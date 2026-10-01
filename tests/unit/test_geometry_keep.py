"""Kept network outputs (``GeometryRequest.keep_forward``): MoGe-2's forward pass does not depend
on the focal length, so the mapper's focal re-run re-solves the first pass's network output with
the new focal instead of running the network again, and must get exactly what running it gives."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.client import protocol as p
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.api import _reconstruct_keyframe
from oh_my_slam.server.models.geometry_moge import KeptForwards, forward_key
from tests.fakes.client import FakeClient, FakeFrame


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_kept_outputs_are_bounded_in_bytes_oldest_dropped_first() -> None:
    kept = KeptForwards(max_bytes=100, max_age_s=1e9)
    for k in range(5):
        kept.put(k, f"v{k}", 30)
    assert len(kept) == 3 and kept.nbytes == 90
    assert kept.pop(0) is None and kept.pop(1) is None  # the oldest went first
    assert kept.pop(2) == "v2" and kept.nbytes == 60
    kept.put("big", "v", 101)  # larger than the bound: not kept, nothing else dropped
    assert "big" not in kept._entries and len(kept) == 2


def test_kept_outputs_are_handed_out_once_and_replaced_by_a_new_keep() -> None:
    kept = KeptForwards(max_bytes=100, max_age_s=1e9)
    kept.put("a", "first", 10)
    kept.put("a", "second", 20)
    assert len(kept) == 1 and kept.nbytes == 20
    assert kept.pop("a") == "second"
    assert kept.pop("a") is None and kept.nbytes == 0


def test_kept_outputs_expire() -> None:
    clock = _Clock()
    kept = KeptForwards(max_bytes=100, max_age_s=60.0, clock=clock)
    kept.put("old", 1, 10)
    clock.t = 50.0
    kept.put("new", 2, 10)
    clock.t = 61.0
    assert kept.pop("old") is None and kept.nbytes == 10
    assert kept.pop("new") == 2


def test_forward_key_is_the_exact_pixels_tokens_and_precision(rng: np.random.Generator) -> None:
    rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    key = forward_key(rgb, 1400, True)
    assert forward_key(rgb.copy(), 1400, True) == key
    assert forward_key(np.asfortranarray(rgb), 1400, True) == key
    other = rgb.copy()
    other[3, 5, 1] ^= 1
    assert forward_key(other, 1400, True) != key
    assert forward_key(rgb, 1200, True) != key
    assert forward_key(rgb, 1400, False) != key
    assert forward_key(rgb.reshape(32, 24, 3), 1400, True) != key


# MoGe-2's architecture at a toy size with random weights, on the CPU in its own process (torch
# never shares a process with Open3D): the server adapter's keep and re-solve against MoGe's own
# ``infer``.
_TINY_MOGE = r"""
import json, sys
import numpy as np, torch
from moge.model.v2 import MoGeModel
from oh_my_slam.server.models import geometry_moge as gm

torch.manual_seed(0)
dims = [64, 32, 16, 16, 16]
heads = dict(dim_in=dims, dim_res_blocks=dims, num_res_blocks=[0, 1, 1, 1, 0],
             res_block_in_norm="none", res_block_hidden_norm="none",
             resamplers=["conv_transpose"] * 3 + ["bilinear"])
tiny = MoGeModel(
    encoder=dict(backbone="dinov2_vits14", intermediate_layers=[2, 5, 8, 11], dim_out=64),
    neck=dict(heads, dim_in=[66, 2, 2, 2, 2], dim_out=None, num_res_blocks=[0, 2, 2, 2, 0]),
    points_head=dict(heads, dim_out=[None] * 4 + [3]),
    normal_head=dict(heads, dim_out=[None] * 4 + [3]),
    mask_head=dict(heads, dim_out=[None] * 4 + [1]),
    scale_head=dict(dims=[384, 64, 1]), remap_output="exp").eval()
mask_out = [m for m in tiny.mask_head.modules() if isinstance(m, torch.nn.Conv2d)][-1]
assert mask_out.out_channels == 1
mask_out.bias.data += 0.45  # a mask with both values (logits about -0.4..0.4)
rng = np.random.default_rng(0)
img = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
tokens = 100
with_normals = tiny.infer(torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0),
                          num_tokens=tokens, use_fp16=False)
MoGeModel.from_pretrained = classmethod(lambda cls, *a, **k: tiny)
g = gm.MoGeGeometry()
g.load("cpu")
g.warmup()
runs = []
network = g._network
g._network = lambda *a, **k: (runs.append(1), network(*a, **k))[1]
other = np.ascontiguousarray(img[::-1])

def same(a, b):
    return {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in a.keys() | b.keys()}

network_first = g.infer_array(img, None, tokens)
network_rerun = g.infer_array(img, 70.0, tokens)
assert len(g.kept) == 0
first = g.infer_array(img, None, tokens, keep=True)
kept_bytes = g.kept.nbytes
entry = next(iter(g.kept._entries.values()))[2]
other_rerun = g.infer_array(other, 70.0, tokens)  # other pixels: not served by the kept output
n_runs = len(runs)
resolved = g.infer_array(img, 70.0, tokens)
resolve_runs = len(runs) - n_runs
after = len(g.kept)
print(json.dumps({
    "no_normal_head": not hasattr(g.model, "normal_head"),
    "as_with_normals": all(np.array_equal(network_first[k], with_normals[k].float().numpy(),
                                          equal_nan=True) for k in ("depth", "mask", "intrinsics")),
    "first": same(first, network_first),
    "resolved": same(resolved, network_rerun),
    "other_is_network": same(other_rerun, g.infer_array(other, 70.0, tokens)),
    "other_differs": not np.array_equal(other_rerun["depth"], resolved["depth"]),
    "focal_matters": not np.array_equal(network_first["depth"], network_rerun["depth"]),
    "mask_mixed": bool(0 < first["mask"].mean() < 1),
    "kept_bytes": kept_bytes,
    "expected_bytes": 3 * 96 * 128 * entry.points.element_size() + 96 * 128 + 4 + 384 * 4,
    "kept_after": after,
    "resolve_runs": resolve_runs,
}))
"""


def test_resolving_a_kept_output_is_running_the_network_again() -> None:
    proc = subprocess.run([sys.executable, "-c", _TINY_MOGE], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    keys = {"depth", "mask", "intrinsics", "descriptor"}
    # the server loads no normal head: the other outputs are those of the model with it
    assert r["no_normal_head"] and r["as_with_normals"]
    assert r["first"] == dict.fromkeys(keys, True)  # keeping changes nothing of the first pass
    assert r["resolved"] == dict.fromkeys(keys, True)  # bit for bit, the descriptor included
    assert r["other_is_network"] == dict.fromkeys(keys, True) and r["other_differs"]
    assert r["focal_matters"] and r["mask_mixed"]
    assert r["resolve_runs"] == 0  # the network did not run for the re-solve
    assert r["kept_bytes"] == r["expected_bytes"] and r["kept_after"] == 0


class _Recording(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[p.GeometryRequest] = []

    def geometry(self, req: p.GeometryRequest) -> p.GeometryResponse:
        self.requests.append(req)
        return super().geometry(req)


@pytest.mark.parametrize("exif", [False, True])
def test_mapper_keeps_the_forward_pass_only_where_the_focal_may_be_re_solved(
        tmp_path: Path, exif: bool) -> None:
    client = _Recording()
    K = Intrinsics(300.0, 300.0, 160.0, 120.0, 320, 240)
    depth = np.full((240, 320), 2.0, np.float32)
    rgb = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    img = client.add(tmp_path / "f000000.jpg", rgb, FakeFrame(depth, K, np.array([0, -1.0, 0])))
    given = Intrinsics(310.0, 310.0, 160.0, 120.0, 320, 240, "exif") if exif else None
    fr = _reconstruct_keyframe(img, client, given, tmp_path / "w")
    colmap = Intrinsics(330.0, 330.0, 160.0, 120.0, 320, 240, "colmap")
    again = _reconstruct_keyframe(img, client, colmap, first=False, rgb=fr.rgb)
    np.testing.assert_array_equal(again.rgb, fr.rgb)
    first, rerun = client.requests
    assert first.keep_forward is (not exif)
    assert not rerun.keep_forward and rerun.fov_x_deg == pytest.approx(colmap.fov_x_deg)
    assert (first.image_path, first.max_side, first.num_tokens) == (
        rerun.image_path, rerun.max_side, rerun.num_tokens)
