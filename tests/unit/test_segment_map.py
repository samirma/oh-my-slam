"""``segment.sh -m`` on a persisted map with the inference server down: no inference, the map is
never modified, and the outputs follow the same rules as for an image (``-f``, ``-o``, ``-d``,
``-p`` with map scope, byte-identical artefacts, the colour contract)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core import paths
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.mapping import store
from oh_my_slam.mapping.api import update
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES
from oh_my_slam.segmentation.colors import UNSEGMENTED
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room, ring

REPO = Path(__file__).resolve().parents[2]


def segment(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(REPO / "segment.sh"), *args], capture_output=True, timeout=120,
                          env=os.environ.copy())


@pytest.fixture(scope="module")
def one_image_map(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A one-keyframe map (identity pose: no COLMAP needed) with three objects."""
    base = tmp_path_factory.mktemp("segmap")
    client = FakeClient()
    imgs = add_frames(client, mapping_room(), ring(1), base / "in", "s")
    update(base / "map", imgs, client=client, progress=lambda m: None)
    return base / "map"


def test_segment_map_works_with_the_server_down_and_never_modifies_it(
        one_image_map: Path, tmp_path: Path) -> None:
    assert not paths.socket_path().exists()  # no inference server in this test session
    before = store.full_tree_hash(one_image_map)
    res = segment("-m", str(one_image_map))
    assert res.returncode == 0, res.stderr.decode()
    doc = json.loads(res.stdout)
    assert validation_errors(doc) == [] and len(doc["openlabel"]["objects"]) == 3
    assert doc["openlabel"]["metadata"]["tool"] == "segment"  # not the mapper that stored it
    stored = store.MapReader(one_image_map).read_json(store.SCENE_JSON)
    assert stored["openlabel"]["metadata"]["tool"] == "mapper"
    assert doc["openlabel"]["objects"] == stored["openlabel"]["objects"]
    assert b"timings: total" in res.stderr  # per-stage timings, as for -i
    art = tmp_path / "art"
    ply = segment("-m", str(one_image_map), "-f", "ply", "-d", str(art), "-p",
                  "label=on,voxel=0.02,normals=on")
    assert ply.returncode == 0, ply.stderr.decode()
    assert sorted(p.name for p in art.iterdir()) == sorted(ARTIFACT_NAMES)
    assert (art / "segments.ply").read_bytes() == ply.stdout  # identical to -f ply
    assert (art / "segmentation.json").read_bytes() == res.stdout  # identical to -f json
    assert parse_header(ply.stdout).comments == [
        "oh-my-slam map frame (z up), metres",
        "attributes color=segment,voxel=0.02,normals=on,label=on,encoding=binary"]
    cloud = parse_ply(ply.stdout)
    assert cloud.label is not None and cloud.rgb is not None and cloud.normals is not None
    objects = doc["openlabel"]["objects"]
    assert set(np.unique(cloud.label)) == {0} | {int(k) for k in objects}
    for key, o in objects.items():  # the colour contract, as in segmentation.json
        np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == int(key)], axis=0),
                                      [o["object_data"]["vec"][0]["val"]])
    np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == 0], axis=0), [UNSEGMENTED])
    out = tmp_path / "result.json"
    quiet = segment("-m", str(one_image_map), "-o", str(out))
    assert quiet.returncode == 0 and quiet.stdout == b"" and out.read_bytes() == res.stdout
    default_ply = segment("-m", str(one_image_map), "-f", "ply")
    assert parse_ply(default_ply.stdout).label is None  # label property off by default
    assert store.full_tree_hash(one_image_map) == before  # nothing in the map changed


@pytest.mark.parametrize(("args", "hint"), [
    (["-f", "ply", "-p", "stride=2"], b"pixel-level attribute"),
    (["-f", "ply", "-p", "edge=0"], b"pixel-level attribute"),
    (["-f", "ply", "-p", "color=rgb"], b"fixed to segment"),
    (["-p", "voxel=0.1"], b"-f ply or -d"),
    (["--min-score", "0.6"], b"--min-score applies to -i only"),
])
def test_segment_map_usage_errors(one_image_map: Path, args: list[str], hint: bytes) -> None:
    res = segment("-m", str(one_image_map), *args)
    assert res.returncode == 2 and res.stdout == b"" and hint in res.stderr, res.stderr
