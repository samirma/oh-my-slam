"""MoGe-2 (ViT-L, the checkpoint with a normal head, which is not loaded): metric point map,
depth, validity mask, intrinsics, plus a global DINOv2 class-token descriptor for retrieval.

MoGe's network output does not depend on the focal length: a known ``fov_x`` enters only its
post-processing, which solves the point map's z shift for that focal and derives depth and
intrinsics from it. A request with ``keep_forward`` therefore keeps the network output for its
pixels (:class:`KeptForwards`), and the next request for the same pixels and tokens re-solves that
output with its own ``fov_x`` instead of running the network again (the mapper's focal re-run).
The forward pass is deterministic, so the result is the one a second forward pass gives, bit for
bit.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import GeometryRequest
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.images import load_rgb, upright_size

REPO_ID = "Ruicheng/moge-2-vitl-normal"

# Kept network outputs: 2.2 MiB for a 768x432 keyframe (the point map before MoGe's output remap,
# fp16, and the binary mask), so about 460 keyframes. The mapper's focal re-run follows its
# inference within minutes; an output nothing re-solved (the focal agreed, the keyframe was not
# posed, the run stopped) expires.
KEEP_MAX_BYTES = 1 << 30
KEEP_MAX_AGE_S = 1800.0


def _use_fp16(device: str) -> bool:
    """fp16 autocast on MPS, as gate G2 decided (``GATE_G2_FP16``)."""
    return device == "mps" and GATE_G2_FP16


# Set from the gate G2 measurement (MPS fp16 vs CPU fp32 median relative depth difference).
GATE_G2_FP16 = True


class KeptForwards:
    """Network outputs kept for a re-solve: bounded in total bytes and in age, oldest dropped
    first; each is handed out once (:meth:`pop`). Used on the GPU worker thread only."""

    def __init__(self, max_bytes: int = KEEP_MAX_BYTES, max_age_s: float = KEEP_MAX_AGE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_bytes = max_bytes
        self.max_age_s = max_age_s
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, int, Any]] = OrderedDict()
        self.nbytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, key: Hashable, value: Any, nbytes: int) -> None:
        self._remove(key)
        self._expire()
        if nbytes > self.max_bytes:
            return
        self._entries[key] = (self._clock(), nbytes, value)
        self.nbytes += nbytes
        while self.nbytes > self.max_bytes:
            self._remove(next(iter(self._entries)))

    def pop(self, key: Hashable) -> Any | None:
        self._expire()
        return self._remove(key)

    def _remove(self, key: Hashable) -> Any | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        self.nbytes -= entry[1]
        return entry[2]

    def _expire(self) -> None:
        now = self._clock()
        while self._entries:
            key, (t, _, _) = next(iter(self._entries.items()))
            if now - t <= self.max_age_s:
                return
            self._remove(key)


def forward_key(rgb: np.ndarray, num_tokens: int, fp16: bool) -> tuple[Hashable, ...]:
    """What the network output depends on: the exact pixels, the token count and the precision."""
    digest = hashlib.blake2b(np.ascontiguousarray(rgb).data, digest_size=16).digest()
    return digest, rgb.shape, rgb.dtype.str, int(num_tokens), bool(fp16)


@dataclass
class _Forward:
    """One network output, on the CPU: the point map as the points head gave it (before MoGe's
    output remap, NCHW), the binary mask, the metric scale and the descriptor."""

    points: Any
    mask: Any
    metric_scale: Any
    descriptor: np.ndarray | None = None

    @property
    def nbytes(self) -> int:
        tensors = sum(int(t.element_size() * t.nelement())
                      for t in (self.points, self.mask, self.metric_scale))
        return tensors + (0 if self.descriptor is None else int(self.descriptor.nbytes))


class MoGeGeometry:
    key = "geometry"
    name = "MoGe-2 ViT-L normal"

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"
        self.fp16 = False
        self._cls: Any = None
        self.precision = "fp32"
        self.kept = KeptForwards()
        self._network: Any = None
        self._keep = False  # capture this forward pass (``_forward``)
        self._replay: _Forward | None = None  # answer this forward pass with a kept output
        self._pre_remap: Any = None
        self._captured: _Forward | None = None

    def load(self, device: str) -> None:
        import torch
        from moge.model.v2 import MoGeModel

        self.device = device
        model = MoGeModel.from_pretrained(REPO_ID)
        if hasattr(model, "normal_head"):
            # The server returns no normals. Each head reads only the shared neck, so without this
            # one the others give the same output, bit for bit, and a forward pass takes 5 % less.
            del model.normal_head
        model = model.to(torch.device(device)).eval()
        if hasattr(model, "enable_pytorch_native_sdpa"):
            model.enable_pytorch_native_sdpa()
        self.fp16 = _use_fp16(device)
        self.precision = "fp16-autocast" if self.fp16 else "fp32"
        encoder = model.encoder
        original = encoder.forward

        def forward_capture(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                self._cls = out[1]
            return out

        encoder.forward = forward_capture
        remap = model._remap_points

        def remap_capture(points: Any) -> Any:
            if self._keep:
                self._pre_remap = points
            return remap(points)

        model._remap_points = remap_capture
        # ``MoGeModel.infer`` runs ``self.forward`` and then its post-processing: the network is
        # reached through ``_forward``, which can keep its output or answer with a kept one.
        self._network = model.forward
        model.forward = self._forward
        self.model = model

    def _forward(self, image: Any, num_tokens: Any) -> dict[str, Any]:
        """``MoGeModel.forward`` (called by ``infer`` inside its autocast), or its kept output.

        A kept output is the network's own: the points head's map before the output remap (so in
        the dtype and memory layout the head gave it), which the remap turns into the points the
        forward pass returns, the mask binarised as ``infer`` binarises it, and the metric scale.
        ``infer`` reads nothing else of the forward pass (the normal head is not loaded)."""
        replay = self._replay
        if replay is not None:
            dev = self.model.device
            points = self.model._remap_points(replay.points.to(dev).permute(0, 2, 3, 1))
            return {"points": points, "mask": replay.mask.to(dev),
                    "metric_scale": replay.metric_scale.to(dev)}
        self._pre_remap = None
        out: dict[str, Any] = self._network(image, num_tokens=num_tokens)
        if self._keep and self._pre_remap is not None and {"mask", "metric_scale"} <= out.keys():
            self._captured = _Forward(
                points=self._pre_remap.permute(0, 3, 1, 2).cpu(),
                mask=(out["mask"].float() > 0.5).cpu(),
                metric_scale=out["metric_scale"].cpu(),
            )
        self._pre_remap = None
        return out

    def warmup(self) -> None:
        import torch

        img = torch.rand(3, 256, 320, device=self.device)
        self.model.infer(img, num_tokens=1200, use_fp16=self.fp16)
        self._cls = None

    def infer_array(self, rgb: np.ndarray, fov_x_deg: float | None, num_tokens: int,
                    fp16: bool | None = None, keep: bool = False) -> dict[str, Any]:
        """Depth, mask and intrinsics (and the descriptor) for an RGB uint8 array. ``keep`` keeps
        the network output for a later re-solve; without it, a kept output of the same pixels and
        tokens is re-solved instead of running the network."""
        import torch

        use_fp16 = self.fp16 if fp16 is None else fp16
        key = forward_key(rgb, num_tokens, use_fp16) if keep or len(self.kept) else None
        replay = None if keep or key is None else self.kept.pop(key)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        t = t.permute(2, 0, 1).float().div_(255.0)
        self._keep, self._replay, self._captured = keep, replay, None
        try:
            out = self.model.infer(t, num_tokens=num_tokens, fov_x=fov_x_deg, use_fp16=use_fp16)
        finally:
            self._keep, self._replay = False, None
        result = {k: out[k].detach().float().cpu().numpy()
                  for k in ("depth", "mask", "intrinsics") if k in out}
        if self._cls is not None:
            cls = self._cls.detach().float()
            cls = torch.nn.functional.normalize(cls.reshape(cls.shape[0], -1), dim=-1)
            result["descriptor"] = cls[0].cpu().numpy()
        elif replay is not None and replay.descriptor is not None:
            result["descriptor"] = replay.descriptor
        self._cls = None
        captured, self._captured = self._captured, None
        if keep and captured is not None and key is not None:
            captured.descriptor = result.get("descriptor")
            self.kept.put(key, captured, captured.nbytes)
        return result

    def run(self, req: GeometryRequest) -> dict[str, Any]:
        rgb = load_rgb(Path(req.image_path), max_side=req.max_side)
        ow, oh = upright_size(Path(req.image_path))
        h, w = rgb.shape[:2]
        out = self.infer_array(rgb, req.fov_x_deg, req.num_tokens, keep=req.keep_forward)
        depth = out["depth"].astype(np.float32)
        mask = out.get("mask")
        valid = np.isfinite(depth) & (depth > 0)
        if mask is not None:
            valid &= mask.astype(bool)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        K = out["intrinsics"]
        fx, fy = float(K[0, 0] * w), float(K[1, 1] * h)
        cx, cy = float(K[0, 2] * w), float(K[1, 2] * h)
        out_dir = Path(req.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(out_dir / "depth.npy", depth)
        atomic_save_npy(out_dir / "mask.npy", valid.astype(np.uint8))
        descriptor = None
        if req.want_descriptor and "descriptor" in out:
            descriptor = [float(v) for v in out["descriptor"]]
        return {
            "width": w,
            "height": h,
            "orig_width": ow,
            "orig_height": oh,
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "fov_x_deg": math.degrees(2 * math.atan(w / (2 * fx))),
            "depth_path": str(out_dir / "depth.npy"),
            "mask_path": str(out_dir / "mask.npy"),
            "descriptor": descriptor,
            "timings": {},
        }
