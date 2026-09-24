"""MoGe-2 (ViT-L, normal head): metric point map, depth, validity mask, normals, intrinsics, plus a
global DINOv2 class-token descriptor for retrieval."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import GeometryRequest
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.images import load_rgb, upright_size

REPO_ID = "Ruicheng/moge-2-vitl-normal"


def _use_fp16(device: str) -> bool:
    """Gate G2 decides the default; ``OH_MY_SLAM_GEOMETRY_FP16=0/1`` overrides."""
    env = os.environ.get("OH_MY_SLAM_GEOMETRY_FP16")
    if env is not None:
        return env == "1"
    return device == "mps" and GATE_G2_FP16


# Set from the gate G2 measurement (MPS fp16 vs CPU fp32 median relative depth difference).
GATE_G2_FP16 = True


class MoGeGeometry:
    key = "geometry"
    name = "MoGe-2 ViT-L normal"
    required = True

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"
        self.fp16 = False
        self._cls: Any = None
        self.precision = "fp32"

    def load(self, device: str) -> None:
        import torch
        from moge.model.v2 import MoGeModel

        self.device = device
        model = MoGeModel.from_pretrained(REPO_ID)
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
        self.model = model

    def warmup(self) -> None:
        import torch

        img = torch.rand(3, 256, 320, device=self.device)
        self.model.infer(img, num_tokens=1200, use_fp16=self.fp16)
        self._cls = None

    def infer_array(self, rgb: np.ndarray, fov_x_deg: float | None, num_tokens: int,
                    fp16: bool | None = None) -> dict[str, Any]:
        """Raw inference on an RGB uint8 array."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        t = t.permute(2, 0, 1).float().div_(255.0)
        out = self.model.infer(
            t,
            num_tokens=num_tokens,
            fov_x=fov_x_deg,
            use_fp16=self.fp16 if fp16 is None else fp16,
        )
        result = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
        if self._cls is not None:
            cls = self._cls.detach().float()
            cls = torch.nn.functional.normalize(cls.reshape(cls.shape[0], -1), dim=-1)
            result["descriptor"] = cls[0].cpu().numpy()
        self._cls = None
        return result

    def run(self, req: GeometryRequest) -> dict[str, Any]:
        rgb = load_rgb(Path(req.image_path), max_side=req.max_side)
        ow, oh = upright_size(Path(req.image_path))
        h, w = rgb.shape[:2]
        out = self.infer_array(rgb, req.fov_x_deg, req.num_tokens)
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
        normals_path = None
        if req.want_normals and "normal" in out:
            normals = np.where(valid[..., None], out["normal"], 0.0).astype(np.float16)
            atomic_save_npy(out_dir / "normals.npy", normals)
            normals_path = str(out_dir / "normals.npy")
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
            "normals_path": normals_path,
            "descriptor": descriptor,
            "timings": {},
        }
