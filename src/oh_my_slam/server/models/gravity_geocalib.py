"""GeoCalib: gravity direction (roll/pitch with uncertainty) and a field-of-view estimate."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import GravityRequest
from oh_my_slam.core.images import load_rgb, upright_size
from oh_my_slam.server.gpu_worker import CpuFinish

_INPUT_SIDE = 640

# GeoCalib's Gravity.vec3d for roll = pitch = 0 is (0, -1, 0) and its camera frame follows OpenCV
# axes, so vec3d is the *up* direction in camera coordinates (verified on real images, see the
# Implementation Notes). UP_SIGN flips it should a future version change the convention.
UP_SIGN = 1.0


# What GeoCalib's network passes on to its least-squares refinement (``GeoCalib.forward``)
_PASSED = ("image", "scales", "prior_gravity", "prior_focal", "prior_dist", "camera_R_rig")


class GeoCalibGravity:
    """GeoCalib in two parts: the network on the device (the GPU thread), and the Levenberg-
    Marquardt refinement of its fields on the CPU, off the GPU thread (``gpu_worker.CpuFinish``).
    On MPS that refinement synchronised with the CPU at each of its 30 steps (its Cholesky solve
    runs on the CPU): 0.25-0.43 s of the GPU thread per image, which the other models waited
    for. Split, the GPU thread spends 0.10 s per image and a CPU thread 0.12 s beside it (40
    ``ainex-captures`` images). The results are ``calibrate``'s within its own run-to-run
    spread on MPS: the same image's up vectors from two calls 0.08° apart (median; 2.6° at most,
    on a keyframe looking down at a bare floor), the split's 0.06° from ``calibrate``'s (2.2°
    at most), focal lengths within 5e-7."""

    key = "gravity"
    name = "GeoCalib pinhole"

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from geocalib import GeoCalib

        self.device = device
        self.model = GeoCalib(weights="pinhole").to(torch.device(device)).eval()

    def warmup(self) -> None:
        noise = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        self.estimate_array(noise, None)

    def estimate_deferred(self, rgb: np.ndarray, focal_px: float | None
                          ) -> Callable[[], dict[str, float | list]]:
        """GeoCalib's network on ``rgb`` (HxWx3 uint8) now, on the device; returns the rest of the
        estimate (``estimate_array``'s result), which needs the CPU only."""
        import torch

        ext = self.model
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
            t = t.permute(2, 0, 1).float().div_(255.0)
            img_data = ext.image_processor(t[None])
            # as ``calibrate``: the focal prior in pixels of the network's input
            data = dict(img_data) | {"camera_R_rig": None}
            if focal_px is not None:
                prior = torch.tensor([float(focal_px)], device=self.device)
                data["prior_focal"] = prior * img_data["scales"][1]
            net = ext.model
            features = {"hl": net.backbone(data)["features"], "ll": net.ll_enc(data)["features"]}
            out = net.perspective_decoder({"features": features})
            out |= {k: data[k] for k in _PASSED if k in data}
            out = {k: _on_cpu(v) for k, v in out.items()}
            img_data = {k: _on_cpu(v) for k, v in img_data.items()}
        height = rgb.shape[0]
        return lambda: self._solve(out, img_data, height)

    def _solve(self, out: dict[str, Any], img_data: dict[str, Any], height: int
               ) -> dict[str, float | list]:
        """GeoCalib's refinement of the network's fields ``out`` (CPU tensors) and its
        post-processing (``calibrate``), summarised for an image ``height`` px tall."""
        import torch
        from geocalib.lm_optimizer import LMOptimizer

        with torch.no_grad():  # grad mode is per thread
            out = out | LMOptimizer({})(out)  # its own: one runs per thread at once
            cam, out = self.model._post_process(out["camera"], img_data, out)
        grav = out["gravity"]
        vec = grav.vec3d.detach().float().cpu().numpy().reshape(-1, 3)[0]
        up = UP_SIGN * vec / np.linalg.norm(vec)

        def scalar(x: Any, default: float = 0.0) -> float:
            if x is None:
                return default
            return float(x.detach().float().cpu().reshape(-1)[0])

        f = scalar(cam.f[..., 1]) if cam.f.shape[-1] > 1 else scalar(cam.f)
        return {
            "up_cam": [float(v) for v in up],
            "roll_deg": math.degrees(scalar(grav.roll)),
            "pitch_deg": math.degrees(scalar(grav.pitch)),
            "roll_unc_deg": math.degrees(scalar(out.get("roll_uncertainty"))),
            "pitch_unc_deg": math.degrees(scalar(out.get("pitch_uncertainty"))),
            "focal_px_small": f,
            "focal_unc_px_small": scalar(out.get("focal_uncertainty")),
            "vfov_deg": math.degrees(2 * math.atan(height / (2 * f))) if f > 0 else 0.0,
        }

    def estimate_array(self, rgb: np.ndarray, focal_px: float | None) -> dict[str, float | list]:
        return self.estimate_deferred(rgb, focal_px)()

    def run(self, req: GravityRequest) -> CpuFinish:
        path = Path(req.image_path)
        rgb = load_rgb(path, max_side=_INPUT_SIDE)
        ow, _ = upright_size(path)
        scale = rgb.shape[1] / ow
        prior = None if req.focal_px is None else req.focal_px * scale
        estimate = self.estimate_deferred(rgb, prior)

        def finish() -> dict[str, Any]:
            est = estimate()
            return {
                "up_cam": est["up_cam"],
                "roll_deg": est["roll_deg"],
                "pitch_deg": est["pitch_deg"],
                "roll_unc_deg": est["roll_unc_deg"],
                "pitch_unc_deg": est["pitch_unc_deg"],
                "focal_px": float(est["focal_px_small"]) / scale,
                "focal_unc_px": float(est["focal_unc_px_small"]) / scale,
                "vfov_deg": est["vfov_deg"],
                "timings": {},
            }

        return CpuFinish(finish)


def _on_cpu(v: Any) -> Any:
    """A tensor copied to the CPU (anything else as it is)."""
    return v.detach().cpu() if hasattr(v, "detach") else v
