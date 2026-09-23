"""GeoCalib: gravity direction (roll/pitch with uncertainty) and a field-of-view estimate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import GravityRequest
from oh_my_slam.core.images import load_rgb, upright_size

_INPUT_SIDE = 640

# GeoCalib's Gravity.vec3d for roll = pitch = 0 is (0, -1, 0) and its camera frame follows OpenCV
# axes, so vec3d is the *up* direction in camera coordinates (verified on real images, see the
# Implementation Notes). UP_SIGN flips it should a future version change the convention.
UP_SIGN = 1.0


class GeoCalibGravity:
    key = "gravity"
    name = "GeoCalib pinhole"
    required = True

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from geocalib import GeoCalib

        self.device = device
        self.model = GeoCalib(weights="pinhole").to(torch.device(device)).eval()

    def warmup(self) -> None:
        import torch

        img = torch.rand(3, 240, 320, device=self.device)
        self.model.calibrate(img)

    def estimate_array(self, rgb: np.ndarray, focal_px: float | None) -> dict[str, float | list]:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        t = t.permute(2, 0, 1).float().div_(255.0)
        priors = None
        if focal_px is not None:
            priors = {"focal": torch.tensor(float(focal_px), device=self.device)}
        res = self.model.calibrate(t, priors=priors)
        grav = res["gravity"]
        cam = res["camera"]
        vec = grav.vec3d.detach().float().cpu().numpy().reshape(-1, 3)[0]
        up = UP_SIGN * vec / np.linalg.norm(vec)

        def scalar(x: Any, default: float = 0.0) -> float:
            if x is None:
                return default
            return float(x.detach().float().cpu().reshape(-1)[0])

        f = scalar(cam.f[..., 1]) if cam.f.shape[-1] > 1 else scalar(cam.f)
        h = rgb.shape[0]
        return {
            "up_cam": [float(v) for v in up],
            "roll_deg": math.degrees(scalar(grav.roll)),
            "pitch_deg": math.degrees(scalar(grav.pitch)),
            "roll_unc_deg": math.degrees(scalar(res.get("roll_uncertainty"))),
            "pitch_unc_deg": math.degrees(scalar(res.get("pitch_uncertainty"))),
            "focal_px_small": f,
            "focal_unc_px_small": scalar(res.get("focal_uncertainty")),
            "vfov_deg": math.degrees(2 * math.atan(h / (2 * f))) if f > 0 else 0.0,
        }

    def run(self, req: GravityRequest) -> dict[str, Any]:
        path = Path(req.image_path)
        rgb = load_rgb(path, max_side=_INPUT_SIDE)
        ow, _ = upright_size(path)
        scale = rgb.shape[1] / ow
        prior = None if req.focal_px is None else req.focal_px * scale
        est = self.estimate_array(rgb, prior)
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
