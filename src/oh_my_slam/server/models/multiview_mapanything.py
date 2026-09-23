"""MapAnything (Apache-2.0 checkpoint): metric multi-view poses and depth, optionally conditioned
on known intrinsics and cam-to-world poses for some views (used to anchor chunks)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import MultiviewRequest
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.images import load_rgb, upright_size

REPO_ID = os.environ.get("OH_MY_SLAM_MAPANYTHING_REPO", "facebook/map-anything-apache")
_LOAD_SIDE = 1024


class MapAnythingMultiview:
    key = "multiview"
    name = "MapAnything (apache)"
    required = True

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from mapanything.models import MapAnything

        self.device = device
        self.model = MapAnything.from_pretrained(REPO_ID).to(torch.device(device)).eval()

    def warmup(self) -> None:
        pass  # a forward pass needs >= 1 real view; the first request warms up

    def run(self, req: MultiviewRequest) -> dict[str, Any]:
        import torch
        from mapanything.utils.image import preprocess_inputs

        views = []
        n = len(req.image_paths)
        for i, p in enumerate(req.image_paths):
            path = Path(p)
            rgb = load_rgb(path, max_side=_LOAD_SIDE)
            ow, oh = upright_size(path)
            view: dict[str, Any] = {"img": torch.from_numpy(rgb)}
            K = None if req.intrinsics is None else req.intrinsics[i]
            if K is not None:
                Kn = np.asarray(K, dtype=np.float32).copy()
                sx, sy = rgb.shape[1] / ow, rgb.shape[0] / oh
                Kn[0] *= sx
                Kn[1] *= sy
                view["intrinsics"] = torch.from_numpy(Kn)
            T = None if req.poses is None else req.poses[i]
            if T is not None:
                view["camera_poses"] = torch.from_numpy(np.asarray(T, dtype=np.float32))
                view["is_metric_scale"] = torch.tensor([bool(req.poses_metric)])
            views.append(view)
        processed = preprocess_inputs(views, resize_mode="longest_side", size=req.resolution)
        with torch.inference_mode():
            preds = self.model.infer(
                processed,
                memory_efficient_inference=n > 8,
                use_amp=self.device != "cpu",
                amp_dtype="fp16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )
        out_dir = Path(req.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result_views = []
        scale = 1.0
        for i, pred in enumerate(preds):
            pose = pred["camera_poses"][0].detach().float().cpu().numpy()
            K = pred["intrinsics"][0].detach().float().cpu().numpy()
            depth = pred["depth_z"][0, ..., 0].detach().float().cpu().numpy()
            mask = pred["mask"][0, ..., 0].detach().cpu().numpy().astype(bool)
            conf = pred["conf"][0].detach().float().cpu().numpy()
            depth = np.where(mask & np.isfinite(depth), depth, 0.0).astype(np.float32)
            entry: dict[str, Any] = {
                "pose": pose.tolist(),
                "intrinsics": K.tolist(),
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
            }
            if req.want_depth:
                atomic_save_npy(out_dir / f"mv_depth_{i:04d}.npy", depth)
                atomic_save_npy(out_dir / f"mv_conf_{i:04d}.npy", conf.astype(np.float16))
                entry["depth_path"] = str(out_dir / f"mv_depth_{i:04d}.npy")
                entry["conf_path"] = str(out_dir / f"mv_conf_{i:04d}.npy")
            result_views.append(entry)
            if "metric_scaling_factor" in pred:
                scale = float(pred["metric_scaling_factor"].reshape(-1)[0])
        if self.device == "mps":
            torch.mps.empty_cache()
        return {"views": result_views, "metric_scale": scale, "timings": {}}
