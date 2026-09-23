"""In-process stand-in for the inference client, driven by synthetic ground truth.

Register an image with its depth grid, intrinsics, up vector, instances and (optionally) its
true camera-to-map pose; the fake then answers geometry/gravity/segment/multiview requests the
way the real server would (files written into ``out_dir``, masks as RLE).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client import protocol as p
from oh_my_slam.core import rle
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.errors import ServerUnavailableError
from oh_my_slam.core.images import save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose


@dataclass
class FakeInstance:
    label: str
    score: float
    mask: NDArray[np.bool_]


@dataclass
class FakeFrame:
    depth: NDArray[np.float32]
    K: Intrinsics  # on the grid (= image size for these tests)
    up_cam: NDArray[np.float64]
    instances: list[FakeInstance] = field(default_factory=list)
    pose: Pose | None = None
    descriptor: NDArray[np.float64] | None = None


class FakeClient:
    def __init__(self, depth_scale: float = 1.0, down: bool = False) -> None:
        self.frames: dict[str, FakeFrame] = {}
        self._alias: dict[str, str] = {}
        self._thumbs: dict[str, NDArray[np.float32]] = {}
        self.calls: Counter[str] = Counter()
        self.depth_scale = depth_scale
        self.down = down

    def add(self, path: Path, rgb: NDArray[np.uint8], frame: FakeFrame) -> Path:
        if path.suffix == ".png":
            from oh_my_slam.core.images import save_png

            save_png(rgb, path)
        else:
            save_jpeg(rgb, path, quality=98)
        self.frames[str(Path(path).resolve())] = frame
        return path

    def clone(self) -> FakeClient:
        return self

    def close(self) -> None:
        pass

    def _get(self, path: str) -> FakeFrame:
        """Registered frame for ``path``; copies (e.g. the mapper's keyframe JPEGs) are matched by
        image content."""
        key = str(Path(path).resolve())
        if key in self.frames:
            return self.frames[key]
        if key in self._alias:
            return self.frames[self._alias[key]]
        from oh_my_slam.core.images import load_rgb

        thumb = load_rgb(Path(path), max_side=48).astype(np.float32)
        best, best_err = None, np.inf
        for k in self.frames:
            t = self._thumb(k)
            if t.shape != thumb.shape:
                continue
            err = float(np.abs(t - thumb).mean())
            if err < best_err:
                best, best_err = k, err
        if best is None or best_err > 6.0:
            raise KeyError(f"fake client: unknown image {path}")
        self._alias[key] = best
        return self.frames[best]

    def _thumb(self, key: str) -> NDArray[np.float32]:
        if key not in self._thumbs:
            from oh_my_slam.core.images import load_rgb

            self._thumbs[key] = load_rgb(Path(key), max_side=48).astype(np.float32)
        return self._thumbs[key]

    # -- client API ------------------------------------------------------------------------------

    def health(self, timeout: float = 0.5) -> p.Health:
        if self.down:
            raise ServerUnavailableError("fake down")
        return p.Health(status="ready")

    def require_ready(self, wait_loading_s: float = 0) -> p.Health:
        return self.health()

    def geometry(self, req: p.GeometryRequest) -> p.GeometryResponse:
        self.calls["geometry"] += 1
        f = self._get(req.image_path)
        out = Path(req.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        depth = (f.depth * self.depth_scale).astype(np.float32)
        atomic_save_npy(out / "depth.npy", depth)
        atomic_save_npy(out / "mask.npy", (depth > 0).astype(np.uint8))
        normals = None
        if req.want_normals:
            atomic_save_npy(out / "normals.npy", np.zeros((*depth.shape, 3), np.float16))
            normals = str(out / "normals.npy")
        h, w = depth.shape
        K = f.K
        if req.fov_x_deg is not None:
            fx = w / (2 * np.tan(np.radians(req.fov_x_deg) / 2))
            K = Intrinsics(fx, fx, w / 2, h / 2, w, h)
        desc = f.descriptor
        if desc is None:
            desc = np.ones(8) / np.sqrt(8)
        return p.GeometryResponse(
            width=w, height=h, orig_width=f.K.width, orig_height=f.K.height,
            intrinsics=p.PixelIntrinsics(fx=K.fx, fy=K.fy, cx=K.cx, cy=K.cy),
            fov_x_deg=float(np.degrees(2 * np.arctan(w / (2 * K.fx)))),
            depth_path=str(out / "depth.npy"), mask_path=str(out / "mask.npy"),
            normals_path=normals, descriptor=[float(v) for v in desc],
        )

    def gravity(self, req: p.GravityRequest) -> p.GravityResponse:
        self.calls["gravity"] += 1
        f = self._get(req.image_path)
        return p.GravityResponse(
            up_cam=[float(v) for v in f.up_cam], roll_deg=0, pitch_deg=0, roll_unc_deg=1.0,
            pitch_unc_deg=1.0, focal_px=f.K.fx, focal_unc_px=1.0, vfov_deg=60.0,
        )

    def segment_image(self, req: p.SegmentRequest) -> p.SegmentResponse:
        self.calls["segment"] += 1
        f = self._get(req.image_path)
        h, w = f.depth.shape
        wanted = set(req.labels)
        inst = [
            p.Instance(label=i.label, score=i.score, source="yoloe",
                       box_xyxy=[0, 0, float(w), float(h)], mask=rle.encode(i.mask))
            for i in f.instances
            if i.label in wanted and i.score >= req.conf
        ]
        return p.SegmentResponse(width=w, height=h, instances=inst)

    def multiview(self, req: p.MultiviewRequest) -> p.MultiviewResponse:
        self.calls["multiview"] += 1
        views = []
        for path in req.image_paths:
            f = self._get(path)
            pose = f.pose or Pose.identity()
            views.append(p.MultiviewView(pose=pose.matrix().tolist(), intrinsics=f.K.K().tolist(),
                                         width=f.K.width, height=f.K.height))
        return p.MultiviewResponse(views=views)
