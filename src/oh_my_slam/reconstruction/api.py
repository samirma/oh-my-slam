"""Single-image reconstruction: intrinsics, metric depth, descriptor and gravity.

Intrinsics priority: explicitly given (e.g. from COLMAP) > EXIF > model estimate; the chosen
source is recorded. The depth grid has the long side <= ``max_side``; colours for points come from
the same resized image (``core.images.load_rgb``), so PLY colours equal the resized pixels.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient, connect
from oh_my_slam.core import paths, timing
from oh_my_slam.core.images import exif_intrinsics, load_rgb
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.ply import PointCloud
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.gravity import GravityEstimate, refine_with_floor
from oh_my_slam.reconstruction.pointcloud import MAX_GRID_SIDE, cloud_mask, frame_cloud

log = get_logger("oh_my_slam.reconstruction")

SINGLE_IMAGE_TOKENS = 2500
KEYFRAME_TOKENS = 1400


@dataclass
class FrameReconstruction:
    image_path: Path
    rgb: NDArray[np.uint8]  # resized image on the depth grid
    depth: NDArray[np.float32]  # metres, 0 = invalid
    valid: NDArray[np.bool_]  # model validity mask
    K_grid: Intrinsics  # intrinsics of the depth grid
    intrinsics: Intrinsics  # full-resolution intrinsics (with source)
    descriptor: NDArray[np.float32] | None = None
    gravity: GravityEstimate | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[0]

    def point_mask(self) -> NDArray[np.bool_]:
        return cloud_mask(self.depth, self.valid)

    def camera_cloud(self, mask: NDArray[Any] | None = None) -> tuple[PointCloud, NDArray[Any]]:
        """Coloured points in the camera frame (OpenCV axes, metres) and their pixel indices."""
        return frame_cloud(self.depth, self.rgb, self.K_grid,
                           self.point_mask() if mask is None else mask)


def connect_server() -> InferenceClient:
    """The inference client for callers that delegate inference to reconstruction (mapper)."""
    return connect()


def _intrinsics_from_grid(g: p.GeometryResponse, source: str) -> Intrinsics:
    grid = Intrinsics(g.intrinsics.fx, g.intrinsics.fy, g.intrinsics.cx, g.intrinsics.cy,
                      g.width, g.height, "model")
    return grid.resized(g.orig_width, g.orig_height).with_source(source)  # type: ignore[arg-type]


def reconstruct_image(
    image_path: Path,
    *,
    intrinsics: Intrinsics | None = None,
    max_side: int = MAX_GRID_SIDE,
    num_tokens: int = SINGLE_IMAGE_TOKENS,
    want_gravity: bool = True,
    want_descriptor: bool = False,
    work_dir: Path | None = None,
    client: InferenceClient | None = None,
) -> FrameReconstruction:
    """Run geometry (and gravity) for one image. ``work_dir`` keeps the server's files."""
    image_path = Path(image_path).resolve()
    client = client or connect()
    intr = intrinsics or exif_intrinsics(image_path)
    fov_x = intr.fov_x_deg if intr is not None else None
    own_dir = work_dir is None
    out_dir = Path(tempfile.mkdtemp(prefix="frame-", dir=paths.scratch_dir())) if own_dir else (
        Path(work_dir)  # type: ignore[arg-type]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with timing.part("geometry"):
        try:
            g = client.geometry(
                p.GeometryRequest(
                    image_path=str(image_path),
                    out_dir=str(out_dir),
                    max_side=max_side,
                    fov_x_deg=fov_x,
                    num_tokens=num_tokens,
                    want_descriptor=want_descriptor,
                )
            )
            depth = np.load(g.depth_path).astype(np.float32)
            valid = np.load(g.mask_path).astype(bool)
        finally:
            if own_dir:
                shutil.rmtree(out_dir, ignore_errors=True)
        rgb = load_rgb(image_path, max_side=max_side)
    if intr is None:
        intr = _intrinsics_from_grid(g, "model")
    K_grid = intr.resized(g.width, g.height)
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(f"grid mismatch: image {rgb.shape[:2]} vs depth {depth.shape}")
    frame = FrameReconstruction(
        image_path=image_path,
        rgb=rgb,
        depth=depth,
        valid=valid,
        K_grid=K_grid,
        intrinsics=intr,
        descriptor=None if g.descriptor is None else np.asarray(g.descriptor, np.float32),
        meta={"geometry_s": g.timings.compute_s, "model_fov_x_deg": g.fov_x_deg},
    )
    if want_gravity:
        frame.gravity = estimate_gravity(frame, client)
    return frame


def estimate_gravity(frame: FrameReconstruction, client: InferenceClient | None = None,
                     refine: bool = True) -> GravityEstimate:
    with timing.part("gravity"):
        return _estimate_gravity(frame, client or connect(), refine)


def _estimate_gravity(frame: FrameReconstruction, client: InferenceClient,
                      refine: bool) -> GravityEstimate:
    gr = client.gravity(p.GravityRequest(image_path=str(frame.image_path),
                                         focal_px=frame.intrinsics.fx))
    prior = GravityEstimate(
        up_cam=np.asarray(gr.up_cam, dtype=np.float64),
        source="geocalib",
        roll_unc_deg=gr.roll_unc_deg,
        pitch_unc_deg=gr.pitch_unc_deg,
    )
    frame.meta["geocalib_focal_px"] = gr.focal_px
    if not refine:
        return prior
    cloud, _ = frame.camera_cloud()
    return refine_with_floor(cloud.xyz, prior)
