"""Map frame and metric scale.

Scale (fixed at map creation): median over frames of the per-frame median ratio between MoGe
metric depth and SfM depth at well-triangulated points (track >= 3, reprojection < 2 px, 3-MAD
rejection, >= 50 points per frame). Gravity: confidence-weighted mean of the per-frame up vectors
rotated into the SfM frame. Map frame (C9): metres, z up, origin at the first keyframe's camera
centre, x = that camera's forward direction projected onto the floor, y = z × x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import Sim3
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction.depth import fit_frame_scale, global_scale, sample_depth
from oh_my_slam.reconstruction.gravity import mean_up

log = get_logger("oh_my_slam.frame")

SCALE_SPREAD_WARN = 0.2


@dataclass
class FrameDepth:
    name: str
    depth: NDArray[np.float32]  # MoGe metric depth on the grid
    K_grid: Intrinsics
    full_size: tuple[int, int]
    up_cam: NDArray[np.float64] | None = None
    up_confidence: float = 1.0


@dataclass
class ScaleResult:
    scale: float
    spread: float
    per_frame: dict[str, float] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=dict)


def grid_uv(uv_full: NDArray[Any], full_size: tuple[int, int], grid: Intrinsics) -> NDArray[Any]:
    """COLMAP pixel coordinates (corner origin) → grid pixel-centre coordinates."""
    sx = grid.width / full_size[0]
    sy = grid.height / full_size[1]
    return np.stack([uv_full[:, 0] * sx - 0.5, uv_full[:, 1] * sy - 0.5], axis=1)


def sample_depth_at(depth: NDArray[Any], uv_full: NDArray[Any], full_size: tuple[int, int],
                    grid: Intrinsics) -> NDArray[np.float64]:
    return sample_depth(depth, grid_uv(uv_full, full_size, grid))


def per_frame_ratio(uv_full: NDArray[Any], z_model: NDArray[Any], f: FrameDepth,
                    min_points: int = 50) -> tuple[float, int]:
    """Robust median of predicted-depth / model-depth for one frame."""
    pred = sample_depth(f.depth, grid_uv(uv_full, f.full_size, f.K_grid))
    fit = fit_frame_scale(z_model, pred, min_points)  # s * z_model ≈ pred
    return (fit.scale if fit.ok else float("nan")), fit.inliers


def metric_scale(model: Any, frames: list[FrameDepth]) -> ScaleResult:
    """Scale that turns SfM units into metres (x_metric = s * x_sfm)."""
    per, counts = {}, {}
    for f in frames:
        if f.name not in model.registered:
            continue
        uv, xyz = model.observations(f.name)
        if len(xyz) == 0:
            continue
        z = model.pose(f.name).inverse().apply(xyz)[:, 2]
        s, n = per_frame_ratio(uv, z, f)
        counts[f.name] = n
        if np.isfinite(s):
            per[f.name] = s
    s, spread = global_scale(list(per.values()))
    if not np.isfinite(s):
        raise ValueError("no frame had enough triangulated points to fix the metric scale")
    if spread > SCALE_SPREAD_WARN:
        log.warning("metric scale varies across frames (IQR/median %.2f > %.2f)", spread,
                    SCALE_SPREAD_WARN)
    return ScaleResult(s, spread, per, counts)


def world_up(poses: dict[str, Pose], frames: list[FrameDepth]) -> NDArray[np.float64]:
    ups, ws = [], []
    for f in frames:
        if f.up_cam is None or f.name not in poses:
            continue
        ups.append(poses[f.name].R @ f.up_cam)
        ws.append(f.up_confidence)
    if not ups:
        return np.array([0.0, 0.0, 1.0])
    return mean_up(ups, ws)


def map_transform(first_pose: Pose, up_world: NDArray[Any], scale: float) -> Sim3:
    """Similarity SfM world → map (see module docstring)."""
    z = np.asarray(up_world, np.float64)
    z /= np.linalg.norm(z)
    fwd = first_pose.R[:, 2]
    x = fwd - (fwd @ z) * z
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down: use the camera x axis
        cx = first_pose.R[:, 0]
        x = cx - (cx @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # rows: map axes in SfM coordinates
    t = -scale * R @ first_pose.t
    return Sim3(scale, R, t)


def align_by_poses(out_poses: list[Pose], ref_poses: list[Pose]) -> Pose:
    """Rigid ``T_ref_out`` mapping poses of an output frame onto reference poses of the same
    cameras: rotation = SVD average of R_ref R_out^T, translation = mean residual. Works for
    rotation-only rigs where the camera centres (almost) coincide."""
    if not out_poses:
        return Pose.identity()
    M = sum(r.R @ o.R.T for o, r in zip(out_poses, ref_poses, strict=True))
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    t = np.mean([r.t - R @ o.t for o, r in zip(out_poses, ref_poses, strict=True)], axis=0)
    return Pose(R, t)


def transform_pose(sim: Sim3, T: Pose) -> Pose:
    M = sim.transform_pose(T.matrix())
    return Pose.from_matrix(M)
