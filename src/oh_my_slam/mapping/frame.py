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

from oh_my_slam.core.geometry import Sim3, rot_z
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


# The up of a map's horizontal surfaces (``horizontal_up``): the normals within LEVEL_MAX_DEG of the
# current up are averaged, then those within LEVEL_CONE_DEG of that mean, LEVEL_PASSES times (it
# stays put after two or three); fewer than LEVEL_MIN_NORMALS of them: none.
LEVEL_MAX_DEG = 10.0
LEVEL_CONE_DEG = 5.0
LEVEL_PASSES = 5
LEVEL_MIN_NORMALS = 500


def horizontal_up(normals: NDArray[Any], up: NDArray[Any]) -> NDArray[np.float64] | None:
    """The mean direction of the surface normals ``normals`` (N, 3; unit) that are horizontal
    surfaces' — within ``LEVEL_MAX_DEG`` of ``up`` either way (a ceiling faces down) — refined
    within ``LEVEL_CONE_DEG`` of itself (``LEVEL_PASSES``); None with fewer than
    ``LEVEL_MIN_NORMALS`` of them. It depends on the surfaces only, not on ``up`` within the cone:
    every pass averages the same pixels."""
    n = np.asarray(normals, np.float64).reshape(-1, 3)
    u = np.asarray(up, np.float64) / np.linalg.norm(up)
    for cone in (LEVEL_MAX_DEG,) + (LEVEL_CONE_DEG,) * LEVEL_PASSES:
        c = n @ u
        near = np.abs(c) >= np.cos(np.radians(cone))
        if int(near.sum()) < LEVEL_MIN_NORMALS:
            return None
        m = (n[near] * np.sign(c[near])[:, None]).sum(axis=0)
        u = m / np.linalg.norm(m)
    return u


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


def similarity_by_poses(out_poses: list[Pose], ref_poses: list[Pose],
                        with_scale: bool = True) -> Sim3:
    """Similarity mapping poses of an output frame onto reference poses of the same cameras:
    rotation = SVD average of R_ref R_out^T, scale = ratio of the camera centres' RMS spreads
    (1 when they coincide, or without ``with_scale``), translation = mean residual."""
    M = sum(r.R @ o.R.T for o, r in zip(out_poses, ref_poses, strict=True))
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    co = np.array([o.t for o in out_poses])
    cr = np.array([r.t for r in ref_poses])
    s = 1.0
    if with_scale:
        so = float(np.sqrt(((co - co.mean(0)) ** 2).sum(1).mean()))
        sr = float(np.sqrt(((cr - cr.mean(0)) ** 2).sum(1).mean()))
        s = sr / so if so > 1e-9 else 1.0
    return Sim3(s, R, cr.mean(0) - s * R @ co.mean(0))


def align_by_poses(out_poses: list[Pose], ref_poses: list[Pose]) -> Pose:
    """Rigid ``T_ref_out`` mapping poses of an output frame onto reference poses of the same
    cameras (``similarity_by_poses`` without scale). Works for rotation-only rigs where the
    camera centres (almost) coincide."""
    if not out_poses:
        return Pose.identity()
    sim = similarity_by_poses(out_poses, ref_poses, with_scale=False)
    return Pose(sim.R, sim.t)


def transform_pose(sim: Sim3, T: Pose) -> Pose:
    M = sim.transform_pose(T.matrix())
    return Pose.from_matrix(M)


def upright_similarity(src: NDArray[Any], dst: NDArray[Any], with_scale: bool = True) -> Sim3:
    """Least-squares similarity of the points ``src`` onto ``dst`` (N, 3) that turns about +z
    only (yaw, scale, translation): both frames are levelled with gravity. Without
    ``with_scale`` the scale is 1 (both frames metric)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    yaw = float(np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                           np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    R = rot_z(yaw)
    var = float(np.sum(a * a))
    s = float(np.sum(b * (a @ R.T)) / var) if var > 1e-12 and with_scale else 1.0
    return Sim3(s, R, md - s * R @ ms)


def robust_upright_similarity(src: NDArray[Any], dst: NDArray[Any], tol: NDArray[Any],
                              samples: int = 2000, seed: int = 0, with_scale: bool = True
                              ) -> tuple[Sim3, NDArray[np.bool_]] | None:
    """``upright_similarity`` of the correspondences ``src`` -> ``dst`` that agree with it within
    ``tol`` (per correspondence, metres): RANSAC over pairs of correspondences at least 10 cm
    apart horizontally (a seeded generator: the same input aligns the same way), then refitted
    to the inliers twice. None without three inliers."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best: tuple[Sim3, NDArray[np.bool_]] | None = None
    for _ in range(samples if n >= 2 else 0):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(src[i, :2] - src[j, :2]) < 0.1:
            continue
        S = upright_similarity(src[[i, j]], dst[[i, j]], with_scale)
        inl = np.linalg.norm(S.apply(src) - dst, axis=1) < tol
        if best is None or inl.sum() > best[1].sum():
            best = (S, inl)
    if best is None or best[1].sum() < 3:
        return None
    inl = best[1]
    for _ in range(2):
        S = upright_similarity(src[inl], dst[inl], with_scale)
        inl = np.linalg.norm(S.apply(src) - dst, axis=1) < tol
    return S, inl

