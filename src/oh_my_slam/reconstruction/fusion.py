"""TSDF fusion of aligned depth maps (Open3D ``VoxelBlockGrid``, CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.types import Pose

MIN_VOXEL = 0.01
MAX_VOXEL = 0.04
TRUNC_VOXELS = 4.0
FUSION_MAX_SIDE = 1024  # depth grid used for integration (gate G7: ~2-8 ms/frame at 1.5 cm)


def choose_voxel_size(median_depth: float, fraction: float = 0.005) -> float:
    """clamp(0.5% of the median depth, 1-4 cm)."""
    return float(np.clip(fraction * median_depth, MIN_VOXEL, MAX_VOXEL))


def fusion_step(shape: tuple[int, ...], max_side: int = FUSION_MAX_SIDE) -> int:
    """The subsampling of a depth grid of ``shape`` (rows, cols) for integration (1: none)."""
    return max(1, int(np.ceil(max(int(shape[0]), int(shape[1])) / max_side)))


def _downsample(depth: NDArray[Any], K: NDArray[Any], max_side: int
                ) -> tuple[NDArray[Any], NDArray[Any]]:
    step = fusion_step(depth.shape, max_side)
    if step <= 1:
        return depth, K
    d = depth[::step, ::step]
    K2 = K.copy()
    K2[0, 0] /= step
    K2[1, 1] /= step
    K2[0, 2] /= step
    K2[1, 2] /= step
    return d, K2


@dataclass
class FusionStats:
    frames: int = 0
    seconds: float = 0.0


class TsdfFusion:
    def __init__(self, voxel_size: float, depth_max: float, block_count: int = 40000,
                 trunc_voxels: float = TRUNC_VOXELS) -> None:
        import open3d as o3d
        import open3d.core as o3c

        self._o3d = o3d
        self._o3c = o3c
        self.voxel_size = voxel_size
        self.depth_max = depth_max
        self.trunc_voxels = float(trunc_voxels)
        self.trunc = self.trunc_voxels * voxel_size
        self.device = o3c.Device("CPU:0")
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1), (1)),
            voxel_size=voxel_size,
            block_resolution=8,
            block_count=block_count,
            device=self.device,
        )
        self.stats = FusionStats()

    def integrate(self, depth: NDArray[Any], K: NDArray[Any], T_world_cam: Pose,
                  max_side: int = FUSION_MAX_SIDE, depth_max: float | None = None) -> None:
        """Integrate one frame; ``depth`` in metres with 0 for invalid/latest-wins-removed pixels,
        up to ``depth_max`` (default: the fusion's)."""
        import time

        o3d, o3c = self._o3d, self._o3c
        t0 = time.perf_counter()
        cut = float(self.depth_max if depth_max is None else depth_max)
        d, Kd = _downsample(np.asarray(depth, np.float32), np.asarray(K, np.float64), max_side)
        d = np.where(np.isfinite(d) & (d > 0) & (d < cut), d, 0.0).astype(np.float32)
        if not (d > 0).any():
            return
        depth_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d)))
        intr = o3c.Tensor(Kd, dtype=o3c.float64)
        extr = o3c.Tensor(T_world_cam.inverse().matrix(), dtype=o3c.float64)
        coords = self.vbg.compute_unique_block_coordinates(
            depth_img, intr, extr, depth_scale=1.0, depth_max=cut,
            trunc_voxel_multiplier=self.trunc_voxels,
        )
        self.vbg.integrate(coords, depth_img, intr, extr, depth_scale=1.0,
                           depth_max=cut, trunc_voxel_multiplier=self.trunc_voxels)
        self.stats.frames += 1
        self.stats.seconds += time.perf_counter() - t0

    def extract_points(self, weight_threshold: float = 1.0) -> NDArray[Any]:
        empty = np.zeros((0, 3))
        if self.vbg.hashmap().size() == 0:
            return empty
        try:
            pcd = self.vbg.extract_point_cloud(weight_threshold=weight_threshold)
        except RuntimeError as e:  # Open3D raises instead of returning no surface points
            if "shape {0}" in str(e):
                return empty
            raise
        return pcd.point.positions.numpy() if "positions" in pcd.point else empty
