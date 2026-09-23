"""TSDF fusion of aligned depth maps (Open3D ``VoxelBlockGrid``, CPU) and mesh extraction."""

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


def _downsample(depth: NDArray[Any], rgb: NDArray[Any], K: NDArray[Any], max_side: int
                ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    h, w = depth.shape
    step = int(np.ceil(max(h, w) / max_side))
    if step <= 1:
        return depth, rgb, K
    d = depth[::step, ::step]
    c = rgb[::step, ::step]
    K2 = K.copy()
    K2[0, 0] /= step
    K2[1, 1] /= step
    K2[0, 2] /= step
    K2[1, 2] /= step
    return d, c, K2


@dataclass
class FusionStats:
    frames: int = 0
    seconds: float = 0.0


class TsdfFusion:
    def __init__(self, voxel_size: float, depth_max: float, block_count: int = 40000) -> None:
        import open3d as o3d
        import open3d.core as o3c

        self._o3d = o3d
        self._o3c = o3c
        self.voxel_size = voxel_size
        self.depth_max = depth_max
        self.trunc = TRUNC_VOXELS * voxel_size
        self.device = o3c.Device("CPU:0")
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=8,
            block_count=block_count,
            device=self.device,
        )
        self.stats = FusionStats()

    def integrate(self, depth: NDArray[Any], rgb: NDArray[np.uint8], K: NDArray[Any],
                  T_world_cam: Pose, max_side: int = FUSION_MAX_SIDE) -> None:
        """Integrate one frame; ``depth`` in metres with 0 for invalid/latest-wins-removed pixels."""
        import time

        o3d, o3c = self._o3d, self._o3c
        t0 = time.perf_counter()
        d, c, Kd = _downsample(np.asarray(depth, np.float32), np.asarray(rgb, np.uint8),
                               np.asarray(K, np.float64), max_side)
        d = np.where(np.isfinite(d) & (d > 0) & (d < self.depth_max), d, 0.0).astype(np.float32)
        if not (d > 0).any():
            return
        depth_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d)))
        color_img = o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(c.astype(np.float32) / 255.0))
        )
        intr = o3c.Tensor(Kd, dtype=o3c.float64)
        extr = o3c.Tensor(T_world_cam.inverse().matrix(), dtype=o3c.float64)
        coords = self.vbg.compute_unique_block_coordinates(
            depth_img, intr, extr, depth_scale=1.0, depth_max=self.depth_max,
            trunc_voxel_multiplier=TRUNC_VOXELS,
        )
        self.vbg.integrate(coords, depth_img, color_img, intr, intr, extr, depth_scale=1.0,
                           depth_max=self.depth_max, trunc_voxel_multiplier=TRUNC_VOXELS)
        self.stats.frames += 1
        self.stats.seconds += time.perf_counter() - t0

    def extract_mesh(self, weight_threshold: float = 1.0) -> Any:
        """Legacy ``open3d.geometry.TriangleMesh`` with vertex colours in [0, 1]."""
        mesh = self.vbg.extract_triangle_mesh(weight_threshold=weight_threshold)
        return mesh.to_legacy()

    def extract_points(self, weight_threshold: float = 1.0) -> tuple[NDArray[Any], NDArray[Any]]:
        pcd = self.vbg.extract_point_cloud(weight_threshold=weight_threshold)
        pts = pcd.point.positions.numpy() if "positions" in pcd.point else np.zeros((0, 3))
        cols = pcd.point.colors.numpy() if "colors" in pcd.point else np.zeros((0, 3))
        return pts, cols
