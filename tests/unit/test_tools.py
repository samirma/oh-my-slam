"""Offline parts of the developer tools (cloud metrics)."""

from __future__ import annotations

import numpy as np

from oh_my_slam.tools.cloud_quality import planar_patches


def test_cloud_quality_planar_patches_detects_layers() -> None:
    rng = np.random.default_rng(0)
    xy = rng.random((200_000, 2)) * 2.0
    one = np.c_[xy, rng.normal(0, 0.002, len(xy))]
    layered = one.copy()
    layered[::3, 2] += 0.03  # a second copy of the floor 3 cm above the first
    single, double = planar_patches(one), planar_patches(layered)
    assert single["thickness_p50_mm"] < 3 and single["off_plane_pct"] < 1
    assert double["thickness_p50_mm"] > 10 and double["off_plane_pct"] > 20
