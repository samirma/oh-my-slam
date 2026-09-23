"""Multi-view metric reconstruction (MapAnything) behind the inference server.

Poses in and out are camera-to-world with OpenCV axes (MapAnything's convention), which is also
ours, so conversion is only between :class:`Pose` and 4x4 lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.client import protocol as p
from oh_my_slam.client.client import InferenceClient, connect
from oh_my_slam.core.types import Intrinsics, Pose

DEFAULT_RESOLUTION = 518  # gate G6


@dataclass
class MultiviewResult:
    pose: Pose  # camera-to-world
    K: NDArray[np.float64]  # 3x3 on the processed grid
    width: int
    height: int
    depth_path: Path | None
    conf_path: Path | None


def run_multiview(
    images: list[Path],
    work_dir: Path,
    intrinsics: list[Intrinsics | None] | None = None,
    poses: list[Pose | None] | None = None,
    resolution: int = DEFAULT_RESOLUTION,
    want_depth: bool = False,
    client: InferenceClient | None = None,
) -> tuple[list[MultiviewResult], float]:
    """Poses (and optionally depth) for ``images``; returns results and the metric scale."""
    client = client or connect()
    req = p.MultiviewRequest(
        image_paths=[str(Path(i).resolve()) for i in images],
        out_dir=str(work_dir),
        intrinsics=None if intrinsics is None else [
            None if k is None else k.K().tolist() for k in intrinsics
        ],
        poses=None if poses is None else [None if t is None else t.matrix().tolist()
                                          for t in poses],
        poses_metric=True,
        resolution=resolution,
        want_depth=want_depth,
    )
    res = client.multiview(req)
    out = [
        MultiviewResult(
            pose=Pose.from_matrix(np.asarray(v.pose, dtype=np.float64)),
            K=np.asarray(v.intrinsics, dtype=np.float64),
            width=v.width,
            height=v.height,
            depth_path=Path(v.depth_path) if v.depth_path else None,
            conf_path=Path(v.conf_path) if v.conf_path else None,
        )
        for v in res.views
    ]
    return out, res.metric_scale
