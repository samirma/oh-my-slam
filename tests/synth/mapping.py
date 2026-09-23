"""Synthetic mapping inputs: render a room from several poses and register the renders with the
fake inference client (true depth with optional per-frame scale noise, instance masks from the
renderer's surface ids)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oh_my_slam.core.types import Intrinsics, Pose
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.scene import Room, orbit_poses, render

K = Intrinsics(300.0, 300.0, 200.0, 150.0, 400, 300)


def add_frames(client: FakeClient, room: Room, poses: list[Pose], folder: Path, prefix: str,
               depth_noise: float = 0.0, seed: int = 0) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i, pose in enumerate(poses):
        r = render(room, pose, K)
        inst = []
        for k, box in enumerate(room.boxes):
            m = r.ids == k + 2
            if m.sum() > 150:
                inst.append(FakeInstance(box.label, 0.85, m))
        scale = 1.0 + (rng.uniform(-depth_noise, depth_noise) if depth_noise else 0.0)
        up = pose.R.T @ np.array([0.0, 0.0, 1.0])
        path = folder / f"{prefix}_{i:03d}.png"
        client.add(path, r.rgb, FakeFrame((r.depth * scale).astype(np.float32), K, up, inst,
                                          pose=pose))
        paths.append(path)
    return paths


def ring(n: int, start: float = 0.0, span: float = 2 * np.pi, radius: float = 2.2,
         height: float = 1.5) -> list[Pose]:
    return orbit_poses(n, radius=radius, height=height, target_z=0.4, start=start, span=span)


def mapping_room() -> Room:
    """Boxes well inside the camera ring so every side is observed."""
    from tests.synth.scene import Box

    return Room(boxes=[
        Box(np.array([0.9, 0.6, 0.4]), np.array([0.9, 0.5, 0.8]), 0.3, (220, 40, 40), "cabinet"),
        Box(np.array([-0.9, -0.5, 0.25]), np.array([0.6, 0.6, 0.5]), -0.4, (40, 180, 60), "box"),
        Box(np.array([0.1, -1.0, 0.4]), np.array([1.4, 0.45, 0.8]), 0.0, (50, 70, 210), "sofa"),
    ])
