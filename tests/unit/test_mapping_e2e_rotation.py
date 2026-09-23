"""Rotation-dominant input (a camera turning in place, like the AiNex robot head): detection,
chunked pose-anchored multi-view poses, then an update anchored on the existing map; plus the
textured mesh, -t single PLY and fixed old poses."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.ply import parse_ply
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.store import MapReader
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room
from tests.synth.scene import look_at

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def turning(n: int, start_deg: float, step_deg: float) -> list[Pose]:
    eye = np.array([0.0, 0.0, 1.3])
    out = []
    for k in range(n):
        a = np.radians(start_deg + k * step_deg)
        out.append(look_at(eye, eye + np.array([np.cos(a), np.sin(a), -0.35])))
    return out


def yaw(T: Pose) -> float:
    f = T.R[:, 2]
    return float(np.degrees(np.arctan2(f[1], f[0])))


def test_rotation_only_map_and_anchored_update(tmp_path: Path) -> None:
    client = FakeClient()
    room = mapping_room()
    first = turning(28, 0.0, 12.0)  # > one multi-view chunk (24)
    add_frames(client, room, first, tmp_path / "a", "a", depth_noise=0.03, seed=4)
    second = turning(8, 5.0, 45.0)
    add_frames(client, room, second, tmp_path / "b", "b", depth_noise=0.03, seed=5)
    mdir = tmp_path / "map"
    msgs: list[str] = []
    res = update(mdir, [tmp_path / "a"], client=client, progress=msgs.append)
    assert any("multi-view fallback (rotation-dominant" in m for m in msgs), msgs
    assert client.calls["multiview"] >= 2  # chunked
    r = MapReader(mdir)
    assert len(r.frames) == 28
    y0 = yaw(r.frames[0].T_map_cam)
    for f, T in zip(r.frames, first, strict=True):
        err = (yaw(f.T_map_cam) - y0 - (yaw(T) - yaw(first[0])) + 180) % 360 - 180
        assert abs(err) < 2.0, (f.name, err)
    centres = np.array([f.T_map_cam.t for f in r.frames])
    assert np.linalg.norm(centres - centres.mean(0), axis=1).max() < 0.3
    old_poses = {f.name: f.T_map_cam.matrix() for f in r.frames}

    # textured mesh
    import trimesh

    scene = trimesh.load(mdir / "mesh" / "mesh.glb", force="scene")
    geom = next(iter(scene.geometry.values()))
    tex = np.asarray(geom.visual.material.baseColorTexture.convert("RGB"))
    assert tex.std() > 10

    # anchored update, -t single -f ply
    res2 = update(mdir, [tmp_path / "b"], mode="single", fmt="ply", client=client,
                  progress=msgs.append)
    assert len(res2.new_frames) == 8
    cloud = parse_ply(res2.payload)
    assert len(cloud) > 1000
    r2 = MapReader(mdir)
    for f in r2.frames:
        if f.name in old_poses:
            np.testing.assert_allclose(f.T_map_cam.matrix(), old_poses[f.name], atol=1e-6)
    new = [f for f in r2.frames if f.update_id == 2]
    for f, T in zip(new, second, strict=True):
        err = (yaw(f.T_map_cam) - y0 - (yaw(T) - yaw(first[0])) + 180) % 360 - 180
        assert abs(err) < 3.0, (f.name, err)
    meta = json.loads((mdir / "map.json").read_text())
    assert meta["updates"][-1]["frames_added"] == res2.new_frames
