"""A one-image map (identity pose, C14) extended by a short sequence (< 3 keyframes → re-map)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.geometry import angle_between_deg
from oh_my_slam.mapping.api import update
from oh_my_slam.mapping.store import MapReader
from oh_my_slam.schema.validate import validation_errors
from tests.fakes.client import FakeClient
from tests.synth.mapping import add_frames, mapping_room, ring

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


def test_single_image_map_then_extension(tmp_path: Path) -> None:
    client = FakeClient()
    room = mapping_room()
    poses = ring(10, span=np.pi)
    imgs = add_frames(client, room, poses, tmp_path / "in", "p", depth_noise=0.02, seed=7)
    mdir = tmp_path / "map"
    res = update(mdir, [imgs[0]], client=client, progress=lambda m: None)
    doc = json.loads(res.payload)
    assert validation_errors(doc) == []
    r = MapReader(mdir)
    assert len(r.frames) == 1 and r.frames[0].pose_source == "identity"
    np.testing.assert_allclose(r.frames[0].T_map_cam.t, 0, atol=1e-9)
    # map z is up: the first camera's up vector maps to +z
    up_map = r.frames[0].T_map_cam.R @ (poses[0].R.T @ np.array([0.0, 0.0, 1.0]))
    assert angle_between_deg(up_map, [0, 0, 1]) < 2.0
    T0 = r.frames[0].T_map_cam.matrix()

    res2 = update(mdir, imgs[1:], client=client, progress=lambda m: None)
    r2 = MapReader(mdir)
    assert len(r2.frames) >= 9
    np.testing.assert_allclose(r2.frames[0].T_map_cam.matrix(), T0, atol=1e-9)
    # relative geometry: distance between the first two cameras matches the synthetic truth
    d_est = np.linalg.norm(r2.frames[1].T_map_cam.t - r2.frames[0].T_map_cam.t)
    d_true = np.linalg.norm(poses[1].t - poses[0].t)
    assert d_est == pytest.approx(d_true, rel=0.15)
    assert json.loads((mdir / "map.json").read_text())["updates"][-1]["notes"]["remap_small"]
    assert res2.rejected == []
