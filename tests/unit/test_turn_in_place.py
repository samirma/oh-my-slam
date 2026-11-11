"""Regression (round-5 replay of d5a3ee6): a first update without priors that turns in place,
its camera 10 cm ahead of the robot's axis, mono or with depth, by either engine of camera-only
frames, commits: nothing rejected, no ``rotation_inconsistent``, and its committed rotations
are the true ones (map.md §1, §4)."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.mapping.anchored import rot_deg
from oh_my_slam.service import prior_first
from tests.synth.room import base_pose
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    frames_for,
)
from tests.unit.test_service_register_ma import ma_frames


@pytest.mark.parametrize("engine", ["mapanything", "geometric"])
@pytest.mark.parametrize(("depth", "step"), [(False, 15.0), (True, 25.0)])
def test_a_turn_in_place_without_priors_commits(client: Client,  # noqa: F811
                                                monkeypatch: pytest.MonkeyPatch, engine: str,
                                                depth: bool, step: float) -> None:
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, engine)
    poses = [base_pose(step * k) for k in range(8)]
    make = ma_frames if engine == "mapanything" else frames_for
    fr, files = make(client, poses, trust=None, depth=depth, height_above_floor_m=0.5)
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    assert job["quality"]["failed_checks"] == []
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 8
    T0 = np.array(job["inputs"][0]["T_map_cam"])
    for i, T in zip(job["inputs"][1:], poses[1:], strict=True):
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        rel_true = np.linalg.inv(poses[0]) @ T
        assert rot_deg(rel_true[:3, :3].T @ rel[:3, :3]) < 1.0
