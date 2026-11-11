"""Locate on a map whose scale comes from camera heights (api.md §6 "Camera-height scale"): the
height disagreement joins the position sigma, ``height_consistent`` reports the 3-sigma test,
and a height-inconsistent hypothesis ranks below a consistent rival."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.synth.room import base_pose
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import client  # noqa: F401


def test_a_locate_on_a_camera_height_scale(client: Client) -> None:  # noqa: F811
    """api.md §6 "Camera-height scale": the located position sigma takes the height
    disagreement, ``height_consistent`` says whether it is within 3 sigma; null on other
    scales."""
    from tests.fakes.service import multipart
    from tests.unit.test_service_prior_first import CAM, frames_for, shot

    cam = {**CAM, "height_above_floor_m": 0.5, "height_sigma_m": 0.01}
    fr, files = frames_for(client, [base_pose(15.0 * k) for k in range(4)], tag="a",
                           depth=False, trust=None)
    job = client.update("m", fr, files, cameras={"cam": cam})
    assert job["state"] == "committed" and job["scale_source"] == "camera_height"
    img, _, _ = shot(base_pose(20))

    def locate(height: float) -> dict[str, Any]:
        data, ctype = multipart({"frames": [{"name": "q", "image": "q", "camera_id": "cam",
                                             "height_above_floor_m": height,
                                             "height_sigma_m": 0.01}],
                                 "record": False}, {"q": img})
        r = client.http.post("/api/maps/m/locate", content=data,
                             headers={"content-type": ctype})
        assert r.status_code == 200, r.text
        out: dict[str, Any] = client.check(r, "Locate")["frames"][0]
        return out

    good, off = locate(0.5), locate(1.5)
    assert good["status"] == off["status"] == "located"
    assert good["height_consistent"] is True and off["height_consistent"] is False
    assert off["sigma_pos_m"] >= abs(np.array(off["T_map_cam"])[2, 3] - 1.5) - 0.01  # floor z
    assert off["sigma_pos_m"] > good["sigma_pos_m"]


def test_height_consistency_ranks_hypotheses() -> None:
    from oh_my_slam.mapping import relocalize as rl

    q = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), np.eye(3), 640,
                 480)
    low, high = base_pose(0, height=0.5), base_pose(0, 1.0, height=1.2)
    assert rl.height_consistent(q, low, 0.01) is None
    assert rl.height_sigma(q, high, 0.01) == 0.01
    q.camera_z, q.camera_z_sigma, q.height_scale = 0.5, 0.01, True
    assert rl.height_consistent(q, low, 0.01) is True
    assert rl.height_consistent(q, high, 0.01) is False
    assert rl.height_sigma(q, high, 0.01) == pytest.approx(math.hypot(0.01, 0.7))
    # without a prior: the consistent one, though weaker
    assert rl.choose(q, [high, low], [100, 60], [(0.01, 0.5)] * 2) == 1
    # with a prior near both: the consistent one ranks first
    q.prior, q.prior_sigma_pos_m, q.prior_sigma_rot_deg = high, 1.0, 5.0
    assert rl.choose(q, [high, low], [100, 60], [(0.01, 0.5)] * 2) == 1
