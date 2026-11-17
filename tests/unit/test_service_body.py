"""Body links through the service (oms/2.15; api.md §4 "Body links"): the request's validation
(``service.app.check_bodies``), the input status's and the dry run's ``body_link``, and the older
mapper's ``engine`` report."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import body as bl
from oh_my_slam.service import prior_first
from oh_my_slam.service.app import check_bodies
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.update import FrameWork, Updater
from tests.synth.room import base_pose
from tests.unit import test_service_api as api_tests
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import CAM, client, dry, frames_for  # noqa: F401

R_BC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    M = np.eye(4)
    M[:3, :3] = [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                 [0.0, 0.0, 1.0]]
    return M


def mount(lever: float = 0.1) -> dict[str, Any]:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R_BC, [lever, 0.0, 0.5]
    return {"T_body_cam": M.tolist(), "source": "measured"}


def motion(frm: str, M: np.ndarray | None = None, ys: str = "imu",
           ts: str = "lidar_match") -> dict[str, Any]:
    return {"from": frm, "T_from_body": (np.eye(4) if M is None else M).tolist(),
            "yaw_source": ys, "sigma_yaw_deg": 1.0, "translation_source": ts,
            "sigma_pos_m": 0.005}


def fbody(frm: str | None = None, **kw: Any) -> dict[str, Any]:
    b: dict[str, Any] = {"image_stamp_s": 1.0, "standing": True, **kw}
    if frm is not None:
        b["motion"] = motion(frm, Rz(15.0))
    return b


# -- through the app ------------------------------------------------------------------------------


def test_a_turn_with_body_links_reports_them(client: Client) -> None:  # noqa: F811
    """A wheeled base turning in place with a fixed mount (the camera's ``body_mount``) and
    linked frames: every input with a motion has its ``body_link`` (schema-checked), as the
    dry run predicted it, the first frame starting the chain."""
    poses = [base_pose(0), base_pose(15), base_pose(30)]
    fr, files = frames_for(client, poses, tag="t")
    fr[0]["body"] = fbody()
    fr[1]["body"] = fbody("t0")
    fr[2]["body"] = fbody("t1")
    cams = {"cam": {**CAM, "body_mount": mount()}}
    pred = dry(client, "m", fr, files, cameras=cams)
    rows = {r["name"]: r for r in pred["frames"]}
    assert "body_link" not in rows["t0"]
    assert rows["t1"]["body_link"]["from"] == "t0"
    assert rows["t1"]["body_link"]["baseline_m"] == pytest.approx(
        2 * 0.1 * math.sin(math.radians(7.5)), rel=1e-3)
    job = client.update("m", fr, files, cameras=cams)
    ins = {r["name"]: r for r in job["inputs"]}
    assert "body_link" not in ins["t0"]
    for n in ("t1", "t2"):
        link = ins[n]["body_link"]
        assert link["from"] == f"t{int(n[1]) - 1}"
        assert link["state"] in ("used", "unused")
    # a later job links from a stored frame; its camera keeps its mount (one of its values)
    fr2, files2 = frames_for(client, [base_pose(45)], tag="u")
    fr2[0]["body"] = fbody("t2")
    job = client.update("m", fr2, files2)
    assert job["inputs"][0]["body_link"]["from"] == "t2"
    doc = client.post_update("m", {"frames": fr2, "cameras": {"cam": {**CAM, "body_mount": mount(
        0.2)}}}, files2, status=409)
    assert doc["error"]["code"] == "camera_conflict"


def test_a_frame_the_checks_reject_still_reports_its_link(client: Client) -> None:  # noqa: F811
    from tests.fakes.service import plain

    fr, files = frames_for(client, [base_pose(0)], tag="a")
    fr[0]["body"] = fbody(mount=mount(), mount_stamp_s=1.0)
    files["p"] = plain(320, 240)
    fr.append({"name": "p", "image": "p", "camera_id": "cam",
               "body": fbody("a0", mount=mount(), mount_stamp_s=5.0, standing=False)})
    job = client.update("m", fr, files, cameras={"cam": CAM})
    ins = {r["name"]: r for r in job["inputs"]}
    assert ins["p"]["status"] == "rejected"
    assert ins["p"]["body_link"]["reason"] == "stale_mount"


def bad(fr: list[dict[str, Any]], files: dict[str, bytes], cams: dict[str, Any] | None,
        c: Client, field: str) -> None:
    req: dict[str, Any] = {"frames": fr}
    if cams is not None:
        req["cameras"] = cams
    doc = c.post_update("m", req, files, status=400)
    assert doc["error"]["code"] == "invalid_request"
    assert doc["error"]["details"]["field"] == field, doc


def test_invalid_bodies_name_the_failing_field(client: Client) -> None:  # noqa: F811
    poses = [base_pose(0), base_pose(15)]
    cams = {"cam": CAM}

    def two(b0: dict[str, Any] | None, b1: dict[str, Any] | None, **cam: Any
            ) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
        fr, files = frames_for(client, poses, tag="f")
        if b0 is not None:
            fr[0]["body"] = b0
        if b1 is not None:
            fr[1]["body"] = b1
        return fr, files, {"cam": {**CAM, **cam}}

    m = {"mount": mount(), "mount_stamp_s": 1.0}
    fr, files, cams = two(fbody(), None)
    bad(fr, files, cams, client, "frames[0].body.mount")
    skew = mount()
    skew["T_body_cam"] = (2 * np.eye(4)).tolist()
    fr, files, cams = two(fbody(mount=skew, mount_stamp_s=1.0), None)
    bad(fr, files, cams, client, "frames[0].body.mount.T_body_cam")
    fr, files, cams = two(None, None, body_mount=skew)
    bad(fr, files, cams, client, "cameras.cam.body_mount.T_body_cam")
    b1 = fbody("f0", **m)
    b1["motion"]["T_from_body"] = (2 * np.eye(4)).tolist()
    fr, files, cams = two(fbody(**m), b1)
    bad(fr, files, cams, client, "frames[1].body.motion.T_from_body")
    fr, files, cams = two(fbody(**m), fbody("f1", **m))
    bad(fr, files, cams, client, "frames[1].body.motion.from")  # itself
    fr, files, cams = two(None, fbody("f0", **m))
    bad(fr, files, cams, client, "frames[1].body.motion.from")  # a frame without body
    fr, files, cams = two(fbody(**m), fbody("nobody", **m))
    bad(fr, files, cams, client, "frames[1].body.motion.from")  # unknown
    fr, files, cams = two(fbody("f1", **m), fbody("f0", **m))
    bad(fr, files, cams, client, "frames[0].body.motion.from")  # a cycle
    fr, files, cams = two(fbody(**m), fbody("f0", **m))
    fr[0]["camera_id"] = "cam2"
    cams["cam2"] = CAM
    bad(fr, files, cams, client, "frames[1].body.motion.from")  # another camera
    still = fbody("f0", **m)
    still["motion"] = motion("f0", np.eye(4), "none", "none")
    fr, files, cams = two(fbody(**m, standing=False), still)
    bad(fr, files, cams, client, "frames[1].body.motion")  # the from frame moves
    half = fbody("f0", **m)
    half["motion"] = motion("f0", np.eye(4), "none", "imu")
    fr, files, cams = two(fbody(**m), half)
    bad(fr, files, cams, client, "frames[1].body.motion")
    moved = fbody("f0", **m)
    moved["motion"] = motion("f0", Rz(1.0), "none", "none")
    fr, files, cams = two(fbody(**m), moved)
    bad(fr, files, cams, client, "frames[1].body.motion")
    fr, files, cams = two(fbody(**m), {**still, "standing": False})
    bad(fr, files, cams, client, "frames[1].body.motion")
    fr, files, cams = two(fbody(**m), still)  # a head pan with the body still: valid
    job = client.update("m", fr, files, cameras=cams)
    assert job["inputs"][1]["body_link"]["from"] == "f0"


def test_check_bodies_against_stored_and_queued_frames() -> None:
    cams = {"cam": {**CAM, "body_mount": mount()}, "c2": CAM}

    def req(*frames: dict[str, Any]) -> dict[str, Any]:
        return {"frames": [{"name": f"n{i}", "image": "x", "camera_id": "cam", **f}
                           for i, f in enumerate(frames)]}

    got = check_bodies(req({"body": fbody("s")}), cams, {"s": "cam"}, {})
    assert got[0].motion is not None and got[0].mount is not None and got[0].mount.fixed
    with pytest.raises(ApiError) as e:
        check_bodies(req({"body": fbody("s")}), cams, {"s": "c2"}, {})
    assert e.value.details["field"] == "frames[0].body.motion.from"
    still = fbody("q")
    still["motion"] = motion("q", np.eye(4), "none", "none")
    queued = FrameWork(0, "q", "cam", b"", body=bl.body_of("q", "cam", fbody(), mount()))
    assert check_bodies(req({"body": still}), cams, {}, {"q": queued})[0].motion is not None
    queued.body = bl.body_of("q", "cam", fbody(standing=False), mount())
    with pytest.raises(ApiError):
        check_bodies(req({"body": still}), cams, {}, {"q": queued})
    queued.body = None
    with pytest.raises(ApiError):
        check_bodies(req({"body": still}), cams, {}, {"q": queued})
    # a stored frame of unknown camera (an older record): accepted, judged when the job runs
    assert check_bodies(req({"body": fbody("s")}), cams, {"s": ""}, {})


def test_the_older_mapper_reports_engine() -> None:
    body = bl.body_of("b", "cam", fbody("a"), mount())
    work = FrameWork(0, "b", "cam", b"", body=body)
    fr = SimpleNamespace(work=work, reason="no_overlap", internal=None, unconfirmed=None,
                         pose=None, sigma_pos_m=None, sigma_rot_deg=None, prior_residual=None,
                         scene_m=None, duplicate_of=None, detail=None)
    row = Updater._input(fr)  # type: ignore[arg-type]
    assert row["body_link"]["reason"] == "engine" and row["body_link"]["from"] == "a"
    assert prior_first.body_link(fr, None, None)["reason"] == "engine"  # type: ignore[arg-type,index]
    fr.work = FrameWork(0, "b", "cam", b"")
    assert prior_first.body_link(fr, None, None) is None  # type: ignore[arg-type]
    assert "body_link" not in Updater._input(fr)  # type: ignore[arg-type]


def test_the_older_mapper_through_the_app(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from oh_my_slam.service.app import Service, create_app
    from oh_my_slam.service.jobs import Jobs
    from oh_my_slam.service.store import Store
    from oh_my_slam.service.timings import Timings
    from tests.fakes.service import FakeBackend, FakeInference

    store = Store(tmp_path / "data")
    store.create()
    backend = FakeBackend()
    timings = Timings()
    jobs = Jobs(store.jobs, Updater(store, backend, timings, prior_first=False),
                timings.predict_update)
    jobs.start()
    service = Service(store, jobs, FakeInference(), backend.require_inference, timings=timings,
                      version="test")
    try:
        with TestClient(create_app(service)) as http:
            c = Client(http, backend, service)
            fr, files, _ = api_tests.frames_of(c, 2)
            fr[0]["body"] = fbody(mount=mount(), mount_stamp_s=1.0)
            fr[1]["body"] = fbody(fr[0]["name"], mount=mount(), mount_stamp_s=1.0)
            job = c.update("m", fr, files, cameras={"cam": api_tests.CAMERA})
            assert job["inputs"][1]["body_link"]["reason"] == "engine"
    finally:
        jobs.stop(timeout=10)
