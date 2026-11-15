"""A monocular scale reads the constant in every response (map.md §1, oms/2.12): a version, a
recorded pose or a job record stored before with the nominal 0.10 answers 0.65 everywhere; the
stored files are not rewritten and other scale sources read as stored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from oh_my_slam.service.jobs import Job
from tests.unit.test_service_nav import Nav, _png, nav, room_map  # noqa: F401 (fixtures)


def _store_scale(app: Nav, name: str, source: str, sigma: float) -> None:
    """The stored version records read as if an earlier release committed them."""
    for p in (app.service.store.dir(name) / "versions").glob("*.json"):
        if p.stem.isdigit():
            d = json.loads(p.read_text())
            d["scale_source"], d["scale_sigma_rel"] = source, sigma
            p.write_text(json.dumps(d))
    derived = app.service.derived  # derived before the edit: read again, as after a restart
    assert derived is not None
    with derived._lock:
        derived._cache.clear()
        derived._latest.clear()


def test_every_read_of_a_monocular_version_reports_the_constant(
        room_map: tuple[Nav, Any, Any]) -> None:  # noqa: F811
    from tests.synth.scene import look_at, render

    app, room, K = room_map
    P = look_at(np.array([1.5, 0.4, 1.3]), np.array([0.1, 0, 0.5]))
    png = _png(render(room, P, K).rgb)
    req = {"frames": [{"name": "q", "image": "q", "camera_id": "cam"}], "record": True}

    def reads() -> list[tuple[str, float]]:
        docs = [app.get("/api/maps/room", "MapSummary"), app.get("/api/maps/room/frames", "Frames"),
                app.get("/api/maps/room/grid", "Grid"),
                app.get("/api/maps/room/objects", "Objects"),
                app.get("/api/maps/room/birdseye", "Birdseye"),
                app.get("/api/maps/room/points?min=-3,-3,-1&max=3,3,3", "Points"),
                app.locate("room", req, {"q": png}),
                app.get("/api/maps/room/pose", "Pose")]
        return [(d["scale_source"], d["scale_sigma_rel"]) for d in docs]

    # a version that is not monocular reads as stored
    _store_scale(app, "room", "camera_height", 0.0123)
    assert set(reads()) == {("camera_height", 0.0123)}
    # a monocular one stored with the nominal 0.10 reads the constant; the file keeps it
    _store_scale(app, "room", "monocular", 0.10)
    assert set(reads()) == {("monocular", 0.65)}
    stored = json.loads((app.service.store.dir("room") / "versions" / "1.json").read_text())
    assert stored["scale_sigma_rel"] == 0.10
    # the recorded pose's own copy, and the version's it falls back on
    path = app.service.store.dir("room") / "pose.json"
    old = json.loads(path.read_text())
    old["scale_sigma_rel"] = 0.10
    path.write_text(json.dumps(old))
    assert app.get("/api/maps/room/pose", "Pose")["scale_sigma_rel"] == 0.65
    del old["scale_source"], old["scale_sigma_rel"]
    path.write_text(json.dumps(old))
    assert app.get("/api/maps/room/pose", "Pose")["scale_sigma_rel"] == 0.65
    # a job record an earlier release wrote (a restart loads it)
    vid = app.get("/api/maps/room/versions", "Versions")["versions"][0]["job_id"]
    job = app.get(f"/api/jobs/{vid}", "Job")
    rec = {**job, "created": 1.0, "ended": 2.0, "frames": 1,
           "scale_source": "monocular", "scale_sigma_rel": 0.10}
    assert Job.from_record(rec).body()["scale_sigma_rel"] == 0.65
    rec["scale_source"] = "priors"
    assert Job.from_record(rec).body()["scale_sigma_rel"] == 0.10
