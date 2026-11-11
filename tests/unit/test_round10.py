"""Round 10: the supporting frames' timing (P19), versions and revert (O2), ignored pixels and
the label vocabulary (P18, oms/2.3), and object extents over many views."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import Drift
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.store import Store
from oh_my_slam.service.update import Updater
from tests.synth.room import base_pose
from tests.unit.test_anchored import frame, new_map
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import CAM, frames_for
from tests.unit.test_service_prior_first import client  # noqa: F401

# -- P19: a frame joins the drift support at its version's commit -------------------------------


def _far(T: np.ndarray, dx: float) -> np.ndarray:
    W = T.copy()
    W[:3, 3] += [dx, 0.0, 0.0]
    return W


def test_a_confirmed_frame_supports_only_from_its_commit() -> None:
    """map.md §1 "Supporting frames" (P19): a drifting frame the job confirms is a supporting
    frame of the committed state, but the job's later predictions (a retry, a rerun, the close)
    still grow the sigma from the support as it was; the next job counts it."""
    d = Drift(np.eye(4), np.eye(4), fit_sigma=(0.01, 0.1), support=[[0.0, 0.0, 0.0]])
    P = _far(np.eye(4), 4.0)
    before = d.predict(P, 0.02, 1.0)
    d.confirm(P, P)
    assert P[:3, 3].tolist() in d.support and d.to_json()["support"] == d.support
    assert "fresh" not in d.to_json()
    assert d.predict(P, 0.02, 1.0)[1:] == before[1:]  # its own bound is not loosened
    d.fresh = []  # the next job (``begin_job``)
    assert d.predict(P, 0.02, 1.0)[1] < before[1]


def test_the_prior_first_job_never_reads_its_own_confirmations(tmp_path: Path) -> None:
    """P19 on the prior-first engine: after a held job's registration closed its drift
    (``_close_drift``), its confirmed frame is in the support but not in what the job's later
    passes predict with; ``begin_job`` of the next job lets it support."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), p="drifting")])
    m.drift = Drift(m.keyframes[0].T, m.keyframes[0].T, fit_sigma=(0.01, 0.1),
                    support=[[50.0, 50.0, 50.0]])
    f = frame("d1", base_pose(10), p="drifting")
    pls = m.update([f], hold=True)
    assert pls[0].status == "registered"
    c = np.asarray(f.prior["T_map_cam"])[:3, 3].tolist()  # type: ignore[index]
    assert c in m.drift.support and c in m.drift.fresh
    in_job = m._predict(f)
    assert in_job is not None
    m.end_job(pls, False)
    m.begin_job()
    after = m._predict(f)
    assert after is not None and after[1] < in_job[1]
    assert an.MAX_EVIDENCE == 500


# -- O2: versions and revert --------------------------------------------------------------------


def _revert(c: Client, to: Any, status: int = 200, raw: bytes | None = None) -> Any:
    r = c.http.post("/api/maps/m/revert", content=raw if raw is not None else json.dumps(
        {"to_version": to}).encode(), headers={"content-type": "application/json"})
    assert r.status_code == status, r.text
    return c.check(r, "MapSummary")


def _three(c: Client) -> None:
    for k, tag in enumerate("abc"):
        fr, files = frames_for(c, [base_pose(25.0 * k)], tag=tag)
        job = c.update("m", fr, files, cameras={"cam": CAM} if k == 0 else None)
        assert job["state"] == "committed" and job["result_version"] == k + 1


def test_versions_list_and_revert_commit_a_copy(client: Client) -> None:  # noqa: F811
    """api.md §5, map.md §2 (O2): the kept versions; a revert commits a new version equal to
    its target (frames, poses, scale, cameras), every read names it; the next update starts
    from it in the same map frame (never re-gauged)."""
    _three(client)
    doc = client.get("/api/maps/m/versions", "Versions")
    assert doc["latest_version"] == 3 and [v["version"] for v in doc["versions"]] == [1, 2, 3]
    assert {v["kind"] for v in doc["versions"]} == {"update"}
    assert [v["frames"] for v in doc["versions"]] == [1, 2, 3]
    assert all(v["job_id"] for v in doc["versions"])
    v1 = client.get("/api/maps/m/frames?version=1", "Frames")
    s = _revert(client, 1)
    assert s["map_version"] == 4 and s["latest_version"] == 4 and s["frames"] == 1
    assert s["cameras"] == client.get("/api/maps/m?version=1", "MapSummary")["cameras"]
    v4 = client.get("/api/maps/m/frames", "Frames")
    assert v4["map_version"] == 4 and v4["frames"] == v1["frames"]
    assert v4["scale_source"] == v1["scale_source"]
    info = client.get("/api/maps/m/versions", "Versions")["versions"][-1]
    assert (info["kind"], info["job_id"], info["reverted_to"]) == ("revert", None, 1)
    g = client.get("/api/maps/m/grid", "Grid")
    assert g["map_version"] == 4
    assert client.get("/api/maps/m/objects", "Objects")["map_version"] == 4
    # the next update from the reverted version: its frames keep their poses
    fr, files = frames_for(client, [base_pose(25.0)], tag="d")
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and job["base_version"] == 4
    v5 = client.get("/api/maps/m/frames", "Frames")
    assert [f["name"] for f in v5["frames"]] == ["a0", "d0"]
    assert v5["frames"][0]["T_map_cam"] == v1["frames"][0]["T_map_cam"]
    # a revert to the latest copies it too; a revert to a revert as well
    assert _revert(client, 4)["map_version"] == 6
    assert client.get("/api/maps/m/frames", "Frames")["frames"] == v1["frames"]


def test_revert_refusals(client: Client) -> None:  # noqa: F811
    r = client.http.post("/api/maps/m/revert", json={"to_version": 1})
    assert r.status_code == 404 and client.check(r, None)["error"]["code"] == "map_not_found"
    assert client.get("/api/maps/m/versions", status=404)["error"]["code"] == "map_not_found"
    _three(client)
    for to in (4, 9):
        e = _revert(client, to, 400)["error"]
        assert e["code"] == "invalid_request" and e["details"]["field"] == "to_version"
    e = _revert(client, 0, 400)["error"]
    assert e["code"] == "invalid_request"
    e = _revert(client, None, 400, raw=b"{not json")["error"]
    assert e["details"]["field"] == "body"
    (client.service.store.dir("m") / "versions" / "1.json").unlink()
    e = _revert(client, 1, 410)["error"]
    assert e["code"] == "version_gone" and e["details"]["field"] == "to_version"
    assert [v["version"] for v in client.get("/api/maps/m/versions", "Versions")["versions"]] \
        == [2, 3]


def test_revert_is_busy_while_a_job_registers_and_queued_jobs_follow_it(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """api.md §5: ``map_busy`` (409, retryable) while a job of the map is registering; a job
    queued meanwhile starts from the reverted version."""
    _three(client)
    gate, started = threading.Event(), threading.Event()
    real = Updater._run

    def slow(self: Updater, *a: Any) -> None:
        started.set()
        gate.wait(20)
        real(self, *a)

    monkeypatch.setattr(Updater, "_run", slow)
    fr, files = frames_for(client, [base_pose(60.0)], tag="e")
    j1 = client.post_update("m", {"frames": fr}, files)
    assert started.wait(10)
    e = _revert(client, 1, 409)["error"]
    assert e["code"] == "map_busy" and e["retryable"] and e["details"]["job_id"] == j1["job_id"]
    gate.set()
    assert client.wait(j1["job_id"])["state"] == "committed"
    monkeypatch.setattr(Updater, "_run", real)
    # a queued job waits for the revert, then starts from it
    jobs = client.service.jobs
    with jobs.holding("m"):
        fr, files = frames_for(client, [base_pose(70.0)], tag="g")
        j2 = client.post_update("m", {"frames": fr}, files)
        time.sleep(0.2)
        assert jobs.get(j2["job_id"]).state == "queued"
        assert client.service.store.revert("m", 1) == 5
    assert client.wait(j2["job_id"])["base_version"] == 5


def test_retention_keeps_the_last_versions(tmp_path: Path) -> None:
    """map.md §2 Retention: at least the last N; older ones deleted (``version_gone``) but one
    a read holds; the mapper maps no kept version uses go with them."""
    store = Store(tmp_path, keep_versions=2)
    store.create()

    def mapper(tag: str) -> Path:
        d = store.work / f"m-{tag}"
        d.mkdir(parents=True)
        (d / "tag").write_text(tag)
        return d

    rec = {"frames": [], "sim3": {}, "scale_source": "priors", "scale_sigma_rel": 0.01}
    assert store.commit("m", rec, mapper("1")) == 1
    with store.hold("m", 1):
        store.commit("m", rec, mapper("2"), {"objects.json": {"objects": []},
                                             "kept_depth.json": {}})
        store.commit("m", rec, None)
        assert store.kept("m") == [1, 2, 3]  # held by a read
    store.commit("m", {**rec, "frames": [{"content": True, "added_version": 2}]}, mapper("4"))
    assert store.kept("m") == [3, 4]
    with pytest.raises(ApiError) as e:
        store.version("m", 1)
    assert e.value.code == "version_gone"
    vdir = store.dir("m") / "versions"
    # version 4's kept frame reads the kept depth of version 2, which added it
    assert (vdir / "2.kept_depth.json").exists() and not (vdir / "2.objects.json").exists()
    assert not (store.dir("m") / "mappers" / "1").exists()
    assert (store.dir("m") / "mappers" / "2" / "tag").read_text() == "2"  # serves version 3
    assert (store.mapper_for("m", 3) or Path()).name == "2"
    assert store.mapper_for("m", 4) == store.mapper_dir("m")
    with pytest.raises(ApiError):
        store.mapper_for("m", 1)
    assert [v["version"] for v in store.versions("m")] == [3, 4]
    # a revert to 3: version 5 holds a clone of the map 3 was committed with
    assert store.revert("m", 3) == 5
    assert (store.mapper_dir("m") / "tag").read_text() == "2"
    assert store.version("m", 5)["reverted_to"] == 3
