"""Round 12 (P22): the versions list takes no ``?version``; a revert keeps the camera registry
and the object ids, reuses its target's derived reads; kept versions share unchanged derived
files; jobs and reads hold their versions; ``map_busy`` and queued jobs; commits of one map
are serialised."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.service import object_ids
from oh_my_slam.service.derived import Derived, DerivedVersion, content_key
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.objects import _mapper_records
from oh_my_slam.service.store import Store
from oh_my_slam.service.update import Updater, update_versions
from tests.synth.room import base_pose
from tests.unit.test_round10 import _revert, _three
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    frames_for,
)

REC = {"frames": [], "sim3": {}, "scale_source": "priors", "scale_sigma_rel": 0.01}


def test_the_versions_list_takes_no_version(client: Client) -> None:  # noqa: F811
    """api.md §1 (SI-21): ``?version`` on ``GET …/versions`` answers 400 ``invalid_request``."""
    _three(client)
    e = client.get("/api/maps/m/versions?version=1", status=400)["error"]
    assert e["code"] == "invalid_request" and e["details"]["field"] == "version"


def test_a_revert_keeps_the_camera_registry(client: Client) -> None:  # noqa: F811
    """map.md §2 Revert (SI-22): a revert version lists every camera defined so far, those
    defined after its target included; a camera id stays known with its values."""
    _three(client)
    fr, files = frames_for(client, [base_pose(80.0)], tag="w")
    for f in fr:
        f["camera_id"] = "cam2"
    cam2 = {**CAM, "fx": 170.0, "fy": 170.0}
    assert client.update("m", fr, files, cameras={"cam2": cam2})["state"] == "committed"
    assert set(client.get("/api/maps/m?version=1", "MapSummary")["cameras"]) == {"cam"}
    s = _revert(client, 1)
    assert s["map_version"] == 5 and set(s["cameras"]) == {"cam", "cam2"}
    assert client.service.store.version("m", 5)["cameras"]["cam2"] == cam2
    # cam2 is still known: defining it again with other values is refused
    fr, files = frames_for(client, [base_pose(90.0)], tag="x")
    for f in fr:
        f["camera_id"] = "cam2"
    data = client.post_update("m", {"frames": fr, "cameras": {"cam2": CAM}}, files, status=409)
    assert data["error"]["code"] == "camera_conflict"


# -- object ids across the history (SI-23) -------------------------------------------------------


def _anchored(root: Path, objects: dict[str, Any] | None = None) -> Path:
    from oh_my_slam.mapping.anchored import STATE_JSON

    root.mkdir(parents=True)
    (root / STATE_JSON).write_text("{}")
    if objects is not None:
        (root / "objects.json").write_text(json.dumps(objects))
    return root


def test_object_ids_count_on_both_engines(tmp_path: Path) -> None:
    assert object_ids.next_id(None) == 1
    a = _anchored(tmp_path / "a")
    assert object_ids.next_id(a) == 1
    object_ids.reserve(a, 7)  # no object yet: an empty object map numbering from 7
    from oh_my_slam.mapping.anchored_objects import ObjectMap

    om = ObjectMap.load(a)
    assert om.next_id == 7 and om.objects == []
    object_ids.reserve(a, 5)  # never lowered
    assert object_ids.next_id(a) == 7
    object_ids.reserve(a, 9)
    assert ObjectMap.load(a).next_id == 9
    # the older mapper: its map.json count, and its objects.json one when it has objects
    old = tmp_path / "old"
    old.mkdir()
    (old / "map.json").write_text(json.dumps({"format_version": 1, "next_object_id": 4}))
    assert object_ids.next_id(old) == 4
    object_ids.reserve(old, 6)
    assert json.loads((old / "map.json").read_text())["next_object_id"] == 6
    (old / "objects.json").write_text(json.dumps({"next_id": 8, "objects": []}))
    assert object_ids.next_id(old) == 8
    object_ids.reserve(old, 11)
    assert json.loads((old / "objects.json").read_text())["next_id"] == 11
    assert json.loads((old / "map.json").read_text())["next_object_id"] == 11
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "objects.json").write_text("{")
    assert object_ids.next_id(bad) == 1
    object_ids.reserve(bad, 3)  # neither engine's layout: nothing to number
    assert object_ids.next_id(bad) == 1


def test_a_revert_never_reuses_an_object_id(tmp_path: Path) -> None:
    """map.md §2 Revert: the clone a revert commits numbers new objects past every id the map
    gave, those of versions reverted away too, across reverts."""
    store = Store(tmp_path)
    store.create()

    def mapper(tag: str, next_id: int) -> Path:
        return _anchored(store.work / f"m-{tag}", {"objects": [], "next_id": next_id,
                                                   "surface_ids": [], "next_surface_id": 1,
                                                   "version": 0, "surfaces": []})

    store.commit("m", REC, mapper("1", 3))
    store.commit("m", REC, mapper("2", 10))  # ids 3..9 created in version 2
    assert store.revert("m", 1) == 3
    assert object_ids.next_id(store.mapper_dir("m")) == 10
    assert store.meta("m")["next_object_id"] == 10  # type: ignore[index]
    # a later update on the reverted line gives 10..11; a revert to 2 starts past them
    store.commit("m", REC, mapper("4", 12))
    assert store.revert("m", 2) == 5
    assert object_ids.next_id(store.mapper_dir("m")) == 12
    # a revert back to 1 again: past every id given on any line
    assert store.revert("m", 1) == 6
    assert object_ids.next_id(store.mapper_dir("m")) == 12


def test_surface_ids_count_and_a_revert_raises_them(tmp_path: Path) -> None:
    """map.md §2 Revert (P23): surface ids follow the object-id rule: a revert raises the
    clone's surface counter to the map's high-water mark, never back to the target's."""
    from oh_my_slam.mapping.anchored_objects import ObjectMap

    a = _anchored(tmp_path / "a", {"objects": [], "next_id": 2, "surface_ids": [],
                                   "next_surface_id": 3, "version": 0, "surfaces": []})
    assert object_ids.next_ids(a) == object_ids.Ids(2, 3)
    object_ids.reserve(a, object_ids.Ids(2, 6))  # the objects' counter stays, surfaces rise
    assert (ObjectMap.load(a).next_id, ObjectMap.load(a).next_surface_id) == (2, 6)
    b = _anchored(tmp_path / "b")
    object_ids.reserve(b, object_ids.Ids(1, 4))
    assert ObjectMap.load(b).next_surface_id == 4
    # the older mapper numbers no surfaces: its objects file gets no surface counter
    old = tmp_path / "old"
    old.mkdir()
    (old / "objects.json").write_text(json.dumps({"next_id": 2, "objects": []}))
    object_ids.reserve(old, object_ids.Ids(5, 7))
    assert json.loads((old / "objects.json").read_text()) == {"next_id": 5, "objects": []}
    store = Store(tmp_path / "s")
    store.create()

    def mapper(tag: str, objects: int, surfaces: int) -> Path:
        return _anchored(store.work / f"m-{tag}", {"objects": [], "next_id": objects,
                                                   "surface_ids": [],
                                                   "next_surface_id": surfaces,
                                                   "version": 0, "surfaces": []})

    store.commit("m", REC, mapper("1", 2, 2))
    store.commit("m", REC, mapper("2", 2, 5))  # surfaces 2..4 in version 2
    assert store.revert("m", 1) == 3
    assert object_ids.next_ids(store.mapper_dir("m")) == object_ids.Ids(2, 5)
    assert store.id_marks("m") == object_ids.Ids(2, 5)


def test_a_new_mapper_map_is_seeded_with_the_marks(tmp_path: Path) -> None:
    """map.md §2 Revert (P23): after a revert to a version without a stored mapper map, the
    next mapper map a job builds numbers from the recorded high-water marks."""
    from oh_my_slam.mapping.anchored_objects import ObjectMap
    from oh_my_slam.mapping.store import MapReader

    store = Store(tmp_path)
    store.create()
    assert store.seed_ids("m", tmp_path / "x", anchored=True) is False  # no mark: nothing
    store.commit("m", REC, None)
    assert store.seed_ids("m", tmp_path / "x", anchored=True) is False
    store.commit("m", REC, _anchored(store.work / "m2", {
        "objects": [], "next_id": 8, "surface_ids": [], "next_surface_id": 4, "version": 0,
        "surfaces": []}))
    assert store.revert("m", 1) == 3  # version 1 had no mapper map
    assert store.mapper_for("m", 3) is None and not store.mapper_dir("m").exists()
    assert store.id_marks("m") == object_ids.Ids(8, 4)
    a = tmp_path / "anchored"
    assert store.seed_ids("m", a, anchored=True)
    assert (ObjectMap.load(a).next_id, ObjectMap.load(a).next_surface_id) == (8, 4)
    old = tmp_path / "old"
    assert store.seed_ids("m", old, anchored=False)
    assert object_ids.next_id(old) == 8
    reader = MapReader(old)  # a new map the older mapper extends
    assert reader.frames == [] and reader.meta["update_count"] == 0


def test_jobs_seed_a_new_mapper_map_on_both_engines(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A job on a map with versions but no stored mapper map builds a new one numbering past
    the recorded marks: the prior-first engine's object map, the older mapper's ``map.json``."""
    from oh_my_slam.service import update as update_mod

    _three(client)
    store = client.service.store
    meta = store.meta("m")
    assert meta is not None
    meta.update(next_object_id=40, next_surface_id=9)
    store._write_meta("m", meta)
    import shutil as sh

    sh.rmtree(store.mapper_dir("m"))
    seen: list[tuple[bool, int]] = []
    real = Store.seed_ids

    def spy(self: Store, name: str, mapper: Path, anchored: bool) -> bool:
        out = real(self, name, mapper, anchored)
        seen.append((anchored, object_ids.next_id(mapper)))
        return out

    monkeypatch.setattr(Store, "seed_ids", spy)
    fr, files = frames_for(client, [base_pose(60.0)], tag="e")
    assert client.update("m", fr, files)["state"] == "committed"
    assert seen == [(True, 40)]
    surfaces = client.get("/api/maps/m/objects", "Objects")["surfaces"]
    assert surfaces and min(x["id"] for x in surfaces) >= 9
    # the older mapper's path: the folder it extends holds the seeded count
    from tests.fakes.service import CAMERA
    from tests.unit.test_service_api import frames_of

    old = client.backend.register
    counts: list[int] = []

    def register(mapper_dir: Path, *a: Any, **k: Any) -> Any:
        counts.append(object_ids.next_id(mapper_dir))
        return old(mapper_dir, *a, **k)

    monkeypatch.setattr(client.backend, "register", register)
    up = Updater(store, client.backend, prior_first=False)
    frames, files, _ = frames_of(client, 2)
    client.service.jobs.runner = up
    assert client.update("o", frames, files, {"cam": CAMERA})["state"] == "committed"
    meta = store.meta("o")
    assert meta is not None
    meta.update(next_object_id=30)
    store._write_meta("o", meta)
    sh.rmtree(store.mapper_dir("o"))
    frames, files, _ = frames_of(client, 2, start=2)
    client.update("o", frames, files)  # the new map lacks the stored frames: may be rejected
    assert counts == [1, 30] and seen[-1] == (False, 30)
    assert update_mod is not None


def test_older_mapper_update_ids_follow_the_versions() -> None:
    """map.md §5 ``first_seen_version`` (SI-23): the older mapper's update ids become versions
    through each version's line: a version with no mapper update and a revert shift nothing."""
    v1 = {"update_versions": update_versions(None, 1, 1)}
    v2 = {**v1}  # every frame kept at its prior: no mapper update
    v3 = {"update_versions": update_versions(v2, 2, 3)}
    assert v3["update_versions"] == {"1": 1, "2": 3}
    assert update_versions(v3, None, 4) == {"1": 1, "2": 3}
    rec = {"frames": [], "update_versions": {"1": 1, "2": 3}}
    meta = {"objects": [
        {"id": 1, "label": "box", "score": 0.9, "created_update": 2, "last_seen_update": 2,
         "centre_m": [0.0, 0.0, 0.5], "size_m": [0.2, 0.2, 0.2], "yaw_deg": 0.0},
        {"id": 2, "label": "cup", "score": 0.8, "created_update": 1,
         "centre_m": [1.0, 0.0, 0.5], "size_m": [0.1, 0.1, 0.1], "yaw_deg": 0.0}]}
    dv = DerivedVersion("m", 3, rec, meta, Path("."), [])
    dv._cloud = {"xyz": np.zeros((2, 3), np.float32), "rgb": np.zeros((2, 3), np.uint8),
                 "object_id": np.array([1, 2], np.uint32)}
    got = {o["id"]: o for o in _mapper_records(dv)["objects"]}
    assert (got[1]["first_seen_version"], got[1]["last_seen_version"]) == (3, 3)
    assert (got[2]["first_seen_version"], got[2]["last_seen_version"]) == (1, 1)


# -- derived reads shared between versions (SI-24, M-21) -----------------------------------------


def _npz(p: Path, **arrays: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arrays)


def test_two_identical_versions_share_one_file(tmp_path: Path) -> None:
    """map.md §2 Sharing: equal derived files of two versions are one file (one inode, one
    blob), whatever their zip encoding; a blob goes with the last version linking it."""
    store = Store(tmp_path)
    store.create()
    d = Derived(store, lambda p: None)  # type: ignore[arg-type,return-value]
    a, b, c = (d.dir("m") / k for k in ("1", "2", "3"))
    _npz(a / "volume.npz", x=np.arange(10))
    b.mkdir(parents=True)
    np.savez_compressed(b / "volume.npz", x=np.arange(10))  # other bytes, same arrays
    _npz(c / "volume.npz", x=np.arange(11))
    assert (a / "volume.npz").read_bytes() != (b / "volume.npz").read_bytes()
    assert content_key(a / "volume.npz") == content_key(b / "volume.npz")
    for v in (a, b, c):
        d._share("m", v)
    assert (a / "volume.npz").stat().st_ino == (b / "volume.npz").stat().st_ino
    assert (a / "volume.npz").stat().st_ino != (c / "volume.npz").stat().st_ino
    blobs = d.dir("m") / "blobs"
    assert len(list(blobs.glob("*.npz"))) == 2
    for v in (a, c):
        for p in v.iterdir():
            p.unlink()
    d.forget("m", 3)
    assert len(list(blobs.glob("*.npz"))) == 1  # version 3's blob: no version links it
    assert np.load(b / "volume.npz")["x"].tolist() == list(range(10))


def test_a_revert_reuses_its_targets_derived_reads(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §2 Revert (SI-24): the revert's derived files are its target's (one inode
    each), its answers and locate index the target's renamed to its version."""
    _three(client)
    g1 = client.get("/api/maps/m/grid?version=1", "Grid")
    o1 = client.get("/api/maps/m/objects?version=1", "Objects")
    derived = client.service.derived
    assert derived is not None
    calls: list[int] = []
    real = Derived._derive

    def spy(self: Derived, name: str, n: int, *a: Any, **k: Any) -> None:
        calls.append(n)
        real(self, name, n, *a, **k)

    monkeypatch.setattr(Derived, "_derive", spy)
    s = _revert(client, 1)
    monkeypatch.undo()
    assert s["map_version"] == 4 and calls == []  # nothing derived again
    d = client.service.store.dir("m") / "derived"
    for f in ("volume.npz", "cloud.npz", "locate.npz"):
        assert (d / "4" / f).stat().st_ino == (d / "1" / f).stat().st_ino
    assert json.loads((d / "4" / "meta.json").read_text())["version"] == 4
    g4 = client.get("/api/maps/m/grid", "Grid")
    assert g4["map_version"] == 4 and {**g4, "map_version": 1} == g1
    o4 = client.get("/api/maps/m/objects", "Objects")
    assert o4 == {**o1, "map_version": 4}
    loc = derived.get("m", 4)
    assert loc._index is derived.get("m", 1)._index


def test_a_revert_derives_when_its_target_cannot_be_reused(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    _three(client)
    derived = client.service.derived
    assert derived is not None

    def gone(*a: Any) -> Any:
        raise ApiError("version_gone", "gone")

    monkeypatch.setattr(derived, "get", gone)
    assert derived._reuse("m", 4, {"reverted_to": 1}, Path("."), {}) is None

    monkeypatch.undo()
    # the target's files gone (its version in memory): derived like any version
    shutil.rmtree(client.service.store.dir("m") / "derived" / "2")
    s = _revert(client, 2)
    assert s["map_version"] == 4 and s["frames"] == 2
    meta = client.service.store.dir("m") / "derived" / "4" / "meta.json"
    assert json.loads(meta.read_text())["version"] == 4


# -- holds (SI-25) --------------------------------------------------------------------------------


def test_held_versions_stay_beyond_the_last_n(tmp_path: Path) -> None:
    """map.md §2 Holds: a held version older than the newest N is kept on top of them, and goes
    at the first commit after its last hold ended; the latest is read and held in one step."""
    store = Store(tmp_path, keep_versions=1)
    store.create()
    with store.hold_latest("m") as none:
        assert none is None
    with pytest.raises(ApiError) as e, store.hold_latest("m", required=True):
        pass
    assert e.value.code == "map_not_found"
    store.commit("m", REC, None)
    with store.hold_latest("m") as n:
        assert n == 1
        store.commit("m", REC, None)
        store.commit("m", REC, None)
        assert store.kept("m") == [1, 3]  # the newest one, and the held one on top
    assert store.kept("m") == [1, 3]  # no commit since the hold ended
    store.commit("m", REC, None)
    assert store.kept("m") == [4]


def test_a_job_holds_its_base_version(client: Client,  # noqa: F811
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _three(client)
    store = client.service.store
    seen: list[dict[tuple[str, int], int]] = []
    real = Updater._run

    def look(self: Updater, *a: Any) -> None:
        seen.append(dict(store._holds))
        real(self, *a)

    monkeypatch.setattr(Updater, "_run", look)
    fr, files = frames_for(client, [base_pose(60.0)], tag="e")
    job = client.update("m", fr, files)
    assert job["base_version"] == 3 and seen == [{("m", 3): 1}]
    assert store._holds == {}


def test_a_dry_run_and_a_versioned_locate_hold_their_version(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from oh_my_slam.service import nav
    from tests.fakes.service import multipart

    _three(client)
    store = client.service.store
    seen: list[dict[tuple[str, int], int]] = []
    real_check = Updater._check

    def check(self: Updater, *a: Any) -> Any:
        seen.append(dict(store._holds))
        return real_check(self, *a)

    monkeypatch.setattr(Updater, "_check", check)
    fr, files = frames_for(client, [base_pose(60.0)], tag="e")
    data, ctype = multipart({"frames": fr}, files)
    r = client.http.post("/api/maps/m/updates?dry_run=true", content=data,
                         headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    assert seen == [{("m", 3): 1}] and store._holds == {}
    real_frames = nav.locate_frames

    def frames(*a: Any) -> Any:
        seen.append(dict(store._holds))
        return real_frames(*a)

    monkeypatch.setattr(nav, "locate_frames", frames)
    fr, files = frames_for(client, [base_pose(25.0)], tag="q", depth=False, trust=None)
    data, ctype = multipart({"frames": fr, "version": 2, "record": False}, files)
    r = client.http.post("/api/maps/m/locate", content=data, headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    assert r.json()["map_version"] == 2 and seen[-1] == {("m", 2): 1}
    assert store._holds == {}


# -- map_busy and queued jobs (SI-29) -------------------------------------------------------------


def test_delete_is_busy_while_a_job_is_queued_and_revert_is_not(
        client: Client) -> None:  # noqa: F811
    """api.md §5: delete is refused ``map_busy`` while a job is only queued; a revert proceeds,
    and the queued job then runs from the reverted version, its ``base_version`` null until it
    starts."""
    _three(client)
    jobs = client.service.jobs
    with jobs.holding("m"):
        fr, files = frames_for(client, [base_pose(70.0)], tag="g")
        j = client.post_update("m", {"frames": fr}, files)
        time.sleep(0.2)
        assert client.get(f"/api/jobs/{j['job_id']}", "Job")["base_version"] is None
        r = client.http.delete("/api/maps/m")
        assert r.status_code == 409 and client.check(r, None)["error"]["code"] == "map_busy"
        assert _revert(client, 1)["map_version"] == 4  # nested in the outer hold
        time.sleep(0.2)
        assert jobs.get(j["job_id"]).state == "queued"  # the outer hold still pauses it
    done = client.wait(j["job_id"])
    assert done["state"] == "committed" and done["base_version"] == 4
    assert [f["name"] for f in client.get("/api/maps/m/frames", "Frames")["frames"]] \
        == ["a0", "g0"]


def test_two_reverts_of_a_map_are_serialised(client: Client,  # noqa: F811
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _three(client)
    store = client.service.store
    real = Store._commit
    inside = threading.Event()
    gate = threading.Event()
    order: list[int] = []

    def slow(self: Store, name: str, record: dict[str, Any], *a: Any, **k: Any) -> int:
        order.append(int(record["reverted_to"]))
        if len(order) == 1:
            inside.set()
            gate.wait(10)
        return real(self, name, record, *a, **k)

    monkeypatch.setattr(Store, "_commit", slow)
    got: dict[int, int] = {}
    t1 = threading.Thread(target=lambda: got.__setitem__(1, store.revert("m", 1)))
    t1.start()
    assert inside.wait(10)
    t2 = threading.Thread(target=lambda: got.__setitem__(2, store.revert("m", 2)))
    t2.start()
    time.sleep(0.3)
    assert order == [1]  # the second waits for the first's commit
    gate.set()
    t1.join(20)
    t2.join(20)
    assert got == {1: 4, 2: 5}
    assert store.version("m", 4)["reverted_to"] == 1 and len(store.version("m", 4)["frames"]) == 1
    assert store.version("m", 5)["reverted_to"] == 2 and len(store.version("m", 5)["frames"]) == 2


def test_labels_compare_unicode_case_folded() -> None:
    """map.md §5 (SI-28): labels are equal when their Unicode case-folded forms are, after
    trimming, ``_`` read as a space and whitespace runs collapsed; the first form is kept."""
    from oh_my_slam.segmentation.detect import normalize_label
    from oh_my_slam.service.update import map_labels

    assert normalize_label(" Straße_Sign ") == normalize_label("STRASSE   sign") == "strasse sign"
    assert map_labels({"labels": ["Straße sign"]}, ["strasse_SIGN", "red mug"]) \
        == ["Straße sign", "red mug"]
