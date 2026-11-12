"""Round 10: the supporting frames' timing (P19), versions and revert (O2), ignored pixels and
the label vocabulary (P18, oms/2.3), and object extents over many views."""

from __future__ import annotations

import json
import shutil
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


# -- P18: ignored pixels contribute nothing, on every engine ------------------------------------


def _png8(a: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


FEET_ROW = 190  # the rows below it show the robot's own feet, 0.32 m away


def _feet_frames(c: Client, poses: list[np.ndarray], tag: str, depth: bool = True
                 ) -> tuple[list[Any], dict[str, bytes]]:
    import hashlib

    from tests.synth.room import png_depth
    from tests.unit.test_service_prior_first import shot

    fr, files = frames_for(c, poses, tag=tag, depth=depth)
    for i, T in enumerate(poses):
        img, _, z = shot(T)
        z = z.copy()
        z[FEET_ROW:, 100:220] = 0.32
        if depth:
            files[f"{tag}{i}d"] = png_depth(z)
        c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = z
    return fr, files


def _near_occupied(c: Client, name: str) -> int:
    import base64

    g = c.get(f"/api/maps/{name}/grid?cell_m=0.05", "Grid")
    st = np.frombuffer(base64.b64decode(g["layers"]["state"]["data"]), np.uint8) \
        .reshape(g["shape"])
    ox, oy = g["origin_m"]
    rows, cols = np.nonzero(st == 2)
    xy = np.c_[ox + (cols + 0.5) * 0.05, oy + (rows + 0.5) * 0.05]
    cams = np.array([np.asarray(f["T_map_cam"])[:2, 3]
                     for f in c.get(f"/api/maps/{name}/frames", "Frames")["frames"]])
    d = np.linalg.norm(xy[:, None, :] - cams[None], axis=2).min(1) if len(xy) else np.zeros(0)
    return int((d < 0.5).sum())


@pytest.mark.parametrize("depth", [True, False], ids=["sensor", "monocular"])
def test_a_camera_seeing_its_own_feet_in_its_mask_leaves_no_trace(
        client: Client, depth: bool) -> None:  # noqa: F811
    """map.md §2 "Ignored pixels" (P18), prior-first engine: the feet a camera sees in its
    ignore mask give no occupied cell near it (the reported phantom cells 0.30-0.34 m away), no
    cloud point, no keypoint (registration or locate) and no stored depth; without the mask the
    same feet are occupied (the control)."""
    mask = np.zeros((240, 320), np.uint8)
    mask[FEET_ROW:, :] = 255
    poses = [base_pose(30.0 * k) for k in range(3)]
    fr, files = _feet_frames(client, poses, "c", depth)
    assert client.update("control", fr, files, cameras={"cam": CAM})["state"] == "committed"
    assert _near_occupied(client, "control") > 0
    cam = {**CAM, "ignore_mask": "mask"}
    fr, files = _feet_frames(client, poses, "m", depth)
    job = client.update("m", fr, {**files, "mask": _png8(mask)}, cameras={"cam": cam})
    assert job["state"] == "committed"
    assert _near_occupied(client, "m") == 0
    from oh_my_slam.mapping.anchored import AnchoredMap

    am = AnchoredMap.load(client.service.store.mapper_dir("m"))
    for k in am.keyframes:
        am._load_kf(k)
        assert k.feats is not None and (k.feats.kps[:, 1] < FEET_ROW).all()
        ign = am.ignore_of(k.id)
        d = am.depth_of(k.id)
        assert ign is not None and d is not None and not d[ign].any()
    # the cloud has no point at the feet (0.32 m ahead, below the camera)
    pts = client.get("/api/maps/m/points?min=-0.6,-0.6,0.05&max=0.6,0.6,0.45", "Points")
    ctrl = client.get("/api/maps/control/points?min=-0.6,-0.6,0.05&max=0.6,0.6,0.45", "Points")
    assert ctrl["xyz"]["shape"][0] > 0 and pts["xyz"]["shape"][0] == 0
    # the locate index's keypoints avoid them too
    feats = list((client.service.store.dir("m") / "derived" / "features").glob("*-m*.npz"))
    assert feats and all((np.load(p)["uv"][:, 1] < FEET_ROW).all() for p in feats)


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")
def test_colmap_extracts_no_keypoint_on_ignored_pixels(tmp_path: Path) -> None:
    """The older mapper's SIFT (Homebrew colmap CLI, ``Sfm.extract``): no keypoint on a
    keyframe's ignored pixels (``frame_masks/<name>.png``), so no match (SIFT's or LightGlue's,
    both on those keypoints) either; a keyframe without a mask keeps them all."""
    import pycolmap

    from oh_my_slam.mapping.sfm import CameraPrior, Sfm, ignore_mask_path
    from tests.fakes.service import textured

    frames = tmp_path / "frames"
    frames.mkdir()
    for n in ("a.jpg", "b.jpg"):
        (frames / n).write_bytes(textured(3, 320, 240))
    mask = np.zeros((240, 320), np.uint8)
    mask[:, :160] = 255
    p = ignore_mask_path(frames, "a.jpg")
    p.parent.mkdir()
    p.write_bytes(_png8(mask))
    sfm = Sfm(tmp_path / "db.db", frames, tmp_path / "work")
    sfm.extract(["a.jpg", "b.jpg"], CameraPrior(320, 240, focal=300.0))
    db = pycolmap.Database.open(str(tmp_path / "db.db"))
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        a, b = (np.asarray(db.read_keypoints(ids[n])) for n in ("a.jpg", "b.jpg"))
    finally:
        db.close()
    assert len(a) > 20 and (a[:, 0] >= 159.0).all()
    assert (b[:, 0] < 159.0).sum() > 20
    assert Sfm(tmp_path / "db2.db", frames, tmp_path / "w2")._keypoint_masks(["b.jpg"]) is None


def test_detections_mostly_on_ignored_pixels_are_dropped() -> None:
    """map.md §2 (P18): a detection with more than ``IGNORED_DETECTION_MAX_SHARE`` of its mask
    on the ignored pixels gives no object and no vote; a camera whose view of an object falls
    on its ignored pixels does not count as seeing it."""
    from oh_my_slam.core.constants import IGNORED_DETECTION_MAX_SHARE
    from oh_my_slam.core.images import on_grid
    from oh_my_slam.core.types import Intrinsics
    from oh_my_slam.mapping import anchored_objects as ao
    from oh_my_slam.segmentation.detect import Detection

    assert IGNORED_DETECTION_MAX_SHARE == 0.5
    ign = np.zeros((20, 30), bool)
    ign[10:] = True
    low = np.zeros((20, 30), bool)
    low[12:] = True
    half = np.zeros((20, 30), bool)
    half[5:15] = True
    dets = [Detection("rug", 0.9, "default", low, (0, 12, 30, 20)),
            Detection("chair", 0.8, "default", half, (0, 5, 30, 15)),
            Detection("box", 0.7, "default", np.zeros((4, 4), bool), (0, 0, 30, 6))]
    assert [d.label for d in ao.unignored(dets, ign)] == ["chair", "box"]
    assert ao.unignored(dets, None) == dets
    assert on_grid(None, (2, 2)) is None and on_grid(np.zeros((4, 4), bool), (2, 2)) is None
    assert on_grid(ign, (40, 60)).shape == (40, 60)  # type: ignore[union-attr]
    K = Intrinsics(30.0, 30.0, 15.0, 10.0, 30, 20)
    T = np.eye(4)
    below = np.array([0.0, 0.2, 1.0])  # row 16: ignored
    above = np.array([0.0, -0.2, 1.0])  # row 4
    assert ao.in_view(below, T, K) and not ao.in_view(below, T, K, ign)
    assert ao.in_view(above, T, K, ign)
    assert ao.in_view(below, T, K, on_grid(ign, (40, 60))) is False


def test_mapanything_depth_on_ignored_pixels_counts_for_nothing() -> None:
    from oh_my_slam.core.types import Intrinsics
    from oh_my_slam.mapping import register_ma as rm

    ign = np.zeros((20, 30), bool)
    ign[10:] = True
    v = rm.MvView(np.eye(4), np.ones((10, 15), np.float32), np.eye(3))
    K = Intrinsics(30.0, 30.0, 15.0, 10.0, 30, 20)
    chunk = [rm.New(0, Path("a"), K, ignore=ign), rm.New(1, Path("b"), K)]
    out = rm._without_ignored([v, v, v], 1, chunk)
    assert out[0] is v and out[2] is v
    assert not out[1].depth[5:].any() and out[1].depth[:5].all()


# -- P18: the label vocabulary (oms/2.3) --------------------------------------------------------


def test_labels_are_validated_as_the_field_labels(client: Client) -> None:  # noqa: F811
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    for bad in ([""], ["   "], ["x" * 65], ["ok"] * 65, [3]):
        req = {"frames": fr, "cameras": {"cam": CAM}, "labels": bad}
        doc = client.post_update("m", req, files, status=400)
        assert doc["error"]["code"] == "invalid_request"
        assert doc["error"]["details"]["field"] == "labels", bad
    doc = client.post_update("m", {"frames": fr, "cameras": {"cam": CAM}, "nope": 1}, files,
                             status=400)
    assert doc["error"]["details"]["field"] != "labels"


def test_the_label_vocabulary_is_the_union_of_the_updates(client: Client) -> None:  # noqa: F811
    """map.md §5 (oms/2.3): each update's labels join the version's vocabulary (case-insensitive,
    trimmed, never the defaults); detection prompts with the defaults and it; the summary
    reports it, a dry run the merged one; a revert restores it."""
    from oh_my_slam.segmentation.detect import default_vocabulary

    fr, files = frames_for(client, [base_pose(0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM}, labels=[" Red Mug ", "chair"])
    assert job["state"] == "committed"
    assert client.backend.prompted and client.backend.prompted[-1] == ["Red Mug", "chair"]
    assert client.get("/api/maps/m", "MapSummary")["labels"] == ["Red Mug", "chair"]
    fr, files = frames_for(client, [base_pose(20)], tag="b")
    req = {"frames": fr, "labels": ["red mug", "Lamp Shade"]}
    from tests.unit.test_service_prior_first import dry

    d = dry(client, "m", fr, files, labels=["red mug", "Lamp Shade"])
    defaults = list(default_vocabulary())
    assert d["labels"][:len(defaults)] == defaults
    extra = d["labels"][len(defaults):]
    assert "Red Mug" in extra and "Lamp Shade" in extra and "red mug" not in extra
    assert "chair" in defaults and "chair" not in extra
    job = client.wait(client.post_update("m", req, files)["job_id"])
    assert job["state"] == "committed"
    s = client.get("/api/maps/m", "MapSummary")
    assert s["labels"] == ["Red Mug", "chair", "Lamp Shade"]
    assert client.get("/api/maps/m?version=1", "MapSummary")["labels"] == ["Red Mug", "chair"]
    assert _revert(client, 1)["labels"] == ["Red Mug", "chair"]


def test_prompts_merge_the_defaults_case_insensitively() -> None:
    from oh_my_slam.segmentation import detect as dt

    defaults = list(dt.default_prompts())
    first = dt.default_vocabulary()[0]
    got = dt.prompts([first.upper(), "  Red  Mug ", "red mug", "", "blue_box"])
    assert got[:len(defaults)] == defaults and got[len(defaults):] == ["Red  Mug", "blue_box"]
    assert dt.label_of("Red  Mug") == "red mug"
    assert dt.vocabulary(["Red Mug"])[-1] == "Red Mug" and dt.prompts() == defaults


# -- extent_seen over many views ----------------------------------------------------------------


def test_objects_seen_from_many_views_are_seen_in_full(tmp_path: Path) -> None:
    """map.md §5 ``extent_seen`` from the views that bound the box: objects on a table seen
    uncut from around and above, two rings of keyframes and one overhead, are ``full`` on every
    axis (the prior-first engine records each sighting's ray and cut borders)."""
    import math

    from oh_my_slam.mapping.anchored import Cloud
    from oh_my_slam.mapping.anchored_objects import ObjectMap
    from tests.synth.table import TableScene
    from tests.unit.test_anchored_objects import view

    sc = TableScene()
    eyes = [(0.6 + r * math.cos(2 * math.pi * k / 12), 0.4 + r * math.sin(2 * math.pi * k / 12), h)
            for k in range(12) for h, r in ((1.6, 1.0), (1.2, 1.3))] + [(0.6, 0.41, 2.2)]
    cloud = Cloud()
    om = ObjectMap(tmp_path)
    views = [view(f"k{i}", e, sc, cloud) for i, e in enumerate(eyes)]
    om.update(views, 1, 0.0)
    om.finish(cloud.points(), 1.5, 1, np.array([v.centre for v in views]), 0.0)
    recs = om.records()["objects"]
    assert {o["label"] for o in recs} >= {"box", "basket", "apple"}
    for o in recs:
        assert o["extent_seen"] == {"x": "full", "y": "full", "z": "full"}, o["label"]


def test_the_older_mapper_s_recorded_views_give_its_extents() -> None:
    """``service.objects._mapper_extent``: the older mapper records each object's views (its
    ``Sighting``: keyframe, centroid, border share), so its extents follow the same rule; a view
    whose keyframe the version does not hold counts for nothing; none recorded: partial."""
    from oh_my_slam.service.objects import _mapper_extent

    cams = {"f000000": np.array([-2.0, 0.0, 0.5]), "f000001": np.array([0.0, -2.0, 0.5])}
    side = {"frame": "f000000", "centroid": [0.0, 0.0, 0.5], "uncut": True}
    o = {"yaw_deg": 0.0, "views": [side, {"frame": "f000001", "centroid": [0.0, 0.0, 0.5],
                                          "uncut": False},
                                   {"frame": "gone", "centroid": [0, 0, 0], "uncut": True},
                                   {"frame": "f000000", "centroid": [-2.0, 0.0, 0.5],
                                    "uncut": True}]}
    assert _mapper_extent(o, cams) == {"x": "partial", "y": "full", "z": "full"}
    assert _mapper_extent({"yaw_deg": 10.0}, cams) == {"x": "partial", "y": "partial",
                                                       "z": "partial"}
