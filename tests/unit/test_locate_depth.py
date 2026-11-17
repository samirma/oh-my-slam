"""oms/2.17: a locate frame's own depth image (api.md §6 "Query depth"; ``relocalize``,
``nav``, ``derived``).

The locate scenes are synthetic: a box room whose walls, floor and ceiling carry points with
unique descriptors, rendered analytically (depth along the optical axis), keyframes taken from
two spots turning, each with its sensor depth on a reduced grid. A rival is planted as texture
aliasing: part of the query's keypoints show map points as a camera at another pose would see
them (a tiled wall seen one tile off), so that the images alone hold a second hypothesis."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.mapping import relocalize as rl
from oh_my_slam.service import nav as nav_mod
from oh_my_slam.service.backend import InternalMap
from oh_my_slam.service.derived import Derived, sensor_grid
from oh_my_slam.service.errors import ApiError
from oh_my_slam.service.store import DEPTH_UNUSED
from tests.synth import rooms
from tests.unit.test_service_nav import Nav, _png, nav, room_map  # noqa: F401  (fixtures)

K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
W, H = 320, 240
BOX = (np.array([-1.6, -1.5, 0.0]), np.array([1.6, 3.0, 2.5]))
LONG = (np.array([-50.0, -1.5, 0.0]), np.array([50.0, 3.0, 2.5]))  # no side walls in view
TRUTH = rooms.camera(0.1, 0.2, 1.2, 90.0, -10.0)


def _shifted(T: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = T.copy()
    out[:3, 3] = T[:3, 3] + d
    return out


AHEAD = _shifted(TRUTH, 0.6 * TRUTH[:3, 2])  # the same view 0.6 m along the optical axis
ASIDE = _shifted(TRUTH, np.array([0.6, 0.0, 0.0]))  # one tile along the wall


def render(T: np.ndarray, box: tuple[np.ndarray, np.ndarray], w: int = W, h: int = H,
           Kc: np.ndarray = K) -> np.ndarray:
    """The depth (along the optical axis) of the inside of an axis-aligned box seen by a camera
    at ``T`` inside it."""
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u + 0.5 - Kc[0, 2]) / Kc[0, 0], (v + 0.5 - Kc[1, 2]) / Kc[1, 1],
                  np.ones((h, w))], -1) @ T[:3, :3].T
    c = T[:3, 3]
    best = np.full((h, w), np.inf)
    lo, hi = box
    with np.errstate(divide="ignore", invalid="ignore"):
        for ax in range(3):
            for b in (lo[ax], hi[ax]):
                t = (b - c[ax]) / d[..., ax]
                p = c + t[..., None] * d
                inside = np.all((p >= lo - 1e-6) & (p <= hi + 1e-6), axis=-1)
                ok = (t > 1e-6) & inside
                best = np.where(ok & (t < best), t, best)
    return np.where(np.isfinite(best), best, 0.0).astype(np.float32)


def scene(box: tuple[np.ndarray, np.ndarray] = BOX, n: int = 6000, seed: int = 0
          ) -> tuple[np.ndarray, np.ndarray]:
    """Points on the box's six faces (within 6 m of the cameras) and their descriptors."""
    rng = np.random.default_rng(seed)
    lo, hi = np.maximum(box[0], -6.0), np.minimum(box[1], 6.0)
    P = rng.uniform(lo, hi, (n, 3))
    ax = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    P[np.arange(n), ax] = np.where(side == 0, box[0][ax], box[1][ax])
    keep = np.all(np.abs(P) < 50.0, axis=1)
    return P[keep], rng.integers(0, 256, (int(keep.sum()), 128)).astype(np.uint8)


def view(T: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pixels, indices) of the points a camera at ``T`` sees (a box hides nothing inside)."""
    uv, z = rl.project(T, K, P)
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[ok].astype(np.float32), np.flatnonzero(ok)


def keyframe(name: str, T: np.ndarray, P: np.ndarray, D: np.ndarray,
             box: tuple[np.ndarray, np.ndarray] | None) -> rl.Keyframe:
    """A keyframe with its points and, given its ``box``, its sensor depth on an 80 x 60 grid."""
    uv, idx = view(T, P)
    k = rl.Keyframe(name, T, K, uv, D[idx], P[idx].astype(np.float32), 0.01, 0.1, True)
    if box is not None:
        Kg = K.copy()
        Kg[:2] /= 4.0
        k.sensor, k.sensor_K = render(T, box, 80, 60, Kg), Kg
    return k


def index(P: np.ndarray, D: np.ndarray, box: tuple[np.ndarray, np.ndarray] | None
          ) -> rl.Index:
    kfs = [keyframe(f"k{i}{j}", rooms.camera(x, y, 1.2, yaw, -10.0), P, D, box)
           for i, (x, y) in enumerate(((0.0, 0.0), (0.2, 0.3)))
           for j, yaw in enumerate((70.0, 80.0, 90.0, 100.0, 110.0))]
    return rl.Index(kfs)


def query(P: np.ndarray, D: np.ndarray, alias: np.ndarray | None,
          box: tuple[np.ndarray, np.ndarray] | None, share: float = 0.4) -> rl.Query:
    """The frame at ``TRUTH``: most of its keypoints its own view of the map points, ``share`` of
    them the view from ``alias`` (the aliased texture), with its depth rendered at the truth."""
    rng = np.random.default_rng(3)
    uv, idx = view(TRUTH, P)
    own = rng.random(len(idx)) >= share
    uv_l, d_l = [uv[own]], [D[idx[own]]]
    if alias is not None:
        uv_a, idx_a = view(alias, P)
        pick = np.isin(idx_a, idx[~own])
        uv_l.append(uv_a[pick])
        d_l.append(D[idx_a[pick]])
    uv_q = np.concatenate(uv_l) + rng.normal(0, 0.3, (sum(len(u) for u in uv_l), 2))
    q = rl.Query(uv_q.astype(np.float32), np.concatenate(d_l), K, W, H)
    if box is not None:
        q.depth = render(TRUTH, box)
    return q


def _near(o: rl.Outcome, T: np.ndarray) -> bool:
    assert o.T is not None
    dp, dr = rl.pose_distance(o.T, T)
    return dp < 0.05 and dr < 1.0


# -- relocalize ------------------------------------------------------------------------------------


def test_a_rival_at_another_depth_is_refused_with_query_depth_and_reported_without() -> None:
    P, D = scene()
    idx = index(P, D, BOX)
    bare = rl.locate(query(P, D, AHEAD, None), idx)
    assert bare.located and _near(bare, TRUTH)
    assert bare.depth_pose is None and bare.depth_refused == 0
    assert bare.rivals and bare.rivals[0]["pos_m"] == pytest.approx(0.6, abs=0.05)
    assert bare.rivals[0]["rot_deg"] < 1.0 and bare.ambiguity is not None
    got = rl.locate(query(P, D, AHEAD, BOX), idx)
    assert got.located and _near(got, TRUTH)  # the true pose stays located, unchanged
    assert got.T is not None and bare.T is not None and np.allclose(got.T, bare.T)
    assert (got.sigma_pos_m, got.sigma_rot_deg) == (bare.sigma_pos_m, bare.sigma_rot_deg)
    assert got.rivals == [] and got.ambiguity is None
    assert got.depth_pose == rl.CONFIRMED and got.depth_refused >= 1


def test_a_twin_along_a_wall_is_refused_where_the_depth_sees_the_wall_end() -> None:
    P, D = scene()
    idx = index(P, D, BOX)
    bare = rl.locate(query(P, D, ASIDE, None), idx)
    assert any(r["pos_m"] == pytest.approx(0.6, abs=0.05) for r in bare.rivals)
    got = rl.locate(query(P, D, ASIDE, BOX), idx)
    assert got.located and _near(got, TRUTH) and got.rivals == []
    assert got.depth_pose == rl.CONFIRMED and got.depth_refused >= 1


def test_a_twin_the_depth_cannot_tell_keeps_its_rival() -> None:
    """A wall and a floor with no end in view: the twin one tile along the wall agrees with every
    depth (map.md §4, the accepted degenerate case)."""
    P, D = scene(LONG)
    idx = index(P, D, LONG)
    got = rl.locate(query(P, D, ASIDE, LONG), idx)
    assert got.located and _near(got, TRUTH)
    assert any(r["pos_m"] == pytest.approx(0.6, abs=0.05) for r in got.rivals)
    assert got.depth_pose == rl.CONFIRMED


def test_monocular_references_test_nothing() -> None:
    P, D = scene()
    got = rl.locate(query(P, D, AHEAD, BOX), index(P, D, None))
    assert got.located and got.rivals and got.depth_pose == rl.UNTESTED
    assert got.depth_refused == 0


def test_a_depth_without_values_tests_nothing() -> None:
    """Every pixel ignored (the masks empty the depth): no point is measured."""
    P, D = scene()
    q = query(P, D, AHEAD, BOX)
    assert q.depth is not None
    q.depth = np.zeros_like(q.depth)
    got = rl.locate(q, index(P, D, BOX))
    assert got.located and got.rivals and got.depth_pose == rl.UNTESTED


def test_a_depth_that_contradicts_every_hypothesis_is_depth_inconsistent() -> None:
    """An obstacle right before the camera (or a changed scene): the depth refuses every pose
    the images give."""
    P, D = scene()
    q = query(P, D, AHEAD, BOX)
    assert q.depth is not None
    q.depth = np.full_like(q.depth, 0.4)
    got = rl.locate(q, index(P, D, BOX))
    assert not got.located and got.reason == "depth_inconsistent"
    assert got.depth_refused >= 1 and got.depth_pose is None


def test_the_twin_comparison_refuses_only_beyond_the_margin(monkeypatch: pytest.MonkeyPatch
                                                            ) -> None:
    hyps = [rl.Hypothesis(np.eye(4), 10, np.zeros(1, bool)) for _ in range(4)]
    verdicts = iter([(rl.CONFIRMED, 0), (rl.CONFIRMED, 1), (rl.UNTESTED, None),
                     (rl.CONFIRMED, 0)])
    scores = iter([1.0, 0.96, 0.92])
    monkeypatch.setattr(rl, "depth_verdict", lambda q, h, kfs, index: next(verdicts))
    monkeypatch.setattr(rl, "agreement_over", lambda T, q, refs: next(scores))
    idx = rl.Index([rl.Keyframe(f"k{i}", np.eye(4), K, np.zeros((0, 2), np.float32),
                                np.zeros((0, 128), np.uint8), np.zeros((0, 3), np.float32))
                    for i in range(2)])
    q = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), K, W, H)
    assert rl.depth_verdicts(q, hyps, np.zeros(1, np.int64), idx) == [
        rl.CONFIRMED, rl.CONFIRMED, rl.UNTESTED, rl.REFUSED]
    # no reference measures enough points: nothing to compare
    monkeypatch.setattr(rl, "depth_verdict", lambda q, h, kfs, index: (rl.CONFIRMED, 0))
    monkeypatch.setattr(rl, "agreement_over", lambda T, q, refs: None)
    assert rl.depth_verdicts(q, hyps[:2], np.zeros(1, np.int64), idx) == [rl.CONFIRMED] * 2


def test_agreement_over_and_the_pair_rules() -> None:
    P, D = scene()
    k = keyframe("k", rooms.camera(0.0, 0.0, 1.2, 90.0, -10.0), P, D, BOX)
    q = query(P, D, None, BOX)
    assert rl.agreement_over(TRUTH, q, [k]) == pytest.approx(1.0)
    assert rl.agreement_over(AHEAD, q, [k]) < 0.9
    assert rl.agreement_over(TRUTH, q, []) is None
    seen = rl.DepthCounts(100, 40, 0, 0, 60)
    assert not rl.contradicts(seen) and rl.contradicts(rl.DepthCounts(100, 35, 10, 0, 55))
    assert not rl.contradicts(rl.DepthCounts(100, 20, 10, 0, 70))  # too few measured
    assert rl.distinct([np.eye(4), np.eye(4), AHEAD, TRUTH, TRUTH]) == 3


def test_a_thin_pair_hands_the_decision_to_the_next_reference() -> None:
    """The reference with most inliers barely sees the frame's depth (a far grid of a few
    values): the next one decides."""
    P, D = scene()
    idx = index(P, D, BOX)
    assert idx.keyframes[0].sensor is not None
    for k in idx.keyframes[:5]:  # the first spot's: nearly blank grids
        assert k.sensor is not None
        k.sensor = np.where(np.arange(k.sensor.size).reshape(k.sensor.shape) % 97 == 0,
                            k.sensor, 0.0).astype(np.float32)
    got = rl.locate(query(P, D, AHEAD, BOX), idx)
    assert got.located and _near(got, TRUTH) and got.depth_pose == rl.CONFIRMED


def test_a_turn_fit_is_untested(monkeypatch: pytest.MonkeyPatch) -> None:
    P, D = scene()
    q = query(P, D, None, BOX)
    monkeypatch.setattr(rl, "pnp", lambda *a, **k: None)
    monkeypatch.setattr(rl, "_turn", lambda *a, **k: rl.Outcome(T=TRUTH, method="turn"))
    got = rl.locate(q, index(P, D, BOX))
    assert got.method == "turn" and got.depth_pose == rl.UNTESTED
    q.depth = None
    assert rl.locate(q, index(P, D, BOX)).depth_pose is None


def test_reduced_depth_keeps_every_s_th_pixel() -> None:
    d = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    r = rl.reduced_depth(d)
    assert r.shape == (120, 160) and r[0, 0] == d[2, 2] and r[1, 1] == d[6, 6]
    assert rl.reduced_depth(d[:100, :150]).shape == (100, 150)


# -- the service -----------------------------------------------------------------------------------


def _png16(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def test_locate_takes_a_depth_part_validated_as_an_update_s(
        room_map: tuple[Nav, Any, Any],  # noqa: F811
        monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.synth.scene import look_at
    from tests.synth.scene import render as render3d

    app, room, Kc = room_map
    P = look_at(np.array([1.4, 0.4, 1.3]), np.array([0.0, 0.0, 0.5]))
    r = render3d(room, P, Kc)
    mm = np.clip(np.round(r.depth * 1000), 0, 65535)
    f = {"name": "q", "image": "q", "camera_id": "cam", "depth": "d"}
    files = {"q": _png(r.rgb), "d": _png16(mm)}
    doc = app.locate("room", {"frames": [f]}, files)
    res = doc["frames"][0]
    # this map's keyframes carry no sensor depth: located from the images, nothing tested
    assert res["status"] == "located" and res["depth_check"] == {"pose": "untested",
                                                                 "refused": 0}
    plain_doc = app.locate("room", {"frames": [{k: v for k, v in f.items() if k != "depth"}]},
                           files)
    assert "depth_check" not in plain_doc["frames"][0]
    assert plain_doc["frames"][0]["T_map_cam"] == res["T_map_cam"]
    # validation: another size, an 8-bit PNG, a missing part, a field outside the schema
    for bad, code in (({"d": _png16(mm[:100, :100])}, 400), ({"d": _png(r.rgb)}, 400),
                      ({}, 400)):
        got = app.locate("room", {"frames": [f]}, {"q": files["q"], **bad}, code)
        assert got["error"]["code"] == "invalid_request"
        assert got["error"]["details"]["field"] == "frames[0].depth"
    # a frame refused by its depth, and one rejected before matching, still say it
    monkeypatch.setattr(rl, "locate", lambda q, index: rl.Outcome(reason="depth_inconsistent",
                                                                  depth_refused=2))
    res = app.locate("room", {"frames": [f]}, files)["frames"][0]
    assert (res["status"], res["reason"]) == ("rejected", "depth_inconsistent")
    assert res["depth_check"] == {"pose": None, "refused": 2}
    from tests.fakes.service import plain

    res = app.locate("room", {"frames": [f]}, {"q": plain(320, 240), "d": files["d"]})["frames"][0]
    assert res["reason"] == "plain" and res["depth_check"] == {"pose": None, "refused": 0}


def test_the_query_depth_is_read_with_the_camera_s_scale_and_emptied_by_the_masks() -> None:
    from oh_my_slam.service.cameras import Camera

    cam = Camera(300.0, 300.0, 160.0, 120.0, 320, 240, depth_scale_m=0.002)
    rgb = np.random.default_rng(0).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    depth = np.full((240, 320), 1000, np.uint16)
    mask = np.zeros((240, 320), bool)
    mask[:, :100] = True
    fr = nav_mod.LocateFrame("q", cam, rgb, mask, None, None, None, depth)
    q, _ = nav_mod._query(fr, 0.0)
    assert q.depth is not None and q.depth.shape == (240, 320)
    assert np.all(q.depth[:, :100] == 0.0) and np.allclose(q.depth[:, 100:], 2.0)
    q, _ = nav_mod._query(nav_mod.LocateFrame("q", cam, rgb, None, None, None, None), 0.0)
    assert q.depth is None


# -- the locate index ------------------------------------------------------------------------------


def _internal(depth: np.ndarray | None, ignore: np.ndarray | None = None) -> InternalMap:
    return InternalMap([], None, None, False, lambda name: depth, ignore=lambda name: ignore)


def test_sensor_grid_takes_only_a_sensor_depth_in_use() -> None:
    from oh_my_slam.mapping.anchor import Sim3

    d = np.full((480, 640), 2.0, np.float32)
    sim = Sim3()
    f = {"depth_kind": "sensor", "internal": "k0"}
    g = sensor_grid(f, _internal(d), sim)
    assert g is not None and g.shape == (120, 160) and np.allclose(g, 2.0)
    ignore = np.zeros((480, 640), bool)
    ignore[:240] = True
    g = sensor_grid(f, _internal(d, ignore), sim)
    assert g is not None and np.all(g[:60] == 0.0) and np.allclose(g[60:], 2.0)
    assert sensor_grid({**f, "depth_kind": "mono"}, _internal(d), sim) is None
    assert sensor_grid({**f, DEPTH_UNUSED: True}, _internal(d), sim) is None
    assert sensor_grid(f, None, sim) is None
    assert sensor_grid({"depth_kind": "sensor"}, _internal(d), sim) is None
    assert sensor_grid(f, _internal(None), sim) is None


def _sensor_map(app: Nav) -> tuple[Path, dict[str, Any]]:
    """The room map's version 1 with every frame recorded as a sensor-depth keyframe, its
    derived reads removed (derived again by the next reader)."""
    import shutil

    store = app.service.store
    vfile = next(p for p in store.dir("room").rglob("1.json") if "versions" in p.parts)
    rec = json.loads(vfile.read_text())
    for f in rec["frames"]:
        f["depth_kind"] = "sensor"
    vfile.write_text(json.dumps(rec))
    shutil.rmtree(store.dir("room") / "derived" / "1")
    return vfile, rec


def test_the_index_keeps_the_sensor_depth_and_reads_an_older_index_from_the_mapper(
        room_map: tuple[Nav, Any, Any],  # noqa: F811
        monkeypatch: pytest.MonkeyPatch) -> None:
    app, _, _ = room_map
    _sensor_map(app)
    store = app.service.store
    derived = Derived(store, app.backend.read)
    idx = derived.index(derived.get("room", 1))
    assert idx.keyframes and all(k.sensor is not None and k.sensor_K is not None
                                 for k in idx.keyframes)
    k = idx.keyframes[0]
    assert k.sensor is not None and k.sensor_K is not None
    assert max(k.sensor.shape) <= rl.SENSOR_SIDE
    assert k.sensor_K[0, 2] == pytest.approx(k.sensor.shape[1] / 2.0)
    z = np.load(store.dir("room") / "derived" / "1" / "locate.npz")
    assert "sensor" in z.files and "s0" in z.files
    # an index written before oms/2.17: the grids from the version's mapper map
    path = store.dir("room") / "derived" / "1" / "locate.npz"
    kept = {name: z[name] for name in z.files if not name.startswith("s")}
    path.unlink()  # a shared blob's link: replaced, not written through
    np.savez(path, **kept)
    again = Derived(store, app.backend.read)
    idx2 = again.index(again.get("room", 1))
    assert all(k.sensor is not None for k in idx2.keyframes)
    assert np.allclose(idx2.keyframes[0].sensor, k.sensor, atol=0.01)
    from oh_my_slam.service import derived as derived_mod

    with monkeypatch.context() as m:  # a keyframe without one there keeps none
        m.setattr(derived_mod, "sensor_grid", lambda f, internal, sim: None)
        fourth = Derived(store, app.backend.read)
        assert all(k.sensor is None for k in fourth.index(fourth.get("room", 1)).keyframes)
    # ... and none when that map is gone or unreadable
    for broken in (lambda name, n: None,
                   lambda name, n: (_ for _ in ()).throw(ApiError("version_gone", "gone", {}))):
        monkeypatch.setattr(store, "mapper_for", broken)
        third = Derived(store, app.backend.read)
        assert all(k.sensor is None for k in third.index(third.get("room", 1)).keyframes)


def test_a_rival_planted_at_another_depth_through_the_service_index(
        room_map: tuple[Nav, Any, Any]) -> None:  # noqa: F811
    """End to end with the service's own index of a sensor-depth map: the frame's depth confirms
    its pose and is reported."""
    from tests.synth.scene import look_at
    from tests.synth.scene import render as render3d

    app, room, Kc = room_map
    _sensor_map(app)
    app.service.derived = Derived(app.service.store, app.backend.read)
    P = look_at(np.array([1.4, 0.4, 1.3]), np.array([0.0, 0.0, 0.5]))
    r = render3d(room, P, Kc)
    f = {"name": "q", "image": "q", "camera_id": "cam", "depth": "d"}
    files = {"q": _png(r.rgb), "d": _png16(np.clip(np.round(r.depth * 1000), 0, 65535))}
    res = app.locate("room", {"frames": [f]}, files)["frames"][0]
    assert res["status"] == "located" and res["depth_check"]["pose"] == "confirmed"
    assert math.isfinite(res["sigma_pos_m"])


# -- outside the prior (api.md §6 "Ambiguity", oms/2.17) -------------------------------------------


def _prior(q: rl.Query, pos: float, rot: float) -> rl.Query:
    q.prior = TRUTH.copy()
    q.prior_sigma_pos_m, q.prior_sigma_rot_deg = pos, rot
    return q


def test_each_rival_says_whether_the_prior_s_conflict_bound_excludes_it() -> None:
    P, D = scene()
    idx = index(P, D, BOX)
    bare = rl.locate(query(P, D, AHEAD, None), idx)
    assert bare.rivals and all("outside_prior" not in r for r in bare.rivals)
    assert bare.ambiguity is not None and "outside_prior" not in bare.ambiguity
    tight = rl.locate(_prior(query(P, D, AHEAD, None), 0.01, 1.0), idx)
    assert tight.located and _near(tight, TRUTH)
    assert [r["outside_prior"] for r in tight.rivals] == [True] * len(tight.rivals)
    assert tight.ambiguity is not None and tight.ambiguity["outside_prior"] is True
    assert set(tight.ambiguity) == {"pos_m", "rot_deg", "outside_prior"}
    # the rivals are the same, only flagged; a prior wide enough holds the rival within its bound
    assert [(r["pos_m"], r["support"]) for r in tight.rivals] == [
        (r["pos_m"], r["support"]) for r in bare.rivals]
    wide = rl.locate(_prior(query(P, D, AHEAD, None), 1.0, 30.0), idx)
    assert wide.rivals and wide.rivals[0]["outside_prior"] is False


def test_outside_prior_is_the_conflict_bound_and_rivals_of_carries_it() -> None:
    q = rl.Query(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), K, W, H)
    assert rl.outside_prior(q, [TRUTH], [(0.01, 0.1)]) is None
    _prior(q, 0.05, 2.0)
    far = _shifted(TRUTH, np.array([0.3, 0.0, 0.0]))
    near = _shifted(TRUTH, np.array([0.12, 0.0, 0.0]))
    # 3 x hypot(0.05, 0.02) = 0.16 m: 0.12 m within, 0.3 m outside; the rival's own sigma widens it
    assert rl.outside_prior(q, [near, far, far], [(0.02, 0.1), (0.02, 0.1), (0.2, 0.1)]) == [
        False, True, False]
    got = rl.rivals_of(TRUTH, [near, far], [10, 8], 10, [False, True])
    assert [r["outside_prior"] for r in got] == [False, True]
    assert "outside_prior" not in rl.rivals_of(TRUTH, [far], [8], 10)[0]


def test_a_turn_fit_keeps_the_points_stage_s_refused_count() -> None:
    base = rl.Outcome(votes=np.array([3, 1]), nearest=0, depth_refused=2)
    got = rl._with_votes(rl.Outcome(T=TRUTH, method="turn"), base)
    assert (got.depth_refused, got.nearest, list(got.votes)) == (2, 0, [3, 1])
