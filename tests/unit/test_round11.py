"""Round 11 (spec P20): the two-view measurement on repetitive texture (map.md §1 "Measurable
pair", §2 steps 2 and 8, §4 rotation consistency). A square-tiled floor makes the largest
epipolar consensus of frames turned by about 90 deg a false geometry; the pair's own depth
images, the cycle rule without depth, the depth confirmation of every engine's candidate
poses, the engine order of a job and the single-anchor rotation sigma floor."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import register_ma
from oh_my_slam.mapping import standpoint as sp
from oh_my_slam.service import prior_first
from tests.synth.room import K_ROBOT, Room, base_pose, render
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    dry,
    frames_for,
)

TILED = Room(floor="tiles")
STEP = 28.0  # the client's wheeled turn: pairs two and three steps apart are 56 and 84 deg
N = 5


def _turn(i: int) -> np.ndarray:
    return base_pose(10.0 + STEP * i, 0.3, -0.2, 0.5, 0.1, -25.0)


@functools.cache
def _frame(i: int) -> tuple[an.Features, np.ndarray]:
    rgb, z = render(TILED, _turn(i), K_ROBOT)
    return an.features(rgb), z


def _view(i: int, depth: bool = True, R: np.ndarray | None = None) -> sp.View:
    f, z = _frame(i)
    T = _turn(i)
    return sp.View(f"f{i}", K_ROBOT, f, (T[:3, :3] if R is None else R).copy(), T[:3, 3].copy(),
                   False, i, depth=(lambda: (z, K_ROBOT)) if depth else None)


def _err(i: int, j: int, tv: sp.TwoView) -> float:
    true = _turn(i)[:3, :3].T @ _turn(j)[:3, :3]
    return an.rot_deg(true.T @ tv.R_ab)


def test_a_tile_lattice_geometry_is_never_measured_with_depth() -> None:
    """The largest consensus of a pair 84 or 112 deg apart is the lattice's, about 90 deg off,
    with over 100 inliers (pinned); with the depth images the pair measures correctly or is not
    measurable, and the 28 and 56 deg pairs measure within 0.5 deg (PnP on the lifted inliers)."""
    lattice = 0
    for i in range(N):
        for j in range(i + 1, N):
            plain = sp.two_view(_view(i, False), _view(j, False))
            tv = sp.two_view(_view(i), _view(j))
            assert plain is not None and tv is not None
            if j - i >= 3:
                if plain.measurable and _err(i, j, plain) > 80.0:
                    lattice += 1  # the failure the depth images remove
                assert not tv.measurable or _err(i, j, tv) < 3.0
                assert tv.depth
            else:
                assert tv.measurable and tv.depth and _err(i, j, tv) < 0.5
    assert lattice >= 1


def _views(depth: bool, yaw: dict[int, float] | None = None) -> list[sp.View]:
    out = []
    for i in range(N):
        R = _turn(i)[:3, :3]
        if yaw and i in yaw:
            R = an.exp_so3(np.array([0.0, 0.0, np.radians(yaw[i])])) @ R
        out.append(_view(i, depth, R))
    return out


@pytest.mark.parametrize("depth", [True, False])
def test_the_rotation_check_passes_a_correct_tiled_turn_and_catches_a_wrong_one(
        depth: bool) -> None:
    """The true rotations pass (no pair beyond 10 deg: the lattice pairs are left out, by the
    depth images or by the cycle rule); one rotation yawed 15 deg, or 90 deg (the lattice's own
    turn, so no confirmation may pass it), fails through the confirmed neighbouring pairs."""
    pairs = sp.Pairs()
    bad, rows = sp.check(_views(depth), pairs)
    assert not bad and rows and max(r["difference_deg"] for r in rows) < 10.0
    for deg in (15.0, 90.0):
        bad, rows = sp.check(_views(depth, {2: deg}), pairs)
        assert bad and max(r["difference_deg"] for r in rows) > 10.0


def _tv(inliers: int, deg: float, depth: bool = False) -> sp.TwoView:
    return sp.TwoView(an.exp_so3(np.array([0.0, 0.0, np.radians(deg)])), 5.0, inliers, 0.5,
                      0.01, True, depth)


def test_the_cycle_rule_without_depth() -> None:
    """map.md §1: a pair disagreeing by more than 10 deg with the chain of strictly stronger
    measured pairs is not measurable; a pair with no stronger chain passes; pairs as strong do
    not judge each other; a pair both of whose frames carry depth keeps its confirmation; the
    outcome depends neither on the request order nor on any pose."""
    names = ["a", "b", "c", "d"]
    fits = {("a", "b"): _tv(500, 20.0), ("b", "c"): _tv(400, 20.0),
            ("a", "c"): _tv(200, -50.0),  # 90 deg off its chain a-b-c
            ("c", "d"): _tv(300, 20.0), ("b", "d"): _tv(300, 40.0),  # as strong: both kept
            ("a", "d"): _tv(150, 55.0)}  # 5 deg off: kept
    T = [base_pose(0.0)] * 4

    def run(order: list[int], fits: dict[tuple[str, str], sp.TwoView]) -> set[frozenset[str]]:
        f, _ = _frame(0)
        views = [sp.View(names[k], K_ROBOT, f, T[k][:3, :3], T[k][:3, 3], False, k)
                 for k in order]
        pairs = sp.Pairs()
        pairs.fits.update(fits)
        return {frozenset((views[i].name, views[j].name))
                for i, j, _ in sp.evaluated(views, pairs)}

    got = run([0, 1, 2, 3], fits)
    assert frozenset("ac") not in got and len(got) == 5
    assert run([3, 1, 0, 2], fits) == got
    assert frozenset("ac") in run([0, 1, 2, 3], {**fits, ("a", "c"): _tv(200, -50.0, True)})
    alone = {("a", "c"): _tv(200, -50.0)}  # no stronger chain: passes
    assert run([0, 1, 2, 3], {**{k: None for k in fits}, **alone}) == {frozenset("ac")}


def test_two_view_without_a_confirmed_hypothesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """With depth on both sides and no hypothesis the depth confirms (nor a PnP), the pair keeps
    its fit but is not measurable; a reversed fit carries the same verdict."""
    a, b = _view(0), _view(1)
    monkeypatch.setattr(sp, "depth_pose", lambda *x, **k: None)
    tv = sp.two_view(a, b)
    assert tv is not None and tv.depth and not tv.confirmed and not tv.measurable
    pairs = sp.Pairs()
    pairs.fits[("a", "b")] = tv
    back = pairs.get(sp.View("b", K_ROBOT, b.feats, b.R, b.c), sp.View("a", K_ROBOT, a.feats,
                                                                       a.R, a.c))
    assert back is not None and not back.measurable and back.depth
    monkeypatch.undo()
    uv = np.full((3, 2), 100.0)
    few = an.Epipolar(uv, uv, np.ones((3, 3)), np.ones((3, 3)), np.eye(3), 0.01)
    assert sp.depth_pose(a, b, few, np.eye(3), (_frame(0)[1], K_ROBOT),
                         (_frame(1)[1], K_ROBOT)) is None  # under six lifted inliers


def test_depth_confirmation_is_two_way() -> None:
    """``anchored.depth_confirmed``: the true relative pose of two depth images agrees both
    ways; the lattice turn (90 deg) or a view that sees nothing of the other does not."""
    za, zb = _frame(0)[1], _frame(1)[1]
    Ta, Tb = _turn(0), _turn(1)
    rel = np.linalg.inv(Ta) @ Tb
    A = an.DepthRef(np.eye(4), za, K_ROBOT.K())
    assert an.depth_confirmed(A, an.DepthRef(rel, zb, K_ROBOT.K()))
    turned = rel.copy()
    turned[:3, :3] = an.exp_so3(np.array([0.0, np.radians(90.0), 0.0])) @ rel[:3, :3]
    assert not an.depth_confirmed(A, an.DepthRef(turned, zb, K_ROBOT.K()))
    away = rel.copy()
    away[:3, 3] += np.array([0.0, 0.0, -50.0])
    assert not an.depth_confirmed(A, an.DepthRef(away, zb, K_ROBOT.K()))


# -- the service: engine order, the dry run, the sigma floor ------------------------------------


def _turn_frames(c: Client, poses: list[np.ndarray], depth: bool, off_deg: float
                 ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Frames without priors whose stubbed MapAnything rotation is ``off_deg`` off the truth."""
    fr, files = frames_for(c, poses, trust=None, depth=depth, height_above_floor_m=0.5)
    for f, T in zip(fr, poses, strict=True):
        bad = T.copy()
        bad[:3, :3] = an.exp_so3(np.array([0.0, 0.0, np.radians(off_deg)])) @ T[:3, :3]
        c.backend.mv_add(files[f["image"]], bad)
    return fr, files


def test_frames_with_depth_register_as_the_dry_run_predicts(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §2 step 8: a job's frames carrying depth images go to the geometric engine first,
    as the dry run registers them, so their statuses and poses equal the dry run's (MapAnything,
    stubbed 10 deg off, never sees them) and the update passes the rotation check."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    poses = [base_pose(25.0 * k) for k in range(6)]
    fr, files = _turn_frames(client, poses, True, 10.0)
    pred = dry(client, "m", fr, files, cameras={"cam": CAM})
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    assert client.backend.mv_calls == []
    assert [i["status"] for i in job["inputs"]] == [r["predicted_status"]
                                                    for r in pred["frames"]]
    assert pred["quality"]["rotation_pairs"] == job["quality"]["rotation_pairs"]
    T0 = np.array(job["inputs"][0]["T_map_cam"])
    for i, T in zip(job["inputs"][1:], poses[1:], strict=True):
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        assert an.rot_deg((np.linalg.inv(poses[0]) @ T)[:3, :3].T @ rel[:3, :3]) < 0.5
    assert prior_first.camera_only(prior_first_work(fr)) == 0


def prior_first_work(fr: list[dict[str, Any]]) -> Any:
    from oh_my_slam.service.update import FrameWork, UpdateWork

    return UpdateWork("m", [FrameWork(k, f["name"], "cam", b"", depth=b"d" if "depth" in f
                                      else None) for k, f in enumerate(fr)])


def test_a_monocular_frame_of_a_single_anchor_call_reports_the_sigma_floor(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §2 step 8: a MapAnything call anchored on the origin alone has no residual to
    estimate its rotation noise from: its frames report at least the floor."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    poses = [base_pose(25.0 * k, offset=0.0) for k in range(3)]
    fr, files = _turn_frames(client, poses, False, 0.0)
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    got = [i for i in job["inputs"][1:] if i["status"] == "registered"]
    assert got and all(i["sigma_rot_deg"] >= register_ma.SINGLE_ANCHOR_SIGMA_ROT_DEG
                       for i in got)
    assert job["inputs"][0]["sigma_rot_deg"] < register_ma.SINGLE_ANCHOR_SIGMA_ROT_DEG
    placed = register_ma.Placement("registered", anchors=("k1",))
    assert placed.sigma_rot_floor == register_ma.SINGLE_ANCHOR_SIGMA_ROT_DEG
    assert register_ma.Placement("registered", anchors=("k1", "k2")).sigma_rot_floor == 0.0


def test_a_tiled_turn_in_place_without_priors_commits(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The client's case (W1-W3): a no-prior first update turning in place in 28 deg steps over
    a square-tiled floor, with depth, commits without ``rotation_inconsistent``, at the true
    rotations."""
    import tests.unit.test_service_prior_first as spf

    monkeypatch.setattr(spf, "ROOM", TILED)
    spf._shot.cache_clear()
    try:
        poses = [_turn(i) for i in range(6)]
        fr, files = frames_for(client, poses, trust=None, depth=True, height_above_floor_m=0.5)
        job = client.update("m", fr, files, cameras={"cam": CAM})
    finally:
        spf._shot.cache_clear()
    assert job["state"] == "committed", job["quality"]
    assert "rotation_inconsistent" not in job["quality"]["failed_checks"]
    assert all(r["difference_deg"] <= 10.0 for r in job["quality"]["rotation_pairs"])
    reg = [(i, T) for i, T in zip(job["inputs"], poses, strict=True)
           if i["status"] == "registered"]
    assert len(reg) >= 4
    T0, P0 = np.array(reg[0][0]["T_map_cam"]), reg[0][1]
    for i, T in reg[1:]:
        rel = np.linalg.inv(T0) @ np.array(i["T_map_cam"])
        assert an.rot_deg((np.linalg.inv(P0) @ T)[:3, :3].T @ rel[:3, :3]) < 1.0


def test_the_engines_refuse_a_pose_their_depth_does_not_confirm(
        tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """map.md §2 step 2: on the geometric engine a candidate pose of a frame carrying a depth
    image is placed only if it passes the depth confirmation against the depth reference it
    registers to most (the frame is then no placement: rejected no_overlap without a prior);
    MapAnything applies the same test (``register_ma.confirms``)."""
    from tests.unit.test_anchored import frame, new_map

    am = new_map(tmp_path)
    pls = am.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(20.0), p=None)])
    assert [p.status for p in pls] == ["registered", "registered"]
    monkeypatch.setattr(an, "depth_confirmed", lambda *x, **k: False)
    pl = am.update([frame("c", base_pose(10.0), p=None)])[0]
    assert pl.status == "rejected" and pl.reason == "no_overlap"
    monkeypatch.undo()
    k = am.keyframes[0]
    ref = am._grid(k)
    assert ref is not None
    new = register_ma.New(0, tmp_path / "x.jpg", k.K, ref.depth)
    assert register_ma.confirms(new, k.T, ref.depth, ref.K, [ref])
    turned = k.T.copy()  # half a metre ahead: every depth 0.5 m off
    turned[:3, 3] += 0.5 * k.T[:3, 2]
    assert not register_ma.confirms(new, turned, ref.depth, ref.K, [ref])
    assert register_ma.confirms(register_ma.New(1, tmp_path / "y.jpg", new.K), turned,
                                ref.depth, ref.K, [ref])  # a monocular frame: no test
    assert register_ma.confirms(new, turned, ref.depth, ref.K, [])  # no depth reference


# -- spec P21: extent_seen from a top view with a support, masked views (map.md §2, §5) ---------


def _sighting(kid: str, ray: list[float], cut: list[bool] | None = None) -> Any:
    from oh_my_slam.mapping.anchored_objects import Sighting

    return Sighting(kid, 1, [0.0, 0.0, 0.0], 1.0, 100, cut or [False] * 4, ray, [0.0, 0.0, 1.0],
                    0.01, [1.0, 2.0, 3.0], "box", 0.5)


def test_a_top_view_with_a_support_plane_makes_z_full() -> None:
    """map.md §5 ``extent_seen`` (P21): an uncut view of the top (ray z below
    -``TOP_VIEW_COS``) makes z full only with a support plane measured under the footprint;
    a cut top view, or one not steep enough, does not; both engines' rule (``extent_from``)."""
    from oh_my_slam.mapping import anchored_objects as ao

    R = np.eye(3)
    top = [0.0, 0.0, -1.0]
    assert ao.TOP_VIEW_COS == ao.ACROSS_COS
    assert ao.extent_from(R, [(top, [False] * 4)], True)["z"] == "full"
    assert ao.extent_from(R, [(top, [False] * 4)], False)["z"] == "partial"
    assert ao.extent_from(R, [(top, [True, False, False, False])], True)["z"] == "partial"
    shallow = [0.0, 0.62, -0.78]  # |ray z| 0.78: a side view, its bottom cut
    assert ao.extent_from(R, [(shallow, [False, False, False, True])], True)["z"] == "partial"
    o = ao.MapObject(1, {"box": 1.0}, box=ao.Box(np.zeros(3), np.ones(3), 0.0))
    o.sightings = [_sighting("k", top)]
    assert ao.extent_seen(o)["z"] == "partial"  # no support measured
    o.supported_by = {"kind": "object", "id": 2, "verified_version": 1}
    assert ao.extent_seen(o)["z"] == "full"


def test_the_older_mappers_extent_takes_its_support() -> None:
    """``service.objects._mapper_extent``: the support plane that mapper measured (its
    ``support_z``, recorded as ``supported`` by ``derived``) completes a top view."""
    from oh_my_slam.service.objects import _mapper_extent

    cams = {"kf_a": np.array([0.0, 0.0, 2.0])}
    o = {"yaw_deg": 0.0, "views": [{"frame": "kf_a", "centroid": [0.0, 0.0, 0.5],
                                    "uncut": True}]}
    assert _mapper_extent(o, cams)["z"] == "partial"
    assert _mapper_extent({**o, "supported": True}, cams)["z"] == "full"


def _cam_down(z: float = 2.0) -> np.ndarray:
    """A camera above the origin looking straight down (map frame z up)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, z]
    return T


def test_masked_view_measures_the_projected_box() -> None:
    """``core.images.masked_view``: the object's projected box, clipped to the image, on the
    ignored pixels by more than ``IGNORED_DETECTION_MAX_SHARE`` (any grid of the image)."""
    from oh_my_slam.core.images import masked_view
    from oh_my_slam.core.types import Intrinsics

    K = Intrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    box = np.array([[x, y, 2.0] for x in (-0.2, 0.2) for y in (-0.2, 0.2)])  # u 40-60, v 30-50
    ign = np.zeros((80, 100), bool)
    assert not masked_view(box, K, None) and not masked_view(box, K, ign)
    ign[:, :52] = True  # 60 % of the box
    assert masked_view(box, K, ign)
    assert masked_view(box, K, ign[::2, ::2])  # a coarser grid of the same image
    ign[:] = False
    ign[:, :48] = True  # 40 %
    assert not masked_view(box, K, ign)
    ign[:] = True
    assert not masked_view(-box, K, ign)  # behind the camera
    assert not masked_view(box + [10.0, 0.0, 0.0], K, ign)  # off the image


def test_a_masked_view_is_no_view_of_the_object() -> None:
    """map.md §5 "Ignored pixels" (P21): a keyframe whose view holds the object mostly on its
    ignored pixels enters neither the votes nor their denominator, nor the extent evidence."""
    from oh_my_slam.core.types import Intrinsics
    from oh_my_slam.mapping import anchored_objects as ao

    K = Intrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    o = ao.MapObject(1, {"box": 1.0, "bin": 0.5},
                     box=ao.Box(np.zeros(3), np.array([0.4, 0.4, 0.4]), 0.0))
    side = [0.0, 1.0, 0.0]
    o.sightings = [_sighting("a", side), _sighting("b", side)]
    o.sightings[1].label = "bin"
    full = np.ones((80, 100), bool)
    cams = [("a", _cam_down(), K, None), ("b", _cam_down(), K, full), ("c", _cam_down(), K, None)]
    assert ao.masked_by(o, cams) == {"b"}
    assert ao.detection_votes(o, cams) == [{"label": "box", "share": 0.5}]
    assert ao.extent_seen(o)["z"] == "full"
    o.masked = {"a", "b"}
    assert ao.extent_seen(o)["z"] == "partial"
    assert ao.masked_by(ao.MapObject(2, {}), cams) == set()


def test_the_older_mapper_counts_no_masked_view(tmp_path: Any) -> None:
    """``mapping.objects.count_views`` / ``masked_frames`` (P21): a keyframe whose ignore mask
    covers the object's projected box is no view of it; ``record_ignore`` reads the masks
    beside the map's frames (staged copy first)."""
    from types import SimpleNamespace

    from PIL import Image

    from oh_my_slam.core.types import Intrinsics, Pose
    from oh_my_slam.mapping import objects as ob
    from oh_my_slam.segmentation.obb import OBB

    K = Intrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    T = _cam_down()
    pose = Pose(T[:3, :3], T[:3, 3])
    pts = np.random.default_rng(0).uniform(-0.1, 0.1, (50, 3)).astype(np.float32)
    o = ob.MapObject(1, "box", {"box": 1.0}, [0.9], pts,
                     OBB(np.zeros(3), np.eye(3), np.array([0.4, 0.4, 0.4])))
    recs = [SimpleNamespace(index=i, K_grid=K, T_map_cam=pose, image=f"frames/k{i}.jpg")
            for i in range(3)]
    (tmp_path / "frame_masks").mkdir()
    Image.fromarray(np.full((80, 100), 255, np.uint8)).save(tmp_path / "frame_masks" / "k1.jpg.png")
    tx = SimpleNamespace(current=lambda rel: tmp_path / rel)
    ignore_of = ob.record_ignore(tx)
    assert ignore_of(recs[0]) is None and ignore_of(recs[1]) is not None
    assert ob.count_views(o, recs) == 3
    o.masked = ob.masked_frames(o, recs, ignore_of)
    assert o.masked == [1] and ob.count_views(o, recs) == 2
    assert ob.MapObject.from_dict(o.to_dict(), pts).masked == [1]
    o.obb = None
    assert ob.masked_frames(o, recs, ignore_of) == []


def test_a_later_hypothesis_can_be_the_confirmed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``two_view``: the first hypothesis the depth images reject, the next confirmed is the
    fit; one without a rotation fit or too little overlap is passed over."""
    a, b = _view(0), _view(1)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    thin = an.Epipolar(e.ua[:, :] * 0 + e.ua[0], e.ub * 0 + e.ub[0], e.ra, e.rb, e.E, e.thr)
    monkeypatch.setattr(sp, "epipolar_hypotheses", lambda *x, **k: iter([e, thin, e, e]))
    verdicts = iter([False, True])
    monkeypatch.setattr(sp, "depth_confirmed", lambda *x, **k: next(verdicts))
    real = sp._fit
    fits = iter([real, real, lambda *x: None, real])
    monkeypatch.setattr(sp, "_fit", lambda *x: next(fits)(*x))
    tv = sp.two_view(a, b)
    assert tv is not None and tv.measurable and tv.depth and _err(0, 1, tv) < 0.5


def test_a_depth_frame_the_geometric_engine_leaves_unplaced_goes_to_mapanything(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """map.md §2 step 8: MapAnything takes the frames without a prior that the geometric engine
    leaves unplaced, depth frames included (and confirms them by the same depth test)."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    real = an.AnchoredMap.depth_confirms
    monkeypatch.setattr(an.AnchoredMap, "depth_confirms",
                        lambda self, f, *a: f.name != "f2" and real(self, f, *a))
    poses = [base_pose(0.0), base_pose(20.0), base_pose(40.0)]
    fr, files = _turn_frames(client, poses, True, 0.0)
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 3
    assert len(client.backend.mv_calls) == 1 and len(client.backend.mv_calls[0]) >= 2


def test_a_too_close_origin_hands_over_to_a_monocular_frame(
        client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """A no-prior first update whose depth origin is too close, the others monocular: the next
    frame in canonical order (MapAnything's) becomes the origin (map.md §1)."""
    from tests.synth.room import png_depth

    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    fr, files = frames_for(client, [base_pose(0), base_pose(20), base_pose(40)], tag="n",
                           trust=None)
    files["n0d"] = png_depth(np.full((240, 320), 0.15, np.float32))
    for f in fr[1:]:
        del files[f.pop("depth")]
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed", job["quality"]
    assert [i["status"] for i in job["inputs"]][:2] == ["rejected", "registered"]
    assert job["inputs"][0]["reason"] == "too_close"


def test_pnp_and_hypothesis_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """``depth_pose`` without a PnP; the hypotheses exhausted; a monocular keyframe carries no
    depth image for the two-view fits; MapAnything's confirmation takes the reference whose
    depth agrees most."""
    from tests.unit.test_anchored import frame, new_map

    a, b = _view(0), _view(1)
    e = an.epipolar(a.feats, a.K, b.feats, b.K)
    assert e is not None
    assert len(list(an.epipolar_hypotheses(a.feats, a.K, b.feats, b.K, 1))) == 1
    monkeypatch.setattr(sp.cv2, "solvePnPRansac", lambda *x, **k: (False, None, None, None))
    assert sp.depth_pose(a, b, e, np.eye(3), (_frame(0)[1], K_ROBOT),
                         (_frame(1)[1], K_ROBOT)) is None
    monkeypatch.undo()
    am = new_map(tmp_path)
    am.update([frame("m0", base_pose(0.0), p=None, depth=False, mono=1.0)])
    k = am.keyframes[0]
    assert k.depth_kind == "mono" and am.sensor_depth(k) is None
    assert sp.keyframe_depth(am, k) is None
    am2 = new_map(tmp_path / "two")
    am2.update([frame("a", base_pose(0.0), p=None), frame("b", base_pose(15.0), p=None)])
    refs = [r for kf in am2.keyframes for r in [am2._grid(kf)] if r is not None]
    T = am2.keyframes[0].T
    new = register_ma.New(0, tmp_path / "x.jpg", am2.keyframes[0].K, refs[0].depth)
    assert register_ma.confirms(new, T, refs[0].depth, refs[0].K, refs)
    assert register_ma.confirms(new, T, refs[0].depth, refs[0].K, refs[::-1])
