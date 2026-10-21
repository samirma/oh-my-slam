"""A video update the extension cannot register is mapped on its own and placed in the map as one
block: the similarity about the vertical its shared points agree on (``frame``), the points its
matches see at their depth (``panorama.lift``), and the own map's SfM, which registers only its
own keyframes from a copy of the map's database (``Sfm(only=...)``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.geometry import Sim3, rot_z
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import api, panorama
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import sfm as sfm_mod
from oh_my_slam.mapping.panorama import PairMatches
from tests.unit.test_mapping_cov_api import K, ctx_of, jpg, nf, quiet


def _cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform([-3, -3, 0], [3, 3, 2.5], size=(n, 3))


def test_an_upright_similarity_recovers_yaw_scale_and_translation() -> None:
    src = _cloud()
    S = Sim3(1.7, rot_z(np.radians(35.0)), np.array([4.0, -2.0, 0.3]))
    fit = mframe.upright_similarity(src, S.apply(src))
    assert fit.s == pytest.approx(1.7) and np.allclose(fit.R, S.R) and np.allclose(fit.t, S.t)
    same = mframe.upright_similarity(np.zeros((3, 3)), np.ones((3, 3)))  # no spread: scale 1
    assert same.s == 1.0 and np.allclose(same.t, 1.0)


def test_the_robust_fit_keeps_the_agreeing_points_only() -> None:
    src = _cloud(300, seed=1)
    S = Sim3(0.6, rot_z(np.radians(-120.0)), np.array([1.0, 2.0, -0.5]))
    dst = S.apply(src)
    rng = np.random.default_rng(2)
    bad = rng.choice(len(src), 90, replace=False)  # 30 % wrong correspondences
    dst[bad] += rng.uniform(-3, 3, size=(90, 3))
    out = mframe.robust_upright_similarity(src, dst, np.full(len(src), 0.05))
    assert out is not None
    fit, inl = out
    assert fit.s == pytest.approx(0.6, abs=1e-6) and np.allclose(fit.R, S.R, atol=1e-6)
    assert not inl[bad].any() and inl.sum() == len(src) - 90
    # points all at one spot horizontally, or a single point: nothing to turn about
    assert mframe.robust_upright_similarity(src[:1], dst[:1], np.full(1, 0.05)) is None
    stack = np.column_stack([np.zeros(5), np.zeros(5), np.arange(5.0)])
    assert mframe.robust_upright_similarity(stack, stack, np.full(5, 0.05)) is None


def test_matched_keypoints_are_lifted_with_the_depth_of_their_view() -> None:
    grid = K.resized(64, 48)
    depth = np.full((48, 64), 2.0, np.float32)
    depth[:, 32:] = 0.0  # no depth on the right half
    pose = Pose(rot_z(np.radians(90.0)), np.array([1.0, 0.0, 0.0]))
    view = panorama.View(pose, K, depth, grid, (640, 480))
    P = panorama.lift(view, np.array([[160.0, 240.0], [100.0, 240.0], [600.0, 240.0]]))
    assert np.allclose(P[0], pose.R @ [2.0 * (160 - 320) / 500, 0.0, 2.0] + pose.t, atol=1e-6)
    assert np.allclose(P[1], pose.R @ [2.0 * (100 - 320) / 500, 0.0, 2.0] + pose.t, atol=1e-6)
    assert np.isnan(P[2]).all()


def test_an_own_map_registers_its_keyframes_only(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    import pycolmap

    seen: list[Any] = []
    monkeypatch.setattr(pycolmap, "global_mapping", lambda db, img, out, opts:
                        seen.append(("global", list(opts.image_names))) or {})
    monkeypatch.setattr(pycolmap, "incremental_mapping", lambda db, img, out, opts, input_path:
                        seen.append(("incremental", list(opts.image_names))) or {})
    own = sfm_mod.Sfm(tmp_path / "db.db", tmp_path, tmp_path / "w", only=["b.jpg", "a.jpg"])
    assert own.map_global(tmp_path / "g") is None and own.map_incremental(tmp_path / "i") is None
    sfm_mod.Sfm(tmp_path / "db.db", tmp_path, tmp_path / "w").map_global(tmp_path / "g")
    assert seen == [("global", ["a.jpg", "b.jpg"]), ("incremental", ["a.jpg", "b.jpg"]),
                    ("global", [])]
    monkeypatch.setattr(pycolmap, "incremental_mapping", lambda db, img, out, opts, input_path:
                        seen.append(("refine_focal", opts.ba_refine_focal_length)) or {})
    own.map_incremental(tmp_path / "i", hold_focal=True)
    own.map_incremental(tmp_path / "i")
    assert seen[-2:] == [("refine_focal", False), ("refine_focal", True)]


# --- placing the own map ------------------------------------------------------------------------


def _align_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pairs: int = 4,
                 scale: float = 1.25) -> SimpleNamespace:
    """The map's keyframe 0 and the new keyframe 10 are one camera before a wall 2 m away; the
    own map's frame is the map's through ``S`` (its units 1 / ``scale`` of a metre)."""
    S = Sim3(scale, rot_z(np.radians(40.0)), np.array([0.5, -0.2, 0.1]))
    T_old = Pose.identity()
    T_own = mframe.transform_pose(S.inverse(), T_old)
    uv = np.random.default_rng(3).uniform([40, 40], [600, 440], size=(160, 2))
    root = tmp_path / "map"
    (root / "per_frame" / "f000000").mkdir(parents=True)
    np.save(root / "per_frame" / "f000000" / "depth.npy", np.full((48, 64), 2.0, np.float16))
    old = [SimpleNamespace(image=f"frames/{jpg(0)}", name="f000000", T_map_cam=T_old, K=K,
                           K_grid=K.resized(64, 48), width=640, height=480)]
    new = nf(10)
    new.frame.depth = np.full((48, 64), 2.0 / scale, np.float32)
    matches = [PairMatches(jpg(0), jpg(10), uv[k::pairs], uv[k::pairs]) for k in range(pairs)]
    matches.append(PairMatches(jpg(10), jpg(11), uv, uv))  # among the new: not used
    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: matches)
    sfm = SimpleNamespace(db=Path("db"), image_intrinsics=lambda names: {n: (1, K) for n in names})
    ctx = ctx_of([new, nf(11)], old, tx=SimpleNamespace(current=lambda rel: root / rel))
    return SimpleNamespace(ctx=ctx, sfm=sfm, own={jpg(10): T_own, jpg(11): T_own}, S=S,
                           T_old=T_old)


def test_an_own_map_is_placed_by_the_points_it_shares_with_the_map(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _align_world(monkeypatch, tmp_path)
    w.ctx.notes["own_map"] = {"scale": 2.0}
    msgs: list[str] = []
    placed = api._align_own_map(w.ctx, w.sfm, w.own, msgs.append)
    assert set(placed) == {jpg(10), jpg(11)}  # the whole block, matched or not
    assert np.allclose(placed[jpg(10)].matrix(), w.T_old.matrix(), atol=1e-2)
    note = w.ctx.notes["own_map"]
    assert note["aligned"] and note["pairs"] == 4 and note["inliers"] >= 100
    assert note["scale"] == pytest.approx(2.0 * 1.25, abs=1e-2) and abs(note["yaw_deg"] - 40) < 0.5
    assert "placed by" in msgs[-1]


@pytest.mark.parametrize("pairs,scale", [(2, 1.25), (4, 3.5)])
def test_an_own_map_too_loosely_or_oddly_tied_to_the_map_is_not_placed(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pairs: int, scale: float) -> None:
    w = _align_world(monkeypatch, tmp_path, pairs=pairs, scale=scale)
    msgs: list[str] = []
    assert api._align_own_map(w.ctx, w.sfm, w.own, msgs.append) == {}
    assert w.ctx.notes["own_map"]["aligned"] is False and "not placed" in msgs[-1]


def test_an_own_map_without_shared_points_is_not_placed(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    w = _align_world(monkeypatch, tmp_path)
    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: [])
    msgs: list[str] = []
    assert api._align_own_map(w.ctx, w.sfm, w.own, msgs.append) == {}
    assert w.ctx.notes["own_map"] == {"aligned": False} and "no points" in msgs[-1]


# --- mapping the new keyframes on their own -----------------------------------------------------


class _OwnModel:
    def __init__(self, poses: dict[str, Pose], method: str, metric: bool) -> None:
        self.poses, self.method = poses, method
        self.notes: dict[str, Any] = {"metric": True} if metric else {}
        self.transformed = False

    def transform(self, s: float, R: Any, t: Any) -> None:
        self.transformed = True

    @property
    def registered(self) -> list[str]:
        return sorted(self.poses)

    def pose(self, name: str) -> Pose:
        return self.poses[name]


@pytest.mark.parametrize("metric,scale,expected", [(True, None, 1.0), (False, 2.0, 2.0),
                                                   (False, None, 1.0)])
def test_new_keyframes_are_mapped_on_their_own_in_a_levelled_metric_frame(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metric: bool, scale: float | None,
        expected: float) -> None:
    (tmp_path / "db.db").write_bytes(b"db")
    poses = {jpg(i): Pose(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in (10, 11, 12, 13)}
    seen: dict[str, Any] = {}

    def run(ctx: Any, is_video: bool, client: Any, progress: Any, own_db: Path | None = None
            ) -> _OwnModel:
        seen.update(new=[nf.kf.name for nf in ctx.new], old=ctx.old_frames, db=own_db.read_bytes())
        progress("posed")
        return _OwnModel(poses, "sfm-global", metric)

    def metric_scale(model: Any, fds: Any) -> Any:
        if scale is None:
            raise ValueError("no points")
        return SimpleNamespace(scale=scale)

    monkeypatch.setattr(api, "_run_sfm", run)
    monkeypatch.setattr(mframe, "metric_scale", metric_scale)
    monkeypatch.setattr(mframe, "world_up", lambda p, f: np.array([0.0, 0.0, 1.0]))
    ctx = ctx_of([nf(i) for i in (10, 11, 12, 13, 14)], [SimpleNamespace()], tmp=tmp_path)
    msgs: list[str] = []
    out = api._own_map(ctx, SimpleNamespace(db=tmp_path / "db.db"),
                       {jpg(i) for i in (10, 11, 12, 13)}, None, msgs.append)
    assert out is not None
    placed, method, is_metric = out.poses, out.method, out.metric
    assert out.model.transformed  # the reconstruction is brought into the levelled frame
    assert is_metric is (metric or scale is not None)
    assert method == "sfm-global" and seen["old"] == [] and seen["db"] == b"db"
    assert seen["new"] == [f"f{i:06d}" for i in (10, 11, 12, 13)]
    d = np.linalg.norm(placed[jpg(13)].t - placed[jpg(10)].t)
    assert d == pytest.approx(3.0 * expected)
    assert ctx.notes["own_map"] == {"method": "sfm-global", "posed": 4, "of": 4,
                                    "scale": expected, "metric": is_metric}
    assert msgs == ["mapping 4 keyframes on their own", "own map: posed"]


def test_an_own_map_of_fewer_than_three_posed_keyframes_is_none(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "db.db").write_bytes(b"db")
    monkeypatch.setattr(api, "_run_sfm", lambda *a, **kw: _OwnModel(
        {jpg(10): Pose.identity()}, "multiview", True))
    ctx = ctx_of([nf(10), nf(11)], [SimpleNamespace()], tmp=tmp_path)
    assert api._own_map(ctx, SimpleNamespace(db=tmp_path / "db.db"), {jpg(10), jpg(11)}, None,
                        quiet) is None
    assert ctx.notes["own_map"]["posed"] == 1


def test_keyframes_the_extension_registered_place_the_own_map_by_their_centres(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A block whose keyframes share no points with the stored ones (they meet the map through
    keyframes incremental SfM registered) is placed by those keyframes' centres."""
    w = _align_world(monkeypatch, tmp_path)
    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: [])
    ids = range(10, 10 + api.OWN_ALIGN_MIN_INLIERS + 2)
    own = {jpg(i): Pose(np.eye(3), np.array([0.3 * i, 0.1 * (i % 2), 0.0])) for i in ids}
    S = Sim3(1.1, rot_z(np.radians(-25.0)), np.array([2.0, 1.0, 0.0]))
    last = jpg(max(ids))
    posed = {n: mframe.transform_pose(S, T) for n, T in own.items() if n != last}
    w.ctx.new += [nf(i) for i in ids if i > 11]
    placed = api._align_own_map(w.ctx, w.sfm, own, quiet, posed=posed)
    assert set(placed) == set(own)
    assert np.allclose(placed[last].t, S.apply(own[last].t[None])[0], atol=1e-6)
    assert w.ctx.notes["own_map"]["pairs"] == len(posed)  # each registered keyframe counts once


# --- the extension's paths ----------------------------------------------------------------------


class _ExtSfm:
    def __init__(self, inc: Any = None) -> None:
        self.inc = inc
        self.db = Path("db")
        self.extended: dict[str, Any] = {}

    def connected(self, seeds: set[str], candidates: set[str]) -> set[str]:
        return set(candidates)

    def model_cameras(self, path: Path) -> set[int]:
        return {1}

    def map_incremental(self, out: Path, **kw: Any) -> Any:
        return self.inc

    def image_intrinsics(self, names: set[str]) -> dict[str, tuple[int, Any]]:
        return {n: (1, K) for n in names}

    def extend_with_poses(self, base: Path, poses: dict[str, Pose], out: Path, method: str,
                          cameras: Any = None) -> Any:
        self.extended = {"poses": dict(poses), "method": method}
        return SimpleNamespace(method=method, registered=sorted(poses))


def _ext(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n: int = 14, aligned: bool = True,
         lens: int | None = None, metric: bool = True) -> SimpleNamespace:
    from tests.unit.test_mapping_cov_api import FakeModel, _ext_world

    tx, old = _ext_world(tmp_path)
    new = [nf(100 + k) for k in range(n)]
    names = {jpg(100 + k) for k in range(n)}
    inc = FakeModel({**{jpg(i): r.T_map_cam for i, r in enumerate(old)},
                     jpg(100): Pose.identity(), jpg(101): Pose.identity()})
    seen: dict[str, Any] = {"refined": []}
    monkeypatch.setattr(api, "_contradicting", lambda s, p, k, judged: {})
    monkeypatch.setattr(api, "_depth_consistent", lambda c, m, name: True)
    monkeypatch.setattr(api, "_own_map", lambda ctx, sfm, names_, client, progress: (
        seen.__setitem__("own_of", set(names_)) or api.OwnMap(
            {n_: Pose.identity() for n_ in names_ if n_ != jpg(113)}, "sfm-global", metric,
            None)))  # type: ignore[arg-type]

    def align(ctx: Any, sfm: Any, own: Any, progress: Any, posed: Any = None,
              with_scale: bool = True) -> Any:
        seen.update(posed=set(posed or {}), with_scale=with_scale)
        return dict(own) if aligned else {}

    monkeypatch.setattr(api, "_align_own_map", align)
    monkeypatch.setattr(api, "_multiview_poses", lambda ctx, todo, pool, client, temporal=False:
                        {v.name: Pose.identity() for v in todo})

    def refine(ctx: Any, sfm: Any, poses: Any, free: set[str], refine_focal: bool,
               rotation: bool, model: Any = None, hold_centres: bool = False) -> Any:
        seen["refined"].append((set(free), hold_centres))
        ctx.notes["pose_refinement"] = {"median_after_deg": 0.3}
        return poses, 1.0, 0.0

    monkeypatch.setattr(api, "_refine_multiview", refine)
    monkeypatch.setattr(api, "_depth_contradicted_placements",
                        lambda ctx, sfm, poses, free, K, video: (
                            seen.__setitem__("judged", set(free)) or set()))
    ctx = ctx_of(new, old, tx=tx, lens_camera=lens)
    return SimpleNamespace(ctx=ctx, sfm=_ExtSfm(inc), names=names, seen=seen)


def test_a_video_the_extension_leaves_unregistered_is_placed_as_its_own_map(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _ext(monkeypatch, tmp_path)
    model = api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert model.method == "sfm-incremental+own-map+multiview"
    assert w.seen["own_of"] == w.names  # all of the video, registered or not
    assert w.seen["posed"] == {jpg(100), jpg(101)}  # placed by those too
    assert w.seen["with_scale"] is False  # both metric: the block keeps its size
    assert w.ctx.notes["mv_names"] == [jpg(113)]  # the one the own map did not hold
    assert w.seen["refined"] == [(w.names, False)]  # every new keyframe moves


def test_keyframes_whose_placement_their_links_contradict_are_left_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _ext(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "_depth_contradicted_placements",
                        lambda ctx, sfm, poses, free, K, video: (
                            w.seen.__setitem__("judged", (set(free), set(K), video))
                            or {jpg(105), jpg(106)}))
    api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert w.seen["judged"] == (w.names, w.names, True)  # every keyframe the own map moved
    assert set(w.sfm.extended["poses"]) == w.names - {jpg(105), jpg(106)}
    # keyframes SfM alone registered are not judged
    w = _ext(monkeypatch, tmp_path / "s", n=2)
    api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert "judged" not in w.seen and set(w.sfm.extended["poses"]) == w.names


def test_photos_or_an_unplaced_own_map_take_the_multiview_path(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _ext(monkeypatch, tmp_path, aligned=False)
    model = api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert model.method == "sfm-incremental+multiview"
    assert w.seen["refined"] == [(w.names - {jpg(100), jpg(101)}, False)]
    w = _ext(monkeypatch, tmp_path / "p")
    api._extend(w.ctx, w.sfm, w.names, False, False, None, quiet, 0.0)
    assert "own_of" not in w.seen  # photos: no own map
    w = _ext(monkeypatch, tmp_path / "f", n=8)
    api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert "own_of" not in w.seen  # too few unregistered keyframes


def test_an_own_map_without_sfm_points_is_aligned_densely(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    w = _ext(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "_own_map", lambda ctx, sfm, names_, client, progress: api.OwnMap(
        {n_: Pose.identity() for n_ in names_}, "multiview", True, None))  # type: ignore[arg-type]
    api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert w.ctx.notes["mv_names"] == sorted(w.names)


@pytest.mark.parametrize("metric,map_scale,with_scale", [
    (False, "moge_over_sfm_median", True),  # the block has no scale of its own
    (True, "multiview_depth", True),  # the map's is a pan's, at its own depth
])
def test_the_block_is_scaled_when_one_of_the_scales_is_weak(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metric: bool, map_scale: str,
        with_scale: bool) -> None:
    w = _ext(monkeypatch, tmp_path, metric=metric)
    w.ctx.meta["scale"] = {"method": map_scale}
    api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert w.seen["with_scale"] is with_scale


@pytest.mark.parametrize("median,taken", [(0.4, True), (1.4, False)])
def test_a_wide_angle_camera_turning_in_place_gets_one_centre(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, median: float, taken: bool) -> None:
    w = _ext(monkeypatch, tmp_path, n=4, lens=2)

    def refine(ctx: Any, sfm: Any, poses: Any, free: set[str], refine_focal: bool,
               rotation: bool, model: Any = None, hold_centres: bool = False) -> Any:
        w.seen["refined"].append((set(free), hold_centres))
        ctx.notes["pose_refinement"] = {"median_after_deg": median if hold_centres else 0.3}
        if hold_centres:
            ctx.pose_support["x"] = (median, 1)
        return poses, 1.0, 0.0

    monkeypatch.setattr(api, "_refine_multiview", refine)
    centres = {jpg(100): [0.0, 0, 0], jpg(101): [0.02, 0, 0], jpg(102): [0.5, 0, 0],
               jpg(103): [0.04, 0, 0]}
    monkeypatch.setattr(api, "_multiview_poses", lambda ctx, todo, pool, client, temporal=False:
                        {v.name: Pose(np.eye(3), np.array(centres[v.name])) for v in todo})
    api._extend(w.ctx, w.sfm, w.names, False, False, None, quiet, 0.0)
    note = w.ctx.notes["one_centre"]
    assert note["taken"] is taken and note["keyframes"] == 4
    placed = w.sfm.extended["poses"]
    if taken:
        # fewer than three SfM keyframes: the median of all their centres, x 0.02
        assert all(np.allclose(placed[n].t, [0.02, 0, 0]) for n in w.names)
        assert w.seen["refined"][-1] == (w.names, True)
    else:
        assert np.allclose(placed[jpg(102)].t, [0.5, 0, 0])  # left as it was, notes restored
        assert w.ctx.notes["pose_refinement"] == {"median_after_deg": 0.3}
        assert "x" not in w.ctx.pose_support


def test_one_centre_needs_three_keyframes() -> None:
    assert api._one_centre(ctx_of([]), None, {jpg(1): Pose.identity()}, {jpg(1), jpg(2)}) is None  # type: ignore[arg-type]


# --- loosely verified pairs ---------------------------------------------------------------------


class _LooseSfm:
    def __init__(self, inliers: dict[frozenset[str], int]) -> None:
        self.inliers = inliers
        self.rematched: set[frozenset[str]] = set()

    def verified_pairs(self, min_inliers: int = 15) -> dict[frozenset[str], tuple[int, int]]:
        return {p: (n, 2) for p, n in self.inliers.items() if n >= min_inliers}

    def rematch_lightglue(self, pairs: set[frozenset[str]]) -> dict[str, int]:
        self.rematched = set(pairs)
        return {"pairs": len(pairs), "verified_before": 1, "verified_after": 2}


def test_loosely_verified_map_and_loop_pairs_are_matched_again_most_similar_first(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.test_mapping_cov_api import _old

    e = np.eye(8)
    old = [_old(0, tmp_path, e[0]), _old(1, tmp_path, None)]
    new = [nf(100, e[0]), nf(101, e[1]), nf(100 + api.SEQ_OVERLAP + 5, e[1] + e[2])]
    far = 100 + api.SEQ_OVERLAP + 5
    names = {0: jpg(0), 1: jpg(1), 100: jpg(100), 101: jpg(101), far: jpg(far)}
    pairs = {(0, 100), (1, 101), (100, 101), (101, far), (0, 101)}
    sfm = _LooseSfm({frozenset((jpg(0), jpg(100))): 120,  # verified well: kept
                     frozenset((jpg(1), jpg(101))): 20, frozenset((jpg(101), jpg(far))): 3})
    ctx = ctx_of(new, old, tx=SimpleNamespace(current=lambda rel: tmp_path / rel))
    monkeypatch.setattr(api, "LOOSE_MAX_PAIRS", 2)
    msgs: list[str] = []
    api._strengthen_loose_pairs(ctx, sfm, pairs, names, msgs.append)  # type: ignore[arg-type]
    # candidates: (1,101) without a stored descriptor (similarity 0), the loop (101, far) at 0.71
    # and (0, 101) at 0: the two most similar, ties by ids
    assert sfm.rematched == {frozenset((jpg(101), jpg(far))), frozenset((jpg(0), jpg(101)))}
    assert ctx.notes["loose_pairs"] == {"candidates": 3, "pairs": 2, "verified_before": 1,
                                        "verified_after": 2}
    assert "LightGlue on 2 loosely verified pairs" in msgs[0]
    # a new video's sequential pairs only: nothing loose, nothing matched
    sfm2 = _LooseSfm({})
    ctx2 = ctx_of(new[:2], [], tx=None)
    api._strengthen_loose_pairs(ctx2, sfm2, {(100, 101)}, names, quiet)  # type: ignore[arg-type]
    assert sfm2.rematched == set() and "loose_pairs" not in ctx2.notes


def test_an_own_map_is_posed_from_its_database_copy_without_matching(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.test_mapping_cov_api import FakeModel

    made: list[Any] = []

    class _Sfm:
        def __init__(self, db: Path, image_dir: Path, work: Path, only: Any = None) -> None:
            made.append((db, sorted(only) if only is not None else None))

        def two_view_stats(self, names: Any = None) -> dict[str, float]:
            return {"verified_pairs": 4.0, "rotation_fraction": 0.0, "planar_fraction": 0.0}

        def largest_component(self, names: set[str]) -> set[str]:
            return set(names)

        def rotation_pairs(self) -> set[frozenset[str]]:
            return set()

        def map_global(self, out: Path) -> None:
            return None

        def map_incremental(self, out: Path) -> None:
            return None

    monkeypatch.setattr(api, "check_versions", lambda: "4.2.1")
    monkeypatch.setattr(api, "Sfm", _Sfm)
    monkeypatch.setattr(api, "_features_and_matching", lambda *a: pytest.fail("matched"))
    sentinel = FakeModel({jpg(1): Pose.identity()})
    monkeypatch.setattr(api, "_vetted", lambda ctx, m, turns: None)
    monkeypatch.setattr(api, "_plausible_focal", lambda ctx, m, prior: m)
    monkeypatch.setattr(api, "_multiview_poses", lambda *a, **kw: {jpg(1): Pose.identity()})
    monkeypatch.setattr(api, "_refine_multiview", lambda *a, **kw: ({jpg(1): Pose.identity()},
                                                                    1.0, 0.0))
    _Sfm.triangulate_with_poses = lambda self, poses, out, focal_scale=1.0, distortion=0.0: (  # type: ignore[attr-defined]
        sentinel)
    tx = SimpleNamespace(stage=lambda rel: tmp_path / rel)
    ctx = ctx_of([nf(1), nf(2)], [], tmp=tmp_path, tx=tx)
    assert api._run_sfm(ctx, True, None, quiet, own_db=tmp_path / "copy.db") is sentinel
    assert made == [(tmp_path / "copy.db", [jpg(1), jpg(2)])]


def test_a_metric_block_keeps_its_size() -> None:
    src = _cloud(80, seed=5)
    S = Sim3(1.0, rot_z(np.radians(70.0)), np.array([1.0, 0.0, 0.2]))
    out = mframe.robust_upright_similarity(src, S.apply(src) * 1.02, np.full(80, 0.2),
                                           with_scale=False)
    assert out is not None and out[0].s == 1.0


@pytest.mark.parametrize("update_", [False, True])
def test_only_a_video_added_to_a_map_has_its_loose_pairs_matched_again(
        monkeypatch: pytest.MonkeyPatch, update_: bool) -> None:
    calls: list[str] = []
    monkeypatch.setattr(api, "_strengthen_weak_links", lambda *a: calls.append("weak"))
    monkeypatch.setattr(api, "_strengthen_loose_pairs", lambda *a: calls.append("loose"))
    monkeypatch.setattr(api, "_calibrate_new_camera", lambda *a: False)
    sfm = SimpleNamespace(extract=lambda names, prior, video=False: 1,
                          match_pairs=lambda pairs, names: len(pairs))
    old = [SimpleNamespace(index=0)] if update_ else []
    monkeypatch.setattr(api, "_pairs_update", lambda ctx, v: {(0, 1)})
    ctx = ctx_of([nf(1), nf(2)], old)
    api._features_and_matching(ctx, sfm, {0: jpg(0), 1: jpg(1), 2: jpg(2)}, None, True, None,  # type: ignore[arg-type]
                               quiet)
    assert calls == (["weak", "loose"] if update_ else ["weak"])


# --- a pan from one spot rebuilt with a video ---------------------------------------------------


class _PanSfm:
    def __init__(self, inc: Any) -> None:
        self.inc = inc
        self.extended: dict[str, Any] = {}
        self.incremental: dict[str, Any] = {}

    def image_intrinsics(self, names: set[str]) -> dict[str, tuple[int, Any]]:
        return {n: (2, K) for n in names}

    def model_cameras(self, path: Path) -> set[int]:
        return {1}

    def map_incremental(self, out: Path, **kw: Any) -> Any:
        self.incremental = kw
        return self.inc

    def extend_with_poses(self, base: Path, poses: dict[str, Pose], out: Path,
                          method: str) -> Any:
        self.extended = {"poses": dict(poses), "method": method}
        return SimpleNamespace(method=method, notes={})


def _pan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inc_names: list[str],
         own: bool = True) -> SimpleNamespace:
    from tests.unit.test_mapping_cov_api import FakeModel

    stored = [nf(i) for i in range(3)]
    video = [nf(10 + i) for i in range(5)]
    seen: dict[str, Any] = {}
    model = SimpleNamespace(write=lambda path: seen.__setitem__("base", path))
    monkeypatch.setattr(api, "_own_map", lambda ctx, sfm, names, client, progress: (
        seen.__setitem__("own_of", set(names)) or (api.OwnMap(
            {n: Pose.identity() for n in names}, "sfm-global", True, model) if own else None)))
    monkeypatch.setattr(api, "_multiview_poses", lambda ctx, todo, pool, client, temporal=False:
                        seen.__setitem__("mv", ([v.name for v in todo], sorted(
                            v.name for v in pool))) or {v.name: Pose.identity() for v in todo})
    monkeypatch.setattr(api, "_one_centre", lambda ctx, sfm, poses, names, model=None:
                        seen.__setitem__("one", set(names)) or None)
    inc = FakeModel({n: Pose.identity() for n in inc_names})
    ctx = ctx_of(stored + video, tmp=tmp_path,
                 rebuild=api.Rebuild({f"f{i:06d}": 1 for i in range(3)}, {}, {}, 1, {}, {}, []))
    return SimpleNamespace(ctx=ctx, sfm=_PanSfm(inc), seen=seen,
                           names={jpg(i) for i in (0, 1, 2, 10, 11, 12, 13, 14)})


def test_a_pan_is_registered_onto_the_video_mapped_on_its_own(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _pan(monkeypatch, tmp_path, [jpg(0), jpg(1), jpg(10)])
    model = api._pan_onto_video(w.ctx, w.sfm, w.names, None, quiet, 0.0)  # type: ignore[arg-type]
    assert w.seen["own_of"] == {jpg(10 + i) for i in range(5)}  # the video alone
    assert w.sfm.incremental["constant_cameras"] == {1, 2}  # the pan's lens held
    assert w.seen["mv"][0] == [jpg(2)]  # the one incremental mapping left
    assert w.seen["one"] == {jpg(0), jpg(1), jpg(2)}
    assert set(w.sfm.extended["poses"]) == {jpg(0), jpg(1), jpg(2)}  # the video's are kept
    assert model.notes == {"metric": True, "scale_method": "own_map"}
    assert w.ctx.notes["mv_names"] == [jpg(2)]


def test_a_video_that_does_not_map_on_its_own_abandons_the_rebuild(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from oh_my_slam.core.errors import RegistrationError

    w = _pan(monkeypatch, tmp_path, [], own=False)
    with pytest.raises(RegistrationError):
        api._pan_onto_video(w.ctx, w.sfm, w.names, None, quiet, 0.0)  # type: ignore[arg-type]
    w = _pan(monkeypatch, tmp_path / "b", [jpg(0), jpg(1), jpg(2)])
    w.sfm.inc = None  # incremental mapping registers nothing: every pan keyframe by multi-view
    api._pan_onto_video(w.ctx, w.sfm, w.names, None, quiet, 0.0)  # type: ignore[arg-type]
    assert w.seen["mv"][0] == [jpg(0), jpg(1), jpg(2)]


def test_a_rebuild_with_a_video_matches_the_pan_with_all_of_it_and_maps_it_onto_the_video(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    matched: list[set[tuple[int, int]]] = []
    monkeypatch.setattr(api, "_strengthen_weak_links", lambda *a: None)
    sfm = SimpleNamespace(extract=lambda names, prior, video=False: 1,
                          match_pairs=lambda pairs, names: matched.append(set(pairs)) or 0)
    rb = api.Rebuild({"f000000": 1, "f000001": 1}, {}, {}, 1, {}, {}, [])
    ctx = ctx_of([nf(0), nf(1), nf(50), nf(51)], rebuild=rb)
    api._features_and_matching(ctx, sfm, {i: jpg(i) for i in (0, 1, 50, 51)}, None, True,  # type: ignore[arg-type]
                               None, quiet)
    assert {(0, 50), (0, 51), (1, 50), (1, 51)} <= matched[0]
    # and _run_sfm takes the pan-onto-video path
    monkeypatch.setattr(api, "check_versions", lambda: "4.2.1")
    monkeypatch.setattr(api, "_features_and_matching", lambda *a: 0)
    monkeypatch.setattr(api, "Sfm", lambda *a, **kw: SimpleNamespace(
        two_view_stats=lambda names=None: {"rotation_fraction": 0.0}))
    monkeypatch.setattr(api, "_pan_onto_video", lambda ctx_, sfm_, names, client, progress, t0:
                        "pan")
    ctx.tx = SimpleNamespace(stage=lambda rel: tmp_path / rel, clone_for_edit=lambda rel: tmp_path)
    assert api._run_sfm(ctx, True, None, quiet) == "pan"


@pytest.mark.parametrize("floor_z,shift", [(1.9, -0.1), (1.0, None), (None, None)])
def test_a_placed_block_is_put_on_the_map_s_floor_when_it_is_close(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, floor_z: float | None,
        shift: float | None) -> None:
    """The fixture's camera looks up at a surface 2 m above it: the block's lowest supported
    horizontal surface. A map floor 10 cm below it takes the block down; one 1 m away does not."""
    w = _align_world(monkeypatch, tmp_path)
    if floor_z is not None:
        w.ctx.meta["floor_z"] = floor_z
    placed = api._align_own_map(w.ctx, w.sfm, w.own, quiet)
    note = w.ctx.notes["own_map"]
    assert note.get("floor_shift_m") == (None if shift is None else pytest.approx(shift, abs=0.02))
    dz = placed[jpg(10)].t[2] - w.T_old.t[2]
    assert dz == pytest.approx(shift or 0.0, abs=0.02)


def test_pairs_verified_well_are_not_matched_again(tmp_path: Path) -> None:
    from tests.unit.test_mapping_cov_api import _old

    sfm = _LooseSfm({frozenset((jpg(0), jpg(100))): 90})
    ctx = ctx_of([nf(100)], [_old(0, tmp_path, None)])
    api._strengthen_loose_pairs(ctx, sfm, {(0, 100)}, {0: jpg(0), 100: jpg(100)}, quiet)  # type: ignore[arg-type]
    assert sfm.rematched == set() and "loose_pairs" not in ctx.notes


def test_a_pan_registered_whole_needs_no_multiview(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    w = _pan(monkeypatch, tmp_path, [jpg(0), jpg(1), jpg(2)])
    api._pan_onto_video(w.ctx, w.sfm, w.names, None, quiet, 0.0)  # type: ignore[arg-type]
    assert "mv" not in w.seen and "mv_names" not in w.ctx.notes


def test_a_video_that_does_not_map_on_its_own_is_placed_as_before(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _ext(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "_own_map", lambda *a: None)
    model = api._extend(w.ctx, w.sfm, w.names, False, True, None, quiet, 0.0)
    assert model.method == "sfm-incremental+multiview" and "posed" not in w.seen


def test_held_centres_refine_with_the_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def refine(pairs: Any, views: Any, free: Any, **kw: Any) -> Any:
        seen.update(kw)
        return panorama.PoseFit({}, 1.0, 0, 0, 1.0, 0.5)

    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: [])
    monkeypatch.setattr(panorama, "refine_poses", refine)
    sfm = SimpleNamespace(db=Path("db"), image_intrinsics=lambda names: {})
    api._refine_multiview(ctx_of([nf(1)]), sfm, {jpg(1): Pose.identity()}, {jpg(1)},  # type: ignore[arg-type]
                          refine_focal=False, rotation=False, hold_centres=True)
    assert seen["hold_centres"] is True and "use_depth" not in seen


def test_a_block_without_a_floor_of_its_own_is_left_where_it_was(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _align_world(monkeypatch, tmp_path)
    w.ctx.meta["floor_z"] = 1.9
    for f in w.ctx.new:
        f.frame.depth = np.zeros_like(f.frame.depth)  # no depth to find a floor in
    w.ctx.new[1].frame.depth = None
    S = Sim3(1.0, np.eye(3), np.zeros(3))
    assert api._on_the_floor(w.ctx, w.own, S, {}) is S


# --- placements the depth of their links contradicts --------------------------------------------


def _links_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """The map's keyframe 0 and every new keyframe look at a wall 2 m ahead (depth 2 m): placed
    where keyframe 0 is, a new keyframe's matched points meet keyframe 0's; placed 5 m aside,
    they are 5 m off. Keyframe 16 is registered by SfM, the others are judged as placed by
    their own map or multi-view."""
    root = tmp_path / "map"
    (root / "per_frame" / "f000000").mkdir(parents=True)
    np.save(root / "per_frame" / "f000000" / "depth.npy", np.full((48, 64), 2.0, np.float16))
    old = [SimpleNamespace(image=f"frames/{jpg(0)}", name="f000000", T_map_cam=Pose.identity(),
                           K=K, K_grid=K.resized(64, 48), width=640, height=480)]
    here, aside = Pose.identity(), Pose(np.eye(3), np.array([5.0, 0.0, 0.0]))
    poses = {jpg(10): here, jpg(11): here, jpg(12): aside, jpg(13): aside, jpg(14): aside,
             jpg(15): aside, jpg(16): here, jpg(17): here}
    uv = np.random.default_rng(4).uniform([40, 40], [600, 440], size=(40, 2))
    few = uv[:api.LINK_MIN_POINTS - 1]

    def link(a: int, b: int, pts: np.ndarray = uv) -> PairMatches:
        return PairMatches(jpg(a), jpg(b), pts, pts)

    matches = [
        link(0, 10),  # agrees: 10 is confirmed by the map
        link(10, 11),  # agrees: 11 through 10
        link(0, 12), link(12, 13),  # 12 disagrees with the map; 13 hangs on 12 alone
        link(10, 12),  # disagrees
        link(10, 14, few),  # judges nothing: 14 keeps its place
        link(0, 16), link(16, 17),  # 17 is confirmed by a keyframe SfM registered
        link(16, 13),  # disagrees
        link(16, 99),  # 99 was not placed: not judged against
    ]  # 15 has no link at all
    names_seen: list[set[str]] = []

    def verified_matches(db: Path, names: Any) -> list[PairMatches]:
        names_seen.append(set(names))
        return [p for p in matches if {p.a, p.b} <= set(names)]

    monkeypatch.setattr(panorama, "verified_matches", verified_matches)
    pairs = {frozenset((p.a, p.b)): (len(p.uv_a), 2) for p in matches}
    sfm = SimpleNamespace(db=Path("db"), verified_pairs=lambda: pairs)
    ctx = ctx_of([nf(i) for i in (10, 11, 12, 13, 14, 15, 16, 17, 99)], old,
                 tx=SimpleNamespace(current=lambda rel: root / rel))
    return SimpleNamespace(ctx=ctx, sfm=sfm, poses=poses, names_seen=names_seen,
                           free={jpg(i) for i in (10, 11, 12, 13, 14, 15, 17)})


def test_keyframes_their_links_place_elsewhere_or_hold_by_nothing_are_left_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _links_world(monkeypatch, tmp_path)
    K_ = {n: K for n in w.poses}
    off = api._depth_contradicted_placements(w.ctx, w.sfm, w.poses, w.free, K_)  # type: ignore[arg-type]
    assert off == {jpg(12), jpg(13), jpg(15)}
    assert w.ctx.notes["links_contradicted"] == [jpg(12), jpg(13), jpg(15)]
    assert jpg(99) not in w.names_seen[0]  # only placed keyframes are judged against
    # registered by SfM, in blocks: 12 and 13 placed elsewhere, 15 held by nothing (11 and 16
    # confirmed, 14 by a link that judges nothing)
    free = {jpg(10), jpg(16), jpg(17)}
    assert api._depth_contradicted_placements(w.ctx, w.sfm, w.poses, free,
                                              K_) == {jpg(12), jpg(13), jpg(15)}  # type: ignore[arg-type]
    # a block with a confirmed keyframe (16) keeps those its depth disagrees with (13, 12)
    assert api._depth_contradicted_placements(w.ctx, w.sfm, w.poses, {jpg(10), jpg(17)},
                                              K_) == {jpg(15)}  # type: ignore[arg-type]
    held = {n: T for n, T in w.poses.items() if n != jpg(15)}  # nothing left out
    assert api._depth_contradicted_placements(w.ctx, w.sfm, held, {jpg(10), jpg(17)},
                                              K_) == set()  # type: ignore[arg-type]
    # a video's few kept keyframes that every link to the rest disagrees with go with the rest
    assert api._depth_contradicted_placements(w.ctx, w.sfm, w.poses, w.free, K_,  # type: ignore[arg-type]
                                              video=True) == {jpg(12), jpg(13), jpg(15)}
    few = {n: w.poses[n] for n in (jpg(10), jpg(12), jpg(13), jpg(15))}
    assert api._depth_contradicted_placements(w.ctx, w.sfm, few, set(few), K_,  # type: ignore[arg-type]
                                              video=True) == set(few)
    assert api._depth_contradicted_placements(w.ctx, w.sfm, few, set(few),  # type: ignore[arg-type]
                                              K_) == {jpg(12), jpg(13), jpg(15)}
    # ... unless a link to the rest agrees, or none joins them
    lone = {n: w.poses[n] for n in (jpg(10), jpg(15))}
    assert api._depth_contradicted_placements(w.ctx, w.sfm, lone, set(lone), K_,  # type: ignore[arg-type]
                                              video=True) == {jpg(15)}
    assert api._depth_contradicted_placements(w.ctx, w.sfm, w.poses, set(), K_) == set()  # type: ignore[arg-type]
