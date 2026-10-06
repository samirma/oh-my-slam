"""``mapper.sh update`` decisions (``mapping.api``) taken apart from COLMAP and the inference
server: matching pairs, the SfM fallback chain, joining keyframes onto the main reconstruction,
the depth alignment, levelling, staging and the rebuild of a weak map. COLMAP's results are
stand-ins (``FakeSfm``, ``FakeModel``) so that every branch is driven deliberately."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from oh_my_slam.core.errors import RegistrationError
from oh_my_slam.core.geometry import rot_z, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, retrieval, store
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import trajectory as traj
from oh_my_slam.mapping.panorama import PairMatches, PoseFit
from oh_my_slam.reconstruction import depth as rdepth
from oh_my_slam.reconstruction.depth import ScaleFit
from oh_my_slam.reconstruction.gravity import GravityEstimate
from tests.synth.scene import look_at

K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap")
GRID = K.resized(64, 48)
UP = np.array([0.0, 0.0, 1.0])


def quiet(msg: str) -> None:
    pass


def nf(i: int, desc: np.ndarray | None = None, up_cam: np.ndarray | None = None,
       pose: Pose | None = None, exif: Intrinsics | None = None) -> SimpleNamespace:
    """A new keyframe ``f<i>`` (as ``NewFrame``): depth and validity on a 64 x 48 grid."""
    grav = None if up_cam is None else GravityEstimate(np.asarray(up_cam, float), "geocalib",
                                                       1.0, 1.0)
    frame = SimpleNamespace(descriptor=desc, intrinsics=K, K_grid=GRID, grid_size=(64, 48),
                            depth=np.full((48, 64), 2.0, np.float32),
                            valid=np.ones((48, 64), bool), gravity=grav,
                            rgb=np.zeros((48, 64, 3), np.uint8))
    kf = SimpleNamespace(index=i, name=f"f{i:06d}", path=Path(f"/x/f{i:06d}.jpg"), exif=exif,
                         source=f"in{i}.jpg")
    return SimpleNamespace(kf=kf, frame=frame, full_size=(640, 480), record=None, depth=None,
                           dets=[], pose=pose)


def jpg(i: int) -> str:
    return f"f{i:06d}.jpg"


def ctx_of(new: list[Any], old: list[Any] | None = None, tmp: Path | None = None,
           **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(new=new, old_frames=old or [], notes={}, pose_support={}, rejected=[],
                           rescaled={}, meta={}, update_id=1, rebuild=None, features=None,
                           work=tmp or Path("/nonexistent"), tx=kw.pop("tx", None), **kw)


class FakeModel:
    """An SfM model: named camera-to-world poses, tracks (for support) and a similarity."""

    def __init__(self, poses: dict[str, Pose], tracks: list[set[str]] | None = None,
                 method: str = "sfm-global") -> None:
        self.poses = dict(poses)
        self.tracks = [set(t) for t in (tracks if tracks is not None else [set(poses)] * 50)]
        self.method = method
        self.notes: dict[str, Any] = {}
        self.others: list[Any] = []
        self.written: list[Path] = []
        self.rec = SimpleNamespace(exists_camera=lambda cid: False, cameras={},
                                   deregister_frame=lambda fid: self.poses.pop(fid),
                                   find_image_with_name=lambda n: SimpleNamespace(frame_id=n))

    @property
    def registered(self) -> list[str]:
        return sorted(self.poses)

    def pose(self, name: str) -> Pose:
        return self.poses[name]

    def intrinsics(self, name: str) -> Intrinsics:
        return K

    def camera_id(self, name: str) -> int:
        return 1

    def image_stats(self, name: str) -> dict[str, float]:
        return {"observations": 500.0, "reproj_error": 0.5}

    def supported(self, min_points: int) -> set[str]:
        return {n for n in self.poses if sum(n in t for t in self.tracks) >= min_points}

    def point_counts(self) -> dict[str, int]:
        return {n: sum(n in t for t in self.tracks) for n in self.poses}

    def covisibility(self) -> dict[frozenset[str], int]:
        out: dict[frozenset[str], int] = {}
        for t in self.tracks:
            for a in t:
                for b in t:
                    if a < b:
                        out[frozenset((a, b))] = out.get(frozenset((a, b)), 0) + 1
        return out

    def deregister(self, names: set[str]) -> None:
        for n in names:
            self.poses.pop(n, None)
        self.tracks = [t - names for t in self.tracks if len(t - names) >= 2]

    def transform(self, s: float, R: np.ndarray, t: np.ndarray) -> None:
        self.poses = {n: Pose(R @ T.R, s * R @ T.t + t) for n, T in self.poses.items()}

    def write(self, path: Path) -> None:
        self.written.append(path)

    def baseline_ratio(self) -> float:
        return 0.5


def line(names: list[str], step: float = 1.0) -> dict[str, Pose]:
    """Keyframes along x, ``step`` apart, looking along +y."""
    return {n: look_at(np.array([step * k, 0.0, 1.5]), np.array([step * k, 3.0, 1.5]))
            for k, n in enumerate(names)}


# --- pairs and anchors --------------------------------------------------------------------------


def test_pairs_of_a_new_video_without_descriptors_are_sequential() -> None:
    new = [nf(i) for i in range(30)]
    ids = list(range(30))
    assert api._pairs_new_map(new, is_video=True) == retrieval.sequential_pairs(ids,
                                                                                api.SEQ_OVERLAP)


def test_many_photos_are_paired_by_retrieval() -> None:
    rng = np.random.default_rng(0)
    n = api.PHOTO_EXHAUSTIVE_MAX + 1
    desc = rng.normal(size=(n, 8))
    pairs = api._pairs_new_map([nf(i, desc[i]) for i in range(n)], is_video=False)
    ids = list(range(n))
    assert pairs == retrieval.sequential_pairs(ids, api.SEQ_OVERLAP) | retrieval.top_k_pairs(
        desc, desc, api.RETRIEVAL_TOP_K, ids, ids, 0)
    assert len(pairs) < n * (n - 1) // 2


def _old(i: int, tmp: Path, desc: np.ndarray | None) -> store.FrameRecord:
    rec = store.FrameRecord(i, f"f{i:06d}", f"frames/{jpg(i)}", "", 1, 640, 480, K,
                            Pose.identity(), 64, 48)
    if desc is not None:
        (tmp / "per_frame" / rec.name).mkdir(parents=True, exist_ok=True)
        np.save(tmp / store.frame_file(rec.name, "descriptor.npy"), desc.astype(np.float32))
    return rec


def test_an_update_of_a_large_map_is_paired_by_retrieval(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_old = api.UPDATE_EXHAUSTIVE_MAX + 1
    old_desc = rng.normal(size=(n_old, 8))
    old = [_old(i, tmp_path, old_desc[i] if i % 2 else None) for i in range(n_old)]
    tx = SimpleNamespace(current=lambda rel: tmp_path / rel)
    new_desc = rng.normal(size=(3, 8))
    new = [nf(1000 + k, new_desc[k]) for k in range(3)]
    new_ids = [1000, 1001, 1002]
    held = [i for i in range(n_old) if i % 2]  # keyframes stored without a descriptor: not paired
    pairs = api._pairs_update(ctx_of(new, old, tx=tx), is_video=False)
    assert pairs == retrieval.all_pairs(new_ids) | retrieval.top_k_pairs(
        new_desc, old_desc.astype(np.float32)[held], api.RETRIEVAL_TOP_K, new_ids, held, 0)
    # a new keyframe without a descriptor: every stored keyframe
    new[1].frame.descriptor = None
    assert api._pairs_update(ctx_of(new, old, tx=tx), is_video=True) == \
        retrieval.sequential_pairs(new_ids, api.SEQ_OVERLAP) | retrieval.all_pairs(
            new_ids, list(range(n_old)))
    # no stored descriptor at all: the new keyframes among themselves only
    bare = [_old(i, tmp_path / "bare", None) for i in range(n_old)]
    new[1].frame.descriptor = new_desc[1]
    assert api._pairs_update(ctx_of(new, bare, tx=SimpleNamespace(
        current=lambda rel: tmp_path / "bare" / rel)), is_video=False) == retrieval.all_pairs(
        new_ids)


def test_anchors_without_descriptors_are_the_latest_posed_views() -> None:
    posed = [api.PoolView(f"p{k}", Path("x"), K, None, Pose.identity()) for k in range(6)]
    chunk = [api.PoolView("c", Path("y"), K, np.ones(4, np.float32))]
    assert [p.name for p in api._pick_anchors(chunk, posed, 4)] == ["p2", "p3", "p4", "p5"]


def test_a_keyframe_without_triangulated_points_has_no_sparse_scale() -> None:
    model = SimpleNamespace(observations=lambda n: (np.zeros((0, 2)), np.zeros((0, 3))))
    assert api._sparse_scale(nf(0), model, jpg(0)) is None  # type: ignore[arg-type]


def test_no_model_is_vetted_as_none() -> None:
    assert api._vetted(ctx_of([]), None, set()) is None  # type: ignore[arg-type]


def test_model_intrinsics_fall_back_to_the_database_camera() -> None:
    other = Intrinsics(400.0, 400.0, 320.0, 240.0, 640, 480, "colmap")
    sfm = SimpleNamespace(image_intrinsics=lambda names: {n: (3, other) for n in names})
    model = FakeModel({})
    assert api._model_intrinsics(sfm, model, {"a.jpg"}) == {"a.jpg": other}  # type: ignore[arg-type]


# --- the SfM fallback chain of a new map --------------------------------------------------------


class FakeSfm:
    """COLMAP stand-in: the reconstructions the test hands it, every other call recorded."""

    global_model: FakeModel | None = None
    incremental_model: FakeModel | None = None
    component: ClassVar[set[str]] = set()

    def __init__(self, db: Path, image_dir: Path, work: Path) -> None:
        self.db = db
        self.calls: list[str] = []

    def extract(self, names: list[str], prior: Any, video: bool = False) -> int:
        return 1

    def match_pairs(self, pairs: set[tuple[int, int]], names: dict[int, str]) -> int:
        return len(pairs)

    def two_view_stats(self, names: set[str] | None = None) -> dict[str, float]:
        return {"verified_pairs": 10.0, "rotation_fraction": 0.0, "planar_fraction": 0.0}

    def largest_component(self, names: set[str]) -> set[str]:
        return set(self.component)

    def rotation_pairs(self) -> set[frozenset[str]]:
        return set()

    def map_global(self, out: Path) -> FakeModel | None:
        return self.global_model

    def map_incremental(self, out: Path, **kw: Any) -> FakeModel | None:
        return self.incremental_model

    def triangulate_with_poses(self, poses: dict[str, Pose], out: Path,
                               focal_scale: float = 1.0) -> FakeModel:
        m = FakeModel(poses, method="multiview")
        m.notes["focal_scale"] = focal_scale
        return m


@pytest.fixture
def sfm_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    monkeypatch.setattr(api, "check_versions", lambda: "4.2.1")
    monkeypatch.setattr(api, "Sfm", FakeSfm)
    seen: dict[str, Any] = {}

    def multiview(ctx: Any, todo: list[Any], pool: list[Any], client: Any,
                  temporal: bool = False) -> dict[str, Pose]:
        seen["todo"] = [v.name for v in todo]
        return {v.name: Pose.identity() for v in todo}

    def refine(ctx: Any, sfm: Any, poses: dict[str, Pose], free: set[str], refine_focal: bool,
               rotation: bool, model: Any = None) -> tuple[dict[str, Pose], float]:
        seen["free"], seen["rotation"] = free, rotation
        return poses, 1.02

    monkeypatch.setattr(api, "_multiview_poses", multiview)
    monkeypatch.setattr(api, "_refine_multiview", refine)
    tx = SimpleNamespace(clone_for_edit=lambda rel: tmp_path / rel,
                         stage=lambda rel: tmp_path / "staging" / rel)
    ctx = ctx_of([nf(i, np.ones(8)) for i in range(4)], tmp=tmp_path, tx=tx)
    return SimpleNamespace(ctx=ctx, seen=seen)


def test_too_few_keyframes_placed_by_global_and_incremental_fall_back_to_multiview(
        sfm_chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeSfm, "global_model", None)
    monkeypatch.setattr(FakeSfm, "incremental_model", FakeModel(line([jpg(0)])))  # 1 of 4
    monkeypatch.setattr(FakeSfm, "component", {jpg(2)})  # one keyframe: not a component
    msgs: list[str] = []
    model = api._run_sfm(sfm_chain.ctx, False, None, msgs.append)
    assert model.method == "multiview"
    assert model.notes == {"focal_scale": 1.02, "reason": "SfM placed too few frames",
                           "metric": True}
    assert sfm_chain.seen["todo"] == [jpg(i) for i in range(4)]  # every keyframe
    assert sfm_chain.seen["free"] == {jpg(1), jpg(2), jpg(3)}  # the first holds the gauge
    assert any("multi-view fallback (SfM placed too few frames); 4 connected" in m for m in msgs)


def test_sfm_points_without_metric_scale_fall_back_to_multiview(
        sfm_chain: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeSfm, "global_model", FakeModel(line([jpg(i) for i in range(4)])))
    monkeypatch.setattr(FakeSfm, "component", {jpg(i) for i in range(4)})

    def no_scale(model: Any, frames: Any) -> Any:
        raise ValueError("no frame had enough triangulated points")

    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    model = api._run_sfm(sfm_chain.ctx, False, None, quiet)
    assert model.notes["reason"] == "no metric scale from the SfM points"
    assert sfm_chain.ctx.notes["baseline_ratio"] == 0.5
    assert sfm_chain.seen["rotation"] is True  # refined as rotation-dominant input


# --- multi-view refinement ----------------------------------------------------------------------


def test_refinement_takes_part_only_keyframes_with_intrinsics(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import panorama

    pairs = [PairMatches(jpg(0), jpg(1), np.zeros((20, 2)), np.zeros((20, 2))),
             PairMatches(jpg(1), "stray.jpg", np.zeros((20, 2)), np.zeros((20, 2)))]
    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: pairs)
    seen: dict[str, Any] = {}

    def refine(p: Any, views: dict[str, Any], free: Any, refine_focal: bool = False) -> PoseFit:
        seen["views"] = sorted(views)
        return PoseFit({jpg(1): Pose.identity()}, 1.0, 2, 40, 1.0, 0.1, {jpg(1): 0.1},
                       {jpg(1): 40})

    monkeypatch.setattr(panorama, "refine_poses", refine)
    sfm = SimpleNamespace(db=Path("db"), image_intrinsics=lambda names: {
        n: (1, K) for n in names if n != "stray.jpg"})
    ctx = ctx_of([nf(0), nf(1)])
    poses = {jpg(0): Pose.identity(), jpg(1): Pose.identity()}
    out, focal = api._refine_multiview(ctx, sfm, poses, {jpg(1)}, False, False)  # type: ignore[arg-type]
    assert seen["views"] == [jpg(0), jpg(1)]  # stray.jpg: no intrinsics, not stored
    assert focal == 1.0 and ctx.pose_support == {jpg(1): (0.1, 40)}
    assert ctx.notes["pose_refinement"]["pairs"] == 2


# --- joining keyframes onto the main reconstruction ---------------------------------------------


class JoinSfm:
    def __init__(self, connected: set[str], more: FakeModel | None = None) -> None:
        self.db = Path("db")
        self._connected = connected
        self.more = more
        self.extended: dict[str, Any] = {}

    def image_intrinsics(self, names: set[str]) -> dict[str, tuple[int, Intrinsics]]:
        return {n: (1, K) for n in names}

    def connected(self, seeds: set[str], candidates: set[str]) -> set[str]:
        return candidates & self._connected

    def model_cameras(self, path: Path) -> set[int]:
        return {1}

    def map_incremental(self, out: Path, **kw: Any) -> FakeModel | None:
        return self.more

    def extend_with_poses(self, base: Path, poses: dict[str, Pose], out: Path, method: str,
                          cameras: Any = None) -> FakeModel:
        self.extended = {"poses": poses, "method": method}
        return FakeModel(poses, method=method)


def test_joining_photos_without_metric_scale_leaves_the_unplaced_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A keyframe posed against its matches is registered again, one its depth contradicts and
    one left without support are joined like unplaced ones — but without a metric scale there is
    nothing to place them with: they are left out, and the update says so."""
    names = [jpg(i) for i in range(6)]
    poses = line(names)
    model = FakeModel(poses)
    model.notes["reconstructions"] = [6]
    more = FakeModel({jpg(5): poses[jpg(5)]}, tracks=[{jpg(5)}] * 30)
    sfm = JoinSfm(connected={jpg(3), jpg(5), jpg(6)}, more=more)
    calls: list[set[str]] = []

    def contradicting(s: Any, p: Any, k: Any, judged: set[str]) -> dict[str, float]:
        calls.append(set(judged))
        return {jpg(5): 0.5} if len(calls) == 1 else {}

    monkeypatch.setattr(api, "_fix_blocks", lambda c, s, m, j: {})
    monkeypatch.setattr(api, "_contradicting", contradicting)
    monkeypatch.setattr(api, "_deregister_contradicted",
                        lambda c, m, n, photos: ({jpg(4): 3.0}, {jpg(3)}))

    def no_scale(m: Any, f: Any) -> Any:
        raise ValueError("no scale")

    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    ctx = ctx_of([nf(i) for i in range(7)], tmp=tmp_path)
    msgs: list[str] = []
    out = api._join_unplaced(ctx, sfm, model, {jpg(i) for i in range(7)}, set(), False, None,  # type: ignore[arg-type]
                             msgs.append)
    join = ctx.notes["sfm_join"]
    assert join["contradicted"] == {jpg(5): 0.5} and join["reregistered"] == [jpg(5)]
    assert join["depth_contradicted"] == {jpg(4): 3.0}
    assert join["unsupported_after_join"] == [jpg(3)]
    assert join["no_connection"] == [jpg(3), jpg(4), jpg(6)]  # no scale to place them with
    assert join["multiview"] == [] and join["merged_by_shared_keyframes"] == []
    assert calls[1] == {jpg(5)}  # the re-registered pose is judged again
    assert sfm.extended["method"] == "sfm-global+reregistered"
    assert list(sfm.extended["poses"]) == [jpg(5)]
    assert out.method == "sfm-global+reregistered" and out.notes["reconstructions"] == [6]
    assert any("3 keyframes have no verified matches" in m for m in msgs)


def _other_reconstruction(poses: dict[str, Pose]) -> Any:
    """A pycolmap reconstruction of the keyframes at ``poses``, each with 25 triangulated
    points (enough support to be vetted)."""
    import pycolmap

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 500.0, 640, 480))
    ids = {}
    for k, (n, T) in enumerate(sorted(poses.items())):
        pts = [pycolmap.Point2D(np.array([10.0 + j, 20.0])) for j in range(25)]
        Tcw = T.inverse()
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=n, camera_id=1, image_id=k + 1, points2D=pts),
            pycolmap.Rigid3d(pycolmap.Rotation3d(Tcw.R), Tcw.t))
        ids[n] = k + 1
    for j in range(25):
        track = pycolmap.Track()
        for i in ids.values():
            track.add_element(i, j)
        rec.add_point3D(np.array([0.1 * j, 3.0, 1.5]), track, np.zeros(3, np.uint8))
    return rec


def test_joining_a_video_merges_another_reconstruction_and_guards_the_re_placed(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Another reconstruction sharing keyframes joins through the similarity of the shared
    poses; of the re-placed keyframes, one its matches contradict, one placed onto another
    keyframe's centre and one its gravity contradicts are left out."""
    names = [jpg(i) for i in range(6)]
    poses = line(names)
    model = FakeModel(poses)
    model.others = [_other_reconstruction({**{n: poses[n] for n in names[:4]},
                                           jpg(9): line(names + [jpg(9)])[jpg(9)]}),
                    # shares one keyframe only: nothing to join it by
                    _other_reconstruction({jpg(0): poses[jpg(0)],
                                           jpg(8): line(names + [jpg(9), jpg(8)])[jpg(8)]})]
    realigned = {jpg(2): poses[jpg(2)]}

    def fix_blocks(c: Any, s: Any, m: FakeModel, j: dict[str, Any]) -> dict[str, Pose]:
        m.deregister(set(realigned))
        return realigned

    monkeypatch.setattr(api, "_fix_blocks", fix_blocks)
    monkeypatch.setattr(api, "_contradicting", lambda s, p, k, judged: {})
    monkeypatch.setattr(api, "_deregister_contradicted", lambda c, m, n, photos: ({}, set()))
    monkeypatch.setattr(api.mframe, "metric_scale",
                        lambda m, f: mframe.ScaleResult(2.0, 0.0, {}, {}))
    turned = Pose(rot_z(np.pi / 2) @ poses[jpg(0)].R, poses[jpg(0)].t)  # f7 on f0's centre
    placed = {jpg(6): look_at(np.array([14.0, 0.0, 1.5]), np.array([14.0, 3.0, 1.5])),
              jpg(7): Pose(turned.R, 2.0 * turned.t)}

    def multiview(ctx: Any, todo: list[Any], pool: list[Any], client: Any,
                  temporal: bool = False) -> dict[str, Pose]:
        assert temporal and [v.name for v in todo] == [jpg(6), jpg(7)]
        return {v.name: placed[v.name] for v in todo}

    def refine(ctx: Any, sfm: Any, p: dict[str, Pose], free: set[str], refine_focal: bool,
               rotation: bool, model: Any = None) -> tuple[dict[str, Pose], float]:
        ctx.pose_support.update({jpg(6): (23.0, 300), jpg(7): (float("inf"), 0),
                                 jpg(2): (0.2, 300)})
        return p, 1.0

    monkeypatch.setattr(api, "_multiview_poses", multiview)
    monkeypatch.setattr(api, "_refine_multiview", refine)
    # gravity: up in each camera; f2's estimate is 40 degrees off its pose
    every = {**line(names + [jpg(9)]), **placed}
    new = []
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 9):
        T = every[jpg(i)]
        up_cam = T.R.T @ UP
        if i == 2:
            up_cam = T.R.T @ (rotation_between(UP, [np.sin(0.7), 0.0, np.cos(0.7)]) @ UP)
        new.append(nf(i, up_cam=up_cam))
    ctx = ctx_of(new, tmp=tmp_path)
    sfm = JoinSfm(connected={jpg(6), jpg(7)})
    new_names = {jpg(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 9)}
    out = api._join_unplaced(ctx, sfm, model, new_names, set(), True, None, quiet)  # type: ignore[arg-type]
    join = ctx.notes["sfm_join"]
    assert join["merged_by_shared_keyframes"] == [jpg(9)]
    assert join["residual_rejected"] == [jpg(6)]
    assert join["collapsed_rejected"] == [jpg(7)]
    assert join["gravity_rejected"] == [jpg(2)]
    assert join["multiview"] == [] and join["no_connection"] == []
    assert model.notes["prescale"] == 2.0
    np.testing.assert_allclose(sfm.extended["poses"][jpg(9)].t, 2.0 * np.array([6.0, 0, 1.5]),
                               atol=1e-9)  # merged, then brought to metres
    assert out.method == "sfm-global+merged"


def test_depth_judgement_needs_a_metric_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_scale(m: Any, f: Any) -> Any:
        raise ValueError("no scale")

    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    ctx = ctx_of([nf(0)])
    assert api._depth_contradicted(ctx, FakeModel(line([jpg(0)])), {jpg(0)}, True) == {}  # type: ignore[arg-type]


GOOD = ScaleFit(1.0, 200, 0.05)


def _depth_world(monkeypatch: pytest.MonkeyPatch, sparse: dict[str, ScaleFit],
                 dense: ScaleFit) -> tuple[Any, FakeModel]:
    names = sorted(sparse)
    ctx = ctx_of([nf(int(n[1:7])) for n in names])
    model = FakeModel(line(names, 0.3))
    monkeypatch.setattr(api, "_frame_depths", lambda c: [])
    monkeypatch.setattr(api.mframe, "metric_scale",
                        lambda m, f: mframe.ScaleResult(1.0, 0.0, {}, {}))
    monkeypatch.setattr(api, "_sparse_scale", lambda n, m, name: sparse[name])
    monkeypatch.setattr(rdepth, "dense_scale", lambda *a, **k: dense)
    return ctx, model


def test_photos_outside_the_confident_depth_range_and_poses_the_dense_test_keeps(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = {jpg(0): GOOD, jpg(1): GOOD, jpg(2): ScaleFit(1.5, 200, 0.05),
              jpg(3): ScaleFit(3.0, 200, 0.05)}
    ctx, model = _depth_world(monkeypatch, sparse, dense=ScaleFit(1.05, 5000, 0.1))
    names = set(sparse) | {"stored.jpg"}  # a stored keyframe serves as a reference only
    # f2: within the rejection range but a photo outside the confident one; f3's pose the
    # dense test supports
    assert api._depth_contradicted(ctx, model, names, photos=True) == {jpg(2): 1.5}  # type: ignore[arg-type]
    assert api._depth_contradicted(ctx, model, names, photos=False) == {}  # type: ignore[arg-type]


def test_deregistering_a_contradicted_keyframe_that_changes_no_other_ends_the_search(
        monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(line([jpg(i) for i in range(4)]),
                      tracks=[{jpg(0), jpg(1)}] * 25 + [{jpg(2), jpg(3)}] * 40)
    calls: list[set[str]] = []

    def judged(ctx: Any, m: Any, names: set[str], photos: bool,
               judge: set[str] | None = None) -> dict[str, float]:
        calls.append(set(judge or names))
        return {jpg(0): 4.0}

    monkeypatch.setattr(api, "_depth_contradicted", judged)
    off, unsupported = api._deregister_contradicted(None, model, set(model.registered), True)  # type: ignore[arg-type]
    assert off == {jpg(0): 4.0} and unsupported == {jpg(1)}
    assert len(calls) == 1  # f1 went with f0, f2 and f3 lost nothing: nothing to judge again
    assert model.registered == [jpg(2), jpg(3)]


@pytest.mark.parametrize(("connected", "more"), [
    (set(), None),  # no verified match to the model's keyframes
    ({jpg(1)}, None),  # the incremental mapper gives nothing
    ({jpg(1)}, FakeModel({jpg(1): Pose.identity()}, tracks=[])),  # a pose without support
])
def test_registering_again_gives_nothing_without_a_supported_pose(
        connected: set[str], more: FakeModel | None, tmp_path: Path) -> None:
    model = FakeModel(line([jpg(0), jpg(2)]))
    sfm = JoinSfm(connected=connected, more=more)
    assert api._reregister(ctx_of([], tmp=tmp_path), sfm, model, {jpg(1)}) == {}  # type: ignore[arg-type]
    assert model.written == ([] if not connected else [tmp_path / "sfm_rereg_base"])


def test_blocks_hanging_on_nothing_are_deregistered(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(line([jpg(i) for i in range(4)]))

    def no_scale(m: Any, f: Any) -> Any:
        raise ValueError("no scale")

    seen: dict[str, Any] = {}

    def fix(poses: Any, ratios: Any, covis: Any, links: Any, ups: Any) -> traj.BlockFix:
        seen["ratios"] = ratios
        return traj.BlockFix({}, [], {jpg(3)})

    warned: list[str] = []
    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    monkeypatch.setattr(traj, "fix_blocks", fix)
    monkeypatch.setattr(api.log, "warning", lambda msg, *a: warned.append(msg % a))
    sfm = SimpleNamespace(verified_pairs=lambda: {})
    join: dict[str, Any] = {}
    assert api._fix_blocks(ctx_of([nf(i) for i in range(4)]), sfm, model, join) == {}  # type: ignore[arg-type]
    assert seen["ratios"] == {} and model.registered == [jpg(0), jpg(1), jpg(2)]
    assert join["fixed_blocks"] == [] and "1 such keyframes hang on nothing" in warned[0]


def test_realigned_blocks_are_levelled_with_their_gravity_again() -> None:
    poses = line([jpg(0), jpg(1), jpg(2)])
    tilt = rotation_between(UP, [np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))])
    free = {n: Pose(tilt @ poses[n].R, poses[n].t) for n in (jpg(1), jpg(2))}
    join = {"fixed_blocks": [{"keyframes": [jpg(1), jpg(2)], "about": jpg(1)}]}
    api._relevel_blocks(ctx_of([nf(i) for i in range(3)]), {jpg(0): poses[jpg(0)]}, free, join)
    assert "relevelled_deg" not in join["fixed_blocks"][0]  # no gravity: nothing to level by
    ups = [nf(i, up_cam=poses[jpg(i)].R.T @ UP) for i in range(3)]
    api._relevel_blocks(ctx_of(ups), {jpg(0): poses[jpg(0)]}, free, join)
    assert join["fixed_blocks"][0]["relevelled_deg"] == pytest.approx(10.0, abs=0.01)
    for n in (jpg(1), jpg(2)):
        np.testing.assert_allclose(free[n].R, poses[n].R, atol=1e-9)


def test_a_video_run_floating_away_is_moved_between_its_neighbours() -> None:
    names = [jpg(i) for i in range(6)]
    poses = line(names)
    fixed = {n: poses[n] for n in (jpg(0), jpg(1), jpg(4), jpg(5))}
    far = {n: Pose(poses[n].R, poses[n].t + [0.0, 50.0, 0.0]) for n in (jpg(2), jpg(3))}
    join: dict[str, Any] = {}
    free = dict(far)
    api._attach_floating(ctx_of([nf(i) for i in range(6)]), fixed, free, {}, join)
    assert free[jpg(2)] is far[jpg(2)] and join == {}  # no gravity: nothing to turn it about
    ups = [nf(i, up_cam=poses[jpg(i)].R.T @ UP) for i in range(6)]
    mv = {jpg(3): far[jpg(3)]}
    free = {jpg(2): far[jpg(2)]}
    api._attach_floating(ctx_of(ups), fixed, free, mv, join)
    assert join["attached_by_neighbours"] == [{"keyframes": [jpg(2), jpg(3)], "before": jpg(1),
                                               "after": jpg(4)}]
    np.testing.assert_allclose(free[jpg(2)].t, poses[jpg(2)].t, atol=1e-9)
    np.testing.assert_allclose(mv[jpg(3)].t, poses[jpg(3)].t, atol=1e-9)


# --- extending a map ----------------------------------------------------------------------------


def _ext_world(tmp_path: Path, n_old: int = 3) -> tuple[SimpleNamespace, list[Any]]:
    root = tmp_path / "map"
    (root / store.SFM_MODEL).mkdir(parents=True)
    (root / store.SFM_MODEL / "images.bin").write_bytes(b"")
    old = [store.FrameRecord(i, f"f{i:06d}", f"frames/{jpg(i)}", "", 1, 640, 480, K,
                             line([jpg(i)])[jpg(i)], 64, 48) for i in range(n_old)]
    return SimpleNamespace(root=root, current=lambda rel: root / rel), old


class ExtSfm(JoinSfm):
    def __init__(self, inc: FakeModel | None) -> None:
        super().__init__(connected={jpg(10), jpg(11)})
        self.inc = inc

    def map_incremental(self, out: Path, **kw: Any) -> FakeModel | None:
        return self.inc


def test_a_map_of_fewer_than_three_keyframes_is_mapped_again(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tx, old = _ext_world(tmp_path, n_old=2)
    sentinel = FakeModel({}, method="remapped")
    monkeypatch.setattr(api, "_remap_small", lambda *a: sentinel)
    ctx = ctx_of([nf(10)], old, tx=tx)
    assert api._extend(ctx, ExtSfm(None), {jpg(10)}, False, None, quiet, 0.0) is sentinel  # type: ignore[arg-type]


@pytest.fixture
def extension(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def multiview(ctx: Any, todo: list[Any], pool: list[Any], client: Any,
                  temporal: bool = False) -> dict[str, Pose]:
        seen["todo"] = [v.name for v in todo]
        seen["pool"] = sorted(v.name for v in pool if v.pose is not None)
        return {v.name: Pose.identity() for v in todo}

    monkeypatch.setattr(api, "_multiview_poses", multiview)
    monkeypatch.setattr(api, "_refine_multiview",
                        lambda ctx, sfm, p, free, refine_focal, rotation, model=None: (p, 1.0))
    monkeypatch.setattr(api, "_contradicting", lambda s, p, k, judged: {})
    return seen


def test_without_an_incremental_extension_the_new_keyframes_are_placed_by_multiview(
        extension: dict[str, Any], tmp_path: Path) -> None:
    tx, old = _ext_world(tmp_path)
    ctx = ctx_of([nf(10), nf(11)], old, tx=tx)
    sfm = ExtSfm(None)
    model = api._extend(ctx, sfm, {jpg(10), jpg(11)}, False, None, quiet, 0.0)  # type: ignore[arg-type]
    assert model.method == "multiview" and ctx.notes["mv_names"] == [jpg(10), jpg(11)]
    assert extension["pool"] == [jpg(i) for i in range(3)]  # anchored on the map's keyframes


def test_an_extension_moving_a_fixed_frame_or_contradicting_the_depth(
        extension: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tx, old = _ext_world(tmp_path)
    inc = FakeModel({**{jpg(i): r.T_map_cam for i, r in enumerate(old)},
                     jpg(10): Pose.identity(), jpg(11): Pose.identity()})
    inc.notes["moved_fixed"] = [jpg(1)]
    monkeypatch.setattr(api, "_depth_consistent", lambda c, m, name: name != jpg(11))
    ctx = ctx_of([nf(10), nf(11)], old, tx=tx)
    model = api._extend(ctx, ExtSfm(inc), {jpg(10), jpg(11)}, False, None, quiet, 0.0)  # type: ignore[arg-type]
    assert ctx.notes["sfm_moved_fixed"] == [jpg(1)]
    assert model.method == "sfm-incremental+multiview" and ctx.notes["mv_names"] == [jpg(11)]
    assert extension["pool"] == [jpg(0), jpg(1), jpg(2), jpg(10)]


def test_new_keyframes_posed_against_their_matches_are_noted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_contradicting", lambda s, p, k, judged: {jpg(10): 0.41234})
    inc = FakeModel({jpg(10): Pose.identity()})
    ctx = ctx_of([nf(10)])
    assert api._contradicted_new(ctx, None, inc, {jpg(10)}) == {jpg(10): 0.41234}  # type: ignore[arg-type]
    assert ctx.notes["sfm_contradicted"] == {jpg(10): 0.412}


# --- focal re-run, map frame, depth alignment, levelling ----------------------------------------


def test_keyframes_whose_sfm_focal_differs_are_reconstructed_again_with_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Only registered keyframes without EXIF whose SfM focal length differs by more than 3 %
    are re-run, with the SfM intrinsics; a client that is its own clone is shared, not
    closed."""
    sfm_K = Intrinsics(560.0, 560.0, 320.0, 240.0, 640, 480, "colmap")  # 12 % off
    model = FakeModel(line([jpg(0), jpg(1), jpg(2)]))
    monkeypatch.setattr(model, "intrinsics", lambda n: K if n == jpg(1) else sfm_K)
    seen: list[tuple[str, Intrinsics, bool]] = []

    def again(path: Path, client: Any, intrinsics: Intrinsics, work_dir: Any = None,
              first: bool = True, rgb: Any = None) -> SimpleNamespace:
        seen.append((path.name, intrinsics, first))
        return SimpleNamespace(depth=np.full((48, 64), 3.0, np.float32),
                               valid=np.ones((48, 64), bool), intrinsics=intrinsics,
                               K_grid=intrinsics.resized(64, 48))

    monkeypatch.setattr(api, "_reconstruct_keyframe", again)
    closed: list[int] = []
    client = SimpleNamespace(close=lambda: closed.append(1))
    client.clone = lambda: client
    new = [nf(0), nf(1), nf(2, exif=K), nf(3)]  # f1: within 3 %, f2: EXIF, f3: not registered
    msgs: list[str] = []
    api._rerun_focal(ctx_of(new), model, client, msgs.append)  # type: ignore[arg-type]
    assert seen == [("f000000.jpg", sfm_K, False)] and closed == []
    assert new[0].frame.intrinsics == sfm_K and float(new[0].frame.depth[0, 0]) == 3.0
    assert new[1].frame.intrinsics == K
    assert msgs == ["re-running geometry for 1 keyframes with the SfM focal length"]


def test_the_map_frame_without_metric_scale_keeps_the_sfm_units(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def no_scale(m: Any, f: Any) -> Any:
        raise ValueError("no scale")

    monkeypatch.setattr(api.mframe, "metric_scale", no_scale)
    poses = line([jpg(0), jpg(1)])
    model = FakeModel(poses)
    ctx = ctx_of([nf(0, up_cam=poses[jpg(0)].R.T @ UP), nf(1, up_cam=poses[jpg(1)].R.T @ UP)])
    api._define_map_frame(ctx, model, quiet)  # type: ignore[arg-type]
    assert ctx.meta["scale"] == {"sfm_to_metric": 1.0, "spread": 0.0, "frames": 0,
                                 "method": "moge_over_sfm_median"}
    np.testing.assert_allclose(model.pose(jpg(0)).t, 0.0, atol=1e-9)  # origin at the first
    np.testing.assert_allclose(model.pose(jpg(1)).t, [0.0, -1.0, 0.0], atol=1e-9)


def test_a_keyframe_neither_sparse_nor_dense_scale_supports_is_rejected(
        monkeypatch: pytest.MonkeyPatch) -> None:
    poses = line([jpg(1), jpg(2)], 0.2)
    model = FakeModel(poses)
    fits = {jpg(1): GOOD, jpg(2): ScaleFit(3.0, 200, 0.05)}
    monkeypatch.setattr(api, "_sparse_scale", lambda n, m, name: fits[name])
    monkeypatch.setattr(rdepth, "dense_scale", lambda *a, **k: ScaleFit(3.1, 5000, 0.05))
    ctx = ctx_of([nf(0), nf(1), nf(2)])  # f0 is not registered
    api._align_depths(ctx, model)  # type: ignore[arg-type]
    assert ctx.new[0].record is None
    assert ctx.new[1].record.stats["depth_scale_method"] == "sparse"
    assert ctx.rejected == ["f000002"] and ctx.new[2].record is None
    assert model.registered == [jpg(1)]  # deregistered


def test_depth_nodes_skip_unplaced_keyframes_and_one_holds_the_gauge() -> None:
    placed = nf(1)
    placed.record = store.FrameRecord(1, "f000001", "", "", 1, 640, 480, K, Pose.identity(),
                                      64, 48, stats={"depth_scale_method": "dense"})
    placed.depth = placed.frame.depth
    ctx = ctx_of([nf(0), placed])
    nodes = api._scale_nodes(ctx)  # type: ignore[arg-type]
    assert [n.name for n in nodes] == ["f000001"] and not nodes[0].free
    msgs: list[str] = []
    api._adjust_depth_scales(ctx, msgs.append)  # type: ignore[arg-type]
    assert msgs == [] and "depth_scale_adjustment" not in ctx.notes  # one keyframe: nothing


def test_levelling_needs_a_floor_with_enough_points_and_a_plane(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.core import geometry
    from oh_my_slam.mapping import objects

    rng = np.random.default_rng(0)
    model = FakeModel(line([jpg(0)]))
    before = model.pose(jpg(0))
    ctx = ctx_of([nf(0)])
    for floor in ((np.zeros((0, 3)), None), (rng.uniform(-1, 1, (100, 3)) * [1, 1, 0], 0.0)):
        monkeypatch.setattr(objects, "map_floor", lambda c, per_frame=None, f=floor: f)
        api._level_with_floor(ctx, model, quiet)  # type: ignore[arg-type]
    pts = rng.uniform(-2, 2, (2000, 3)) * [1, 1, 0]
    monkeypatch.setattr(objects, "map_floor", lambda c, per_frame=None: (pts, 0.0))
    monkeypatch.setattr(geometry, "ransac_plane", lambda *a, **k: None)
    api._level_with_floor(ctx, model, quiet)  # type: ignore[arg-type]
    assert "floor_levelling_deg" not in ctx.notes
    np.testing.assert_allclose(model.pose(jpg(0)).matrix(), before.matrix())


def test_levelling_rotates_the_placed_keyframes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from oh_my_slam.mapping import objects

    rng = np.random.default_rng(1)
    tilt = rotation_between(UP, [np.sin(np.radians(3)), 0.0, np.cos(np.radians(3))])
    pts = (rng.uniform(-2, 2, (3000, 3)) * [1, 1, 0]) @ tilt.T
    monkeypatch.setattr(objects, "map_floor", lambda c, per_frame=None: (pts, 0.0))
    placed = nf(1)
    placed.record = store.FrameRecord(1, "f000001", "", "", 1, 640, 480, K, Pose.identity(),
                                      64, 48)
    ctx = ctx_of([nf(0), placed])
    model = FakeModel(line([jpg(1)]))
    msgs: list[str] = []
    api._level_with_floor(ctx, model, msgs.append)  # type: ignore[arg-type]
    assert ctx.notes["floor_levelling_deg"] == pytest.approx(3.0, abs=0.2)
    assert ctx.new[0].record is None
    assert traj.rotation_deg(placed.record.T_map_cam.R, np.eye(3)) == pytest.approx(3.0, abs=0.2)
    assert msgs and "levelled with the floor plane" in msgs[0]


def test_staging_skips_unplaced_keyframes_and_drops_rejected_images(tmp_path: Path) -> None:
    staged: dict[str, Any] = {}
    tx = SimpleNamespace(staging=tmp_path, save_npy=lambda rel, a: staged.setdefault(rel, a),
                         write_bytes=lambda rel, b: staged.setdefault(rel, b))
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "f000002.jpg").write_bytes(b"jpg")
    placed = nf(1)
    placed.record, placed.depth = SimpleNamespace(), placed.frame.depth
    ctx = ctx_of([nf(0), placed], tx=tx)
    ctx.rejected = ["f000002"]
    api._stage_frames(ctx)  # type: ignore[arg-type]
    assert sorted(staged) == ["per_frame/f000001/depth.npy", "per_frame/f000001/valid.png"]
    assert not (tmp_path / "frames" / "f000002.jpg").exists()


# --- placing an update --------------------------------------------------------------------------


@pytest.mark.parametrize(("registered", "placed", "error"), [
    (set(), set(), "nothing registered"),
    ({jpg(0)}, set(), "no input frame could be placed"),
    ({jpg(0)}, {0}, None),
])
def test_placing_fails_without_a_registered_or_placed_keyframe(
        monkeypatch: pytest.MonkeyPatch, registered: set[str], placed: set[int],
        error: str | None) -> None:
    new = [nf(0), nf(1)]
    monkeypatch.setattr(api, "_run_sfm", lambda c, v, cl, p: FakeModel(
        {n: Pose.identity() for n in registered}))
    for name in ("_rerun_focal", "_define_map_frame", "_level_with_floor"):
        monkeypatch.setattr(api, name, lambda *a: None)
    monkeypatch.setattr(api, "_adjust_depth_scales", lambda c, p: None)

    def align(ctx: Any, model: Any) -> None:
        for i in placed:
            ctx.new[i].record = SimpleNamespace()

    monkeypatch.setattr(api, "_align_depths", align)
    msgs: list[str] = []
    if error is None:
        api._place(ctx_of(new), False, None, msgs.append)  # type: ignore[arg-type]
        assert msgs == ["left out 1 unplaceable keyframes: f000001"]
    else:
        with pytest.raises(RegistrationError, match=error):
            api._place(ctx_of(new), False, None, msgs.append)  # type: ignore[arg-type]


# --- rebuilding a weak map ----------------------------------------------------------------------


def _weak_map(root: Path) -> list[store.FrameRecord]:
    """A one-keyframe map whose detections carry ids the map merged (7 -> 3 -> 2)."""
    from oh_my_slam.core import rle
    from oh_my_slam.core.images import save_jpeg

    rec = store.FrameRecord(0, "f000000", "frames/f000000.jpg", "in.jpg", 1, 640, 480, K,
                            Pose.identity(), 64, 48, pose_source="identity")
    mask = np.zeros((48, 64), bool)
    mask[10:20, 10:20] = True
    insts = [
        {"object_id": 7, "first_id": 7, "label": "cup", "score": 0.9, "mask": rle.encode(mask)},
        {"object_id": 4, "label": "book", "score": 0.8, "mask": rle.encode(mask)},  # old map
        {"object_id": 0, "label": "lamp", "score": 0.7, "mask": rle.encode(mask)},  # removed
        {"object_id": 5, "label": "mug", "score": 0.7,
         "mask": rle.encode(np.zeros((48, 64), bool))},  # no pixel
        {"object_id": 6, "label": "pen", "score": 0.7,
         "mask": rle.encode(np.ones((8, 8), bool))},  # another grid
    ]
    with store.MapTransaction(root) as tx:
        save_jpeg(np.full((480, 640, 3), 90, np.uint8), tx.stage(rec.image))
        tx.save_npy(store.frame_file(rec.name, "depth.npy"), np.full((48, 64), 2.0, np.float16))
        tx.write_json(store.frame_file(rec.name, "instances.json"), {"instances": insts})
        tx.write_json(store.FRAMES_JSON, {"frames": [rec.to_dict()]})
        tx.write_json(store.OBJECTS_JSON, {"next_id": 8, "objects": [],
                                           "merged_into": {"7": 3, "3": 2}})
        tx.commit({"update_count": 1, "next_object_id": 8, "next_frame_index": 1,
                   "updates": [{"kind": "images", "notes": {}}], "scale": {"x": 1}})
    return [rec]


def test_stored_keyframes_bring_their_detections_with_their_first_ids(tmp_path: Path) -> None:
    root = tmp_path / "map"
    (rec,) = _weak_map(root)
    with store.MapTransaction(root) as tx:
        prior: dict[int, int] = {}
        first: dict[int, int] = {}
        frame = api._stored_frame(tx, rec, prior, first, lambda oid: {7: 2}.get(oid, oid))
        assert [d.label for d in frame.dets] == ["cup", "book", "lamp"]  # masks on its grid
        cup, book, lamp = frame.dets
        assert first == {id(cup): 7, id(book): 4} and prior == {id(cup): 2, id(book): 4}
        assert frame.frame.descriptor is None and frame.frame.gravity is None
        assert tx.stage(rec.image).exists()  # its image staged again


def test_an_abandoned_rebuild_extends_the_map_as_before(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    root = tmp_path / "map"
    plan = _weak_map(root)
    seen: dict[str, Any] = {}

    def fails(ctx: Any, is_video: bool, client: Any, progress: Any) -> Any:
        seen["prior"] = sorted(ctx.rebuild.prior.values())
        raise RegistrationError("none of the input frames overlaps the map")

    monkeypatch.setattr(api, "_place", fails)
    msgs: list[str] = []
    with store.MapTransaction(root) as tx:
        meta = store.read_meta_or_default(tx)
        saved = json.loads(json.dumps(meta))
        ctx, model = api._try_rebuild(tx, meta, plan, [], plan, 2, tmp_path / "work", False,
                                      None, msgs.append)
        assert model is None and ctx.old_frames == plan and ctx.rebuild is None
        assert ctx.notes["restart_abandoned"] == {
            "left_out": ["(none of the input frames overlaps the map)"]}
        assert meta == saved  # the map's metadata as it was (scale, next ids)
        assert tx.current(store.FRAMES_JSON) == root / store.FRAMES_JSON  # read again
    assert seen["prior"] == [2, 4]  # 7 resolved through the map's merges: 7 -> 3 -> 2
    assert any("rebuild abandoned" in m for m in msgs)
