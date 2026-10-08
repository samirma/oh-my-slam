"""Mapping building blocks at their edges: the map frame and metric scale (``frame``), retrieval
pairs, trajectory helpers, latest wins (``validity``) and the on-disk map (``store``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import retrieval, store, validity
from oh_my_slam.mapping import trajectory as traj
from tests.synth.scene import look_at

GRID = Intrinsics(30.0, 30.0, 20.0, 15.0, 40, 30)


# --- frame: metric scale, gravity, map frame ----------------------------------------------------


class _Model:
    """SfM model of keyframes at the origin whose observations lie at z = 1 in front of them."""

    def __init__(self, registered: list[str], points: dict[str, int]) -> None:
        self.registered = registered
        self.points = points

    def pose(self, name: str) -> Pose:
        return Pose.identity()

    def observations(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        n = self.points.get(name, 0)
        rng = np.random.default_rng(len(name))
        uv = rng.uniform([2, 2], [38, 28], (n, 2))
        xyz = np.column_stack([(uv - [20.0, 15.0]) / 30.0, np.ones(n)])
        return uv, xyz


def _depth(name: str, value: float) -> mframe.FrameDepth:
    return mframe.FrameDepth(name, np.full((30, 40), value, np.float32), GRID, (40, 30))


def test_metric_scale_is_the_median_ratio_and_warns_about_its_spread(
        monkeypatch: pytest.MonkeyPatch) -> None:
    warned: list[str] = []
    monkeypatch.setattr(mframe.log, "warning", lambda msg, *a: warned.append(msg % a))
    model = _Model(["a", "b", "c", "empty"], {"a": 200, "b": 200, "c": 200})
    frames = [_depth("a", 1.0), _depth("b", 2.0), _depth("c", 3.0), _depth("empty", 2.0),
              _depth("unregistered", 5.0)]
    res = mframe.metric_scale(model, frames)
    assert res.scale == pytest.approx(2.0)
    assert res.per_frame == pytest.approx({"a": 1.0, "b": 2.0, "c": 3.0})
    assert set(res.points) == {"a", "b", "c"}  # no observations, not registered: not measured
    assert res.spread == pytest.approx(0.5)
    assert warned and "varies across frames" in warned[0]


def test_metric_scale_needs_a_frame_with_enough_points() -> None:
    model = _Model(["a", "b"], {"a": 0, "b": 10})  # b: too few points for a ratio
    with pytest.raises(ValueError, match="enough triangulated points"):
        mframe.metric_scale(model, [_depth("a", 1.0), _depth("b", 1.0)])


def test_world_up_uses_posed_frames_with_gravity_and_defaults_to_z() -> None:
    no_up = _depth("a", 1.0)
    unposed = mframe.FrameDepth("b", no_up.depth, GRID, (40, 30), np.array([0.0, -1.0, 0.0]))
    assert mframe.world_up({"a": Pose.identity()}, [no_up, unposed]).tolist() == [0.0, 0.0, 1.0]
    posed = mframe.FrameDepth("c", no_up.depth, GRID, (40, 30), np.array([0.0, -1.0, 0.0]))
    up = mframe.world_up({"a": Pose.identity(), "c": Pose.identity()}, [no_up, unposed, posed])
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-9)


def test_map_frame_of_a_camera_looking_straight_down_takes_its_x_axis() -> None:
    down = look_at(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]), up=np.array([0, 1.0, 0]))
    sim = mframe.map_transform(down, np.array([0.0, 0.0, 1.0]), 1.0)
    first = mframe.transform_pose(sim, down)
    np.testing.assert_allclose(first.t, 0.0, atol=1e-9)
    np.testing.assert_allclose(first.R[:, 0], [1.0, 0.0, 0.0], atol=1e-9)  # camera x -> map x
    np.testing.assert_allclose(first.R[:, 2], [0.0, 0.0, -1.0], atol=1e-9)  # still looking down


# --- retrieval ----------------------------------------------------------------------------------


def test_top_k_pairs_never_pair_an_image_with_itself() -> None:
    desc = np.eye(4)[[0, 0, 1, 1]] + 0.01 * np.arange(4)[:, None]
    ids = [10, 11, 12, 13]
    # min_gap -1 excludes nothing by capture order: the image itself still never pairs
    pairs = retrieval.top_k_pairs(desc, desc, 1, ids, ids, min_gap=-1)
    assert pairs == {(10, 11), (12, 13)}
    assert all(a != b for a, b in pairs)


# --- trajectory ---------------------------------------------------------------------------------


def _at(x: float, y: float = 0.0, yaw: float = 0.0) -> Pose:
    return Pose(rot_z(yaw), np.array([x, y, 0.0]))


def test_summary_of_a_trajectory_without_steps() -> None:
    assert traj.summary({"a": _at(0)}, ["a", "b"]) == {"keyframes": 1}


def test_a_keyframe_without_ratio_or_links_is_a_block_of_its_own() -> None:
    covis = {frozenset(("a", "b")): 40, frozenset(("c", "d")): 40}
    blocks = traj.scale_blocks({"a": 1.0, "b": 1.02}, covis, ["a", "b", "c", "d", "e"])
    # c and d share points only with each other, e with nothing: no ratio and no vote to join
    assert blocks == [{"a", "b"}, {"c"}, {"d"}, {"e"}]


def test_tilted_keyframes_need_a_reference_gravity() -> None:
    poses = {"a": _at(0), "b": _at(1)}
    ups = {"b": np.array([0.0, -1.0, 0.0])}
    assert traj.tilted_keyframes(poses, ups, {"a"}, {"b"}) == set()  # a has no gravity estimate


def test_floating_runs_need_fixed_keyframes_that_move() -> None:
    poses = {"a": _at(0), "b": _at(0), "c": _at(9)}
    assert traj.floating_runs(poses, ["a", "b", "c"], {"c"}) == []


def test_attach_run_with_one_neighbour_or_none() -> None:
    poses = {"a": _at(0), "r1": _at(5, yaw=0.3), "r2": _at(6, yaw=0.3), "z": _at(10)}
    up = np.array([0.0, 0.0, 1.0])
    after = traj.attach_run(poses, ["r1", "r2"], None, "z", up)
    np.testing.assert_allclose(after["r2"].t, poses["z"].t)  # its end at the neighbour's centre
    np.testing.assert_allclose(after["r1"].t - after["r2"].t, poses["r1"].t - poses["r2"].t)
    np.testing.assert_allclose(after["r1"].R, poses["r1"].R)  # shape, scale and tilt kept
    assert traj.attach_run(poses, ["r1", "r2"], None, None, up) == {}


def test_attach_run_of_one_keyframe_is_centred_without_turning() -> None:
    poses = {"a": _at(0), "r": _at(5, 3, yaw=0.3), "z": _at(2)}
    moved = traj.attach_run(poses, ["r"], "a", "z", np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(moved["r"].t, [1.0, 0.0, 0.0])  # between its neighbours
    np.testing.assert_allclose(moved["r"].R, poses["r"].R)  # no direction to turn it by


# --- validity -----------------------------------------------------------------------------------


K_VIEW = Intrinsics(80.0, 80.0, 40.0, 30.0, 80, 60)


def _wall_depth(z: float, patch: float | None = None) -> np.ndarray:
    """A wall ``z`` m in front of the camera, with an object ``patch`` m away in the middle."""
    depth = np.full((60, 80), z, np.float32)
    if patch is not None:
        depth[18:42, 24:56] = patch
    return depth


def _wall(z: float, valid: bool = True, patch: float | None = None) -> validity.View:
    depth = _wall_depth(z, patch)
    return validity.View(depth, np.full(depth.shape, valid), K_VIEW, Pose.identity())


def test_views_without_usable_pixels_contradict_nothing() -> None:
    old = _wall(3.0)
    blind = _wall(5.0, valid=False)
    assert not validity.contradicted_cells(old, [blind]).any()
    assert not validity.contradicted_cells(blind, [_wall(5.0)]).any()
    # the object in front of the wall is gone: where it stood is free space now
    cells = validity.contradicted_cells(_wall(3.0, patch=2.0), [_wall(3.0), _wall(3.0)])
    px = validity.cells_to_pixels(cells, (60, 80))
    assert px[22:38, 28:52].all() and not px[:10].any()


K_DOOR = Intrinsics(160.0, 160.0, 80.0, 60.0, 160, 120)


def _door(far: float | None, slope: float = 0.0, patch: bool = False) -> validity.View:
    """A wall 2 m in front of the camera with a door on the right, through which a corridor is
    seen ``far`` m away (its depth ``slope`` times steeper per row from the middle), and an
    object 1.5 m away on the left (``patch``)."""
    depth = np.full((120, 160), 2.0, np.float32)
    if far is not None:
        rows = np.arange(20, 100, dtype=np.float32)[:, None]
        depth[20:100, 100:140] = far * (1.0 + slope * (rows - 60.0) / 40.0)
    if patch:
        depth[40:80, 20:60] = 1.5
    return validity.View(depth, np.ones(depth.shape, bool), K_DOOR, Pose.identity())


def test_the_far_field_of_a_static_scene_is_not_a_change(monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    """Later keyframes that see the corridor through the door at a third of its depth
    (monocular depth of the far field) do not invalidate it: on the old keyframe's far cells the
    margin is at least the disagreement the pair shows there. Where the object stood, which
    they see empty, is still free space now."""
    old = _door(8.0, patch=True)
    later = [_door(8.0 / 3, slope=0.1), _door(8.0 / 3, slope=-0.1)]
    px = validity.cells_to_pixels(validity.contradicted_cells(old, later), (120, 160))
    assert px[44:76, 24:56].all()  # the object's place
    assert not px[24:96, 104:136].any()  # the corridor
    # with no far field, both later keyframes contradict the corridor
    monkeypatch.setattr(validity, "FAR_REL", 100.0)
    px = validity.cells_to_pixels(validity.contradicted_cells(old, later), (120, 160))
    assert px[44:76, 24:56].all() and px[24:96, 104:136].all()
    assert validity.far_tolerance(np.ones(validity.FAR_MIN_SAMPLES - 1)) == 0.0  # too few
    assert validity.far_tolerance(np.full(validity.FAR_MIN_SAMPLES, -1.0)) == 1.0


def test_a_keyframe_without_usable_depth_has_no_far_field() -> None:
    blind = _door(8.0)
    blind.valid[:] = False
    assert not validity.contradicted_cells(blind, [_door(8.0 / 3)]).any()


class _Tx:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.written: dict[str, bytes] = {}

    def current(self, rel: str) -> Path:
        return self.root / rel

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.written[rel] = data


def _stored(root: Path, name: str, depth: np.ndarray | None) -> store.FrameRecord:
    rec = store.FrameRecord(0, name, f"frames/{name}.jpg", "", 1, 80, 60, K_VIEW, Pose.identity(),
                            80, 60)
    if depth is not None:
        (root / "per_frame" / name).mkdir(parents=True)
        np.save(root / store.frame_file(name, "depth.npy"), depth.astype(np.float16))
    return rec


def test_latest_wins_skips_stored_keyframes_without_depth(tmp_path: Path) -> None:
    from types import SimpleNamespace

    new = SimpleNamespace(record=SimpleNamespace(stats={}, pose_source="multiview",
                                                 K_grid=K_VIEW, T_map_cam=Pose.identity()),
                          depth=np.full((60, 80), 5.0, np.float32),
                          frame=SimpleNamespace(valid=np.ones((60, 80), bool)))
    tx = _Tx(tmp_path)
    # f000001 saw an object in front of the wall that the new keyframes see through
    old = [_stored(tmp_path, "f000000", None), _stored(tmp_path, "f000001", _wall_depth(5.0, 3.0)),
           _stored(tmp_path, "f000002", _wall_depth(5.0))]
    ctx = SimpleNamespace(new=[new, new], old_frames=old, tx=tx, notes={})
    msgs: list[str] = []
    validity.apply_latest_wins(ctx, [], msgs.append)
    assert set(tx.written) == {store.frame_file("f000001", "valid.png")}  # no depth: no view
    assert ctx.notes["latest_wins"]["frames_changed"] == 1
    assert msgs and "1 older keyframes" in msgs[0]
    assert validity.stored_view(tx.current, old[0]) is None


# --- store --------------------------------------------------------------------------------------


def _map(root: Path) -> None:
    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": [_stored(root, "f000000", None).to_dict()]})
        tx.commit({"update_count": 1})


def test_reader_without_instances_or_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _map(root)
    reader = store.MapReader(root)
    fr = reader.frames[0]
    assert reader.instances(fr) == []
    assert reader.descriptor(fr) is None
    (root / "per_frame" / fr.name).mkdir(parents=True)
    np.save(root / "per_frame" / fr.name / "descriptor.npy", np.arange(4, dtype=np.float32))
    assert reader.descriptor(fr).tolist() == [0.0, 1.0, 2.0, 3.0]  # type: ignore[union-attr]


def test_leaving_a_transaction_twice_releases_the_lock_once(tmp_path: Path) -> None:
    root = tmp_path / "m"
    tx = store.MapTransaction(root).__enter__()
    tx.__exit__(None, None, None)
    tx.__exit__(None, None, None)
    with store.MapTransaction(root) as again:  # the lock is free
        assert again.created


def test_resume_discards_what_a_rebuild_staged_of_the_derived_files(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _map(root)
    with store.MapTransaction(root) as tx:
        tx.start_over(keep=(store.SFM_DB,))
        tx.write_bytes("frames/f000000.jpg", b"image")
        tx.write_bytes(store.SFM_DB, b"db")
        tx.write_bytes(store.SFM_DB + "-wal", b"wal")
        tx.write_bytes(store.SFM_MODEL + "/images.bin", b"model")
        tx.write_bytes("per_frame/f000000/depth.npy", b"depth")
        tx.write_bytes("objects/points_000001.npy", b"points")
        tx.resume()
        left = sorted(str(p.relative_to(tx.staging)) for p in tx.staging.rglob("*")
                      if p.is_file())
        assert left == ["frames/f000000.jpg"]  # the keyframe images stay staged
        assert tx.current(store.FRAMES_JSON) == root / store.FRAMES_JSON  # read again


def test_a_rebuild_replaces_the_map_layout_only(tmp_path: Path) -> None:
    """``start_over``: the files the map layout owns go unless staged again; an earlier ``-o``
    result and the user's own files in the folder stay."""
    root = tmp_path / "m"
    _map(root)
    (root / "r.json").write_text("{}")
    (root / "notes").mkdir()
    (root / "notes" / "todo.txt").write_text("x")
    (root / "objects").mkdir(exist_ok=True)
    (root / "objects" / "points_000001.npy").write_bytes(b"p")
    with store.MapTransaction(root) as tx:
        tx.start_over(keep=(store.SFM_DB,))
        tx.commit({"update_count": 2})
    assert (root / "r.json").exists() and (root / "notes" / "todo.txt").exists()
    assert not (root / "objects" / "points_000001.npy").exists()


def test_deleting_a_staged_file_drops_the_staged_copy(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _map(root)
    with store.MapTransaction(root) as tx:
        tx.write_bytes("extra.bin", b"x")
        tx.delete("extra.bin")
        assert not (tx.staging / "extra.bin").exists()
        tx.commit({"update_count": 2})
    assert not (root / "extra.bin").exists()


def test_roll_forward_skips_files_applied_before_an_interruption(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _map(root)
    tx = store.MapTransaction(root).__enter__()
    tx.write_bytes("a.bin", b"a")
    tx.write_bytes("b.bin", b"b")
    tx.write_json(store.MAP_JSON, {"update_count": 2})
    (tx.staging / store.COMMIT).write_text(json.dumps(
        {"files": ["a.bin", "b.bin", store.MAP_JSON], "delete": []}))
    tx.__exit__(None, None, None)
    (tx.staging / "a.bin").replace(root / "a.bin")  # applied before the interruption
    with store.MapTransaction(root):
        pass
    assert (root / "a.bin").read_bytes() == b"a" and (root / "b.bin").read_bytes() == b"b"
    assert json.loads((root / store.MAP_JSON).read_text())["update_count"] == 2


def test_a_map_without_a_cloud_reads_as_an_empty_one() -> None:
    from types import SimpleNamespace

    from oh_my_slam.mapping.export import map_cloud

    cloud = map_cloud(SimpleNamespace(exists=lambda rel: False))  # type: ignore[arg-type]
    assert cloud.xyz.shape == (0, 3) and cloud.label is not None and len(cloud.label) == 0
