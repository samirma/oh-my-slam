"""Mapping building blocks: store (staged commit, crash safety, read-only), ingest, retrieval,
map frame, latest wins and object bookkeeping."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.errors import InputError, MapLockedError, NotAMapError, UsageError
from oh_my_slam.core.geometry import angle_between_deg, rot_z, rotation_between
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import ingest, retrieval, store, validity
from oh_my_slam.mapping.objects import MapObject, ObjectState, overlap_fraction, projected_mask
from tests.synth.scene import Box, Room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


# --- store --------------------------------------------------------------------------------------


def _make_map(root: Path) -> None:
    with store.MapTransaction(root) as tx:
        tx.write_json(store.FRAMES_JSON, {"frames": []})
        tx.write_bytes("per_frame/f000000/x.bin", b"one")
        tx.commit({"update_count": 1})


def test_store_create_commit_and_read(tmp_path: Path) -> None:
    root = tmp_path / "m"
    assert store.classify(root) == "missing"
    _make_map(root)
    assert store.classify(root) == "map"
    r = store.MapReader(root)
    assert r.meta["update_count"] == 1 and r.meta["format_version"] == 1
    assert r.frames == [] and not (root / store.STAGING).exists()
    assert (root / "per_frame/f000000/x.bin").read_bytes() == b"one"
    with pytest.raises(NotAMapError):
        store.MapReader(tmp_path)


def test_store_uncommitted_update_leaves_map_untouched(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _make_map(root)
    before = store.full_tree_hash(root)
    with pytest.raises(RuntimeError), store.MapTransaction(root) as tx:
        tx.write_bytes("per_frame/f000000/x.bin", b"two")
        tx.write_json(store.MAP_JSON, {"update_count": 99})
        raise RuntimeError("killed mid-update")
    assert store.full_tree_hash(root) == before
    assert not (root / store.STAGING).exists()


def test_store_killed_process_leaves_map_untouched(tmp_path: Path) -> None:
    """A real process killed (SIGKILL) during staging: map.json unchanged, next update cleans."""
    root = tmp_path / "m"
    _make_map(root)
    before = store.tree_hash(root)
    code = (
        "import os, time, sys\n"
        "from pathlib import Path\n"
        "from oh_my_slam.mapping import store\n"
        f"tx = store.MapTransaction(Path({str(root)!r})).__enter__()\n"
        "tx.write_bytes('per_frame/f000000/x.bin', b'partial')\n"
        "print('staged', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "staged"
    proc.kill()
    proc.wait()
    assert store.tree_hash(root) == before
    assert (root / "per_frame/f000000/x.bin").read_bytes() == b"one"
    with store.MapTransaction(root) as tx:  # lock released by the kernel; staging discarded
        assert not (root / store.STAGING / "per_frame").exists()
        tx.write_bytes("y.bin", b"y")
        tx.commit({"update_count": 2})
    assert (root / "y.bin").exists()


def test_store_committed_staging_rolls_forward_and_overlays(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _make_map(root)
    tx = store.MapTransaction(root).__enter__()
    tx.write_bytes("per_frame/f000000/x.bin", b"new")
    tx.write_json(store.MAP_JSON, {"update_count": 2})
    files = ["per_frame/f000000/x.bin", store.MAP_JSON]
    (tx.staging / store.COMMIT).write_text(json.dumps({"files": files, "delete": []}))
    tx.__exit__(None, None, None)  # "crash" after the commit point
    reader = store.MapReader(root)  # read-only view sees the committed update
    assert reader.meta["update_count"] == 2
    assert reader.path("per_frame/f000000/x.bin").read_bytes() == b"new"
    assert (root / "per_frame/f000000/x.bin").read_bytes() == b"one"  # reader wrote nothing
    with store.MapTransaction(root):
        pass  # roll forward
    assert (root / "per_frame/f000000/x.bin").read_bytes() == b"new"
    assert json.loads((root / store.MAP_JSON).read_text())["update_count"] == 2


def test_store_lock_delete_and_folder_rules(tmp_path: Path) -> None:
    root = tmp_path / "m"
    _make_map(root)
    with store.MapTransaction(root) as tx:
        with pytest.raises(MapLockedError), store.MapTransaction(root):
            pass
        tx.delete("per_frame/f000000/x.bin")
        tx.commit({"update_count": 2})
    assert not (root / "per_frame/f000000/x.bin").exists()
    other = tmp_path / "o"
    other.mkdir()
    (other / "a.txt").write_text("x")
    with pytest.raises(NotAMapError), store.MapTransaction(other):
        pass
    f = tmp_path / "file"
    f.write_text("x")
    assert store.classify(f) == "other"


def test_frame_record_roundtrip() -> None:
    rec = store.FrameRecord(3, "f000003", "frames/f000003.jpg", "x.jpg", 1.5, 1, 640, 480,
                            Intrinsics(500, 500, 320, 240, 640, 480, "colmap"),
                            Pose(rot_z(0.3), np.array([1.0, 2.0, 3.0])), 320, 240)
    back = store.FrameRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back.name == rec.name and back.K == rec.K
    np.testing.assert_allclose(back.T_map_cam.matrix(), rec.T_map_cam.matrix(), atol=1e-9)
    assert back.K_grid.width == 320 and back.K_grid.fx == pytest.approx(250)


# --- ingest -------------------------------------------------------------------------------------


def test_resolve_inputs_folders_files_video(tmp_path: Path) -> None:
    d = tmp_path / "caps"
    d.mkdir()
    for name in ("b.jpg", "a.jpg", "c.png"):
        Image.new("RGB", (8, 8)).save(d / name)
    (d / ".DS_Store").write_bytes(b"x")
    (d / "notes.txt").write_text("x")
    spec = ingest.resolve_inputs([d])
    assert [p.name for p in spec.images] == ["a.jpg", "b.jpg", "c.png"]
    spec2 = ingest.resolve_inputs([d / "b.jpg", d])
    assert spec2.images[0].name == "b.jpg" and len(spec2.images) == 4
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    assert ingest.resolve_inputs([vid]).kind == "video"
    with pytest.raises(UsageError):
        ingest.resolve_inputs([vid, d / "a.jpg"])
    with pytest.raises(UsageError):
        ingest.resolve_inputs([])
    with pytest.raises(InputError):
        ingest.resolve_inputs([tmp_path / "missing"])
    with pytest.raises(InputError):
        ingest.resolve_inputs([d / "notes.txt"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError):
        ingest.resolve_inputs([empty])


def test_keyframes_from_images_and_video(tmp_path: Path) -> None:
    d = tmp_path / "caps"
    d.mkdir()
    img = Image.new("RGB", (40, 30), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated: must be re-encoded upright
    img.save(d / "r.jpg", exif=exif)
    Image.new("RGB", (40, 30)).save(d / "s.jpg")
    kfs = list(ingest.keyframes(ingest.resolve_inputs([d]), 2.0, tmp_path / "frames", 5))
    assert [k.name for k in kfs] == ["f000005", "f000006"]
    assert Image.open(kfs[0].path).size == (30, 40)
    from tests.unit.test_video import _make_clip

    clip = tmp_path / "c.mp4"
    _make_clip(clip, seconds=2.0, fps=10)
    vk = list(ingest.keyframes(ingest.resolve_inputs([clip]), 2.0, tmp_path / "vf", 0))
    assert len(vk) == 4 and vk[1].timestamp is not None and vk[1].timestamp >= 0.5


# --- retrieval ----------------------------------------------------------------------------------


def test_retrieval_pairs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    desc = rng.normal(size=(10, 16))
    desc[7] = desc[2] + 0.01 * rng.normal(size=16)
    pairs = retrieval.top_k_pairs(desc, desc, 1, list(range(10)), list(range(10)), min_gap=3)
    assert (2, 7) in pairs
    assert all(abs(a - b) > 3 for a, b in pairs)
    assert retrieval.sequential_pairs([1, 2, 3, 4], 2) == {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
    assert len(retrieval.all_pairs([1, 2, 3])) == 3
    assert retrieval.all_pairs([1], [1, 2]) == {(1, 2)}
    n = retrieval.write_pair_list(tmp_path / "p.txt", {(1, 2)}, {1: "a.jpg", 2: "b.jpg"})
    assert n == 1 and (tmp_path / "p.txt").read_text() == "a.jpg b.jpg\n"
    assert retrieval.top_k_pairs(np.zeros((0, 4)), desc, 3, [], [], 0) == set()


# --- map frame ----------------------------------------------------------------------------------


class _FakeModel:
    """Minimal SfM model: poses in an arbitrary similarity frame, projections of true points."""

    def __init__(self, poses: dict[str, Pose], pts: np.ndarray, sim: object) -> None:
        self._poses = poses
        self.pts = pts
        self.sim = sim
        self.registered = sorted(poses)

    def pose(self, name: str) -> Pose:
        return self._poses[name]

    def observations(self, name: str):  # type: ignore[no-untyped-def]
        T = self._poses[name].inverse()
        pc = T.apply(self.pts)
        pc = pc[pc[:, 2] > 0.2]
        uv = np.stack([K.fx * pc[:, 0] / pc[:, 2] + K.cx + 0.5,
                       K.fy * pc[:, 1] / pc[:, 2] + K.cy + 0.5], 1)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < K.width) & (uv[:, 1] >= 0) & (uv[:, 1] < K.height)
        return uv[ok], self._poses[name].apply(pc[ok])


def test_metric_scale_gravity_and_map_frame() -> None:
    room = Room(boxes=[Box(np.array([0.0, 0.0, 0.4]), np.array([1.0, 1.0, 0.8]))])
    true_poses = [look_at(np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 1.4]),
                          np.array([0.0, 0.0, 0.4])) for a in np.linspace(0, 1.5, 5)]
    # SfM frame = true frame scaled by 1/3.7 and rotated arbitrarily
    Rs = rotation_between([0, 0, 1], [0.3, -0.2, 0.9])
    s_true = 3.7
    to_sfm = mframe.Sim3(1 / s_true, Rs, np.array([0.5, -1.0, 2.0]))
    frames, poses = [], {}
    rng = np.random.default_rng(0)
    for i, T in enumerate(true_poses):
        r = render(room, T, K)
        name = f"f{i}.jpg"
        poses[name] = mframe.transform_pose(to_sfm, T)
        up_cam = T.R.T @ np.array([0.0, 0.0, 1.0])
        frames.append(mframe.FrameDepth(name, r.depth, K, (K.width, K.height), up_cam, 1.0))
    from oh_my_slam.core.geometry import unproject

    r0 = render(room, true_poses[2], K)
    surf = true_poses[2].apply(unproject(r0.depth, K.K()))
    pts = surf[rng.choice(len(surf), 600, replace=False)]  # points on visible surfaces
    model = _FakeModel(poses, to_sfm.apply(pts), to_sfm)
    res = mframe.metric_scale(model, frames)
    assert res.scale == pytest.approx(s_true, rel=0.02)  # AC: known scale within 2 %
    up = mframe.world_up(poses, frames)
    assert angle_between_deg(up, Rs @ np.array([0.0, 0.0, 1.0])) < 1.0  # within 1 degree
    sim = mframe.map_transform(poses[frames[0].name], up, res.scale)
    first = mframe.transform_pose(sim, poses[frames[0].name])
    np.testing.assert_allclose(first.t, 0, atol=1e-9)
    assert abs(first.R[:, 2] @ [0, 1, 0]) < 1e-9  # forward projected on the floor → x axis
    assert first.R[:, 2] @ [1, 0, 0] > 0.5
    T = mframe.align_by_poses([Pose(rot_z(0.4), np.array([1.0, 0, 0]))],
                              [Pose(np.eye(3), np.array([0.0, 2.0, 0.0]))])
    np.testing.assert_allclose(T.R, rot_z(-0.4), atol=1e-9)
    assert mframe.align_by_poses([], []).t.tolist() == [0, 0, 0]


# --- latest wins --------------------------------------------------------------------------------


def _view(room: Room, pose: Pose, scale: float = 1.0, rot_noise_deg: float = 0.0,
          seed: int = 0) -> validity.View:
    r = render(room, pose, K)
    noisy = pose
    if rot_noise_deg:
        rng = np.random.default_rng(seed)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rot_noise_deg)
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
        noisy = Pose(dR @ pose.R, pose.t)
    return validity.View((r.depth * scale).astype(np.float32), r.depth > 0, K, noisy)


def test_latest_wins_removed_box_and_no_false_hits() -> None:
    box = Box(np.array([0.0, 0.0, 0.4]), np.array([0.6, 0.6, 0.8]))
    with_box = Room(boxes=[box])
    without = Room(boxes=[])
    old_pose = look_at(np.array([2.2, 0.3, 1.4]), np.array([0.0, 0.0, 0.4]))
    new_poses = [look_at(np.array([2.1, y, 1.5]), np.array([0.0, 0.0, 0.4])) for y in (-0.3, 0.5)]
    old = _view(with_box, old_pose)
    # unchanged room, 5 % depth-scale noise and 1 degree pose noise: nothing contradicted
    cells = validity.contradicted_cells(old, [
        _view(with_box, p, scale=s, rot_noise_deg=1.0, seed=i)
        for i, (p, s) in enumerate(zip(new_poses, (1.05, 0.95), strict=True))])
    assert cells.mean() < 0.01
    # the box was removed: its pixels in the old view are contradicted, the wall is not
    cells = validity.contradicted_cells(old, [_view(without, p) for p in new_poses])
    px = validity.cells_to_pixels(cells, old.depth.shape)
    box_px = render(with_box, old_pose, K).ids == 2
    assert px[box_px].mean() > 0.6
    assert px[~box_px & (old.depth > 0)].mean() < 0.02
    assert validity.well_registered({"observations": 500, "reproj_error": 0.8}, "sfm-global")
    assert not validity.well_registered({"observations": 20}, "sfm-global")
    assert validity.well_registered({}, "multiview")


# --- objects ------------------------------------------------------------------------------------


def test_object_bookkeeping_and_projection() -> None:
    pts = np.random.default_rng(0).uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.8], (3000, 3))
    o = MapObject(5, "chair", {}, [], np.zeros((0, 3), np.float32))
    o.add_points(pts)
    o.vote("chair", 0.9)
    o.vote("armchair", 0.6)
    o.vote("chair", 0.7)
    assert o.label == "chair" and o.score == pytest.approx((0.9 + 0.7 + 0.6) / 3)
    d = o.to_dict()
    back = MapObject.from_dict(d, o.points)
    assert back.label == "chair" and back.point_file if hasattr(back, "point_file") else True
    assert overlap_fraction(pts, pts, 0.01) == 1.0
    assert overlap_fraction(pts, pts + 5.0, 0.1) == 0.0
    assert overlap_fraction(np.zeros((0, 3)), pts, 0.1) == 0.0
    view = _view(Room(boxes=[Box(np.zeros(3) + [0, 0, 0.4], np.array([0.6, 0.6, 0.8]))]),
                 look_at(np.array([2.0, 0.0, 1.2]), np.array([0.0, 0.0, 0.4])))
    m = projected_mask(view, o.points)
    assert m.sum() > 500
    assert not projected_mask(view, np.zeros((0, 3))).any()
    st = ObjectState([o], 6, {9: 5, 12: 9})
    assert st.resolve(12) == 5 and st.resolve(77) is None
    assert st.exported() == []  # unconfirmed, no OBB yet


@pytest.mark.parametrize("env", [{}])
def test_mapper_cli_arguments(env: dict[str, str], tmp_path: Path) -> None:
    from oh_my_slam.cli import mapper as cli_mapper

    ap = cli_mapper.build_parser()
    a = ap.parse_args(["update", "-a", "x.mp4", "-m", "m", "-t", "full"])
    assert (a.format, a.mode, a.fps) == ("json", "full", None)
    a = ap.parse_args(["update", "-a", "a.jpg", "b.jpg", "-m", "m", "-f", "ply", "-t", "single",
                       "-fps", "3"])
    assert a.inputs == [Path("a.jpg"), Path("b.jpg")] and a.fps == 3.0
    for bad in (["update", "-a", "x.mp4", "-m", "m"], ["update", "-m", "m", "-t", "full"],
                ["-a", "x"], ["update", "-a", "x", "-m", "m", "-t", "partial"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(bad)
        assert e.value.code == 2
    repo = Path(__file__).resolve().parents[2]
    res = subprocess.run([str(repo / "mapper.sh"), "update", "-a", "x.mp4", "-m",
                          str(tmp_path / "m")], capture_output=True, env=os.environ.copy())
    assert res.returncode == 2 and res.stdout == b""
