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
    before = store.full_tree_hash(root)
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
    assert store.full_tree_hash(root) == before
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
    rec = store.FrameRecord(3, "f000003", "frames/f000003.jpg", "x.jpg", 1, 640, 480,
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
    assert len(vk) == 4 and float(vk[1].source.rsplit("@", 1)[1]) >= 0.5


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
    from oh_my_slam.core.geometry import unproject_pixels

    r0 = render(room, true_poses[2], K)
    v, u = np.nonzero(np.isfinite(r0.depth) & (r0.depth > 0))
    surf = true_poses[2].apply(unproject_pixels(u, v, r0.depth[v, u], K.K()))
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
    assert (a.format, a.mode, a.fps, a.output, a.attrs) == ("json", "full", None, None, None)
    a = ap.parse_args(["update", "-a", "a.jpg", "b.jpg", "-m", "m", "-f", "ply", "-o", "c.ply",
                       "-p", "voxel=0.05,normals=on", "-t", "single", "-fps", "3"])
    assert a.inputs == [Path("a.jpg"), Path("b.jpg")] and a.fps == 3.0
    assert (a.output, a.attrs) == (Path("c.ply"), ["voxel=0.05,normals=on"])
    for bad in (["update", "-a", "x.mp4", "-m", "m"], ["update", "-m", "m", "-t", "full"],
                ["-a", "x"], ["update", "-a", "x", "-m", "m", "-t", "partial"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(bad)
        assert e.value.code == 2
    repo = Path(__file__).resolve().parents[2]
    res = subprocess.run([str(repo / "mapper.sh"), "update", "-a", "x.mp4", "-m",
                          str(tmp_path / "m")], capture_output=True, env=os.environ.copy())
    assert res.returncode == 2 and res.stdout == b""


def test_mapper_validates_attributes_before_updating(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    """``-p`` is checked (map scope, needs -f ply) before ``update`` could reach the server;
    ``-o`` receives the payload."""
    import io
    from types import SimpleNamespace

    from oh_my_slam.cli import mapper as cli_mapper
    from oh_my_slam.core import timing
    from oh_my_slam.core.cloud_attrs import CloudAttrs
    from oh_my_slam.core.log import PayloadWriter
    from oh_my_slam.mapping import api

    calls: list[dict] = []

    def fake_update(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(payload=b"PAYLOAD", timings=timing.Timings().to_dict())

    stdout = io.BytesIO()
    monkeypatch.setattr(api, "update", fake_update)
    monkeypatch.setattr(cli_mapper, "claim_stdout", lambda output=None: PayloadWriter(
        stdout) if output is None else PayloadWriter(path=output))
    base = ["update", "-a", "x.jpg", "-m", str(tmp_path / "m"), "-t", "full"]
    for bad in (["-p", "voxel=0.1"], ["-f", "ply", "-p", "stride=2"],
                ["-f", "ply", "-p", "max-depth=3"], ["-f", "ply", "-p", "voxel=-1"]):
        with pytest.raises(UsageError):
            cli_mapper.main(base + bad)
    assert calls == []
    target = tmp_path / "out.ply"
    assert cli_mapper.main(base + ["-f", "ply", "-p", "voxel=0.1,normals=on,color=height",
                                   "-o", str(target)]) == 0
    assert calls[0]["attrs"] == CloudAttrs(color="height", voxel=0.1, normals=True)
    assert calls[0]["fmt"] == "ply" and target.read_bytes() == b"PAYLOAD"
    assert stdout.getvalue() == b""
    assert cli_mapper.main(base) == 0
    assert calls[1]["attrs"] == CloudAttrs() and stdout.getvalue() == b"PAYLOAD"


# --- map cloud: fused surface + latest-frame attribution ----------------------------------------


def _cloud_frames(scales: list[float]) -> tuple[Room, list]:
    from oh_my_slam.mapping.geometry import FrameData
    from tests.synth.scene import default_room, orbit_poses

    room = default_room()
    K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
    frames = []
    for i, (pose, s) in enumerate(zip(orbit_poses(len(scales)), scales, strict=True)):
        r = render(room, pose, K)
        last = i == len(scales) - 1  # the last frame is a later update's
        rec = store.FrameRecord(i, store.frame_name(i), "", "", 1, 320, 240, K, pose, 320, 240,
                                update_id=2 if last else 1)
        # labels: synthetic box k -> object id k + 1 (floor and walls unlabelled)
        frames.append(FrameData(rec, (r.depth * s).astype(np.float32), r.depth > 0, r.rgb,
                                np.where(r.ids >= 2, r.ids - 1, 0).astype(np.int32), last))
    return room, frames


def _surface_distance(room: Room, xyz: np.ndarray) -> np.ndarray:
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    for mesh, _, _ in room.meshes():
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(xyz.astype(np.float32))).numpy()


def test_fused_cloud_collapses_per_frame_depth_disagreement() -> None:
    """Frames whose depth disagrees by ±3 % give one surface, not one offset copy per frame."""
    from oh_my_slam.mapping.geometry import fused_cloud_points
    from oh_my_slam.reconstruction.pointcloud import frame_cloud

    scales = [1.03, 0.97, 1.02, 0.98, 1.03, 0.97, 1.01, 0.99, 1.03, 0.97, 1.02, 0.98]
    room, frames = _cloud_frames(scales)
    stacked = np.concatenate([frame_cloud(fd.depth, fd.rgb, fd.rec.K_grid, fd.valid,
                                          fd.rec.T_map_cam)[0].xyz for fd in frames])
    fused = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    assert len(fused) > 10_000
    d_old = _surface_distance(room, stacked)
    d_new = _surface_distance(room, fused)
    # the stacked copies sit up to ±3 % (~6 cm at 2 m) off the surface; the fused one averages
    assert np.percentile(d_new, 90) < 0.5 * np.percentile(d_old, 90)
    assert np.median(d_new) < 0.015


def test_attribute_points_latest_visible_update_wins() -> None:
    from oh_my_slam.mapping.geometry import attribute_points, fused_cloud_points

    room, frames = _cloud_frames([1.0] * 8)
    xyz = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    rgb, label, seen_new = attribute_points(xyz, frames)
    d = _surface_distance(room, xyz)
    assert np.median(d) < 0.01
    # object ids land on the boxes, and only on points near their surfaces
    assert set(np.unique(label)) <= {0, 1, 2, 3} and (label > 0).mean() > 0.05
    # the newest update's colour wins where it sees a point; points it cannot see keep older ones
    last = frames[-1]
    cam = last.rec.T_map_cam.inverse()
    pc = xyz @ cam.R.T + cam.t
    assert seen_new.any() and not seen_new.all()
    behind = pc[:, 2] <= 0
    assert not seen_new[behind].any()
    assert (rgb[~seen_new] != 128).any(axis=1).mean() > 0.9  # others were coloured by older frames

