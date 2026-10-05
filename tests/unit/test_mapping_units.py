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


def test_store_overlay_falls_back_to_the_applied_file(tmp_path: Path) -> None:
    """An update killed partway through applying its commit: the files it already moved into
    place are no longer staged, so a reader takes them from the map itself."""
    root = tmp_path / "m"
    _make_map(root)
    tx = store.MapTransaction(root).__enter__()
    tx.write_bytes("per_frame/f000000/x.bin", b"new")
    tx.write_bytes("per_frame/f000000/y.bin", b"staged")
    tx.write_json(store.MAP_JSON, {"update_count": 2})
    files = ["per_frame/f000000/x.bin", "per_frame/f000000/y.bin", store.MAP_JSON]
    (tx.staging / store.COMMIT).write_text(json.dumps({"files": files, "delete": []}))
    tx.__exit__(None, None, None)
    (tx.staging / "per_frame/f000000/x.bin").replace(root / "per_frame/f000000/x.bin")  # applied
    reader = store.MapReader(root)
    assert reader.path("per_frame/f000000/x.bin") == root / "per_frame/f000000/x.bin"
    assert reader.path("per_frame/f000000/x.bin").read_bytes() == b"new"
    assert reader.path("per_frame/f000000/y.bin").read_bytes() == b"staged"  # still staged
    assert reader.meta["update_count"] == 2


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


def test_folder_is_empty_only_without_entries_or_with_tool_leftovers(tmp_path: Path) -> None:
    """Spec 2.3: a map is created when the folder is missing or empty. A `.git` (or any other
    hidden entry) makes it non-empty and is refused untouched; `.DS_Store` and this tool's own
    `.lock` / `.staging` leftovers do not."""
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    only_ds = tmp_path / "ds"
    only_ds.mkdir()
    (only_ds / ".DS_Store").write_text("")
    leftovers = tmp_path / "leftovers"
    (leftovers / ".staging").mkdir(parents=True)
    (leftovers / ".lock").write_text("")
    for empty in (nothing, only_ds, leftovers):
        assert store.classify(empty) == "empty", empty
        with store.MapTransaction(empty) as tx:
            assert tx.created
    dotgit = tmp_path / "dotgit"
    (dotgit / ".git").mkdir(parents=True)
    hidden_file = tmp_path / "hidden_file"
    hidden_file.mkdir()
    (hidden_file / ".env").write_text("x")
    for other in (dotgit, hidden_file):
        before = sorted(p.name for p in other.iterdir())
        assert store.classify(other) == "other", other
        with pytest.raises(NotAMapError), store.MapTransaction(other):
            pass
        assert sorted(p.name for p in other.iterdir()) == before  # untouched, no .lock either


def test_frame_record_roundtrip() -> None:
    rec = store.FrameRecord(3, "f000003", "frames/f000003.jpg", "x.jpg", 1, 640, 480,
                            Intrinsics(500, 500, 320, 240, 640, 480, "colmap"),
                            Pose(rot_z(0.3), np.array([1.0, 2.0, 3.0])), 320, 240)
    back = store.FrameRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back.name == rec.name and back.K == rec.K
    np.testing.assert_allclose(back.T_map_cam.matrix(), rec.T_map_cam.matrix(), atol=1e-9)
    assert back.K_grid.width == 320 and back.K_grid.fx == pytest.approx(250)


def test_scene_metadata_floats_are_rounded_for_byte_parity(tmp_path: Path) -> None:
    """Threaded SfM makes the map's scale and floor height differ by ~1e-12 between identical
    runs: the scene JSON rounds them as it rounds poses and cuboids, so the two runs' documents
    are byte-identical."""
    from oh_my_slam.mapping import export

    rec = store.FrameRecord(0, "f000000", "frames/f000000.jpg", "x.jpg", 1, 640, 480,
                            Intrinsics(500, 500, 320, 240, 640, 480, "colmap"),
                            Pose(rot_z(0.3), np.array([1.0, 2.0, 3.0])), 320, 240)

    def doc(eps: float) -> bytes:
        meta = {"scale": {"sfm_to_metric": 0.123456789 + eps, "spread": 0.0421 + eps,
                          "frames": 13, "method": "sfm"},
                "floor_z": np.float64(-1.2345678912 + eps), "update_count": 1,
                "map_frame": {"units": "m", "gravity_aligned": True}}
        return json.dumps(export.full_scene(tmp_path, meta, [rec], [])).encode()

    a, b = doc(0.0), doc(3e-12)
    assert a == b
    md = json.loads(a)["openlabel"]["metadata"]
    assert md["scale"] == {"sfm_to_metric": 0.123457, "spread": 0.0421, "frames": 13,
                           "method": "sfm"}
    assert md["floor_z"] == -1.234568 and md["map_frame"]["gravity_aligned"] is True


# --- ingest -------------------------------------------------------------------------------------


def test_resolve_inputs_images_in_order_or_one_video(tmp_path: Path) -> None:
    """Spec §2.3: ``-i`` is image(s) or a video; images keep the order given, folders are refused."""
    d = tmp_path / "caps"
    d.mkdir()
    for name in ("b.jpg", "a.jpg", "c.png"):
        Image.new("RGB", (8, 8)).save(d / name)
    (d / "notes.txt").write_text("x")
    spec = ingest.resolve_inputs([d / "b.jpg", d / "a.jpg", d / "c.png"])
    assert spec.kind == "images" and [p.name for p in spec.images] == ["b.jpg", "a.jpg", "c.png"]
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
    with pytest.raises(InputError, match="is a folder"):
        ingest.resolve_inputs([d])


def test_keyframes_from_images_and_video(tmp_path: Path) -> None:
    d = tmp_path / "caps"
    d.mkdir()
    img = Image.new("RGB", (40, 30), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated: must be re-encoded upright
    img.save(d / "r.jpg", exif=exif)
    Image.new("RGB", (40, 30)).save(d / "s.jpg")
    kfs = list(ingest.keyframes(ingest.resolve_inputs([d / "r.jpg", d / "s.jpg"]), 2.0,
                                tmp_path / "frames", 5))
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
def test_mapper_cli_arguments(env: dict[str, str], tmp_path: Path,
                              capsys: pytest.CaptureFixture[str]) -> None:
    """Only ``-i`` and ``-m`` are required (``-t`` defaults to full); ``-a`` no longer exists."""
    from oh_my_slam.cli import mapper as cli_mapper

    ap = cli_mapper.build_parser()
    a = ap.parse_args(["update", "-i", "x.mp4", "-m", "m"])
    assert (a.inputs, a.map) == ([Path("x.mp4")], Path("m"))
    assert (a.format, a.mode, a.fps, a.output, a.attrs) == ("json", "full", None, None, None)
    assert ap.parse_args(["update", "-i", "x.mp4", "-m", "m", "-t", "full"]).mode == "full"
    a = ap.parse_args(["update", "-i", "a.jpg", "b.jpg", "-m", "m", "-f", "ply", "-o", "c.ply",
                       "-p", "voxel=0.05,normals=on", "-t", "single", "-fps", "3"])
    assert a.inputs == [Path("a.jpg"), Path("b.jpg")] and (a.mode, a.fps) == ("single", 3.0)
    assert (a.output, a.attrs) == (Path("c.ply"), ["voxel=0.05,normals=on"])
    for bad, message in ((["update", "-m", "m"], "required: -i"),
                         (["update", "-i", "x.mp4"], "required: -m"),
                         (["update", "-a", "x.mp4", "-m", "m"], "required: -i"),
                         (["update", "-i", "x.mp4", "-m", "m", "-a", "y.mp4"],
                          "unrecognized arguments: -a"),
                         (["-i", "x"], "invalid choice"),
                         (["update", "-i", "x", "-m", "m", "-t", "partial"], "invalid choice")):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(bad)
        assert e.value.code == 2
        assert message in capsys.readouterr().err, bad
    repo = Path(__file__).resolve().parents[2]
    res = subprocess.run([str(repo / "mapper.sh"), "update", "-a", "x.mp4", "-m",
                          str(tmp_path / "m")], capture_output=True, env=os.environ.copy())
    assert res.returncode == 2 and res.stdout == b"" and b"required: -i" in res.stderr
    assert not (tmp_path / "m").exists()


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
    (tmp_path / "x.jpg").write_bytes(b"")  # the inputs are checked before update() runs
    base = ["update", "-i", str(tmp_path / "x.jpg"), "-m", str(tmp_path / "m")]
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
    assert cli_mapper.main(base) == 0  # defaults: the whole map as JSON to stdout
    assert calls[1]["attrs"] == CloudAttrs() and stdout.getvalue() == b"PAYLOAD"
    assert (calls[1]["mode"], calls[1]["fmt"], calls[1]["fps"]) == ("full", "json", 2.0)
    assert cli_mapper.main(base + ["-t", "single"]) == 0
    assert calls[2]["mode"] == "single"


def test_mapper_ignores_fps_for_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Spec §2.3: ``-fps`` is ignored for images, so any value (0, negative) only warns; for a
    video a non-positive value is a usage error. Through the shell with the server down, images
    with ``-fps 0`` reach the server check (exit 3, not 2) after the warning."""
    from types import SimpleNamespace

    from oh_my_slam.cli import mapper as cli_mapper
    from oh_my_slam.core import timing
    from oh_my_slam.core.log import PayloadWriter
    from oh_my_slam.mapping import api

    calls: list[dict] = []
    warnings: list[str] = []
    monkeypatch.setattr(api, "update", lambda *a, **k: calls.append(k) or SimpleNamespace(
        payload=b"{}\n", timings=timing.Timings().to_dict()))
    monkeypatch.setattr(cli_mapper, "claim_stdout", lambda output=None: PayloadWriter(
        path=tmp_path / "out.json"))
    monkeypatch.setattr(cli_mapper.log, "warning", lambda msg, *a: warnings.append(msg % a))
    for fps in ("0", "-1", "3"):
        for name in ("a.jpg", "b.jpg"):  # the inputs are checked before update() runs
            (tmp_path / name).write_bytes(b"")
        assert cli_mapper.main(["update", "-i", str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg"),
                                "-m", str(tmp_path / "m"),
                                "-fps", fps]) == 0
        assert warnings.pop() == "-fps applies to video input only; ignored for images"
    assert len(calls) == 3
    from oh_my_slam.mapping.ingest import DEFAULT_FPS

    assert [c["fps"] for c in calls] == [DEFAULT_FPS] * 3  # ignored: the default is passed on
    for fps in ("0", "-1"):
        with pytest.raises(UsageError, match="-fps must be positive"):
            cli_mapper.main(["update", "-i", "x.mp4", "-m", str(tmp_path / "m"), "-fps", fps])
    assert len(calls) == 3 and warnings == []

    img = tmp_path / "a.jpg"
    Image.new("RGB", (32, 24)).save(img)
    repo = Path(__file__).resolve().parents[2]
    res = subprocess.run([str(repo / "mapper.sh"), "update", "-i", str(img), "-m",
                          str(tmp_path / "m2"), "-fps", "0"], capture_output=True,
                         env=os.environ.copy(), timeout=60)
    assert res.returncode == 3 and res.stdout == b"", res.stderr
    assert b"-fps applies to video input only; ignored for images" in res.stderr


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


def _wall_frames(spread: float | None = None) -> tuple[Room, list]:
    """One keyframe (0) looking at the +x wall, which no other keyframe has in view, and three
    (1-3) looking at the -x wall; keyframe 0's depth scale spread is ``spread``."""
    from oh_my_slam.mapping.geometry import FrameData

    room = Room()
    K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)
    poses = [look_at(np.array([0.5, 0.0, 1.3]), np.array([3.0, 0.0, 1.0]))]
    poses += [look_at(np.array([0.5, dy, 1.3]), np.array([-3.0, dy, 1.0]))
              for dy in (-0.1, 0.0, 0.1)]
    frames = []
    for i, pose in enumerate(poses):
        r = render(room, pose, K)
        stats = {} if i or spread is None else {"depth_scale_method": "dense",
                                                "depth_scale_spread": spread}
        rec = store.FrameRecord(i, store.frame_name(i), "", "", 1, 320, 240, K, pose, 320, 240,
                                stats=stats)
        frames.append(FrameData(rec, r.depth.astype(np.float32), r.depth > 0, r.rgb,
                                np.zeros(r.depth.shape, np.int32), False))
    return room, frames


def test_a_surface_that_only_one_keyframe_has_in_view_is_in_the_cloud() -> None:
    """CLOUD_MIN_VIEWS drops speckle, not what only one or two keyframes look at (a laptop at the
    corner of two photos): the +x wall, which only keyframe 0 sees, is drawn."""
    from oh_my_slam.mapping.geometry import fused_cloud_points

    room, frames = _wall_frames()
    xyz = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    wall = xyz[:, 0] > 2.5
    assert wall.sum() > 5000 and (xyz[:, 0] < -2.5).sum() > 5000
    assert np.percentile(_surface_distance(room, xyz[wall]), 99) < 0.02


def test_speckle_that_other_keyframes_see_through_needs_the_usual_views() -> None:
    """A blob that one keyframe places in front of the -x wall, where the other keyframes see the
    wall, is not drawn; nor is what only a keyframe with an ill-measured depth scale sees."""
    from oh_my_slam.mapping.geometry import fused_cloud_points

    room, frames = _wall_frames()
    blob = np.zeros(frames[1].depth.shape, bool)
    blob[100:140, 140:180] = True
    frames[1].depth = np.where(blob, 0.6 * frames[1].depth, frames[1].depth).astype(np.float32)
    xyz = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    floating = (xyz[:, 0] < -0.5) & (xyz[:, 0] > -2.5) & (xyz[:, 2] > 0.3) & (xyz[:, 2] < 2.3)
    assert floating.sum() < 20
    room, frames = _wall_frames(spread=0.2)
    xyz = fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    assert (xyz[:, 0] > 2.5).sum() == 0 and (xyz[:, 0] < -2.5).sum() > 5000


def test_cell_culling_selects_every_point_a_keyframe_can_see() -> None:
    """``_Cells`` culls only points a keyframe cannot see: visibility, seeing through and the
    places of removed objects are the same with and without it, for points all around the
    camera, near it, beyond its depth and at the image border."""
    from oh_my_slam.core import rle
    from oh_my_slam.mapping import geometry as g
    from oh_my_slam.mapping.objects import Vacated

    _, frames = _wall_frames()
    rng = np.random.default_rng(3)
    fd = frames[0]
    # points on and around the keyframe's surfaces (back-projected pixels, jittered) + anywhere
    h, w = fd.depth.shape
    v, u = rng.integers(0, h, 40_000), rng.integers(-20, w + 20, 40_000)
    z = fd.depth[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)] * rng.uniform(0.9, 1.1, 40_000)
    K = fd.rec.K_grid
    pc = np.stack([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], axis=1)
    pts = np.concatenate([fd.rec.T_map_cam.apply(pc), rng.uniform(-4, 4, (20_000, 3))])
    cells = g._Cells(pts)
    for f in frames:
        a, b = g._visible(f, pts), g._visible(f, pts, cells)
        assert len(a[0]) > 0 or f is not fd
        assert set(a[0].tolist()) == set(b[0].tolist())
        assert np.array_equal(g._residuals(f, pts)[0], g._residuals(f, pts, cells)[0],
                              equal_nan=True)
    mask = np.zeros((h, w), bool)
    mask[0:60, 280:320] = True  # at the image corner
    place = Vacated(1, 7, {fd.rec.name: rle.encode(mask)}, [frames[1].rec.name], None, 3.0)
    by_name = {f.rec.name: f for f in frames}
    region = g._vacated_region(pts, place, by_name, cells)
    assert region.sum() > 100
    assert np.array_equal(region, g._vacated_region(pts, place, by_name))
    from oh_my_slam.segmentation.api import OBB

    box = OBB(pts[0], np.eye(3), np.array([0.6, 0.4, 0.8]))  # on the wall keyframe 0 sees
    place = Vacated(1, 8, {}, [frames[1].rec.name], box, 3.0)
    region = g._vacated_region(pts, place, by_name, cells)
    assert region.sum() > 100
    assert np.array_equal(region, g._vacated_region(pts, place, by_name))


def test_fusion_in_slabs_or_a_region_is_the_whole_fusion(monkeypatch: pytest.MonkeyPatch
                                                         ) -> None:
    """A large map is fused slab by slab (``TILE_BLOCKS``), and a box of it alone (``region``),
    with the same points as one fusion of every block: a voxel depends only on the frames, a
    surface point on its two voxels, the tests of few views on the point."""
    from oh_my_slam.mapping import geometry as g

    _, frames = _cloud_frames([1.03, 0.97, 1.02, 0.98, 1.01, 0.99])
    frames[2].depth = np.where(frames[2].depth > 2.5, 0.8 * frames[2].depth,
                               frames[2].depth).astype(np.float32)  # a surface few frames see
    whole = g.fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    monkeypatch.setattr(g, "TILE_BLOCKS", 2000)
    tiles = g._tiles
    counts = []
    monkeypatch.setattr(g, "_tiles", lambda *a: counts.append(len(t := tiles(*a))) or t)
    sliced = g.fused_cloud_points(frames, voxel=0.01, depth_max=6.0)
    assert counts[0] > 3
    assert len(whole) > 10_000 and np.array_equal(whole, sliced)
    lo, hi = np.array([-0.5, -1.0, 0.2]), np.array([1.5, 0.7, 1.4])
    inside = whole[np.all((whole >= lo) & (whole <= hi), axis=1)]
    assert len(inside) > 1000
    assert np.array_equal(g.fused_cloud_points(frames, voxel=0.01, depth_max=6.0,
                                               region=(lo, hi)), inside)
    blocks = g._frame_blocks(frames, 0.01, 6.0)  # every keyframe's blocks, for several boxes
    assert np.array_equal(g.fused_cloud_points(frames, voxel=0.01, depth_max=6.0,
                                               region=(lo, hi), blocks=blocks), inside)


def test_border_depth_defers_to_keyframes_that_see_the_surface_centrally() -> None:
    """A keyframe whose depth is 7 % too deep in its outer 15 % (monocular depth is least reliable
    at the image border): where its neighbours see those surfaces near their centre, its border
    takes their depth; its centre, and what no neighbour sees centrally, keep theirs."""
    from oh_my_slam.mapping.geometry import BORDER_BAND, _central, correct_borders

    room, frames = _cloud_frames([1.0] * 12)
    truth = [fd.depth.copy() for fd in frames]
    fd0 = frames[0]
    h, w = fd0.depth.shape
    border = ~_central(np.arange(w)[None, :] + 0.5, np.arange(h)[:, None] + 0.5, w, h,
                       BORDER_BAND)
    fd0.depth = np.where(border, fd0.depth * 1.07, fd0.depth).astype(np.float32)
    assert correct_borders(frames) > 0
    ok = fd0.valid & (truth[0] > 0)
    ratio = fd0.depth[ok & border] / truth[0][ok & border]
    fixed = np.abs(ratio - 1.0) < 0.01
    assert fixed.mean() > 0.5  # most of the border is seen centrally by a neighbour
    # never pushed past the truth (but for a few pixels where a box edge meets the wall)
    assert np.mean((ratio > 0.97) & (ratio < 1.0701)) > 0.995
    np.testing.assert_array_equal(fd0.depth[~border], truth[0][~border])
    for fd, d in zip(frames[1:], truth[1:], strict=True):  # consistent neighbours barely move
        np.testing.assert_array_equal(fd.depth[~border], d[~border])
        change = np.abs(fd.depth[border & (d > 0)] / d[border & (d > 0)] - 1)
        assert np.percentile(change, 95) < 0.005 and np.percentile(change, 99) < 0.04


def test_consensus_puts_disagreeing_keyframes_on_one_surface() -> None:
    """Keyframes whose depth disagrees by ±1.2 % (as neighbouring keyframes do after the depth
    adjustment): each pixel takes the median of the depths the keyframes that see it give along
    its ray, so every keyframe lands within a fraction of a percent of the others — and the TSDF
    fuses one surface instead of a layer per placement. The stored depth is not touched."""
    from oh_my_slam.mapping.geometry import consensus_depths, fused_cloud_points

    scales = [1.012, 0.988, 1.0] * 8
    room, frames = _cloud_frames(scales)
    truth = [fd.depth / s for fd, s in zip(frames, scales, strict=True)]
    box = (np.array([-1.5, -1.5, -0.1]), np.array([1.5, 0.0, 1.0]))  # floor, boxes, sofa
    # at the map's voxel (5 mm, a 4 cm band), which keeps placements 2.4 % apart at 2 m apart
    before = fused_cloud_points(frames, voxel=0.005, depth_max=6.0, region=box)
    moved, dropped = consensus_depths(frames)
    assert moved > 0
    errs = []
    for fd, t in zip(frames, truth, strict=True):
        ok = fd.valid & (t > 0)
        errs.append(np.abs(fd.depth[ok] / t[ok] - 1.0))
    err = np.concatenate(errs)
    assert np.median(err) < 0.002 and np.percentile(err, 90) < 0.006  # 0.012 for 2/3 before
    assert dropped < 0.01 * sum(int(fd.valid.sum()) for fd in frames)  # nothing is seen through
    after = fused_cloud_points(frames, voxel=0.005, depth_max=6.0, region=box)
    d_before, d_after = _surface_distance(room, before), _surface_distance(room, after)
    assert np.percentile(d_after, 90) < 0.5 * np.percentile(d_before, 90)
    assert len(after) < 0.8 * len(before)  # fewer layers


def test_consensus_leaves_out_what_other_keyframes_see_through() -> None:
    """A blob that one keyframe places in front of the -x wall, where two other keyframes of its
    update see the wall, is left out of the fusion (free-space violation); the rest of its
    pixels are not. A keyframe of an older update does not carve a newer one's pixels: what it
    saw through may stand there now (latest wins)."""
    from oh_my_slam.mapping.geometry import consensus_depths

    for newer in (False, True):
        _, frames = _wall_frames()
        blob = np.zeros(frames[1].depth.shape, bool)
        blob[100:140, 140:180] = True
        frames[1].depth = np.where(blob, 0.6 * frames[1].depth,
                                   frames[1].depth).astype(np.float32)
        if newer:
            frames[1].rec.update_id = 2
        consensus_depths(frames)
        drop = frames[1].drop
        assert drop is not None
        if newer:
            assert not drop[blob].any()
        else:
            assert drop[blob].mean() > 0.9
        assert drop[~blob & frames[1].valid].mean() < 0.01


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


# --- SfM extension stays in the map frame -------------------------------------------------------


class _PoseModel:
    """Duck-typed SfmModel: named camera-to-world poses and a similarity transform."""

    def __init__(self, poses: dict[str, Pose]) -> None:
        self.poses = dict(poses)
        self.notes: dict[str, object] = {}

    @property
    def registered(self) -> list[str]:
        return sorted(self.poses)

    def pose(self, name: str) -> Pose:
        return self.poses[name]

    def transform(self, s: float, R: np.ndarray, t: np.ndarray) -> None:
        self.poses = {n: Pose(R @ T.R, s * R @ T.t + t) for n, T in self.poses.items()}


def test_incremental_extension_is_brought_back_onto_the_map() -> None:
    """COLMAP returns an extended model re-normalised by a similarity (fixed frames included):
    it is mapped back so the fixed frames sit at their stored poses. A fixed frame COLMAP dropped
    and registered again elsewhere (a weakly supported stored pose) is left out of the similarity
    and reported — the new frames, registered on the others, still land in the map frame (office
    split: one re-placed keyframe discarded the whole extension, and every new keyframe fell back
    to multi-view poses decimetres off). A model whose fixed frames mostly moved is rejected."""
    from oh_my_slam.mapping.sfm import _back_onto

    rng = np.random.default_rng(1)
    stored = {f"f{i}": look_at(rng.uniform(-2, 2, 3) + [0, 0, 1.5], np.array([0.0, 0.0, 0.4]))
              for i in range(5)}
    new = {"n0": look_at(np.array([2.5, 0.3, 1.4]), np.array([0.0, 0.0, 0.4]))}
    R = rotation_between([0, 0, 1], [0.2, 0.5, 0.8])

    def colmap_output(poses: dict[str, Pose]) -> _PoseModel:
        out = _PoseModel(poses)
        out.transform(3.7, R, np.array([1.0, -2.0, 0.5]))  # COLMAP's normalisation
        return out

    out = colmap_output({**stored, **new})
    sim = mframe.similarity_by_poses([out.pose(n) for n in stored], list(stored.values()))
    assert sim.s == pytest.approx(1 / 3.7)
    back = _back_onto(out, _PoseModel(stored))  # type: ignore[arg-type]
    assert back is not None and "moved_fixed" not in back.notes
    for n, T in {**stored, **new}.items():
        np.testing.assert_allclose(back.pose(n).matrix(), T.matrix(), atol=1e-9)
    replaced = Pose(rot_z(0.05) @ stored["f0"].R, stored["f0"].t + [0.08, 0, 0])
    back = _back_onto(colmap_output({**stored, "f0": replaced, **new}),  # type: ignore[arg-type]
                      _PoseModel(stored))  # type: ignore[arg-type]
    assert back is not None and back.notes["moved_fixed"] == ["f0"]
    for n, T in {**{k: v for k, v in stored.items() if k != "f0"}, **new}.items():
        np.testing.assert_allclose(back.pose(n).matrix(), T.matrix(), atol=1e-9)
    moved = {n: Pose(T.R, T.t + rng.normal(0, 0.5, 3)) for n, T in stored.items()
             if n in ("f0", "f1", "f2")}
    assert _back_onto(colmap_output({**stored, **moved}),  # type: ignore[arg-type]
                      _PoseModel(stored)) is None  # type: ignore[arg-type]


def test_poses_that_contradict_their_matches_are_found() -> None:
    """``sfm.contradicted``: exact matches of five keyframes looking at one scene; one keyframe
    posed 0.5° off (the 6-photo office map's global-mapper pose of f000005 was 0.43° off its
    matches, every correct pose within 0.12°) is found — and only it, although its partners'
    pairs share its error. Two cameras at one centre agree for any translation direction when
    their rotations do."""
    from oh_my_slam.mapping.panorama import PairMatches
    from oh_my_slam.mapping.sfm import contradicted, epipolar_deg

    K = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)
    rng = np.random.default_rng(3)
    X = rng.uniform([-1.5, -1.0, -0.5], [1.5, 1.0, 1.5], (400, 3)) + [0.0, 0.0, 0.0]
    poses = {f"k{i}": look_at(np.array([4.0 * np.cos(a), 4.0 * np.sin(a), 1.2]),
                              np.array([0.0, 0.0, 0.3]))
             for i, a in enumerate(np.linspace(0.0, 0.8, 5))}

    def project(T: Pose) -> np.ndarray:
        pc = T.inverse().apply(X)
        return np.column_stack([K.fx * pc[:, 0] / pc[:, 2] + K.cx, K.fy * pc[:, 1] / pc[:, 2] + K.cy])

    uv = {n: project(T) for n, T in poses.items()}
    names = sorted(poses)
    matches = [PairMatches(a, b, uv[a], uv[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    Ks = {n: K for n in names}
    assert contradicted(poses, Ks, matches) == {}
    off = dict(poses, k2=Pose(rotation_between([0, 0, 1], [np.sin(np.radians(0.5)), 0,
                                                            np.cos(np.radians(0.5))])
                              @ poses["k2"].R, poses["k2"].t))
    bad = contradicted(off, Ks, matches)
    assert list(bad) == ["k2"] and 0.25 < bad["k2"] < 1.0
    # a pure rotation: any translation direction fits the matches of a correct rotation
    turned = Pose(rot_z(0.2) @ poses["k0"].R, poses["k0"].t)
    uv_turned = project(turned)
    for d in ([0.3, -0.5, 0.8], [-1.0, 0.2, 0.1]):
        posed = Pose(turned.R, turned.t + 1e-3 * np.array(d))
        assert epipolar_deg(poses["k0"], K, posed, K, uv["k0"], uv_turned).max() < 1e-6


def test_incremental_extension_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An extension continues the stored model only (COLMAP's further models restart from the
    images left — the stored ones included — in an unrelated frame), holds the map's cameras and
    is seeded; mapping from scratch keeps COLMAP's defaults."""
    import pycolmap

    from oh_my_slam.mapping import sfm as sfm_mod

    seen: list[object] = []

    def fake(db: str, images: str, out: str, opts: object, input_path: str = "") -> dict:
        seen.append(opts)
        return {}

    monkeypatch.setattr(pycolmap, "incremental_mapping", fake)
    model = tmp_path / "model"
    model.mkdir()
    pycolmap.Reconstruction().write(str(model))
    s = sfm_mod.Sfm(tmp_path / "db.db", tmp_path, tmp_path / "work")
    assert s.map_incremental(tmp_path / "a") is None
    assert s.map_incremental(tmp_path / "b", input_path=model, fix_existing=True,
                             constant_cameras={1}) is None
    scratch, ext = seen
    assert scratch.multiple_models and scratch.random_seed == -1  # type: ignore[attr-defined]
    assert not ext.multiple_models and ext.fix_existing_frames  # type: ignore[attr-defined]
    assert ext.random_seed == sfm_mod.EXTEND_SEED  # type: ignore[attr-defined]
    assert set(ext.constant_cameras) == {1}  # type: ignore[attr-defined]


def test_sift_doubles_photos_but_not_hd_video_keyframes(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Photos keep COLMAP's SIFT (the image doubled for the finest octave: the office map's
    global solution depends on it); the keyframes of an HD video are not doubled (their doubled
    octave was most of the COLMAP stage), those of a small video are (alone they keep few
    features). The mapper says which keyframes come from a video."""
    from oh_my_slam.mapping import sfm as sfm_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(sfm_mod, "_run", lambda args, log: seen.append(args))
    monkeypatch.setattr(sfm_mod.Sfm, "_register", lambda self, names, prior: 1)
    s = sfm_mod.Sfm(tmp_path / "db.db", tmp_path, tmp_path / "work")
    s.extract(["a.jpg"], sfm_mod.CameraPrior(4000, 3000, focal=3000.0))
    s.extract(["b.jpg"], sfm_mod.CameraPrior(1920, 1080), video=True)
    s.extract(["c.jpg"], sfm_mod.CameraPrior(640, 480), video=True)
    first = [a[a.index("--SiftExtraction.first_octave") + 1] for a in seen]
    assert first == ["-1", "0", "-1"]
    assert all(a.count("--SiftExtraction.first_octave") == 1 for a in seen)
    assert all("--FeatureExtraction.max_image_size" not in a for a in seen)  # COLMAP's 3200


def test_images_are_registered_in_keyframe_order_before_extraction(tmp_path: Path) -> None:
    """COLMAP's threaded extraction gives the images it registers ids in the order it finishes
    them (the global mapper's result follows the ids): the mapper registers them first, in the
    keyframes' order, with the camera COLMAP's single_camera would create, and a later batch of
    the same size and focal shares it."""
    import pycolmap

    from oh_my_slam.mapping import sfm as sfm_mod

    s = sfm_mod.Sfm(tmp_path / "db.db", tmp_path, tmp_path / "work")
    names = [f"f{k:06d}.jpg" for k in (0, 1, 2, 10)]
    cam = s._register(names, sfm_mod.CameraPrior(4000, 3000))
    db = pycolmap.Database.open(str(tmp_path / "db.db"))
    try:
        assert [(im.image_id, im.name) for im in sorted(db.read_all_images(),
                                                         key=lambda im: im.image_id)] \
            == list(enumerate(names, start=1))
        c = db.read_camera(cam)
        assert list(c.params) == [4800.0, 2000.0, 1500.0] and not c.has_prior_focal_length
    finally:
        db.close()
    assert s._register(names[:2] + ["f000011.jpg"], sfm_mod.CameraPrior(4000, 3000,
                                                                       existing_id=cam)) == cam
    db = pycolmap.Database.open(str(tmp_path / "db.db"))
    try:
        assert db.num_images() == 5
    finally:
        db.close()


def test_weak_links_of_a_walk_are_the_cuts_few_sequential_pairs_span() -> None:
    """A cut of the capture order that fewer than WEAK_CUT_PAIRS verified pairs within the
    sequential window span is weak (loop closures across it do not count); the listed pairs that
    span it at most WEAK_CUT_REACH apart, and that SIFT did not verify, are matched again."""
    from oh_my_slam.mapping.sfm import WEAK_CUT_PAIRS, WEAK_CUT_REACH, weak_link_pairs

    order = [f"k{i:02d}" for i in range(30)]

    def pair(i: int, j: int) -> frozenset[str]:
        return frozenset((order[i], order[j]))

    window = 4
    listed = {pair(i, j) for i in range(30) for j in range(i + 1, min(30, i + 1 + window))}
    listed |= {pair(2, 25), pair(5, 28)}  # loop candidates
    # every sequential pair verified, except across 14|15, where only (13, 15) is: a weak cut
    verified = {p for p in listed if len(p) == 2 and not (
        min(order.index(n) for n in p) <= 14 < max(order.index(n) for n in p))}
    verified |= {pair(13, 15), pair(2, 25), pair(5, 28)}
    assert WEAK_CUT_PAIRS > 1 and WEAK_CUT_REACH >= 2
    cuts, todo = weak_link_pairs(order, listed, verified, window)
    assert cuts == ["k14"]
    spanning = {pair(i, j) for i in range(11, 15) for j in range(15, 19) if j - i <= window}
    assert todo == spanning - {pair(13, 15)}
    # a shorter reach keeps the nearer pairs only; a strong cut needs nothing
    _, near = weak_link_pairs(order, listed, verified, window, reach=2)
    assert near == {pair(14, 15), pair(14, 16)}
    assert weak_link_pairs(order, listed, verified | {pair(14, 15), pair(12, 16)}, window,
                           min_pairs=2) == ([], set())
    # a keyframe matched to nothing leaves both its cuts weak
    alone = {p for p in verified if order[20] not in p}
    cuts, todo = weak_link_pairs(order, listed, alone, window, min_pairs=1)
    assert cuts == [] and todo == set()  # its neighbours' pairs over it still span both cuts
    cuts, _ = weak_link_pairs(order, listed, alone, 1, min_pairs=1)
    assert cuts == ["k14", "k19", "k20"]


def _colmap_db(path: Path, names: list[str]) -> dict[str, int]:
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        cid = db.write_camera(pycolmap.Camera.create_from_model_name(1, "SIMPLE_PINHOLE", 100.0,
                                                                     64, 48))
        return {n: db.write_image(pycolmap.Image(name=n, camera_id=cid)) for n in names}
    finally:
        db.close()


def _write_pair(path: Path, i: int, j: int, inliers: int, raw: int) -> None:
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        db.write_matches(i, j, np.stack([np.arange(raw)] * 2, 1).astype(np.uint32))
        g = pycolmap.TwoViewGeometry()
        g.config = 2  # calibrated
        g.inlier_matches = np.stack([np.arange(inliers)] * 2, 1).astype(np.uint32)
        db.write_two_view_geometry(i, j, g)
    finally:
        db.close()


def _inliers(path: Path, i: int, j: int) -> int:
    import pycolmap

    db = pycolmap.Database.open(str(path))
    try:
        return len(db.read_two_view_geometry(i, j).inlier_matches)
    finally:
        db.close()


def test_lightglue_rematches_weak_pairs_on_the_sift_keypoints(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch
                                                              ) -> None:
    """COLMAP skips a pair the database holds, so SIFT's rows go before LightGlue (on the CPU,
    on the SIFT keypoints) matches the pairs; a pair where SIFT verified more keeps SIFT's rows,
    and every pair keeps them when LightGlue fails (no model, no network)."""
    import pycolmap

    from oh_my_slam.mapping import sfm as sfm_mod

    db_path = tmp_path / "db.db"
    ids = _colmap_db(db_path, ["a.jpg", "b.jpg", "c.jpg"])
    a, b, c = ids["a.jpg"], ids["b.jpg"], ids["c.jpg"]
    _write_pair(db_path, a, b, inliers=0, raw=4)
    _write_pair(db_path, b, c, inliers=12, raw=30)
    seen: list[list[str]] = []

    def lightglue(args: list[str], log: Path) -> None:
        seen.append(args)
        lst = Path(args[args.index("--match_list_path") + 1]).read_text().split()
        assert sorted(lst) == ["a.jpg", "b.jpg", "b.jpg", "c.jpg"]
        db = pycolmap.Database.open(str(db_path))
        try:
            assert not db.exists_matches(a, b) and not db.exists_two_view_geometry(b, c)
        finally:
            db.close()
        _write_pair(db_path, a, b, inliers=40, raw=50)
        _write_pair(db_path, b, c, inliers=8, raw=20)

    monkeypatch.setattr(sfm_mod, "_run", lightglue)
    s = sfm_mod.Sfm(db_path, tmp_path, tmp_path / "work")
    weak = {frozenset(("a.jpg", "b.jpg")), frozenset(("b.jpg", "c.jpg"))}
    assert s.rematch_lightglue(weak) == {"pairs": 2, "verified_before": 0, "verified_after": 1}
    (args,) = seen
    assert args[0] == "matches_importer"
    assert args[args.index("--FeatureMatching.type") + 1] == "SIFT_LIGHTGLUE"
    assert args[args.index("--FeatureMatching.use_gpu") + 1] == "0"
    assert (_inliers(db_path, a, b), _inliers(db_path, b, c)) == (40, 12)  # SIFT's 12 kept
    assert set(s.verified_pairs()) == {frozenset(("a.jpg", "b.jpg"))}

    def fails(args: list[str], log: Path) -> None:
        raise sfm_mod.SfmError("colmap matches_importer failed (1): download failed")

    monkeypatch.setattr(sfm_mod, "_run", fails)
    _write_pair(db_path, a, c, inliers=5, raw=9)
    res = s.rematch_lightglue({frozenset(("a.jpg", "c.jpg"))})
    assert res == {"pairs": 1, "verified_before": 0, "verified_after": 0}
    assert _inliers(db_path, a, c) == 5
    assert s.rematch_lightglue(set())["pairs"] == 0


def test_the_mapper_rematches_the_weak_links_of_the_new_keyframes() -> None:
    """The new keyframes' capture order (by index, not by the order they are listed in) gives the
    cuts; the result goes to the update's notes. Without a weak link nothing is matched again."""
    from types import SimpleNamespace

    from oh_my_slam.mapping import api

    new = [SimpleNamespace(kf=SimpleNamespace(name=f"f{i:06d}", index=i))
           for i in (5, 0, 7, 1, 2, 6, 3, 4)]
    names = {i: f"f{i:06d}.jpg" for i in range(8)}
    pairs = {(i, j) for i in range(8) for j in range(i + 1, min(8, i + 4))}

    class FakeSfm:
        def __init__(self, verified: set[tuple[int, int]]) -> None:
            self.verified = {frozenset((names[a], names[b])): (50, 2) for a, b in verified}
            self.asked: list[set[frozenset[str]]] = []

        def verified_pairs(self) -> dict[frozenset[str], tuple[int, int]]:
            return self.verified

        def rematch_lightglue(self, weak: set[frozenset[str]]) -> dict[str, int]:
            self.asked.append(weak)
            return {"pairs": len(weak), "verified_before": 0, "verified_after": 1}

    msgs: list[str] = []
    ctx = SimpleNamespace(new=new, notes={})
    # every pair verified except across 3|4, which (2, 4) alone spans
    sfm = FakeSfm({(i, j) for i, j in pairs if not i <= 3 < j} | {(2, 4)})
    api._strengthen_weak_links(ctx, sfm, pairs, names, msgs.append)  # type: ignore[arg-type]
    assert sfm.asked == [{frozenset((names[i], names[j]))
                          for i, j in ((1, 4), (2, 5), (3, 4), (3, 5), (3, 6))}]
    assert ctx.notes["weak_links"] == {"cuts_after": ["f000003.jpg"], "pairs": 5,
                                       "verified_before": 0, "verified_after": 1}
    assert len(msgs) == 1 and "LightGlue" in msgs[0]
    strong = FakeSfm(pairs)
    ctx = SimpleNamespace(new=new, notes={})
    api._strengthen_weak_links(ctx, strong, pairs, names, msgs.append)  # type: ignore[arg-type]
    assert strong.asked == [] and ctx.notes == {}


def test_the_near_far_correction_does_not_change_what_a_keyframe_fuses() -> None:
    """A street keyframe: the road 8 m ahead (the median depth) and a facade 27 m away, fused
    up to 30 m. Its near/far correction (exponent 1.2 about its median) places the facade at
    34.4 m: the keyframe still fuses it (its cut moves with the correction, to 39 m), where a
    fixed 30 m cut would drop it. Uncorrected, the same depth beyond 30 m is not fused."""
    from oh_my_slam.core.types import Intrinsics, Pose
    from oh_my_slam.mapping.geometry import FrameData, fused_cloud_points, fusion_depth_max
    from oh_my_slam.mapping.store import FrameRecord

    K = Intrinsics(40.0, 40.0, 32.0, 24.0, 64, 48)
    raw = np.full((48, 64), 8.0)
    raw[:16] = 27.0  # the facade: the top third of the image
    e, med = 1.2, 8.0
    tilted = med * (raw / med) ** e
    assert tilted[0, 0] == pytest.approx(34.4, abs=0.1) and np.median(tilted) == med

    def frames(depth: np.ndarray, exponent: float) -> list[FrameData]:
        out = []
        for k in range(3):
            T = Pose(np.eye(3), np.array([0.05 * k, 0.0, 0.0]))
            rec = FrameRecord(k, f"f{k:06d}", "", "", 1, 64, 48, K, T, 64, 48,
                              stats={"depth_exponent": exponent})
            out.append(FrameData(rec, depth.astype(np.float32), np.ones(depth.shape, bool),
                                 np.zeros(depth.shape + (3,), np.uint8),
                                 np.zeros(depth.shape, np.int32), True))
        return out

    corrected = frames(tilted, e)
    assert fusion_depth_max(corrected[0], 30.0) == pytest.approx(med * (30.0 / med) ** e)
    assert fusion_depth_max(frames(raw, 1.0)[0], 30.0) == 30.0
    far = fused_cloud_points(corrected, voxel=0.1, depth_max=30.0)
    assert (np.abs(far[:, 2] - 34.4) < 0.3).sum() > 50  # the facade, where the correction put it
    beyond = fused_cloud_points(frames(tilted, 1.0), voxel=0.1, depth_max=30.0)
    assert not (beyond[:, 2] > 30.0).any()  # the same depth uncorrected: beyond the cut


def test_a_reader_reads_again_once_while_a_commit_is_applied(tmp_path: Path) -> None:
    """A file can move from the staging folder into place between ``MapReader.path`` and the
    read while an update applies its commit (a rebuild replaces most files): the reader reads once
    more, and only then."""
    from oh_my_slam.mapping import store

    reader = object.__new__(store.MapReader)
    reader.root = tmp_path
    reader._overlay = set()
    calls: list[int] = []

    def flaky() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("moved")
        return 7

    with pytest.raises(FileNotFoundError):
        reader._again(flaky)  # no commit in progress: the error stands
    (tmp_path / store.STAGING).mkdir()
    (tmp_path / store.STAGING / store.COMMIT).write_text("{}")
    calls.clear()
    assert reader._again(flaky) == 7 and len(calls) == 2
