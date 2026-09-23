"""Point clouds, gravity refinement, depth alignment, fusion, mesh clean-up and texturing on
synthetic scenes (no server)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.geometry import angle_between_deg, rotation_between
from oh_my_slam.core.images import save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction import depth as dmod
from oh_my_slam.reconstruction.fusion import TsdfFusion, choose_voxel_size
from oh_my_slam.reconstruction.gravity import GravityEstimate, mean_up, refine_with_floor
from oh_my_slam.reconstruction.mesh import clean_mesh, remove_faces_near
from oh_my_slam.reconstruction.pointcloud import cloud_mask, frame_cloud
from oh_my_slam.reconstruction.texture import (
    TextureView,
    openmvs_available,
    run_openmvs,
    texture_mesh,
    texture_openmvs,
    write_colmap_text,
)
from tests.synth.scene import default_room, look_at, orbit_poses, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def test_frame_cloud_colours_and_edges() -> None:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    r = render(room, pose, K)
    m = cloud_mask(r.depth, r.depth > 0)
    assert m.sum() > 0.8 * (r.depth > 0).sum()
    cloud, idx = frame_cloud(r.depth, r.rgb, K, m, pose)
    np.testing.assert_array_equal(cloud.rgb, r.rgb.reshape(-1, 3)[idx])
    # floor pixels unproject to z ~ 0 in the map frame
    floor = (r.ids.reshape(-1)[idx] == 0)
    assert np.abs(cloud.xyz[floor, 2]).max() < 0.02
    with pytest.raises(ValueError):
        frame_cloud(r.depth, r.rgb[:10], K, m)


def test_gravity_floor_refinement_within_5_degrees(rng: np.random.Generator) -> None:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.3]))
    r = render(room, pose, K)
    cloud, _ = frame_cloud(r.depth, r.rgb, K, r.depth > 0)
    true_up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    # a prior 3 degrees off
    tilt = rotation_between(true_up, true_up + np.array([0.05, 0.0, 0.0]))
    prior = GravityEstimate(tilt @ true_up, "geocalib", 1.0, 1.5)
    est = refine_with_floor(cloud.xyz, prior)
    assert est.source == "geocalib+floor"
    assert angle_between_deg(est.up_cam, true_up) < 0.5
    assert est.floor_height == pytest.approx(1.5, abs=0.03)
    # a prior 20 degrees off is kept (floor normal outside the 5 degree window)
    far = rotation_between(true_up, true_up + np.array([0.4, 0.0, 0.0])) @ true_up
    kept = refine_with_floor(cloud.xyz, GravityEstimate(far, "geocalib"))
    assert kept.source == "geocalib"
    assert refine_with_floor(cloud.xyz[:10], prior) is prior
    d = est.to_dict()
    assert GravityEstimate.from_dict(d).floor_inliers == est.floor_inliers
    assert est.confidence > prior.confidence
    up = mean_up([np.array([0, 0, 1.0]), np.array([0, 0.1, 1.0])], [1.0, 1.0])
    assert angle_between_deg(up, [0, 0.05, 1]) < 0.5


def test_depth_scale_fit_with_noise(rng: np.random.Generator) -> None:
    ref = rng.uniform(1, 5, 500)
    pred = ref / 1.37 * rng.normal(1, 0.02, 500)
    pred[:40] *= 3  # outliers
    fit = dmod.fit_frame_scale(pred, ref)
    assert fit.ok and fit.scale == pytest.approx(1.37, rel=0.01)
    assert fit.spread < 0.1
    bad = dmod.fit_frame_scale(pred[:10], ref[:10])
    assert not bad.ok
    s, spread = dmod.global_scale([1.0, 1.1, 0.9, float("nan")])
    assert s == pytest.approx(1.0) and spread > 0
    assert np.isnan(dmod.global_scale([])[0])
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = dmod.sample_depth(d, np.array([[1.2, 0.9], [9, 9], [0, 0]]))
    assert got[0] == 5 and np.isnan(got[1]) and np.isnan(got[2])


def _fused_room(n: int = 12) -> tuple[TsdfFusion, list, list[Pose]]:
    room = default_room()
    poses = orbit_poses(n)
    renders = [render(room, p, K) for p in poses]
    fusion = TsdfFusion(voxel_size=0.02, depth_max=6.0)
    for r, p in zip(renders, poses, strict=True):
        fusion.integrate(r.depth, r.rgb, K.K(), p)
    return fusion, renders, poses


def test_fusion_mesh_and_speed() -> None:
    fusion, _, _ = _fused_room()
    mesh = fusion.extract_mesh()
    assert len(mesh.triangles) > 1000
    v = np.asarray(mesh.vertices)
    # geometry lies inside the room
    assert v[:, 2].min() > -0.1 and v[:, 2].max() < 2.7
    cols = np.asarray(mesh.vertex_colors)
    assert 0.1 < cols.mean() < 0.9
    pts, pcols = fusion.extract_points()
    assert len(pts) > 1000 and len(pcols) == len(pts)
    assert fusion.stats.frames == 12
    assert choose_voxel_size(1.0) == 0.01 and choose_voxel_size(3.0) == 0.015
    assert choose_voxel_size(50.0) == 0.04
    fusion.integrate(np.zeros((240, 320), np.float32), np.zeros((240, 320, 3), np.uint8), K.K(),
                     Pose.identity())
    assert fusion.stats.frames == 12  # empty depth skipped


def test_mesh_cleanup_removes_small_components() -> None:
    import open3d as o3d

    big = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=40)
    small = o3d.geometry.TriangleMesh.create_sphere(0.05, resolution=3).translate((5, 0, 0))
    m = clean_mesh(big + small, min_component_faces=100)
    assert np.asarray(m.vertices)[:, 0].max() < 1.5
    m2 = clean_mesh(big, max_faces=500)
    assert len(m2.triangles) <= 500
    cut = remove_faces_near(big, np.array([[1.0, 0.0, 0.0]]), 0.3)
    assert len(cut.triangles) < len(big.triangles)
    assert len(remove_faces_near(big, np.zeros((0, 3)), 0.3).triangles) == len(big.triangles)


@pytest.mark.parametrize("prefer_openmvs", [False, True])
def test_texturing_produces_a_textured_glb(tmp_path: Path, prefer_openmvs: bool,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    import trimesh

    if prefer_openmvs and not openmvs_available():
        pytest.skip("OpenMVS binary not installed (scripts/install_tools.sh)")
    caller = tmp_path / "caller"  # OpenMVS must not leave its logs in the caller's directory
    caller.mkdir()
    monkeypatch.chdir(caller)
    fusion, renders, poses = _fused_room(8)
    mesh = clean_mesh(fusion.extract_mesh(), max_faces=60000)
    views = []
    for i, (r, p) in enumerate(zip(renders, poses, strict=True)):
        path = tmp_path / f"img{i}.jpg"
        save_jpeg(r.rgb, path)
        views.append(TextureView(path, K.K(), K.width, K.height, p))
    out = tmp_path / "mesh.glb"
    t0 = time.perf_counter()
    method = texture_mesh(mesh, views, out, tmp_path / "work", prefer_openmvs=prefer_openmvs)
    elapsed = time.perf_counter() - t0
    assert method == ("openmvs" if prefer_openmvs else "atlas"), method
    scene = trimesh.load(out)
    geom = next(iter(scene.geometry.values())) if hasattr(scene, "geometry") else scene
    img = np.asarray(geom.visual.material.baseColorTexture.convert("RGB"))
    assert img.std() > 10  # non-uniform texture (AC13)
    assert elapsed < 120
    assert list(caller.iterdir()) == []
    if prefer_openmvs:
        assert sorted(p.name.split("-")[0] for p in (tmp_path / "work").rglob("*.log")) == [
            "InterfaceCOLMAP", "TextureMesh"]


_FAKE_OPENMVS = """#!/bin/sh
# emulates OpenMVS: the log goes to the -w working folder, else to the current directory
dir="$PWD"
while [ $# -gt 0 ]; do [ "$1" = "-w" ] && dir="$2"; shift; done
echo "$PWD" > "$dir/$(basename "$0")-fake.log"
"""


def test_openmvs_runs_in_its_work_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenMVS tools run with the texturing work folder as ``-w`` and as current directory, so
    their ``*.log`` files never land in the caller's directory (e.g. the repository root)."""
    import open3d as o3d

    fake = tmp_path / "openmvs"
    fake.mkdir()
    for tool in ("InterfaceCOLMAP", "TextureMesh"):
        (fake / tool).write_text(_FAKE_OPENMVS)
        (fake / tool).chmod(0o755)
    monkeypatch.setenv("OH_MY_SLAM_OPENMVS_DIR", str(fake))
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    work = tmp_path / "work"
    assert run_openmvs("InterfaceCOLMAP", ["-i", "x"], work)
    assert run_openmvs("TextureMesh", ["scene.mvs", "-v", "0"], work)
    assert list(caller.iterdir()) == []
    logs = sorted(work.glob("*.log"))
    assert [p.name for p in logs] == ["InterfaceCOLMAP-fake.log", "TextureMesh-fake.log"]
    assert all(p.read_text().strip() == str(work.resolve()) for p in logs)  # cwd = work folder
    img = tmp_path / "a.jpg"
    save_jpeg(np.zeros((10, 10, 3), np.uint8), img)
    mesh = o3d.geometry.TriangleMesh.create_box()
    # the fake produces no GLB: the OpenMVS path reports failure, nothing written to the caller
    assert not texture_openmvs(mesh, [TextureView(img, K.K(), 320, 240, Pose.identity())],
                               tmp_path / "m.glb", tmp_path / "w2")
    assert list(caller.iterdir()) == []
    assert len(list((tmp_path / "w2").glob("*.log"))) == 2


def test_colmap_text_export(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    save_jpeg(np.zeros((10, 10, 3), np.uint8), img)
    write_colmap_text([TextureView(img, K.K(), 320, 240, Pose.identity())], tmp_path / "c")
    cams = (tmp_path / "c/sparse/cameras.txt").read_text()
    assert "PINHOLE 320 240" in cams
    lines = (tmp_path / "c/sparse/images.txt").read_text().splitlines()
    assert lines[1].startswith("1 1.000000000 0.000000000")
    assert (tmp_path / "c/images/000001.jpg").is_symlink()
