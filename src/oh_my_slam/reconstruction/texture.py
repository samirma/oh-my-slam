"""Mesh texturing into a GLB: OpenMVS ``TextureMesh`` (best view per face) with a NumPy fallback
(per-face best-view atlas; Open3D's ``project_images_to_albedo`` is x86_64-only).

Seam levelling is disabled with the OpenMVS 2.4.0 arm64 build: global levelling aborts on TSDF
meshes (``unordered_map::at``) and local levelling paints patches black / saturated colours
(verified on a synthetic room and the ainex map)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import paths
from oh_my_slam.core.geometry import rot_to_quat
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Pose

log = get_logger("oh_my_slam.texture")

EMPTY_COLOR = 0x808080
OPENMVS_TIMEOUT_S = 420


@dataclass
class TextureView:
    image_path: Path  # undistorted pinhole image
    K: NDArray[np.float64]  # 3x3 for this image's resolution
    width: int
    height: int
    T_world_cam: Pose


def openmvs_available() -> bool:
    d = paths.openmvs_dir()
    return (d / "TextureMesh").is_file() and (d / "InterfaceCOLMAP").is_file()


def write_colmap_text(views: list[TextureView], folder: Path) -> None:
    """COLMAP text model (PINHOLE cameras, poses, no points) + symlinked images."""
    sparse = folder / "sparse"
    images = folder / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    cams, imgs = [], []
    for i, v in enumerate(views, start=1):
        fx, fy, cx, cy = v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2]
        cams.append(f"{i} PINHOLE {v.width} {v.height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}")
        T_cw = v.T_world_cam.inverse()
        qx, qy, qz, qw = rot_to_quat(T_cw.R)
        tx, ty, tz = T_cw.t
        name = f"{i:06d}{Path(v.image_path).suffix.lower()}"
        imgs.append(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {i} {name}")
        imgs.append("")
        link = images / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(v.image_path).resolve())
    (sparse / "cameras.txt").write_text("# Camera list\n" + "\n".join(cams) + "\n")
    (sparse / "images.txt").write_text("# Image list\n" + "\n".join(imgs) + "\n")
    (sparse / "points3D.txt").write_text("# 3D point list\n")


def run_openmvs(tool: str, args: list[str], work: Path) -> bool:
    """Run one OpenMVS tool with ``work`` as both its working folder (``-w``) and its current
    directory: OpenMVS writes a ``<Tool>-<id>.log`` into the working folder (the current directory
    by default), so nothing lands in the caller's directory (e.g. the repository)."""
    work.mkdir(parents=True, exist_ok=True)
    cmd = [str(paths.openmvs_dir() / tool), *args, "-w", str(work)]
    res = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                         timeout=OPENMVS_TIMEOUT_S)
    if res.returncode != 0:
        log.warning("%s failed (%d): %s", tool, res.returncode, (res.stdout + res.stderr)[-800:])
        return False
    return True


def texture_openmvs(mesh: Any, views: list[TextureView], out_glb: Path, work: Path) -> bool:
    import open3d as o3d

    work.mkdir(parents=True, exist_ok=True)
    colmap = work / "colmap"
    write_colmap_text(views, colmap)
    mesh_ply = work / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_ascii=False)
    scene = work / "scene.mvs"
    textured = work / "textured.glb"
    ok = run_openmvs("InterfaceCOLMAP", [
        "-i", str(colmap), "-o", str(scene), "--image-folder", str(colmap / "images") + "/",
        "-v", "0"], work)
    ok = ok and run_openmvs("TextureMesh", [
        str(scene), "--mesh-file", str(mesh_ply), "-o", str(textured),
        "--export-type", "glb", "--empty-color", str(EMPTY_COLOR), "--decimate", "1",
        "--close-holes", "0", "--resolution-level", "0", "--max-texture-size", "8192",
        "--global-seam-leveling", "0", "--local-seam-leveling", "0",
        "-v", "0", "--process-priority", "0",
    ], work)
    if not ok or not textured.is_file():
        return False
    return embed_glb(textured, out_glb)


def embed_glb(src: Path, out_glb: Path) -> bool:
    """Re-export a GLB whose textures are external files (OpenMVS writes ``*_0.png`` next to it)
    as a self-contained GLB."""
    import trimesh

    scene = trimesh.load(str(src), force="scene")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_glb.with_suffix(".tmp.glb")
    scene.export(str(tmp), file_type="glb")
    shutil.move(str(tmp), str(out_glb))
    return out_glb.is_file()


def _face_view_choice(verts: NDArray[Any], faces: NDArray[Any], views: list[TextureView],
                      margin: float = 2.0) -> NDArray[np.int64]:
    """Best visible view per face (most frontal and closest), -1 when no view sees it."""
    import open3d as o3d

    tri = verts[faces]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    best = np.full(len(faces), -1, np.int64)
    best_score = np.zeros(len(faces))
    for vi, v in enumerate(views):
        T_cw = v.T_world_cam.inverse()
        pc = centroids @ T_cw.R.T + T_cw.t
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = v.K[0, 0] * pc[:, 0] / z + v.K[0, 2]
            vv = v.K[1, 1] * pc[:, 1] / z + v.K[1, 2]
        to_cam = v.T_world_cam.t[None, :] - centroids
        dist = np.linalg.norm(to_cam, axis=1)
        cos = np.abs((normals * to_cam).sum(1)) / (dist + 1e-12)
        cand = (z > 0.05) & (u >= margin) & (u < v.width - margin) & (vv >= margin) & (
            vv < v.height - margin) & (cos > 0.2)
        if not cand.any():
            continue
        idx = np.nonzero(cand)[0]
        dirs = -to_cam[idx] / dist[idx, None]
        rays = np.concatenate([np.repeat(v.T_world_cam.t[None], len(idx), 0), dirs], 1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        visible = hit >= dist[idx] * 0.995 - 0.01
        score = cos[idx] / np.maximum(z[idx], 1e-3)
        better = visible & (score > best_score[idx])
        best[idx[better]] = vi
        best_score[idx[better]] = score[better]
    return best


def texture_atlas(mesh: Any, views: list[TextureView], out_glb: Path, max_views: int = 80,
                  max_tex: int = 8192) -> bool:
    """Fallback texturing: every face gets a half-cell of a regular atlas filled from its best
    visible view (nearest-best-view per face, no seam levelling)."""
    from oh_my_slam.core.images import load_rgb

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(faces) == 0 or not views:
        return False
    if len(views) > max_views:
        keep = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in keep]
    choice = _face_view_choice(verts, faces, views)
    n_cells = int(np.ceil(len(faces) / 2))
    per_row = int(np.ceil(np.sqrt(n_cells)))
    c = int(np.clip(max_tex // per_row, 2, 16))
    size = per_row * c
    atlas = np.full((size, size, 3), 128, np.uint8)
    # texel centres of one cell and the barycentrics of the two half-cell triangles
    ty, tx = np.mgrid[0:c, 0:c] + 0.5
    lower = (tx + ty) <= c  # upper-left triangle: corners (0,0), (c,0), (0,c)
    a_l = np.stack([1 - (tx + ty) / c, tx / c, ty / c], -1)
    a_u = np.stack([(tx + ty) / c - 1, 1 - tx / c, 1 - ty / c], -1)  # corners (c,c),(0,c),(c,0)
    bary = np.where(lower[..., None], a_l, a_u).reshape(-1, 3)
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(1, keepdims=True)
    half = lower.reshape(-1)
    fid = np.arange(len(faces))
    cell = fid // 2
    upper_tri = fid % 2 == 1
    cx0 = (cell % per_row) * c
    cy0 = (cell // per_row) * c
    uvs = np.zeros((len(faces), 3, 2))
    corners_l = np.array([[0, 0], [c, 0], [0, c]], float)
    corners_u = np.array([[c, c], [0, c], [c, 0]], float)
    corners = np.where(upper_tri[:, None, None], corners_u[None], corners_l[None])
    uvs[..., 0] = (cx0[:, None] + corners[..., 0]) / size
    uvs[..., 1] = (cy0[:, None] + corners[..., 1]) / size
    for vi, v in enumerate(views):
        sel = np.nonzero(choice == vi)[0]
        if len(sel) == 0:
            continue
        img = load_rgb(v.image_path)
        sx, sy = img.shape[1] / v.width, img.shape[0] / v.height
        T_cw = v.T_world_cam.inverse()
        for start in range(0, len(sel), 20000):
            fs = sel[start:start + 20000]
            tri = verts[faces[fs]]  # (n, 3, 3)
            texel_mask = np.where(upper_tri[fs][:, None], ~half[None], half[None])  # (n, c*c)
            pts = np.einsum("tk,nkd->ntd", bary, tri)  # (n, c*c, 3)
            pc = pts @ T_cw.R.T + T_cw.t
            z = np.maximum(pc[..., 2], 1e-6)
            u = np.clip((v.K[0, 0] * pc[..., 0] / z + v.K[0, 2]) * sx, 0, img.shape[1] - 1)
            w = np.clip((v.K[1, 1] * pc[..., 1] / z + v.K[1, 2]) * sy, 0, img.shape[0] - 1)
            col = img[np.rint(w).astype(int), np.rint(u).astype(int)]  # (n, c*c, 3)
            ay = (cy0[fs][:, None] + ty.reshape(-1)[None] - 0.5).astype(int)
            ax = (cx0[fs][:, None] + tx.reshape(-1)[None] - 0.5).astype(int)
            atlas[ay[texel_mask], ax[texel_mask]] = col[texel_mask]
    return write_textured_glb(verts, faces, uvs, atlas, out_glb)


def write_textured_glb(verts: NDArray[Any], faces: NDArray[Any], tri_uvs: NDArray[Any],
                       texture: NDArray[np.uint8], out_glb: Path) -> bool:
    """GLB with per-corner UVs (vertices unrolled per triangle corner) and an sRGB texture."""
    import trimesh
    from PIL import Image

    corners = verts[faces].reshape(-1, 3)
    new_faces = np.arange(len(corners)).reshape(-1, 3)
    uv = tri_uvs.reshape(-1, 2).copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF v axis points down the image
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture), metallicFactor=0.0, roughnessFactor=1.0,
        doubleSided=True,
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    tm = trimesh.Trimesh(vertices=corners, faces=new_faces, visual=visual, process=False)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(out_glb), file_type="glb")
    return out_glb.is_file()


def write_vertex_color_glb(mesh: Any, out_glb: Path) -> bool:
    """Last resort: untextured GLB with the fused vertex colours."""
    import trimesh

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    c = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
    tm = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(out_glb), file_type="glb")
    return out_glb.is_file()


def texture_mesh(mesh: Any, views: list[TextureView], out_glb: Path, work: Path,
                 prefer_openmvs: bool = True) -> str:
    """Texture ``mesh`` into ``out_glb``; returns the method used."""
    if prefer_openmvs and openmvs_available():
        try:
            if texture_openmvs(mesh, views, out_glb, work / "openmvs"):
                return "openmvs"
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("OpenMVS texturing failed: %s", exc)
    try:
        if texture_atlas(mesh, views, out_glb):
            return "atlas"
    except Exception as exc:
        log.warning("atlas texturing failed: %s", exc)
    write_vertex_color_glb(mesh, out_glb)
    return "vertex-colors"
