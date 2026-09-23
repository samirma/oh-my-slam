"""Mesh clean-up: degenerate/duplicate removal, small components, decimation."""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_FACES = 1_000_000


def clean_mesh(mesh: Any, min_component_fraction: float = 0.002, min_component_faces: int = 50,
               max_faces: int = MAX_FACES) -> Any:
    """Return a cleaned copy of a legacy Open3D triangle mesh."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    n = len(m.triangles)
    if n == 0:
        return m
    raw_clusters, raw_counts, _ = m.cluster_connected_triangles()
    clusters = np.asarray(raw_clusters)
    counts = np.asarray(raw_counts)
    keep_min = max(min_component_faces, int(min_component_fraction * n))
    small = counts[clusters] < keep_min
    if small.any() and not small.all():
        m.remove_triangles_by_mask(small)
        m.remove_unreferenced_vertices()
    if len(m.triangles) > max_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=max_faces)
        m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    c = np.asarray(mesh.vertex_colors, dtype=np.float64) if mesh.has_vertex_colors() else None
    return v, f, c


def remove_faces_near(mesh: Any, points: np.ndarray, radius: float) -> Any:
    """Drop faces whose centroid is within ``radius`` of any point (used for removed content)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    if len(points) == 0 or len(mesh.triangles) == 0:
        return mesh
    v, f, _ = mesh_arrays(mesh)
    centroids = v[f].mean(axis=1)
    d, _ = cKDTree(points).query(centroids, k=1, distance_upper_bound=radius)
    mask = np.isfinite(d)
    m = o3d.geometry.TriangleMesh(mesh)
    if mask.any():
        m.remove_triangles_by_mask(mask)
        m.remove_unreferenced_vertices()
    return m
