"""Layering / accuracy metrics of a map's point cloud (read-only; no server, no torch).

    python -m oh_my_slam.tools.cloud_quality --map DIR [--map DIR2 ...]

Per map, one JSON document on stdout:

* ``frame_agreement``: relative depth disagreement between overlapping keyframes (frame i
  back-projected into frame i+gap, visible surfaces only): median and p90 of |z_i→j / z_j - 1|.
  This is what stacks parallel copies of a surface when frames are merged without fusion.
* ``planar_patches``: 10 cm patches of the cloud that are clearly planar: thickness (std along the
  normal) p50/p90 in mm and the share of points more than 1.5 cm off the patch plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.geometry import project
from oh_my_slam.core.ply import read_ply
from oh_my_slam.mapping import store

GAPS = (1, 3, 8, 20)
PATCH_RADIUS = 0.10
OFF_PLANE = 0.015
SEEDS = 6000


def frame_agreement(reader: store.MapReader, gaps: tuple[int, ...] = GAPS, step: int = 3
                    ) -> dict[str, Any]:
    frames = sorted(reader.frames, key=lambda r: r.index)
    cache: dict[int, tuple[NDArray[Any], NDArray[Any]]] = {}

    def data(i: int) -> tuple[NDArray[Any], NDArray[Any]]:
        if i not in cache:
            cache[i] = (reader.depth(frames[i]), reader.valid(frames[i]))
        return cache[i]

    out: dict[str, Any] = {}
    for gap in gaps:
        med, p90 = [], []
        for i in range(0, len(frames) - gap, step):
            (di, vi), (dj, vj) = data(i), data(i + gap)
            ri, rj = frames[i], frames[i + gap]
            v, u = np.nonzero(vi & (di > 0))
            u, v = u[::7], v[::7]
            Ki = ri.K_grid.K()
            z = di[v, u].astype(np.float64)
            pc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
            T = rj.T_map_cam.inverse().compose(ri.T_map_cam)
            uv, zq = project(pc @ T.R.T + T.t, rj.K_grid.K())
            h, w = dj.shape
            with np.errstate(invalid="ignore"):
                uu, vv = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
                ok = (zq > 0.1) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            uu, vv, zq = uu[ok].astype(int), vv[ok].astype(int), zq[ok]
            keep = vj[vv, uu] & (dj[vv, uu] > 0)
            if keep.sum() < 500:
                continue
            r = np.abs(zq[keep] / dj[vv[keep], uu[keep]] - 1)
            r = r[r < 0.3]  # same visible surface only
            med.append(np.median(r))
            p90.append(np.percentile(r, 90))
        if med:
            out[str(gap)] = {"pairs": len(med), "median_pct": round(float(np.median(med)) * 100, 2),
                             "p90_pct": round(float(np.median(p90)) * 100, 2)}
    return out


def planar_patches(xyz: NDArray[Any], radius: float = PATCH_RADIUS, seeds: int = SEEDS
                   ) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    pts = np.asarray(xyz, np.float64)
    if len(pts) < 100:
        return {"patches": 0}
    tree = cKDTree(pts)
    rng = np.random.default_rng(1)
    centres = pts[rng.choice(len(pts), min(seeds, len(pts)), replace=False)]
    thick, off = [], []
    for idx in tree.query_ball_point(centres, radius):
        if len(idx) < 60:
            continue
        q = pts[idx] - pts[idx].mean(0)
        w, V = np.linalg.eigh(q.T @ q / len(idx))
        if w[1] < 9 * w[0]:  # clearly planar patches only (walls, floor, table tops)
            continue
        thick.append(np.sqrt(max(w[0], 0.0)))
        off.append(float((np.abs(q @ V[:, 0]) > OFF_PLANE).mean()))
    if not thick:
        return {"patches": 0}
    t = np.array(thick) * 1000
    return {"patches": len(t), "thickness_p50_mm": round(float(np.median(t)), 2),
            "thickness_p90_mm": round(float(np.percentile(t, 90)), 2),
            "off_plane_pct": round(float(np.mean(off)) * 100, 2)}


def measure(root: Path) -> dict[str, Any]:
    reader = store.MapReader(root)
    xyz = read_ply(reader.path(store.CLOUD_PLY)).xyz if reader.exists(store.CLOUD_PLY) \
        else np.zeros((0, 3), np.float32)
    return {"map": str(reader.root), "keyframes": len(reader.frames), "cloud_points": len(xyz),
            "frame_agreement": frame_agreement(reader), "planar_patches": planar_patches(xyz)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", type=Path, action="append", required=True, help="map folder")
    args = ap.parse_args(argv)
    out = [measure(m) for m in args.map]
    for m in out:
        pp = m["planar_patches"]
        print(f"{Path(m['map']).name}: {m['cloud_points']} pts | patch thickness p50 "
              f"{pp.get('thickness_p50_mm')} mm, off-plane {pp.get('off_plane_pct')} %",
              file=sys.stderr)
    json.dump(out[0] if len(out) == 1 else out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
