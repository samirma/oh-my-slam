"""The bird's-eye reads (map.md §8): ``GET /api/maps/{name}/birdseye.png`` and its metadata
``GET /api/maps/{name}/birdseye`` [``Birdseye``].

Each version is drawn once from what it holds (``derived``): its fused cloud, its objects'
footprints with their ids (the confirmed and moved objects of that version's objects read;
struck ones are not drawn) and its keyframe cameras
(``mapping.birdseye.render``). The image and its metadata are kept with the version's derived
data (``derived/<n>/birdseye.png`` and ``birdseye.json``), so both reads answer the same version
with the same metadata. ``pixel_to_map`` maps a pixel's centre, given by its column and row
indices (u, v), to map x, y: ``[x, y] = A [u, v, 1]``; the PNG's text metadata records it with
``map_version``."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_json
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import birdseye
from oh_my_slam.service.derived import DerivedVersion

PNG = "birdseye.png"
META = "birdseye.json"


def _affine(g: birdseye.Grid) -> tuple[list[list[float]], list[list[float]]]:
    """``pixel_to_map`` for pixel centres by index, and its inverse."""
    s = g.s
    to_map = [[s, 0.0, round(g.x0 + s / 2, 9)], [0.0, -s, round(g.y0 - s / 2, 9)]]
    to_px = [[round(1 / s, 9), 0.0, round(-(g.x0 + s / 2) / s, 9)],
             [0.0, round(-1 / s, 9), round((g.y0 - s / 2) / s, 9)]]
    return to_map, to_px


def _draw(dv: DerivedVersion, objects: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    c = dv.cloud()
    boxes = []
    for o in objects["objects"]:
        if o.get("state") == "struck":  # seen gone: not drawn
            continue
        a = math.radians(float(o["yaw_deg"]))
        R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        boxes.append(birdseye.Footprint(int(o["id"]), tuple(o["color"]),
                                        birdseye.corners(o["centre_m"], o["size_m"], R2)))
    frames = dv.record["frames"]
    cams = [birdseye.Camera(f["name"], "keyframe",
                            Pose.from_matrix(np.asarray(f["T_map_cam"], np.float64)), {})
            for f in frames]
    centres = np.array([np.asarray(f["T_map_cam"], np.float64)[:3, 3] for f in frames])

    def text(g: birdseye.Grid) -> dict[str, str]:
        to_map, to_px = _affine(g)
        return {"map": dv.name, "map_version": str(dv.version),
                "pixel_to_map": json.dumps(to_map, separators=(",", ":")),
                "map_to_pixel": json.dumps(to_px, separators=(",", ":"))}

    png, grid = birdseye.render(c["xyz"], c["rgb"], boxes, cams, centres.reshape(-1, 3),
                                dv.name, text)
    meta = {"map": dv.name, "map_version": dv.version, "width": grid.width,
            "height": grid.height, "pixel_to_map": _affine(grid)[0]}
    return png, meta


def read(dv: DerivedVersion, objects: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """The version's image and its ``Birdseye`` metadata (without ``schema``), drawn once, with
    the version's ``scale_source`` and ``scale_sigma_rel`` (map.md §8)."""
    png_path, meta_path = dv.dir / PNG, dv.dir / META
    scale = {"scale_source": dv.record["scale_source"],
             "scale_sigma_rel": dv.record["scale_sigma_rel"]}
    if meta_path.exists() and png_path.exists():
        out: dict[str, Any] = json.loads(meta_path.read_text())
        return png_path.read_bytes(), {**out, **scale}
    png, meta = _draw(dv, objects)
    atomic_write_bytes(png_path, png)
    atomic_write_json(meta_path, meta)  # last: both exist once it does
    return png, {**meta, **scale}
