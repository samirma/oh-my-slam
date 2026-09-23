"""The object catalogue: ``catalog.csv`` (exact header) and ``catalog.md`` (by volume, desc)."""

from __future__ import annotations

import csv
import io
from typing import Any

from oh_my_slam.segmentation.api import SceneObject

CSV_HEADER = (
    "id", "label", "score", "color_hex", "width_m", "height_m", "depth_m", "volume_m3",
    "center_x", "center_y", "center_z", "pixel_count", "point_count",
)


def catalog_rows(objects: list[SceneObject]) -> list[dict[str, Any]]:
    rows = []
    for o in sorted(objects, key=lambda x: x.id):
        w, d, h = (float(v) for v in o.obb.size)
        cx, cy, cz = (float(v) for v in o.obb.center)
        rows.append({
            "id": o.id, "label": o.label, "score": round(o.score, 4), "color_hex": o.color_hex,
            "width_m": round(w, 3), "height_m": round(h, 3), "depth_m": round(d, 3),
            "volume_m3": round(w * d * h, 4),
            "center_x": round(cx, 3), "center_y": round(cy, 3), "center_z": round(cz, 3),
            "pixel_count": o.pixel_count, "point_count": o.point_count,
        })
    return rows


def catalog_csv(objects: list[SceneObject]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_HEADER), lineterminator="\n")
    writer.writeheader()
    for row in catalog_rows(objects):
        writer.writerow(row)
    return buf.getvalue()


def catalog_md(objects: list[SceneObject], title: str = "Object catalogue") -> str:
    rows = sorted(catalog_rows(objects), key=lambda r: (-r["volume_m3"], r["id"]))
    lines = [
        f"# {title}",
        "",
        f"{len(rows)} objects, ordered by descending volume. Dimensions in metres.",
        "",
        "| swatch | id | label | score | color_hex | width_m | height_m | depth_m | volume_m3 "
        "| center (x, y, z) | pixels | points |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        swatch = f'<span style="color:{r["color_hex"]}">&#9632;</span>'
        lines.append(
            f"| {swatch} | {r['id']} | {r['label']} | {r['score']:.3f} | `{r['color_hex']}` "
            f"| {r['width_m']:.3f} | {r['height_m']:.3f} | {r['depth_m']:.3f} "
            f"| {r['volume_m3']:.4f} | ({r['center_x']:.2f}, {r['center_y']:.2f}, "
            f"{r['center_z']:.2f}) | {r['pixel_count']} | {r['point_count']} |"
        )
    return "\n".join(lines) + "\n"
