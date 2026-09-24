"""The five ``segment.sh -d`` artefacts, written only when an artefact folder is given."""

from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_text
from oh_my_slam.core.images import png_bytes
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.catalog import catalog_csv, catalog_md

ARTIFACT_NAMES = ("segmentation.json", "segmented.png", "catalog.csv", "catalog.md",
                  "segments.ply")


def write_artifacts(
    out_dir: Path,
    scene_json: bytes,
    segmented: NDArray,
    objects: list[SceneObject],
    segments_ply: bytes,
    title: str,
) -> list[Path]:
    """Write exactly the five artefacts; ``scene_json`` and ``segments_ply`` are the exact bytes
    ``-f json`` and ``-f ply`` output for the same run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_dir / "segmentation.json", scene_json)
    atomic_write_bytes(out_dir / "segmented.png", png_bytes(segmented))
    atomic_write_text(out_dir / "catalog.csv", catalog_csv(objects))
    atomic_write_text(out_dir / "catalog.md", catalog_md(objects, title))
    atomic_write_bytes(out_dir / "segments.ply", segments_ply)
    return [out_dir / n for n in ARTIFACT_NAMES]
