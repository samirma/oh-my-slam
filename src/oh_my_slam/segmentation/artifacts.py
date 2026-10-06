"""The four ``segment.sh -d`` artefacts, written only when an artefact folder is given."""

from __future__ import annotations

from pathlib import Path

from oh_my_slam.core.atomic import atomic_write_bytes, atomic_write_text
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.catalog import catalog_csv, catalog_md

ARTIFACT_NAMES = ("segmentation.json", "segmented.png", "catalog.csv", "catalog.md")


def write_artifacts(
    out_dir: Path,
    scene_json: bytes,
    segmented_png: bytes,
    objects: list[SceneObject],
    title: str,
) -> list[Path]:
    """Write exactly the four artefacts; ``scene_json`` and ``segmented_png`` are the exact bytes
    ``-f json`` and ``-f png`` output for the same run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_dir / "segmentation.json", scene_json)
    atomic_write_bytes(out_dir / "segmented.png", segmented_png)
    atomic_write_text(out_dir / "catalog.csv", catalog_csv(objects))
    atomic_write_text(out_dir / "catalog.md", catalog_md(objects, title))
    return [out_dir / n for n in ARTIFACT_NAMES]
