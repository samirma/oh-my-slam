"""Ownership rules (AC20): import-linter contracts plus source checks for who calls which
endpoint, who fits OBBs and who assigns colours."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "oh_my_slam"


def _py_files(sub: str = "") -> list[Path]:
    return sorted((SRC / sub).rglob("*.py"))


def _grep(pattern: str, files: list[Path]) -> set[str]:
    rx = re.compile(pattern)
    hits = set()
    for f in files:
        if rx.search(f.read_text("utf-8")):
            hits.add(str(f.relative_to(SRC)))
    return hits


def test_import_linter_contracts() -> None:
    exe = Path(sys.executable).with_name("lint-imports")
    res = subprocess.run([str(exe)], capture_output=True, text=True, cwd=SRC.parents[1])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "broken" in res.stdout and " 0 broken" in res.stdout


def test_palette_defined_only_in_segmentation_colors() -> None:
    hits = _grep(r"(?i)#e6194b|0xe6194b", _py_files())
    assert hits <= {"segmentation/colors.py"}, hits
    defs = _grep(r"def color_for_id|def color_hex_for_id", _py_files())
    assert defs <= {"segmentation/colors.py"}, defs


def test_obb_fitting_only_in_segmentation_obb() -> None:
    hits = _grep(r"minAreaRect|def fit_obb|def fit_upright_obb|convex_hull", _py_files())
    assert hits <= {"segmentation/obb.py"}, hits


def test_endpoint_callers() -> None:
    """Geometry/gravity/multiview calls live in reconstruction; segment calls in segmentation."""
    files = [f for f in _py_files() if "server" not in f.parts and "client" not in f.parts]
    assert _grep(r"\.geometry\(", files) <= {"reconstruction/api.py", "reconstruction/multiview.py"}
    assert _grep(r"\.gravity\(", files) <= {"reconstruction/api.py", "reconstruction/gravity.py"}
    assert _grep(r"\.multiview\(", files) <= {"reconstruction/multiview.py"}
    assert _grep(r"\.segment_image\(", files) <= {"segmentation/detect.py"}
    for route in ("/v1/geometry", "/v1/gravity", "/v1/segment", "/v1/multiview"):
        owners = _grep(re.escape(route), _py_files())
        assert owners <= {"client/client.py", "server/app.py", "client/protocol.py"}, owners


def test_import_contracts_express_the_ownership_rules() -> None:
    """The §4 rules are import-linter contracts, not only conventions."""
    import tomllib

    cfg = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    contracts = {c["name"]: c for c in cfg["tool"]["importlinter"]["contracts"]}
    by_source: dict[str, set[str]] = {}
    for c in contracts.values():
        if c["type"] == "forbidden":
            for s in c["source_modules"]:
                by_source.setdefault(s, set()).update(c["forbidden_modules"])
    seg_internals = {f"oh_my_slam.segmentation.{m}" for m in ("detect", "lift", "obb", "colors")}
    assert seg_internals <= by_source["oh_my_slam.mapping"]
    assert seg_internals <= by_source["oh_my_slam.viewer"]
    assert "oh_my_slam.client" in by_source["oh_my_slam.mapping"]
    assert {"oh_my_slam.segmentation", "oh_my_slam.mapping"} <= by_source[
        "oh_my_slam.reconstruction"]
    for pkg in ("mapping", "segmentation", "viewer", "server"):
        assert "open3d" in by_source[f"oh_my_slam.{pkg}"], pkg


def test_mapping_delegates_depth_and_segmentation() -> None:
    """Mapping gets keyframe depth from ``reconstruction.api`` and detections, lifting, boxes and
    colours from ``segmentation.api``; it never fits, lifts, detects or colours by itself."""
    files = _py_files("mapping")
    assert _grep(r"reconstruct_image\(", files) == {"mapping/api.py"}
    assert _grep(r"detect_alongside\(", files) == {"mapping/api.py"}
    assert _grep(r"lift_detections\(", files) == {"mapping/objects.py"}
    assert _grep(r"fit_object_obb\(", files) == {"mapping/objects.py"}
    assert not _grep(r"\bfit_obb\(|\blift_mask\(|\bdetect\(|color(_hex)?_for_id\(|"
                     r"segmentation\.(detect|lift|obb|colors)\b", files)
    # segmentation owns lifting and colours; it does not know the mapper's keyframe settings
    seg = _py_files("segmentation")
    assert not _grep(r"KEYFRAME_TOKENS|KEYFRAME_GRID_SIDE|want_descriptor", seg)



# view.sh owns only the web server and UI (spec §2.5). It may call the owners' public APIs — one
# reconstruction + segmentation run (segmentation.api), the read-only map export, the shared
# point-cloud derivation (segmentation.cloud) and attribute definition (core.cloud_attrs) — but never
# the internals behind them, and it re-implements none of their logic.
VIEWER_FORBIDDEN_IMPORTS = (
    r"import torch|oh_my_slam\.client|oh_my_slam\.server"
    r"|segmentation\.(colors|obb|lift|detect)"
    r"|reconstruction\.(pointcloud|fusion|depth|multiview)"
    r"|mapping\.(api|objects|sfm|geometry|fusion)"
)
VIEWER_REIMPLEMENTATION = (
    r"unproject|pixel_mask|depth_edge_mask|depth_normals|point_normals|voxel_keys"
    r"|voxel_downsample|fit_obb|fit_upright_obb|color_for_id|color_hex_for_id|segment_colors"
    r"|height_colors|PALETTE|np\.random|default_rng|K\.fx|\.K\(\)"
)


def test_viewer_only_calls_the_owners_apis() -> None:
    files = _py_files("viewer") + [SRC / "cli" / "view.py"]
    assert not _grep(VIEWER_FORBIDDEN_IMPORTS, files)
    assert not _grep(VIEWER_REIMPLEMENTATION, files)
    bundle = (SRC / "viewer" / "bundle.py").read_text("utf-8")
    # every displayed cloud is the shared derivation, controlled by the shared attribute set
    for call in ("derive_cloud(", "applicable(", "parse_cloud_attrs(", "image_cloud_source(",
                 "reader_source(", "segment_frame(", "single_image_scene(", "scene_bytes("):
        assert call in bundle, call


def test_viewer_page_has_no_palette_or_randomness() -> None:
    """Colours reach the page only as data (scene JSON, derived clouds); nothing random."""
    from oh_my_slam.segmentation.colors import PALETTE_HEX, UNSEGMENTED_HEX

    static = SRC / "viewer" / "static"
    pages = [p for p in static.rglob("*") if p.suffix in (".js", ".html", ".css")
             and "vendor" not in p.relative_to(static).parts]
    assert pages
    for p in pages:
        text = p.read_text("utf-8").lower()
        assert not [h for h in (*PALETTE_HEX, UNSEGMENTED_HEX) if h in text], p
        assert "math.random" not in text, p


@pytest.mark.parametrize("script", ["reconstruct.sh", "mapper.sh", "segment.sh", "view.sh",
                                    "start_inference_server.sh"])
def test_entry_scripts_are_thin_and_never_call_each_other(script: str) -> None:
    root = SRC.parents[1]
    text = (root / script).read_text()
    assert "oms_exec" in text
    for other in ("reconstruct.sh", "mapper.sh", "segment.sh", "view.sh"):
        if other != script:
            assert other not in text
    assert (root / script).stat().st_mode & 0o111
