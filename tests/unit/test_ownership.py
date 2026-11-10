"""Ownership rules (high_level_spec.md §10): import-linter contracts plus source checks for who
calls which endpoint, who fits OBBs and who assigns colours. The HTTP service is the top: it uses
the mapping, reconstruction and segmentation APIs, and nothing imports it."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

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
    """The ownership rules are import-linter contracts, not only conventions."""
    import tomllib

    cfg = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    contracts = {c["name"]: c for c in cfg["tool"]["importlinter"]["contracts"]}
    by_source: dict[str, set[str]] = {}
    for c in contracts.values():
        if c["type"] == "forbidden":
            for s in c["source_modules"]:
                by_source.setdefault(s, set()).update(c["forbidden_modules"])
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert packages == {"client", "core", "mapping", "reconstruction", "segmentation", "server",
                        "service"}
    # no contract names a package that no longer exists
    named = {m for c in contracts.values() for k in ("source_modules", "forbidden_modules")
             for m in c.get(k, []) if m.startswith("oh_my_slam.")}
    named |= {m.strip("() ") for c in contracts.values() for layer in c.get("layers", [])
              for m in layer.split("|")}
    assert {m.split(".")[1] for m in named} <= packages, named
    seg_internals = {f"oh_my_slam.segmentation.{m}" for m in ("detect", "lift", "obb", "colors")}
    assert seg_internals <= by_source["oh_my_slam.mapping"]
    assert "oh_my_slam.client" in by_source["oh_my_slam.mapping"]
    assert {"oh_my_slam.segmentation", "oh_my_slam.mapping"} <= by_source[
        "oh_my_slam.reconstruction"]
    assert "oh_my_slam.core.ply" in by_source["oh_my_slam.segmentation"]  # it writes no PLY
    # Open3D (depth fusion) lives in reconstruction and mapping's geometry
    for pkg in ("segmentation", "service", "server", "client", "core"):
        assert "open3d" in by_source[f"oh_my_slam.{pkg}"], pkg
    for pkg in ("service", "mapping", "segmentation", "reconstruction", "client", "core"):
        assert {"torch", "ultralytics"} <= by_source[f"oh_my_slam.{pkg}"], pkg
    # the service is the top: nothing imports it
    for pkg in packages - {"service"}:
        assert "oh_my_slam.service" in by_source[f"oh_my_slam.{pkg}"], pkg
    # the pipeline never imports the inference server; the server owns models, no pipeline logic
    for pkg in ("mapping", "segmentation", "reconstruction", "client", "core"):
        assert "oh_my_slam.server" in by_source[f"oh_my_slam.{pkg}"], pkg
    assert {"oh_my_slam.mapping", "oh_my_slam.segmentation",
            "oh_my_slam.reconstruction"} <= by_source["oh_my_slam.server"]
    # the service reuses the inference server's lifecycle helpers only
    server_internals = {f"oh_my_slam.server.{m}" for m in ("app", "main", "gpu_worker", "models")}
    assert server_internals <= by_source["oh_my_slam.service"]
    assert "oh_my_slam.server.lifecycle" not in by_source["oh_my_slam.service"]


def test_point_clouds_are_reconstructions_and_colours_segmentations() -> None:
    """§4: the reconstruction code alone derives every emitted cloud (pixel selection, voxel
    thinning, normals, PLY), which segmentation, mapping and the viewer reuse; the segmentation
    code alone assigns colours, which the derivation receives as data and never names."""
    defs = _grep(r"def (derive_cloud|derive_thinned|cloud_ply)\(|"
                 r"class (ImageCloudSource|MapCloudSource|PointColors)\b", _py_files())
    assert defs == {"reconstruction/cloud.py"}, defs
    users = _py_files("segmentation") + _py_files("service")
    assert not _grep(r"\b(pixel_mask|pixel_points|depth_normals|PointNormals|"
                     r"voxel_downsample_indices|budget_voxel_indices|ply_bytes)\(", users)
    assert _grep(r"def (segment_colors|height_colors|color_for_id)\(", _py_files()) == {
        "segmentation/colors.py"}
    assert not _grep(r"segment_colors|height_colors|color_for_id|UNSEGMENTED|PALETTE|VIRIDIS|"
                     r"#808080", _py_files("reconstruction"))
    # segmentation hands its colour assignment to the derivation, as data
    api = (SRC / "segmentation" / "api.py").read_text("utf-8")
    assert "POINT_COLORS = PointColors(segment=segment_colors, height=height_colors)" in api


def test_mapping_delegates_depth_and_segmentation() -> None:
    """Mapping gets keyframe depth from ``reconstruction.api`` and detections, lifting, boxes and
    colours from ``segmentation.api``; it never fits, lifts, detects or colours by itself."""
    files = _py_files("mapping")
    # api: keyframes; locate: the retrieval descriptor of a query image (large maps)
    assert _grep(r"reconstruct_image\(", files) == {"mapping/api.py", "mapping/locate.py"}
    assert _grep(r"detect_alongside\(", files) == {"mapping/api.py"}
    assert _grep(r"lift_detections\(", files) == {"mapping/objects.py"}
    assert _grep(r"fit_object_obb\(", files) == {"mapping/objects.py"}
    assert not _grep(r"\bfit_obb\(|\blift_mask\(|\bdetect\(|color(_hex)?_for_id\(|"
                     r"segmentation\.(detect|lift|obb|colors)\b", files)
    # segmentation owns lifting and colours; it does not know the mapper's keyframe settings
    seg = _py_files("segmentation")
    assert not _grep(r"KEYFRAME_TOKENS|KEYFRAME_GRID_SIDE|want_descriptor", seg)


# Mapping re-implements none of the shared geometry (spec §4, "Neither reconstruction, mapping nor
# the viewer re-implements that logic"): pixels are unprojected and points projected by
# ``core.geometry`` (``unproject_pixels``, ``project``), voxel keys come from ``voxel_keys``,
# normals from ``reconstruction.pointcloud.PointNormals``, and the depth correction of the image
# borders is reconstruction's (``reconstruction.borders``).
MAPPING_REIMPLEMENTATION = (
    r"\(\s*[\w\[\]:, ]+-\s*\w+\.c[xy]\s*\)\s*/\s*\w+\.f[xy]"  # (u - K.cx) / K.fx
    r"|-\s*\w*K\w*\[[01], 2\]\s*\)\s*/\s*\w*K\w*\[[01], [01]\]"  # (u - K[0, 2]) / K[0, 0]
    r"|\.f[xy]\s*\*\s*[\w\[\]:, ]+/\s*\w+\s*\+\s*\w+\.c[xy]"  # K.fx * x / z + K.cx
    r"|K\w*\[[01], [01]\]\s*\*\s*[\w\[\]:, ]+/\s*\w+\s*\+\s*\w*K\w*\[[01], 2\]"
    r"|linalg\.(eigh|eigvalsh)\(|einsum\(\"nki,nkj"  # normals by a local PCA
    r"|np\.floor\([^\n]*/\s*(cell|voxel|\w*VOXEL\w*)\)"  # voxel keys
    r"|BORDER_(SAME|NEIGHBOURS)\s*="  # the border depth correction
)


def test_mapping_reimplements_no_shared_geometry() -> None:
    assert not _grep(MAPPING_REIMPLEMENTATION, _py_files("mapping"))
    # the patterns find the formulas they forbid (as mapping wrote them before)
    for formula in ("(uv[:, 0] - K.cx) / K.fx * z", "(uu[ok] - K.cx) / K.fx * zs",
                    "Kg.fx * Y[:, 0] / zz + Kg.cx", "K.fy * pc[:, 1] / z + K.cy",
                    "(u - K[0, 2]) / K[0, 0] * z", "K[1, 1] * p[:, 1] / z + K[1, 2]",
                    'np.linalg.eigh(np.einsum("nki,nkj->nij", q, q))',
                    "np.floor(flat / BRIDGE_VOXEL)", "np.floor(np.asarray(p, np.float64) / cell)",
                    "BORDER_SAME = 0.1"):
        assert re.search(MAPPING_REIMPLEMENTATION, formula), formula
    # mapping applies reconstruction's border correction to its keyframes, nothing more
    geometry = (SRC / "mapping" / "geometry.py").read_text("utf-8")
    assert "border_depths(" in geometry
    assert "def correct_borders(" in (SRC / "reconstruction" / "borders.py").read_text("utf-8")
