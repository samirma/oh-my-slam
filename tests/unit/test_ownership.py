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
    assert seg_internals <= by_source["oh_my_slam.cli"]
    assert "oh_my_slam.client" in by_source["oh_my_slam.mapping"]
    assert {"oh_my_slam.segmentation", "oh_my_slam.mapping"} <= by_source[
        "oh_my_slam.reconstruction"]
    assert "oh_my_slam.core.ply" in by_source["oh_my_slam.segmentation"]  # it writes no PLY
    for pkg in ("mapping", "segmentation", "viewer", "server", "cli", "commands", "tools"):
        assert "open3d" in by_source[f"oh_my_slam.{pkg}"], pkg
    for pkg in ("cli", "commands", "tools", "viewer", "mapping", "segmentation", "reconstruction",
                "client"):
        assert {"torch", "ultralytics"} <= by_source[f"oh_my_slam.{pkg}"], pkg
    assert "oh_my_slam.client" in by_source["oh_my_slam.viewer"]
    # only the CLI starts the server; the evaluator reads its lifecycle state and nothing else
    for pkg in ("commands", "viewer", "mapping", "segmentation", "reconstruction", "client",
                "schema", "core"):
        assert "oh_my_slam.server" in by_source[f"oh_my_slam.{pkg}"], pkg
    assert "oh_my_slam.commands" in by_source["oh_my_slam.server"]
    # the commands' definitions sit right under the command line, the web service (spec 2.6) and
    # the evaluators (spec 5), which never import one another
    (layers,) = [c["layers"] for c in contracts.values() if c["type"] == "layers"]
    assert layers[:2] == ["oh_my_slam.cli | (oh_my_slam.web) | oh_my_slam.tools",
                          "oh_my_slam.commands"]
    server_internals = {f"oh_my_slam.server.{m}" for m in ("app", "main", "gpu_worker", "models")}
    assert server_internals <= by_source["oh_my_slam.tools"]
    assert "oh_my_slam.server.lifecycle" not in by_source["oh_my_slam.tools"]


def test_point_clouds_are_reconstructions_and_colours_segmentations() -> None:
    """§4: the reconstruction code alone derives every emitted cloud (pixel selection, voxel
    thinning, normals, PLY), which segmentation, mapping and the viewer reuse; the segmentation
    code alone assigns colours, which the derivation receives as data and never names."""
    defs = _grep(r"def (derive_cloud|derive_thinned|cloud_ply)\(|"
                 r"class (ImageCloudSource|MapCloudSource|PointColors)\b", _py_files())
    assert defs == {"reconstruction/cloud.py"}, defs
    users = _py_files("segmentation") + _py_files("viewer") + _py_files("cli")
    assert not _grep(r"\b(pixel_mask|pixel_points|depth_normals|PointNormals|"
                     r"voxel_downsample_indices|budget_voxel_indices|ply_bytes)\(", users)
    assert _grep(r"def (segment_colors|height_colors|color_for_id)\(", _py_files()) == {
        "segmentation/colors.py"}
    assert not _grep(r"segment_colors|height_colors|color_for_id|UNSEGMENTED|PALETTE|VIRIDIS|"
                     r"#808080", _py_files("reconstruction"))
    # segmentation hands its colour assignment to the derivation, as data
    api = (SRC / "segmentation" / "api.py").read_text("utf-8")
    assert "POINT_COLORS = PointColors(segment=segment_colors, height=height_colors)" in api
    rec = (SRC / "cli" / "reconstruct.py").read_text("utf-8")
    assert "from oh_my_slam.reconstruction.cloud import" in rec
    assert "cloud_ply(image_cloud_source(" in rec  # segmentation's colours, as data


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



# view.sh owns only the web server and UI (spec §2.5). It may call the owners' public APIs — one
# reconstruction + segmentation run (segmentation.api), the read-only map export, the shared
# point-cloud derivation (reconstruction.cloud) and attribute definition (core.cloud_attrs) — but never
# the internals behind them, and it re-implements none of their logic.
VIEWER_FORBIDDEN_IMPORTS = (
    r"import torch|oh_my_slam\.client|oh_my_slam\.server"
    r"|segmentation\.(colors|obb|lift|detect)"
    r"|reconstruction\.(pointcloud|fusion|depth|multiview)"
    r"|mapping\.(api|objects|sfm|geometry|fusion)"
)
VIEWER_REIMPLEMENTATION = (
    r"unproject|pixel_mask|depth_edge_mask|depth_normals|PointNormals|voxel_keys"
    r"|voxel_downsample|budget_voxel|fit_obb|fit_upright_obb|color_for_id|color_hex_for_id|segment_colors"
    r"|height_colors|PALETTE|np\.random|default_rng|K\.fx|\.K\(\)"
)


def test_viewer_only_calls_the_owners_apis() -> None:
    files = _py_files("viewer") + [SRC / "cli" / "view.py"]
    assert not _grep(VIEWER_FORBIDDEN_IMPORTS, files)
    assert not _grep(VIEWER_REIMPLEMENTATION, files)
    bundle = (SRC / "viewer" / "bundle.py").read_text("utf-8")
    # every displayed cloud is the shared derivation, controlled by the shared attribute set
    for call in ("derive_thinned(", "applicable(", "parse_cloud_attrs(", "image_cloud_source(",
                 "reader_source(", "segment_frame(", "single_image_scene(", "scene_bytes("):
        assert call in bundle, call


def test_viewer_page_has_no_palette_or_randomness() -> None:
    """Colours reach the page only as data (scene JSON, derived clouds); nothing random."""
    from oh_my_slam.segmentation.colors import PALETTE_HEX, UNSEGMENTED, rgb_to_hex

    static = SRC / "viewer" / "static"
    pages = [p for p in static.rglob("*") if p.suffix in (".js", ".html", ".css")
             and "vendor" not in p.relative_to(static).parts]
    assert pages
    for p in pages:
        text = p.read_text("utf-8").lower()
        assert not [h for h in (*PALETTE_HEX, rgb_to_hex(UNSEGMENTED)) if h in text], p
        assert "math.random" not in text, p


ENTRY_SCRIPTS = ("reconstruct.sh", "mapper.sh", "segment.sh", "view.sh",
                 "start_inference_server.sh", "server.sh")


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_entry_scripts_are_thin_and_never_call_each_other(script: str) -> None:
    root = SRC.parents[1]
    text = (root / script).read_text()
    assert "oms_exec" in text
    for other in ENTRY_SCRIPTS:
        if other != script:
            assert not re.search(rf"(?<![\w]){re.escape(other)}", text), other
    assert (root / script).stat().st_mode & 0o111


def test_web_service_is_a_client_without_models_or_open3d() -> None:
    """server.sh (oh_my_slam.web, spec 2.6) is covered by the ownership contracts: no model
    framework, no Open3D, no inference-server internals; it reuses only the lifecycle helpers."""
    import tomllib

    cfg = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    forbidden: set[str] = set()
    for c in cfg["tool"]["importlinter"]["contracts"]:
        if c["type"] == "forbidden" and "oh_my_slam.web" in c["source_modules"]:
            forbidden.update(c["forbidden_modules"])
    assert {"torch", "ultralytics", "open3d"} <= forbidden
    assert {f"oh_my_slam.server.{m}" for m in ("app", "main", "gpu_worker", "models")} <= forbidden
    # it runs the commands' Python entry points as subprocesses, never the shell scripts
    runner = (SRC / "web" / "runner.py").read_text("utf-8")
    assert '[self.python, "-m", run.module, *run.argv]' in runner
    assert 'return f"oh_my_slam.cli.{' in (SRC / "web" / "operations.py").read_text("utf-8")
    assert not _grep(r"subprocess\.\w+\(\s*\[?[^\]]*\.sh", _py_files("web"))
