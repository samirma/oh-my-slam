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


def test_viewer_has_no_inference_fitting_or_colour_logic() -> None:
    files = _py_files("viewer")
    assert not _grep(r"import torch|from oh_my_slam\.client", files)
    assert not _grep(r"segmentation\.(colors|obb|lift)|reconstruction\.(pointcloud|fusion)", files)


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
