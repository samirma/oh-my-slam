"""The §5 evaluator end to end with the real models (``-m eval``): every entry point runs on
``examples/``, every expected metric gets a value, the contracts hold, and the report is written.

About ten minutes of GPU work. The evaluator stops and restarts the inference server to time its
cold start (and leaves it as it found it), so run this alone:

    OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m eval -s

Missed targets are findings of the run (exit 1), not failures of this test; the test fails when
a command or a metric could not be run or computed, or a contract is broken.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oh_my_slam.tools.evaluate.__main__ import main
from oh_my_slam.tools.evaluate.suite import expected_ids

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(os.environ.get("OH_MY_SLAM_TEST_REAL_SERVER") != "1",
                       reason="set OH_MY_SLAM_TEST_REAL_SERVER=1 (real weights; ~10 min on the GPU)"),
    pytest.mark.timeout(5400),
]


def test_the_evaluator_measures_every_entry_point(tmp_path: Path) -> None:
    out = tmp_path / "evaluation"
    code = main(["--out", str(out), "--baseline", str(tmp_path / "no-baseline.json")])
    assert code in (0, 1)
    result = json.loads((out / "result.json").read_text())
    assert (out / "summary.md").is_file()
    assert "errors" not in result["details"], result["details"]["errors"]
    by_id = {m["id"]: m for m in result["metrics"]}
    missing = {mid: by_id[mid]["error"] for mid in expected_ids() if by_id[mid]["value"] is None}
    assert not missing, missing  # every command ran and every metric was computed
    broken = {mid: m["detail"] for mid, m in by_id.items()
              if mid.startswith("contract.") and not m["passed"]}
    assert not broken, broken
    assert result["summary"]["regressions"] is None  # no baseline: nothing was compared
    # per stage: time and the peak memory of the command and of the server
    for group in ("reconstruct_json", "segment_image", "mapper_single", "segment_map"):
        stages = by_id[f"perf.{group}.wall_s"]["detail"]["stages"]
        assert stages, group
        assert all(s["client_peak_mb"] and s["server_peak_gb"] for s in stages.values()), stages
