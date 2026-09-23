"""Per-stage timing instrumentation (core.timing): exclusive nested stages, concurrent parts,
server request records, no-op outside a collection, the stderr summary and the JSON file."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from oh_my_slam.core import timing


def test_calls_outside_a_collection_are_no_ops() -> None:
    assert timing.current() is None
    with timing.stage("a"), timing.part("b"):
        pass
    timing.record_request("/v1/geometry", 1.0, 0.1, 0.5)
    timing.count(x=1)
    assert timing.current() is None


def test_nested_stages_are_exclusive_and_add_up() -> None:
    with timing.collect() as t:
        with timing.stage("outer"):
            time.sleep(0.02)
            with timing.stage("inner"):
                time.sleep(0.03)
        with timing.stage("inner"):  # repeated stages accumulate
            time.sleep(0.01)
    d = t.to_dict()
    assert d["stages_s"]["inner"] == pytest.approx(0.04, abs=0.02)
    assert d["stages_s"]["outer"] == pytest.approx(0.02, abs=0.02)
    assert sum(d["stages_s"].values()) <= d["total_s"] + 1e-3
    assert timing.current() is None  # collection restored


def test_parts_and_requests_from_threads() -> None:
    with timing.collect() as t:
        def work() -> None:
            for _ in range(10):
                with timing.part("geometry"):
                    pass
                timing.record_request("/v1/geometry", 0.2, 0.05, 0.1)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        timing.count(keyframes_sampled=3)
    d = t.to_dict()
    assert d["parts"]["geometry"]["count"] == 40
    srv = d["server"]["geometry"]  # route shortened to the endpoint name
    assert srv["count"] == 40
    assert srv["compute_s"] == pytest.approx(4.0)
    assert srv["queue_s"] == pytest.approx(2.0)
    assert srv["wall_s"] == pytest.approx(8.0)
    assert d["counts"] == {"keyframes_sampled": 3}
    assert d["peak_rss_mb"]["self"] > 10


def test_report_logs_one_line_and_writes_the_env_file(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    target = tmp_path / "sub" / "timings.json"
    monkeypatch.setenv(timing.ENV_PATH, str(target))
    with timing.collect() as t:
        with timing.stage("inference"):
            timing.record_request("/v1/segment", 0.3, 0.0, 0.25)
    logger = logging.getLogger("test_timing")
    with caplog.at_level(logging.INFO, logger="test_timing"):
        rec = timing.report(t, logger, command="x")
    line = caplog.records[-1].getMessage()
    assert line.startswith("timings: total") and "inference" in line and "segment 1×" in line
    data = json.loads(target.read_text())
    assert data == json.loads(json.dumps(rec))
    assert data["command"] == "x" and "inference" in data["stages_s"]
    # a finished record (dict) can be reported too
    monkeypatch.delenv(timing.ENV_PATH)
    assert timing.report(rec, logger)["stages_s"] == rec["stages_s"]
