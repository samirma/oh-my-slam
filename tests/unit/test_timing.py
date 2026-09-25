"""Per-stage timing instrumentation (core.timing): exclusive nested stages, concurrent parts,
server request records, no-op outside a collection, the stderr summary and the JSON file."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
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


def test_peak_memory_per_stage_and_the_stage_windows() -> None:
    """Each stage's peak resident set comes from the sampler (and the boundary readings); peaks
    include nested stages; every stage has its wall-clock window after ``t0_unix``."""
    t_start = time.time()
    with timing.collect(sample_every=0.01) as t:
        with timing.stage("small"):
            time.sleep(0.08)
        with timing.stage("outer"):
            with timing.stage("big"):
                a = np.ones(25_000_000)  # 200 MB, touched
                time.sleep(0.08)
                del a
            time.sleep(0.02)
        with timing.stage("after"):
            time.sleep(0.08)
    d = t.to_dict()
    peaks = d["stages_peak_rss_mb"]
    assert list(peaks) == list(d["stages_s"])  # same stages, deterministic structure
    assert peaks["big"] >= peaks["small"] + 150
    assert peaks["outer"] >= peaks["big"]  # inclusive of the nested stage
    assert peaks["after"] > 10  # (macOS may keep freed pages resident: no drop is asserted)
    assert t_start - 1e-3 <= d["t0_unix"] <= t_start + 1  # rounded to ms
    names = [w[0] for w in d["stage_windows"]]
    assert names == ["small", "outer", "big", "after"]  # by start time
    small, outer, big, after = d["stage_windows"]
    assert 0 <= small[1] < small[2] <= outer[1] <= big[1] < big[2] <= outer[2] <= after[1]
    assert big[2] - big[1] == pytest.approx(d["stages_s"]["big"], abs=0.01)
    assert t._sampler is None  # the sampler stops with the collection


def test_without_sampling_the_boundaries_still_give_peaks() -> None:
    t = timing.Timings(sample_every=None)
    with t.stage("x"):
        pass
    assert t.to_dict()["stages_peak_rss_mb"]["x"] > 10
