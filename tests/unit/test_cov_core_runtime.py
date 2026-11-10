"""The package logger (``core.log``) and the timing collector's edges (``core.timing``)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from oh_my_slam.core import log, timing

# --- stdout carries the payload only -------------------------------------------------------------

def test_the_package_logger_writes_to_stderr_without_propagating() -> None:
    root = log.get_logger()
    assert root is logging.getLogger("oh_my_slam") and not root.propagate
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert log.get_logger("oh_my_slam.service") is root.getChild("service")
    assert log.get_logger("plain") is root.getChild("plain")


# --- exit codes ----------------------------------------------------------------------------------


# --- timing --------------------------------------------------------------------------------------


def test_sampling_starts_once_and_only_when_asked() -> None:
    t = timing.Timings(sample_every=None)
    t.start_sampling()
    assert t._sampler is None  # boundaries only
    t.stop_sampling()  # nothing to stop
    t = timing.Timings(sample_every=0.01)
    t.start_sampling()
    sampler = t._sampler
    t.start_sampling()
    assert t._sampler is sampler and sampler is not None and sampler.is_alive()
    t.stop_sampling()
    assert t._sampler is None and not sampler.is_alive()


def test_a_stage_reentered_stays_active_until_its_outer_run_ends() -> None:
    t = timing.Timings(sample_every=None)
    with t.stage(timing.Stage.OBJECTS):
        with t.stage(timing.Stage.OBJECTS):
            assert t._active == {"objects": 2}
        assert t._active == {"objects": 1}
    assert t._active == {} and t.stages["objects"] >= 0
    assert [w[0] for w in t.windows] == ["objects", "objects"]


def test_an_unreadable_resident_set_reads_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gone:
        def memory_info(self) -> Any:
            raise ProcessLookupError("no such process")

    monkeypatch.setattr(timing, "_process", Gone())
    assert timing.rss_mb() == 0.0  # instrumentation never fails a command
