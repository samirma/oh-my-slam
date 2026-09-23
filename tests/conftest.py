"""Shared fixtures. Runtime files go to a short per-session directory so Unix socket paths fit."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_runtime() -> Iterator[None]:
    """Keep tests away from the real server runtime dir unless running the `models` suite."""
    if os.environ.get("OH_MY_SLAM_TEST_REAL_SERVER") == "1":
        yield
        return
    short = Path(tempfile.mkdtemp(prefix="oms-", dir="/tmp"))
    old = os.environ.get("OH_MY_SLAM_RUNTIME_DIR")
    os.environ["OH_MY_SLAM_RUNTIME_DIR"] = str(short)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OH_MY_SLAM_RUNTIME_DIR", None)
        else:
            os.environ["OH_MY_SLAM_RUNTIME_DIR"] = old
        shutil.rmtree(short, ignore_errors=True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
