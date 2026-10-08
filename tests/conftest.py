"""Shared fixtures. Runtime files go to a short per-session directory so Unix socket paths fit,
and the unit tests run offline (spec §4): this process connects to this machine only, and the
processes a test starts reach the network through no proxy-aware client (``OFFLINE_ENV``)."""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# the suites that use the real server, a browser or the evaluator may reach further
ONLINE_MARKERS = ("models", "browser", "eval")
# The environment of the processes an offline test starts: every proxy-aware client — libcurl,
# with which the Homebrew COLMAP downloads a model it lacks (sift-lightglue.onnx, when a video's
# weak links are matched again), Python's urllib, httpx, huggingface_hub — sends anything but this
# machine to a proxy on a closed loopback port, so it fails at once instead of reaching the
# network; the Hugging Face hub is offline too.
DEAD_PROXY = "http://127.0.0.1:9"
LOOPBACK = "localhost,127.0.0.1,::1,0.0.0.0"
OFFLINE_ENV = {
    **{k: DEAD_PROXY for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY",
                               "HTTPS_PROXY", "ALL_PROXY")},
    "no_proxy": LOOPBACK, "NO_PROXY": LOOPBACK, "HF_HUB_OFFLINE": "1",
}


class OfflineError(RuntimeError):
    """A unit test tried to reach another machine."""


def is_local(family: int, address: Any) -> bool:
    """A Unix socket, or an address of this machine: loopback, ``localhost``, or the unspecified
    address (0.0.0.0, ::), which a connection takes as this machine."""
    if family == socket.AF_UNIX:
        return True
    host = str(address[0] if isinstance(address, tuple) else address)
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False  # a name, which could be anywhere
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection of this process to another machine (``OfflineError``), and give
    the processes it starts ``OFFLINE_ENV``."""
    if any(request.node.get_closest_marker(m) for m in ONLINE_MARKERS):
        return
    for key, value in OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    # urlopen's opener keeps the proxies it read when built: one built under OFFLINE_ENV is this
    # test's only (a later browser test reaches its local server directly)
    monkeypatch.setattr(urllib.request, "_opener", None)
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(sock: socket.socket, address: Any) -> None:
        if not is_local(sock.family, address):
            raise OfflineError(f"the unit tests run offline (spec §4): refused a connection to "
                               f"{address!r}")

    def connect(self: socket.socket, address: Any) -> None:
        check(self, address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        check(self, address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


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
