"""Spec §4: the unit tests run offline. ``conftest._offline`` refuses every connection of the test
process to another machine, and lets the ones to this machine through (the stub server's Unix
socket, the web service on loopback); the processes a test starts (the commands, COLMAP) send
anything but this machine to a dead proxy (``conftest.OFFLINE_ENV``)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.conftest import OFFLINE_ENV, OfflineError, is_local


@pytest.mark.parametrize("address", [("192.0.2.1", 80), ("2001:db8::1", 443, 0, 0),
                                     ("example.com", 443)])
def test_a_connection_to_another_machine_is_refused(address: tuple[object, ...]) -> None:
    family = socket.AF_INET6 if ":" in str(address[0]) else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        with pytest.raises(OfflineError, match="run offline"):
            s.connect(address)
        with pytest.raises(OfflineError, match="run offline"):
            s.connect_ex(address)


def test_connections_to_this_machine_go_through() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        port = server.getsockname()[1]
        with socket.socket() as a:
            a.connect(("127.0.0.1", port))
        with socket.socket() as b:
            assert b.connect_ex(("127.0.0.1", port)) == 0
    for family, address in [(socket.AF_UNIX, "/tmp/x.sock"), (socket.AF_INET, ("localhost", 1)),
                            (socket.AF_INET, ("0.0.0.0", 1)), (socket.AF_INET6, ("::1", 1)),
                            (socket.AF_INET6, ("::ffff:127.0.0.1", 1)),
                            (socket.AF_INET6, ("fe80::1%lo0", 1))]:
        assert is_local(family, address) == (address != ("fe80::1%lo0", 1)), address


def test_the_processes_a_test_starts_reach_no_other_machine(tmp_path: Path) -> None:
    """The processes a test starts get ``OFFLINE_ENV``: curl (the libcurl the Homebrew COLMAP
    downloads its models with) and Python's urllib send a request for another machine to a proxy
    on a closed loopback port and fail at once; a request for this machine goes straight to it."""
    assert all(os.environ[k] == v for k, v in OFFLINE_ENV.items())
    curl = subprocess.run(["curl", "-sS", "--max-time", "5", "https://192.0.2.1/"],
                          capture_output=True, text=True, timeout=30)
    assert curl.returncode == 7 and "127.0.0.1 port 9" in curl.stderr  # the dead proxy
    py = subprocess.run([sys.executable, "-c", "import urllib.request\n"
                         "urllib.request.urlopen('http://192.0.2.1/', timeout=5)"],
                        capture_output=True, text=True, timeout=30)
    assert py.returncode != 0 and "Connection refused" in py.stderr
    assert urllib.request._opener is None  # this test's own: not one an earlier test built
    (tmp_path / "page").write_text("here")
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for host in ("127.0.0.1", "localhost"):
                local = subprocess.run(
                    ["curl", "-sS", "--max-time", "5",
                     f"http://{host}:{server.server_address[1]}/page"],
                    capture_output=True, text=True, timeout=30)
                assert (local.returncode, local.stdout) == (0, "here"), local.stderr
        finally:
            server.shutdown()
