"""Viewer HTTP server: read-only routes, binary payloads, static files, 127.0.0.1 binding."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud
from oh_my_slam.viewer.bundle import ViewBundle, _thin, _upright_transform
from oh_my_slam.viewer.server import serve, url_of


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"glTF-fake")
    cloud = PointCloud(np.arange(12, dtype=np.float32).reshape(4, 3),
                       np.array([[1, 2, 3]] * 4, np.uint8))
    b = ViewBundle(
        mode="map", title="t", scene={"openlabel": {"metadata": {"schema_version": "1.0.0"}}},
        cloud=cloud, segments=np.full((4, 3), 128, np.uint8), labels=np.array([0, 1, 1, 0]),
        catalog=[{"id": 1}], frustums=[], mesh_path=mesh, segmented_png=b"\x89PNGfake",
        stats={"points": 4},
    )
    httpd = serve(b, 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield url_of(httpd)
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_routes(server: str) -> None:
    assert server.startswith("http://127.0.0.1:")
    code, ctype, body = get(server)
    assert code == 200 and "text/html" in ctype and b"importmap" in body
    code, _, body = get(server + "api/meta")
    meta = json.loads(body)
    assert meta["mode"] == "map" and meta["points"] == 4 and meta["has_mesh"]
    assert json.loads(get(server + "api/scene")[2])["openlabel"]["metadata"]
    assert np.frombuffer(get(server + "api/points.bin")[2], "<f4").reshape(-1, 3).shape == (4, 3)
    assert get(server + "api/colors.bin")[2] == bytes([1, 2, 3] * 4)
    assert set(get(server + "api/segments.bin")[2]) == {128}
    assert np.frombuffer(get(server + "api/labels.bin")[2], "<i4").tolist() == [0, 1, 1, 0]
    assert get(server + "api/catalog")[2] == b'[{"id": 1}]'
    assert get(server + "api/mesh.glb")[2] == b"glTF-fake"
    assert get(server + "api/segmented.png")[1] == "image/png"
    code, ctype, _ = get(server + "static/app.js")
    assert code == 200 and ctype == "text/javascript"
    assert get(server + "static/vendor/three/build/three.module.js")[0] == 200
    assert get(server + "static/../server.py")[0] == 404
    assert get(server + "static/%2e%2e/server.py")[0] == 404
    assert get(server + "nope")[0] == 404
    assert get(server + "favicon.ico")[0] == 204
    req = urllib.request.Request(server + "api/meta", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 405


def test_display_helpers() -> None:
    T = np.array(_upright_transform(np.array([0.0, -1.0, 0.0])))
    np.testing.assert_allclose(T[:3, :3] @ [0, -1, 0], [0, 0, 1], atol=1e-9)  # up → +z
    fwd = T[:3, :3] @ [0, 0, 1]
    assert fwd[1] > 0.99  # the camera looks along +y
    import oh_my_slam.viewer.bundle as vb

    old = vb.MAX_DISPLAY_POINTS
    vb.MAX_DISPLAY_POINTS = 3
    try:
        c = PointCloud(np.zeros((10, 3)), np.zeros((10, 3)))
        thin, (lab,) = _thin(c, [np.arange(10)])
        assert len(thin) == 3 and len(lab) == 3
    finally:
        vb.MAX_DISPLAY_POINTS = old


def test_view_cli_arguments() -> None:
    from oh_my_slam.cli.view import build_parser

    ap = build_parser()
    a = ap.parse_args(["-m", "map"])
    assert a.port == 0 and not a.no_browser
    a = ap.parse_args(["-i", "x.jpg", "--port", "8123", "--no-browser"])
    assert a.port == 8123 and a.no_browser
    for bad in ([], ["-i", "a", "-m", "b"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(bad)
        assert e.value.code == 2
