"""Viewer HTTP server: read-only routes, the /api/cloud document and its validation, static files,
127.0.0.1 binding, display thinning, camera poses from the scene, the display frame."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.geometry import rot_z
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.cloud import derive_cloud, map_cloud_source
from oh_my_slam.viewer.bundle import ViewBundle, scene_cameras, upright_transform
from oh_my_slam.viewer.server import parse_cloud_payload
from tests.browser.scenes import running

RGB = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], np.uint8)


def small_map() -> ViewBundle:
    source = map_cloud_source(np.arange(12, dtype=np.float32).reshape(4, 3), RGB,
                              np.array([0, 1, 1, 0]), {1}, np.zeros((1, 3)))
    return ViewBundle(
        mode="map", title="t", scene={"openlabel": {"metadata": {"schema_version": "1.0.0"}}},
        source=source, catalog=[{"id": 1}], segmented_png=b"\x89PNGfake",
        stats={"objects": 1, "frames": 0},
    )


@pytest.fixture
def server() -> Iterator[str]:
    with running(small_map()) as url:
        yield url


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
    meta = json.loads(get(server + "api/meta")[2])
    assert meta["mode"] == "map" and meta["has_segmented"] and meta["cameras"] == []
    assert json.loads(get(server + "api/scene")[2])["openlabel"]["metadata"]
    assert get(server + "api/catalog")[2] == b'[{"id": 1}]'
    assert get(server + "api/segmented.png")[1] == "image/png"
    code, ctype, _ = get(server + "static/app.js")
    assert code == 200 and ctype == "text/javascript"
    assert get(server + "static/vendor/three/build/three.module.js")[0] == 200
    assert get(server + "static/../server.py")[0] == 404
    assert get(server + "static/%2e%2e/server.py")[0] == 404
    for gone in ("api/points.bin", "api/colors.bin", "api/segments.bin", "nope"):
        assert get(server + gone)[0] == 404
    assert get(server + "favicon.ico")[0] == 204
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        req = urllib.request.Request(server + "api/meta", data=b"x", method=method)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 405


def test_cloud_document(server: str) -> None:
    code, ctype, body = get(server + "api/cloud")
    assert code == 200 and ctype == "application/octet-stream"
    head, arrays = parse_cloud_payload(body)
    assert head["count"] == head["total"] == 4 and head["step"] == 1
    assert head["attrs"] == "color=rgb,voxel=0,normals=off"
    assert {b["offset"] % 4 for b in head["buffers"]} == {0}
    assert arrays["position"].shape == (4, 3)
    np.testing.assert_array_equal(arrays["position"].ravel(), np.arange(12))
    np.testing.assert_array_equal(arrays["color"], RGB)
    assert arrays["label"].tolist() == [0, 1, 1, 0] and "normal" not in arrays
    head, arrays = parse_cloud_payload(get(server + "api/cloud?color=none&normals=on")[2])
    assert set(arrays) == {"position", "label", "normal"}
    assert head["attrs"] == "color=none,voxel=0,normals=on"
    np.testing.assert_allclose(np.linalg.norm(arrays["normal"], axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize(("query", "message"), [
    ("stride=2", "pixel-level attribute"),
    ("edge=0", "pixel-level attribute"),
    ("label=on", "PLY files only"),
    ("encoding=ascii", "PLY files only"),
    ("colour=rgb", "unknown point-cloud attribute 'colour'"),
    ("color=purple", "color must be rgb|segment|height|none"),
    ("voxel=-1", "voxel must be metres >= 0"),
    ("voxel=", "voxel must be metres >= 0"),
    ("normals=yes", "normals must be on|off"),
    ("voxel=0.1&voxel=0.2", "given twice"),
])
def test_invalid_attributes_are_400_with_the_message(server: str, query: str,
                                                     message: str) -> None:
    code, ctype, body = get(server + "api/cloud?" + query)
    assert code == 400 and ctype == "application/json"
    assert message in json.loads(body)["error"]


def test_display_thinning_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import oh_my_slam.viewer.bundle as vb

    rng = np.random.default_rng(0)
    source = map_cloud_source(rng.normal(size=(10, 3)), rng.integers(0, 255, (10, 3), np.uint8),
                              None, set(), np.zeros((1, 3)))
    b = ViewBundle(mode="map", title="t", scene={}, source=source, catalog=[])
    monkeypatch.setattr(vb, "MAX_DISPLAY_POINTS", 3)
    dc = b.cloud(CloudAttrs())
    full = derive_cloud(source, CloudAttrs())
    assert (dc.total, dc.step, len(dc.cloud)) == (10, 4, 3)  # every 4th: 0, 4, 8
    np.testing.assert_array_equal(dc.cloud.xyz, full.xyz[[0, 4, 8]])
    np.testing.assert_array_equal(b.cloud(CloudAttrs()).cloud.xyz, dc.cloud.xyz)
    assert dc.cloud.label is None  # an unsegmented source has no object ids


def test_realistic_maps_are_shown_complete() -> None:
    """Spec §2.5 "complete point cloud": the 5.4 M-point evaluation map (and anything up to 12 M
    points, 60 frames/s in Edge on the M4 Max) is served unthinned."""
    import oh_my_slam.viewer.bundle as vb

    assert vb.MAX_DISPLAY_POINTS >= 12_000_000
    b = ViewBundle(mode="map", title="t", scene={}, catalog=[],
                   source=map_cloud_source(np.zeros((5, 3)), np.zeros((5, 3), np.uint8), None,
                                           set(), np.zeros((1, 3))))
    assert b.meta()["max_points"] == vb.MAX_DISPLAY_POINTS


def test_cameras_are_the_poses_of_the_scene_json() -> None:
    K = Intrinsics(500.0, 510.0, 320.0, 240.0, 640, 480)
    # a single image: the scene is in the camera frame, the camera sits at the identity
    image = ol.document(ol.metadata("x"), {}, coordinate_systems={"camera": ol.sensor_cs()},
                        streams={"camera": ol.camera_stream(K)},
                        frames={"0": ol.frame(0.0, stream_uris={"camera": "x.jpg"})})
    (cam,) = scene_cameras(image)
    np.testing.assert_array_equal(cam["T"], np.eye(4))
    assert cam["position"] == [0.0, 0.0, 0.0] and cam["source"] == "x.jpg"
    assert cam["K"] == [500.0, 510.0, 320.0, 240.0] and cam["size"] == [640, 480]
    # a map: one camera per frame, at the frame's camera-to-map transform
    poses = [Pose(rot_z(0.3 * k), np.array([k, 2.0 * k, 0.5])) for k in range(3)]
    frames = {str(k): ol.frame(float(k), stream_uris={"camera_0": f"f{k}.jpg"},
                               transforms={"camera_0_to_map": ol.transform("camera_0", "map", p)},
                               keyframe=f"f{k:06d}", update_id=1 + k // 2)
              for k, p in enumerate(poses)}
    scene = ol.document(ol.metadata("m"), {},
                        coordinate_systems={"map": ol.map_cs(["camera_0"]),
                                            "camera_0": ol.sensor_cs("map")},
                        streams={"camera_0": ol.camera_stream(K)}, frames=frames)
    cams = scene_cameras(scene)
    assert [c["name"] for c in cams] == ["f000000", "f000001", "f000002"]
    assert [c["update"] for c in cams] == [1, 1, 2]
    assert [c["source"] for c in cams] == ["f0.jpg", "f1.jpg", "f2.jpg"]
    for c, p, fr in zip(cams, poses, frames.values(), strict=True):
        np.testing.assert_allclose(c["T"], p.matrix(), atol=1e-6)
        # the listed camera centre is exactly the translation the scene JSON states
        tr = fr["frame_properties"]["transforms"]["camera_0_to_map"]["transform_src_to_dst"]
        assert c["position"] == tr["translation"] == [float(round(v, 6)) for v in p.t]
        assert np.asarray(c["T"])[:3, 3].tolist() == c["position"]


def test_upright_display_frame() -> None:
    T = upright_transform(np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(T[:3, :3] @ [0, -1, 0], [0, 0, 1], atol=1e-9)  # up → +z
    assert (T[:3, :3] @ [0, 0, 1])[1] > 0.99  # the camera looks along +y
    tilted = np.array([0.0, -np.cos(0.3), np.sin(0.3)])  # camera pitched down by 0.3 rad
    T = upright_transform(tilted)
    np.testing.assert_allclose(T[:3, :3] @ tilted, [0, 0, 1], atol=1e-9)


def test_view_cli_arguments() -> None:
    from oh_my_slam.cli.view import URL_LINE, build_parser

    ap = build_parser()
    a = ap.parse_args(["-m", "map"])
    assert a.port == 0 and not a.no_browser
    a = ap.parse_args(["-i", "x.jpg", "--no-browser"])
    assert a.no_browser and a.image.name == "x.jpg"
    for bad in ([], ["-i", "a", "-m", "b"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(bad)
        assert e.value.code == 2
    m = URL_LINE.match("view.sh: listening on http://127.0.0.1:50123/")
    assert m and m.group(1) == "http://127.0.0.1:50123/"
