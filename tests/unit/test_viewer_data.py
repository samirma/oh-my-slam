"""What ``view.sh`` serves, through its HTTP API: an image reconstructed and segmented once by the
fake inference client, and a map opened read-only without any server. Every cloud is the shared
derivation (the same points and colours as the PLY commands), controls never re-run inference,
camera poses are those of the scene JSON, and colours follow the §2.4 contract."""

from __future__ import annotations

import io
import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.cloud_attrs import CloudScope, applicable, parse_cloud_attrs
from oh_my_slam.core.ply import parse_ply
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id
from oh_my_slam.viewer.bundle import image_bundle, map_bundle
from oh_my_slam.viewer.server import parse_cloud_payload
from tests.browser.scenes import running, synthetic_image, synthetic_map

IMAGE_KEYS = ["color", "stride", "min-depth", "max-depth", "edge", "voxel", "normals"]
MAP_KEYS = ["color", "voxel", "normals"]
ATTR_SETS = ["", "stride=2", "min-depth=2.5&max-depth=4", "edge=0", "voxel=0.05",
             "normals=on", "color=segment", "color=height", "color=none&stride=3&voxel=0.02"]


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def cloud(url: str, query: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    code, body = get(f"{url}api/cloud?{query}")
    assert code == 200, body
    return parse_cloud_payload(body)


def spec(query: str) -> str:
    return query.replace("&", ",")


# ------------------------------------------------------------------------------------------------
# view.sh -i


@pytest.fixture(scope="module")
def image_view(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path, Any, dict]]:
    img, client = synthetic_image(tmp_path_factory.mktemp("viewimg"))
    bundle = image_bundle(img, client)
    calls = dict(client.calls)
    with running(bundle) as url:
        yield url, img, client, calls


def test_image_is_reconstructed_and_segmented_once(image_view: Any) -> None:
    url, _, client, calls = image_view
    assert calls == {"geometry": 1, "gravity": 1, "segment": 1}
    for query in ATTR_SETS * 2:
        cloud(url, query)
    assert dict(client.calls) == calls  # controls re-derive from memory, never re-run inference


def test_image_cloud_is_the_reconstruct_ply(image_view: Any, tmp_path: Path) -> None:
    """For every attribute set the served cloud equals ``reconstruct.sh -f ply -p …``."""
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.cloud import cloud_ply, image_cloud_source

    url, _, _, _ = image_view
    img, client = synthetic_image(tmp_path)
    frame, dets = reconstruct_and_detect(img, client)
    source = image_cloud_source(frame, segment_frame(frame, client=client, detections=dets))
    for query in ATTR_SETS:
        head, arrays = cloud(url, query)
        attrs = parse_cloud_attrs(spec(query), CloudScope.IMAGE)
        ply = parse_ply(cloud_ply(source, attrs))
        assert head["count"] == head["total"] == len(ply) > 0, query
        np.testing.assert_array_equal(arrays["position"], ply.xyz)
        if ply.rgb is None:
            assert "color" not in arrays
        else:
            np.testing.assert_array_equal(arrays["color"], ply.rgb)
        if ply.normals is not None:
            np.testing.assert_array_equal(arrays["normal"], ply.normals)
        assert ("normal" in arrays) == attrs.normals


def test_image_meta_controls_cameras_and_artefacts(image_view: Any) -> None:
    url, img, _, _ = image_view
    meta = json.loads(get(url + "api/meta")[1])
    scene = json.loads(get(url + "api/scene")[1])
    assert meta["mode"] == "image" and meta["title"] == img.name
    # controls: the §2.2 attributes of an image, minus the PLY-only ones, from core.cloud_attrs
    keys = [c["key"] for c in meta["controls"]]
    assert keys == IMAGE_KEYS == [a.key for a in applicable(CloudScope.IMAGE)
                                  if a.key not in ("label", "encoding")]
    by_key = {c["key"]: c for c in meta["controls"]}
    assert by_key["color"]["options"] == ["rgb", "segment", "height", "none"]
    assert by_key["normals"]["kind"] == "toggle" and by_key["stride"]["kind"] == "int"
    assert by_key["max-depth"]["default"] == "inf" and by_key["max-depth"]["off"] == "inf"
    assert by_key["voxel"]["unit"] == "m" and by_key["max-depth"]["max"] >= 4.0
    assert meta["defaults"] == "color=rgb,stride=1,min-depth=0,max-depth=inf,edge=0.04,voxel=0," \
                               "normals=off"
    # the camera at its estimated pose: the scene is in the camera frame, so the identity, with
    # the intrinsics the scene JSON states
    (cam,) = meta["cameras"]
    np.testing.assert_array_equal(cam["T"], np.eye(4))
    pin = scene["openlabel"]["streams"]["camera"]["stream_properties"]["intrinsics_pinhole"]
    m = pin["camera_matrix"]
    assert cam["K"] == [m[0], m[5], m[2], m[6]] and cam["size"] == [pin["width_px"],
                                                                   pin["height_px"]]
    # segmented image and catalogue
    code, png = get(url + "api/segmented.png")
    assert code == 200 and Image.open(io.BytesIO(png)).size == (320, 240)
    rows = json.loads(get(url + "api/catalog")[1])
    assert sorted(r["id"] for r in rows) == sorted(int(k) for k in scene["openlabel"]["objects"])
    assert meta["stats"]["objects"] == len(rows) == 3


def test_image_colours_follow_the_contract(image_view: Any) -> None:
    url, _, _, _ = image_view
    scene = json.loads(get(url + "api/scene")[1])
    _, arrays = cloud(url, "color=segment")
    lab, col = arrays["label"], arrays["color"]
    assert set(np.unique(lab).tolist()) == {0, 1, 2, 3}
    np.testing.assert_array_equal(col[lab == 0], np.tile(UNSEGMENTED, ((lab == 0).sum(), 1)))
    for oid, obj in scene["openlabel"]["objects"].items():
        rgb = next(v["val"] for v in obj["object_data"]["vec"] if v["name"] == "color")
        assert tuple(rgb) == color_for_id(int(oid))
        np.testing.assert_array_equal(np.unique(col[lab == int(oid)], axis=0), [rgb])
    # object ids are carried with every colour mode, so the segmentation layer never changes
    for query in ("", "color=height", "color=none"):
        np.testing.assert_array_equal(cloud(url, query)[1]["label"], lab)


def test_image_depth_range_error_is_actionable(image_view: Any) -> None:
    url, _, _, _ = image_view
    code, body = get(url + "api/cloud?min-depth=3&max-depth=2")
    assert code == 400
    assert json.loads(body)["error"] == "min-depth (3) must be smaller than max-depth (2)"


# ------------------------------------------------------------------------------------------------
# view.sh -m


needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")


@pytest.fixture(scope="module")
def map_view(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path, str]]:
    from oh_my_slam.mapping.store import full_tree_hash

    root = synthetic_map(tmp_path_factory.mktemp("viewmap"))
    before = full_tree_hash(root)
    with running(map_bundle(root)) as url:
        yield url, root, before


@needs_colmap
def test_map_view_is_read_only_and_needs_no_server(map_view: Any) -> None:
    from oh_my_slam.client.client import connect
    from oh_my_slam.core.errors import ServerUnavailableError
    from oh_my_slam.mapping.store import full_tree_hash

    url, root, before = map_view
    with pytest.raises(ServerUnavailableError):
        connect()  # no inference server in this test session
    for query in ("", "voxel=0.05", "normals=on", "color=segment", "color=height", "color=none"):
        cloud(url, query)
    for path in ("api/meta", "api/scene", "api/catalog"):
        assert get(url + path)[0] == 200
    assert full_tree_hash(root) == before


@needs_colmap
def test_map_meta_cameras_and_controls(map_view: Any) -> None:
    from oh_my_slam.mapping.store import MapReader

    url, root, _ = map_view
    meta = json.loads(get(url + "api/meta")[1])
    assert meta["mode"] == "map" and meta["has_segmented"] is False
    assert [c["key"] for c in meta["controls"]] == MAP_KEYS
    assert meta["defaults"] == "color=rgb,voxel=0,normals=off"
    code, body = get(url + "api/cloud?stride=2")
    assert code == 400 and "pixel-level attribute" in json.loads(body)["error"]
    frames = MapReader(root).frames
    assert len(meta["cameras"]) == len(frames) == meta["stats"]["frames"] > 0
    for cam, fr in zip(meta["cameras"], sorted(frames, key=lambda f: f.index), strict=True):
        assert cam["name"] == fr.name
        np.testing.assert_allclose(cam["T"], fr.T_map_cam.matrix(), atol=1e-5)
        np.testing.assert_allclose(cam["K"], [fr.K.fx, fr.K.fy, fr.K.cx, fr.K.cy], atol=1e-5)


@needs_colmap
def test_map_cloud_is_the_segment_m_ply(map_view: Any) -> None:
    """The complete map cloud, coloured by object exactly as ``segment.sh -m -f ply``."""
    from oh_my_slam.mapping.export import map_segment_outputs

    url, root, _ = map_view
    for query in ("", "voxel=0.05&normals=on"):
        head, arrays = cloud(url, "color=segment&" + query)
        attrs = parse_cloud_attrs(spec(query), CloudScope.MAP | CloudScope.SEGMENT)
        _, ply_bytes = map_segment_outputs(root, None, attrs)
        assert ply_bytes is not None
        ply = parse_ply(ply_bytes)
        assert head["count"] == head["total"] == len(ply) > 1000
        np.testing.assert_array_equal(arrays["position"], ply.xyz)
        np.testing.assert_array_equal(arrays["color"], ply.rgb)
    scene = json.loads(get(url + "api/scene")[1])
    assert int((arrays["label"] > 0).sum()) > 0
    assert set(np.unique(arrays["label"]).tolist()) - {0} <= {
        int(k) for k in scene["openlabel"]["objects"]}
