"""Readers of images, layout, geometry, floor plan, validation and the API client."""

from __future__ import annotations

import importlib
import json
import math
import socket
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import rpl_api
import rpl_floorplan as fp
import rpl_geometry as geo
import rpl_images as img
import rpl_layout
import rpl_validate

from . import synth
from .fake_server import Config, Fake


# ---- layout ---------------------------------------------------------------------------------
def test_layout_merge_and_errors(tmp_path: Path) -> None:
    assert rpl_layout.load_layout(None) == rpl_layout.LAYOUT
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"frame_time": {"utc_offset_h": 2}, "new": [1]}))
    lay = rpl_layout.load_layout(p)
    assert lay["frame_time"]["utc_offset_h"] == 2 and lay["new"] == [1]
    assert lay["frame_time"]["pattern"] == rpl_layout.LAYOUT["frame_time"]["pattern"]
    p.write_text("[1]")
    with pytest.raises(rpl_layout.LayoutError, match="JSON object"):
        rpl_layout.load_layout(p)
    with pytest.raises(rpl_layout.LayoutError, match=r"missing key 'b'.*keys present: a"):
        rpl_layout.need({"a": 1}, "b", "here")
    with pytest.raises(rpl_layout.LayoutError, match="expected a JSON object"):
        rpl_layout.need([1], "b", "here")


# ---- images ---------------------------------------------------------------------------------
def test_image_sizes_and_errors() -> None:
    assert img.image_size(synth.jpeg(64, 48)) == (64, 48)
    assert img.image_size(synth.png(5, 4)) == (5, 4)
    assert img.mime(synth.png(1, 1)) == "image/png"
    with pytest.raises(img.ImageError, match="not a JPEG"):
        img.mime(b"GIF89a")
    with pytest.raises(img.ImageError, match="corrupt"):
        img.image_size(b"\xff\xd8\x00\x00\x00\x00")
    fill = b"\xff\xd8\xff" + synth.jpeg(7, 3)[2:]  # a fill byte before the first marker
    assert img.image_size(fill) == (7, 3)
    no_sof = b"\xff\xd8\xff\xe0\x00\x04ab"  # data ends without a scan
    with pytest.raises(img.ImageError, match="start-of-frame"):
        img.image_size(no_sof)
    sig = img.PNG_SIGNATURE
    chunk = struct.pack(">I", 0) + b"tEXt" + b"\x00" * 4
    with pytest.raises(img.ImageError, match="IHDR"):
        img.image_size(sig + chunk)  # no IEND either
    ihdr_second = sig + chunk + synth.png(9, 2)[len(sig):]
    assert img.image_size(ihdr_second) == (9, 2)


def test_exif_focal() -> None:
    e = synth.focal_entries(400.0)
    assert img.exif_focal_px(synth.jpeg(8, 8, synth.tiff_exif(e))) == pytest.approx((400, 400))
    assert img.exif_focal_px(synth.png(2, 2, synth.tiff_exif(e, ">"))) == pytest.approx(
        (400, 400))
    # entries in IFD0, inch unit, other numeric types, skipped kinds, a zero denominator
    inch = [(0x920A, 10, (2, 1)), (0xA20E, 5, (254, 10)), (0xA210, 4, 2), (0x0001, 9, -3),
            (0x0002, 2, 4), (0x0003, 3, 0), (0x0004, 5, (1, 0))]
    fx, fy = img.exif_focal_px(synth.jpeg(8, 8, synth.tiff_exif(inch, exif_ifd=False)))
    assert fx == pytest.approx(2 * 25.4 / 25.4) and fy == fx
    no_unit = synth.focal_entries(10.0, unit=None)
    assert img.exif_focal_px(synth.jpeg(1, 1, synth.tiff_exif(no_unit)), "cm") == \
        pytest.approx((1.0, 1.0))
    assert img.exif_focal_px(synth.jpeg(1, 1)) is None
    assert img.exif_focal_px(synth.png(1, 1)) is None
    assert img.exif_focal_px(synth.jpeg(1, 1, b"XX\x00\x00\x00\x00\x00\x00")) is None
    assert img.exif_focal_px(synth.jpeg(1, 1, b"II")) is None
    assert img.exif_focal_px(synth.jpeg(1, 1, synth.tiff_exif([(0x920A, 5, (1, 1))]))) is None


# ---- geometry -------------------------------------------------------------------------------
def test_geometry_basics() -> None:
    assert geo.mat4(list(range(16))).shape == (4, 4)
    with pytest.raises(ValueError, match="4x4"):
        geo.mat4([1, 2, 3])
    with pytest.raises(ValueError, match="4 numbers"):
        geo.quat_to_R([1, 0, 0])
    R = geo.rot_z(0.3)
    assert np.allclose(geo.quat_to_R([0, 0, math.sin(0.15), math.cos(0.15)], "xyzw"), R)
    assert np.allclose(geo.quat_to_R([math.cos(0.15), 0, 0, math.sin(0.15)]), R)
    assert geo.yaw_of_R(R) == pytest.approx(0.3)
    assert geo.wrap_deg(270) == -90
    assert geo.angle_deg(np.eye(3), R) == pytest.approx(math.degrees(0.3))
    T = geo.make_T(R, [1, 2, 3])
    assert np.allclose(geo.inv_T(T) @ T, np.eye(4))
    assert geo.optical_yaw_deg(synth.R_BODY_OPT) == pytest.approx(0.0)


def test_umeyama() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(10, 3))
    true = geo.Sim3(1.3, geo.rot_z(0.7), np.array([1.0, -2.0, 0.5]))
    sim, rms = geo.umeyama(src, true.points(src))
    assert sim.s == pytest.approx(1.3) and rms < 1e-9
    assert np.allclose(sim.R, true.R) and sim.yaw_deg == pytest.approx(math.degrees(0.7))
    p = np.array([0.1, 0.2, 0.3])
    assert np.allclose(sim.inverse_point(sim.point(p)), p)
    assert np.allclose(sim.pose(np.eye(4))[:3, :3], true.R)
    reflected = src * np.array([1, 1, -1])  # a reflection: the det correction
    sim, _ = geo.umeyama(src, reflected)
    assert np.linalg.det(sim.R) == pytest.approx(1.0)
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])  # collinear: yaw-only fit
    sim, rms = geo.umeyama(line, true.points(line))
    assert sim.s == pytest.approx(1.3) and rms < 1e-9
    R = geo.rot_z(0.4)
    T = geo.rigid_fit(src, src @ R.T + 1.0)
    assert np.allclose(T[:3, :3], R) and np.allclose(T[:3, 3], 1.0)
    assert np.linalg.det(geo.rigid_fit(src, reflected)[:3, :3]) == pytest.approx(1.0)
    T = geo.rigid_fit(line, line + 2.0, R)  # collinear: translation only, rotation kept
    assert np.allclose(T[:3, :3], R) and np.allclose(T @ np.r_[line[1], 1], np.r_[line[1] + 2, 1])
    assert np.allclose(geo.rigid_fit(line[:2], line[:2])[:3, :3], np.eye(3))
    with pytest.raises(ValueError, match="two point pairs"):
        geo.umeyama(line[:1], line[:1])
    with pytest.raises(ValueError, match="coincide"):
        geo.umeyama(np.zeros((3, 3)), line)


# ---- floor plan -----------------------------------------------------------------------------
def plan() -> fp.FloorPlan:
    return fp.load_house(synth.house(), ["Bowl"], floor_z=0.0)


def test_house_reader() -> None:
    p = plan()
    assert len(p.rooms) == 2 and len(p.doors) == 1
    door = p.doors[0]
    assert np.allclose(door.centre, [4.0, 1.45]) and door.width == pytest.approx(0.9)
    labels = {o.label: o for o in p.objects}
    assert labels["Mug"].parent == "DiningTable|1|1" and not labels["Mug"].inside
    assert labels["Apple"].inside and labels["Apple"].parent == "Bowl|1|3"
    assert labels["Sofa"].support_z == 0.0 and labels["Mug"].support_z is None
    assert np.allclose(labels["Mug"].centre, [2.0, 2.5, 0.85])
    # both copies of the door wall are cut
    on_door_line = [(a, b) for a, b in p.walls if abs(a[0] - 4) < 1e-9 and abs(b[0] - 4) < 1e-9]
    assert len(on_door_line) == 4
    assert fp.ray_cast(p, np.array([2.0, 1.45]), 0.0) == pytest.approx(5.0)  # through the door
    assert fp.ray_cast(p, np.array([2.0, 0.5]), 0.0) == pytest.approx(2.0)
    assert fp.ray_cast(p, np.array([2.0, 0.5]), 0.0, max_m=1.0) is None


def test_house_variants_and_errors() -> None:
    walls = [{"id": "wall|1|0.00|0.00|2.00|0.00"},  # no polygon: ends from the id
             {"id": "w2", "polygon": [synth.thor(0, 0), synth.thor(0, 0, 2), synth.thor(0, 2)]}]
    gaps = [{"wall0": "wall|1|0.00|0.00|2.00|0.00",
             "holePolygon": [{"x": 0.0, "y": 0}, {"x": 0.5, "y": 2}]},  # at the wall start
            {"wall0": "wall|1|0.00|0.00|2.00|0.00",
             "holePolygon": [{"x": 1.5, "y": 0}, {"x": 2.0, "y": 2}]}]  # at its end
    p = fp.load_house({"rooms": [{"floorPolygon": [synth.thor(0, 0), synth.thor(2, 0),
                                                   synth.thor(2, 2)]}],
                       "walls": walls, "doors": gaps}, [])
    xs = sorted((round(a[0], 3), round(b[0], 3)) for a, b in p.walls if a[1] == b[1] == 0)
    assert xs == [(0.5, 1.5)]
    assert fp._cut((np.array([0.0, 0]), np.array([1.0, 0])),
                   [fp.Door("d", np.array([2.0, 0]), np.array([3.0, 0]))]) != []
    with pytest.raises(rpl_layout.LayoutError, match="polygon"):
        fp.load_house({"walls": [{"id": "x"}]}, [])
    flat = {"id": "wall|1|0|0|1|0", "polygon": [synth.thor(0, 0), synth.thor(0, 0, 2)]}
    assert np.allclose(fp._wall_ends(flat, "w")[1], [1, 0])  # a degenerate polygon: the id
    with pytest.raises(rpl_layout.LayoutError, match="not a wall id"):
        fp.load_house({"walls": walls, "doors": [{"wall0": "nope", "holePolygon": []}]}, [])
    r = fp.rasterise(fp.FloorPlan(rooms=[np.array([[0, 0], [1, 0], [1, 1.0]])]))
    assert fp.WALL not in r.classes


def test_raster_classes() -> None:
    r = fp.rasterise(plan())
    pts = np.array([[2.0, 1.5], [0.0, 1.5], [4.0, 1.45], [0.12, 1.5], [-3, -3], [4.0, 0.5]])
    assert list(r.classify(pts)) == [fp.FLOOR, fp.WALL, fp.DOOR, fp.BORDER, fp.OUTSIDE, fp.WALL]
    d = fp.poly_distance(np.array([[0.5, 0.5], [5.0, 0.5]]), np.array([[0, 0], [1, 0], [1, 1],
                                                                        [0, 1.0]]))
    assert d[0] == 0 and d[1] == pytest.approx(4.0)


# ---- validation -----------------------------------------------------------------------------
def test_validator_full_and_minimal() -> None:
    v = rpl_validate.Validator()
    assert v.full
    assert v.errors("Health", {}) != []
    assert v.errors("Nope", {}) == ["unknown schema definition 'Nope'"]
    m = rpl_validate.Validator(use_jsonschema=False)
    assert not m.full
    assert m.errors("Frames", {"schema": "oms/2.1", "map": "m", "map_version": 1, "frames": [],
                               "scale_source": "priors", "scale_sigma_rel": 0}) == []
    errs = m.errors("Frames", {"schema": "x/1"})
    assert any("scale_source" in e for e in errs) and any("does not match" in e for e in errs)
    assert m.errors("Health", [1]) == ["<root>: expected an object for Health"]
    m.defs["Mixed"] = {"required": ["a"], "allOf": [{"type": "object"}]}
    assert m.errors("Mixed", {"a": 1}) == []


def test_validate_without_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    mod = importlib.reload(rpl_validate)
    try:
        assert mod.jsonschema is None and not mod.Validator().full
    finally:
        monkeypatch.undo()
        importlib.reload(rpl_validate)


# ---- API client -----------------------------------------------------------------------------
def test_api_errors_and_retries() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    api = rpl_api.Api(f"http://127.0.0.1:{port}", rpl_validate.Validator(), retries=1,
                      retry_wait_s=0.0)
    with pytest.raises(rpl_api.ApiError) as e:
        api.call("GET", "/api/health", "GET health", "Health")
    assert e.value.status == 0 and e.value.code == "connection"
    with Fake(Config()) as fake:
        api = rpl_api.Api(fake.url, rpl_validate.Validator())
        with pytest.raises(rpl_api.ApiError) as e:
            api.call("GET", "/api/maps/none", "GET map", "MapSummary")
        assert e.value.code == "map_not_found" and not e.value.retryable
        assert api.log.calls[-1].schema == "ErrorBody" and api.log.calls[-1].schema_errors == []
        clone = api.clone()
        assert clone.log is api.log and clone.session is not api.session


def test_standalone_imports() -> None:
    """The tool imports only the standard library, numpy, requests, optional jsonschema and its
    own modules: it runs from a bare checkout, never importing the service."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"numpy", "requests", "jsonschema"}
    folder = Path(rpl_api.__file__).parent
    own = {p.stem for p in folder.glob("*.py")}
    for path in folder.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                top = n.split(".")[0]
                assert top in allowed or top in own, f"{path.name} imports {n}"
