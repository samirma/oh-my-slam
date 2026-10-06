"""reconstruct.sh / segment.sh logic in-process (fake client, captured payload): formats (the
scene, the depth image, the point cloud, the segmented image), ``-o``, ``-d`` artefacts, ``-p``
point-cloud attributes, and the reconstruct/segment JSON agreement."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.cli import reconstruct as cli_reconstruct
from oh_my_slam.cli import segment as cli_segment
from oh_my_slam.cli.common import ArgumentParser
from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.log import PayloadWriter
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, hex_to_rgb
from oh_my_slam.segmentation.render import DIM
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


class Capture:
    """Stands in for ``claim_stdout``: stdout is a buffer, ``-o`` goes to the real file writer."""

    def __init__(self) -> None:
        self.buf = io.BytesIO()
        self.writer = PayloadWriter(self.buf)

    def __call__(self, output: Path | None = None) -> PayloadWriter:
        return self.writer if output is None else PayloadWriter(path=output)

    def take(self) -> bytes:
        data = self.buf.getvalue()
        self.__init__()  # type: ignore[misc]
        return data


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    room = default_room()
    pose = look_at(np.array([2.6, 2.2, 1.6]), np.array([0.0, 0.0, 0.4]))
    r = render(room, pose, K)
    inst = [FakeInstance(b.label, 0.9 - 0.15 * k, r.ids == k + 2)
            for k, b in enumerate(room.boxes)]
    client = FakeClient()
    up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    depth = r.depth.copy()
    depth[:4, :6] = 0  # no valid depth there (the fake's mask: depth > 0)
    img = client.add(tmp_path / "room.png", r.rgb, FakeFrame(depth, K, up, inst, pose=pose))
    import oh_my_slam.client.client as cc

    monkeypatch.setattr(cc, "connect", lambda require=True: client)
    monkeypatch.setattr(cli_reconstruct, "connect", lambda require=True: client)
    cap = Capture()
    monkeypatch.setattr(cli_reconstruct, "claim_stdout", cap)
    monkeypatch.setattr(cli_segment, "claim_stdout", cap)
    return img, cap, client


def _without_tool(doc: dict[str, Any]) -> dict[str, Any]:
    doc = json.loads(json.dumps(doc))
    doc["openlabel"]["metadata"].pop("tool")
    return doc


def test_reconstruct_json(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_reconstruct.main(["-i", str(img)]) == 0
    doc = json.loads(cap.take())
    assert validation_errors(doc) == []
    types = sorted(o["type"] for o in doc["openlabel"]["objects"].values())
    assert types == ["box", "cabinet", "sofa"]
    for o in doc["openlabel"]["objects"].values():
        assert o["object_data"]["num"][0]["val"] >= 0.5


def test_reconstruct_json_matches_segment_with_default_options(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_reconstruct.main(["-i", str(img)]) == 0
    rec = json.loads(cap.take())
    assert cli_segment.main(["-i", str(img)]) == 0
    seg = json.loads(cap.take())
    assert rec["openlabel"]["metadata"]["tool"] == "reconstruct"
    assert seg["openlabel"]["metadata"]["tool"] == "segment"
    assert _without_tool(rec) == _without_tool(seg)  # same objects, ids, colours and OBBs


def test_reconstruct_ply_runs_only_the_inference_its_attributes_need(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, client = env
    for attrs, gravity, segment in [(None, 0, 0), ("color=height", 1, 0),
                                    ("color=segment", 1, 1), ("label=on", 1, 1),
                                    ("color=none,normals=on,voxel=0.05", 0, 0)]:
        client.calls.clear()
        args = ["-i", str(img), "-f", "ply"] + ([] if attrs is None else ["-p", attrs])
        assert cli_reconstruct.main(args) == 0
        data = cap.take()
        assert (client.calls["geometry"], client.calls["gravity"], client.calls["segment"]) == (
            1, gravity, segment), attrs
        cloud = parse_ply(data)
        assert len(cloud) > 1000
        assert (cloud.label is not None) == (attrs == "label=on")
        assert (cloud.rgb is None) == (attrs is not None and "color=none" in attrs)
        assert (cloud.normals is not None) == (attrs is not None and "normals=on" in attrs)
    head = parse_header(data)
    assert head.comments[1] == ("attributes color=none,stride=1,min-depth=0,max-depth=inf,"
                                "edge=0.04,voxel=0.05,normals=on,label=off,encoding=binary")


def test_reconstruct_segment_ply_follows_the_colour_contract(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_reconstruct.main(["-i", str(img)]) == 0
    objects = json.loads(cap.take())["openlabel"]["objects"]
    assert cli_reconstruct.main(["-i", str(img), "-f", "ply", "-p",
                                 "color=segment,label=on,encoding=ascii"]) == 0
    cloud = parse_ply(cap.take())
    assert cloud.label is not None and cloud.rgb is not None
    assert set(np.unique(cloud.label)) == {0} | {int(k) for k in objects}
    for key, o in objects.items():
        triple = tuple(o["object_data"]["vec"][0]["val"])
        assert triple == hex_to_rgb(o["object_data"]["text"][0]["val"])
        np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == int(key)], axis=0),
                                      [triple])
    np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == 0], axis=0), [UNSEGMENTED])


def decode_png(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im)


def test_reconstruct_depth_is_the_models_depth_as_a_16_bit_png(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """-f depth: one 16-bit single-channel PNG of the input's pixel size, each pixel the depth in
    1/256 m (README), 0 where the model gives none; only the depth model runs, and no point-cloud
    attribute applies (the flying pixels the PLY drops are kept)."""
    img, cap, client = env
    client.calls.clear()
    assert cli_reconstruct.main(["-i", str(img), "-f", "depth"]) == 0
    data = cap.take()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert dict(client.calls) == {"geometry": 1}  # no gravity, no segmentation
    values = decode_png(data)
    assert values.dtype == np.uint16 and values.shape == (K.height, K.width)
    frame = client.frames[str(img.resolve())]
    valid = frame.depth > 0
    assert not valid.all() and (values[~valid] == 0).all()
    np.testing.assert_array_equal(values[valid], np.rint(frame.depth[valid] * 256))
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        assert im.info["Description"].endswith("metres = value / 256; 0 = no valid depth")
    assert cli_reconstruct.main(["-i", str(img), "-f", "ply", "-p", "edge=0"]) == 0
    assert len(parse_ply(cap.take())) == int(valid.sum())  # every valid pixel, as in the PNG
    target = tmp_path / "depth.png"
    assert cli_reconstruct.main(["-i", str(img), "-f", "depth", "-o", str(target)]) == 0
    assert cap.take() == b"" and target.read_bytes() == data
    client.calls.clear()
    with pytest.raises(UsageError, match="only the PLY output has: use -f ply"):
        cli_reconstruct.main(["-i", str(img), "-f", "depth", "-p", "voxel=0.1"])
    assert sum(client.calls.values()) == 0  # refused before any inference


def test_reconstruct_output_file_and_option_errors(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, client = env
    target = tmp_path / "out" / "cloud.ply"
    assert cli_reconstruct.main(["-i", str(img), "-f", "ply", "-o", str(target)]) == 0
    assert cap.take() == b"" and len(parse_ply(target.read_bytes())) > 1000
    assert cli_reconstruct.main(["-i", str(img), "-o", str(tmp_path / "scene.json")]) == 0
    assert cap.take() == b"" and json.loads((tmp_path / "scene.json").read_text())["openlabel"]
    client.calls.clear()
    for args in (["-p", "voxel=0.1"],  # -p needs -f ply
                 ["-f", "depth", "-p", "voxel=0.1"],
                 ["-f", "ply", "-p", "colour=rgb"], ["-f", "ply", "-p", "stride=0"],
                 ["-f", "ply", "-o", str(tmp_path)]):  # -o is a folder
        with pytest.raises(UsageError):
            cli_reconstruct.main(["-i", str(img), *args])
    assert sum(client.calls.values()) == 0  # rejected before any inference
    with pytest.raises(InputError):
        cli_reconstruct.main(["-i", str(tmp_path / "nope.jpg")])


def test_segment_artifacts_are_the_outputs_of_the_same_run(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    out = tmp_path / "seg"
    assert cli_segment.main(["-i", str(img), "-f", "png", "-d", str(out)]) == 0
    png = cap.take()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES) == [
        "catalog.csv", "catalog.md", "segmentation.json", "segmented.png"]
    assert (out / "segmented.png").read_bytes() == png  # byte-identical to -f png
    scene = (out / "segmentation.json").read_bytes()
    assert cli_segment.main(["-i", str(img), "-d", str(out)]) == 0
    payload = cap.take()
    assert (out / "segmentation.json").read_bytes() == payload == scene  # identical to -f json
    assert (out / "segmented.png").read_bytes() == png
    assert cli_segment.main(["-i", str(img), "-f", "png"]) == 0
    assert cap.take() == png


def test_segment_png_is_the_segmented_image(env) -> None:  # type: ignore[no-untyped-def]
    """-f png: the input image (on the reconstruction's grid) dimmed, each object's mask painted
    opaque in exactly its colour (§2.4 colour contract), and no other colour."""
    img, cap, _ = env
    assert cli_segment.main(["-i", str(img)]) == 0
    objects = json.loads(cap.take())["openlabel"]["objects"]
    assert cli_segment.main(["-i", str(img), "-f", "png"]) == 0
    rgb = decode_png(cap.take())
    assert rgb.dtype == np.uint8 and rgb.shape == (K.height, K.width, 3)
    colours = {tuple(c) for c in np.unique(rgb.reshape(-1, 3), axis=0).tolist()}
    painted = {c for c in colours if max(c) > int(255 * DIM)}
    assert painted == {color_for_id(int(k)) for k in objects}  # exact, never blended
    for key, o in objects.items():
        pixels = int((rgb == color_for_id(int(key))).all(axis=2).sum())
        assert pixels == o["object_data"]["num"][1]["val"]  # its pixel_count


# the catalog.csv header, literally as specs/segment.md states it
SPEC_CSV_HEADER = ("id,label,score,color_hex,width_m,height_m,depth_m,volume_m3,center_x,center_y,"
                   "center_z,pixel_count,point_count")
MD_ROW = re.compile(r'^\| <span style="color:(#[0-9a-f]{6})">&#9632;</span> \| (\d+) \| .*? \| '
                    r"`(#[0-9a-f]{6})` \|")


def assert_catalogue_colours(folder: Path) -> None:
    """The colour contract across the files ``segment.sh -d`` wrote: for every object of
    ``segmentation.json``, its ``color_hex`` there, the ``color_hex`` column of ``catalog.csv``
    and both the swatch colour and the hex of its ``catalog.md`` row are one value; the CSV header
    is the spec's literal and neither catalogue has an object the scene lacks."""
    import csv

    objects = json.loads((folder / "segmentation.json").read_text())["openlabel"]["objects"]
    scene = {}
    for key, o in objects.items():
        text = {t["name"]: t["val"] for t in o["object_data"]["text"]}
        vec = {v["name"]: v["val"] for v in o["object_data"]["vec"]}
        assert hex_to_rgb(text["color_hex"]) == tuple(vec["color"]), key
        scene[int(key)] = text["color_hex"]
    assert scene
    lines = (folder / "catalog.csv").read_text().splitlines()
    assert lines[0] == SPEC_CSV_HEADER
    rows = list(csv.DictReader(lines))
    assert {int(r["id"]): r["color_hex"] for r in rows} == scene
    md = [m.groups() for line in (folder / "catalog.md").read_text().splitlines()
          if (m := MD_ROW.match(line))]
    assert {int(i): swatch for swatch, i, _ in md} == scene
    assert {int(i): code for _, i, code in md} == scene


def test_catalogue_colours_are_the_scene_colours(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    out = tmp_path / "seg"
    assert cli_segment.main(["-i", str(img), "-d", str(out)]) == 0
    cap.take()
    assert_catalogue_colours(out)


def test_segment_min_score_keeps_ids_and_colours(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_segment.main(["-i", str(img)]) == 0
    full = json.loads(cap.take())["openlabel"]["objects"]
    assert cli_segment.main(["-i", str(img), "--min-score", "0.8"]) == 0
    hi = json.loads(cap.take())["openlabel"]["objects"]
    assert hi and set(hi) < set(full)
    for k, o in hi.items():
        assert o == full[k]  # same id, label, colour and OBB
        assert o["object_data"]["num"][0]["val"] >= 0.8
    # the spec sets no lower bound: a low threshold is accepted and only adds objects
    assert cli_segment.main(["-i", str(img), "--min-score", "0.1"]) == 0
    low = json.loads(cap.take())["openlabel"]["objects"]
    assert {k: low[k] for k in full} == full
    # nor any bound at all: below the detector's own floor, or above 1 (no object left)
    assert cli_segment.main(["-i", str(img), "--min-score", "0.01"]) == 0
    assert set(json.loads(cap.take())["openlabel"]["objects"]) >= set(full)
    assert cli_segment.main(["-i", str(img), "--min-score", "1.5"]) == 0
    assert not json.loads(cap.take())["openlabel"].get("objects")


def test_segment_output_file_and_no_files_by_default(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    before = set(tmp_path.rglob("*"))
    assert cli_segment.main(["-i", str(img)]) == 0
    assert set(tmp_path.rglob("*")) == before  # without -o and -d nothing is written
    plain = cap.take()
    target = tmp_path / "scene.json"
    assert cli_segment.main(["-i", str(img), "-o", str(target)]) == 0
    assert cap.take() == b"" and target.read_bytes() == plain
    assert cli_segment.main(["-i", str(img), "-f", "png"]) == 0
    png = cap.take()
    image = tmp_path / "out" / "segmented.png"
    assert cli_segment.main(["-i", str(img), "-f", "png", "-o", str(image)]) == 0
    assert cap.take() == b"" and image.read_bytes() == png
    assert set(tmp_path.rglob("*")) == before | {target, image.parent, image}


def test_segment_usage_errors(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, _cap, client = env
    for args in (["-i", str(img), "--min-score", "abc"],
                 ["-i", str(img), "--min-score", "nan"],
                 ["-i", str(img), "-f", "png", "-o", str(tmp_path)]):  # -o is a folder
        with pytest.raises(UsageError):
            cli_segment.main(args)
    assert sum(client.calls.values()) == 0
    # segment.sh takes -i only: no map (-m), no point cloud (-f ply, -p)
    for args in (["-i", str(img), "-m", str(tmp_path)], ["-m", str(tmp_path)], [],
                 ["-i", str(img), "-f", "ply"], ["-i", str(img), "-p", "voxel=0.1"]):
        with pytest.raises(SystemExit) as e:
            cli_segment.main(args)
        assert e.value.code == 2
    assert sum(client.calls.values()) == 0


def test_spec_usage_lines_parse_with_defaults() -> None:
    r = cli_reconstruct.build_parser().parse_args(["-i", "x.jpg"])
    assert (r.format, r.output, r.attrs) == ("json", None, None)
    r = cli_reconstruct.build_parser().parse_args(
        ["-i", "x.jpg", "-f", "ply", "-p", "color=height,voxel=0.01,normals=on", "-o", "c.ply"])
    assert (r.format, r.output, r.attrs) == ("ply", Path("c.ply"),
                                             ["color=height,voxel=0.01,normals=on"])
    r = cli_reconstruct.build_parser().parse_args(["-i", "x.jpg", "-f", "depth", "-o", "d.png"])
    assert (r.format, r.output, r.attrs) == ("depth", Path("d.png"), None)
    s = cli_segment.build_parser().parse_args(["-i", "x.jpg"])
    assert (s.format, s.output, s.artifacts, s.min_score) == ("json", None, None, None)
    s = cli_segment.build_parser().parse_args(
        ["-i", "x.jpg", "-f", "png", "-o", "o.png", "-d", "d", "--min-score", "0.3"])
    assert (s.format, s.output, s.artifacts, s.min_score) == (
        "png", Path("o.png"), Path("d"), "0.3")
    assert not {"map", "attrs"} & set(vars(s))
    from oh_my_slam.cli import mapper as cli_mapper

    m = cli_mapper.build_parser().parse_args(["locate", "-i", "x.jpg", "-m", "map"])
    assert (m.command, m.inputs, m.map, m.format, m.output, m.attrs, m.mode) == (
        "locate", [Path("x.jpg")], Path("map"), "json", None, None, "single")
    m = cli_mapper.build_parser().parse_args(
        ["locate", "-i", "x.jpg", "y.jpg", "-m", "map", "-f", "ply", "-o", "o.ply", "-p",
         "voxel=0.1", "-t", "full"])
    assert (m.inputs, m.format, m.output, m.attrs, m.mode) == (
        [Path("x.jpg"), Path("y.jpg")], "ply", Path("o.ply"), ["voxel=0.1"], "full")
    u = cli_mapper.build_parser().parse_args(["update", "-i", "x.jpg", "-m", "map"])
    assert (u.command, u.format, u.mode, u.fps) == ("update", "json", "full", None)
    with pytest.raises(SystemExit) as e:
        ArgumentParser(prog="x").parse_args(["--bad"])
    assert e.value.code == 2
