"""reconstruct.sh / segment.sh logic in-process (fake client, captured payload): formats, ``-o``,
``-d`` artefacts, ``-p`` point-cloud attributes, and the reconstruct/segment JSON agreement."""

from __future__ import annotations

import io
import json
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
from oh_my_slam.segmentation.colors import UNSEGMENTED, hex_to_rgb
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
    img = client.add(tmp_path / "room.png", r.rgb, FakeFrame(r.depth, K, up, inst, pose=pose))
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


def test_reconstruct_output_file_and_option_errors(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, client = env
    target = tmp_path / "out" / "cloud.ply"
    assert cli_reconstruct.main(["-i", str(img), "-f", "ply", "-o", str(target)]) == 0
    assert cap.take() == b"" and len(parse_ply(target.read_bytes())) > 1000
    assert cli_reconstruct.main(["-i", str(img), "-o", str(tmp_path / "scene.json")]) == 0
    assert cap.take() == b"" and json.loads((tmp_path / "scene.json").read_text())["openlabel"]
    client.calls.clear()
    for args in (["-p", "voxel=0.1"],  # -p needs -f ply
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
    assert cli_segment.main(["-i", str(img), "-f", "ply", "-d", str(out), "-p",
                             "voxel=0.02,normals=on"]) == 0
    ply = cap.take()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)
    assert (out / "segments.ply").read_bytes() == ply  # byte-identical to -f ply
    cloud = parse_ply(ply)
    assert cloud.label is None and cloud.normals is not None  # label property off by default
    assert parse_header(ply).comments[1].startswith("attributes color=segment,")
    assert cli_segment.main(["-i", str(img), "-d", str(out)]) == 0
    payload = cap.take()
    assert (out / "segmentation.json").read_bytes() == payload  # identical to -f json
    assert cli_segment.main(["-i", str(img), "-f", "ply"]) == 0
    assert (out / "segments.ply").read_bytes() == cap.take()


def test_attributes_never_change_objects_ids_or_colours(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_segment.main(["-i", str(img)]) == 0
    plain = cap.take()
    assert cli_segment.main(["-i", str(img), "-d", str(tmp_path / "a"), "-p",
                             "stride=3,edge=0,voxel=0.1,normals=on,label=on,encoding=ascii,"
                             "min-depth=1,max-depth=3"]) == 0
    assert cap.take() == plain
    assert (tmp_path / "a" / "segmentation.json").read_bytes() == plain


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


def test_segment_output_file_and_no_files_by_default(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    before = set(tmp_path.rglob("*"))
    assert cli_segment.main(["-i", str(img)]) == 0
    assert set(tmp_path.rglob("*")) == before  # without -o and -d nothing is written
    plain = cap.take()
    target = tmp_path / "scene.json"
    assert cli_segment.main(["-i", str(img), "-o", str(target)]) == 0
    assert cap.take() == b"" and target.read_bytes() == plain
    assert set(tmp_path.rglob("*")) == before | {target}


def test_segment_usage_errors(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, _cap, client = env
    for args in (["-m", str(tmp_path), "--min-score", "0.6"],
                 ["-i", str(img), "--min-score", "abc"],
                 ["-i", str(img), "--min-score", "1.5"],
                 ["-i", str(img), "--min-score", "0.1"],  # below the detection floor
                 ["-i", str(img), "-p", "voxel=0.1"],  # -p needs -f ply or -d
                 ["-i", str(img), "-f", "ply", "-p", "color=rgb"],  # colour fixed to segment
                 ["-m", str(tmp_path), "-f", "ply", "-p", "stride=2"]):  # pixel-level on a map
        with pytest.raises(UsageError):
            cli_segment.main(args)
    assert sum(client.calls.values()) == 0
    for args in (["-i", str(img), "-m", str(tmp_path)], []):
        with pytest.raises(SystemExit) as e:
            cli_segment.main(args)
        assert e.value.code == 2


def test_spec_usage_lines_parse_with_defaults() -> None:
    r = cli_reconstruct.build_parser().parse_args(["-i", "x.jpg"])
    assert (r.format, r.output, r.attrs) == ("json", None, None)
    r = cli_reconstruct.build_parser().parse_args(
        ["-i", "x.jpg", "-f", "ply", "-p", "color=segment,voxel=0.01,normals=on", "-o", "c.ply"])
    assert (r.format, r.output, r.attrs) == ("ply", Path("c.ply"),
                                             ["color=segment,voxel=0.01,normals=on"])
    s = cli_segment.build_parser().parse_args(["-i", "x.jpg"])
    assert (s.format, s.output, s.artifacts, s.attrs, s.min_score) == (
        "json", None, None, None, None)
    s = cli_segment.build_parser().parse_args(
        ["-i", "x.jpg", "-f", "ply", "-o", "o.ply", "-d", "d", "-p", "voxel=0.1",
         "--min-score", "0.3"])
    assert (s.format, s.output, s.artifacts, s.attrs, s.min_score) == (
        "ply", Path("o.ply"), Path("d"), ["voxel=0.1"], "0.3")
    s = cli_segment.build_parser().parse_args(
        ["-m", "map", "-f", "json", "-o", "o.json", "-d", "d", "-p", "label=on"])
    assert s.map == Path("map") and s.artifacts == Path("d")
    with pytest.raises(SystemExit) as e:
        ArgumentParser(prog="x").parse_args(["--bad"])
    assert e.value.code == 2
