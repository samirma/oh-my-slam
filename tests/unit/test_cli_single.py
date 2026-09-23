"""reconstruct.sh / segment.sh logic in-process (fake client, captured payload)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.cli import reconstruct as cli_reconstruct
from oh_my_slam.cli import segment as cli_segment
from oh_my_slam.cli.common import ArgumentParser, parse_labels
from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.log import PayloadWriter
from oh_my_slam.core.ply import parse_ply
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


class Capture:
    def __init__(self) -> None:
        self.buf = io.BytesIO()
        self.writer = PayloadWriter(self.buf)

    def __call__(self) -> PayloadWriter:
        return self.writer


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


def test_reconstruct_json_and_ply(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_reconstruct.main(["-i", str(img)]) == 0
    doc = json.loads(cap.buf.getvalue())
    assert validation_errors(doc) == []
    types = sorted(o["type"] for o in doc["openlabel"]["objects"].values())
    assert types == ["box", "cabinet", "sofa"]
    for o in doc["openlabel"]["objects"].values():
        assert o["object_data"]["num"][0]["val"] >= 0.5


def test_reconstruct_ply(env) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    assert cli_reconstruct.main(["-i", str(img), "-f", "ply"]) == 0
    cloud = parse_ply(cap.buf.getvalue())
    assert len(cloud) > 50000 and cloud.label is None


def test_reconstruct_missing_image(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InputError):
        cli_reconstruct.main(["-i", str(tmp_path / "nope.jpg")])


def test_segment_with_artifacts_labels_and_min_score(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, cap, _ = env
    out = tmp_path / "seg"
    assert cli_segment.main(["-i", str(img), "-o", str(out)]) == 0
    payload = cap.buf.getvalue()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)
    assert (out / "segmentation.json").read_bytes() == payload
    full = json.loads(payload)["openlabel"]["objects"]

    cap.__init__()
    assert cli_segment.main(["-i", str(img), "--labels", "sofa,espresso machine"]) == 0
    only = json.loads(cap.buf.getvalue())["openlabel"]["objects"]
    assert {o["type"] for o in only.values()} == {"sofa"}

    cap.__init__()
    assert cli_segment.main(["-i", str(img), "--min-score", "0.8"]) == 0
    hi = json.loads(cap.buf.getvalue())["openlabel"]["objects"]
    assert set(hi) <= set(full)
    for k, o in hi.items():
        assert o["type"] == full[k]["type"]
        assert o["object_data"]["num"][0]["val"] >= 0.8

    cap.__init__()
    assert cli_segment.main(["-i", str(img), "-f", "ply"]) == 0
    assert parse_ply(cap.buf.getvalue()).label is not None


def test_segment_without_o_writes_nothing(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, _cap, _ = env
    before = set(tmp_path.rglob("*"))
    assert cli_segment.main(["-i", str(img)]) == 0
    assert set(tmp_path.rglob("*")) == before


def test_segment_usage_errors(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    img, _cap, _ = env
    with pytest.raises(UsageError):
        cli_segment.main(["-m", str(tmp_path), "--labels", "chair"])
    with pytest.raises(UsageError):
        cli_segment.main(["-m", str(tmp_path), "--min-score", "0.3"])
    with pytest.raises(UsageError):
        cli_segment.main(["-i", str(img), "--min-score", "abc"])
    with pytest.raises(UsageError):
        cli_segment.main(["-i", str(img), "--min-score", "1.5"])
    with pytest.raises(SystemExit) as e:
        cli_segment.main(["-i", str(img), "-m", str(tmp_path)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli_segment.main([])
    assert e.value.code == 2


def test_spec_usage_lines_parse_with_defaults() -> None:
    r = cli_reconstruct.build_parser().parse_args(["-i", "x.jpg"])
    assert r.format == "json"
    assert cli_reconstruct.build_parser().parse_args(["-i", "x.jpg", "-f", "ply"]).format == "ply"
    s = cli_segment.build_parser().parse_args(["-i", "x.jpg"])
    assert (s.format, s.out, s.min_score, s.labels) == ("json", None, None, None)
    s = cli_segment.build_parser().parse_args(
        ["-i", "x.jpg", "-o", "o", "-f", "ply", "--min-score", "0.3", "--labels", "a,b,c"])
    assert parse_labels(s.labels) == ["a", "b", "c"] and s.format == "ply"
    s = cli_segment.build_parser().parse_args(["-m", "map", "-o", "o", "-f", "json"])
    assert s.map == Path("map")
    assert parse_labels(" , ") is None and parse_labels(None) is None
    with pytest.raises(SystemExit) as e:
        ArgumentParser(prog="x").parse_args(["--bad"])
    assert e.value.code == 2
