"""The commands' single source of truth (``cli/spec.py``, spec §2.6 "Single source of truth"):
every argparse parser is built from the registry and its export covers every option; the
validation rules raise the commands' own errors; stage names are registered; the exit-code →
HTTP rule; live progress events (``core.timing``)."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from oh_my_slam.cli import mapper as cli_mapper
from oh_my_slam.cli import reconstruct as cli_reconstruct
from oh_my_slam.cli import segment as cli_segment
from oh_my_slam.cli import spec
from oh_my_slam.cli import view as cli_view
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    ExitCode,
    InputError,
    NotAMapError,
    UsageError,
    error_code,
    http_status,
)
from oh_my_slam.core.timing import Stage

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "oh_my_slam"
CLIS = {"reconstruct.sh": cli_reconstruct, "mapper.sh": cli_mapper, "segment.sh": cli_segment,
        "view.sh": cli_view}


def _leaf_parsers(ap: argparse.ArgumentParser) -> dict[str | None, argparse.ArgumentParser]:
    for a in ap._actions:
        if isinstance(a, argparse._SubParsersAction):
            return dict(a.choices.items())
    return {None: ap}


def _options(ap: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in ap._actions if a.option_strings and a.dest != "help"]


def test_every_command_parser_is_the_registry() -> None:
    """Each CLI's parser is built from the registry: same options, defaults, choices, help."""
    assert set(CLIS) == {p.prog for p in spec.PROGRAMS}
    for program in spec.PROGRAMS:
        assert CLIS[program.prog].PROGRAM is program
        leaves = _leaf_parsers(CLIS[program.prog].build_parser())
        assert set(leaves) == {c.name for c in program.commands}
        for cmd in program.commands:
            actions = _options(leaves[cmd.name])
            assert [a.option_strings for a in actions] == [[o.flag] for o in cmd.options]
            for a, o in zip(actions, cmd.options, strict=True):
                assert (a.dest, a.help, a.required) == (o.name, o.help, o.required)
                assert a.choices == (o.choices if o.kind is spec.Kind.ENUM else None)
                if o.kind is spec.Kind.ENUM:
                    assert a.default == o.default


def test_describe_covers_every_option_of_every_parser() -> None:
    """describe() is JSON and lists, per operation, exactly the options its parser accepts in
    that mode (the other mode's selector excluded)."""
    d = json.loads(json.dumps(spec.describe()))
    ops = {o["id"]: o for o in d["operations"]}
    assert set(ops) == {"reconstruct.sh", "mapper.sh update", "mapper.sh locate",
                        "segment.sh -i", "segment.sh -m", "view.sh -i", "view.sh -m"}
    for program in spec.PROGRAMS:
        leaves = _leaf_parsers(CLIS[program.prog].build_parser())
        for cmd in program.commands:
            flags = {a.option_strings[0]: a for a in _options(leaves[cmd.name])}
            for mode in cmd.modes:
                op = ops[cmd.label(mode)]
                others = {cmd.option(m.selector).flag for m in cmd.modes
                          if m is not mode and m.selector}
                expected = {f for f in flags if f not in others}
                if mode.name == "map":
                    expected.discard("--min-score")
                assert {p["flag"] for p in op["parameters"]} == expected, op["id"]
                for p in op["parameters"]:
                    a = flags[p["flag"]]
                    assert (p["name"], p["help"]) == (a.dest, a.help)
                    assert p["required"] == (a.required or p["name"] == mode.selector)
                assert op["stages"] == [str(s) for s in mode.stages]
                assert all(s in d["stages"] for s in op["stages"])
                codes = {e["code"] for e in op["errors"]}
                assert "usage" in codes
                assert ("server_unavailable" in codes) == (op["inference"] != "never"), op["id"]
    seg_i = ops["segment.sh -i"]
    p = {x["name"]: x for x in seg_i["parameters"]}
    assert p["min_score"]["default"] == 0.5 and p["format"]["choices"] == ["json", "ply"]
    assert p["attrs"]["default"].startswith("color=segment,stride=1")
    assert p["attrs"]["applies"] == [{"option": "format", "in": ["ply"]},
                                     {"option": "artifacts", "is": "given"}]
    assert {x["key"] for x in p["attrs"]["attributes"]} >= {"stride", "edge"}
    seg_m = {x["name"]: x for x in ops["segment.sh -m"]["parameters"]}
    assert "stride" not in {x["key"] for x in seg_m["attrs"]["attributes"]}
    update = {x["name"]: x for x in ops["mapper.sh update"]["parameters"]}
    assert update["fps"]["default"] == 2.0 and update["fps"]["applies"] == [
        {"option": "inputs", "is": "video"}]
    assert {o["name"] for o in seg_i["outputs"]} == {
        "result", "segmentation.json", "segmented.png", "catalog.csv", "catalog.md",
        "segments.ply"}


def test_an_option_added_to_the_registry_reaches_parser_and_export(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = spec.RECONSTRUCT.commands[0]
    extra = spec.Option("--gain", "gain", spec.Kind.NUMBER, "a new option", type=float,
                        default=1.0)
    program = dataclasses.replace(spec.RECONSTRUCT, commands=(
        dataclasses.replace(cmd, options=(*cmd.options, extra)),))
    monkeypatch.setattr(spec, "PROGRAMS", (program,))
    assert spec.build_parser(program).parse_args(["-i", "x", "--gain", "2"]).gain == 2.0
    (op,) = spec.describe()["operations"]
    assert op["parameters"][-1] == {
        "name": "gain", "flag": "--gain", "kind": "number", "help": "a new option",
        "required": False, "default": 1.0, "choices": None, "multiple": False,
        "repeatable": False, "applies": [], "applies_text": ""}


def _validate(program: spec.Program, argv: list[str], **kw):  # type: ignore[no-untyped-def]
    args = spec.build_parser(program).parse_args(argv)
    cmd = program.command(args.command if program.subcommands else None)
    return spec.validate(cmd, args, **kw)


def test_validation_rules_raise_the_commands_errors(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    R, M, S, V = spec.RECONSTRUCT, spec.MAPPER, spec.SEGMENT, spec.VIEW
    cases = [
        (R, ["-i", str(img), "-p", "voxel=1"], UsageError, "only the PLY output has"),
        (R, ["-i", str(tmp_path / "no.jpg")], InputError, "image not found"),
        (R, ["-i", str(img), "-o", str(tmp_path)], UsageError, "is a folder"),
        (M, ["update", "-i", "v.mp4", "-m", "m", "-fps", "0"], UsageError, "-fps must be"),
        (M, ["update", "-i", "a.jpg", "-m", "m", "-f", "ply", "-p", "stride=2"], UsageError,
         "pixel-level"),
        (M, ["locate", "-i", "v.mp4", "-m", "m"], UsageError, "not a video"),
        (M, ["locate", "-i", str(img), "-m", str(tmp_path / "none")], InputError, "no map in"),
        (S, ["-m", "m", "--min-score", "0.4"], UsageError, "applies to -i only"),
        (S, ["-i", str(img), "--min-score", "nan"], UsageError, "finite number"),
        (S, ["-i", str(img), "-p", "voxel=1"], UsageError, "use -f ply or -d"),
        (V, ["-i", str(tmp_path / "no.jpg")], InputError, "image not found"),
    ]
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "f").write_text("x")
    cases.append((M, ["locate", "-i", str(img), "-m", str(tmp_path / "other")], NotAMapError,
                  "is not a map"))
    for program, argv, exc, message in cases:
        with pytest.raises(exc, match=re.escape(message)):
            _validate(program, argv)
    warnings: list[str] = []
    v = _validate(M, ["update", "-i", "a.jpg", "-m", "m", "-fps", "-3"], warn=warnings.append)
    assert v.fps == 2.0 and not v.is_video and warnings == [
        "-fps applies to video input only; ignored for images"]
    v = _validate(S, ["-i", str(img), "-d", str(tmp_path / "d"), "-p", "voxel=0.1",
                      "--min-score", "0.3"])
    assert v.min_score == 0.3 and v.attrs == CloudAttrs(color="segment", voxel=0.1)
    assert (tmp_path / "d").is_dir()  # -d is prepared before any work
    assert _validate(R, ["-i", str(img)]).attrs == CloudAttrs()


def test_exit_codes_map_to_http_statuses() -> None:
    assert set(HTTP_STATUS) == set(ExitCode)
    assert http_status(ExitCode.USAGE) == 400
    assert http_status(ExitCode.SERVER_UNAVAILABLE) == 503
    assert http_status(ExitCode.INTERNAL) == 500 and http_status(130) == 500
    for code in (ExitCode.NOT_A_MAP, ExitCode.NOT_REGISTERED, ExitCode.MAP_LOCKED):
        assert 400 <= http_status(code) < 500
    assert error_code(4) == "not_a_map" and error_code(130) == "internal"
    exported = {e["code"]: e["http_status"] for e in spec.describe()["exit_codes"]}
    assert exported["map_locked"] == 409 and exported["usage"] == 400


def _stage_literals() -> dict[str, set[str]]:
    """``stage("…")`` string literals per source file."""
    found: dict[str, set[str]] = {}
    for f in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text("utf-8"))):
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                owner = fn.value.id if isinstance(fn, ast.Attribute) and isinstance(
                    fn.value, ast.Name) else ""
                if name == "stage" and owner in ("", "timing"):
                    found.setdefault(str(f.relative_to(SRC)), set()).add(node.args[0].value)
    return found


def test_every_timing_stage_is_registered() -> None:
    """Every stage a command records is a ``Stage`` and listed for some operation; the commands
    and ``mapping/locate.py`` use the constants."""
    literals = _stage_literals()
    assert set().union(*literals.values()) <= {str(s) for s in Stage}, literals
    assert not any(f.startswith("cli/") or f == "mapping/locate.py" for f in literals), literals
    listed = {s for _p, _c, m in spec.operations() for s in m.stages}
    assert listed == set(Stage)


def test_progress_events_reach_listeners() -> None:
    events: list[dict] = []
    with timing.listen(events.append), timing.collect(sample_every=None):
        with timing.stage(Stage.EXPORT):
            timing.progress(1, 2)
            timing.count(keyframes_sampled=2)
            with timing.part("lift"):
                pass
    assert [e["event"] for e in events] == [
        "begin", "stage_start", "progress", "count", "part", "stage_end", "finish"]
    assert events[2] == {"event": "progress", "stage": "export", "done": 1, "total": 2}
    assert events[3] == {"event": "count", "keyframes_sampled": 2}
    timing.progress(1, 1)  # outside a collection: nothing
    assert len(events) == 7


def test_progress_file_keeps_stdout_and_stderr_unchanged(tmp_path: Path) -> None:
    """``OH_MY_SLAM_PROGRESS=<path>``: one JSON line per event in that file, and the command's
    stdout and human stderr are those of a run without it; mapping reports its sizes."""
    from oh_my_slam.mapping.api import update
    from tests.fakes.client import FakeClient
    from tests.synth.mapping import add_frames, mapping_room, ring

    client = FakeClient()
    imgs = add_frames(client, mapping_room(), ring(1), tmp_path / "in", "s")
    events: list[dict] = []
    with timing.listen(events.append):
        update(tmp_path / "map", imgs, client=client, progress=lambda m: None)
    assert {"event": "progress", "stage": "inference", "done": 1, "total": 1} in events
    assert any(e["event"] == "count" and e.get("keyframes_sampled") == 1 for e in events)

    def run(env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run([str(REPO / "segment.sh"), "-m", str(tmp_path / "map")],
                              capture_output=True, timeout=120, env={**os.environ, **env})

    def human(err: bytes) -> list[str]:  # the timing figures change from run to run
        return [ln for ln in err.decode().splitlines()
                if "timings:" not in ln and "done in" not in ln]

    plain = run({})
    sink = tmp_path / "progress.jsonl"
    live = run({timing.ENV_PROGRESS: str(sink)})
    assert plain.returncode == live.returncode == 0, live.stderr.decode()
    assert live.stdout == plain.stdout and human(live.stderr) == human(plain.stderr)
    lines = [json.loads(ln) for ln in sink.read_text().splitlines()]
    assert lines[0] == {"event": "begin"} and lines[-1]["event"] == "finish"
    assert [e["stage"] for e in lines if e["event"] == "stage_start"] == ["export", "write"]
