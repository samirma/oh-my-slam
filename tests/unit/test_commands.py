"""The commands' single source of truth (``commands/spec.py``, spec §2.6 "Single source of truth"):
every argparse parser is built from the registry and its export covers every option; API
parameters parse through the same parser; the validation rules raise the commands' own errors,
and a dry run reports them all per parameter; stage names are registered and the stages the
commands record are the declared ones; the exit-code → HTTP rule."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oh_my_slam.cli import mapper as cli_mapper
from oh_my_slam.cli import reconstruct as cli_reconstruct
from oh_my_slam.cli import segment as cli_segment
from oh_my_slam.cli import view as cli_view
from oh_my_slam.commands import spec
from oh_my_slam.commands.parser import ParameterError
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import ATTRIBUTES, CloudAttrs
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
from tests.unit.test_cli_single import env  # noqa: F401  (the fake-server fixture)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "oh_my_slam"
CLIS = {"reconstruct.sh": cli_reconstruct, "mapper.sh": cli_mapper, "segment.sh": cli_segment,
        "view.sh": cli_view}
R, M, S, V = spec.RECONSTRUCT, spec.MAPPER, spec.SEGMENT, spec.VIEW


def _leaf_parsers(ap: argparse.ArgumentParser) -> dict[str | None, argparse.ArgumentParser]:
    for a in ap._actions:
        if isinstance(a, argparse._SubParsersAction):
            return dict(a.choices.items())
    return {None: ap}


def _options(ap: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in ap._actions if a.option_strings and a.dest != "help"]


# --- the parsers and the export -------------------------------------------------------------------


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
                        "segment.sh -i", "view.sh -i", "view.sh -m"}
    for program in spec.PROGRAMS:
        leaves = _leaf_parsers(CLIS[program.prog].build_parser())
        for cmd in program.commands:
            flags = {a.option_strings[0]: a for a in _options(leaves[cmd.name])}
            for mode in cmd.modes:
                op = ops[cmd.label(mode)]
                others = {cmd.option(m.selector).flag for m in cmd.modes
                          if m is not mode and m.selector}
                expected = {f for f in flags if f not in others}
                assert {p["flag"] for p in op["parameters"]} == expected, op["id"]
                for p in op["parameters"]:
                    a = flags[p["flag"]]
                    assert (p["name"], p["help"]) == (a.dest, a.help)
                    assert p["required"] == (a.required or p["name"] == mode.selector)
                assert op["stages"] == [str(s) for s in mode.stages]
                codes = {e["code"] for e in op["errors"]}
                assert "usage" in codes
                assert ("server_unavailable" in codes) == (op["inference"] != "never"), op["id"]
    seg_i = {x["name"]: x for x in ops["segment.sh -i"]["parameters"]}
    assert seg_i["min_score"]["default"] == 0.5 and seg_i["min_score"]["finite"] is True
    assert seg_i["min_score"]["applies"] == [] and seg_i["min_score"]["applies_text"] == ""
    assert seg_i["format"]["choices"] == ["json", "png"] and seg_i["format"]["default"] == "json"
    assert seg_i["image"]["accepts"] == sorted([".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff",
                                                ".webp"])
    assert seg_i["image"]["must_exist"] is True and seg_i["image"]["required"] is True
    assert seg_i["artifacts"]["must_exist"] is False
    assert "attrs" not in seg_i  # segment.sh writes no point cloud (spec §2.4)
    rec = {x["name"]: x for x in ops["reconstruct.sh"]["parameters"]}
    assert rec["format"]["choices"] == ["json", "depth", "ply"]
    attrs = rec["attrs"]
    assert attrs["default"].startswith("color=rgb,stride=1")
    assert attrs["applies"] == [{"option": "format", "in": ["ply"]}]
    schema = {x["key"]: x["schema"] for x in attrs["attributes"]}
    assert schema["color"] == {"type": "enum", "choices": ["rgb", "segment", "height", "none"]}
    assert schema["stride"] == {"type": "integer", "minimum": 1}
    by_format = {o["format"]: o for o in ops["reconstruct.sh"]["outputs"]}
    assert set(by_format) == {"json", "png", "ply"}
    depth = by_format["png"]
    assert depth["when"] == [{"option": "format", "in": ["depth"]}] and not depth["object_regions"]
    assert depth["media_type"] == "image/png" and "16-bit" in depth["text"]
    seg_out = [o for o in ops["segment.sh -i"]["outputs"] if o["via"] == "stdout"]
    assert [(o["format"], o["when"], o["object_regions"]) for o in seg_out] == [
        ("json", [{"option": "format", "in": ["json"]}], False),
        ("png", [{"option": "format", "in": ["png"]}], True)]
    update = {x["name"]: x for x in ops["mapper.sh update"]["parameters"]}
    assert update["fps"]["default"] == 2.0 and update["fps"]["exclusive_minimum"] == 0
    assert update["fps"]["omit_if_default"] is True
    (video,) = update["fps"]["applies"]
    assert video["option"] == "inputs" and video["is"] == "video" and ".mp4" in video["suffixes"]
    assert set(video["suffixes"]) <= set(update["inputs"]["accepts"])
    seg_d = {o["name"]: o for o in ops["segment.sh -i"]["outputs"]}
    assert seg_d["segmented.png"]["object_regions"] and not seg_d["catalog.csv"]["object_regions"]
    assert update["inputs"]["ordered"] is True and ".mp4" in update["inputs"]["accepts"]
    assert update["map"]["must_exist"] is False
    assert ops["mapper.sh locate"]["inference"] == "conditional"
    from oh_my_slam.mapping.api import UPDATE_EXHAUSTIVE_MAX

    assert ops["mapper.sh locate"]["inference_condition"] == {
        "map_keyframes_greater_than": UPDATE_EXHAUSTIVE_MAX}
    assert {o["name"] for o in ops["segment.sh -i"]["outputs"]} == {
        "result", "segmentation.json", "segmented.png", "catalog.csv", "catalog.md"}
    assert {"code": "interrupted", "exit_code": 130, "http_status": 499} in d["exit_codes"]


def test_an_option_added_to_the_registry_reaches_parser_and_export(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = R.commands[0]
    extra = spec.Option("--gain", "gain", spec.Kind.NUMBER, "a new option", type=float,
                        default=1.0, minimum=0)
    program = dataclasses.replace(R, commands=(
        dataclasses.replace(cmd, options=(*cmd.options, extra)),))
    monkeypatch.setattr(spec, "PROGRAMS", (program,))
    assert spec.build_parser(program).parse_args(["-i", "x", "--gain", "2"]).gain == 2.0
    new_cmd = program.commands[0]
    assert spec.parse(new_cmd, new_cmd.modes[0], {"image": "x", "gain": 3}).gain == 3.0
    (op,) = spec.describe()["operations"]
    assert {
        "name": "gain", "flag": "--gain", "kind": "number", "help": "a new option",
        "required": False, "default": 1.0, "minimum": 0}.items() <= op["parameters"][-1].items()


def test_applicability_conditions_read_the_parsed_arguments() -> None:
    """``When``: an option given (the -d artefacts), a value among some (-f png: the segmented
    image), or exactly one video (-fps)."""
    given, png = spec.When("artifacts"), spec.When("format", ("png",))
    video = spec.When("inputs", video=True)
    args = argparse.Namespace(artifacts=Path("d"), format="png", inputs=[Path("a.MP4")])
    assert given.holds(args) and png.holds(args) and video.holds(args)
    args = argparse.Namespace(artifacts=None, format="json", inputs=[Path("a.jpg")])
    assert not (given.holds(args) or png.holds(args) or video.holds(args))
    assert not video.holds(argparse.Namespace(inputs=[Path("a.mp4"), Path("b.mp4")]))
    seg = {o["name"]: o for o in spec.describe()["operations"][3]["outputs"]
           if o["via"] == "-d"}
    assert seg["segmented.png"]["when"] == [given.describe()] == [{"option": "artifacts",
                                                                   "is": "given"}]


def test_cloud_attribute_schemas_match_their_parsers() -> None:
    """The typed attribute schema the API publishes accepts and refuses what ``-p`` does."""
    for a in ATTRIBUTES:
        sc = a.schema
        if sc["type"] == "enum":
            for choice in sc["choices"]:
                a.parse(choice)
            with pytest.raises(ValueError):
                a.parse("bogus")
            continue
        if "minimum" in sc:
            a.parse(str(sc["minimum"]))
            with pytest.raises(ValueError):
                a.parse(str(sc["minimum"] - 0.5))
        if "exclusive_minimum" in sc:
            with pytest.raises(ValueError):
                a.parse(str(sc["exclusive_minimum"]))
        if sc["type"] == "integer":
            with pytest.raises(ValueError):
                a.parse("2.5")
        else:
            finite = sc.get("finite", True)
            if finite:
                with pytest.raises(ValueError):
                    a.parse("inf")
            else:
                assert math.isinf(a.parse("inf"))


# --- parameters → arguments, validation, dry run --------------------------------------------------


def _validate(program: spec.Program, argv: list[str], **kw: Any) -> argparse.Namespace:
    args = spec.build_parser(program).parse_args(argv)
    cmd = program.command(args.command if program.subcommands else None)
    return spec.validate(cmd, args, **kw)


def test_parameters_parse_through_the_commands_parser(tmp_path: Path) -> None:
    seg = S.command()
    params = {"image": tmp_path / "a.jpg", "format": "png", "min_score": 0.3,
              "output": "-odd.png"}
    args = spec.parse(seg, spec.SEGMENT_IMAGE, params)
    cli = cli_segment.build_parser().parse_args(
        ["-i", str(tmp_path / "a.jpg"), "-f", "png", "--min-score", "0.3", "-o=-odd.png"])
    assert args == cli and args.output == Path("-odd.png")
    rec = R.command()
    args = spec.parse(rec, rec.modes[0], {"image": "a.jpg", "format": "ply",
                                          "attrs": ["voxel=0.1", "normals=on"]})
    assert args.attrs == ["voxel=0.1", "normals=on"]
    with pytest.raises(UsageError, match=re.escape(
            "argument -f: invalid choice: 'ply' (choose from json, png)")):
        spec.parse(seg, spec.SEGMENT_IMAGE, {"image": "a.jpg", "format": "ply"})
    with pytest.raises(UsageError, match="the following arguments are required: -m"):
        spec.parse(M.command("update"), M.command("update").modes[0], {"inputs": ["a.jpg"]})
    with pytest.raises(UsageError, match="the following arguments are required: -i"):
        spec.parse(seg, spec.SEGMENT_IMAGE, {"format": "png"})
    with pytest.raises(UsageError, match=re.escape(
            "unrecognized parameters for segment.sh -i: attrs, map")):
        spec.parse(seg, spec.SEGMENT_IMAGE, {"image": "a.jpg", "map": "m", "attrs": "voxel=1"})
    up = M.command("update")
    args = spec.parse(up, up.modes[0], {"inputs": ["a.jpg", "b.jpg"], "map": "m", "fps": 2.0})
    assert args.fps is None and args.inputs == [Path("a.jpg"), Path("b.jpg")]  # default omitted
    with pytest.raises(UsageError, match="invalid float value: 'abc'"):
        spec.parse(up, up.modes[0], {"inputs": ["v.mp4"], "map": "m", "fps": "abc"})
    view = V.command()
    assert spec.parse(view, view.mode("map"), {"map": "m", "no_browser": True}).no_browser
    with pytest.raises(UsageError, match=re.escape("one of the arguments -i -m is required")):
        V.command().mode_of(argparse.Namespace(image=None, map=None))


def test_validation_rules_raise_the_commands_errors(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "f").write_text("x")
    other = str(tmp_path / "other")
    cases = [
        (R, ["-i", str(img), "-p", "voxel=1"], UsageError, "only the PLY output has"),
        (R, ["-i", str(img), "-f", "depth", "-p", "voxel=1"], UsageError,
         "-p sets point-cloud attributes, which only the PLY output has: use -f ply"),
        (R, ["-i", str(tmp_path / "no.jpg")], InputError, "image not found"),
        (R, ["-i", str(img), "-o", str(tmp_path)], UsageError, "is a folder"),
        (R, ["-i", str(img), "-o", str(img / "x.json")], UsageError,
         f"-o {img}: cannot write there (File exists)"),
        (M, ["update", "-i", "v.mp4", "-m", "m", "-fps", "0"], UsageError, "-fps must be"),
        (M, ["update", "-i", "a.jpg", "-m", "m", "-f", "ply", "-p", "stride=2"], UsageError,
         "pixel-level"),
        (M, ["update", "-i", str(tmp_path / "no.jpg"), "-m", "m"], InputError, "input not found"),
        (M, ["update", "-i", str(tmp_path), "-m", "m"], InputError, "is a folder"),
        (M, ["update", "-i", str(img), "-m", other], NotAMapError, "is not empty and not a map"),
        (M, ["locate", "-i", "v.mp4", "-m", "m"], UsageError, "not a video"),
        (M, ["locate", "-i", str(img), "-m", str(tmp_path / "none")], InputError, "no map in"),
        (M, ["locate", "-i", str(img), "-m", other], NotAMapError, "is not a map"),
        (S, ["-i", str(img), "--min-score", "nan"], UsageError, "finite number"),
        (S, ["-i", str(img), "-o", str(tmp_path)], UsageError, "is a folder"),
        (S, ["-i", str(tmp_path / "no.jpg")], InputError, "image not found"),
        (V, ["-i", str(tmp_path / "no.jpg")], InputError, "image not found"),
        (V, ["-m", other], NotAMapError, "not a map folder"),
    ]
    for program, argv, exc, message in cases:
        with pytest.raises(exc, match=re.escape(message)):
            _validate(program, argv)
    assert not (tmp_path / "m").exists()  # the map is never created by validation
    warnings: list[str] = []
    v = _validate(M, ["update", "-i", str(img), "-m", str(tmp_path / "new"), "-fps", "-3"],
                  warn=warnings.append)
    assert v.fps == 2.0 and not v.is_video and v.input_spec.images == [img]
    assert warnings == ["-fps applies to video input only; ignored for images"]
    v = _validate(S, ["-i", str(img), "-d", str(tmp_path / "d"), "-f", "png",
                      "--min-score", "0.3"])
    assert v.min_score == 0.3 and v.format == "png" and not hasattr(v, "attrs")
    assert (tmp_path / "d").is_dir()  # -d is prepared before any work
    assert _validate(R, ["-i", str(img)]).attrs == CloudAttrs()
    assert _validate(R, ["-i", str(img), "-f", "depth"]).attrs == CloudAttrs()
    assert _validate(R, ["-i", str(img), "-f", "ply", "-p", "voxel=0.1"]).attrs == CloudAttrs(
        voxel=0.1)


def test_a_dry_run_reports_every_problem_per_parameter_and_changes_nothing(
        tmp_path: Path) -> None:
    seg = S.command()
    problems = {p.rule: p for p in spec.dry_run(seg, spec.SEGMENT_IMAGE, {
        "image": tmp_path / "none.jpg", "min_score": "abc",
        "artifacts": tmp_path / "new" / "d", "output": tmp_path})}
    assert set(problems) == {"min_score", "output_writable", "image_exists"}
    assert problems["min_score"].parameters[0] == "min_score"
    assert problems["image_exists"].describe() == {
        "rule": "image_exists", "parameters": ["image"],
        "message": f"image not found: {tmp_path / 'none.jpg'}", "code": "usage",
        "exit_code": 2, "http_status": 400}
    assert not (tmp_path / "new").exists()  # the -d folder is only checked
    afile = tmp_path / "afile"
    afile.write_text("x")
    (p,) = spec.dry_run(seg, spec.SEGMENT_IMAGE, {"image": afile, "artifacts": afile})
    assert p.message == f"-d {afile}: cannot write there (File exists)"
    # argparse's problems and the rules' together, per field
    up = M.command("update")
    found = spec.dry_run(up, up.modes[0], {"inputs": [tmp_path / "none.jpg"], "map": afile,
                                           "format": "xml", "fps": "abc", "attrs": ["voxel=1"]})
    assert spec.by_parameter(found) == {
        "format": ["argument -f: invalid choice: 'xml' (choose from json, ply)"],
        "fps": ["argument -fps: invalid float value: 'abc'"],
        "inputs": [f"input not found: {tmp_path / 'none.jpg'}"],
        "map": [f"{afile.resolve()} is not empty and not a map; use a new or empty folder"]}
    # (-p is not checked: it concerns -f, which argparse refused)
    missing = spec.dry_run(up, up.modes[0], {"inputs": ["a.jpg"], "format": "xml"})
    assert [(p.rule, p.parameters) for p in missing] == [("arguments", ("format",)),
                                                         ("arguments", ("map",))]


def test_argument_errors_name_their_parameters() -> None:
    seg, up = S.command(), M.command("update")
    cases = [
        (seg, spec.SEGMENT_IMAGE, {"image": "a.jpg", "format": "xml"}, ("format",)),
        (up, up.modes[0], {"inputs": ["v.mp4"], "map": "m", "fps": "abc"}, ("fps",)),
        (up, up.modes[0], {"inputs": ["a.jpg"]}, ("map",)),
        (up, up.modes[0], {}, ("inputs", "map")),
        (seg, spec.SEGMENT_IMAGE, {"image": "a.jpg", "attrs": "voxel=1"}, ("attrs",)),
    ]
    for cmd, mode, params, names in cases:
        with pytest.raises(ParameterError) as e:
            spec.parse(cmd, mode, params)
        assert e.value.parameters == names, (params, e.value)
        assert e.value.exit_code is ExitCode.USAGE
    # a relative path starting with "-" among several values stays a path
    args = spec.parse(up, up.modes[0], {"inputs": ["-a.jpg", "b.jpg"], "map": "m"})
    assert args.inputs == [Path("./-a.jpg"), Path("b.jpg")]


def test_describe_is_cheap() -> None:
    """Reading the definitions loads none of the pipeline (the web service's start-up)."""
    code = ("import sys, time; t = time.perf_counter(); from oh_my_slam.commands import spec; "
            "spec.describe(); print(time.perf_counter() - t); "
            "print(sorted(m for m in sys.modules if m.startswith('oh_my_slam.') and "
            "m.split('.')[1] not in ('commands', 'core')))")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60, cwd=REPO)
    seconds, loaded = res.stdout.splitlines()
    assert loaded == "[]" and float(seconds) < 0.5, res.stdout


def test_exit_codes_map_to_http_statuses() -> None:
    assert set(HTTP_STATUS) == set(ExitCode)
    assert http_status(ExitCode.USAGE) == 400
    assert http_status(ExitCode.SERVER_UNAVAILABLE) == 503
    assert http_status(ExitCode.INTERNAL) == 500 and http_status(77) == 500
    assert http_status(-9) == 500  # a command stopped by a signal
    for code in (ExitCode.NOT_A_MAP, ExitCode.NOT_REGISTERED, ExitCode.MAP_LOCKED):
        assert 400 <= http_status(code) < 500
    assert http_status(130) == 499 and error_code(130) == "interrupted"
    assert error_code(4) == "not_a_map" and error_code(77) == "internal"
    exported = {e["code"]: e["http_status"] for e in spec.describe()["exit_codes"]}
    assert exported["map_locked"] == 409 and exported["usage"] == 400


# --- stages -------------------------------------------------------------------------------------


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
    """No stage is named by a string literal (``Stage`` is typed, so mypy checks the rest), and
    every ``Stage`` belongs to some operation."""
    assert _stage_literals() == {}
    listed = {s for _p, _c, m in spec.operations() for s in m.stages}
    assert listed == set(Stage)


def _declared(cmd: spec.Command, mode: spec.Mode) -> set[str]:
    return {str(s) for s in mode.stages}


def test_recorded_stages_are_the_declared_ones(env, tmp_path: Path,  # type: ignore[no-untyped-def]  # noqa: F811
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """reconstruct.sh (each format), segment.sh and mapper.sh update / locate (fake server) record
    only stages their operation declares."""
    import shutil

    from oh_my_slam.mapping.api import update
    from oh_my_slam.mapping.locate import locate, open_map, resolve_images
    from tests.fakes.client import FakeClient
    from tests.synth.mapping import add_frames, mapping_room, ring

    img, cap, _client = env
    runs = [
        (R.command(), R.command().modes[0], lambda: cli_reconstruct.main(["-i", str(img)])),
        (R.command(), R.command().modes[0], lambda: cli_reconstruct.main(
            ["-i", str(img), "-f", "ply", "-p", "color=segment"])),
        (R.command(), R.command().modes[0], lambda: cli_reconstruct.main(
            ["-i", str(img), "-f", "depth"])),
        (S.command(), spec.SEGMENT_IMAGE, lambda: cli_segment.main(
            ["-i", str(img), "-d", str(tmp_path / "art")])),
        (S.command(), spec.SEGMENT_IMAGE, lambda: cli_segment.main(["-i", str(img), "-f", "png"])),
    ]
    client = FakeClient()
    room = mapping_room()
    keys = add_frames(client, room, ring(1), tmp_path / "k", "k")
    mdir = tmp_path / "map"
    queries = add_frames(client, room, ring(2, start=0.08, span=0.16), tmp_path / "q", "q")
    up = M.command("update")
    runs += [
        (up, up.modes[0], lambda: update(mdir, keys, client=client, progress=lambda m: None)),
    ]
    if shutil.which("colmap"):
        lo = M.command("locate")
        runs.append((lo, lo.modes[0], lambda: locate(open_map(mdir), resolve_images(queries),
                                                     mode="full", progress=lambda m: None)))
    seen: list[str] = []
    stage = timing.Timings.stage

    def spy(self: timing.Timings, name: Stage) -> Any:
        seen.append(str(name))
        return stage(self, name)

    monkeypatch.setattr(timing.Timings, "stage", spy)
    for cmd, mode, run in runs:
        seen.clear()
        run()
        cap.take()  # the fake stdout takes one payload per run
        assert seen and set(seen) <= _declared(cmd, mode), (cmd.label(mode), seen)


def test_dry_run_reports_a_bad_value_after_an_unknown_parameter(tmp_path: Path) -> None:
    """An unknown parameter is dropped, then the next parse still reports the others; argv_of is
    the command line parse parses (the web service runs the command with it)."""
    seg = spec.SEGMENT.command()
    problems = spec.dry_run(seg, spec.SEGMENT_IMAGE, {"image": str(tmp_path / "a.jpg"),
                                                      "map": "m", "format": "xml"})
    assert [p.parameters for p in problems][:2] == [("map",), ("format",)]
    assert spec.argv_of(seg, spec.SEGMENT_IMAGE, {"image": "a.jpg", "format": "png",
                                                  "min_score": None}) == ["-i=a.jpg", "-f=png"]
