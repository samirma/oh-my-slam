"""The commands' single source of truth: every mode, option, default, validation rule, output,
error and timing stage of ``reconstruct.sh``, ``mapper.sh update`` / ``locate``, ``segment.sh -i`` /
``-m`` and ``view.sh -i`` / ``-m``, declared once as data.

Each command builds its argparse parser (:func:`build_parser`) and runs its validation
(:func:`validate`) from these definitions, and :func:`describe` exports them as JSON-serialisable
data, so that the web service of spec §2.6 ("Single source of truth") derives its operations,
parameters, defaults, validation, error codes and stages from the same definitions: a new or
changed option here reaches the commands and the API alike.

* An :class:`Option` records its flag, name, :class:`Kind`, choices, default, required-ness, help
  text and where it applies (:class:`When`).
* A :class:`Rule` is one validation step, run in order before any work starts; it raises the
  commands' own errors (``core/errors.py``) with their own messages.
* A :class:`Mode` (``segment.sh -i`` / ``-m``, …) is one API operation: its rules, inference need,
  outputs, run-time errors and timing stages (``core.timing.Stage``).

The rules live here, in the top layer, because some of them use the mapping package (``mapper.sh
locate`` opens the map); the web service sits beside the commands and imports this module.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from oh_my_slam.cli.common import ArgumentParser
from oh_my_slam.core.atomic import preflight_dir, preflight_file
from oh_my_slam.core.cloud_attrs import (
    CloudAttrs,
    CloudScope,
    applicable,
    help_text,
    parse_cloud_attrs,
)
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    ExitCode,
    InputError,
    MapLockedError,
    NotAMapError,
    OhMySlamError,
    RegistrationError,
    ServerUnavailableError,
    UsageError,
    error_code,
)
from oh_my_slam.core.images import VIDEO_SUFFIXES
from oh_my_slam.core.timing import Stage
from oh_my_slam.mapping.ingest import DEFAULT_FPS
from oh_my_slam.segmentation.detect import DEFAULT_MIN_SCORE


class Kind(StrEnum):
    """What an option's value is (the web service picks the form field and the upload from it)."""

    IMAGE = "image"  # path in: one RGB image
    IMAGES = "images"  # path in: one or more images
    IMAGES_OR_VIDEO = "images_or_video"  # path in: images, or exactly one video
    MAP = "map"  # path in: a map folder
    FILE_OUT = "file_out"  # path out: a file
    FOLDER_OUT = "folder_out"  # path out: a folder
    ENUM = "enum"
    NUMBER = "number"
    FLAG = "flag"
    ATTRS = "attrs"  # the -p point-cloud attributes (core/cloud_attrs.py)


@dataclass(frozen=True)
class When:
    """An applicability condition read from another option: it is given (``values`` empty), its
    value is one of ``values``, or (``video``) it names exactly one video file."""

    option: str
    values: tuple[str, ...] = ()
    video: bool = False

    def holds(self, args: argparse.Namespace) -> bool:
        v = getattr(args, self.option, None)
        if self.video:
            return isinstance(v, list) and len(v) == 1 and Path(v[0]).suffix.lower() \
                in VIDEO_SUFFIXES
        if self.values:
            return v in self.values
        return v is not None

    def describe(self) -> dict[str, Any]:
        if self.video:
            return {"option": self.option, "is": "video"}
        if self.values:
            return {"option": self.option, "in": list(self.values)}
        return {"option": self.option, "is": "given"}


@dataclass(frozen=True)
class Option:
    flag: str
    name: str  # argparse dest and API parameter name
    kind: Kind
    help: str
    required: bool = False
    default: Any = None  # the effective default (argparse's only for an ENUM: the others stay
    #                      None when not given, which the rules and the commands tell apart)
    choices: tuple[str, ...] | None = None
    multiple: bool = False  # one or more values (-i of mapper.sh)
    repeatable: bool = False  # may be given several times (-p)
    metavar: str | None = None
    type: Callable[[str], Any] | None = None  # argparse conversion (None: the text)
    modes: tuple[str, ...] | None = None  # the modes it belongs to (None: all)
    applies: tuple[When, ...] = ()  # applies when any of these holds (empty: always)
    applies_text: str = ""  # the same, in the command's words
    group: str | None = None  # mutually exclusive group (the modes' selectors)

    def add_to(self, ap: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup) -> None:
        kw: dict[str, Any] = {"dest": self.name, "help": self.help}
        if self.kind is Kind.FLAG:
            ap.add_argument(self.flag, action="store_true", **kw)
            return
        if self.kind is Kind.ENUM:
            kw.update(choices=self.choices, default=self.default)
        if self.multiple:
            kw["nargs"] = "+"
        if self.repeatable:
            kw["action"] = "append"
        if self.required:
            kw["required"] = True
        if self.type is not None:
            kw["type"] = self.type
        if self.metavar is not None:
            kw["metavar"] = self.metavar
        ap.add_argument(self.flag, **kw)


@dataclass
class Context:
    """What a rule reads (``args``: as parsed) and fills (``values``: as the command uses them)."""

    args: argparse.Namespace
    values: argparse.Namespace
    mode: Mode
    warn: Callable[[str], None]


@dataclass(frozen=True)
class Rule:
    name: str
    options: tuple[str, ...]  # the parameters it concerns (where a form flags the error)
    text: str  # what it enforces
    check: Callable[[Context], None]
    errors: tuple[type[OhMySlamError], ...] = (UsageError,)


_MEDIA = {"json": "application/json", "ply": "application/octet-stream", "png": "image/png",
          "csv": "text/csv", "markdown": "text/markdown", "html": "text/html",
          "map": "inode/directory"}


@dataclass(frozen=True)
class Output:
    name: str  # "result" (stdout or -o), a file name, or the folder written
    via: str  # "stdout" (or the -o file) | "-d" | "-m" | "browser"
    format: str  # json | ply | png | csv | markdown | map | html
    text: str
    when: tuple[When, ...] = ()  # produced when any holds (empty: always)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "via": self.via, "format": self.format,
                "media_type": _MEDIA[self.format], "text": self.text,
                "when": [w.describe() for w in self.when]}


@dataclass(frozen=True)
class Mode:
    name: str | None  # None: the command has a single mode
    selector: str | None  # the option whose presence selects it (-i / -m)
    rules: tuple[Rule, ...]
    inference: str  # "required" | "never" | "conditional"
    inference_text: str
    stages: tuple[Stage, ...]
    outputs: tuple[Output, ...]
    errors: tuple[type[OhMySlamError], ...]  # raised while it runs (after validation)
    attrs_scope: CloudScope | None = None  # the -p scope


@dataclass(frozen=True)
class Command:
    prog: str  # the shell entry point
    name: str | None  # subcommand (mapper.sh update / locate)
    help: str  # subcommand help, or the parser's description
    options: tuple[Option, ...]
    modes: tuple[Mode, ...]
    exclusive_required: tuple[str, ...] = ()  # one of these is required (the mode selectors)

    def mode_of(self, args: argparse.Namespace) -> Mode:
        for m in self.modes:
            if m.selector is None or getattr(args, m.selector, None) is not None:
                return m
        raise UsageError(f"one of {', '.join(self.flags(self.exclusive_required))} is required")

    def flags(self, names: tuple[str, ...]) -> list[str]:
        by = {o.name: o.flag for o in self.options}
        return [by[n] for n in names]

    def option(self, name: str) -> Option:
        return next(o for o in self.options if o.name == name)

    def label(self, mode: Mode) -> str:
        """The command as typed (``segment.sh -i``, ``mapper.sh update``): the operation id."""
        parts = [self.prog] + ([self.name] if self.name else [])
        if mode.selector is not None:
            parts.append(self.option(mode.selector).flag)
        return " ".join(parts)

    def mode_options(self, mode: Mode) -> list[Option]:
        others = {m.selector for m in self.modes if m is not mode and m.selector}
        return [o for o in self.options if o.name not in others
                and (o.modes is None or mode.name in o.modes)]


@dataclass(frozen=True)
class Program:
    prog: str
    description: str
    commands: tuple[Command, ...]  # one, or the subcommands

    @property
    def subcommands(self) -> bool:
        return self.commands[0].name is not None

    def command(self, name: str | None = None) -> Command:
        return next(c for c in self.commands if c.name == name)


# --- rules ----------------------------------------------------------------------------------------


def _attrs_rule(writes_ply: tuple[When, ...], requires: str) -> Rule:
    """``-p``: refused without a PLY output, then parsed for the mode's scope (spec §2.2)."""

    def check(ctx: Context) -> None:
        values = ctx.args.attrs
        scope = ctx.mode.attrs_scope
        assert scope is not None
        if values and not any(w.holds(ctx.args) for w in writes_ply):
            raise UsageError(f"-p sets point-cloud attributes, which {requires}")
        ctx.values.attrs = parse_cloud_attrs(values, scope)

    return Rule("attrs", ("attrs", *(w.option for w in writes_ply)),
                f"-p sets point-cloud attributes, which {requires}; every key and value is "
                "valid for the command (spec §2.2)", check)


def _output_check(ctx: Context) -> None:
    if ctx.args.output is not None:
        preflight_file(ctx.args.output, "-o")


OUTPUT_RULE = Rule("output_writable", ("output",),
                   "the -o file can be written (it is not a folder), checked before any work",
                   _output_check)


def _artifacts_check(ctx: Context) -> None:
    if ctx.args.artifacts is not None:
        preflight_dir(ctx.args.artifacts, "-d")


ARTIFACTS_RULE = Rule("artifacts_writable", ("artifacts",),
                      "the -d folder is created and can be written, checked before any work",
                      _artifacts_check)


def _image_check(ctx: Context) -> None:
    image: Path = ctx.args.image
    if not image.is_file():
        raise InputError(f"image not found: {image}")


IMAGE_RULE = Rule("image_exists", ("image",), "the -i image exists", _image_check, (InputError,))


def _fps_check(ctx: Context) -> None:
    a = ctx.args
    is_video = When("inputs", video=True).holds(a)
    fps = DEFAULT_FPS if a.fps is None else a.fps
    if a.fps is not None and not is_video:
        ctx.warn("-fps applies to video input only; ignored for images")
        fps = DEFAULT_FPS  # ignored (spec §2.3), whatever its value
    if fps <= 0:
        raise UsageError("-fps must be positive")
    ctx.values.fps = fps
    ctx.values.is_video = is_video


FPS_RULE = Rule("fps", ("fps", "inputs"),
                "-fps must be positive for a video; for images it is ignored with a warning",
                _fps_check)


def _min_score_check(ctx: Context) -> None:
    value = ctx.args.min_score
    if value is None:
        ctx.values.min_score = DEFAULT_MIN_SCORE
        return
    try:
        s = float(value)
    except ValueError as exc:
        raise UsageError(f"--min-score must be a number, got {value!r}") from exc
    if not math.isfinite(s):
        raise UsageError(f"--min-score must be a finite number, got {value!r}")
    ctx.values.min_score = s


MIN_SCORE_RULE = Rule("min_score", ("min_score",), "--min-score is a finite number",
                      _min_score_check)


def _no_min_score_check(ctx: Context) -> None:
    if ctx.args.min_score is not None:
        raise UsageError("-m exports the map's persistent objects; --min-score applies to -i only")


NO_MIN_SCORE_RULE = Rule("min_score_image_only", ("min_score", "map"),
                         "--min-score applies to -i only", _no_min_score_check)


def _locate_images_check(ctx: Context) -> None:
    from oh_my_slam.mapping.locate import resolve_images

    ctx.values.images = resolve_images(ctx.args.inputs)


def _locate_map_check(ctx: Context) -> None:
    from oh_my_slam.mapping.locate import open_map

    ctx.values.reader = open_map(ctx.args.map)


def _locate_output_check(ctx: Context) -> None:
    from oh_my_slam.mapping.locate import check_output

    check_output(ctx.args.map, ctx.args.output)


LOCATE_RULES = (
    Rule("images_only", ("inputs",), "-i names image files; a video is refused",
         _locate_images_check, (UsageError, InputError)),
    Rule("existing_map", ("map",),
         "-m is an existing map; a missing or empty folder is not created, a non-empty folder "
         "that is not a map is refused", _locate_map_check, (InputError, NotAMapError)),
    Rule("output_outside_map", ("output", "map"),
         "-o is not inside the map folder, which locate never writes", _locate_output_check),
)


# --- the commands ---------------------------------------------------------------------------------

_PLY = When("format", ("ply",))
_JSON = When("format", ("json",))
_D = When("artifacts")
_FORMAT_HELP = "output format (default: json)"
_SERVER = (ServerUnavailableError,)


def _format() -> Option:
    return Option("-f", "format", Kind.ENUM, _FORMAT_HELP, default="json", choices=("json", "ply"))


def _output() -> Option:
    return Option("-o", "output", Kind.FILE_OUT,
                  "write the result to FILE instead of stdout (stdout then stays empty)",
                  metavar="FILE", type=Path)


def _attrs(scope: CloudScope, requires: str, applies: tuple[When, ...], applies_text: str
           ) -> Option:
    return Option("-p", "attrs", Kind.ATTRS, f"{help_text(scope)}; {requires}", repeatable=True,
                  metavar="ATTRS", applies=applies, applies_text=applies_text)


def _result(scene: str, cloud: str) -> tuple[Output, Output]:
    return (Output("result", "stdout", "json", scene, (_JSON,)),
            Output("result", "stdout", "ply", cloud, (_PLY,)))


_PLY_ONLY = "only the PLY output has: use -f ply"
_SCENE = "the OpenLABEL 1.0.0 scene description (spec §3)"

RECONSTRUCT = Program("reconstruct.sh", "Single-image reconstruction (stdout or -o file).", (
    Command("reconstruct.sh", None, "Single-image reconstruction (stdout or -o file).", (
        Option("-i", "image", Kind.IMAGE, "input RGB image", required=True, type=Path),
        _format(),
        _output(),
        _attrs(CloudScope.IMAGE, "requires -f ply", (_PLY,), "only with -f ply"),
    ), (
        Mode(None, None, (_attrs_rule((_PLY,), _PLY_ONLY), OUTPUT_RULE, IMAGE_RULE),
             "required", "reconstructs the image with the inference server",
             (Stage.CONNECT, Stage.INFERENCE, Stage.SEGMENT, Stage.EXPORT, Stage.WRITE),
             _result(f"{_SCENE}: objects, labels, scores, colours and OBBs in the camera frame",
                     "the point cloud (camera frame, metres) shaped by -p"),
             _SERVER, CloudScope.IMAGE),
    )),
))

_MAP_RUN_ERRORS = (InputError, NotAMapError, RegistrationError, MapLockedError,
                   ServerUnavailableError)
_g = "{:g}".format

MAPPER = Program("mapper.sh", "Multi-frame mapping (persistent map).", (
    Command("mapper.sh", "update", "create or extend a map", (
        Option("-i", "inputs", Kind.IMAGES_OR_VIDEO, "image files, or exactly one video",
               required=True, multiple=True, type=Path),
        Option("-m", "map", Kind.MAP, "map folder", required=True, type=Path),
        _format(),
        _output(),
        _attrs(CloudScope.MAP, "requires -f ply", (_PLY,), "only with -f ply"),
        Option("-t", "mode", Kind.ENUM,
               "full = whole map with all keyframe poses; single = new input only "
               "(default: full)", default="full", choices=("full", "single")),
        Option("-fps", "fps", Kind.NUMBER,
               f"video frames per second to sample (default: {_g(DEFAULT_FPS)}; ignored for "
               "images)", default=DEFAULT_FPS, type=float, applies=(When("inputs", video=True),),
               applies_text="video input only; ignored for images"),
    ), (
        Mode(None, None, (_attrs_rule((_PLY,), _PLY_ONLY), OUTPUT_RULE, FPS_RULE),
             "required", "infers depth and objects of every new keyframe",
             (Stage.SETUP, Stage.INGEST, Stage.INFERENCE, Stage.SFM, Stage.FEATURES_MATCHING,
              Stage.POSE_REFINEMENT, Stage.FOCAL_RERUN, Stage.MAP_FRAME, Stage.DEPTH_ALIGNMENT,
              Stage.PERSIST_FRAMES, Stage.VALIDITY, Stage.OBJECTS, Stage.CLOUD, Stage.EXPORT,
              Stage.COMMIT),
             (*_result(f"{_SCENE} of the whole map (-t full) or of the new input (-t single), "
                       "map coordinates",
                       "the map cloud (-t full) or the new frames' points (-t single)"),
              Output("map", "-m", "map", "the map folder, created or extended")),
             _MAP_RUN_ERRORS, CloudScope.MAP),
    )),
    Command("mapper.sh", "locate", "camera pose of images in an existing map (read-only)", (
        Option("-i", "inputs", Kind.IMAGES, "one or more image files (a video is refused)",
               required=True, multiple=True, type=Path),
        Option("-m", "map", Kind.MAP, "existing map folder", required=True, type=Path),
        _format(),
        _output(),
        _attrs(CloudScope.MAP, "requires -f ply", (_PLY,), "only with -f ply"),
        Option("-t", "mode", Kind.ENUM,
               "single = the located camera poses only; full = the whole map plus the "
               "located poses (default: single)", default="single", choices=("full", "single")),
    ), (
        Mode(None, None, (_attrs_rule((_PLY,), _PLY_ONLY), *LOCATE_RULES, OUTPUT_RULE),
             "conditional", "only for retrieval in maps of more keyframes than are matched "
             "exhaustively",
             (Stage.SETUP, Stage.FEATURES_MATCHING, Stage.POSE, Stage.EXPORT),
             _result(f"{_SCENE}: the located camera poses (-t single), or the whole map plus "
                     "them (-t full)",
                     "the map points visible from the located cameras (-t single) or the whole "
                     "map cloud (-t full); the located poses in the header"),
             (UsageError, ServerUnavailableError), CloudScope.MAP),
    )),
))

_SEGMENT_ATTRS_HELP = ("shapes the -f ply output and segments.ply, so it needs -f ply or -d; with "
                       "-m the pixel-level keys (stride, min-depth, max-depth, edge) are refused")
_SEGMENT_PLY = "shape the PLY output: use -f ply or -d <folder>"
_ARTEFACTS = (
    Output("segmentation.json", "-d", "json", f"{_SCENE}, identical to -f json", (_D,)),
    Output("segmented.png", "-d", "png", "the image (for a map, keyframes) with each instance "
           "mask painted in its object's colour", (_D,)),
    Output("catalog.csv", "-d", "csv", "one row per object", (_D,)),
    Output("catalog.md", "-d", "markdown", "the catalogue as a table by descending volume",
           (_D,)),
    Output("segments.ply", "-d", "ply", "the object-coloured cloud, identical to -f ply", (_D,)),
)
_SEGMENT_RULES = (_attrs_rule((_PLY, _D), _SEGMENT_PLY), OUTPUT_RULE, ARTIFACTS_RULE)

SEGMENT = Program("segment.sh", "Instance segmentation → JSON + OBBs, artefacts.", (
    Command("segment.sh", None, "Instance segmentation → JSON + OBBs, artefacts.", (
        Option("-i", "image", Kind.IMAGE, "input RGB image", type=Path, group="source"),
        Option("-m", "map", Kind.MAP, "existing map folder (read-only)", type=Path,
               group="source"),
        _format(),
        _output(),
        _attrs(CloudScope.IMAGE | CloudScope.SEGMENT, _SEGMENT_ATTRS_HELP, (_PLY, _D),
               "only with -f ply or -d"),
        Option("-d", "artifacts", Kind.FOLDER_OUT,
               "also write segmentation.json, segmented.png, catalog.csv, catalog.md and "
               "segments.ply into FOLDER", metavar="FOLDER", type=Path),
        Option("--min-score", "min_score", Kind.NUMBER,
               f"drop detections below this score (default {_g(DEFAULT_MIN_SCORE)}; -i only)",
               default=DEFAULT_MIN_SCORE, modes=("image",), applies_text="-i only"),
    ), (
        Mode("image", "image", (MIN_SCORE_RULE, *_SEGMENT_RULES, IMAGE_RULE),
             "required", "segments the image with the inference server",
             (Stage.CONNECT, Stage.INFERENCE, Stage.SEGMENT, Stage.EXPORT, Stage.ARTIFACTS,
              Stage.WRITE),
             (*_result(f"{_SCENE} (camera frame)", "the object-coloured point cloud"),
              *_ARTEFACTS),
             _SERVER, CloudScope.IMAGE | CloudScope.SEGMENT),
        Mode("map", "map", (NO_MIN_SCORE_RULE, *_SEGMENT_RULES),
             "never", "exports the map's persistent objects without inference",
             (Stage.EXPORT, Stage.ARTIFACTS, Stage.WRITE),
             (*_result(f"{_SCENE} of the map's objects (map coordinates)",
                       "the object-coloured map cloud"), *_ARTEFACTS),
             (NotAMapError,), CloudScope.MAP | CloudScope.SEGMENT),
    ), exclusive_required=("image", "map")),
))

VIEW = Program("view.sh", "Browser visualisation of an image or a map.", (
    Command("view.sh", None, "Browser visualisation of an image or a map.", (
        Option("-i", "image", Kind.IMAGE, "RGB image to reconstruct and segment", type=Path,
               group="source"),
        Option("-m", "map", Kind.MAP, "map folder (opened read-only)", type=Path,
               group="source"),
        Option("--no-browser", "no_browser", Kind.FLAG, "do not open a browser", default=False),
    ), (
        Mode("image", "image", (IMAGE_RULE,), "required",
             "reconstructs and segments the image with the inference server", (),
             (Output("viewer", "browser", "html", "the viewer page (URL on stderr)"),),
             _SERVER),
        Mode("map", "map", (), "never", "opens the persisted map read-only", (),
             (Output("viewer", "browser", "html", "the viewer page (URL on stderr)"),),
             (NotAMapError,)),
    ), exclusive_required=("image", "map")),
))

PROGRAMS: tuple[Program, ...] = (RECONSTRUCT, MAPPER, SEGMENT, VIEW)


# --- argparse and validation ----------------------------------------------------------------------


def _add_options(ap: argparse.ArgumentParser, cmd: Command) -> None:
    groups: dict[str, argparse._MutuallyExclusiveGroup] = {}
    for o in cmd.options:
        if o.group is None:
            o.add_to(ap)
            continue
        if o.group not in groups:
            groups[o.group] = ap.add_mutually_exclusive_group(required=bool(cmd.exclusive_required))
        o.add_to(groups[o.group])


def build_parser(program: Program) -> ArgumentParser:
    """The command's argparse parser, built from its definitions."""
    ap = ArgumentParser(prog=program.prog, description=program.description)
    if not program.subcommands:
        _add_options(ap, program.commands[0])
        return ap
    sub = ap.add_subparsers(dest="command", required=True, parser_class=ArgumentParser)
    for cmd in program.commands:
        _add_options(sub.add_parser(cmd.name, help=cmd.help), cmd)  # type: ignore[arg-type]
    return ap


def validate(cmd: Command, args: argparse.Namespace,
             warn: Callable[[str], None] | None = None) -> argparse.Namespace:
    """Run the mode's rules in order on parsed ``args`` (raising the command's own error on the
    first that fails) and return the values the command uses: ``args`` with ``attrs`` parsed to
    :class:`CloudAttrs`, ``fps`` / ``min_score`` resolved, and whatever a rule prepared (the
    images and map reader of ``locate``). ``warn`` receives the warnings (ignored options)."""
    mode = cmd.mode_of(args)
    ctx = Context(args, argparse.Namespace(**vars(args)), mode, warn or (lambda _msg: None))
    for rule in mode.rules:
        rule.check(ctx)
    return ctx.values


# --- export as data -------------------------------------------------------------------------------


def _errors(classes: tuple[type[OhMySlamError], ...]) -> list[dict[str, Any]]:
    by_code: dict[ExitCode, list[str]] = {}
    for c in classes:
        names = by_code.setdefault(c.exit_code, [])
        if c.__name__ not in names:
            names.append(c.__name__)
    return [{"code": error_code(code), "exit_code": int(code), "http_status": HTTP_STATUS[code],
             "errors": names} for code, names in sorted(by_code.items())]


def _attributes(scope: CloudScope) -> list[dict[str, Any]]:
    d = CloudAttrs.defaults(scope)
    fixed = CloudScope.SEGMENT in scope
    return [{"key": a.key, "values": "segment" if fixed and a.key == "color" else a.metavar,
             "default": a.format(getattr(d, a.field)), "effect": a.effect}
            for a in applicable(scope)]


def _option(cmd: Command, mode: Mode, o: Option) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": o.name, "flag": o.flag, "kind": str(o.kind), "help": o.help,
        "required": o.required or o.name == mode.selector,
        "default": None if o.name == mode.selector else o.default,
        "choices": list(o.choices) if o.choices else None,
        "multiple": o.multiple, "repeatable": o.repeatable,
        "applies": [w.describe() for w in o.applies], "applies_text": o.applies_text,
    }
    if o.kind is Kind.ATTRS:
        assert mode.attrs_scope is not None
        out["attributes"] = _attributes(mode.attrs_scope)
        out["default"] = CloudAttrs.defaults(mode.attrs_scope).describe(mode.attrs_scope)
    return out


def operations() -> list[tuple[Program, Command, Mode]]:
    return [(p, c, m) for p in PROGRAMS for c in p.commands for m in c.modes]


def describe() -> dict[str, Any]:
    """Every operation (command mode) as JSON-serialisable data: its parameters (names, kinds,
    defaults, help, choices, applicability), validation rules, inference need, outputs, errors
    (with exit code, machine-readable code and HTTP status) and timing stages; plus the exit-code
    table. The web service generates its OpenAPI document and forms from it."""
    ops = []
    for prog, cmd, mode in operations():
        rule_errors = tuple(e for r in mode.rules for e in r.errors)
        ops.append({
            "id": cmd.label(mode),
            "prog": prog.prog,
            "command": cmd.name,
            "mode": mode.name,
            "description": cmd.help,
            "inference": mode.inference,
            "inference_text": mode.inference_text,
            "parameters": [_option(cmd, mode, o) for o in cmd.mode_options(mode)],
            "rules": [{"name": r.name, "parameters": list(r.options), "text": r.text,
                       "codes": sorted({error_code(e.exit_code) for e in r.errors})}
                      for r in mode.rules],
            "outputs": [out.describe() for out in mode.outputs],
            "errors": _errors((UsageError, *rule_errors, *mode.errors)),
            "stages": [str(s) for s in mode.stages],
        })
    return {
        "operations": ops,
        "exit_codes": [{"code": error_code(c), "exit_code": int(c), "http_status": s}
                       for c, s in HTTP_STATUS.items()],
        "stages": [str(s) for s in Stage],
    }
