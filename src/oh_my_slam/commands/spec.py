"""The commands' single source of truth: every mode, option, default, validation rule, output and
timing stage of ``reconstruct.sh``, ``mapper.sh update`` / ``locate``, ``segment.sh -i`` and
``view.sh -i`` / ``-m``, declared once as data.

Each command builds its argparse parser (:func:`build_parser`) and runs its validation
(:func:`validate`) from these definitions. The web service of spec §2.6 ("Single source of
truth") uses the same ones for the programs it offers (``Program.service``: every mode of
``reconstruct.sh``, ``mapper.sh`` and ``segment.sh``): :func:`parse` turns API parameters into
the command's arguments through the same parser (so its messages are argparse's), :func:`dry_run`
reports every problem of a request per parameter without touching the filesystem before the
request runs, and :func:`describe` exports everything as JSON-serialisable data for the OpenAPI
document and the forms. A new or changed option here reaches the commands and the API alike.

* An :class:`Option` records its flag, name, :class:`Kind`, choices, default, bounds, required-ness,
  help text, where it applies (:class:`When`) and the modes it belongs to (all by default; an
  option given to another mode is refused by :func:`validate`).
* A :class:`Rule` is one validation step, run in order before any work starts: a pure ``check``
  (reads only) that raises the commands' own errors (``core/errors.py``) with their own messages,
  and an optional ``prepare`` with the side effect the command needs before it starts (``-d`` is
  created).
* A :class:`Mode` (``view.sh -i`` / ``-m``, …) is one API operation: its rules, inference need,
  outputs and timing stages (``core.timing.Stage``). Errors at run time are those of the exit-code
  table (``core.errors.HTTP_STATUS``).

The package sits directly under the command line and the web service in the import layers and
imports the pipeline packages only inside the rules that need them, so reading the definitions is
cheap.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from oh_my_slam.commands.parser import ArgumentParser, ParameterError, RaisingParser
from oh_my_slam.core.cloud_attrs import (
    CloudAttrs,
    CloudScope,
    applicable,
    help_text,
    parse_cloud_attrs,
)
from oh_my_slam.core.constants import (
    DEFAULT_FPS,
    DEFAULT_MIN_SCORE,
    DEPTH_UNITS_PER_METRE,
    IMAGE_SUFFIXES,
    NO_DEPTH,
    UPDATE_EXHAUSTIVE_MAX,
    VIDEO_SUFFIXES,
)
from oh_my_slam.core.errors import (
    HTTP_STATUS,
    ExitCode,
    InputError,
    NotAMapError,
    OhMySlamError,
    ServerUnavailableError,
    UsageError,
    error_code,
    internal_message,
)
from oh_my_slam.core.timing import Stage


class Kind(StrEnum):
    """What an option's value is (the web service picks the form field and the upload from it; the
    path-out kinds only choose where the command writes, so they are no API parameters)."""

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


ACCEPTS: dict[Kind, frozenset[str]] = {  # file suffixes a path-in kind takes
    Kind.IMAGE: IMAGE_SUFFIXES,
    Kind.IMAGES: IMAGE_SUFFIXES,
    Kind.IMAGES_OR_VIDEO: IMAGE_SUFFIXES | VIDEO_SUFFIXES,
}


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
        if self.video:  # with the suffixes that make a file a video
            return {"option": self.option, "is": "video", "suffixes": sorted(VIDEO_SUFFIXES)}
        if self.values:
            return {"option": self.option, "in": list(self.values)}
        return {"option": self.option, "is": "given"}


class _OneAttrs(argparse.Action):
    """``-p``: spec §2.2 defines one ``-p key=value[,key=value…]``, so a second one is an argument
    error (exit 2), never merged with the first nor silently replacing it."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        if getattr(namespace, self.dest) is not None:
            raise argparse.ArgumentError(self, "given more than once: give every point-cloud "
                                         "attribute in one -p key=value[,key=value...] "
                                         "(e.g. -p color=rgb,voxel=0.01)")
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class Option:
    flag: str
    name: str  # argparse dest and API parameter name
    kind: Kind
    help: str
    required: bool = False
    # The effective default. argparse gets it only for an ENUM; any other option stays None when it
    # is not given, which the rules and the commands tell apart from a given value.
    default: Any = None
    choices: tuple[str, ...] | None = None
    multiple: bool = False  # one or more values (-i of mapper.sh)
    ordered: bool = False  # the order of the values matters (mapper.sh update -i: latest wins)
    must_exist: bool | None = None  # a path in that must exist (None: not a path in)
    minimum: float | None = None
    exclusive_minimum: float | None = None
    finite: bool = False
    omit_if_default: bool = False  # giving the default is not the same as not giving it (-fps)
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
        if self.kind is Kind.ATTRS:
            kw["action"] = _OneAttrs
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
    options: tuple[str, ...]  # the parameters it concerns (the first is where a form flags it)
    text: str  # what it enforces
    check: Callable[[Context], None]  # reads only; raises the command's error
    errors: tuple[type[OhMySlamError], ...] = (UsageError,)
    prepare: Callable[[Context], None] | None = None  # the side effect before the work starts


@dataclass(frozen=True)
class Problem:
    """One failed rule of :func:`dry_run`."""

    rule: str
    parameters: tuple[str, ...]
    message: str
    exit_code: ExitCode

    def describe(self) -> dict[str, Any]:
        return {"rule": self.rule, "parameters": list(self.parameters), "message": self.message,
                "code": error_code(self.exit_code), "exit_code": int(self.exit_code),
                "http_status": HTTP_STATUS[self.exit_code]}


_MEDIA = {"json": "application/json", "ply": "application/octet-stream", "png": "image/png",
          "csv": "text/csv", "markdown": "text/markdown", "html": "text/html",
          "map": "inode/directory"}
_SUFFIX = {"json": ".json", "ply": ".ply", "png": ".png"}  # of each format a result has (stdout)


def media_type(fmt: str) -> str:
    """The media type of an output format (``Output.format``)."""
    return _MEDIA[fmt]


def suffix_of(fmt: str) -> str:
    """The file suffix of a result's format, one a command writes to stdout ('' for another)."""
    return _SUFFIX.get(fmt, "")


@dataclass(frozen=True)
class Output:
    name: str  # "result" (stdout or -o), a file name, or the folder written
    via: str  # "stdout" (or the -o file) | "-d" | "-m" | "browser"
    format: str  # json | ply | png | csv | markdown | map | html
    text: str
    when: tuple[When, ...] = ()  # produced when any holds (empty: always)
    # an image whose pixels are painted in the objects' colours (the §2.4 colour contract): the
    # object under a pixel is the one of that colour
    object_regions: bool = False

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "via": self.via, "format": self.format,
                "media_type": _MEDIA[self.format], "text": self.text,
                "when": [w.describe() for w in self.when], "object_regions": self.object_regions}


@dataclass(frozen=True)
class Mode:
    name: str | None  # None: the command has a single mode
    selector: str | None  # the option whose presence selects it (-i / -m)
    rules: tuple[Rule, ...]
    inference: str  # "required" | "never" | "conditional"
    inference_text: str
    stages: tuple[Stage, ...]
    outputs: tuple[Output, ...]
    attrs_scope: CloudScope | None = None  # the -p scope
    inference_condition: Mapping[str, Any] | None = None  # when "conditional"
    # errors a run can end with besides its rules' and the inference server's (--status of a
    # service that does not run: exit 3)
    errors: tuple[type[OhMySlamError], ...] = ()
    # "starts" or "stops" a long-lived server (commands.entry_points): the user runs it
    lifecycle: str | None = None

    def scope(self) -> CloudScope:
        assert self.attrs_scope is not None, "this mode writes no point cloud"
        return self.attrs_scope


@dataclass(frozen=True)
class Command:
    prog: str  # the shell entry point
    name: str | None  # subcommand (mapper.sh update / locate)
    help: str  # subcommand help, or the parser's description
    options: tuple[Option, ...]
    modes: tuple[Mode, ...]
    exclusive_required: tuple[str, ...] = ()  # one of these is required (the mode selectors)

    def mode_of(self, args: argparse.Namespace) -> Mode:
        """The mode whose selector is given (a flag selector is given when set), else the mode
        without one."""
        given = [m for m in self.modes if m.selector is not None
                 and getattr(args, m.selector, None) not in (None, False)]
        for m in (*given, *(m for m in self.modes if m.selector is None)):
            return m
        flags = " ".join(self.option(n).flag for n in self.exclusive_required)
        raise UsageError(f"one of the arguments {flags} is required")

    def mode(self, name: str | None) -> Mode:
        return next(m for m in self.modes if m.name == name)

    def option(self, name: str) -> Option:
        return next(o for o in self.options if o.name == name)

    def label(self, mode: Mode) -> str:
        """The command as typed (``view.sh -i``, ``mapper.sh update``): the operation id."""
        parts = [self.prog] + ([self.name] if self.name else [])
        if mode.selector is not None:
            parts.append(self.option(mode.selector).flag)
        return " ".join(parts)

    def mode_options(self, mode: Mode) -> list[Option]:
        """The options of ``mode``: all but the other modes' selectors and the options that
        belong to other modes only (``Option.modes``)."""
        others = {m.selector for m in self.modes if m is not mode and m.selector}
        return [o for o in self.options if o.name not in others
                and (o.modes is None or mode.name in o.modes)]

    def foreign(self, mode: Mode, args: argparse.Namespace) -> list[Option]:
        """The options given in ``args`` that are not options of ``mode`` (argparse leaves an
        option that was not given None, a flag False and an enum its default)."""
        own = self.mode_options(mode)
        values = {o.name: getattr(args, o.name, None) for o in self.options}
        return [o for o in self.options if o not in own
                and values[o.name] is not None and values[o.name] is not False  # (0 is given)
                and not (o.kind is Kind.ENUM and values[o.name] == o.default)]


@dataclass(frozen=True)
class Program:
    prog: str
    description: str
    commands: tuple[Command, ...]  # one, or the subcommands
    # True: the web service (spec §2.6) offers every mode as an API operation; False: a command
    # only (view.sh, whose output is the browser)
    service: bool = True

    @property
    def subcommands(self) -> bool:
        return self.commands[0].name is not None

    def command(self, name: str | None = None) -> Command:
        return next(c for c in self.commands if c.name == name)


# --- rules ----------------------------------------------------------------------------------------


_PLY = When("format", ("ply",))
_JSON = When("format", ("json",))
_DEPTH = When("format", ("depth",))
_PNG = When("format", ("png",))
_D = When("artifacts")
_PLY_ONLY = "only the PLY output has: use -f ply"


def _attrs_check(ctx: Context) -> None:
    """``-p``: refused without a PLY output (``-f json``, ``-f depth``), then parsed for the
    mode's scope (spec §2.2); both before any inference."""
    text = ctx.args.attrs
    if text is not None and not _PLY.holds(ctx.args):
        raise UsageError(f"-p sets point-cloud attributes, which {_PLY_ONLY}")
    ctx.values.attrs = parse_cloud_attrs(text, ctx.mode.scope())


ATTRS_RULE = Rule("attrs", ("attrs", "format"),
                  f"-p sets point-cloud attributes, which {_PLY_ONLY}; every key and value is "
                  "valid for the command (spec §2.2)", _attrs_check)


def _output_check(ctx: Context) -> None:
    from oh_my_slam.core.atomic import check_file

    if ctx.args.output is not None:
        check_file(ctx.args.output, "-o")  # the command's -o writer prepares it


OUTPUT_RULE = Rule("output_writable", ("output",),
                   "the -o file can be written (it is not a folder), checked before any work",
                   _output_check)


def _artifacts_check(ctx: Context) -> None:
    from oh_my_slam.core.atomic import check_dir

    if ctx.args.artifacts is not None:
        check_dir(ctx.args.artifacts, "-d")


def _artifacts_prepare(ctx: Context) -> None:
    from oh_my_slam.core.atomic import preflight_dir

    if ctx.args.artifacts is not None:
        preflight_dir(ctx.args.artifacts, "-d")


ARTIFACTS_RULE = Rule("artifacts_writable", ("artifacts",),
                      "the -d folder can be created and written, checked before any work",
                      _artifacts_check, prepare=_artifacts_prepare)


def _image_check(ctx: Context) -> None:
    from oh_my_slam.core.images import open_header

    image: Path = ctx.args.image
    if not image.is_file():
        raise InputError(f"image not found: {image}")
    if image.suffix.lower() not in ACCEPTS[Kind.IMAGE]:
        raise InputError(f"unsupported input (not an image): {image}; -i takes "
                         f"{', '.join(sorted(ACCEPTS[Kind.IMAGE]))}")
    with open_header(image):  # an image that cannot be read: exit 2 before the server
        pass


IMAGE_RULE = Rule("image_exists", ("image",),
                  "the -i image exists, has an accepted suffix and can be read, checked before "
                  "the inference server is contacted", _image_check, (InputError,))


def _map_check(ctx: Context) -> None:
    from oh_my_slam.mapping.store import MapReader

    ctx.values.reader = MapReader(ctx.args.map)  # the command reads the map through it


MAP_RULE = Rule("existing_map", ("map",), "-m is a map (it is opened read-only)", _map_check,
                (NotAMapError,))


def _fps_check(ctx: Context) -> None:
    a = ctx.args
    is_video = When("inputs", video=True).holds(a)
    fps = DEFAULT_FPS if a.fps is None else a.fps
    if a.fps is not None and not is_video:
        ctx.warn("-fps applies to video input only; ignored for images")
        fps = DEFAULT_FPS  # ignored (spec §2.3), whatever its value
    if not math.isfinite(fps) or fps <= 0:
        raise UsageError(f"-fps must be a positive finite number, got {_g(fps)}")
    ctx.values.fps = fps
    ctx.values.is_video = is_video


FPS_RULE = Rule("fps", ("fps", "inputs"),
                "-fps must be a positive finite number for a video; for images it is ignored "
                "with a warning", _fps_check)


def _update_inputs_check(ctx: Context) -> None:
    from oh_my_slam.mapping.ingest import resolve_inputs

    ctx.values.input_spec = resolve_inputs(ctx.args.inputs)


def _update_map_check(ctx: Context) -> None:
    from oh_my_slam.mapping.store import refuse_non_map

    refuse_non_map(ctx.args.map)


UPDATE_RULES = (
    Rule("update_inputs", ("inputs",),
         "-i names existing image files, in order, or exactly one video",
         _update_inputs_check, (UsageError, InputError)),
    Rule("map_is_map_or_new", ("map",),
         "-m is a map, an empty folder or a new one; any other folder is refused and left "
         "untouched", _update_map_check, (NotAMapError,)),
)


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

_g = "{:g}".format


def _format(choices: tuple[str, ...] = ("json", "ply"), what: str = "") -> Option:
    return Option("-f", "format", Kind.ENUM,
                  f"output format{f': {what}' if what else ''} (default: json)", default="json",
                  choices=choices)


def _output() -> Option:
    return Option("-o", "output", Kind.FILE_OUT,
                  "write the result to FILE instead of stdout (stdout then stays empty)",
                  metavar="FILE", type=Path, must_exist=False)


def _attrs(scope: CloudScope) -> Option:
    return Option("-p", "attrs", Kind.ATTRS, f"{help_text(scope)}; requires -f ply",
                  metavar="ATTRS", applies=(_PLY,), applies_text="only with -f ply")


def _image(help: str, **kw: Any) -> Option:
    return Option("-i", "image", Kind.IMAGE, help, type=Path, must_exist=True, **kw)


def _map(help: str, must_exist: bool, **kw: Any) -> Option:
    return Option("-m", "map", Kind.MAP, help, type=Path, must_exist=must_exist, **kw)


def _scene(text: str) -> Output:
    return Output("result", "stdout", "json", text, (_JSON,))


def _cloud(text: str) -> Output:
    return Output("result", "stdout", "ply", text, (_PLY,))


_SCENE = "the OpenLABEL 1.0.0 scene description (spec §3)"
_DEPTH_IMAGE = ("the depth image: one 16-bit single-channel PNG of the input's pixel size, each "
                "pixel the metric depth along the optical axis in units of "
                f"1/{DEPTH_UNITS_PER_METRE} m, {NO_DEPTH} where the model gives no valid depth")
_SEGMENTED = ("the segmented image: the input image at the reconstruction's working resolution "
              "(long side at most 1024 px), dimmed, each instance mask painted opaque in its "
              "object's colour")

RECONSTRUCT = Program("reconstruct.sh", "Single-image reconstruction (stdout or -o file).", (
    Command("reconstruct.sh", None, "Single-image reconstruction (stdout or -o file).", (
        _image("input RGB image", required=True),
        _format(("json", "depth", "ply"), "json = the scene description, depth = the depth "
                "image (16-bit PNG), ply = the point cloud"),
        _output(),
        _attrs(CloudScope.IMAGE),
    ), (
        Mode(None, None, (ATTRS_RULE, OUTPUT_RULE, IMAGE_RULE),
             "required", "reconstructs the image with the inference server",
             (Stage.CONNECT, Stage.INFERENCE, Stage.SEGMENT, Stage.EXPORT, Stage.WRITE),
             (_scene(f"{_SCENE}: objects, labels, scores, colours and OBBs in the camera frame"),
              Output("result", "stdout", "png", _DEPTH_IMAGE, (_DEPTH,)),
              _cloud("the point cloud (camera frame, metres) shaped by -p")),
             CloudScope.IMAGE),
    )),
))

MAPPER = Program("mapper.sh", "Multi-frame mapping (persistent map).", (
    Command("mapper.sh", "update", "create or extend a map", (
        Option("-i", "inputs", Kind.IMAGES_OR_VIDEO, "image files, or exactly one video",
               required=True, multiple=True, ordered=True, must_exist=True, type=Path),
        _map("map folder", False, required=True),
        _format(),
        _output(),
        _attrs(CloudScope.MAP),
        Option("-t", "mode", Kind.ENUM,
               "full = whole map with all keyframe poses; single = new input only "
               "(default: full)", default="full", choices=("full", "single")),
        Option("-fps", "fps", Kind.NUMBER,
               f"video frames per second to sample (default: {_g(DEFAULT_FPS)}; ignored for "
               "images)", default=DEFAULT_FPS, exclusive_minimum=0, finite=True,
               omit_if_default=True,
               type=float, applies=(When("inputs", video=True),),
               applies_text="video input only; ignored for images"),
    ), (
        Mode(None, None, (ATTRS_RULE, OUTPUT_RULE, FPS_RULE, *UPDATE_RULES),
             "required", "infers depth and objects of every new keyframe",
             (Stage.SETUP, Stage.INGEST, Stage.INFERENCE, Stage.SFM, Stage.FEATURES_MATCHING,
              Stage.POSE_REFINEMENT, Stage.FOCAL_RERUN, Stage.MAP_FRAME, Stage.DEPTH_ALIGNMENT,
              Stage.PERSIST_FRAMES, Stage.VALIDITY, Stage.OBJECTS, Stage.CLOUD, Stage.EXPORT,
              Stage.COMMIT),
             (_scene(f"{_SCENE} of the whole map (-t full) or of the new input (-t single), map "
                     "coordinates"),
              _cloud("the map cloud (-t full) or the new frames' points (-t single)"),
              Output("map", "-m", "map", "the map folder, created or extended")),
             CloudScope.MAP),
    )),
    Command("mapper.sh", "locate", "camera pose of images in an existing map (read-only)", (
        Option("-i", "inputs", Kind.IMAGES, "one or more image files (a video is refused)",
               required=True, multiple=True, must_exist=True, type=Path),
        _map("existing map folder", True, required=True),
        _format(),
        _output(),
        _attrs(CloudScope.MAP),
        Option("-t", "mode", Kind.ENUM,
               "single = the located camera poses only; full = the whole map plus the "
               "located poses (default: single)", default="single", choices=("full", "single")),
    ), (
        Mode(None, None, (ATTRS_RULE, *LOCATE_RULES, OUTPUT_RULE),
             "conditional", "only for retrieval in maps of more keyframes than are matched "
             "exhaustively",
             (Stage.SETUP, Stage.FEATURES_MATCHING, Stage.POSE, Stage.EXPORT),
             (_scene(f"{_SCENE}: the located camera poses (-t single), or the whole map plus "
                     "them (-t full)"),
              _cloud("the map points visible from the located cameras (-t single) or the whole "
                     "map cloud (-t full); the located poses in the header")),
             CloudScope.MAP, {"map_keyframes_greater_than": UPDATE_EXHAUSTIVE_MAX}),
    )),
))

SEGMENT_IMAGE = Mode(  # its one mode, like reconstruct.sh's: -i is a required option
    None, None, (MIN_SCORE_RULE, OUTPUT_RULE, ARTIFACTS_RULE, IMAGE_RULE),
    "required", "segments the image with the inference server",
    (Stage.CONNECT, Stage.INFERENCE, Stage.SEGMENT, Stage.EXPORT, Stage.ARTIFACTS, Stage.WRITE),
    (_scene(f"{_SCENE} (camera frame)"),
     Output("result", "stdout", "png", _SEGMENTED, (_PNG,), object_regions=True),
     Output("segmentation.json", "-d", "json", f"{_SCENE}, identical to -f json", (_D,)),
     Output("segmented.png", "-d", "png", "the segmented image, identical to -f png", (_D,),
            object_regions=True),
     Output("catalog.csv", "-d", "csv", "one row per object", (_D,)),
     Output("catalog.md", "-d", "markdown", "the catalogue as a table by descending volume",
            (_D,))))

SEGMENT = Program("segment.sh", "Instance segmentation → JSON + OBBs or segmented image, "
                  "artefacts.", (
    Command("segment.sh", None, "Instance segmentation → JSON + OBBs or segmented image, "
            "artefacts.", (
        _image("input RGB image", required=True),
        _format(("json", "png"), "json = the scene description, png = the segmented image"),
        _output(),
        Option("-d", "artifacts", Kind.FOLDER_OUT,
               "also write segmentation.json, segmented.png, catalog.csv and catalog.md into "
               "FOLDER", metavar="FOLDER", type=Path, must_exist=False),
        Option("--min-score", "min_score", Kind.NUMBER,
               f"drop detections below this score (default {_g(DEFAULT_MIN_SCORE)})",
               default=DEFAULT_MIN_SCORE, finite=True),
    ), (SEGMENT_IMAGE,)),
))

_VIEWER = (Output("viewer", "browser", "html", "the viewer page (URL on stderr)"),)

VIEW = Program("view.sh", "Browser visualisation of an image or a map.", (
    Command("view.sh", None, "Browser visualisation of an image or a map.", (
        _image("RGB image to reconstruct and segment", group="source"),
        _map("map folder (opened read-only)", True, group="source"),
        Option("--no-browser", "no_browser", Kind.FLAG, "do not open a browser", default=False),
    ), (
        Mode("image", "image", (IMAGE_RULE,), "required",
             "reconstructs and segments the image with the inference server", (), _VIEWER),
        Mode("map", "map", (MAP_RULE,), "never", "opens the persisted map read-only", (),
             _VIEWER),
    ), exclusive_required=("image", "map")),
), service=False)

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


def build_parser(program: Program, parser_class: type[ArgumentParser] = ArgumentParser
                 ) -> ArgumentParser:
    """The command's argparse parser, built from its definitions."""
    ap = parser_class(prog=program.prog, description=program.description)
    if not program.subcommands:
        _add_options(ap, program.commands[0])
        return ap
    sub = ap.add_subparsers(dest="command", required=True, parser_class=parser_class)
    for cmd in program.commands:
        _add_options(sub.add_parser(cmd.name, help=cmd.help), cmd)  # type: ignore[arg-type]
    return ap


def program_of(cmd: Command) -> Program:
    return next(p for p in PROGRAMS if cmd in p.commands)


def _text(value: Any) -> str:
    return str(value)


def _value(value: Any) -> str:
    """One of several values (``-i a b``): a relative path starting with "-" gets "./", so argparse
    never reads it as an option."""
    text = _text(value)
    return f"./{text}" if text.startswith("-") else text


def argv_of(cmd: Command, mode: Mode, params: Mapping[str, Any]) -> list[str]:
    """The command line (after the program name) for API parameters (option name → value; a list
    for an option that takes several values): what :func:`parse` parses, and what the web service
    runs the command with. Parameters that are None, and those equal to the default of an
    ``omit_if_default`` option, are not passed. An unknown parameter is a
    :class:`ParameterError`."""
    opts = cmd.mode_options(mode)
    names = {o.name for o in opts}
    unknown = sorted(n for n, v in params.items() if v is not None and n not in names)
    if unknown:
        raise ParameterError(f"unrecognized parameters for {cmd.label(mode)}: "
                             f"{', '.join(unknown)}", tuple(unknown))
    argv = [cmd.name] if cmd.name else []
    for o in opts:
        v = params.get(o.name)
        if v is None or (o.omit_if_default and v == o.default):
            continue
        if o.kind is Kind.FLAG:
            argv += [o.flag] if v else []
            continue
        if o.multiple:
            argv += [o.flag, *map(_value, v if isinstance(v, list | tuple) else [v])]
        else:  # flag=value: a value that starts with "-" stays a value
            argv.append(f"{o.flag}={_text(v)}")
    return argv


def parse(cmd: Command, mode: Mode, params: Mapping[str, Any]) -> argparse.Namespace:
    """The command's parsed arguments for API parameters: the argv the command would get
    (:func:`argv_of`), parsed by its own parser, so a bad value is a :class:`ParameterError` with
    argparse's message and the parameters it concerns."""
    return build_parser(program_of(cmd), RaisingParser).parse_args(argv_of(cmd, mode, params))


def validate(cmd: Command, args: argparse.Namespace,
             warn: Callable[[str], None] | None = None) -> argparse.Namespace:
    """Run the mode's rules in order on parsed ``args`` — each rule's check, then its preparation —
    raising the command's own error on the first that fails, and return the values the command
    uses: ``args`` with ``attrs`` parsed to :class:`CloudAttrs`, ``fps`` / ``min_score`` resolved,
    and whatever a rule prepared (the input spec of ``update``, the images and map reader of
    ``locate``). ``warn`` receives the warnings (ignored options)."""
    return validate_deferring(cmd, args, warn)[0]


def validate_deferring(cmd: Command, args: argparse.Namespace,
                       warn: Callable[[str], None] | None = None, defer: tuple[Rule, ...] = ()
                       ) -> tuple[argparse.Namespace, Callable[[], None]]:
    """:func:`validate`, with the preparation of the rules in ``defer`` (e.g. creating the ``-d``
    folder) left to the returned callable, which the command calls once it may write — after its
    inference-server check — so a run refused for that reason leaves nothing behind. Every check
    still runs first, in order, after the refusal of an option that belongs to another mode
    (``Option.modes``: ``server.sh --status --port 8``)."""
    mode = cmd.mode_of(args)
    foreign = [o.flag for o in cmd.foreign(mode, args)]
    if foreign:
        raise UsageError(f"{', '.join(foreign)} {'does' if len(foreign) == 1 else 'do'} not "
                         f"apply to {cmd.label(mode)}")
    ctx = Context(args, argparse.Namespace(**vars(args)), mode, warn or (lambda _msg: None))
    later: list[Callable[[Context], None]] = []
    for rule in mode.rules:
        rule.check(ctx)
        if rule.prepare is not None:
            if rule in defer:
                later.append(rule.prepare)
            else:
                rule.prepare(ctx)

    def prepare() -> None:
        for p in later:
            p(ctx)

    return ctx.values, prepare


def dry_run(cmd: Command, mode: Mode, params: Mapping[str, Any]) -> list[Problem]:
    """Every problem of an API request, without touching anything: argparse's (a bad value, a
    missing parameter) and every rule's check (none prepares anything), each with the parameters
    it concerns, so a form flags them all next to their fields at once (:func:`by_parameter`).
    A parameter argparse refuses is left out of the following parses and rules. A check that
    crashes is the command's internal error (exit 1), told as ``run_main`` tells it."""
    problems: list[Problem] = []
    current = dict(params)
    while True:
        try:
            args = parse(cmd, mode, current)
            break
        except ParameterError as exc:
            problems.append(Problem("arguments", exc.parameters, str(exc), exc.exit_code))
            dropped = [n for n in exc.parameters if current.get(n) is not None]
            if not dropped:  # a missing parameter: no complete arguments to check further
                return problems
            for n in dropped:
                current[n] = None
    refused = {n for p in problems for n in p.parameters}
    ctx = Context(args, argparse.Namespace(**vars(args)), mode, lambda _msg: None)
    for rule in mode.rules:
        if refused & set(rule.options):
            continue
        try:
            rule.check(ctx)
        except OhMySlamError as exc:
            problems.append(Problem(rule.name, rule.options, str(exc), exc.exit_code))
        except Exception as exc:  # e.g. a corrupt map.json
            problems.append(Problem(rule.name, rule.options, internal_message(exc),
                                    ExitCode.INTERNAL))
    return problems


def _map_keyframes_greater_than(args: argparse.Namespace, limit: Any) -> bool:
    from oh_my_slam.mapping.store import MapReader

    return len(MapReader(args.map).frames) > int(limit)


# How each ``Mode.inference_condition`` key is evaluated on the parsed arguments (read-only).
CONDITIONS: dict[str, Callable[[argparse.Namespace, Any], bool]] = {
    "map_keyframes_greater_than": _map_keyframes_greater_than,
}


def needs_inference(mode: Mode, args: argparse.Namespace) -> bool:
    """Whether a run of ``mode`` with ``args`` uses the inference server: always, never, or —
    "conditional" — when its condition holds (any condition that cannot be evaluated counts as
    needing it)."""
    if mode.inference != "conditional":
        return mode.inference == "required"
    try:
        return any(CONDITIONS[k](args, v) for k, v in (mode.inference_condition or {}).items())
    except (KeyError, OhMySlamError, OSError, ValueError):
        return True


def by_parameter(problems: list[Problem]) -> dict[str, list[str]]:
    """The messages of ``problems`` per parameter: each under the first one it concerns (the
    field a form flags), or under "" when it concerns none."""
    out: dict[str, list[str]] = {}
    for p in problems:
        out.setdefault(p.parameters[0] if p.parameters else "", []).append(p.message)
    return out


# --- export as data -------------------------------------------------------------------------------


def _code(code: ExitCode) -> dict[str, Any]:
    return {"code": error_code(code), "exit_code": int(code), "http_status": HTTP_STATUS[code]}


def errors_of(mode: Mode) -> list[dict[str, Any]]:
    """The errors validation can raise, per exit code (argparse's usage errors included), and
    those the mode declares (``Mode.errors``)."""
    classes: list[type[OhMySlamError]] = [UsageError]
    classes += [e for r in mode.rules for e in r.errors]
    if mode.inference != "never":
        classes.append(ServerUnavailableError)
    classes += mode.errors
    by_code: dict[ExitCode, list[str]] = {}
    for c in classes:
        names = by_code.setdefault(c.exit_code, [])
        if c.__name__ not in names:
            names.append(c.__name__)
    return [{**_code(code), "errors": names} for code, names in sorted(by_code.items())]


def _attributes(scope: CloudScope) -> list[dict[str, Any]]:
    d = CloudAttrs()
    return [{"key": a.key, "schema": dict(a.schema), "default": a.format(getattr(d, a.field)),
             "effect": a.effect}
            for a in applicable(scope)]


def _option(mode: Mode, o: Option) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": o.name, "flag": o.flag, "kind": str(o.kind), "help": o.help,
        "required": o.required or o.name == mode.selector,
        "default": None if o.name == mode.selector else o.default,
        "choices": list(o.choices) if o.choices else None,
        "multiple": o.multiple, "ordered": o.ordered,
        "accepts": sorted(ACCEPTS[o.kind]) if o.kind in ACCEPTS else None,
        "must_exist": o.must_exist,
        "minimum": o.minimum, "exclusive_minimum": o.exclusive_minimum, "finite": o.finite,
        "omit_if_default": o.omit_if_default,
        "applies": [w.describe() for w in o.applies], "applies_text": o.applies_text,
    }
    if o.kind is Kind.ATTRS:
        out["attributes"] = _attributes(mode.scope())
        out["default"] = CloudAttrs().describe(mode.scope())
    return out


def operations() -> list[tuple[Program, Command, Mode]]:
    return [(p, c, m) for p in PROGRAMS for c in p.commands for m in c.modes]


def operation_id(prog: Program, cmd: Command, mode: Mode) -> str:
    """A mode's URL-safe id, the web service's operation id: program, subcommand and mode joined
    (``mapper-update``)."""
    parts = [prog.prog.removesuffix(".sh"), cmd.name, mode.name]
    return "-".join(p for p in parts if p)


def writes_map(cmd: Command, mode: Mode) -> Option | None:
    """The option naming the map ``mode`` writes (an output written ``via`` that option)."""
    flags = {o.via for o in mode.outputs}
    return next((o for o in cmd.mode_options(mode) if o.flag in flags and o.kind is Kind.MAP),
                None)


def result_format(cmd: Command, mode: Mode, params: Mapping[str, Any]) -> str | None:
    """Format of ``mode``'s result for API parameters (its stdout output whose condition holds),
    read through the command's own parser (:func:`parse`); None if they do not parse."""
    try:
        args = parse(cmd, mode, params)
    except OhMySlamError:
        return None
    for out in mode.outputs:
        if out.via == "stdout" and (not out.when or any(w.holds(args) for w in out.when)):
            return out.format
    return None


def describe() -> dict[str, Any]:
    """Every operation (command mode) as JSON-serialisable data: its parameters (names, kinds,
    defaults, help, choices, accepted files, bounds, applicability), validation rules, inference
    need, outputs, the errors validation can raise and the timing stages; plus the exit-code table
    (every code a run can end with and its HTTP status). The web service generates its OpenAPI
    document and forms from it."""
    ops = []
    for prog, cmd, mode in operations():
        ops.append({
            "id": cmd.label(mode),
            "prog": prog.prog,
            "command": cmd.name,
            "mode": mode.name,
            "description": cmd.help,
            "inference": mode.inference,
            "inference_text": mode.inference_text,
            "inference_condition": dict(mode.inference_condition) if mode.inference_condition
            else None,
            "parameters": [_option(mode, o) for o in cmd.mode_options(mode)],
            "rules": [{"name": r.name, "parameters": list(r.options), "text": r.text,
                       "codes": sorted({error_code(e.exit_code) for e in r.errors})}
                      for r in mode.rules],
            "outputs": [out.describe() for out in mode.outputs],
            "errors": errors_of(mode),
            "stages": [str(s) for s in mode.stages],
        })
    return {
        "operations": ops,
        "exit_codes": [_code(c) for c in HTTP_STATUS],
        "stages": [str(s) for s in Stage],
    }
