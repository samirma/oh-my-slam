"""One API operation per command mode, derived from ``oh_my_slam.commands.spec`` (spec §2.6
"Single source of truth"): nothing here names a command, a mode or an option. A request — option
name → value — becomes the command's own command line in generic steps, by option kind:

* path inputs (``Kind.IMAGE`` …, ``Kind.MAP``) are workspace paths, resolved and confined to the
  workspace (an upload is ``uploads/<id>/<file>``; a map is ``<name>`` or ``maps/<name>``);
* the result file (``Kind.FILE_OUT``, ``-o``) and the artefact folder (``Kind.FOLDER_OUT``,
  ``-d``) are plain names inside the job's ``out/`` folder (the result defaults to
  ``result.<ext>`` of the result's format), so every result is a file of the job;
* the command's own parser and rules then check the request (``spec.dry_run``), so the messages
  are the command's and nothing is queued for an invalid request; ``spec.needs_inference`` says
  whether the job joins the inference queue.

A job is a list of steps, each a Python entry point run as a subprocess: the command itself
(``oh_my_slam.cli.<command>``), and — for a mode whose output is the browser (``view.sh``), or a
single-image request that asks for its viewer — the viewer's own bundle writer
(``python -m oh_my_slam.viewer.bundle save``), which saves the viewer's data in the job's
``viewer/`` folder, so that the service serves that viewer in-process, after a page reload or a
service restart too. A browser mode runs only that step: opening a browser and serving the page
are the service's part.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oh_my_slam.commands import spec
from oh_my_slam.commands.spec import Command, Kind, Mode, Problem, Program
from oh_my_slam.core.errors import HTTP_STATUS, ExitCode, OhMySlamError, UsageError

PATH_IN = frozenset({Kind.IMAGE, Kind.IMAGES, Kind.IMAGES_OR_VIDEO, Kind.MAP})
OUT_DIR = "out"  # the job's folder for everything the command writes
VIEWER_DIR = "viewer"  # the job's saved viewer bundle
VIEWER_MODULE = "oh_my_slam.viewer.bundle"
VIEWER_PROG = "view.sh"  # the viewer step reports its errors as view.sh does
VIEWER_KINDS = {Kind.IMAGE: "image", Kind.MAP: "map"}
EXTENSIONS = {"json": ".json", "ply": ".ply", "png": ".png", "csv": ".csv", "markdown": ".md",
              "html": ".html"}


@dataclass(frozen=True)
class Operation:
    program: Program
    command: Command
    mode: Mode

    @property
    def id(self) -> str:
        """URL-safe id: program, subcommand and mode joined (``segment-image``)."""
        parts = [self.program.prog.removesuffix(".sh"), self.command.name, self.mode.name]
        return "-".join(p for p in parts if p)

    @property
    def label(self) -> str:
        """The command as typed (``segment.sh -i``), the id of ``spec.describe()``."""
        return self.command.label(self.mode)

    @property
    def module(self) -> str:
        """The command's Python entry point, the module ``scripts/_common.sh`` execs."""
        return f"oh_my_slam.cli.{self.program.prog.removesuffix('.sh')}"

    @property
    def options(self) -> list[spec.Option]:
        return self.command.mode_options(self.mode)

    @property
    def browser(self) -> bool:
        return any(o.via == "browser" for o in self.mode.outputs)

    @property
    def viewer_input(self) -> spec.Option | None:
        """The input a viewer of a request shows: a browser mode's selector (else its first image
        or map input); for any other mode its single image, if it takes one."""
        opts = self.options
        if self.browser:
            found = [o for o in opts if o.name == self.mode.selector and o.kind in VIEWER_KINDS]
            found += [o for o in opts if o.kind in VIEWER_KINDS]
            return found[0] if found else None
        return next((o for o in opts if o.kind is Kind.IMAGE), None)

    def writes_map(self) -> spec.Option | None:
        """The option naming the map the mode writes (an output written ``via`` that option)."""
        flags = {o.via for o in self.mode.outputs}
        return next((o for o in self.options if o.flag in flags and o.kind is Kind.MAP), None)


def operations() -> dict[str, Operation]:
    return {op.id: op for op in (Operation(p, c, m) for p, c, m in spec.operations())}


def problem(parameters: tuple[str, ...], message: str, code: ExitCode = ExitCode.USAGE,
            rule: str = "workspace") -> Problem:
    return Problem(rule, parameters, message, code)


@dataclass
class Step:
    """One subprocess of a job: ``python -m <module> <argv>``; ``prog`` names its error lines."""

    prog: str
    module: str
    argv: list[str]
    timings: bool = True  # records OH_MY_SLAM_TIMINGS (the command's own record)


@dataclass
class Prepared:
    """A request turned into the job's steps, or the problems that refuse it."""

    problems: list[Problem] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    command: list[str] = field(default_factory=list)  # as typed, with workspace paths
    uploads: list[str] = field(default_factory=list)
    writes: str | None = None  # the map folder the job writes
    result: str | None = None  # the -o file name in out/
    result_format: str | None = None
    inference: bool = True  # the job uses the inference server
    viewer: bool = False  # the job saves a viewer


def _values(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list | tuple) else [v]


def _result_format(op: Operation, params: Mapping[str, Any]) -> str | None:
    """Format of the mode's result for these parameters (its ``result`` output whose condition
    holds), read through the command's own parser; None if they do not parse."""
    try:
        args = spec.parse(op.command, op.mode, params)
    except OhMySlamError:
        return None
    for out in op.mode.outputs:
        if out.name == "result" and (not out.when or any(w.holds(args) for w in out.when)):
            return out.format
    return None


def prepare(op: Operation, raw: Any, workspace: Any, job_dir: Path, viewer: bool = False
            ) -> Prepared:
    """Translate an API request into the job's steps and check it with the command's own parser
    and rules (nothing is written). ``viewer``: also save the viewer of the request's image."""
    from oh_my_slam.web.workspace import plain_name

    prep = Prepared()
    if not isinstance(raw, Mapping):
        prep.problems.append(problem((), "the request body must be a JSON object of parameters"))
        return prep
    by_name = {o.name: o for o in op.options}
    actual: dict[str, Any] = {}
    shown: dict[str, Any] = {}
    out = job_dir / OUT_DIR
    for name, value in raw.items():
        o = by_name.get(name)
        if o is None or value is None:  # unknown ones: the command's parser names them
            actual[name] = shown[name] = value
            continue
        try:
            if o.kind is Kind.FLAG and not isinstance(value, bool):
                raise UsageError(f"{o.flag} is a flag: give true or false")
            if not (o.multiple or o.repeatable) and isinstance(value, list | tuple):
                raise UsageError(f"{o.flag} takes one value")
            if o.kind in PATH_IN:
                paths = []
                for v in _values(value):
                    p = workspace.map_path(v) if o.kind is Kind.MAP else workspace.resolve(v)
                    uid = workspace.upload_of(p)
                    if uid is not None and not p.exists():
                        raise UsageError(
                            f"{v}: that upload no longer exists (an upload is deleted when its "
                            "job ends); upload the file again")
                    if uid is not None:
                        prep.uploads.append(uid)
                    paths.append(p)
                actual[name] = [str(p) for p in paths] if o.multiple else str(paths[0])
                shown[name] = [workspace.relative(p) for p in paths] if o.multiple \
                    else workspace.relative(paths[0])
                if op.writes_map() is o:
                    prep.writes = str(paths[0])
            elif o.kind in (Kind.FILE_OUT, Kind.FOLDER_OUT):
                plain_name(value, f"{o.flag} (a name in the job's folder)")
                actual[name] = str(out / value)
                shown[name] = value
            else:
                actual[name] = shown[name] = value
        except OhMySlamError as exc:
            prep.problems.append(problem((name,), str(exc), exc.exit_code))
    viewed = op.viewer_input
    if viewer and not op.browser and (viewed is None or actual.get(viewed.name) is None):
        prep.problems.append(problem((), f"{op.label} has no single image to show in a viewer"))
    if prep.problems:
        return prep
    file_out = next((o for o in op.options if o.kind is Kind.FILE_OUT), None)
    prep.result_format = _result_format(op, actual)
    if file_out is not None:
        if actual.get(file_out.name) is None:
            name = "result" + EXTENSIONS.get(prep.result_format or "", "")
            actual[file_out.name] = str(out / name)
            shown[file_out.name] = name
        prep.result = Path(actual[file_out.name]).name
        for o in op.options:  # the result file and a -d folder never share a name
            if o.kind is Kind.FOLDER_OUT and actual.get(o.name) is not None \
                    and Path(actual[o.name]).name == prep.result:
                prep.problems.append(problem((o.name,), f"{o.flag} {shown[o.name]} is the name of "
                                             "the result file; give the folder another name"))
        if prep.problems:
            return prep
    prep.problems = spec.dry_run(op.command, op.mode, actual)
    if prep.problems:
        return prep
    args = spec.parse(op.command, op.mode, actual)
    prep.inference = spec.needs_inference(op.mode, args)
    prep.command = [op.program.prog, *spec.argv_of(op.command, op.mode, shown)]
    if not op.browser:
        prep.steps.append(Step(op.program.prog, op.module,
                               spec.argv_of(op.command, op.mode, actual)))
    if (op.browser or viewer) and viewed is not None:
        kind = VIEWER_KINDS[viewed.kind]
        prep.steps.append(Step(VIEWER_PROG, VIEWER_MODULE,
                               ["save", str(job_dir / VIEWER_DIR), kind, str(actual[viewed.name])],
                               timings=op.browser))
        prep.viewer = True
        prep.inference = prep.inference or kind == "image"
    return prep


def error_body(problems: list[Problem]) -> tuple[int, dict[str, Any]]:
    """HTTP status and body of a refused request: the first problem's code by the generic exit
    status → HTTP rule, every problem, and the messages per parameter (for the form)."""
    first = problems[0]
    status = HTTP_STATUS[first.exit_code]
    return status, {"error": {
        **first.describe(),
        "problems": [p.describe() for p in problems],
        "by_parameter": spec.by_parameter(problems),
    }}
