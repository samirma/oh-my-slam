"""One API operation per command mode, derived from ``oh_my_slam.commands.spec`` (spec §2.6
"Single source of truth"): nothing here names a command, a mode or an option. A request — option
name → value — becomes the command's own command line in four generic steps, by option kind:

* path inputs (``Kind.IMAGE`` …, ``Kind.MAP``) are workspace paths, resolved and confined to the
  workspace (an upload is ``uploads/<id>/<file>``; a map is ``<name>`` or ``maps/<name>``);
* the result file (``Kind.FILE_OUT``, ``-o``) and the artefact folder (``Kind.FOLDER_OUT``,
  ``-d``) are plain names inside the job's ``out/`` folder (the result defaults to
  ``result.<ext>`` of the result's format), so every result is a file of the job;
* a mode whose output is the browser (``view.sh``) never opens one on the service's machine: its
  viewer is served by the service (``BROWSER_FLAG``);
* the command's own parser and rules then check the request (``spec.dry_run``), so the messages
  are the command's and nothing is queued for an invalid request.
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
# The option that keeps a command whose output is the browser from opening one (view.sh): the
# service shows that viewer itself, under /viewer/job/<id>/.
BROWSER_FLAG = "no_browser"
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
    def uses_inference(self) -> bool:
        """Runs in the inference queue: the mode needs the inference server, or may need it."""
        return self.mode.inference != "never"

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
class Prepared:
    """A request turned into the command's command line, or the problems that refuse it."""

    problems: list[Problem] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)  # after ``python -m <module>``
    command: list[str] = field(default_factory=list)  # as typed, with workspace paths
    uploads: list[str] = field(default_factory=list)
    writes: str | None = None  # the map folder the job writes
    result: str | None = None  # the -o file name in out/
    result_format: str | None = None


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


def prepare(op: Operation, raw: Any, workspace: Any, job_dir: Path) -> Prepared:
    """Translate an API request into the command's command line and check it with the command's
    own parser and rules (nothing is written)."""
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
    if op.browser and BROWSER_FLAG in by_name:
        actual[BROWSER_FLAG] = shown[BROWSER_FLAG] = True
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
    prep.problems = spec.dry_run(op.command, op.mode, actual)
    if not prep.problems:
        prep.argv = spec.argv_of(op.command, op.mode, actual)
        prep.command = [op.program.prog, *spec.argv_of(op.command, op.mode, shown)]
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
