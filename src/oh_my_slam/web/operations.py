"""One API operation per mode of the commands the service offers, derived from
``oh_my_slam.commands.spec`` (spec §2.6 "Single source of truth"): nothing here names a command, a
mode or an option. The programs are those the registry marks ``Program.service`` (``reconstruct.sh``,
``mapper.sh``, ``segment.sh``; ``view.sh`` stays a command only). A request — option name → value —
becomes the command's own command line by option kind:

* path inputs (``Kind.IMAGE`` …, ``Kind.MAP``) are workspace paths, resolved and confined to the
  workspace (an upload is ``uploads/<id>/<file>``; a map is ``<name>`` or ``maps/<name>``);
* the options that only choose where the command writes (``WHERE``: ``-o``, ``-d``) are not
  parameters: the result is the response itself, the command's stdout, and the files a command
  writes to a folder are not offered;
* the command's own parser and rules then check the request (``spec.dry_run``), so the messages are
  the command's and nothing runs for an invalid request; ``spec.needs_inference`` says whether the
  request waits for its turn at the inference server.

A request runs the command's Python entry point (``oh_my_slam.cli.<command>``, the module the shell
script execs) as a subprocess: ``web.runner``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oh_my_slam.commands import spec
from oh_my_slam.commands.spec import Command, Kind, Mode, Problem, Program
from oh_my_slam.core.errors import HTTP_STATUS, ExitCode, OhMySlamError, UsageError

PATH_IN = frozenset({Kind.IMAGE, Kind.IMAGES, Kind.IMAGES_OR_VIDEO, Kind.MAP})
WHERE = frozenset({Kind.FILE_OUT, Kind.FOLDER_OUT})  # options that only choose where it writes
RESULT = "stdout"  # the outputs the response carries: those the command writes to stdout


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
        """The mode's API parameters: its options but those that only choose where it writes."""
        return [o for o in self.command.mode_options(self.mode) if o.kind not in WHERE]

    def writes_map(self) -> spec.Option | None:
        """The option naming the map the mode writes (an output written ``via`` that option)."""
        flags = {o.via for o in self.mode.outputs}
        return next((o for o in self.options if o.flag in flags and o.kind is Kind.MAP), None)

    def entry(self, d: Mapping[str, Any]) -> dict[str, Any]:
        """The mode's ``spec.describe()`` entry as the API offers it: its parameters, their
        applicability, its outputs and its rules without the options that only choose where the
        command writes (and the outputs written there)."""
        names = {o.name for o in self.options}
        flags = {o.flag for o in self.options}
        params = [{**p, "applies": [w for w in p["applies"] if w["option"] in names]}
                  for p in d["parameters"] if p["name"] in names]
        outputs = [o for o in d["outputs"] if o["via"] == RESULT or o["via"] in flags]
        rules = [{**r, "parameters": [n for n in r["parameters"] if n in names]}
                 for r in d["rules"] if r["parameters"][:1] and r["parameters"][0] in names]
        return {**d, "parameters": params, "outputs": outputs, "rules": rules}


def operations() -> dict[str, Operation]:
    return {op.id: op for op in (Operation(p, c, m) for p, c, m in spec.operations()
                                 if p.service)}


def problem(parameters: tuple[str, ...], message: str, code: ExitCode = ExitCode.USAGE,
            rule: str = "workspace") -> Problem:
    return Problem(rule, parameters, message, code)


@dataclass
class Prepared:
    """A request turned into the command's argv, or the problems that refuse it."""

    problems: list[Problem] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)  # the command's, with absolute paths
    command: list[str] = field(default_factory=list)  # as typed, with workspace paths
    uploads: list[str] = field(default_factory=list)  # the uploads the request names
    writes: str | None = None  # the map folder the request writes
    result_format: str | None = None  # the format of the command's stdout
    inference: bool = True  # the request uses the inference server


def _values(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list | tuple) else [v]


def result_format(op: Operation, params: Mapping[str, Any]) -> str | None:
    """Format of the mode's result for these parameters (its stdout output whose condition holds),
    read through the command's own parser; None if they do not parse."""
    try:
        args = spec.parse(op.command, op.mode, params)
    except OhMySlamError:
        return None
    for out in op.mode.outputs:
        if out.via == RESULT and (not out.when or any(w.holds(args) for w in out.when)):
            return out.format
    return None


def media_of(fmt: str | None) -> str:
    """The media type of a response carrying a result of format ``fmt``."""
    return spec.media_type(fmt) if fmt else "application/octet-stream"


def prepare(op: Operation, raw: Any, workspace: Any) -> Prepared:
    """Translate an API request into the command's argv and check it with the command's own
    parser and rules (nothing is written). The uploads a request names are listed even when it is
    refused: it consumes them all the same."""
    prep = Prepared()
    if not isinstance(raw, Mapping):
        prep.problems.append(problem((), "the request body must be a JSON object of parameters"))
        return prep
    by_name = {o.name: o for o in op.options}
    where = {o.name: o for o in op.command.mode_options(op.mode) if o.kind in WHERE}
    actual: dict[str, Any] = {}
    shown: dict[str, Any] = {}
    for name, value in raw.items():
        if name in where:
            prep.problems.append(problem((name,), f"unrecognized parameters for {op.label}: "
                                         f"{name} ({where[name].flag} chooses where the command "
                                         "writes; the response is the result itself)"))
            continue
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
                    if uid is not None:
                        prep.uploads.append(uid)
                        if not p.exists():
                            raise UsageError(
                                f"{v}: that upload no longer exists (an upload is deleted when "
                                "the request it was given to ends); upload the file again")
                    paths.append(p)
                actual[name] = [str(p) for p in paths] if o.multiple else str(paths[0])
                shown[name] = [workspace.relative(p) for p in paths] if o.multiple \
                    else workspace.relative(paths[0])
                if op.writes_map() is o:
                    prep.writes = str(paths[0])
            else:
                actual[name] = shown[name] = value
        except OhMySlamError as exc:
            prep.problems.append(problem((name,), str(exc), exc.exit_code))
    prep.uploads = list(dict.fromkeys(prep.uploads))
    if prep.problems:
        return prep
    prep.problems = spec.dry_run(op.command, op.mode, actual)
    if prep.problems:
        return prep
    prep.result_format = result_format(op, actual)
    prep.inference = inference_of(op, actual)
    prep.command = [op.program.prog, *spec.argv_of(op.command, op.mode, shown)]
    prep.argv = spec.argv_of(op.command, op.mode, actual)
    return prep


def inference_of(op: Operation, actual: Mapping[str, Any]) -> bool:
    """Whether a request of ``op`` with these parameters uses the inference server (a conditional
    mode's condition is read from what it reads, e.g. the map's keyframes)."""
    try:
        return spec.needs_inference(op.mode, spec.parse(op.command, op.mode, actual))
    except OhMySlamError:
        return op.mode.inference != "never"


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
