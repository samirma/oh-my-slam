"""The argparse parser classes the commands' definitions (``commands/spec.py``) are built with,
and :func:`run_main`, the process wrapper of every entry point (exceptions → exit codes)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from oh_my_slam.core.errors import ExitCode, OhMySlamError, UsageError

if TYPE_CHECKING:
    from _typeshed import SupportsWrite


class ArgumentParser(argparse.ArgumentParser):
    """argparse with single-dash long options kept (``-fps``), errors routed to exit 2, and the
    help on stderr: it is human-facing, and stdout carries the result only (spec §4; ``view.sh``
    never writes to stdout)."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(int(ExitCode.USAGE), f"{self.prog}: error: {message}\n")

    def print_help(self, file: SupportsWrite[str] | None = None) -> None:
        super().print_help(sys.stderr if file is None else file)

    def print_usage(self, file: SupportsWrite[str] | None = None) -> None:
        super().print_usage(sys.stderr if file is None else file)


class ParameterError(UsageError):
    """An argument error of :class:`RaisingParser`: argparse's own message, and the names of the
    parameters (options' ``dest``) it concerns, so a form can flag the fields."""

    def __init__(self, message: str, parameters: tuple[str, ...]) -> None:
        super().__init__(message)
        self.parameters = parameters


_FLAG = re.compile(r"(?<![\w-])-{1,2}[A-Za-z][\w-]*")


class RaisingParser(ArgumentParser):
    """The same parser for callers that are not a command (the web service): an argument error is
    a :class:`ParameterError` carrying argparse's own message, and nothing is printed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exit_on_error", False)  # ArgumentError reaches parse_known_args
        super().__init__(*args, **kwargs)

    def _names(self, flags: Sequence[str]) -> tuple[str, ...]:
        dest = {s: a.dest for a in self._actions for s in a.option_strings}
        return tuple(dict.fromkeys(dest[f] for f in flags if f in dest))

    def error(self, message: str) -> NoReturn:
        # "the following arguments are required: -i, -m", "one of the arguments -i -m is required"
        raise ParameterError(message, self._names(_FLAG.findall(message)))

    def parse_known_args(self, args: Sequence[str] | None = None,  # type: ignore[override]
                         namespace: argparse.Namespace | None = None
                         ) -> tuple[argparse.Namespace, list[str]]:
        try:
            return super().parse_known_args(args, namespace)
        except argparse.ArgumentError as err:  # a bad value, two exclusive options, a missing one
            flags = err.argument_name.split("/") if err.argument_name else _FLAG.findall(str(err))
            raise ParameterError(str(err), self._names(flags)) from None


def run_main(prog: str, main: Callable[[list[str]], int], argv: list[str] | None = None) -> NoReturn:
    """Run ``main`` and exit with the mapped code; human-facing text goes to stderr only."""
    args = sys.argv[1:] if argv is None else argv
    try:
        code = main(args)
    except OhMySlamError as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        code = int(exc.exit_code)
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        code = int(ExitCode.INTERRUPTED)
    except BrokenPipeError:
        code = 0
    except Exception as exc:
        print(f"{prog}: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = int(ExitCode.INTERNAL)
    sys.stderr.flush()
    _save_coverage()
    # os._exit avoids interpreter teardown noise from native libraries on stdout/stderr.
    os._exit(code)


def _save_coverage() -> None:
    """Persist coverage data when running under pytest-cov (os._exit skips atexit hooks)."""
    cov_mod = sys.modules.get("coverage")
    if cov_mod is None:
        return
    try:
        cov = cov_mod.Coverage.current()
        if cov is not None:
            cov.stop()
            cov.save()
    except Exception:  # never let coverage bookkeeping change an exit code
        pass
