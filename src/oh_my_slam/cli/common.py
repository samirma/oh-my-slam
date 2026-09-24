"""Shared command-line plumbing: argument errors → exit 2, exceptions → exit codes, one payload,
and the ``-o <file>`` / ``-p <attrs>`` options of the commands that write a result."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope, help_text, parse_cloud_attrs
from oh_my_slam.core.errors import ExitCode, OhMySlamError, UsageError
from oh_my_slam.core.log import get_logger


class ArgumentParser(argparse.ArgumentParser):
    """argparse with single-dash long options kept (``-fps``) and errors routed to exit 2."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(int(ExitCode.USAGE), f"{self.prog}: error: {message}\n")


def add_result_options(ap: argparse.ArgumentParser, attrs_help: str) -> None:
    """``-o <file>`` (the result goes there, stdout stays empty) and ``-p <attrs>``."""
    ap.add_argument("-o", dest="output", type=Path, metavar="FILE",
                    help="write the result to FILE instead of stdout (stdout then stays empty)")
    ap.add_argument("-p", dest="attrs", action="append", metavar="ATTRS", help=attrs_help)


def attrs_help(scope: CloudScope, requires: str) -> str:
    return f"{help_text(scope)}; {requires}"


def cloud_attrs_arg(values: list[str] | None, scope: CloudScope, *, writes_ply: bool,
                    requires: str) -> CloudAttrs:
    """The validated ``-p`` attributes, checked before any server connection or inference; ``-p``
    without a PLY output is a usage error."""
    if values and not writes_ply:
        raise UsageError(f"-p sets point-cloud attributes, which {requires}")
    return parse_cloud_attrs(values, scope)


def run_main(prog: str, main: Callable[[list[str]], int], argv: list[str] | None = None) -> NoReturn:
    """Run ``main`` and exit with the mapped code; human-facing text goes to stderr only."""
    args = sys.argv[1:] if argv is None else argv
    log = get_logger()
    try:
        code = main(args)
    except OhMySlamError as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        code = int(exc.exit_code)
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        code = 130
    except BrokenPipeError:
        code = 0
    except Exception as exc:
        if os.environ.get("OH_MY_SLAM_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        else:
            log.debug("internal error", exc_info=True)
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
