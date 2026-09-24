"""Shared command-line plumbing: argument errors → exit 2, exceptions → exit codes, one payload."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable
from typing import NoReturn

from oh_my_slam.core.errors import ExitCode, OhMySlamError
from oh_my_slam.core.log import get_logger


class ArgumentParser(argparse.ArgumentParser):
    """argparse with single-dash long options kept (``-fps``) and errors routed to exit 2."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(int(ExitCode.USAGE), f"{self.prog}: error: {message}\n")


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
