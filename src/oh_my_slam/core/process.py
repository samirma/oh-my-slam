"""The process wrapper of every entry point (the commands, ``server.sh`` and the viewer's bundle
writer): exceptions become exit codes, human-facing text goes to stderr only."""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable
from typing import NoReturn

from oh_my_slam.core.errors import ExitCode, OhMySlamError


def run_main(prog: str, main: Callable[[list[str]], int], argv: list[str] | None = None) -> NoReturn:
    """Run ``main`` and exit with the mapped code; human-facing text goes to stderr only. Ctrl-C
    is a ``KeyboardInterrupt`` (exit 130) even in a process started with SIGINT ignored (a shell
    ``&`` job, a parent that ignores it), so every entry point can be interrupted."""
    args = sys.argv[1:] if argv is None else argv
    default_sigint()
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


def default_sigint() -> None:
    """Make SIGINT Python's KeyboardInterrupt again (main thread only); a process whose SIGINT is
    handled rather than ignored also passes the default on to the programs it starts (exec resets
    handled signals, but keeps ignored ones)."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.default_int_handler)


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
