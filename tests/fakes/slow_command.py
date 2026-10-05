"""A stand-in command for the web service's job runner tests:
``python -m tests.fakes.slow_command --seconds=S [--code=N] [--ignore-sigint]`` runs one ``setup``
stage for ``S`` seconds with progress ticks (the real ``core.timing`` events), prints a line on
stderr and exits with ``N`` (2: a usage error with a message) — or 130 when interrupted, as every
command does through ``run_main``; ``--ignore-sigint`` makes it deaf to Ctrl-C.

``SLOW_COMMAND`` is the same command as a registry entry (``slow.sh``, no inference), for tests
that add it to ``commands.spec.PROGRAMS``."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from oh_my_slam.core import timing
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.process import run_main
from oh_my_slam.core.timing import Stage

STEPS = 10
MODULE = "tests.fakes.slow_command"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="slow.sh")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--code", type=int, default=0)
    ap.add_argument("--ignore-sigint", action="store_true")
    args = ap.parse_args(argv)
    if args.ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    with timing.collect():
        with timing.stage(Stage.SETUP):
            for i in range(STEPS):
                time.sleep(args.seconds / STEPS)
                timing.progress(i + 1, STEPS)
    print(f"slow.sh: slept {args.seconds:g} s", file=sys.stderr, flush=True)
    if args.code == 2:
        raise UsageError("asked to fail")
    return args.code


def registry_program() -> object:
    """``slow.sh`` as a ``commands.spec.Program`` (one mode, never inference)."""
    from oh_my_slam.commands import spec

    mode = spec.Mode(None, None, (), "never", "needs nothing", (Stage.SETUP,), ())
    cmd = spec.Command("slow.sh", None, "sleep", (
        spec.Option("--seconds", "seconds", spec.Kind.NUMBER, "how long", type=float),
        spec.Option("--ignore-sigint", "ignore_sigint", spec.Kind.FLAG, "deaf to Ctrl-C",
                    default=False),
    ), (mode,))
    return spec.Program("slow.sh", "sleep", (cmd,))


if __name__ == "__main__":
    run_main("slow.sh", main)
