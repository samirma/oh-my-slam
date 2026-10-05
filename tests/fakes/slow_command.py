"""A stand-in command for the web service's job runner tests: ``python -m tests.fakes.slow_command
<seconds> [<exit code>]`` runs one ``setup`` stage for ``seconds`` with progress ticks (the real
``core.timing`` events), prints a line on stderr and exits with ``exit code`` — or 130 when
interrupted, as every command does through ``run_main``."""

from __future__ import annotations

import sys
import time

from oh_my_slam.commands.parser import run_main
from oh_my_slam.core import timing
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.timing import Stage

STEPS = 10


def main(argv: list[str]) -> int:
    seconds = float(argv[0])
    code = int(argv[1]) if len(argv) > 1 else 0
    with timing.collect():
        with timing.stage(Stage.SETUP):
            for i in range(STEPS):
                time.sleep(seconds / STEPS)
                timing.progress(i + 1, STEPS)
    print(f"slow_command: slept {seconds} s", file=sys.stderr, flush=True)
    if code == 2:
        raise UsageError("asked to fail")
    return code


if __name__ == "__main__":
    run_main("slow_command", main)
