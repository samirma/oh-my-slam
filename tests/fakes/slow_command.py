"""A stand-in command for the web service's request tests:
``python -m tests.fakes.slow_command --seconds=S [--code=N] [--ignore-sigint]`` runs one ``setup``
stage for ``S`` seconds (the real ``core.timing`` record, written to ``OH_MY_SLAM_TIMINGS`` with
its summary line), prints a line on stderr and its result on stdout — one JSON line with its pid
and the ``time.monotonic()`` (system-wide) of its start and end, so a test can order runs — and
exits with ``N`` (2: a usage error with a message) — or 130 when interrupted, as every command does
through ``run_main``; ``--ignore-sigint`` makes it deaf to Ctrl-C.

``registry_program()`` is the same command as a registry entry (``slow.sh``, no inference unless
asked, a JSON result on stdout), for tests that add it to ``commands.spec.PROGRAMS``;
``map_registry_program()`` a mode of it that takes a map (``-m``, read only) and writes ``note.md``
into a ``-d`` folder, for the web application's registry-change test (a new mode, a new option, a
new output)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from oh_my_slam.core import timing
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.process import run_main
from oh_my_slam.core.timing import Stage

STEPS = 10
MODULE = "tests.fakes.slow_command"
NOTE = "note.md"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="slow.sh")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--code", type=int, default=0)
    ap.add_argument("--ignore-sigint", action="store_true")
    ap.add_argument("-m", dest="map")
    ap.add_argument("-d", dest="folder")
    ap.add_argument("-i", dest="image")
    ap.add_argument("--mood")
    args = ap.parse_args(argv)
    if args.ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    start = time.monotonic()
    with timing.collect() as tm:
        with timing.stage(Stage.SETUP):
            for _ in range(STEPS):
                time.sleep(args.seconds / STEPS)
    timing.report(tm, logging.getLogger("slow.sh"), command="slow.sh")
    print(f"slow.sh: slept {args.seconds:g} s", file=sys.stderr, flush=True)
    print(json.dumps({"slept": args.seconds, "pid": os.getpid(), "start": start,
                      "end": time.monotonic()}), flush=True)
    if args.folder:
        Path(args.folder).mkdir(parents=True, exist_ok=True)
        (Path(args.folder) / NOTE).write_text(f"slept {args.seconds:g} s over {args.map}\n")
    if args.code == 2:
        raise UsageError("asked to fail")
    return args.code


def registry_program(prog: str = "slow.sh", inference: str = "never") -> object:
    """``slow.sh`` as a ``commands.spec.Program`` (one mode, its result a JSON line on stdout);
    ``prog`` and ``inference`` name another such program (e.g. one that needs the inference
    server: ``"required"``)."""
    from oh_my_slam.commands import spec

    mode = spec.Mode(None, None, (), inference, "needs nothing", (Stage.SETUP,), (
        spec.Output("result", "stdout", "json", f"what {prog} says it did"),
        spec.Output("map", "-m", "map", "the map it names (left as it is)", (spec.When("map"),))))
    cmd = spec.Command(prog, None, "sleep", (
        spec.Option("--seconds", "seconds", spec.Kind.NUMBER, "how long", type=float),
        spec.Option("--code", "code", spec.Kind.NUMBER, "the exit status", type=int),
        spec.Option("--ignore-sigint", "ignore_sigint", spec.Kind.FLAG, "deaf to Ctrl-C",
                    default=False),
        spec.Option("-m", "map", spec.Kind.MAP, "a map it writes (it does not)",
                    must_exist=False),
        spec.Option("-i", "image", spec.Kind.IMAGE, "an image it reads (it does not)",
                    must_exist=True),
    ), (mode,))
    return spec.Program(prog, "sleep", (cmd,))


MOOD = "mood"  # an option kind the web application has never seen
NAP_LIMIT = "--seconds must be at most 100 for a nap over an image"


def image_registry_program() -> object:
    """``nap.sh -i IMAGE [--seconds S] [--code N] [--mood M]`` (run by this module): a mode
    that takes a single image,
    with an option of a kind no command has (``MOOD``) and a rule with its own message
    (``NAP_LIMIT``); never inference."""
    from oh_my_slam.commands import spec

    def nap(ctx: spec.Context) -> None:
        if (ctx.args.seconds or 0) > 100:
            raise UsageError(NAP_LIMIT)

    mode = spec.Mode(None, None, (spec.Rule("nap_limit", ("seconds",), "a nap lasts 100 s at most",
                                            nap),),
                     "never", "needs nothing", (Stage.SETUP,), ())
    cmd = spec.Command("nap.sh", None, "sleep over an image", (
        spec.Option("-i", "image", spec.Kind.IMAGE, "the image to sleep over", required=True,
                    must_exist=True),
        spec.Option("--seconds", "seconds", spec.Kind.NUMBER, "how long", type=float),
        spec.Option("--code", "code", spec.Kind.NUMBER, "the exit status", type=int),
        spec.Option("--mood", "mood", MOOD, "how the nap feels"),  # type: ignore[arg-type]
    ), (mode,))
    return spec.Program("nap.sh", "sleep over an image", (cmd,))


def map_registry_program() -> object:
    """``slow.sh -m MAP [--seconds S] [-d DIR]`` as a ``commands.spec.Program``: one mode that takes
    a map and writes ``note.md`` into ``-d`` (never inference)."""
    from oh_my_slam.commands import spec

    folder = spec.When("folder")
    mode = spec.Mode(None, None, (), "never", "needs nothing", (Stage.SETUP,), (
        spec.Output(NOTE, "-d", "markdown", "a note on the nap", (folder,)),))
    cmd = spec.Command("slow.sh", None, "sleep over a map", (
        spec.Option("-m", "map", spec.Kind.MAP, "the map to sleep over", required=True,
                    must_exist=True),
        spec.Option("--seconds", "seconds", spec.Kind.NUMBER, "how long", type=float),
        spec.Option("-d", "folder", spec.Kind.FOLDER_OUT, "also write a note into this folder"),
    ), (mode,))
    return spec.Program("slow.sh", "sleep", (cmd,))


if __name__ == "__main__":
    run_main("slow.sh", main)
