"""The argparse parser classes the commands' definitions (``commands/spec.py``) are built with."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, NoReturn

from oh_my_slam.core.errors import ExitCode, UsageError

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


class RaisingParser(ArgumentParser):
    """The same parser for callers that are not a command (the web service): an argument error is
    a :class:`UsageError` carrying argparse's own message, and nothing is printed."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)
