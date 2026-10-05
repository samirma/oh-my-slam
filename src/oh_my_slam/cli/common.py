"""Shared command-line plumbing: argument errors → exit 2, exceptions → exit codes, one payload.
The commands' options and validation are defined in ``commands/spec.py``; the process plumbing
lives in ``commands/parser.py`` so that ``server.sh`` (``oh_my_slam.web``) shares it."""

from __future__ import annotations

from oh_my_slam.commands.parser import ArgumentParser as ArgumentParser
from oh_my_slam.commands.parser import run_main as run_main
