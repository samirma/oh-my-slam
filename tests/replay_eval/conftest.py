"""The replay evaluator is a standalone script folder: its modules import as top-level names."""

from __future__ import annotations

import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "replay_eval"
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))
