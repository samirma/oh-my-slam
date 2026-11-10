"""The evaluators are standalone script folders (tools/replay_eval/, tools/evaluate/): their
modules import as top-level names."""

from __future__ import annotations

import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "replay_eval"
for tool in (TOOL, TOOL.parent / "evaluate"):
    if str(tool) not in sys.path:
        sys.path.insert(0, str(tool))
