"""Defaults and input types the commands' option definitions (``oh_my_slam.commands``) share with
the code that applies them. No imports, so reading the definitions (``-h``, the web service's
API description) loads nothing heavy."""

from __future__ import annotations

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

DEFAULT_FPS = 2.0  # mapper.sh update -fps: video frames sampled per second (spec §2.3)
DEFAULT_MIN_SCORE = 0.5  # segment.sh -i --min-score: detection confidence threshold (spec §2.4)
DEFAULT_DATA = "~/oh-my-slam-data"  # server.sh --data: the workspace of maps and uploads (§2.6)

# Map registration: a map of up to UPDATE_EXHAUSTIVE_MAX keyframes is matched exhaustively (no
# inference server needed); a larger one through the RETRIEVAL_TOP_K most similar keyframes by the
# server's retrieval descriptor, so mapper.sh locate needs the server only then.
UPDATE_EXHAUSTIVE_MAX = 150
RETRIEVAL_TOP_K = 30
