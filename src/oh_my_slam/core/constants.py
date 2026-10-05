"""Defaults and input types the commands' option definitions (``oh_my_slam.commands``) share with
the code that applies them. No imports, so reading the definitions (``-h``, the web service's
API description) loads nothing heavy."""

from __future__ import annotations

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

DEFAULT_FPS = 2.0  # mapper.sh update -fps: video frames sampled per second (spec §2.3)
DEFAULT_MIN_SCORE = 0.5  # segment.sh -i --min-score: detection confidence threshold (spec §2.4)
