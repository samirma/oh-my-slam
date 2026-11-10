"""Defaults and input types shared across the package. No imports, so reading them loads
nothing heavy."""

from __future__ import annotations

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

# frames sampled per second of a video (the mapper's internal sequence input; the service sends
# images only)
DEFAULT_FPS = 2.0
DEFAULT_MIN_SCORE = 0.5  # detection confidence threshold
SERVICE_PORT = 9091  # server.sh --port (specs/api.md §1)

# server.sh starts and stops the inference server (specs/inference.md); the errors of the code
# that needs it name the command that restarts both
START_INFERENCE_SERVER = "./server.sh"
RESTART_INFERENCE_SERVER = "./server.sh --stop && ./server.sh"

# The reconstruction's working resolution (spec §2.2): the long side, in pixels, of the depth
# grid an image is reconstructed on, whose pixels the pixel-level point-cloud attributes count
MAX_GRID_SIDE = 1024

# Map registration: a map of up to UPDATE_EXHAUSTIVE_MAX keyframes is matched exhaustively (no
# inference server needed); a larger one through the RETRIEVAL_TOP_K most similar keyframes by the
# server's retrieval descriptor, so locate needs the server only then.
UPDATE_EXHAUSTIVE_MAX = 150
RETRIEVAL_TOP_K = 30
