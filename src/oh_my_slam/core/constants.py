"""Defaults and input types the commands' option definitions (``oh_my_slam.commands``) share with
the code that applies them. No imports, so reading the definitions (``-h``, the web service's
API description) loads nothing heavy."""

from __future__ import annotations

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

DEFAULT_FPS = 2.0  # mapper.sh update -fps: video frames sampled per second (spec §2.3)
DEFAULT_MIN_SCORE = 0.5  # segment.sh --min-score: detection confidence threshold (spec §2.4)
DEFAULT_DATA = "~/oh-my-slam-data"  # server.sh --data: the workspace of maps and uploads (§2.6)

# The inference server's entry point (spec §2.1, ``commands.entry_points``) and the command that
# starts it as the user types it at the root of the checkout (``CHECKOUT``), which the errors of
# the commands that need it, the web service's health and the agent skill name.
CHECKOUT = "./"
INFERENCE_SERVER_PROG = "start_inference_server.sh"
START_INFERENCE_SERVER = CHECKOUT + INFERENCE_SERVER_PROG

# reconstruct.sh -f depth (spec §2.2): a 16-bit PNG whose pixel value is the metric depth along the
# optical axis times DEPTH_UNITS_PER_METRE (1/256 m ≈ 3.9 mm steps, up to 255.996 m), NO_DEPTH
# where the model gives no valid depth
DEPTH_UNITS_PER_METRE = 256
NO_DEPTH = 0

# The reconstruction's working resolution (spec §2.2): the long side, in pixels, of the depth
# grid an image is reconstructed on, whose pixels the pixel-level point-cloud attributes count
MAX_GRID_SIDE = 1024

# Map registration: a map of up to UPDATE_EXHAUSTIVE_MAX keyframes is matched exhaustively (no
# inference server needed); a larger one through the RETRIEVAL_TOP_K most similar keyframes by the
# server's retrieval descriptor, so mapper.sh locate needs the server only then.
UPDATE_EXHAUSTIVE_MAX = 150
RETRIEVAL_TOP_K = 30
