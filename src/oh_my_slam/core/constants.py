"""Defaults and input types shared across the package. No imports, so reading them loads
nothing heavy."""

from __future__ import annotations

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
# the source of a keyframe that an earlier tool sampled from a video (``<video>@<time>``); there
# is no video input any more (high_level_spec.md §13)
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

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

# The registration minimum (api.md §4, §6): a pose needs this many inliers, and the update and
# locate ``few_features`` check asks this many keypoints (one constant for both).
REGISTRATION_MIN = 15

# The floor band (map.md §5-§6): a surface within max(FLOOR_BAND_MIN_M, FLOOR_BAND_SIGMAS x the
# floor's height sigma) of the floor estimate is floor: it never makes an occupied cell, a
# passage or a floor-standing object. FLOOR_SIGMA_DEFAULT_M: the sigma of a floor nothing
# measured (too few floor points, or a version recorded before oms/2.2).
FLOOR_BAND_MIN_M = 0.06
FLOOR_BAND_SIGMAS = 3.0
FLOOR_SIGMA_DEFAULT_M = 0.02
# map.md §2 "Ignored pixels": a detection with more than this share of its mask (its box when it
# has no mask) inside the ignored pixels (the camera's and the frame's ignore masks) is dropped
IGNORED_DETECTION_MAX_SHARE = 0.5


def floor_band_m(floor_sigma_m: float) -> float:
    """The floor band's half thickness for a floor of height sigma ``floor_sigma_m``."""
    return max(FLOOR_BAND_MIN_M, FLOOR_BAND_SIGMAS * floor_sigma_m)
