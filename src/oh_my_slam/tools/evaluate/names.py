"""The capture file-name grammar of ``examples/ainex-captures`` (high_level_spec.md §5).

``NNN_<motion>_<tilt>.jpg`` — ``NNN`` is the capture order; ``<motion>`` the commanded yaw relative
to frame 001, positive to the left:

* ``bootstrap`` → 0°; ``bootstrap_side1`` / ``bootstrap_side2`` → 0° (after a small sideways step);
* ``bootstrap_leftYYY`` and ``bootstrap_leftYYY_side`` → +YYY°;
* ``left_YYY`` → +YYY°; ``right_to_YYY`` → +YYY° (turning back towards 0°); ``right_YYY`` → −YYY°.

``<tilt>`` is ``level``, or ``up`` / ``down`` relative to the ``level`` frame of the same motion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TILTS = ("level", "up", "down")
IMAGE_SUFFIXES = (".jpg", ".jpeg")
# The spec's same-heading pairs (capture numbers): 053 returns to 001's heading, and 078
# (right_150 = -150°) meets 026 (left_210 = +210°), which closes the 360° loop.
SAME_HEADING_PAIRS = ((1, 53), (26, 78))

_NAME = re.compile(
    r"^(?P<index>\d{3})_(?P<motion>"
    r"bootstrap(?:_side[12]|_left(?P<boot_left>\d{3})(?:_side)?)?"
    r"|left_(?P<left>\d{3})|right_to_(?P<right_to>\d{3})|right_(?P<right>\d{3})"
    r")_(?P<tilt>level|up|down)\.jpe?g$",
    re.IGNORECASE,
)


def wrap_deg(angle: float) -> float:
    """Angle in degrees wrapped to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Capture:
    name: str  # file name
    index: int  # capture order NNN
    motion: str  # e.g. "left_060", "bootstrap_left030_side"
    yaw_deg: float  # commanded yaw relative to frame 001, positive to the left
    tilt: str  # "level" | "up" | "down"


def parse_capture(name: str) -> Capture:
    """Parse one capture file name; ``ValueError`` if it does not follow the grammar."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"not a capture name (NNN_<motion>_<tilt>.jpg): {name!r}")
    groups = m.groupdict()
    if groups["right"] is not None:
        yaw = -float(groups["right"])
    else:
        digits = groups["boot_left"] or groups["left"] or groups["right_to"]
        yaw = float(digits) if digits is not None else 0.0
    return Capture(name, int(groups["index"]), groups["motion"].lower(), yaw,
                   groups["tilt"].lower())


def captures_in(folder: Path) -> list[Capture]:
    """Every capture image of ``folder`` in capture order; hidden files (e.g. ``.DS_Store``) and
    non-images are ignored, an image that does not follow the grammar is an error."""
    names = [p.name for p in Path(folder).iterdir()
             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES]
    caps = [parse_capture(n) for n in names]
    return sorted(caps, key=lambda c: c.index)


def level_siblings(captures: list[Capture]) -> dict[str, Capture]:
    """For each ``up`` / ``down`` capture, the ``level`` capture of the same motion (the nearest in
    capture order); tilted captures without one are left out."""
    levels: dict[str, list[Capture]] = {}
    for c in captures:
        if c.tilt == "level":
            levels.setdefault(c.motion, []).append(c)
    out = {}
    for c in captures:
        if c.tilt != "level" and levels.get(c.motion):
            out[c.name] = min(levels[c.motion], key=lambda lv: abs(lv.index - c.index))
    return out


def same_heading_pairs(captures: list[Capture]) -> list[tuple[Capture, Capture]]:
    """The spec's same-heading pairs present in ``captures`` (checked against the grammar)."""
    by_index = {c.index: c for c in captures}
    pairs = []
    for a, b in SAME_HEADING_PAIRS:
        if a in by_index and b in by_index:
            ca, cb = by_index[a], by_index[b]
            if abs(wrap_deg(ca.yaw_deg - cb.yaw_deg)) > 1e-9:
                raise ValueError(f"{ca.name} and {cb.name} do not share a commanded heading")
            pairs.append((ca, cb))
    return pairs
