"""The capture file-name grammars of the capture sequences (high_level_spec.md §5).

``examples/ainex-captures``: ``NNN_<motion>_<tilt>.jpg`` — ``NNN`` is the capture order;
``<motion>`` the commanded yaw relative to frame 001, positive to the left:

* ``bootstrap`` → 0°; ``bootstrap_side1`` / ``bootstrap_side2`` → 0° (after a small sideways step);
* ``bootstrap_leftYYY`` and ``bootstrap_leftYYY_side`` → +YYY°;
* ``left_YYY`` → +YYY°; ``right_to_YYY`` → +YYY° (turning back towards 0°); ``right_YYY`` → −YYY°.

``<tilt>`` is ``level``, or ``up`` / ``down`` relative to the ``level`` frame of the same motion.

``examples/camera`` (``PanCapture``): ``img_NNN_pPP_<tilt>.jpg`` — ``NNN`` is the capture order;
``pPP`` the pan position, numbered in pan order: the camera turns left as ``PP`` increases, by a
step angle the name does not record; ``<tilt>`` is ``down``, ``mid`` or ``up``, and the three tilts
of one pan position are consecutive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
_PAN_NAME = re.compile(r"^img_(?P<index>\d{3})_p(?P<pan>\d{2})_(?P<tilt>down|mid|up)\.jpe?g$",
                       re.IGNORECASE)


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


@dataclass(frozen=True)
class PanCapture:
    name: str  # file name
    index: int  # capture order NNN
    pan: int  # pan position PP, in pan order: the camera turns left as it increases
    tilt: str  # "down" | "mid" | "up"


AnyCapture = Capture | PanCapture


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


def parse_pan_capture(name: str) -> PanCapture:
    """Parse one capture file name of ``examples/camera``; ``ValueError`` if it does not follow
    the grammar."""
    m = _PAN_NAME.match(name)
    if m is None:
        raise ValueError(f"not a pan-tilt capture name (img_NNN_pPP_<tilt>.jpg): {name!r}")
    return PanCapture(name, int(m["index"]), int(m["pan"]), m["tilt"].lower())


def _image_names(folder: Path) -> list[str]:
    """The image file names of ``folder``: hidden files (e.g. ``.DS_Store``) and non-images are
    ignored."""
    return [p.name for p in Path(folder).iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES]


def captures_in(folder: Path) -> list[Capture]:
    """Every capture image of ``folder`` in capture order; hidden files (e.g. ``.DS_Store``) and
    non-images are ignored, an image that does not follow the grammar is an error."""
    return sorted((parse_capture(n) for n in _image_names(folder)), key=lambda c: c.index)


def pan_captures_in(folder: Path) -> list[PanCapture]:
    """``captures_in`` for the pan-tilt grammar (``examples/camera``)."""
    return sorted((parse_pan_capture(n) for n in _image_names(folder)), key=lambda c: c.index)


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


def pan_positions(captures: list[PanCapture]) -> dict[int, list[PanCapture]]:
    """The captures of each pan position (in capture order), by pan position in pan order."""
    out: dict[int, list[PanCapture]] = {}
    for c in sorted(captures, key=lambda c: (c.pan, c.index)):
        out.setdefault(c.pan, []).append(c)
    return out


def mid_siblings(captures: list[PanCapture]) -> dict[str, PanCapture]:
    """For each ``up`` / ``down`` capture, the ``mid`` capture of the same pan position (the
    nearest in capture order); tilted captures without one are left out."""
    out = {}
    for group in pan_positions(captures).values():
        mids = [c for c in group if c.tilt == "mid"]
        for c in group:
            if c.tilt != "mid" and mids:
                out[c.name] = min(mids, key=lambda mid: abs(mid.index - c.index))
    return out


def tilt_pairs(captures: list[PanCapture]) -> list[tuple[PanCapture, PanCapture]]:
    """Every pair of tilts of one pan position (the earlier capture first): the camera turned
    only about its tilt axis between them, so they share a heading."""
    return [(a, b) for group in pan_positions(captures).values()
            for i, a in enumerate(group) for b in group[i + 1:]]
