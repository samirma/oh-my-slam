"""The point-cloud attributes of every emitted PLY (spec §2.2): definition, defaults, validation.

One definition shared by ``reconstruct.sh -f ply``, ``mapper.sh -f ply``, ``segment.sh`` (``-f ply``
and ``segments.ply``) and the ``view.sh`` controls. ``-p key=value[,key=value…]`` is parsed here;
keys that are not given keep their defaults, and every problem is a :class:`UsageError` (exit 2)
raised before any inference runs.

Scopes: ``IMAGE`` (a single image: every key) or ``MAP`` (a persisted map: the pixel-level keys
``stride``, ``min-depth``, ``max-depth`` and ``edge`` apply before unprojection and are refused),
optionally combined with ``SEGMENT`` (``segment.sh``: ``color`` defaults to ``segment`` and any other
value is refused).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Flag, auto
from typing import Any, Literal

from oh_my_slam.core.errors import UsageError

ColorMode = Literal["rgb", "segment", "height", "none"]
Encoding = Literal["binary", "ascii"]

COLOR_MODES: tuple[ColorMode, ...] = ("rgb", "segment", "height", "none")
ENCODINGS: tuple[Encoding, ...] = ("binary", "ascii")
DEFAULT_EDGE = 0.04  # relative depth jump that marks a flying pixel


class CloudScope(Flag):
    IMAGE = auto()
    MAP = auto()
    SEGMENT = auto()


@dataclass(frozen=True)
class CloudAttrs:
    """Effective attributes (field names are the keys with ``-`` → ``_``)."""

    color: ColorMode = "rgb"
    stride: int = 1
    min_depth: float = 0.0
    max_depth: float = math.inf
    edge: float = DEFAULT_EDGE
    voxel: float = 0.0
    normals: bool = False
    label: bool = False
    encoding: Encoding = "binary"

    @staticmethod
    def defaults(scope: CloudScope) -> CloudAttrs:
        return CloudAttrs(color="segment") if CloudScope.SEGMENT in scope else CloudAttrs()

    def items(self, scope: CloudScope) -> list[tuple[str, str]]:
        """``(key, value)`` of every attribute that applies to ``scope``, in table order."""
        return [(a.key, a.format(getattr(self, a.field))) for a in ATTRIBUTES
                if CloudScope.MAP not in scope or not a.pixel_level]

    def describe(self, scope: CloudScope) -> str:
        """``key=value,…`` of the applicable attributes, defaults included (the PLY comment);
        ``parse_cloud_attrs`` reads it back."""
        return ",".join(f"{k}={v}" for k, v in self.items(scope))


# --- the attribute table ---------------------------------------------------------------------------


def _choice(options: tuple[str, ...]) -> Callable[[str], str]:
    def parse(v: str) -> str:
        if v not in options:
            raise ValueError
        return v
    return parse


def _on_off(v: str) -> bool:
    if v not in ("on", "off"):
        raise ValueError
    return v == "on"


def _stride(v: str) -> int:
    if not re.fullmatch(r"\d+", v) or int(v) < 1:
        raise ValueError
    return int(v)


def _metres(minimum: float, allow_inf: bool = False, strict: bool = False
            ) -> Callable[[str], float]:
    def parse(v: str) -> float:
        x = float(v)
        if math.isnan(x) or (math.isinf(x) and not (allow_inf and x > 0)):
            raise ValueError
        if x < minimum or (strict and x == minimum):
            raise ValueError
        return x
    return parse


def _num(x: Any) -> str:
    x = float(x)
    if math.isinf(x):
        return "inf"
    return str(int(x)) if x.is_integer() else repr(x)


@dataclass(frozen=True)
class AttrSpec:
    key: str  # as written in -p
    field: str  # CloudAttrs field
    values: str  # accepted values, for help and errors
    effect: str  # one line for help texts and UI tooltips
    pixel_level: bool  # applies before unprojection, so single images only
    parse: Callable[[str], Any]  # raises ValueError on a bad value
    format: Callable[[Any], str]


ATTRIBUTES: tuple[AttrSpec, ...] = (
    AttrSpec("color", "color", "rgb|segment|height|none",
             "per-point colour: image colour, object colour, height ramp, or no colour",
             False, _choice(COLOR_MODES), str),
    AttrSpec("stride", "stride", "an integer >= 1", "keep every n-th pixel along each image axis",
             True, _stride, str),
    AttrSpec("min-depth", "min_depth", "metres >= 0", "drop pixels closer than this depth",
             True, _metres(0.0), _num),
    AttrSpec("max-depth", "max_depth", "metres > 0 (or inf)", "drop pixels farther than this depth",
             True, _metres(0.0, allow_inf=True, strict=True), _num),
    AttrSpec("edge", "edge", "a relative depth jump >= 0",
             "drop flying pixels on depth discontinuities (0 disables)", True, _metres(0.0), _num),
    AttrSpec("voxel", "voxel", "metres >= 0",
             "keep one point per voxel of this size (0 = off; colours are not averaged)",
             False, _metres(0.0), _num),
    AttrSpec("normals", "normals", "on|off", "add nx ny nz float properties", False, _on_off,
             lambda b: "on" if b else "off"),
    AttrSpec("label", "label", "on|off", "add an int label property (object id, 0 = unsegmented)",
             False, _on_off, lambda b: "on" if b else "off"),
    AttrSpec("encoding", "encoding", "binary|ascii",
             "binary_little_endian 1.0 or ASCII PLY", False, _choice(ENCODINGS), str),
)
_BY_KEY = {a.key: a for a in ATTRIBUTES}
KEYS = tuple(_BY_KEY)


def applicable(scope: CloudScope) -> tuple[AttrSpec, ...]:
    """The attributes that may be set in ``scope`` (``color`` is listed but fixed for SEGMENT)."""
    return tuple(a for a in ATTRIBUTES if CloudScope.MAP not in scope or not a.pixel_level)


def help_text(scope: CloudScope) -> str:
    """One-line description of ``-p`` for ``scope``, with its defaults."""
    d = CloudAttrs.defaults(scope)
    parts = []
    for a in applicable(scope):
        values = "segment (fixed)" if a.key == "color" and CloudScope.SEGMENT in scope \
            else a.values.replace(" ", "")
        parts.append(f"{a.key}={values} [{a.format(getattr(d, a.field))}]")
    return "point-cloud attributes key=value[,key=value...]: " + ", ".join(parts)


# --- parsing and validation ---------------------------------------------------------------------


def _pairs(spec: str | Iterable[str] | Mapping[str, str] | None) -> list[tuple[str, str]]:
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        return [(str(k).strip(), str(v).strip()) for k, v in spec.items()]
    texts = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for text in texts:
        for item in text.split(","):
            if not item.strip():
                continue
            key, sep, value = item.partition("=")
            if not sep:
                raise UsageError(f"point-cloud attribute {item.strip()!r} is not key=value "
                                 f"(e.g. -p color=rgb,voxel=0.01)")
            out.append((key.strip(), value.strip()))
    return out


def parse_cloud_attrs(spec: str | Iterable[str] | Mapping[str, str] | None,
                      scope: CloudScope) -> CloudAttrs:
    """Validated attributes for ``scope`` from ``-p`` text(s) or a key → value mapping."""
    if (CloudScope.IMAGE in scope) == (CloudScope.MAP in scope):
        raise ValueError("scope must contain exactly one of IMAGE and MAP")
    attrs = CloudAttrs.defaults(scope)
    seen: set[str] = set()
    for key, value in _pairs(spec):
        a = _BY_KEY.get(key)
        if a is None:
            raise UsageError(f"unknown point-cloud attribute {key!r}; valid keys: "
                             + ", ".join(x.key for x in applicable(scope)))
        if key in seen:
            raise UsageError(f"point-cloud attribute {key!r} is given twice")
        seen.add(key)
        if a.pixel_level and CloudScope.MAP in scope:
            raise UsageError(f"{key} is a pixel-level attribute: it applies before unprojection, "
                             "to single images only (reconstruct.sh, segment.sh -i); a map's "
                             "points are already 3D — use voxel to thin them")
        try:
            parsed = a.parse(value)
        except ValueError:
            raise UsageError(f"{key} must be {a.values}, got {value!r}") from None
        if key == "color" and CloudScope.SEGMENT in scope and parsed != "segment":
            raise UsageError(f"segment.sh colours points by object: color is fixed to segment "
                             f"(got {value!r})")
        attrs = replace(attrs, **{a.field: parsed})
    if attrs.min_depth >= attrs.max_depth:
        raise UsageError(f"min-depth ({_num(attrs.min_depth)}) must be smaller than max-depth "
                         f"({_num(attrs.max_depth)})")
    return attrs
