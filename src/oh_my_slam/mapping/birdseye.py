"""The bird's-eye image of a map: ``mapper.sh locate -f png`` (specs/mapper.md "Bird's-eye image").

An orthographic projection along the map's up axis (+z) onto its horizontal plane, one scale for
both axes, as one 8-bit RGB PNG: image right is map +x, image up is map +y (seen from above). The
image always covers the whole map — every map point, every OBB footprint, every keyframe camera
centre — and the cameras it draws, whatever its layers and scope, so one map and one set of
cameras give one pixel grid. From bottom to top:

* ``cloud`` — the map cloud as ``color=rgb`` derives it (``reconstruction.cloud``), each pixel the
  highest point that falls in it, without the ceiling or roof (``ceiling_cut``).
* ``obb`` — every object of the map as the footprint of its upright OBB, outlined opaque in its
  colour (``OUTLINE_PX``, no anti-aliasing: the colour contract's exact triple), with its id in
  that colour on a tag of the background colour at the footprint's centre.
* the cameras: keyframes (``-t full``) as small discs, the last known camera as a diamond, a
  located camera as a triangle pointing where it looks, tagged with its ``Camera.tag`` (the
  number of its image among the inputs, from 1) where no other camera's tag is; each with its
  viewing direction projected onto the horizontal plane.

Scale: ``MIN_METRES_PER_PX`` (1 cm) per pixel, or the smallest 1-2-5 step that keeps the drawn
extent within ``MAX_SIDE_PX``; the grid is aligned to the scale, with ``MARGIN_PX`` around it. The
PNG's text metadata (``Description`` and the keys of ``METADATA``) maps pixels to map coordinates
and records the layers, the ceiling cut, the cameras in the JSON result's representation and a
legend of the marks drawn (``MARKS``: shape, colours, meaning)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import export, store

LAYERS = ("cloud", "obb")
MIN_METRES_PER_PX = 0.01
MAX_SIDE_PX = 2000  # the drawn extent's long side at most (the margins come on top)
MARGIN_PX = 24  # around the drawn extent: room for the markers and tags at its edge
BACKGROUND = (0x15, 0x17, 0x1C)  # the viewer's canvas (viewer/static/lib/viewer.js)
OUTLINE_PX = 2
LABEL_PX = 12  # font size of the id tags (Pillow's default font)
TAG_PAD_PX = 2
# cameras: a located camera and the last known camera white with a black edge, a keyframe the
# viewer's frustum colour (viewer/static/lib/cameras.js FRAME_COLOR); none of them is an object
# colour (OKLab chroma below the palette's 0.08, or lighter than its 0.93)
CAMERA_FILL = (0xFF, 0xFF, 0xFF)
CAMERA_EDGE = (0x00, 0x00, 0x00)
KEYFRAME_FILL = (0xC9, 0xD1, 0xDC)
LOCATED_PX = (11.0, 7.0, 7.0)  # triangle: apex ahead of the centre, base behind it, half-width
LAST_KNOWN_PX = 8.0  # diamond: half-diagonal
KEYFRAME_PX = 3.0  # disc radius
HEADING_PX = {"located": 20.0, "last_known": 20.0, "keyframe": 9.0}
# Ceiling or roof (ceiling_cut): over the points higher than every keyframe camera centre by
# CEILING_CLEARANCE_M, in CEILING_BIN_M bins, a layer is CEILING_SLAB_BINS consecutive bins holding
# at least CEILING_DENSITY times the median non-empty layer's points and covering at least
# CEILING_COVER of
# the map's footprint (its CEILING_CELL_M cells that hold any point). Everything from
# CEILING_MARGIN_M below the lowest such layer up is left out.
CEILING_CLEARANCE_M = 0.10
CEILING_BIN_M = 0.05
CEILING_SLAB_BINS = 3
CEILING_DENSITY = 2.0
CEILING_COVER = 0.35
CEILING_CELL_M = 0.25
CEILING_MARGIN_M = 0.15


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# the legend: each mark the image can draw, by layer or camera kind, with its shape and colours
# ("fill", "edge": #rrggbb) and what it shows, in the order it is drawn
MARKS: dict[str, dict[str, str]] = {
    "cloud": {"shape": "points", "text": "the map's points seen from above, in their own colours"},
    "obb": {"shape": "outline", "text": "an object's OBB footprint, outlined in the object's "
                                        "colour and tagged with its id"},
    "keyframe": {"shape": "disc", "fill": _hex(KEYFRAME_FILL), "edge": _hex(CAMERA_EDGE),
                 "text": "a keyframe's camera, its tick pointing where it looked"},
    "last_known": {"shape": "diamond", "fill": _hex(CAMERA_FILL), "edge": _hex(CAMERA_EDGE),
                   "text": "the map's last known camera, its line pointing where it looked"},
    "located": {"shape": "triangle", "fill": _hex(CAMERA_FILL), "edge": _hex(CAMERA_EDGE),
                "text": "a located camera, its tip and its line pointing where it looks, tagged "
                        "with the number of its image among the inputs (1: the first)"},
}
# the text metadata: key -> what it holds
METADATA = {
    "pixel_to_map": "2 x 3 affine matrix M: map (x, y) = M (u, v, 1)",
    "map_to_pixel": "its inverse: pixel (u, v) = M (x, y, 1)",
    "metres_per_pixel": "the scale",
    "layers": "the layers drawn under the cameras (-l)",
    "ceiling_cut_z": "the height from which map points are left out (null: none)",
    "cameras": "the cameras drawn, as the JSON result represents them",
    "map": "the map folder",
    "legend": "the marks drawn (MARKS): each one's mark, shape, colours and meaning",
}


@dataclass(frozen=True)
class Camera:
    """A camera the image draws: ``kind`` "located", "last_known" or "keyframe"; ``record`` its
    JSON representation (``locate.camera_record``), keyed ``key`` in the metadata; ``tag`` the
    text a located camera is tagged with (none: no tag)."""

    key: str
    kind: str
    T_map_cam: Pose
    record: dict[str, Any]
    tag: str = ""


@dataclass(frozen=True)
class Grid:
    """The pixel grid: pixel (u, v) — continuous, (0, 0) the top-left corner of the top-left
    pixel, so pixel column i, row j has its centre at (i + 0.5, j + 0.5) — shows map (x0 + s u,
    y0 - s v)."""

    x0: float
    y0: float
    s: float
    width: int
    height: int

    def pixels(self, xy: NDArray[Any]) -> NDArray[np.float64]:
        """Continuous pixel coordinates (N, 2) of map points (N, 2+)."""
        xy = np.asarray(xy, np.float64).reshape(-1, np.shape(xy)[-1])
        return np.column_stack([(xy[:, 0] - self.x0) / self.s, (self.y0 - xy[:, 1]) / self.s])

    def pixel_to_map(self) -> list[list[float]]:
        return [[self.s, 0.0, self.x0], [0.0, -self.s, self.y0]]

    def map_to_pixel(self) -> list[list[float]]:
        return [[_r(1 / self.s), 0.0, _r(-self.x0 / self.s)],
                [0.0, _r(-1 / self.s), _r(self.y0 / self.s)]]


def _r(v: float) -> float:
    """``v`` without the noise of binary floating point (the grid is a whole number of pixels of
    a 1-2-5 scale)."""
    return round(v, 9) + 0.0  # + 0.0: never -0.0


def nice_scale(extent: float) -> float:
    """Metres per pixel: ``MIN_METRES_PER_PX``, or the smallest 1-2-5 step that draws ``extent``
    metres in at most ``MAX_SIDE_PX`` pixels."""
    need = max(MIN_METRES_PER_PX, extent / MAX_SIDE_PX)
    k = math.floor(math.log10(need))
    return next(s for s in (float(f"{step}e{k}") for step in (1, 2, 5, 10))
                if s >= need * (1 - 1e-9))


def grid_for(xy: NDArray[Any]) -> Grid:
    """The grid that covers the map points ``xy`` (N, 2; at least one), aligned to the scale,
    with ``MARGIN_PX`` around them."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    s = nice_scale(float(max(hi - lo)))
    i0, i1 = math.floor(lo[0] / s), math.floor(hi[0] / s)
    j0, j1 = math.floor(lo[1] / s), math.floor(hi[1] / s)
    return Grid(_r((i0 - MARGIN_PX) * s), _r((j1 + 1 + MARGIN_PX) * s), s,
                i1 - i0 + 1 + 2 * MARGIN_PX, j1 - j0 + 1 + 2 * MARGIN_PX)


def ceiling_cut(xyz: NDArray[Any], camera_top: float | None) -> float | None:
    """The height from which the points ``xyz`` are a ceiling or roof (see ``CEILING_*``), or
    None: no layer above the cameras (``camera_top``: the highest keyframe camera centre) is
    dense and wide enough."""
    if camera_top is None or not len(xyz):
        return None
    z = np.asarray(xyz[:, 2], np.float64)
    start = camera_top + CEILING_CLEARANCE_M
    above = z >= start
    if not above.any():
        return None
    b = np.floor((z[above] - start) / CEILING_BIN_M).astype(np.int64)
    counts = np.bincount(b)
    slabs = np.convolve(counts, np.ones(CEILING_SLAB_BINS, np.int64), "valid") \
        if len(counts) >= CEILING_SLAB_BINS else counts[:0]
    if not len(slabs):
        return None
    cells = np.floor(np.asarray(xyz[:, :2], np.float64) / CEILING_CELL_M).astype(np.int64)
    key = cells[:, 0] * (1 << 32) + cells[:, 1]
    footprint = len(np.unique(key))
    key_above = key[above]
    level = float(np.median(slabs[slabs > 0]))  # the bins over the highest point hold none
    for i in np.flatnonzero(slabs >= CEILING_DENSITY * level):
        layer = (b >= i) & (b < i + CEILING_SLAB_BINS)
        if len(np.unique(key_above[layer])) >= CEILING_COVER * footprint:
            return float(start + i * CEILING_BIN_M - CEILING_MARGIN_M)
    return None


def highest_rgb(grid: Grid, xyz: NDArray[Any], rgb: NDArray[Any]
                ) -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
    """(flat pixel index, colour) of every pixel that points fall in: the colour of its highest
    point (equal heights: the later point)."""
    uv = np.floor(grid.pixels(np.asarray(xyz)[:, :2])).astype(np.int64)
    pix = uv[:, 1] * grid.width + uv[:, 0]
    order = np.lexsort((np.asarray(xyz)[:, 2], pix))
    p = pix[order]
    last = np.ones(len(p), bool)
    last[:-1] = p[1:] != p[:-1]
    return p[last], np.asarray(rgb, np.uint8)[order[last]]


def footprint(obj: Any) -> NDArray[np.float64]:
    """The four horizontal corners (4, 2) of an object's upright OBB, in order around it."""
    hw, hd = obj.obb.size[0] / 2, obj.obb.size[1] / 2
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    return np.asarray(obj.obb.center[:2] + local @ obj.obb.R[:2, :2].T, np.float64)


def heading(T: Pose) -> NDArray[np.float64]:
    """The viewing direction (camera +z) projected onto the horizontal plane, as a unit vector;
    for a camera looking straight up or down, its image's up direction instead."""
    d = np.asarray(T.R[:2, 2], np.float64)
    if float(np.hypot(*d)) <= 1e-6:
        d = -np.asarray(T.R[:2, 1], np.float64)  # a rotation's z and y are never both vertical
    return d / float(np.hypot(*d))


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.load_default(size)


def _draw_objects(draw: Any, grid: Grid, objects: Sequence[Any]) -> None:
    """Every footprint's outline, then every id tag: centred above its footprint (inside the
    image), so that it never hides its own outline."""
    corners = {o.id: grid.pixels(footprint(o)) - 0.5 for o in objects}
    for o in objects:
        pts = [tuple(p) for p in corners[o.id]]
        draw.line([*pts, pts[0]], fill=tuple(o.color), width=OUTLINE_PX, joint="curve")
    font = _font(LABEL_PX)
    for o in objects:
        c = corners[o.id]
        left, top, right, bottom = draw.textbbox((0, 0), str(o.id), font=font)
        w, h = right - left + 2 * TAG_PAD_PX, bottom - top + 2 * TAG_PAD_PX
        x = round(min(max(float(c[:, 0].mean()) - w / 2, 0), grid.width - w))
        y = round(min(max(float(c[:, 1].min()) - OUTLINE_PX - h, 0), grid.height - h))
        draw.rectangle((x, y, x + w - 1, y + h - 1), fill=BACKGROUND, outline=tuple(o.color))
        draw.text((x + TAG_PAD_PX - left, y + TAG_PAD_PX - top), str(o.id), fill=tuple(o.color),
                  font=font)


TAG_OFFSETS_PX = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))
TAG_GAP_PX = 10  # from a camera's centre to its tag, then a ring farther out per try


def tag_box(c: NDArray[Any], w: int, h: int, taken: list[tuple[int, int, int, int]],
            width: int, height: int) -> tuple[int, int, int, int]:
    """Where a camera's tag of ``w`` x ``h`` pixels goes, near its centre ``c``: the first spot
    around it (below, above, right, left, then the diagonals, on rings ``TAG_GAP_PX`` apart)
    inside the image that overlaps no box of ``taken``, else the first one."""
    spots = []
    for ring in (1, 2, 3):
        for dx, dy in TAG_OFFSETS_PX:
            x = c[0] + dx * TAG_GAP_PX * ring - w / 2 + dx * w / 2
            y = c[1] + dy * TAG_GAP_PX * ring - h / 2 + dy * h / 2
            x, y = round(min(max(x, 0), width - w)), round(min(max(y, 0), height - h))
            spots.append((x, y, x + w - 1, y + h - 1))
    free = (b for b in spots if not any(b[0] <= t[2] and t[0] <= b[2] and b[1] <= t[3]
                                         and t[1] <= b[3] for t in taken))
    return next(free, spots[0])


def _draw_tag(draw: Any, grid: Grid, cam: Camera, taken: list[tuple[int, int, int, int]]) -> None:
    """A located camera's tag: its text, black on white with a black edge, where no other tag
    is."""
    font = _font(LABEL_PX)
    left, top, right, bottom = draw.textbbox((0, 0), cam.tag, font=font)
    w, h = right - left + 2 * TAG_PAD_PX, bottom - top + 2 * TAG_PAD_PX
    box = tag_box(grid.pixels(cam.T_map_cam.t[:2])[0] - 0.5, w, h, taken, grid.width,
                  grid.height)
    taken.append(box)
    draw.rectangle(box, fill=CAMERA_FILL, outline=CAMERA_EDGE)
    draw.text((box[0] + TAG_PAD_PX - left, box[1] + TAG_PAD_PX - top), cam.tag, fill=CAMERA_EDGE,
              font=font)


def _draw_camera(draw: Any, grid: Grid, cam: Camera) -> None:
    c = grid.pixels(cam.T_map_cam.t[:2])[0] - 0.5
    d = heading(cam.T_map_cam) * [1.0, -1.0]  # image v grows down
    n = np.array([-d[1], d[0]])
    tip = tuple(c + d * HEADING_PX[cam.kind])
    if cam.kind == "keyframe":
        draw.line([tuple(c), tip], fill=KEYFRAME_FILL, width=1)
        r = KEYFRAME_PX
        draw.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=KEYFRAME_FILL,
                     outline=CAMERA_EDGE)
        return
    draw.line([tuple(c), tip], fill=CAMERA_EDGE, width=4)
    draw.line([tuple(c), tip], fill=CAMERA_FILL, width=2)
    if cam.kind == "located":
        ahead, behind, half = LOCATED_PX
        shape = [c + d * ahead, c - d * behind + n * half, c - d * behind - n * half]
    else:
        r = LAST_KNOWN_PX
        shape = [c + d * r, c + n * r, c - d * r, c - n * r]
    draw.polygon([tuple(p) for p in shape], fill=CAMERA_FILL, outline=CAMERA_EDGE)


def birdseye_png(reader: store.MapReader, cameras: Sequence[Camera],
                 layers: Sequence[str] = LAYERS) -> bytes:
    """The bird's-eye image of the map ``reader`` reads with ``cameras`` (the result's, and for
    ``-t full`` every keyframe's), drawing ``layers`` (``LAYERS``) under them."""
    from PIL import Image, ImageDraw

    from oh_my_slam.reconstruction.cloud import derive_cloud

    unknown = sorted(set(layers) - set(LAYERS))
    if unknown or not layers:
        raise ValueError(f"bird's-eye layers must be some of {LAYERS}, got {list(layers)}")
    _, objects = export.map_objects(reader)
    cloud = derive_cloud(export.map_source(export.map_cloud(reader), objects, reader.frames),
                         CloudAttrs(color="rgb"))
    xyz = np.asarray(cloud.xyz, np.float64).reshape(-1, 3)
    centres = np.array([fr.T_map_cam.t for fr in reader.frames]).reshape(-1, 3)
    extent = [xyz[:, :2], centres[:, :2], *(footprint(o) for o in objects),
              np.array([c.T_map_cam.t[:2] for c in cameras]).reshape(-1, 2)]
    grid = grid_for(np.concatenate(extent))
    cut = ceiling_cut(xyz, float(centres[:, 2].max()) if len(centres) else None)
    img = np.empty((grid.height, grid.width, 3), np.uint8)
    img[:] = BACKGROUND
    if "cloud" in layers and len(xyz):
        keep = slice(None) if cut is None else xyz[:, 2] < cut
        pix, rgb = highest_rgb(grid, xyz[keep], np.asarray(cloud.rgb)[keep])
        img.reshape(-1, 3)[pix] = rgb
    canvas = Image.fromarray(img)
    draw = ImageDraw.Draw(canvas)
    draw.fontmode = "1"  # no anti-aliasing: every pixel drawn is exactly its colour
    if "obb" in layers:
        _draw_objects(draw, grid, objects)
    order = {"keyframe": 0, "last_known": 1, "located": 2}
    ordered = sorted(cameras, key=lambda c: order[c.kind])
    for cam in ordered:
        _draw_camera(draw, grid, cam)
    taken: list[tuple[int, int, int, int]] = []
    for cam in ordered:  # over every marker, each where no other tag is
        if cam.tag:
            _draw_tag(draw, grid, cam, taken)
    drawn = ",".join(k for k in LAYERS if k in layers)
    kinds = {c.kind for c in cameras}
    legend = [{"mark": k, **MARKS[k]} for k in MARKS if k in layers or k in kinds]
    text = {
        "Description": description(reader.root.name, grid.s, drawn),
        "pixel_to_map": _compact(grid.pixel_to_map()),
        "map_to_pixel": _compact(grid.map_to_pixel()),
        "metres_per_pixel": _compact(grid.s),
        "layers": drawn,
        "ceiling_cut_z": _compact(None if cut is None else round(cut, 6)),
        "cameras": _compact({c.key: c.record for c in cameras}),
        "map": str(reader.root),
        "legend": _compact(legend),
    }
    return png_bytes(np.asarray(canvas), text=text)


LAYER_WORDS = {"cloud": "its point cloud", "obb": "its objects' OBBs"}


def description(name: str, s: float, layers: str) -> str:
    """The image's ``Description``, in words: the map, the projection and its axes, the scale and
    what it shows."""
    shown = " and ".join(LAYER_WORDS[k] for k in layers.split(","))
    return (f"Bird's-eye image of the map {name}, seen from straight above (an orthographic "
            f"projection along the map's +z: image right is map +x, image up is map +y) at "
            f"{s:g} m per pixel, showing {shown} under the cameras. Its text metadata maps its "
            "pixels to map coordinates.")


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
