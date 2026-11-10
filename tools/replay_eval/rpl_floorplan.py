"""The ProcTHOR floor plan in the simulator world: rooms, walls, door gaps, objects, a raster and
ray casting.

Simulator world (x, y, z) = THOR (x, z, y): THOR's y is up. A ProcTHOR wall is a zero-thickness
segment (one per room side). A door's ``holePolygon`` x values are offsets along its ``wall0``
from the wall's first polygon point; the gap between them is the door's free width.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rpl_geometry import Arr
from rpl_layout import LayoutError, need

OUTSIDE, FLOOR, WALL, BORDER, DOOR = 0, 1, 2, 3, 4
WALL_ID = re.compile(r"^wall\|[^|]*\|(-?[\d.]+)\|(-?[\d.]+)\|(-?[\d.]+)\|(-?[\d.]+)")


@dataclass
class Door:
    id: str
    a: Arr  # gap end points (world x, y)
    b: Arr

    @property
    def centre(self) -> Arr:
        return (self.a + self.b) / 2.0

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


@dataclass
class TruthObject:
    id: str
    label: str
    centre: Arr  # world x, y, z
    parent: str | None = None  # id of the object it rests on or is in
    inside: bool = False  # True when the parent is a container
    support_z: float | None = None  # truth height of the surface it rests on, when known
    size: Arr | None = None  # world extent, when the truth gives one


@dataclass
class FloorPlan:
    rooms: list[Arr] = field(default_factory=list)  # polygons (n, 2)
    walls: list[tuple[Arr, Arr]] = field(default_factory=list)  # solid segments, door gaps cut
    doors: list[Door] = field(default_factory=list)
    objects: list[TruthObject] = field(default_factory=list)

    @property
    def has_rooms(self) -> bool:
        return bool(self.rooms)


def thor_xy(p: Any, where: str) -> Arr:
    return np.array([float(need(p, "x", where)), float(need(p, "z", where))])


def label_of(thor_id: str) -> str:
    return thor_id.split("|")[0]


def _wall_ends(wall: dict[str, Any], where: str) -> tuple[Arr, Arr]:
    poly = wall.get("polygon")
    if poly:
        p0 = thor_xy(poly[0], where)
        for q in poly[1:]:
            p1 = thor_xy(q, where)
            if np.linalg.norm(p1 - p0) > 1e-6:
                return p0, p1
    m = WALL_ID.match(str(wall.get("id", "")))
    if m is None:
        raise LayoutError(f"{where}: a wall needs 'polygon' or an id 'wall|room|x0|z0|x1|z1'")
    x0, z0, x1, z1 = (float(v) for v in m.groups())
    return np.array([x0, z0]), np.array([x1, z1])


def _cut(seg: tuple[Arr, Arr], gaps: list[Door], tol: float = 0.02) -> list[tuple[Arr, Arr]]:
    """The parts of a wall segment outside the door gaps that lie on it."""
    p0, p1 = seg
    d = p1 - p0
    length = float(np.linalg.norm(d))
    u = d / length
    normal = np.array([-u[1], u[0]])
    cuts: list[tuple[float, float]] = []
    for g in gaps:
        if abs(float(normal @ (g.a - p0))) < tol and abs(float(normal @ (g.b - p0))) < tol:
            lo, hi = sorted((float(u @ (g.a - p0)), float(u @ (g.b - p0))))
            if hi > 0 and lo < length:
                cuts.append((max(lo, 0.0), min(hi, length)))
    pieces: list[tuple[Arr, Arr]] = []
    start = 0.0
    for lo, hi in sorted(cuts):
        if lo > start + 1e-6:
            pieces.append((p0 + u * start, p0 + u * lo))
        start = max(start, hi)
    if length > start + 1e-6:
        pieces.append((p0 + u * start, p1))
    return pieces


def _objects(items: list[Any], containers: set[str], parent: TruthObject | None,
             out: list[TruthObject], floor_z: float | None) -> None:
    for i, o in enumerate(items):
        where = f"house objects[{i}]"
        oid = str(need(o, "id", where))
        pos = need(o, "position", where)
        centre = np.array([float(need(pos, "x", where)), float(need(pos, "z", where)),
                           float(need(pos, "y", where))])
        obj = TruthObject(oid, label_of(oid), centre,
                          parent=parent.id if parent else None,
                          inside=bool(parent and parent.label in containers),
                          support_z=floor_z if parent is None else None)
        out.append(obj)
        _objects(o.get("children") or [], containers, obj, out, floor_z)


def load_house(house: dict[str, Any], containers: list[str],
               floor_z: float | None = None) -> FloorPlan:
    """The floor plan of a ProcTHOR house JSON (keys rooms, walls, doors, objects)."""
    plan = FloorPlan()
    for i, room in enumerate(house.get("rooms") or []):
        poly = need(room, "floorPolygon", f"house rooms[{i}]")
        plan.rooms.append(np.array([thor_xy(p, f"house rooms[{i}].floorPolygon") for p in poly]))
    walls: dict[str, tuple[Arr, Arr]] = {}
    for i, wall in enumerate(house.get("walls") or []):
        walls[str(wall.get("id", i))] = _wall_ends(wall, f"house walls[{i}]")
    for i, door in enumerate(house.get("doors") or []):
        where = f"house doors[{i}]"
        wid = str(need(door, "wall0", where))
        if wid not in walls:
            raise LayoutError(f"{where}: wall0 '{wid}' is not a wall id")
        hole = need(door, "holePolygon", where)
        p0, p1 = walls[wid]
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        lo, hi = sorted(float(need(h, "x", where)) for h in hole[:2])
        plan.doors.append(Door(str(door.get("id", i)), p0 + u * lo, p0 + u * hi))
    for seg in walls.values():
        plan.walls.extend(_cut(seg, plan.doors))
    _objects(house.get("objects") or [], set(containers), None, plan.objects, floor_z)
    return plan


def seg_distance(P: Arr, a: Arr, b: Arr) -> Arr:
    """Distance of each point of P (n, 2) to segment ab."""
    d = b - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.asarray(np.linalg.norm(P - (a + t[:, None] * d), axis=1), dtype=np.float64)


def inside(P: Arr, poly: Arr) -> Arr:
    """Even-odd point-in-polygon test of each point of P (n, 2)."""
    x, y = P[:, 0], P[:, 1]
    result = np.zeros(len(P), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        result ^= crosses & (x < xc)
    return result


def poly_distance(P: Arr, poly: Arr) -> Arr:
    """0 inside the polygon, else the distance to its boundary."""
    d = np.min([seg_distance(P, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))],
               axis=0)
    return np.where(inside(P, poly), 0.0, d)


@dataclass
class Raster:
    """Truth classes of the floor plan on a world grid: FLOOR inside rooms, WALL within
    ``wall_half_m`` of a solid wall, BORDER (not judged) up to ``border_m`` from it, DOOR on a
    door gap, OUTSIDE elsewhere."""

    origin: Arr
    cell: float
    classes: Any  # (rows, cols) uint8

    def classify(self, P: Arr) -> Any:
        """The class at each world point (n, 2); OUTSIDE off the raster."""
        ij = np.floor((P - self.origin) / self.cell).astype(int)
        rows, cols = self.classes.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < cols) & (ij[:, 1] >= 0) & (ij[:, 1] < rows)
        out = np.full(len(P), OUTSIDE, dtype=np.uint8)
        out[ok] = self.classes[ij[ok, 1], ij[ok, 0]]
        return out



def rasterise(plan: FloorPlan, cell: float = 0.05, wall_half_m: float = 0.075,
              border_m: float = 0.15, margin_m: float = 0.5) -> Raster:
    pts = np.vstack(plan.rooms + [np.array([a, b]) for a, b in plan.walls])
    lo = pts.min(axis=0) - margin_m
    hi = pts.max(axis=0) + margin_m
    cols, rows = (np.ceil((hi - lo) / cell)).astype(int)
    xs = lo[0] + (np.arange(cols) + 0.5) * cell
    ys = lo[1] + (np.arange(rows) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    P = np.column_stack([gx.ravel(), gy.ravel()])
    classes = np.full(len(P), OUTSIDE, dtype=np.uint8)
    for poly in plan.rooms:
        classes[inside(P, poly)] = FLOOR
    for door in plan.doors:
        classes[seg_distance(P, door.a, door.b) <= wall_half_m] = DOOR
    if plan.walls:  # a solid wall wins over a door gap at the jambs
        dist = np.min([seg_distance(P, a, b) for a, b in plan.walls], axis=0)
        classes[(dist > wall_half_m) & (dist <= border_m) & (classes == FLOOR)] = BORDER
        classes[dist <= wall_half_m] = WALL
    return Raster(lo, cell, classes.reshape(rows, cols))


def ray_cast(plan: FloorPlan, origin: Arr, yaw_deg: float, max_m: float = 50.0) -> float | None:
    """Distance from ``origin`` (x, y) along heading ``yaw_deg`` to the nearest solid wall."""
    d = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    best: float | None = None
    for a, b in plan.walls:
        e = b - a
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        w = a - origin
        t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
        u = (d[0] * w[1] - d[1] * w[0]) / den
        if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9 and t <= max_m and (best is None or t < best):
            best = float(t)
    return best
