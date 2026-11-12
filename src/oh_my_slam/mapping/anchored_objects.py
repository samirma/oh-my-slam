"""Objects, support surfaces and relations of an anchored map (O8; map.md §5).

The prior-first engine (``mapping.anchored``) fuses each update's new keyframes only; so does
this module, keyframe by keyframe in request order, after the keyframes are placed and fused:

* **Observations.** A keyframe's detections (``segmentation.api.detect_on_grid``, masks on its
  depth grid) are lifted into the map frame on its fused depth (``lift_on_grid``, without the
  support a mask bled onto and without the floor named as an object). Each one is a ``Sighting``:
  its centre, distance, pixel count, which image borders cut its mask, its viewing direction,
  the pose's sigma, and its colour (``robust_rgb``).
* **Association** (``match``, reusable by frame analysis): an observation goes to the object it
  overlaps most — the share of its points inside the object's box grown by ``MATCH_GROW_M`` (or
  of the object's inside its own; both, for other labels), or the nearness of the centres for
  compatible labels — one
  object per observation and one observation per object in a keyframe. Compatible labels match
  from ``MATCH_MIN`` (a generic label, ``GENERIC_LABELS``, goes with any), others (a
  flickering label) only from ``MATCH_ANY``; a detection cut by the image border with half its
  points within an object's grown box is that object, whatever its label (a fragment of it), and
  votes for its label with a quarter of its score; round produce the detector confuses
  (``PRODUCE``) goes together. The label has the most evidence (``MapObject.evidence``: the
  scores of the sightings detected as it and those the detector gave the others under it,
  ``Detection.alternatives``), weighted by how well the box and its relief seen from above
  (``relief``) fit the class (``segmentation.api.label_fit``), a specific one first
  (``MapObject.label``). An unmatched
  observation starts a new object, its id the next of ``next_id`` (never reused).
* **Latest wins** (map.md §3). Every object that a keyframe has in view (``IN_VIEW_SHARE`` of its
  samples in the image, within the fused depth) but did not detect is judged on that keyframe's
  depth (``judge``): samples seen through by more than ``tau`` (max(``SEEN_TAU_M``,
  ``SEEN_TAU_REL`` x depth)) against samples seen in place; samples hidden by something nearer
  and those in the bottom ``FOOT_SHARE`` of the box (the support at its foot looks the same with
  or without it) tell nothing. An object the depth sees through (``see_through``: a keyframe
  detected it after a keyframe's depth had seen it gone, or in several of its sightings the
  depth inside its mask lies beyond its own points — a glass globe, a window, trees behind one) is never judged on depth: only a keyframe that has its place in view,
  unhidden and large enough, and does not detect it counts against it, and one such keyframe
  never strikes it. Nor does a keyframe judge an object on depth where one of its detections of
  a comparable size covers the object's place (``_detected_there``: a glass detected under
  another label). After an object's last sighting in the update, it is ``struck``
  when the keyframes that saw it gone outnumber those that saw it in place (a single one must see
  ``GONE_ALONE`` of it through, with a depth image: one monocular view is not enough). A struck object keeps its record for that version and is removed
  by the next update; the cloud voxels inside its box that its witnesses see through are carved
  (``carve``), so the surface behind it shows the latest view.
* **Moved.** A new object of a compatible label, similar size (``MOVE_SIZE``) and colour
  (``MOVE_RGB``) that appears in the update in which an object is struck takes that object's id:
  ``state: moved`` with ``moved_from_m`` its old centre.
* **Published.** Detections are kept from a score of 0.2 (``service.backend.MAP_MIN_SCORE``),
  without masks over 40 % of the image (background) or nearer the camera than 12 cm (the robot).
  An object is listed once two keyframes detected it, or one did, whole (its mask touching no
  image border) and with a score of at least ``TRUSTED_SCORE``, and no keyframe that had it in view saw its place without detecting
  it; once listed it stays listed. A single
  detection that later keyframes see gone is dropped; two or more make it ``struck``.
* **Boxes** are upright boxes (``segmentation.api.fit_object_obb``) fitted to the points of its
  sightings that agree on the centre; floor-standing classes are grounded on the map's floor.
* **Surfaces** (``find_surfaces``): horizontal planes of the fused cloud's points that face up
  (normals within 25 degrees of up, turned towards the cameras: no object sides, walls or
  ceiling) — cells of
  ``SURFACE_CELL_M`` whose points concentrate (``FLAT_SHARE``) in a slab of ``SLAB_BINS`` bins either side at a
  height that many cells share; their connected areas over ``SURFACE_MIN_AREA_M2``. Ids are
  stable across versions (``assign_surface_ids``: height within ``SURFACE_ID_DZ`` and overlapping
  extent).
* **Relations.** ``supported_by``: the surface or the top of an object within
  ``SUPPORT_TOL_M`` of the box's bottom under its centre (a surface when both fit, unless the
  object's top is clearly nearer and it is no furniture top; of surfaces as near, the largest;
  a support object is at least as large as the object and no floor covering, which is listed
  only on a room-sized surface); the box, as fitted, rests on it: a view from above sees only
  an object's top, so its height is completed down to the support, found up to
  ``SUPPORT_REACH_M`` below it (``z`` stays ``partial``
  unless a sighting saw its height from the side). ``inside_of``:
  the centre within a container's box (``CONTAINERS``: a bowl, a box, a drawer, a cup) that is
  larger than the object, which then is supported by it too. ``verified_version``: the latest
  version that observed the object or its support with the relation holding.
* **Uncertainty.** ``sigma_centre_m`` combines the sightings' pose sigmas and depth noise (over
  their number) and, while an axis is seen only in part, the spread of their centres.
  ``extent_seen`` per box axis (``extent_from``): ``full`` when a sighting whose mask no image
  border cuts sees the axis across (``ACROSS_COS``), or sightings from both ends of it see it;
  ``z`` when a sighting whose mask neither the top nor the bottom border cuts sees it from the
  side, or when an uncut sighting sees its top from above (``TOP_VIEW_COS``) and a support
  plane is measured under its footprint (``supported_by``: it bounds the bottom). A view where
  the object lies mostly on ignored pixels (``masked_by``) is no view of it: no vote, no vote
  denominator, no extent evidence.

State lives in the mapper folder of each version: ``objects.json`` (objects without points,
surface ids, ``next_id``) and ``objects.npz`` (each object's points and their sighting)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.constants import (
    FLOOR_SIGMA_DEFAULT_M,
    IGNORED_DETECTION_MAX_SHARE,
    floor_band_m,
)
from oh_my_slam.core.geometry import project, unproject_pixels, voxel_keys
from oh_my_slam.core.images import masked_view, on_grid
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.segmentation.api import (
    DOOR_LABEL,
    OBB,
    PASSAGE_LABELS,
    Detection,
    Region,
    compatible,
    door_leaf,
    fit_object_obb,
    label_fit,
    lift_on_grid,
    on_the_floor,
    raised_regions,
    surface_label,
    trim_support,
)

if TYPE_CHECKING:
    from oh_my_slam.mapping.anchored import AnchoredMap, Keyframe, Placement

log = get_logger("oh_my_slam.mapping.anchored_objects")

Array = NDArray[np.float64]

OBJECTS_JSON = "objects.json"
OBJECTS_NPZ = "objects.npz"
UP = np.array([0.0, 0.0, 1.0])

SIGHTING_POINTS = 600  # points kept per sighting (thinned to POINT_VOXEL_M first)
POINT_VOXEL_M = 0.01
SHAPING_SIGHTINGS = 8  # the largest sightings keep their points
MATCH_GROW_M = 0.05
MATCH_MIN = 0.3
MATCH_ANY = 0.7
MATCH_CENTRE_M = 0.15  # compatible labels: centres this near (plus half the size) match
BORDER_PX = 2  # a mask within this of an image border is cut by it
RGB_ERODE_PX = 2
RGB_TRIM = (10.0, 90.0)  # luminance percentiles kept for the robust mean
RGB_VIEWS = 3
JUDGE_SAMPLES = 300
IN_VIEW_SHARE = 0.6
IN_VIEW_MARGIN_PX = 2
JUDGE_MAX_DEPTH_M = 6.0  # as anchored.FUSE_MAX_DEPTH_M
SEEN_TAU_M = 0.04
SEEN_TAU_REL = 0.04
FOOT_SHARE = 0.25
JUDGE_MIN = 15
GONE_SHARE = 0.6
PLACE_SHARE = 0.4
GONE_ALONE = 0.7
MOVE_SIZE = 1.6
MOVE_RGB = 60.0
MOVE_MIN_M = 0.10
AGREE_REL = 2.5  # sightings whose centres are within this many median deviations shape the box
AGREE_MIN_M = 0.05
DEPTH_NOISE_REL = 0.02
SENSOR_DEPTH_REL = 0.01  # a depth image's relative noise
MONO_DEPTH_REL = 0.05  # monocular depth's, once brought onto the map
MIN_SIGMA_M = 0.005
ACROSS_COS = 0.8
# a view sees an object's top from above when its ray's z is below minus this (the same |ray z|
# threshold as a side view's, ``ACROSS_COS``; map.md §5 ``extent_seen``, P21)
TOP_VIEW_COS = ACROSS_COS
ENDS_COS = 0.3
SURFACE_CELL_M = 0.05
SURFACE_BIN_M = 0.01
SLAB_BINS = 2  # the slab: this many bins either side (5 cm: monocular depth spreads a plane)
BAND_BINS = 10  # the column a cell's slab share is counted in: +-10 cm
FLAT_SHARE = 0.5
SURFACE_MIN_AREA_M2 = 0.06
SURFACE_MIN_POINTS = 200
UP_NORMAL_Z = 0.9  # a point of a support plane's top: its normal within 25 degrees of up
CORE_SHARE = 0.25  # of the least area: cells left after one erosion (no lines of cells)
SURFACE_PEAK_BINS = 2
SURFACE_ID_DZ = 0.04
SURFACE_ID_IOU = 0.25
SURFACE_ID_CENTRE_M = 0.3
SUPPORT_TOL_M = 0.05
SUPPORT_PREFER_OBJECT_M = 0.01
CARVE_GROW_M = 0.02
BEHIND_SHARE = 0.3  # of a detection's mask: depth beyond its own points (it is seen through)
SEE_THROUGH_VIEWS = 2
ABSENT_SHARE = 0.5  # a see-through object's verdict when a keyframe does not detect it
DETECT_MIN_PX = 200  # ... judged only where it would cover this many pixels
COVERED_SHARE = 0.5  # a keyframe that detects something over this share of an object's place
COVER_SIZE = 4.0  # ... at most this many times its size, sees it there: no depth verdict
# labels the detector gives what it cannot name better: they never outvote a specific label that
# has SPECIFIC_SHARE of their votes, and they go with any label
GENERIC_LABELS = frozenset({"ball", "food", "fruit", "vegetable", "toy", "container"})
SPECIFIC_SHARE = 0.25
# relief (``relief``): of an object at most RELIEF_MAX_M across, seen from above (a ray within
# 45 degrees of down) in RELIEF_VIEWS sightings, its top inside (RELIEF_INNER of its half
# footprint) against its rim (from RELIEF_RIM): lower by RELIEF_M or RELIEF_SHARE of its height
# is "open", higher by as much "closed"
RELIEF_MAX_M = 0.6
RELIEF_DOWN = 0.7
RELIEF_VIEWS = 1
RELIEF_INNER = 0.4
RELIEF_RIM = 0.6
RELIEF_M = 0.01
RELIEF_SHARE = 0.25
RELIEF_MIN_POINTS = 10
MIN_OBJECT_M = 0.03  # a box smaller than this on its longest side is a fragment, not listed
TRUSTED_SCORE = 0.5  # a single detection lists an object only from this score
MAX_MASK_SHARE = 0.4  # a mask over more of the image than this is the background
NEAR_CAMERA_M = 0.12  # a detection nearer the camera than this is on the robot
PRODUCE = frozenset({"apple", "orange", "tomato", "peach", "lemon", "pear", "onion", "potato"})
FRAGMENT_INSIDE = 0.5  # a border-cut detection with this share of its points in an object's box
CUT_VOTE = 0.25  # (grown by MATCH_GROW_M) is that object; it votes with this share of its score
# floor coverings: never a support, and listed only on a room-sized surface (ROOM_FLOOR_M2)
FLOOR_COVERINGS = frozenset({"rug", "carpet", "mat", "doormat", "bath mat"})
ROOM_FLOOR_M2 = 3.0
SUPPORT_AREA_RATIO = 1.0  # a support object's footprint is at least the object's
SUPPORT_SAME_M = 0.01
SUPPORT_REACH_M = 0.30  # surfaces this near in gap: the larger one supports
VIEW_NEAR_M = 0.05  # a point nearer a camera than this is not in its view (detection_votes)
CONTAINERS = frozenset({
    "bowl", "box", "cardboard box", "basket", "bin", "trash can", "recycling bin", "bucket",
    "drawer", "cup", "mug", "pot", "pan", "sink", "tray", "crate", "container", "storage box",
    "lunch box", "tissue box", "toolbox", "jar", "vase", "bathtub", "glass",
})


# -- one keyframe as the objects see it ----------------------------------------------------------


@dataclass
class View:
    """A new keyframe: its pose (``T_map_cam``) and sigma, its depth grid (metres as fused, 0:
    none) with that grid's intrinsics, its image on the grid, its detections (masks on the grid)
    and its ignore mask (on the grid)."""

    kid: str
    T: Array
    sigma_pos_m: float
    K: Intrinsics
    depth: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    dets: list[Detection]
    ignore: NDArray[np.bool_] | None = None
    depth_rel: float = DEPTH_NOISE_REL  # relative noise of its depth (monocular: more)

    @property
    def centre(self) -> Array:
        out: Array = self.T[:3, 3]
        return out

    def camera(self, X: Array) -> Array:
        """Points of the map frame in the camera frame."""
        out: Array = (np.asarray(X, np.float64) - self.centre) @ self.T[:3, :3]
        return out

    def pixels(self, X: Array) -> tuple[Array, Array, Array]:
        """The pixel (u, v; -1 behind or too near) and depth of map-frame points."""
        uv, z = project(self.camera(X), self.K.K())
        ok = z > 0.05
        u = np.where(ok, uv[:, 0], -1.0)
        v = np.where(ok, uv[:, 1], -1.0)
        return u, v, z


@dataclass
class Sighting:
    kid: str
    version: int
    centre: list[float]
    distance: float
    pixels: int
    cut: list[bool]  # mask cut by the left, right, top, bottom border
    ray: list[float]  # unit direction from the camera to the centre
    cam: list[float]  # the camera's centre
    sigma_pos_m: float
    rgb: list[float]
    label: str
    score: float
    depth_rel: float = DEPTH_NOISE_REL  # the keyframe depth's relative noise
    # see-through evidence (``see_through``): the share of the mask's depth beyond the object's
    # own points, and the median range of the mask's depth from the camera
    behind: float = 0.0
    mask_range: float = 0.0

    @property
    def truncated(self) -> bool:
        return any(self.cut)

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_json(d: dict[str, Any]) -> Sighting:
        return Sighting(**d)


def yaw_deg(a: float) -> float:
    """An upright box's yaw in (-90, 90], to 3 decimals."""
    y = round((a + 90.0) % 180.0 - 90.0, 3)
    return 90.0 if y <= -90.0 else y


@dataclass
class Box:
    centre: Array
    size: Array  # x (longer horizontal), y, z
    yaw_deg: float

    @property
    def R(self) -> Array:
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def bottom(self) -> float:
        return float(self.centre[2] - self.size[2] / 2)

    @property
    def top(self) -> float:
        return float(self.centre[2] + self.size[2] / 2)

    def contains(self, X: Array, margin: float = 0.0) -> NDArray[np.bool_]:
        local = (np.asarray(X, np.float64).reshape(-1, 3) - self.centre) @ self.R
        out: NDArray[np.bool_] = np.all(np.abs(local) <= self.size / 2 + margin, axis=1)
        return out

    def footprint_contains(self, xy: Array, margin: float = 0.0) -> bool:
        local = (np.array([xy[0], xy[1], self.centre[2]]) - self.centre) @ self.R
        return bool(np.all(np.abs(local[:2]) <= self.size[:2] / 2 + margin))

    def footprint(self) -> list[list[float]]:
        h = self.size[:2] / 2
        R = self.R[:2, :2]
        return [(R @ np.array(c) + self.centre[:2]).round(4).tolist()
                for c in ((-h[0], -h[1]), (h[0], -h[1]), (h[0], h[1]), (-h[0], h[1]))]

    def to_json(self) -> dict[str, Any]:
        return {"centre": self.centre.tolist(), "size": self.size.tolist(),
                "yaw_deg": self.yaw_deg}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Box:
        return Box(np.asarray(d["centre"], np.float64), np.asarray(d["size"], np.float64),
                   float(d["yaw_deg"]))

    @staticmethod
    def of(obb: OBB) -> Box:
        """The upright box of an OBB: x the longer horizontal side, yaw in (-90, 90]."""
        size = np.asarray(obb.size, np.float64).copy()
        ax = np.asarray(obb.R, np.float64)[:, 0]
        if size[1] > size[0]:
            size[0], size[1] = size[1], size[0]
            ax = np.asarray(obb.R, np.float64)[:, 1]
        return Box(np.asarray(obb.center, np.float64).copy(), size,
                   yaw_deg(math.degrees(math.atan2(float(ax[1]), float(ax[0])))))


@dataclass
class MapObject:
    id: int
    votes: dict[str, float]
    sightings: list[Sighting] = field(default_factory=list)
    points: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    source: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    box: Box | None = None
    state: str = "confirmed"
    moved_from: list[float] | None = None
    first_seen_version: int = 0
    last_seen_version: int = 0
    published: bool = False
    misses: int = 0  # keyframes that saw its place in place without detecting it
    # keyframes that detected it after a keyframe's depth saw it gone (``see_through``), and
    # the latest verdict on it ("seen", "gone", "in_place")
    contradicted: int = 0
    last_verdict: str | None = None
    supported_by: dict[str, Any] | None = None
    inside_of: dict[str, Any] | None = None
    struck_version: int | None = None
    # the scores the detector gave its sightings under the labels they were not detected as
    # (``Detection.alternatives``), summed: label -> score
    others: dict[str, float] = field(default_factory=dict)
    # the keyframes whose view is no view of it (``masked_by``: it lies mostly on their ignored
    # pixels), set on its box as fitted (``ObjectMap.finish``); not stored
    masked: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        """The label with the most evidence (``evidence``), the specific ones first: a specific
        label with at least ``SPECIFIC_SHARE`` of the leading generic one's evidence wins (an
        apple also scored as a ball or a fruit is an apple)."""
        ev = self.evidence()
        best = max(sorted(ev), key=lambda k: ev[k])
        specific = [k for k in sorted(ev) if k not in GENERIC_LABELS]
        if not specific or best not in GENERIC_LABELS:
            return best
        top = max(specific, key=lambda k: ev[k])
        return top if ev[top] >= SPECIFIC_SHARE * ev[best] else best

    def evidence(self) -> dict[str, float]:
        """Each label's evidence: the scores of the sightings detected as it (``votes``) and
        those the detector gave the other sightings under it (``others``: label scores), summed
        over the views, times how well the object's box and relief fit the label
        (``segmentation.api.label_fit``: a mug's rim seen from above, scored as a satellite dish,
        defers to the cup it was also scored as)."""
        ev = {k: self.votes.get(k, 0.0) + self.others.get(k, 0.0)
              for k in sorted(set(self.votes) | set(self.others))}
        if self.box is None or len(ev) < 2:
            return ev
        seen = extent_seen(self)
        whole = seen["x"] == "full" and seen["y"] == "full"
        size = (float(self.box.size[0]), float(self.box.size[1]), float(self.box.size[2]))
        shape = relief(self)
        return {k: v * label_fit(k, size, whole, shape) for k, v in ev.items()}

    @property
    def score(self) -> float:
        return max((s.score for s in self.sightings), default=0.0)

    def detected_as(self) -> list[str]:
        """The labels it was detected as, most votes first; the label chosen first when only
        the label scores (``others``) gave it."""
        out = sorted(self.votes, key=lambda k: (-self.votes[k], k))
        label = self.label
        return out if label in out else [label, *out]

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "votes": self.votes,
                "sightings": [s.to_json() for s in self.sightings],
                "box": None if self.box is None else self.box.to_json(), "state": self.state,
                "moved_from": self.moved_from, "first_seen_version": self.first_seen_version,
                "last_seen_version": self.last_seen_version, "published": self.published,
                "misses": self.misses, "supported_by": self.supported_by,
                "inside_of": self.inside_of, "struck_version": self.struck_version,
                "contradicted": self.contradicted, "last_verdict": self.last_verdict,
                "others": self.others}

    @staticmethod
    def from_json(d: dict[str, Any]) -> MapObject:
        return MapObject(int(d["id"]), {k: float(v) for k, v in d["votes"].items()},
                         [Sighting.from_json(s) for s in d["sightings"]],
                         box=None if d["box"] is None else Box.from_json(d["box"]),
                         state=d["state"], moved_from=d["moved_from"],
                         first_seen_version=int(d["first_seen_version"]),
                         last_seen_version=int(d["last_seen_version"]),
                         published=bool(d["published"]), misses=int(d["misses"]),
                         supported_by=d["supported_by"], inside_of=d["inside_of"],
                         struck_version=d["struck_version"],
                         contradicted=int(d.get("contradicted", 0)),
                         last_verdict=d.get("last_verdict"),
                         others={k: float(v) for k, v in d.get("others", {}).items()})


@dataclass
class Observation:
    """One lifted detection of a keyframe (map frame)."""

    label: str
    score: float
    points: Array
    mask: NDArray[np.bool_]
    sighting: Sighting
    alternatives: dict[str, float] = field(default_factory=dict)

    @property
    def centre(self) -> Array:
        return np.asarray(self.sighting.centre, np.float64)


# -- association (also frame analysis') ----------------------------------------------------------


def alike(a: str, b: str) -> bool:
    """Compatible labels (``segmentation.api.compatible``), one of them generic
    (``GENERIC_LABELS``: a ball, a fruit, a container may be anything of its shape), or both
    round produce the detector confuses (``PRODUCE``: an apple seen as an orange)."""
    return compatible(a, b) or a in GENERIC_LABELS or b in GENERIC_LABELS or (
        a in PRODUCE and b in PRODUCE)


def _extent(points: Array) -> float:
    if not len(points):
        return 0.0
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return float(np.max(hi - lo))


def affinity(label: str, points: Array, centre: Array, o: MapObject, cut: bool = False
             ) -> float:
    """How strongly an observation (its label, map-frame points and centre; ``cut``: its mask
    touches the image border) is object ``o``."""
    if o.box is None or not len(points):
        return 0.0
    inside = float(np.mean(o.box.contains(points, MATCH_GROW_M)))
    if cut and inside >= FRAGMENT_INSIDE:
        # a fragment at the image border, within or touching the object's box: the object,
        # whatever the label it was given
        return max(MATCH_MIN, inside * MATCH_ANY)
    back = 0.0
    if len(o.points):
        lo, hi = points.min(0) - MATCH_GROW_M, points.max(0) + MATCH_GROW_M
        back = float(np.mean(np.all((o.points >= lo) & (o.points <= hi), axis=1)))
    if alike(label, o.label) or any(alike(label, k) for k in o.votes):
        a = max(inside, back)
        gate = MATCH_CENTRE_M + 0.5 * max(_extent(points), float(np.max(o.box.size)))
        d = float(np.linalg.norm(centre - o.box.centre))
        if d < gate:
            a = max(a, MATCH_MIN + (1.0 - MATCH_MIN) * (1.0 - d / gate) * 0.5)
        return a if a >= MATCH_MIN else 0.0
    # another label (a flickering one): the two must hold each other, so that a large object
    # (a table) never takes the detection of what stands on it
    a = min(inside, back)
    return a if a >= MATCH_ANY else 0.0


def match(obs: list[tuple[str, Array, Array]], objects: list[MapObject],
          cut: list[bool] | None = None) -> list[int | None]:
    """The object id of each observation (label, map-frame points, centre; ``cut``: its mask
    touches the image border) of one image, or None: strongest affinity first, one object per
    observation and one observation per object. Frame analysis (O7) gives its detections the ids
    of the map objects they match."""
    pairs = []
    for i, (label, pts, c) in enumerate(obs):
        for o in objects:
            a = affinity(label, pts, c, o, bool(cut[i]) if cut is not None else False)
            if a > 0:
                pairs.append((-a, i, o.id))
    out: list[int | None] = [None] * len(obs)
    used: set[int] = set()
    for _, i, oid in sorted(pairs):
        if out[i] is None and oid not in used:
            out[i] = oid
            used.add(oid)
    return out


# -- observations of a keyframe ------------------------------------------------------------------


def robust_rgb(rgb: NDArray[np.uint8], mask: NDArray[np.bool_]) -> list[float] | None:
    """The robust mean sRGB of a mask's pixels: inside the mask eroded by ``RGB_ERODE_PX``,
    the pixels between the ``RGB_TRIM`` luminance percentiles."""
    inner = ndimage.binary_erosion(mask, iterations=RGB_ERODE_PX)
    if inner.sum() < 10:
        inner = mask
    px = rgb[inner].astype(np.float64)
    if not len(px):
        return None
    lum = px @ np.array([0.2126, 0.7152, 0.0722])
    lo, hi = np.percentile(lum, RGB_TRIM)
    keep = (lum >= lo) & (lum <= hi)
    return [round(float(v), 2) for v in (px[keep] if keep.any() else px).mean(0)]


def _thin(points: Array, cap: int = SIGHTING_POINTS) -> Array:
    if not len(points):
        return points
    q = voxel_keys(points, POINT_VOXEL_M)
    _, first = np.unique(q, axis=0, return_index=True)
    pts = points[np.sort(first)]
    if len(pts) > cap:
        pts = pts[np.linspace(0, len(pts) - 1, cap).astype(np.intp)]
    return pts


def observe(view: View, version: int, floor_z: float | None,
            floor_sigma: float = FLOOR_SIGMA_DEFAULT_M) -> list[Observation]:
    """The lifted detections of a keyframe (see the module docstring). A detection naming an
    opening (``detect.PASSAGE_LABELS``) is no object; a floor-standing one that lies within the
    floor band (``core.constants.floor_band_m`` of ``floor_sigma``) of the floor ``floor_z`` is
    the floor (map.md §5)."""
    if not view.dets:
        return []
    depth = np.where(np.isfinite(view.depth), view.depth, 0.0).astype(np.float32)
    valid = depth > 0
    if view.ignore is not None:
        valid &= ~view.ignore
    pose = Pose.from_matrix(view.T)
    insts = [trim_support(i, depth, view.K, valid, pose)
             for i in lift_on_grid(view.dets, depth, view.K, pose, valid)]
    h, w = depth.shape
    out = []
    for inst in insts:
        pts = np.asarray(inst.lifted.points, np.float64)
        if len(pts) < 10 or inst.mask.mean() > MAX_MASK_SHARE:
            continue  # too few points, or the background (a tree, a sign over the whole view)
        if float(np.median(np.linalg.norm(pts - view.centre, axis=1))) < NEAR_CAMERA_M:
            continue  # on the camera (a gripper, a lens hood)
        if inst.detection.label in PASSAGE_LABELS:
            continue  # a doorway is a passage of the grid, not an object (map.md §5)
        if floor_z is not None and on_the_floor(inst, UP, floor_z, floor_band_m(floor_sigma)):
            continue
        rgb = robust_rgb(view.rgb, inst.mask) or [128.0, 128.0, 128.0]
        rows, cols = np.flatnonzero(inst.mask.any(1)), np.flatnonzero(inst.mask.any(0))
        cut = [bool(cols[0] < BORDER_PX), bool(cols[-1] >= w - BORDER_PX),
               bool(rows[0] < BORDER_PX), bool(rows[-1] >= h - BORDER_PX)]
        c = np.median(pts, axis=0)
        ray = c - view.centre
        dist = float(np.linalg.norm(ray))
        behind, mask_range = _behind(inst.mask, depth, valid, view, pts)
        s = Sighting(view.kid, version, c.round(5).tolist(), round(dist, 4),
                     int(inst.mask.sum()), cut, (ray / max(dist, 1e-9)).round(5).tolist(),
                     view.centre.round(5).tolist(), float(view.sigma_pos_m), rgb,
                     inst.detection.label, round(float(inst.detection.score), 4),
                     view.depth_rel, round(behind, 4), round(mask_range, 4))
        out.append(Observation(inst.detection.label, float(inst.detection.score), pts,
                               inst.mask, s, dict(inst.detection.alternatives)))
    return out


def _behind(mask: NDArray[np.bool_], depth: NDArray[np.float32], valid: NDArray[np.bool_],
            view: View, pts: Array) -> tuple[float, float]:
    """(share of the inner mask's depth beyond the object's own points by more than ``tau``,
    median range of the inner mask's depth): what a detection sees through (a glass, a window,
    trees behind one)."""
    inner = ndimage.binary_erosion(mask, iterations=RGB_ERODE_PX) & valid
    if inner.sum() < 10:
        inner = mask & valid
    vv, uu = np.nonzero(inner)
    r_mask = np.linalg.norm(unproject_pixels(uu, vv, depth[vv, uu], view.K.K()), axis=1)
    far = float(np.percentile(np.linalg.norm(pts - view.centre, axis=1), 90))
    tau = max(SEEN_TAU_M, SEEN_TAU_REL * far)
    return float(np.mean(r_mask > far + tau)), float(np.median(r_mask))


def _detected_there(view: View, u: Array, v: Array, px: float) -> bool:
    """Whether one of the keyframe's detections covers ``COVERED_SHARE`` of an object's samples
    (their pixels ``u``, ``v``) with a mask at most ``COVER_SIZE`` times the pixels the object
    covers (``px``): the keyframe sees an object there, whatever its label or depth."""
    if not len(u):
        return False
    uu, vv = u.astype(np.intp), v.astype(np.intp)
    for det in view.dets:
        if det.mask[vv, uu].mean() >= COVERED_SHARE and det.mask.sum() <= COVER_SIZE * max(px, 1):
            return True
    return False


def see_through(o: MapObject) -> bool:
    """Whether the depth sees through the object (see the module docstring): a keyframe detected
    it after a keyframe's depth had seen it gone (``contradicted``), or several of its
    sightings, at least ``SEE_THROUGH_VIEWS`` and a third of them, see the depth inside its mask
    beyond its own points (``BEHIND_SHARE`` of the mask), or the mask's median depth beyond its
    box by more than half its size."""
    if o.contradicted:
        return True
    if o.box is None:
        return False
    half = 0.5 * float(np.max(o.box.size))
    n = 0
    for s in o.sightings:
        box_range = float(np.linalg.norm(o.box.centre - np.asarray(s.cam)))
        tau = max(SEEN_TAU_M, SEEN_TAU_REL * box_range)
        if s.behind >= BEHIND_SHARE or s.mask_range > box_range + max(tau, half):
            n += 1
    return n >= SEE_THROUGH_VIEWS and 3 * n >= len(o.sightings)


def judge(o: MapObject, view: View) -> tuple[str, float] | None:
    """A keyframe's verdict on an object it did not detect: ("gone" | "in_place", share seen
    through), or None when it does not judge it (out of view, too far, hidden). An object the
    depth sees through (``see_through``) is judged by its detection alone: "gone" at
    ``ABSENT_SHARE`` (so that one keyframe never strikes it) when the keyframe has its place in
    view, unhidden, at a size the detector sees (``DETECT_MIN_PX``)."""
    if o.box is None or not len(o.points):
        return None
    pts = o.points[o.box.contains(o.points, 0.01)]
    if len(pts) < JUDGE_MIN:
        return None
    if len(pts) > JUDGE_SAMPLES:
        pts = pts[np.linspace(0, len(pts) - 1, JUDGE_SAMPLES).astype(np.intp)]
    u, v, z = view.pixels(pts)
    h, w = view.depth.shape
    m = IN_VIEW_MARGIN_PX
    inside = (u >= m) & (u < w - m) & (v >= m) & (v < h - m)
    if inside.mean() < IN_VIEW_SHARE or float(np.median(z[inside])) > JUDGE_MAX_DEPTH_M:
        return None
    foot = pts[:, 2] < o.box.bottom + FOOT_SHARE * o.box.size[2]
    sel = inside & ~foot
    if view.ignore is not None:
        sel[sel] &= ~view.ignore[v[sel].astype(np.intp), u[sel].astype(np.intp)]
    d = view.depth[v[sel].astype(np.intp), u[sel].astype(np.intp)].astype(np.float64)
    zs = z[sel]
    has = d > 0
    d, zs = d[has], zs[has]
    tau = np.maximum(SEEN_TAU_M, SEEN_TAU_REL * zs)
    z_med = float(np.median(z[inside]))
    px = float(np.median([s.pixels * (s.distance / z_med) ** 2 for s in o.sightings])) \
        if o.sightings else 0.0
    if _detected_there(view, u[inside], v[inside], px):
        return None  # the keyframe detects something of its size there (another label)
    if see_through(o):
        hidden = float(np.mean(d < zs - tau)) if len(zs) else 1.0
        if hidden >= 0.5 or px < DETECT_MIN_PX:
            return None
        return "gone", ABSENT_SHARE
    through = int(np.sum(d > zs + tau))
    place = int(np.sum(np.abs(d - zs) <= tau))
    judged = through + place
    if judged < JUDGE_MIN or judged < 0.5 * len(zs):
        return None
    share = through / judged
    if share >= GONE_SHARE:
        return "gone", share
    if share <= PLACE_SHARE:
        return "in_place", share
    return None


# -- surfaces -------------------------------------------------------------------------------------


@dataclass
class Surface:
    height_m: float
    polygon: list[list[float]]
    area_m2: float
    centre: list[float]
    bbox: list[float]  # x0, y0, x1, y1
    id: int = 0

    def contains(self, xy: Array) -> bool:
        """Whether a point's x, y lies within the outline (a cell's width of margin)."""
        poly = np.asarray(self.polygon, np.float32).reshape(-1, 1, 2)
        d = cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), True)
        return bool(d >= -SURFACE_CELL_M)

    def body(self) -> dict[str, Any]:
        return {"id": self.id, "height_m": round(self.height_m, 4), "polygon_m": self.polygon,
                "area_m2": round(self.area_m2, 4)}


def _polygon(mask: NDArray[np.bool_], x0: int, y0: int) -> list[list[float]]:
    """The outline of a cell mask (rows y, cols x from cell x0, y0) in map x, y."""
    m = np.pad(mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.7, True).reshape(-1, 2)
    if len(approx) < 3:  # a cell or a line of cells: its bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        approx = np.array([[x - 0.5, y - 0.5], [x + w - 0.5, y - 0.5],
                           [x + w - 0.5, y + h - 0.5], [x - 0.5, y + h - 0.5]])
    pts = (np.asarray(approx, np.float64) - 1 + 0.5 + np.array([x0, y0])) * SURFACE_CELL_M
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]


def find_surfaces(xyz: NDArray[Any], top_z: float | None = None,
                  viewpoints: NDArray[Any] | None = None) -> list[Surface]:
    """Horizontal support planes of a cloud (map frame, z up; see the module docstring), lowest
    first; none above ``top_z``. With ``viewpoints`` (the cameras), only the points whose normal,
    turned towards the cameras, points up (``UP_NORMAL_Z``) take part: a plane's top, not the
    sides of what stands on it, nor walls, nor a ceiling."""
    p = np.asarray(xyz, np.float64)
    if viewpoints is not None and len(p) >= SURFACE_MIN_POINTS:
        from oh_my_slam.reconstruction.pointcloud import PointNormals

        n = PointNormals(p, viewpoints).at(np.arange(len(p)))
        p = p[n[:, 2] >= UP_NORMAL_Z]
    if len(p) < SURFACE_MIN_POINTS:
        return []
    cxy = voxel_keys(p[:, :2], SURFACE_CELL_M)
    iz = voxel_keys(p[:, 2], SURFACE_BIN_M)
    lo = cxy.min(0)
    cxy -= lo
    iz0 = int(iz.min())
    iz -= iz0
    W = int(cxy[:, 0].max()) + 1
    NZ = int(iz.max()) + 1 + 2 * BAND_BINS
    izp = iz + BAND_BINS
    cell = cxy[:, 1] * W + cxy[:, 0]
    key = cell * NZ + izp
    uk, cnt = np.unique(key, return_counts=True)
    cs = np.concatenate([[0], np.cumsum(cnt)])

    def window(r: int) -> NDArray[np.int64]:
        a = np.searchsorted(uk, uk - r, "left")
        b = np.searchsorted(uk, uk + r, "right")
        out: NDArray[np.int64] = cs[b] - cs[a]
        return out

    slab, band = window(SLAB_BINS), window(BAND_BINS)
    flat = (slab >= 2) & (slab >= FLAT_SHARE * band)
    f_cell, f_z = uk[flat] // NZ, uk[flat] % NZ
    if not len(f_cell):
        return []
    hist = np.bincount(f_z, minlength=NZ).astype(np.float64)
    min_cells = SURFACE_MIN_AREA_M2 / SURFACE_CELL_M ** 2
    peak = ndimage.maximum_filter1d(hist, 2 * SURFACE_PEAK_BINS + 1)
    zs: list[int] = []
    for b in sorted(np.flatnonzero((hist >= peak) & (hist >= min_cells)).tolist(),
                    key=lambda b: (-hist[b], b)):
        if all(abs(b - c) > 2 * SLAB_BINS for c in zs):  # a plateau: its strongest bin
            zs.append(b)
    out: list[Surface] = []
    for b in zs:
        cells = np.unique(f_cell[np.abs(f_z - b) <= SLAB_BINS])
        cx, cy = cells % W, cells // W
        x0, y0 = int(cx.min()), int(cy.min())
        grid = np.zeros((int(cy.max()) - y0 + 1, int(cx.max()) - x0 + 1), bool)
        grid[cy - y0, cx - x0] = True
        closed = ndimage.binary_closing(np.pad(grid, 1), np.ones((3, 3), bool))[1:-1, 1:-1]
        lab, n = ndimage.label(closed | grid, np.ones((3, 3), bool))
        core = ndimage.binary_erosion(closed, np.ones((3, 3), bool))
        for k in range(1, n + 1):
            comp = lab == k
            seen = comp & grid
            # a line of cells (a wall's points) is no plane: it has no core
            if seen.sum() < min_cells or (core & comp).sum() < CORE_SHARE * min_cells:
                continue
            rr, cc = np.nonzero(seen)
            comp_cells = (rr + y0) * W + (cc + x0)
            pt = np.isin(cell, comp_cells) & (np.abs(izp - b) <= SLAB_BINS)
            height = float(np.median(p[pt, 2]))
            gx, gy = cc + x0 + lo[0], rr + y0 + lo[1]
            out.append(Surface(height, _polygon(comp, x0 + int(lo[0]), y0 + int(lo[1])),
                               float(comp.sum()) * SURFACE_CELL_M ** 2,
                               [round(float(np.mean(p[pt, 0])), 4),
                                round(float(np.mean(p[pt, 1])), 4)],
                               [float(gx.min()) * SURFACE_CELL_M,
                                float(gy.min()) * SURFACE_CELL_M,
                                float(gx.max() + 1) * SURFACE_CELL_M,
                                float(gy.max() + 1) * SURFACE_CELL_M]))
    out = [s for s in out if top_z is None or s.height_m <= top_z]
    return sorted(out, key=lambda s: (s.height_m, s.centre))


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assign_surface_ids(surfaces: list[Surface], known: list[dict[str, Any]], next_id: int
                       ) -> tuple[list[dict[str, Any]], int]:
    """Stable surface ids: a surface takes the id of the known one nearest in height (within
    ``SURFACE_ID_DZ``) whose extent overlaps it, else the next id; returns the registry."""
    cands = []
    for i, s in enumerate(surfaces):
        for e in known:
            dz = abs(s.height_m - e["h"])
            if dz > SURFACE_ID_DZ:
                continue
            near = math.hypot(s.centre[0] - e["c"][0], s.centre[1] - e["c"][1])
            if _iou(s.bbox, e["bbox"]) >= SURFACE_ID_IOU or near <= SURFACE_ID_CENTRE_M:
                cands.append((dz, near, i, e["id"]))
    used: set[int] = set()
    for _, _, i, eid in sorted(cands):
        if surfaces[i].id or eid in used:
            continue
        surfaces[i].id = eid
        used.add(eid)
    reg = {e["id"]: e for e in known}
    for s in surfaces:
        if not s.id:
            s.id = next_id
            next_id += 1
        reg[s.id] = {"id": s.id, "h": round(s.height_m, 4), "c": s.centre, "bbox": s.bbox}
    return sorted(reg.values(), key=lambda e: e["id"]), next_id


# -- the objects of a map ------------------------------------------------------------------------


@dataclass
class Struck:
    """An object struck by an update and the keyframes that saw it gone (cloud carving)."""

    box: Box
    witnesses: list[View]


class ObjectMap:
    """The objects of an anchored map in its mapper folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects: list[MapObject] = []
        self.next_id = 1
        self.surface_ids: list[dict[str, Any]] = []
        self.next_surface_id = 1
        self.surfaces: list[Surface] = []
        self.version = 0
        self.floor_z: float | None = None  # the floor of the latest update (box grounding)
        self.floor_sigma = FLOOR_SIGMA_DEFAULT_M  # its height sigma (the floor band)
        # the image-confirmed keyframes (id, T_map_cam, intrinsics) the votes count views over
        # (``detection_votes``), set by ``finish``
        self.cameras: list[tuple[str, Array, Intrinsics, NDArray[np.bool_] | None]] = []

    @staticmethod
    def load(root: Path) -> ObjectMap:
        m = ObjectMap(root)
        p = root / OBJECTS_JSON
        if not p.exists():
            return m
        d = json.loads(p.read_text())
        m.objects = [MapObject.from_json(o) for o in d["objects"]]
        m.next_id = int(d["next_id"])
        m.surface_ids = list(d["surface_ids"])
        m.next_surface_id = int(d["next_surface_id"])
        m.version = int(d["version"])
        m.surfaces = [Surface(s["height_m"], s["polygon_m"], s["area_m2"], s["centre"],
                              s["bbox"], s["id"]) for s in d["surfaces"]]
        z = root / OBJECTS_NPZ
        if z.exists():
            with np.load(z) as arrays:
                for o in m.objects:
                    o.points, o.source = arrays[f"p{o.id}"], arrays[f"s{o.id}"]
        return m

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        for o in self.objects:
            arrays[f"p{o.id}"] = o.points.astype(np.float32)
            arrays[f"s{o.id}"] = o.source.astype(np.int32)
        tmp = self.root / "objects.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(self.root / OBJECTS_NPZ)
        atomic_write_json(self.root / OBJECTS_JSON, {
            "objects": [o.to_json() for o in self.objects], "next_id": self.next_id,
            "surface_ids": self.surface_ids, "next_surface_id": self.next_surface_id,
            "version": self.version,
            "surfaces": [{**s.body(), "centre": s.centre, "bbox": s.bbox}
                         for s in self.surfaces]})

    def by_id(self) -> dict[int, MapObject]:
        return {o.id: o for o in self.objects}

    def live(self) -> list[MapObject]:
        """The listed objects that are not struck."""
        return [o for o in self.objects if listed(o) and o.state != "struck"]

    # -- an update ------------------------------------------------------------------------------

    def update(self, views: list[View], version: int, floor_z: float | None,
               floor_sigma: float = FLOOR_SIGMA_DEFAULT_M) -> list[Struck]:
        """Fold the new keyframes ``views`` (request order) into the objects of ``version``, on
        the version's floor estimate (``floor_z``, ``floor_sigma``: map.md §5); returns the
        objects struck, for carving."""
        self.version = version
        self.floor_z, self.floor_sigma = floor_z, floor_sigma
        self.objects = [o for o in self.objects if o.state != "struck"]
        events: dict[int, list[tuple[str, float, View]]] = {}
        created: set[int] = set()
        for view in views:
            obs = observe(view, version, floor_z, floor_sigma)
            cands = [o for o in self.objects if o.box is not None]
            ids = match([(ob.label, ob.points, ob.centre) for ob in obs], cands,
                        [ob.sighting.truncated for ob in obs])
            by_id = self.by_id()
            seen: set[int] = set()
            for ob, oid in zip(obs, ids, strict=True):
                if oid is None:
                    o = MapObject(self.next_id, {}, first_seen_version=version)
                    self.next_id += 1
                    self.objects.append(o)
                    created.add(o.id)
                else:
                    o = by_id[oid]
                self._add(o, ob, version, floor_z)
                if o.last_verdict == "gone":
                    o.contradicted += 1  # its depth said gone, the detector sees it
                o.last_verdict = "seen"
                events.setdefault(o.id, []).append(("seen", 1.0, view))
                seen.add(o.id)
            for o in cands:
                if o.id not in seen:
                    verdict = judge(o, view)
                    if verdict is not None:
                        o.last_verdict = verdict[0]
                        events.setdefault(o.id, []).append((*verdict, view))
        struck: list[tuple[MapObject, list[View]]] = []
        dropped: set[int] = set()
        for o in self.objects:
            ev = events.get(o.id, [])
            log.debug("object %d %s: %s", o.id, o.label,
                      [(e[0], round(e[1], 2), e[2].kid) for e in ev])
            last = max((i for i, e in enumerate(ev) if e[0] == "seen"), default=-1)
            gone = [e for e in ev[last + 1:] if e[0] == "gone"]
            place = sum(e[0] == "in_place" for e in ev[last + 1:])
            o.misses += place
            # one keyframe alone strikes only with a depth image: monocular depth sees past
            # glass and foliage (a tree behind a window, a window pane) from one view or another
            alone = len(gone) == 1 and gone[0][1] >= GONE_ALONE and \
                gone[0][2].depth_rel <= SENSOR_DEPTH_REL
            if gone and len(gone) > place and (len(gone) >= 2 or alone):
                if not o.published and len(o.sightings) < 2:
                    dropped.add(o.id)  # a single detection seen gone was never there
                else:
                    o.published = True  # seen twice, then gone: listed once, as struck
                    struck.append((o, [e[2] for e in gone]))
                continue
            if last >= 0 and o.id not in created:
                o.state, o.moved_from = "confirmed", None  # seen again where it is
            # one sighting lists an object only whole in the image (no fragment at its border)
            # and trusted (TRUSTED_SCORE); below that, two keyframes must detect it
            if not o.published and last >= 0 and (len(o.sightings) >= 2 or (
                    o.misses == 0 and not o.sightings[0].truncated
                    and o.sightings[0].score >= TRUSTED_SCORE)):
                o.published = True
        self.objects = [o for o in self.objects if o.id not in dropped]
        self._moves([o for o, _ in struck], created, version)
        out = []
        for o, witnesses in struck:
            if o.state != "moved":
                assert o.box is not None
                o.state, o.struck_version = "struck", version
                out.append(Struck(o.box, witnesses))
        return out

    def _add(self, o: MapObject, ob: Observation, version: int, floor_z: float | None) -> None:
        # a detection cut by the image border votes for its label with CUT_VOTE of its score:
        # the label of a whole view wins
        cut = CUT_VOTE if ob.sighting.truncated else 1.0
        o.votes[ob.label] = round(o.votes.get(ob.label, 0.0) + ob.score * cut, 4)
        for alt, v in ob.alternatives.items():
            o.others[alt] = round(o.others.get(alt, 0.0) + v * cut, 4)
        o.sightings.append(ob.sighting)
        k = len(o.sightings) - 1
        pts = _thin(ob.points).astype(np.float32)
        o.points = np.concatenate([o.points, pts])
        o.source = np.concatenate([o.source, np.full(len(pts), k, np.int32)])
        o.last_seen_version = version
        # only the largest sightings keep their points
        big = sorted(range(len(o.sightings)), key=lambda i: -o.sightings[i].pixels)
        keep = set(big[:SHAPING_SIGHTINGS])
        sel = np.isin(o.source, list(keep))
        o.points, o.source = o.points[sel], o.source[sel]
        self._fit(o, floor_z)  # the next keyframe of the update matches the box as it is now

    def _moves(self, struck: list[MapObject], created: set[int], version: int) -> None:
        """A new object like one struck in this update (label, size, colour) is that object,
        moved: it takes its id."""
        fresh = [o for o in self.objects if o.id in created and o.box is not None]
        used: set[int] = set()
        for old in sorted(struck, key=lambda o: o.id):
            if old.box is None:
                continue
            best = None
            for n in fresh:
                if n.id in used or n.box is None or not alike(n.label, old.label):
                    continue
                if float(np.linalg.norm(n.box.centre - old.box.centre)) < MOVE_MIN_M:
                    continue
                a, b = np.sort(n.box.size), np.sort(old.box.size)
                if np.any(np.maximum(a, b) > MOVE_SIZE * np.maximum(np.minimum(a, b), 0.02)):
                    continue
                dc = float(np.linalg.norm(np.subtract(mean_rgb(n), mean_rgb(old))))
                if dc > MOVE_RGB:
                    continue
                if best is None or dc < best[0]:
                    best = (dc, n)
            if best is None:
                continue
            n = best[1]
            used.add(n.id)
            old.moved_from = old.box.centre.round(4).tolist()
            old.state = "moved"
            old.sightings, old.points, old.source = n.sightings, n.points, n.source
            old.box = n.box
            old.votes = {k: old.votes.get(k, 0.0) + v for k, v in n.votes.items()}
            old.others = {k: old.others.get(k, 0.0) + n.others.get(k, 0.0)
                          for k in sorted(set(old.others) | set(n.others))}
            old.last_seen_version = version
            old.misses = 0
            self.objects.remove(n)

    def _fit(self, o: MapObject, floor_z: float | None) -> None:
        """The box of the sightings that agree on the centre."""
        idx = sorted(set(o.source.tolist()))
        if not idx:
            return
        C = np.array([o.sightings[i].centre for i in idx])
        med = np.median(C, axis=0)
        dev = np.linalg.norm(C - med, axis=1)
        tol = max(AGREE_MIN_M, AGREE_REL * float(np.median(dev)))
        agree = [i for i, d in zip(idx, dev, strict=True) if d <= tol]
        pts = o.points[np.isin(o.source, agree)]
        o.box = Box.of(fit_object_obb(pts, o.label, UP, floor_z))

    def carve(self, struck: list[Struck], xyz: Array) -> NDArray[np.bool_]:
        """The cloud points (``xyz``, map frame) to drop: inside a struck object's box (grown by
        ``CARVE_GROW_M``) and seen through by one of its witnesses."""
        drop = np.zeros(len(xyz), bool)
        for s in struck:
            inside = np.flatnonzero(s.box.contains(xyz, CARVE_GROW_M))
            if not len(inside):
                continue
            for view in s.witnesses:
                uf, vf, z = view.pixels(xyz[inside])
                h, w = view.depth.shape
                u, v = np.round(uf).astype(np.intp), np.round(vf).astype(np.intp)
                ok = (uf >= 0) & (u >= 0) & (u < w) & (vf >= 0) & (v >= 0) & (v < h)
                d = np.zeros(len(z))
                d[ok] = view.depth[v[ok], u[ok]]
                tau = np.maximum(SEEN_TAU_M, SEEN_TAU_REL * z)
                drop[inside[ok & (d > z + tau)]] = True
        return drop

    # -- what the cloud says ------------------------------------------------------------------

    def finish(self, xyz: Array, top_z: float | None, version: int,
               viewpoints: Array | None = None, floor_z: float | None = None,
               cameras: list[tuple[str, Array, Intrinsics, NDArray[np.bool_] | None]] | None
               = None) -> None:
        """Surfaces from the fused cloud (the points facing up, seen from ``viewpoints``), then
        every listed object's relations, on its box as fitted (it rests on its support anew).
        ``cameras``: the map's image-confirmed keyframes, which ``detection_votes`` counts."""
        self.cameras = list(cameras or [])
        self.surfaces = find_surfaces(xyz, top_z, viewpoints)
        self.surface_ids, self.next_surface_id = assign_surface_ids(
            self.surfaces, self.surface_ids, self.next_surface_id)
        for o in self.objects:
            if o.label in FLOOR_COVERINGS and o.published and o.box is not None and \
                    not any(s.area_m2 >= ROOM_FLOOR_M2 and abs(o.box.bottom - s.height_m)
                            <= SUPPORT_TOL_M for s in self.surfaces):
                o.published = False  # a rug on a worktop: a false detection
        live = self.live()
        for o in live:
            self._fit(o, self.floor_z if floor_z is None else floor_z)
        for o in self.objects:  # views it lies mostly masked in: no view of it (map.md §5)
            o.masked = masked_by(o, self.cameras)
        for o in live:
            self._relate(o, live, version)
        for o in live:  # on its box as it rests on its support
            o.masked = masked_by(o, self.cameras)

    def _relate(self, o: MapObject, live: list[MapObject], version: int) -> None:
        assert o.box is not None
        box = o.box
        observed = o.last_seen_version == version
        inside: MapObject | None = None
        least = math.inf
        for p in live:
            if p is o or p.box is None or p.label not in CONTAINERS:
                continue
            vol = float(np.prod(p.box.size))
            if vol > np.prod(box.size) and p.box.contains(box.centre[None], 0.01)[0] \
                    and vol < least:
                inside, least = p, vol
        support: tuple[str, int, float] | None = None
        if inside is not None:
            support = ("object", inside.id, 0.0)
        else:
            # an object whose height no sighting saw (seen from above) may stand on a support
            # as far below its visible top part as SUPPORT_REACH_M
            # (its height seen from the side: a support it rests on is not evidence here)
            reach = SUPPORT_TOL_M if extent_seen(o, support=False)["z"] == "full" \
                else SUPPORT_REACH_M
            best_s = None
            for s in sorted(self.surfaces, key=lambda s: -s.area_m2):
                gap = box.bottom - s.height_m
                if -SUPPORT_TOL_M <= gap <= reach and s.contains(box.centre) and (
                        best_s is None or abs(gap) < abs(best_s[1]) - SUPPORT_SAME_M):
                    best_s = (s, gap)  # the nearest in height, the largest when as near
            best_o = None
            area = float(box.size[0] * box.size[1])
            for p in live:
                if p is o or p.box is None or p.box.top > box.top or \
                        p.label in FLOOR_COVERINGS or \
                        float(p.box.size[0] * p.box.size[1]) < SUPPORT_AREA_RATIO * area:
                    continue
                gap = box.bottom - p.box.top
                if -SUPPORT_TOL_M <= gap <= reach and p.box.footprint_contains(box.centre) and (
                        best_o is None or abs(gap) < abs(best_o[1])):
                    best_o = (p, gap)
            if best_o is not None and (best_s is None or (
                    abs(best_o[1]) + SUPPORT_PREFER_OBJECT_M < abs(best_s[1])
                    and not surface_label(best_o[0].label))):
                support = ("object", best_o[0].id, best_o[1])
            elif best_s is not None:
                support = ("surface", best_s[0].id, best_s[1])
        prev_sup, prev_in = o.supported_by, o.inside_of
        sup_obs = observed
        if support is not None:
            kind, sid, gap = support
            if kind == "object":
                q = next(p for p in live if p.id == sid)
                sup_obs = sup_obs or q.last_seen_version == version
            else:
                sup_obs = True  # the surfaces are measured again by every update
            if gap != 0.0:
                # the box rests on its support (a top-down view sees only its top)
                level = next(s.height_m for s in self.surfaces if s.id == sid) \
                    if kind == "surface" else next(p for p in live if p.id == sid).box.top
                top = max(box.top, level + POINT_VOXEL_M)
                box.centre[2] = (top + level) / 2
                box.size[2] = top - level
            same = prev_sup is not None and prev_sup["kind"] == kind and prev_sup["id"] == sid
            o.supported_by = {"kind": kind, "id": sid,
                              "verified_version": version if sup_obs or not same
                              else prev_sup["verified_version"]}  # type: ignore[index]
        else:
            o.supported_by = None
        if inside is not None:
            same = prev_in is not None and prev_in["id"] == inside.id
            o.inside_of = {"id": inside.id, "verified_version": version if observed or not same
                           else prev_in["verified_version"]}  # type: ignore[index]
        else:
            o.inside_of = None

    # -- records ---------------------------------------------------------------------------------

    def records(self) -> dict[str, Any]:
        """The ``Objects`` content of this version, without ``region`` (the grid's) and
        ``color`` (the display colour of the id, segmentation's): the service sets them. Every
        listed object and the surfaces, map frame."""
        out = []
        for o in sorted(self.objects, key=lambda o: o.id):
            if not listed(o):
                continue
            out.append({
                "id": o.id, "label": o.label, "detected_as": o.detected_as(),
                "score": round(o.score, 4), "centre_m": o.box.centre.round(4).tolist(),
                "size_m": o.box.size.round(4).tolist(), "yaw_deg": yaw_deg(o.box.yaw_deg),
                "sigma_centre_m": round(sigma_centre(o), 4), "extent_seen": extent_seen(o),
                "mean_rgb": [round(v) for v in mean_rgb(o)],
                "detection_votes": detection_votes(o, self.cameras),
                "state": o.state,
                "moved_from_m": o.moved_from, "first_seen_version": o.first_seen_version,
                "last_seen_version": o.last_seen_version,
                "supported_by": o.supported_by, "inside_of": o.inside_of,
                "seen_from": sorted({(round(s.cam[0], 3), round(s.cam[1], 3))
                                     for s in o.sightings})})
        return {"objects": out, "surfaces": [s.body() for s in self.surfaces]}


# -- per-object summaries ------------------------------------------------------------------------


def listed(o: MapObject) -> bool:
    """Whether the objects read lists an object: published, with a box of at least
    ``MIN_OBJECT_M`` on its longest side (a few pixels of a mask lifted are no object); a door
    only as a leaf (``detect.door_leaf``, map.md §5)."""
    return o.published and o.box is not None and float(np.max(o.box.size)) >= MIN_OBJECT_M \
        and (o.label != DOOR_LABEL or door_leaf(o.box.size))


def in_view(X: Array, T: Array, K: Intrinsics, ignore: NDArray[np.bool_] | None = None
            ) -> bool:
    """Whether the camera ``T`` (``T_map_cam``) with intrinsics ``K`` has the map point ``X``
    in its view frustum: in front of it and inside its image, not on its ignored pixels
    (``ignore``, any grid of the image: map.md §2, they give no vote)."""
    pc = (np.asarray(X, np.float64) - T[:3, 3]) @ T[:3, :3]
    if pc[2] <= VIEW_NEAR_M:
        return False
    u, v = K.fx * pc[0] / pc[2] + K.cx, K.fy * pc[1] / pc[2] + K.cy
    if not (0.0 <= u < K.width and 0.0 <= v < K.height):
        return False
    if ignore is None:
        return True
    h, w = ignore.shape[:2]
    return not bool(ignore[min(int(v * h / K.height), h - 1), min(int(u * w / K.width), w - 1)])


def unignored(dets: list[Detection], ignore: NDArray[np.bool_] | None) -> list[Detection]:
    """The detections of a view (masks on the grid of ``ignore``) without those lying mostly
    on its ignored pixels: more than ``IGNORED_DETECTION_MAX_SHARE`` of the mask (of the box
    without one) inside them (map.md §2 "Ignored pixels"). The rest lose those pixels when they
    are lifted (``observe``: ``valid``)."""
    if ignore is None:
        return dets
    out = []
    for d in dets:
        area = d.mask if d.mask.shape == ignore.shape and d.mask.any() \
            else _box_mask(d.box, ignore.shape)
        n = int(area.sum())
        if n and float((area & ignore).sum()) / n > IGNORED_DETECTION_MAX_SHARE:
            continue
        out.append(d)
    return out


def _box_mask(box: tuple[float, float, float, float], shape: tuple[int, ...]
              ) -> NDArray[np.bool_]:
    m = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = (round(c) for c in box)
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


def detection_votes(o: MapObject,
                    cameras: list[tuple[str, Array, Intrinsics, NDArray[np.bool_] | None]]
                    ) -> list[dict[str, Any]]:
    """``detection_votes`` (map.md §5): per label, the keyframes that detected the object as it
    over the image-confirmed keyframes whose view frustum holds its box centre, detected or not
    (``cameras``; its sightings come from confirmed keyframes only, and every keyframe that
    detected it counts as seeing it), highest share first."""
    masked = masked_by(o, cameras)  # no view of it: neither a vote nor its denominator
    frames: dict[str, set[str]] = {}
    for s in o.sightings:
        if s.kid not in masked:
            frames.setdefault(s.label, set()).add(s.kid)
    seen = set().union(*frames.values()) if frames else set()
    if o.box is not None:
        seen |= {kid for kid, T, K, ign in cameras
                 if kid not in masked and in_view(o.box.centre, T, K, ign)}
    n = max(len(seen), 1)
    out = [{"label": k, "share": round(min(len(v) / n, 1.0), 4)} for k, v in frames.items()]
    return sorted(out, key=lambda e: (-e["share"], e["label"]))


def relief(o: MapObject) -> str | None:
    """``"open"``, ``"closed"`` or None (unknown) for the top of an object seen from above (see
    ``RELIEF_MAX_M``): its points inside the footprint against those at its rim."""
    if o.box is None or float(np.max(o.box.size[:2])) > RELIEF_MAX_M or \
            sum(s.ray[2] <= -RELIEF_DOWN for s in o.sightings) < RELIEF_VIEWS:
        return None
    local = (np.asarray(o.points, np.float64) - o.box.centre) @ o.box.R
    half = np.maximum(o.box.size[:2] / 2, 1e-6)
    r = np.hypot(local[:, 0] / half[0], local[:, 1] / half[1])
    inner, rim = local[r < RELIEF_INNER, 2], local[(r >= RELIEF_RIM) & (r <= 1.0), 2]
    if len(inner) < RELIEF_MIN_POINTS or len(rim) < RELIEF_MIN_POINTS:
        return None
    step = max(RELIEF_M, RELIEF_SHARE * float(o.box.size[2]))
    top = float(np.percentile(rim, 90))
    if float(np.median(inner)) < top - step:
        return "open"
    if float(np.median(inner)) > float(np.median(rim)) + step:
        return "closed"
    return None


def mean_rgb(o: MapObject) -> list[float]:
    """The colour of the ``RGB_VIEWS`` sightings that saw the object best (uncut first, then by
    pixels): their pixel-weighted mean."""
    best = sorted(o.sightings, key=lambda s: (s.truncated, -s.pixels))[:RGB_VIEWS]
    if not best:
        return [128.0, 128.0, 128.0]
    w = np.array([s.pixels for s in best], np.float64)
    c = np.array([s.rgb for s in best], np.float64)
    out: list[float] = np.clip((c * w[:, None]).sum(0) / w.sum(), 0, 255).tolist()
    return out


def sigma_centre(o: MapObject) -> float:
    """One sigma of the centre: each sighting's pose sigma and depth noise, over their number;
    while a box axis is seen only in part, also the spread of the sightings' centres (each sees
    another part of it) over their number."""
    s = o.sightings
    if not s:
        return 0.0
    noise = float(np.sqrt(np.mean([x.sigma_pos_m ** 2 + (x.depth_rel * x.distance) ** 2
                                   for x in s]) / len(s)))
    spread = 0.0
    if len(s) >= 2 and "partial" in extent_seen(o).values():
        C = np.array([x.centre for x in s])
        spread = float(np.sqrt(np.sum((C - C.mean(0)) ** 2) / (len(s) - 1) / len(s)))
    return max(MIN_SIGMA_M, math.hypot(spread, noise))


def masked_by(o: MapObject,
              cameras: list[tuple[str, Array, Intrinsics, NDArray[np.bool_] | None]]
              ) -> set[str]:
    """The keyframes of ``cameras`` whose view is no view of the object (map.md §5 "Ignored
    pixels", ``core.images.masked_view``): its box projected there lies mostly on their ignored
    pixels. Empty without a box."""
    if o.box is None:
        return set()
    h = o.box.size / 2
    local = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1])
                      for z in (-h[2], h[2])])
    corners = local @ o.box.R.T + o.box.centre
    return {kid for kid, T, K, ign in cameras
            if masked_view((corners - T[:3, 3]) @ T[:3, :3], K, ign)}


def extent_seen(o: MapObject, support: bool = True) -> dict[str, str]:
    """``full`` or ``partial`` per box axis (see the module docstring), from its sightings in
    the views that are views of it (not ``MapObject.masked``; ``extent_from``), with the support
    it rests on (``supported_by``) bounding its bottom unless ``support`` is false."""
    if o.box is None:
        return {"x": "partial", "y": "partial", "z": "partial"}
    return extent_from(o.box.R, [(s.ray, s.cut) for s in o.sightings if s.kid not in o.masked],
                       support and o.supported_by is not None)


def extent_from(R: Array, views: list[tuple[Any, Any]], supported: bool = False
                ) -> dict[str, str]:
    """``extent_seen`` of a box (axes ``R``, columns x, y, z) from the views that bound it
    (map.md §5): each its unit ray from the camera to the box (map frame) and which image borders
    (left, right, top, bottom) cut its mask. An axis x or y is ``full`` when an uncut view sees it
    across (|cos| to the ray at most ``ACROSS_COS``) or views from both of its ends see it
    (``ENDS_COS``); ``z`` when a view whose mask neither the top nor the bottom border cuts sees
    it from the side (|ray z| at most ``ACROSS_COS``; a view the left or right border cuts still
    sees the top and the bottom; a view from above sees the top alone), or when ``supported`` (a
    support plane measured under its footprint, from any views, bounds the bottom) and a view
    that no border cuts sees its top from above (ray z below -``TOP_VIEW_COS``, map.md §5, P21).
    Every engine's views: the prior-first engine's sightings, the older mapper's recorded ones
    (``service.objects``)."""
    out = {"x": "partial", "y": "partial", "z": "partial"}
    whole = [np.asarray(r, np.float64) for r, cut in views if not any(cut)]
    rays = [np.asarray(r, np.float64) for r, _ in views]
    for k, name in ((0, "x"), (1, "y")):
        a = R[:, k]
        dots = [float(np.dot(r, a)) for r in rays]
        across = any(abs(float(np.dot(r, a))) <= ACROSS_COS for r in whole)
        ends = any(d > ENDS_COS for d in dots) and any(d < -ENDS_COS for d in dots)
        if across or ends:
            out[name] = "full"
    if any(not cut[2] and not cut[3] and abs(float(np.asarray(r)[2])) <= ACROSS_COS
           for r, cut in views):
        out["z"] = "full"
    if supported and any(float(np.asarray(r)[2]) < -TOP_VIEW_COS for r in whole):
        out["z"] = "full"
    return out


# -- an update's objects (map.md §5), orchestrated for the prior-first engine ---------------------

# the closer look at a keyframe looking down at its support (image, raised areas, detections,
# grid): its detections, more and with their other labels
Closer = Callable[[Path, list[Region], list[Detection], tuple[int, int]], list[Detection]]


def _on_grid(a: NDArray[Any], shape: tuple[int, int], interp: int) -> NDArray[Any]:
    if a.shape[:2] == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=interp)


def fuse_objects(am: AnchoredMap,
                 frames: list[tuple[Placement, NDArray[np.bool_] | None, Path | None,
                                    Future[list[Detection]]]],
                 version: int, closer: Closer,
                 again: list[tuple[Keyframe, Future[list[Detection]]]] | None = None
                 ) -> tuple[ObjectMap, int]:
    """The objects of the new image-confirmed keyframes (``frames``: each placement, its ignore
    mask, its image and its detections), fused in request order into the map's, then those of the stored keyframes this update re-registered
    (``again``, their stored image's detections): struck objects carved from the cloud, then
    the surfaces, the relations and each object's votes over the confirmed keyframes that see
    it. A frame kept at its prior gives no object (map.md §2), but for the too-close exception
    (``Keyframe.detects``: kept only for too_close, with a metric prior; api.md §4), whose
    detections count at its prior pose (on its depth image, else its monocular depth) and which
    counts as confirmed in the votes.
    ``closer`` looks closer at a keyframe looking down at its support
    (``segmentation.api.raised_regions``; the backend's ``look_closer``). Returns the objects and how many are listed."""
    om = ObjectMap.load(am.root)
    by_id = {k.id: k for k in am.keyframes}
    todo: list[tuple[Keyframe, NDArray[np.float32], NDArray[np.bool_] | None, Path | None,
                     Future[list[Detection]]]] = []
    for pl, own_ignore, path, fut in frames:
        kf = by_id.get(pl.keyframe or "")
        if kf is None or not kf.detects() or kf.grid is None or kf.K_grid is None:
            continue
        todo.append((kf, kf.grid, own_ignore, path, fut))
    # a re-registered keyframe has its stored image (``AnchoredMap.reregister`` set ``rgb``)
    todo += [(kf, ref.depth, None, am.image_path(kf.id), fut) for kf, fut in again or []
             for ref in [am._grid(kf)] if ref is not None]
    views = []
    for kf, grid, own_ignore, path, fut in todo:
        assert kf.rgb is not None and kf.K_grid is not None
        shape = grid.shape
        dets = []
        for d in fut.result():
            m = _on_grid(d.mask.astype(np.uint8), shape, cv2.INTER_NEAREST) > 0
            sx, sy = shape[1] / d.mask.shape[1], shape[0] / d.mask.shape[0]
            b = d.box
            dets.append(Detection(d.label, d.score, d.source, m,
                                  (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)))
        # the union of its camera's and its own mask (map.md §2 "Ignored pixels"), kept with
        # the keyframe: a re-registered stored keyframe's too
        ignore = on_grid(own_ignore if own_ignore is not None else am.ignore_of(kf.id), shape)
        dets = unignored(dets, ignore)
        rel = SENSOR_DEPTH_REL if kf.depth_kind == "sensor" else MONO_DEPTH_REL
        regions = raised_regions(grid, kf.K_grid, kf.T, rel, None if ignore is None else ~ignore)
        if regions:  # a view looking down at its support
            assert path is not None  # a frame kept has its image
            dets = unignored(closer(path, regions, dets, (shape[1], shape[0])), ignore)
        views.append(View(kf.id, kf.T, kf.sigma_pos_m, kf.K_grid, grid,
                          _on_grid(kf.rgb, shape, cv2.INTER_AREA), dets, ignore, rel))
    # the version's one floor estimate (map.md §5): the objects' and, through the version
    # record, the grid's (``run_anchored``: ``om.floor_z``, ``om.floor_sigma``)
    floor, floor_sigma = am.floor()
    struck = om.update(views, version, floor, floor_sigma)
    if struck:
        am.cloud.drop(om.carve(struck, am.cloud.points()))
    confirmed = am.confirmed()
    top = max((float(k.centre[2]) for k in confirmed), default=None)
    om.finish(am.cloud.points(), top, version,
              np.array([k.centre for k in confirmed]).reshape(-1, 3), floor,
              [(k.id, k.T, k.K, am.ignore_of(k.id)) for k in am.keyframes if k.detects()])
    return om, len(om.records()["objects"])
