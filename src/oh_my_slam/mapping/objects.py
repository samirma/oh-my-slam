"""Persistent objects: identity across keyframes and updates, evidence accumulation, OBB refits
(through the segmentation API), confirmation, merging and removal on evidence of absence.

Semantics (spec §2.3):

* **One update, one observation.** The order of an update's keyframes never matters. All of its
  instances are grouped at once (``_group``): strongest agreement first, never two instances of
  one keyframe in one group, and an object the map already held absorbs a group only when most of
  the group's instances match it directly. A group without such an object becomes a new object.
* **Evidence is order-free.** An object's evidence — label votes, scores, the keyframes that
  detected it, one ``Sighting`` per detection, its mean viewing distance and a canonical point set
  (``canonical_points``) — is a pure function of the instances it received, so its label and OBB
  do not depend on the order or the grouping into updates in which that evidence arrived.
* **Confirmation** is recomputed from all evidence (``confirm``): detections with a reliable mask
  (not mostly in the image-border band, where monocular depth is unreliable) in >= 2 distinct
  keyframes — or in the one keyframe that had the object in view, when no other keyframe of the
  map did (``views_in_frustum``; occlusion is ignored, so a detection whose depth puts it behind
  another surface cannot confirm itself). Unconfirmed objects are kept, never exported, so a later
  update can still confirm them.
* **Boxes** are fitted (by segmentation) to the points of the sightings that agree with each
  other (``fit_points``): monocular depth of small objects varies between keyframes, and the union
  of inconsistent sightings is a streak along the viewing rays, not the object.
* **Latest wins across updates.** An update whose keyframes, as a whole, see through an object
  removes it or gives it a strike; an update that re-detects it or sees it in place clears its
  strikes (``_absence``). Keyframes of the same update never remove each other's objects.
* **Ids** come from ``next_object_id`` and are never reused. It counts the map's detections: the
  detections of an update's keyframes are numbered in keyframe order, continuing the count, and a
  new object takes the number of its first detection (earliest keyframe, then the detector's
  order). Ids therefore have gaps, and they are bookkeeping only, but they depend on nothing
  except which detections form the object — not on the unconfirmed candidates, merges or removals
  of earlier updates — so a sequence mapped in one update or split over several in order gets the
  same ids wherever it associates the same detections. A merge keeps the lower id; merged and
  removed ids disappear for good (merges are recorded in ``merged_into`` so old per-frame instance
  files still resolve).
* **Split surfaces.** A horizontal surface that runs out of the view and under the items on it (a
  counter top around a book) is often split by the detector into several instances of one kind.
  Instances of one keyframe that ``split_surface`` allows and that touch with continuous depth
  (``surface_pieces``) are one instance of that keyframe (``join_instances``); pieces seen from
  different keyframes that share surface at one height are merged (``_one_surface``).
* **Merging** joins duplicates: objects with compatible labels that overlap, and — whatever their
  labels — objects of comparable size that occupy the same space (most of either one's points on
  the other's surface) and that no keyframe detected as two instances: the detector's label
  flickered between keyframes (a door seen as a wardrobe). The merged object's label is the one
  with the most evidence (score-weighted votes); the others are exported as ``detected_as``.
* **Depth-explained duplicates.** The per-keyframe depth scale drifts along a long sequence, so
  the keyframes that close a loop can place an object 10-15 % nearer or further than those that
  saw it first: two copies along the same viewing rays, too far apart for the overlap tests (a
  faucet 0.4 m from itself at 3 m). Objects of compatible labels that no keyframe detected
  together merge when their sightings agree once the depth ratio measured between their
  keyframes is removed (``_depth_explained``).
* **Point counts** are those of the map cloud: the points attributed to the object
  (``set_cloud_counts``), as the ``point_count`` of a single image counts its cloud's points. An
  object is exported only with at least ``EXPORT_MIN_CLOUD_POINTS`` of them, so every exported
  object appears in the cloud (``segments.ply``, ``color=segment``) in its colour.
* **Boxes cover the observed surface.** A box is fitted to the points the keyframes saw: an
  object seen only from the front (a refrigerator against a wall) has the depth of its visible
  surface, not its physical depth; no class-typical size is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from oh_my_slam.core import rle
from oh_my_slam.core.geometry import depth_edge_mask, voxel_keys
from oh_my_slam.core.types import Pose
from oh_my_slam.mapping.store import OBJECTS_JSON, frame_file
from oh_my_slam.mapping.validity import (
    BORDER,
    View,
    keyframe_view,
    stored_view,
    tau,
    well_registered,
)
from oh_my_slam.reconstruction.gravity import floor_candidate_height
from oh_my_slam.segmentation.api import (
    OBB,
    LiftedInstance,
    SceneObject,
    compatible,
    fit_object_obb,
    join_instances,
    lift_detections,
    obb_iou_upright,
    split_surface,
)

POINT_CAP = 30000
POINT_VOXEL = 0.01
IOU_GATE = 0.3
CENTROID_GATE = 1.5
# Confirmation: reliable detections in this many distinct keyframes (when another keyframe had the
# object in view). A detection is reliable unless most of its mask lies in the image-border band
# that ``View.usable`` excludes (``BORDER``): there the object is cut off and its monocular depth
# is unreliable, which is where one-off phantoms come from (a counter's edge at the bottom of a
# downward view, labelled "carpet" and placed inside the counter).
CONFIRM_DETECTIONS = 2
BORDER_EVIDENCE = 0.5  # largest share of a reliable detection's mask in the border band
IN_VIEW_SHARE = 0.5  # a keyframe has an object in view when this share of its points projects in
VIEW_SAMPLES = 200  # points per object for the in-view test
MERGE_OVERLAP = 0.5
MERGE_BOX_IOU = 0.3  # compatible objects whose boxes overlap this much are one physical object
MERGE_CONTAINMENT = 0.6
# Objects of incompatible labels are one object when no keyframe detected both, the smaller one's
# size is at least this share of the larger's (not a part of it or an item resting on it) and
# MERGE_OVERLAP of either one's points lie on the other's surface.
MERGE_SCALE = 0.5
# Box fit: two sightings agree when their bounds overlap or are at most max(CONSENSUS_TOL_MIN,
# CONSENSUS_TOL_REL · viewing distance) apart (monocular depth noise); with at least CONSENSUS_MIN
# sightings the box is fitted to the points in the bounds of those agreeing with the best one.
CONSENSUS_MIN = 3
CONSENSUS_TOL_MIN = 0.05
CONSENSUS_TOL_REL = 0.03
CONSENSUS_MARGIN = 0.02
REMOVE_FRACTION = 0.6
REMOVE_FRACTION_FEW = 0.8
REMOVE_MIN_FRAMES = 3
FEW_MIN_INLIERS = 150
MIN_VISIBLE_SAMPLES = 50
FEW_MIN_OBSERVATIONS = 4
ABSENCE_TAU_MIN = 0.25
ABSENCE_TAU_REL = 0.15
MAP_FLOOR_MAX_BELOW = 0.10
MASK_DILATE = 3
PAIR_NEIGHBOURS = 10  # keyframes (nearest by viewpoint) whose instances each keyframe's meet
EARLIER_VIEWS = 2  # an existing object's detecting keyframes (nearest by viewpoint) re-checked
# Split surfaces: two instances of one keyframe are pieces of one surface when at least
# SURFACE_CONTACT_PX pixels of one lie within SURFACE_CONTACT_BAND px of the other, on valid
# pixels free of depth edges, with depth continuous across the contact (median relative
# difference <= SURFACE_CONTACT_REL; separate surfaces that meet in the image differ by a depth
# edge there, > 4 %).
SURFACE_CONTACT_PX = 50
SURFACE_CONTACT_BAND = 2
SURFACE_CONTACT_REL = 0.03
# ... and two objects are pieces of one surface seen from different keyframes (``_one_surface``)
# when they share at least SURFACE_SHARED_POINTS canonical points (1 cm voxels: 100 cm² of
# surface) within SURFACE_PLAN_M horizontally and SURFACE_HEIGHT_TOL vertically — the depth noise
# of a surface seen at a grazing angle is mostly a height offset — at median heights within
# SURFACE_HEIGHT_TOL.
SURFACE_SHARED_POINTS = 100
SURFACE_PLAN_M = 0.05
SURFACE_HEIGHT_TOL = 0.1
# Depth-explained duplicates (``_depth_explained``): the DEPTH_PAIRS pairs of detecting keyframes
# nearest by viewpoint are compared; a pair counts when its keyframes' depths disagree by at least
# DEPTH_RATIO_MIN — beyond the alignment noise (consecutive keyframes agree within ~2 %), else the
# overlap tests are valid and decide — and the sightings agree once the ratio is removed. The
# objects' centres must lie within DEPTH_RATIO_MAX of their distance (plus their extents).
DEPTH_PAIRS = 3
DEPTH_RATIO_MIN = 0.05
DEPTH_RATIO_MAX = 0.3
RATIO_MIN_POINTS = 500  # shared surface points for a keyframe pair's depth ratio
# An object is exported only with at least this many map-cloud points: every exported object is
# then drawn in the cloud (segments.ply, color=segment) in its colour. A confirmed object may have
# none when its surface did not survive the fusion (seen by fewer than 3 keyframes, e.g. a pendant
# lamp) or when fewer than a third of the keyframes that see its surface detected it.
EXPORT_MIN_CLOUD_POINTS = 1
UP = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------------------------------------
# canonical point sets


def _voxel_hash(keys: NDArray[np.int64]) -> NDArray[np.uint64]:
    """Well-mixed 64-bit hash of integer voxel keys (deterministic across runs)."""
    k = keys.astype(np.uint64)
    h = (k[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (k[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) \
        ^ (k[:, 2] * np.uint64(0x165667B19E3779F9))
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h


def canonical_points(points: NDArray[Any]) -> NDArray[np.float32]:
    """One point per ``POINT_VOXEL`` voxel (the one nearest the voxel centre) and at most
    ``POINT_CAP`` of them (the voxels with the smallest hash), sorted by voxel.

    A pure function of the point *set* that composes — ``canonical(canonical(a) ∪ b) ==
    canonical(a ∪ b)`` — so an object's points, and the OBB fitted to them, do not depend on the
    order in which its instances arrived or on how they were split into updates."""
    p = np.asarray(points, np.float32).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros((0, 3), np.float32)
    p64 = p.astype(np.float64)
    keys = voxel_keys(p64, POINT_VOXEL)
    d = np.sum((p64 - (keys + 0.5) * POINT_VOXEL) ** 2, axis=1)
    order = np.lexsort((p64[:, 2], p64[:, 1], p64[:, 0], d, keys[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
    sel = order[first]
    p, keys = p[sel], keys[sel]
    if len(p) > POINT_CAP:
        h = _voxel_hash(keys)
        keep = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], h))[:POINT_CAP]
        p = p[np.sort(keep)]
    return p


# ------------------------------------------------------------------------------------------------
# objects and instances


@dataclass(frozen=True)
class Sighting:
    """Summary of one detection of an object: its keyframe, lifted point count, the share of its
    mask in the image-border band, and the centroid and robust (2-98 %) bounds of its points in
    map coordinates."""

    frame: int
    points: int
    border: float
    centroid: tuple[float, float, float]
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @property
    def reliable(self) -> bool:
        return self.border <= BORDER_EVIDENCE

    def key(self) -> tuple[Any, ...]:
        return (self.frame, self.points, self.centroid, self.lo, self.hi, self.border)

    def to_list(self) -> list[float]:
        return [self.frame, self.points, self.border, *self.centroid, *self.lo, *self.hi]

    @staticmethod
    def from_list(v: list[float]) -> Sighting:
        def t(x: list[float]) -> tuple[float, float, float]:
            return (float(x[0]), float(x[1]), float(x[2]))
        return Sighting(int(v[0]), int(v[1]), float(v[2]), t(v[3:6]), t(v[6:9]), t(v[9:12]))


@dataclass
class MapObject:
    id: int
    label: str
    label_votes: dict[str, float]
    scores: list[float]
    points: NDArray[np.float32]
    obb: OBB | None = None
    frames: list[int] = field(default_factory=list)  # keyframes that detected it (sorted)
    confirmed: bool = False
    strikes: int = 0
    views_in_frustum: int = 0  # keyframes of the map that have it in view (or detected it)
    created_update: int = 0
    last_seen_update: int = 0
    obs_depth: float = 2.0  # mean camera distance of its detections
    pixel_count: int = 0
    sightings: list[Sighting] = field(default_factory=list)  # one per detection (sorted)
    cloud_points: int | None = None  # map-cloud points attributed to it (None: not yet counted)

    @property
    def observations(self) -> int:
        """Keyframes in which the object was detected."""
        return len(self.frames)

    def reliable_frames(self) -> set[int]:
        """Keyframes with a reliable detection (``Sighting.reliable``); keyframes without a
        sighting (maps written before sightings were recorded) count as reliable."""
        seen = {s.frame for s in self.sightings}
        return {s.frame for s in self.sightings if s.reliable} | (set(self.frames) - seen)

    def _add_sightings(self, new: list[Sighting]) -> None:
        self.sightings = sorted([*self.sightings, *new], key=Sighting.key)

    @property
    def score(self) -> float:
        top = sorted(self.scores, reverse=True)[:3]
        return float(np.mean(top)) if top else 0.0

    @property
    def centroid(self) -> NDArray[np.float64]:
        return self.points.mean(0).astype(np.float64) if len(self.points) else np.zeros(3)

    def add_points(self, pts: NDArray[Any]) -> None:
        self.points = canonical_points(np.concatenate([self.points, np.asarray(pts, np.float32)]))

    def vote(self, label: str, score: float) -> None:
        self.label_votes[label] = self.label_votes.get(label, 0.0) + score
        self.label = max(self.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.scores = sorted([*self.scores, float(score)], reverse=True)[:10]

    def add(self, obs: list[Observation], uid: int) -> None:
        """Evidence of instances of update ``uid`` (in any order: the result is the same)."""
        obs = sorted(obs, key=lambda ob: ob.key)
        n0 = len(self.frames)
        for ob in obs:
            self.vote(ob.label, ob.score)
        self.add_points(np.concatenate([ob.points for ob in obs]))
        self.frames = sorted(set(self.frames) | {ob.frame for ob in obs})
        self.obs_depth = float((self.obs_depth * n0 + sum(ob.depth for ob in obs))
                               / (n0 + len(obs)))
        self.pixel_count = max([self.pixel_count] + [ob.pixel_count for ob in obs])
        self.last_seen_update = max(self.last_seen_update, uid)
        self._add_sightings([ob.sighting for ob in obs])

    def absorb(self, gone: MapObject) -> None:
        """Merge another object's evidence into this one."""
        for lab, v in gone.label_votes.items():
            self.label_votes[lab] = self.label_votes.get(lab, 0.0) + v
        self.label = max(self.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.scores = sorted(self.scores + gone.scores, reverse=True)[:10]
        self.add_points(gone.points)
        n_a, n_b = len(self.frames), len(gone.frames)
        if n_a + n_b:
            self.obs_depth = (self.obs_depth * n_a + gone.obs_depth * n_b) / (n_a + n_b)
        self.frames = sorted(set(self.frames) | set(gone.frames))
        self.views_in_frustum = max(self.views_in_frustum, gone.views_in_frustum)
        self.strikes = min(self.strikes, gone.strikes)
        self.created_update = min(self.created_update, gone.created_update)
        self.last_seen_update = max(self.last_seen_update, gone.last_seen_update)
        self.pixel_count = max(self.pixel_count, gone.pixel_count)
        self._add_sightings(gone.sightings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "label_votes": self.label_votes,
            "scores": self.scores, "score": self.score,
            "obb": None if self.obb is None else self.obb.to_dict(),
            "observations": self.observations, "frames": self.frames,
            "confirmed": self.confirmed, "strikes": self.strikes,
            "views_in_frustum": self.views_in_frustum, "created_update": self.created_update,
            "last_seen_update": self.last_seen_update, "obs_depth": self.obs_depth,
            "pixel_count": self.pixel_count, "point_count": self.point_count,
            "cloud_points": self.cloud_points, "point_file": points_file(self.id),
            "sightings": [s.to_list() for s in self.sightings],
        }

    @staticmethod
    def from_dict(d: dict[str, Any], points: NDArray[np.float32]) -> MapObject:
        cloud = d.get("cloud_points")
        return MapObject(
            id=int(d["id"]), label=d["label"], label_votes=dict(d.get("label_votes", {})),
            scores=list(d.get("scores", [])), points=points,
            obb=None if d.get("obb") is None else OBB.from_dict(d["obb"]),
            frames=sorted({int(f) for f in d.get("frames", [])}),
            confirmed=bool(d.get("confirmed", False)), strikes=int(d.get("strikes", 0)),
            views_in_frustum=int(d.get("views_in_frustum", 0)),
            created_update=int(d.get("created_update", 0)),
            last_seen_update=int(d.get("last_seen_update", 0)),
            obs_depth=float(d.get("obs_depth", 2.0)), pixel_count=int(d.get("pixel_count", 0)),
            sightings=sorted((Sighting.from_list(v) for v in d.get("sightings", [])),
                             key=Sighting.key),
            cloud_points=None if cloud is None else int(cloud),
        )

    @property
    def point_count(self) -> int:
        """Points of the map cloud attributed to the object (as a single image's ``point_count``
        counts its cloud's points); the stored sample's size for a map written before counts
        were recorded."""
        return len(self.points) if self.cloud_points is None else self.cloud_points

    def scene_object(self) -> SceneObject:
        assert self.obb is not None
        return SceneObject(
            id=self.id, label=self.label, score=self.score, obb=self.obb,
            pixel_count=self.pixel_count, point_count=self.point_count,
            observations=self.observations, confirmed=self.confirmed, frames=list(self.frames),
            labels=(self.label, *(lab for lab, _ in sorted(self.label_votes.items(),
                                                           key=lambda kv: (-kv[1], kv[0]))
                                  if lab != self.label)),
        )


@dataclass
class Observation:
    """One lifted instance of a keyframe of this update, in map coordinates: a detection, or the
    pieces of a split surface joined (``members``: the detections it stands for)."""

    frame: int  # keyframe index
    view: View
    inst: LiftedInstance
    points: NDArray[np.float32]
    centroid: NDArray[np.float64]
    depth: float  # median camera distance
    extent: float
    members: tuple[LiftedInstance, ...] = ()
    _tree: cKDTree | None = field(default=None, repr=False)

    @staticmethod
    def of(frame: int, view: View, inst: LiftedInstance,
           members: tuple[LiftedInstance, ...] = ()) -> Observation:
        pts = np.asarray(inst.lifted.points, np.float64)
        dist = float(np.median(np.linalg.norm(pts - view.T_map_cam.t, axis=1)))
        return Observation(frame, view, inst, pts.astype(np.float32), pts.mean(0), dist,
                           _extent(pts), members or (inst,))

    @property
    def sighting(self) -> Sighting:
        lo, hi = np.percentile(self.points.astype(np.float64), [2, 98], axis=0)
        c = self.centroid

        def t(x: NDArray[Any]) -> tuple[float, float, float]:
            return (float(x[0]), float(x[1]), float(x[2]))
        return Sighting(self.frame, len(self.points), border_share(self.inst.mask), t(c), t(lo),
                        t(hi))

    @property
    def label(self) -> str:
        return self.inst.detection.label

    @property
    def score(self) -> float:
        return float(self.inst.detection.score)

    @property
    def pixel_count(self) -> int:
        return int(self.inst.mask.sum())

    @property
    def radius(self) -> float:
        """Distance within which another point set counts as the same surface."""
        return max(0.05, 0.02 * self.depth)

    @property
    def key(self) -> tuple[str, float, float, float, float, int]:
        """Content key: a total order that does not depend on the keyframes' order."""
        c = self.centroid
        return (self.label, float(c[0]), float(c[1]), float(c[2]), -self.score,
                -self.pixel_count)

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree


def points_file(oid: int) -> str:
    return f"objects/points_{oid:06d}.npy"


@dataclass
class ObjectState:
    objects: list[MapObject]
    next_id: int
    merged_into: dict[int, int] = field(default_factory=dict)
    floor_z: float | None = None
    observed: set[int] = field(default_factory=set)  # ids observed in this update
    summary: dict[str, Any] = field(default_factory=dict)

    def by_id(self) -> dict[int, MapObject]:
        return {o.id: o for o in self.objects}

    def resolve(self, oid: int) -> int | None:
        seen = set()
        while oid in self.merged_into and oid not in seen:
            seen.add(oid)
            oid = self.merged_into[oid]
        return oid if oid in self.by_id() else None

    def exported(self) -> list[SceneObject]:
        """The confirmed objects with a box and points in the map cloud (not yet counted: maps
        written before cloud counts were recorded)."""
        return [o.scene_object() for o in sorted(self.objects, key=lambda o: o.id)
                if o.confirmed and o.obb is not None
                and (o.cloud_points is None or o.cloud_points >= EXPORT_MIN_CLOUD_POINTS)]


def load_state(current: Any, meta: dict[str, Any]) -> ObjectState:
    import json

    p = current(OBJECTS_JSON)
    if not p.exists():
        return ObjectState([], int(meta.get("next_object_id", 1)), floor_z=meta.get("floor_z"))
    d = json.loads(p.read_text())
    objs = []
    for od in d.get("objects", []):
        pp = current(points_file(int(od["id"])))
        pts = np.load(pp).astype(np.float32) if pp.exists() else np.zeros((0, 3), np.float32)
        objs.append(MapObject.from_dict(od, pts))
    return ObjectState(objs, int(d.get("next_id", meta.get("next_object_id", 1))),
                       {int(k): int(v) for k, v in d.get("merged_into", {}).items()},
                       floor_z=d.get("floor_z", meta.get("floor_z")))


# ------------------------------------------------------------------------------------------------
# geometry helpers


def _visible_pixels(view: View, pts: NDArray[Any]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(u, v) pixels of ``view`` onto which the in-view, unoccluded points of ``pts`` project."""
    if len(pts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    inside, z, d, u, v = view.lookup_pixels(pts)
    vis = inside & (z <= d + tau(z))
    return u[vis], v[vis]


def projected_mask(view: View, pts: NDArray[Any], dilate: int = MASK_DILATE) -> NDArray[np.bool_]:
    """Pixels of ``view`` covered by the visible (unoccluded) points of an object."""
    mask = np.zeros(view.depth.shape, bool)
    u, v = _visible_pixels(view, pts)
    if len(u) == 0:
        return mask
    mask[v, u] = True
    return ndimage.binary_dilation(mask, iterations=dilate) if dilate else mask


def projected_iou(view: View, mask: NDArray[np.bool_], pts: NDArray[Any],
                  dilate: int = MASK_DILATE) -> float:
    """IoU of ``mask`` with the pixels ``pts`` cover in ``view`` (``projected_mask``), computed on
    the bounding box of both (same result as on the whole image, much cheaper)."""
    u, v = _visible_pixels(view, pts)
    if len(u) == 0:
        return 0.0
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    r0, r1 = int(v.min()), int(v.max())
    c0, c1 = int(u.min()), int(u.max())
    if len(rows):
        r0, r1 = min(r0, int(rows[0])), max(r1, int(rows[-1]))
        c0, c1 = min(c0, int(cols[0])), max(c1, int(cols[-1]))
    r0, r1 = max(0, r0 - dilate), min(h, r1 + dilate + 1)
    c0, c1 = max(0, c0 - dilate), min(w, c1 + dilate + 1)
    pm = np.zeros((r1 - r0, c1 - c0), bool)
    pm[v - r0, u - c0] = True
    if dilate:
        pm = ndimage.binary_dilation(pm, iterations=dilate)
    m = mask[r0:r1, c0:c1]
    union = np.logical_or(pm, m).sum()
    return float(np.logical_and(pm, m).sum() / union) if union else 0.0


def overlap_fraction(a: NDArray[Any], b: NDArray[Any], radius: float,
                     tree: cKDTree | None = None) -> float:
    """Fraction of points of ``a`` within ``radius`` of ``b`` (``tree``: a KD-tree of ``b``)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa = a if len(a) <= 4000 else a[np.linspace(0, len(a) - 1, 4000).astype(int)]
    d, _ = (tree or cKDTree(b)).query(sa, k=1, distance_upper_bound=radius)
    return float(np.isfinite(d).mean())


def _extent(pts: NDArray[Any]) -> float:
    """Horizontal diagonal of a point set (large objects are seen piecewise)."""
    if len(pts) < 2:
        return 0.0
    lo, hi = np.percentile(np.asarray(pts)[:, :2], [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def absence_tau(z: NDArray[Any]) -> NDArray[Any]:
    """Margin for "seen through": wider than the pixel test (monocular depth of thin or small
    objects is less reliable than of large surfaces)."""
    return np.maximum(ABSENCE_TAU_MIN, ABSENCE_TAU_REL * z)


def visibility_evidence(view: View, pts: NDArray[Any]) -> tuple[int, int]:
    """(seen-through, consistent) counts of an object's in-view, unoccluded surface samples."""
    if len(pts) > 3000:
        pts = pts[np.linspace(0, len(pts) - 1, 3000).astype(int)]
    inside, z, d = view.lookup(pts)
    z, d = z[inside], d[inside]
    t = absence_tau(z)
    through = int((d - z > t).sum())
    consistent = int((np.abs(d - z) <= t).sum())
    return through, consistent


def border_share(mask: NDArray[np.bool_]) -> float:
    """Share of a mask's pixels in the image-border band that ``View.usable`` excludes."""
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return 0.0
    h, w = m.shape
    bh, bw = max(1, int(BORDER * h)), max(1, int(BORDER * w))
    inner = int(m[bh:h - bh, bw:w - bw].sum())
    return float((n - inner) / n)


def depth_ratio(a: View, b: View) -> float | None:
    """How much further keyframe ``b`` places the surfaces both keyframes see than ``a`` does:
    the median, over ``a``'s surface points that ``b`` sees, of ``b``'s observed depth over the
    point's depth in ``b``'s camera (the same surface only: within a factor 1.3, larger
    differences are occlusions). Monocular depth maps disagree by a largely global factor (as in
    ``validity``). None when they share fewer than ``RATIO_MIN_POINTS`` points."""
    pts, _, _ = a.grid_points()
    if len(pts) == 0:
        return None
    inside, z, d = b.lookup(pts)
    r = d[inside] / z[inside]
    r = r[(r > 1 / 1.3) & (r < 1.3)]
    return float(np.median(r)) if len(r) >= RATIO_MIN_POINTS else None


def _bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def continuous_contact(a: NDArray[np.bool_], b: NDArray[np.bool_], depth: NDArray[Any],
                       good: NDArray[np.bool_]) -> bool:
    """Whether masks ``a`` and ``b`` touch along a continuous surface: at least
    ``SURFACE_CONTACT_PX`` pixels of ``b`` within ``SURFACE_CONTACT_BAND`` px of ``a`` are
    ``good`` (valid, no depth edge), and there ``b``'s depth continues ``a``'s (the median relative
    difference to ``a``'s mean depth around each contact pixel is at most
    ``SURFACE_CONTACT_REL``)."""
    ba, bb = _bbox(a), _bbox(b)
    if ba is None or bb is None:
        return False
    m = SURFACE_CONTACT_BAND + 1
    r0, r1 = max(ba[0], bb[0]) - m, min(ba[1], bb[1]) + m
    c0, c1 = max(ba[2], bb[2]) - m, min(ba[3], bb[3]) + m
    if r1 <= r0 or c1 <= c0:  # the boxes (with the band) do not meet
        return False
    h, w = a.shape
    sl = (slice(max(0, r0 - m), min(h, r1 + m)), slice(max(0, c0 - m), min(w, c1 + m)))
    A, B, G, D = a[sl], b[sl], good[sl], np.asarray(depth, np.float64)[sl]
    near = ndimage.binary_dilation(A, iterations=SURFACE_CONTACT_BAND) & B & G
    size = 2 * SURFACE_CONTACT_BAND + 1
    wa = ndimage.uniform_filter((A & G).astype(np.float64), size)
    sa = ndimage.uniform_filter(np.where(A & G, D, 0.0), size)
    ok = near & (wa > 1e-9)
    if int(ok.sum()) < SURFACE_CONTACT_PX:
        return False
    rel = D[ok] / (sa[ok] / wa[ok]) - 1.0
    return abs(float(np.median(rel))) <= SURFACE_CONTACT_REL


def surface_pieces(view: View, insts: list[LiftedInstance]) -> list[list[int]]:
    """The instances of one keyframe grouped into objects: pieces of one horizontal surface that
    the detector split around the items on it (labels allowed by ``split_surface``, touching with
    continuous depth, ``continuous_contact``) form one group; every other instance is alone.
    Groups are listed by their first instance."""
    n = len(insts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if split_surface(insts[i].detection.label, insts[j].detection.label)]
    if pairs:
        ok = view.valid & (view.depth > 0)
        good = ok & ~depth_edge_mask(np.where(ok, view.depth, 0.0))
        for i, j in pairs:
            if find(i) != find(j) and continuous_contact(insts[i].mask, insts[j].mask,
                                                         view.depth, good):
                parent[max(find(i), find(j))] = min(find(i), find(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def in_view(pts: NDArray[Any], K: Any, T_map_cam: Any) -> bool:
    """Whether a keyframe (grid intrinsics ``K`` and pose) has at least ``IN_VIEW_SHARE`` of the
    points in front of it and inside its image. Occlusion is ignored: the question is whether the
    keyframe could have detected the object, and an object whose depth is wrong would otherwise
    hide from every other keyframe."""
    if len(pts) == 0:
        return False
    pc = T_map_cam.inverse().apply(np.asarray(pts, np.float64))
    z = pc[:, 2]
    w, h = K.width, K.height
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * pc[:, 0] / z + K.cx
        v = K.fy * pc[:, 1] / z + K.cy
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return bool(inside.mean() >= IN_VIEW_SHARE)


def count_views(obj: MapObject, records: list[Any]) -> int:
    """Keyframes of the map (``records``) that detected the object or have it in view."""
    pts = obj.points
    if len(pts) > VIEW_SAMPLES:
        pts = pts[np.linspace(0, len(pts) - 1, VIEW_SAMPLES).astype(int)]
    detected = set(obj.frames)
    return sum(1 for r in records if r.index in detected or in_view(pts, r.K_grid, r.T_map_cam))


def below_floor(obj: MapObject, floor_z: float | None, margin: float = 0.3) -> bool:
    """An object entirely below the map floor is not physical (e.g. seen in a mirror or through
    a reflective screen, where monocular depth places it behind the glass)."""
    if floor_z is None or len(obj.points) < 10:
        return False
    return float(np.percentile(obj.points[:, 2], 90)) < floor_z - margin


def agreeing_sightings(obj: MapObject) -> list[Sighting]:
    """The sightings that agree with the one most others agree with (ties: more points, then
    content). Two sightings agree when their bounds overlap, allowing a gap of the depth noise at
    the object's distance: the partial views of a large object overlap one another, while
    sightings of a small object that monocular depth scattered along the viewing rays do not."""
    s = obj.sightings
    if not s:
        return []
    lo = np.array([x.lo for x in s])
    hi = np.array([x.hi for x in s])
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * obj.obs_depth)
    near = np.all((lo[:, None] <= hi[None] + tol) & (lo[None] <= hi[:, None] + tol), axis=2)
    best = min(range(len(s)), key=lambda i: (-int(near[i].sum()), -s[i].points, s[i].key()))
    return [x for x, ok in zip(s, near[best], strict=True) if ok]


def fit_points(obj: MapObject) -> NDArray[np.float32]:
    """The points a box is fitted to: with at least ``CONSENSUS_MIN`` sightings (all recorded),
    those inside the bounds of the agreeing sightings (``agreeing_sightings``), else all."""
    s = obj.sightings
    if len(s) < CONSENSUS_MIN or {x.frame for x in s} != set(obj.frames):
        return obj.points
    keep = agreeing_sightings(obj)
    if len(keep) == len(s):
        return obj.points
    lo = np.min([x.lo for x in keep], axis=0) - CONSENSUS_MARGIN
    hi = np.max([x.hi for x in keep], axis=0) + CONSENSUS_MARGIN
    inside = np.all((obj.points >= lo) & (obj.points <= hi), axis=1)
    return obj.points[inside] if inside.sum() >= 10 else obj.points


def refit(obj: MapObject, floor_z: float | None) -> None:
    if len(obj.points) < 10:
        return
    obj.obb = fit_object_obb(fit_points(obj), obj.label, UP, floor_z)


def confirm(obj: MapObject) -> None:
    """Confirmed when reliable detections (``MapObject.reliable_frames``) come from at least
    ``CONFIRM_DETECTIONS`` keyframes, or from one when no other keyframe of the map has the object
    in view (``views_in_frustum`` <= 1: a single image, or a place only one keyframe saw)."""
    need = CONFIRM_DETECTIONS if obj.views_in_frustum > 1 else 1
    obj.confirmed = len(obj.reliable_frames()) >= need


# ------------------------------------------------------------------------------------------------
# the update


def map_floor(ctx: Any, per_frame: int | None = None) -> tuple[NDArray[np.float64], float | None]:
    """Map-frame grid points of this update's confidently placed keyframes (about ``per_frame``
    of each, all when None) and the floor height among them: the lowest well-supported
    horizontal surface. Maps accumulate stray points below the floor (see-through shelves,
    reflections, drift), so up to 10 % of the points may lie beneath it (single images: 3 %)."""
    pts = []
    placed = [nf for nf in ctx.new if nf.record is not None and nf.depth is not None]
    for nf in sorted(placed, key=lambda nf: nf.record.order_key):
        if nf.record.low_confidence:
            continue
        p, _, _ = keyframe_view(nf).grid_points()
        pts.append(p if per_frame is None else p[:: max(1, len(p) // per_frame)])
    if not pts:
        return np.zeros((0, 3)), None
    allp = np.concatenate(pts)
    return allp, floor_candidate_height(allp[:, 2], max_below=MAP_FLOOR_MAX_BELOW)


def _instances_json(frame_items: list[tuple[int, LiftedInstance]]) -> dict[str, Any]:
    return {"instances": [
        {"object_id": oid, "label": inst.detection.label, "score": inst.detection.score,
         "source": inst.detection.source, "mask": rle.encode(inst.mask)}
        for oid, inst in frame_items
    ]}


def update_objects(ctx: Any, records: list[Any], progress: Any) -> ObjectState:
    tx = ctx.tx
    state = load_state(tx.current, ctx.meta)
    uid = ctx.update_id
    _, fz = map_floor(ctx)
    if state.floor_z is None or (fz is not None and not ctx.old_frames):
        state.floor_z = fz
    ctx.meta["floor_z"] = state.floor_z
    old_ids = {o.id for o in state.objects}

    # 1. every instance of the update, lifted into the map, with the number of its detection:
    #    all detections of the update's keyframes (placed or not) in keyframe order, continuing
    #    the map's count; the pieces of a split surface are one instance (numbered by the first)
    first_new = state.next_id
    first_number: dict[int, int] = {}
    count = first_new
    for nf in sorted(ctx.new, key=lambda nf: nf.kf.index):
        first_number[nf.kf.index] = count
        count += len(nf.dets)
    new_views: dict[int, tuple[Any, View]] = {}
    per_frame: dict[int, list[int]] = {}
    obs: list[Observation] = []
    number: list[int] = []  # detection number per observation
    for nf in ctx.new:
        if nf.record is None or nf.depth is None:
            continue
        rec = nf.record
        view = keyframe_view(nf)
        new_views[rec.index] = (nf, view)
        insts = lift_detections(nf.frame, nf.dets, rec.T_map_cam, depth=nf.depth,
                                valid=view.valid)
        pieces = surface_pieces(view, insts)
        per_frame[rec.index] = list(range(len(obs), len(obs) + len(pieces)))
        rank = {id(det): j for j, det in enumerate(nf.dets)}
        for group in pieces:
            members = tuple(insts[k] for k in group)
            obs.append(Observation.of(rec.index, view, join_instances(list(members)), members))
            number.append(first_number[nf.kf.index]
                          + min(rank[id(m.detection)] for m in members))
    views = _Views(ctx, {f: v for f, (_, v) in new_views.items()}, records)

    # 2. group them (order-free) and fold each group into an existing or a new object; new
    #    objects get provisional ids >= first_new (content order) until their final numbering
    owner: list[int] = [0] * len(obs)  # object id per observation
    touched: set[int] = set()
    groups = _group(obs, state.objects, _Earlier(ctx, state, views))
    existing = state.by_id()
    fresh_groups = sorted((m for oid, m in groups if oid is None),
                          key=lambda m: min(obs[i].key for i in m))
    targets: list[tuple[MapObject, list[int]]] = [
        (existing[oid], m) for oid, m in groups if oid is not None]
    for k, m in enumerate(fresh_groups):
        o = MapObject(first_new + k, obs[m[0]].label, {}, [], np.zeros((0, 3), np.float32),
                      created_update=uid)
        state.objects.append(o)
        targets.append((o, m))
    for o, m in targets:
        o.add([obs[i] for i in m], uid)
        o.strikes = 0  # re-detected: the latest observation says it is there
        touched.add(o.id)
        for i in m:
            owner[i] = o.id
    for o in state.objects:  # provisional boxes: the merge test compares boxes of new objects too
        if o.id in touched:
            refit(o, state.floor_z)

    # 3. merge duplicates (lower id kept; keepers are refitted), then count the keyframes of the
    #    whole map that have each object in view and confirm
    alias: dict[int, int] = {}
    merged = _merge(state, touched, alias, views)
    for o in state.objects:
        o.views_in_frustum = count_views(o, records)
        confirm(o)

    # 4. drop non-physical objects and objects this update shows to be gone
    dropped = {o.id for o in state.objects if below_floor(o, state.floor_z)}
    removed = _absence([o for o in state.objects if o.id in old_ids and o.id not in touched
                        and o.id not in dropped], new_views)
    gone = dropped | set(removed)
    for oid in sorted(gone | set(alias)):
        if oid < first_new:
            tx.delete(points_file(oid))
    state.objects = [o for o in state.objects if o.id not in gone]

    # 5. final ids of the new objects: the number of their first detection (bookkeeping)
    def resolved(oid: int) -> int:
        while oid in alias:
            oid = alias[oid]
        return oid

    first_detection: dict[int, int] = {}
    for i, oid in enumerate(owner):
        k = resolved(oid)
        first_detection[k] = min(first_detection.get(k, number[i]), number[i])
    fresh = [o for o in state.objects if o.id >= first_new]
    rename = {o.id: first_detection[o.id] for o in fresh}
    for o in fresh:
        o.id = rename[o.id]
    state.next_id = count
    for old, keeper in alias.items():
        if old < first_new:  # a stored id; its keeper has a lower id, so is stored too
            state.merged_into[old] = keeper
    touched = {rename.get(t, t) for t in touched if t not in alias and t not in gone}

    def final_id(oid: int) -> int:
        oid = resolved(oid)
        if oid in gone:
            return 0 if oid >= first_new else oid  # a removed id resolves to nothing
        return rename.get(oid, oid)

    for f, idxs in per_frame.items():
        nf, _ = new_views[f]
        items = [(final_id(owner[i]), m) for i in idxs for m in obs[i].members]
        tx.write_json(frame_file(nf.record.name, "instances.json"), _instances_json(items))
    state.observed = touched
    for o in state.objects:
        if o.id in touched:
            tx.save_npy(points_file(o.id), o.points.astype(np.float32))
    save_state(tx, state)
    confirmed = sum(o.confirmed for o in state.objects)
    state.summary = {
        "instances": len(obs), "touched": len(touched), "new": len(fresh), "merged": merged,
        "removed": sorted(removed), "below_floor_dropped": len(dropped),
        "total": len(state.objects), "confirmed": confirmed,
        "unconfirmed": len(state.objects) - confirmed,
    }
    progress(f"objects: {confirmed} confirmed of {len(state.objects)}; "
             f"{len(removed)} removed, {merged} merged")
    return state


def save_state(tx: Any, state: ObjectState) -> None:
    tx.write_json(OBJECTS_JSON, {
        "next_id": state.next_id,
        "floor_z": state.floor_z,
        "merged_into": {str(k): v for k, v in sorted(state.merged_into.items())},
        "objects": [o.to_dict() for o in sorted(state.objects, key=lambda o: o.id)],
    })


def set_cloud_counts(tx: Any, state: ObjectState, labels: NDArray[Any]) -> None:
    """Record each object's point count in the map cloud (``labels``: object id per cloud point)
    and store the state again."""
    ids = np.asarray(labels, np.int64).reshape(-1)
    counts = np.bincount(ids[ids > 0]) if (ids > 0).any() else np.zeros(1, np.int64)
    for o in state.objects:
        o.cloud_points = int(counts[o.id]) if o.id < len(counts) else 0
    save_state(tx, state)


# ------------------------------------------------------------------------------------------------
# association


def _object_affinity(ob: Observation, o: MapObject, tree: cKDTree,
                     earlier: _Earlier | None = None) -> float:
    """How well an instance matches an existing object: max(projected-mask IoU, share of the
    instance's points on the object's); when neither reaches the gate, also the IoU of the
    object's mask in its detecting keyframes of earlier updates (``_Earlier``) with the instance's
    projected points — the other direction of ``_pair_affinity``, so an object whose depth
    differs between the updates (its points lie behind the new keyframe's surface, hidden) is
    matched as it would be within one update."""
    s = max(projected_iou(ob.view, ob.inst.mask, o.points),
            overlap_fraction(ob.points, o.points, ob.radius, tree))
    if s < IOU_GATE and earlier is not None:
        for view, mask in earlier.masks(o, ob):
            s = max(s, projected_iou(view, mask, ob.points))
    return s


class _Views:
    """The map's keyframes by index: their poses, their views (this update's from memory, earlier
    ones loaded lazily from the map) and the depth ratio of keyframe pairs (``depth_ratio``),
    cached."""

    def __init__(self, ctx: Any, new: dict[int, View], records: list[Any]) -> None:
        self.ctx = ctx
        self.records = {r.index: r for r in records}
        self.views: dict[int, View | None] = dict(new)
        self.ratios: dict[tuple[int, int], float | None] = {}

    def pose(self, index: int) -> Pose | None:
        rec = self.records.get(index)
        return None if rec is None else rec.T_map_cam

    def get(self, index: int) -> View | None:
        if index not in self.views:
            rec = self.records.get(index)
            self.views[index] = (None if rec is None or self.ctx is None
                                 else stored_view(self.ctx.tx.current, rec))
        return self.views[index]

    def ratio(self, a: int, b: int) -> float | None:
        if (a, b) not in self.ratios:
            va, vb = self.get(a), self.get(b)
            self.ratios[(a, b)] = None if va is None or vb is None else depth_ratio(va, vb)
        return self.ratios[(a, b)]


class _Earlier:
    """Keyframes of earlier updates that detected an existing object: their stored views and the
    object's masks in them (``instances.json``), loaded lazily. ``masks`` gives the
    ``EARLIER_VIEWS`` of them nearest to an instance's viewpoint."""

    def __init__(self, ctx: Any, state: ObjectState, views: _Views) -> None:
        self.ctx = ctx
        self.state = state
        self.records = {r.index: r for r in ctx.old_frames}
        self.views = views
        self.instances: dict[int, list[dict[str, Any]]] = {}

    def _view(self, rec: Any) -> View | None:
        return self.views.get(rec.index)

    def _mask(self, rec: Any, oid: int, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
        if rec.index not in self.instances:
            import json

            p = self.ctx.tx.current(frame_file(rec.name, "instances.json"))
            self.instances[rec.index] = (json.loads(p.read_text()).get("instances", [])
                                         if p.exists() else [])
        masks = [rle.decode(x["mask"]) for x in self.instances[rec.index]
                 if self.state.resolve(int(x["object_id"])) == oid]
        masks = [m for m in masks if m.shape == shape]
        return np.logical_or.reduce(masks) if masks else None

    def masks(self, o: MapObject, ob: Observation) -> list[tuple[View, NDArray[np.bool_]]]:
        T = ob.view.T_map_cam
        scale = max(0.5, ob.depth)

        def distance(f: int) -> tuple[float, int]:
            R = self.records[f].T_map_cam
            return (float(np.linalg.norm(R.t - T.t)) / scale
                    + 1.0 - float(R.R[:, 2] @ T.R[:, 2]), f)

        out = []
        for f in sorted((f for f in o.frames if f in self.records), key=distance)[:EARLIER_VIEWS]:
            view = self._view(self.records[f])
            if view is None:
                continue
            mask = self._mask(self.records[f], o.id, view.depth.shape)
            if mask is not None:
                out.append((view, mask))
        return out


def _pair_affinity(a: Observation, b: Observation) -> float:
    """How well two instances of different keyframes agree (symmetric): the IoU of each one's
    mask with the other's projected points; when neither reaches the gate, also the share of
    either's points on the other's (a partial view whose projection is occluded)."""
    s = max(projected_iou(a.view, a.inst.mask, b.points),
            projected_iou(b.view, b.inst.mask, a.points))
    if s >= IOU_GATE:
        return s
    return max(s, overlap_fraction(a.points, b.points, a.radius, b.tree()),
               overlap_fraction(b.points, a.points, b.radius, a.tree()))


def _compatible_matrix(a: list[str], b: list[str]) -> NDArray[np.bool_]:
    """``compatible`` for every label pair (evaluated once per distinct pair)."""
    ua, ub = sorted(set(a)), sorted(set(b))
    ia, ib = {x: k for k, x in enumerate(ua)}, {x: k for k, x in enumerate(ub)}
    table = np.array([[compatible(x, y) for y in ub] for x in ua], bool).reshape(len(ua), len(ub))
    return table[np.array([ia[x] for x in a], int)[:, None], np.array([ib[y] for y in b], int)]


def _near(ca: NDArray[Any], ea: NDArray[Any], cb: NDArray[Any], eb: NDArray[Any]
          ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Index pairs (i, j) whose centroids lie within max(CENTROID_GATE, 0.6 · larger extent)."""
    if len(ca) == 0 or len(cb) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    r = max(CENTROID_GATE, 0.6 * float(max(ea.max(), eb.max())))
    lists = cKDTree(ca).query_ball_tree(cKDTree(cb), r)
    i = np.array([k for k, js in enumerate(lists) for _ in js], np.int64)
    j = np.array([x for js in lists for x in js], np.int64)
    if len(i) == 0:
        return i, j
    gate = np.maximum(CENTROID_GATE, 0.6 * np.maximum(ea[i], eb[j]))
    ok = np.linalg.norm(ca[i] - cb[j], axis=1) <= gate
    return i[ok], j[ok]


def _neighbour_frames(views: dict[int, View]) -> set[tuple[int, int]]:
    """Keyframe pairs whose instances are compared: each keyframe with its ``PAIR_NEIGHBOURS``
    nearest by viewpoint (camera distance over the scene depth, plus 1 − cos of the angle between
    the optical axes; ties by pose), so the work grows linearly with the update. Chains of
    neighbours still link an object's views across the update, and a place revisited from the
    same viewpoint is a neighbour whenever it was captured."""
    keys = sorted(views, key=lambda f: (*views[f].T_map_cam.t.tolist(),
                                        *views[f].T_map_cam.R.reshape(-1).tolist(), f))
    if len(keys) <= PAIR_NEIGHBOURS + 1:
        return {(a, b) for a in keys for b in keys if a != b}
    C = np.array([views[f].T_map_cam.t for f in keys])
    F = np.array([views[f].T_map_cam.R[:, 2] for f in keys])
    depths = [float(np.median(v.depth[v.valid])) for v in views.values() if v.valid.any()]
    scale = max(0.5, float(np.median(depths))) if depths else 2.0
    d = np.linalg.norm(C[:, None] - C[None], axis=2) / scale + (1.0 - F @ F.T)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1, kind="stable")[:, :PAIR_NEIGHBOURS]
    out: set[tuple[int, int]] = set()
    for i, row in enumerate(near):
        for j in row.tolist():
            out.add((keys[i], keys[j]))
            out.add((keys[j], keys[i]))
    return out


def _candidate_pairs(obs: list[Observation], frames: set[tuple[int, int]] | None = None
                     ) -> list[tuple[int, int]]:
    """Instance pairs (i < j) of different keyframes (of neighbouring keyframes when ``frames``
    is given) with compatible labels and nearby centroids."""
    if len(obs) < 2:
        return []
    c = np.array([ob.centroid for ob in obs])
    e = np.array([ob.extent for ob in obs])
    i, j = _near(c, e, c, e)
    frame = np.array([ob.frame for ob in obs])
    compat = _compatible_matrix([ob.label for ob in obs], [ob.label for ob in obs])
    ok = (i < j) & (frame[i] != frame[j]) & compat[i, j]
    return sorted((a, b) for a, b in zip(i[ok].tolist(), j[ok].tolist(), strict=True)
                  if frames is None or (obs[a].frame, obs[b].frame) in frames)


def _candidate_objects(obs: list[Observation], objs: list[MapObject]
                       ) -> list[tuple[int, int]]:
    """(instance, existing object) pairs with compatible labels and the instance's centroid
    within max(CENTROID_GATE, 0.6 · the object's extent) of the object's."""
    have = [j for j, o in enumerate(objs) if len(o.points)]
    if not obs or not have:
        return []
    c = np.array([ob.centroid for ob in obs])
    co = np.array([objs[j].centroid for j in have])
    eo = np.array([_extent(objs[j].points) for j in have])
    i, k = _near(c, np.zeros(len(obs)), co, eo)
    compat = _compatible_matrix([ob.label for ob in obs], [objs[j].label for j in have])
    ok = compat[i, k]
    return sorted(zip(i[ok].tolist(), [have[x] for x in k[ok].tolist()], strict=True))


def _group(obs: list[Observation], objects: list[MapObject], earlier: _Earlier | None = None
           ) -> list[tuple[int | None, list[int]]]:
    """Group this update's instances into objects without regard to the keyframes' order.

    Edges — instance/instance and instance/existing object — are taken strongest first (ties
    by content) and join two groups unless that would put two instances of one keyframe or two
    existing objects in one group, or give an existing object a group most of whose instances do
    not match it directly (a group that grew around a new object is not absorbed through one
    weak link). Returns (existing object id or None, instance indices) per group."""
    n = len(obs)
    objs = list(objects)
    trees: dict[int, cKDTree] = {}
    to_obj: dict[tuple[int, int], float] = {}
    # (-strength, kind, content key, content key / object id, node a, node b); kind 0 = existing
    edges: list[tuple[float, int, tuple[Any, ...], tuple[Any, ...], int, int]] = []
    for i, j in _candidate_objects(obs, objs):
        if j not in trees:
            trees[j] = cKDTree(objs[j].points)
        s = _object_affinity(obs[i], objs[j], trees[j], earlier)
        if s >= IOU_GATE:
            to_obj[(i, j)] = s
            edges.append((-s, 0, obs[i].key, (objs[j].id,), i, n + j))
    views = {ob.frame: ob.view for ob in obs}
    for i, j in _candidate_pairs(obs, _neighbour_frames(views)):
        s = _pair_affinity(obs[i], obs[j])
        if s >= IOU_GATE:
            ka, kb = sorted([obs[i].key, obs[j].key])
            edges.append((-s, 1, ka, kb, i, j))
    edges.sort(key=lambda e: e[:4])

    parent = list(range(n + len(objs)))
    frames: dict[int, set[int]] = {i: {ob.frame} for i, ob in enumerate(obs)}
    frames.update({n + j: set() for j in range(len(objs))})
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    members.update({n + j: [] for j in range(len(objs))})
    anchor: dict[int, int | None] = {i: None for i in range(n)}
    anchor.update({n + j: j for j in range(len(objs))})

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def absorbs(j: int, group: list[int]) -> bool:
        """Existing object ``j`` takes ``group`` only if most of its instances match it."""
        hits = sum((i, j) in to_obj for i in group)
        return 2 * hits >= len(group)

    for *_, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb or frames[ra] & frames[rb]:
            continue
        ja, jb = anchor[ra], anchor[rb]
        if ja is not None and jb is not None:
            continue
        if ja is not None and not absorbs(ja, members[rb]):
            continue
        if jb is not None and not absorbs(jb, members[ra]):
            continue
        parent[rb] = ra
        frames[ra] |= frames.pop(rb)
        members[ra] += members.pop(rb)
        anchor[ra] = ja if ja is not None else jb
        anchor.pop(rb)
    out: list[tuple[int | None, list[int]]] = []
    for r, m in members.items():
        if m:
            k = anchor[r]
            out.append((None if k is None else objs[k].id, sorted(m)))
    return out


# ------------------------------------------------------------------------------------------------
# merging


def containment(a: OBB, b: OBB, pad: float = 0.1, samples: int = 3000) -> float:
    """Fraction of the smaller box (both padded by ``pad``) inside the larger one. Thin objects
    (TVs, pictures, doors) seen from two sides give boxes offset in depth whose IoU is small."""
    pa = OBB(a.center, a.R, a.size + 2 * pad)
    pb = OBB(b.center, b.R, b.size + 2 * pad)
    small, big = (pa, pb) if pa.volume <= pb.volume else (pb, pa)
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (samples, 3)) * small.size
    pts = local @ small.R.T + small.center
    return float(big.contains(pts).mean())


def _scale(o: MapObject) -> float:
    """Size of an object: the largest robust (2-98 %) extent — horizontal diagonal or height — of
    the points its box is fitted to (``fit_points``)."""
    pts = fit_points(o)
    if len(pts) < 2:
        return 0.0
    lo, hi = np.percentile(pts[:, 2], [2, 98])
    return max(_extent(pts), float(hi - lo))


def _depth_explained(a: MapObject, b: MapObject, views: _Views) -> float:
    """>= 1 when two objects of compatible labels that no keyframe detected together are one
    object placed twice by keyframes whose depths disagree (a loop closed by keyframes whose
    depth scale drifted).

    The ``DEPTH_PAIRS`` pairs (sighting of ``a``, sighting of ``b``) whose keyframes are nearest by
    viewpoint are compared. A pair supports the merge when its keyframes' depths disagree by at
    least ``DEPTH_RATIO_MIN`` (``depth_ratio``) and ``b``'s sighting, scaled about its keyframe's
    camera centre by the inverse of that ratio (into ``a``'s keyframe's depth), overlaps ``a``'s
    sighting within the depth-noise tolerance of the box consensus. Returns the supporting pairs
    over a strict majority of the pairs compared."""
    if not compatible(a.label, b.label) or set(a.frames) & set(b.frames):
        return 0.0
    far = max(a.obs_depth, b.obs_depth)
    reach = DEPTH_RATIO_MAX * far + (_extent(a.points) + _extent(b.points)) / 2
    if not a.sightings or not b.sightings or np.linalg.norm(a.centroid - b.centroid) > reach:
        return 0.0
    scale = max(0.5, min(a.obs_depth, b.obs_depth))
    pairs: list[tuple[float, tuple[Any, ...], tuple[Any, ...], Sighting, Sighting]] = []
    for sa in a.sightings:
        Ta = views.pose(sa.frame)
        if Ta is None:
            continue
        for sb in b.sightings:
            Tb = views.pose(sb.frame)
            if Tb is not None:
                d = (float(np.linalg.norm(Ta.t - Tb.t)) / scale
                     + 1.0 - float(Ta.R[:, 2] @ Tb.R[:, 2]))
                pairs.append((d, sa.key(), sb.key(), sa, sb))
    pairs = sorted(pairs, key=lambda p: p[:3])[:DEPTH_PAIRS]
    if not pairs:
        return 0.0
    tol = max(CONSENSUS_TOL_MIN, CONSENSUS_TOL_REL * far)
    support = 0
    for *_, sa, sb in pairs:
        r = views.ratio(sa.frame, sb.frame)
        Tb = views.pose(sb.frame)
        if r is None or Tb is None or abs(np.log(r)) < np.log1p(DEPTH_RATIO_MIN):
            continue
        c = Tb.t
        lo = c + (np.asarray(sb.lo) - c) / r
        hi = c + (np.asarray(sb.hi) - c) / r
        if np.all((np.asarray(sa.lo) <= hi + tol) & (lo <= np.asarray(sa.hi) + tol)):
            support += 1
    return support / (len(pairs) // 2 + 1)


def _one_surface(a: MapObject, b: MapObject) -> float:
    """>= 1 when two objects are pieces of one horizontal surface seen from different keyframes
    (a counter top around the camera, labelled desk from one side and bed from another): some
    labels of the two (their label or any label they were detected as) are pieces of one surface
    (``split_surface``), no keyframe detected both (it would have seen two things there), their
    median heights agree within ``SURFACE_HEIGHT_TOL`` and they share surface: the points of
    ``a`` within ``SURFACE_PLAN_M`` horizontally and ``SURFACE_HEIGHT_TOL`` vertically of
    ``b``'s, over ``SURFACE_SHARED_POINTS``."""
    la, lb = {a.label, *a.label_votes}, {b.label, *b.label_votes}
    if not any(split_surface(x, y) for x in sorted(la) for y in sorted(lb)):
        return 0.0
    if set(a.frames) & set(b.frames) or len(a.points) == 0 or len(b.points) == 0:
        return 0.0
    if abs(float(np.median(a.points[:, 2]) - np.median(b.points[:, 2]))) > SURFACE_HEIGHT_TOL:
        return 0.0
    k = np.array([1.0, 1.0, SURFACE_PLAN_M / SURFACE_HEIGHT_TOL])
    d, _ = cKDTree(b.points * k).query(a.points * k, k=1, distance_upper_bound=SURFACE_PLAN_M)
    return float(np.isfinite(d).sum()) / SURFACE_SHARED_POINTS


def _merge_strength(a: MapObject, b: MapObject, views: _Views | None = None) -> float:
    """>= 1 when two objects are one physical object. Compatible labels: >= 50 % of the smaller
    one's points on the other, or boxes overlapping (IoU >= 0.3, or the smaller padded box >= 60 %
    inside the other), or (with ``views``) copies placed by keyframes whose depths disagree
    (``_depth_explained``). Incompatible labels (the detector's label flickered between
    keyframes): no keyframe detected both (it would have seen two things there), their sizes are
    comparable (``MERGE_SCALE``: neither is a part of the other or an item resting on it) and
    >= 50 % of either one's points lie on the other's surface. Whatever the labels: pieces of one
    horizontal surface (``_one_surface``). The value orders the merges (strongest first)."""
    same_kind = compatible(a.label, b.label)
    if not same_kind and set(a.frames) & set(b.frames):
        return 0.0
    if np.linalg.norm(a.centroid - b.centroid) > max(CENTROID_GATE * 2,
                                                     _extent(a.points) + _extent(b.points)):
        return 0.0
    surface = _one_surface(a, b)
    if not same_kind:
        sa, sb = _scale(a), _scale(b)
        if min(sa, sb) < MERGE_SCALE * max(sa, sb):
            return surface
        radius = max(0.05, 0.02 * min(a.obs_depth, b.obs_depth))
        return max(surface, overlap_fraction(a.points, b.points, radius) / MERGE_OVERLAP,
                   overlap_fraction(b.points, a.points, radius) / MERGE_OVERLAP)
    small, big = (a, b) if len(a.points) <= len(b.points) else (b, a)
    radius = max(0.05, 0.02 * small.obs_depth)
    s = max(surface, overlap_fraction(small.points, big.points, radius) / MERGE_OVERLAP)
    if a.obb is not None and b.obb is not None:
        s = max(s, obb_iou_upright(a.obb, b.obb, samples=3000) / MERGE_BOX_IOU,
                containment(a.obb, b.obb) / MERGE_CONTAINMENT)
    if s < 1.0 and views is not None:
        s = max(s, _depth_explained(a, b, views))
    return s


def _content_key(o: MapObject) -> tuple[Any, ...]:
    return (o.label, *o.centroid.tolist(), len(o.points))


def _merge(state: ObjectState, touched: set[int], alias: dict[int, int],
           views: _Views | None = None) -> int:
    """Merge duplicates among the objects (at least one of each pair touched by this update),
    strongest pair first; the lower id is kept and ``alias`` maps each merged id to its keeper.
    ``views`` (the map's keyframes) enables the depth-explained test (``_depth_explained``)."""
    strength: dict[tuple[int, int], float] = {}

    def pairs_of(o: MapObject) -> None:
        for p in state.objects:
            if p.id == o.id or not (o.id in touched or p.id in touched):
                continue
            a, b = (o, p) if o.id < p.id else (p, o)
            s = _merge_strength(a, b, views)
            if s >= 1.0:
                strength[(a.id, b.id)] = s
            else:
                strength.pop((a.id, b.id), None)

    by = state.by_id()
    for o in sorted(state.objects, key=lambda o: o.id):
        if o.id in touched:
            pairs_of(o)
    merged = 0
    while strength:
        (ka, kb), _ = min(strength.items(), key=lambda kv: (
            -kv[1], sorted([_content_key(by[kv[0][0]]), _content_key(by[kv[0][1]])])))
        keep, gone = by[ka], by[kb]  # ka < kb: the lower id is kept
        keep.absorb(gone)
        refit(keep, state.floor_z)
        alias[gone.id] = keep.id
        state.objects = [o for o in state.objects if o.id != gone.id]
        del by[gone.id]
        touched.add(keep.id)
        touched.discard(gone.id)
        strength = {k: v for k, v in strength.items() if gone.id not in k and keep.id not in k}
        pairs_of(keep)
        merged += 1
    return merged


# ------------------------------------------------------------------------------------------------
# absence


def _absence(candidates: list[MapObject], new_views: dict[int, tuple[Any, View]]) -> list[int]:
    """Objects of earlier updates that this update, as a whole, shows to be gone.

    Each well-registered keyframe of the update that sees enough of an object's surface (in view,
    unoccluded; out of view or occluded is unknown, never free) gives one verdict: through when
    >= 60 % of the samples are seen through, in place when <= 40 % are. The update contradicts
    the object when most of those keyframes see through it. A contradiction by >= 3 keyframes
    removes the object; by fewer, it removes an established object (>= 4 detections) only when
    every one of them sees >= 80 % through from a well-supported pose, else it is a strike and a
    second strike (from a later update) removes it. An update that sees the object in place
    clears its strikes (latest wins)."""
    removed = []
    for o in candidates:
        verdicts: list[tuple[float, bool]] = []  # (fraction seen through, pose well supported)
        for nf, view in new_views.values():
            if not well_registered(nf.record.stats, nf.record.pose_source):
                continue
            through, consistent = visibility_evidence(view, o.points)
            n = through + consistent
            if n < MIN_VISIBLE_SAMPLES:
                continue
            supported = (nf.record.stats.get("observations", FEW_MIN_INLIERS) >= FEW_MIN_INLIERS
                         or nf.record.pose_source in ("identity", "multiview"))
            verdicts.append((through / n, supported))
        if not verdicts:
            continue
        strong = [(f, ok) for f, ok in verdicts if f >= REMOVE_FRACTION]
        in_place = [f for f, _ in verdicts if f <= 1.0 - REMOVE_FRACTION]
        if 2 * len(strong) <= len(verdicts):
            if 2 * len(in_place) > len(verdicts):
                o.strikes = 0
            continue
        few_rule = (all(f >= REMOVE_FRACTION_FEW and ok for f, ok in strong)
                    and o.observations >= FEW_MIN_OBSERVATIONS)
        if len(strong) >= REMOVE_MIN_FRAMES or few_rule:
            removed.append(o.id)
        else:
            o.strikes += 1
            if o.strikes >= 2:
                removed.append(o.id)
    return removed


def label_map_for(reader_instances: list[dict[str, Any]], shape: tuple[int, int],
                  state: ObjectState) -> NDArray[np.int32]:
    """Per-pixel persistent object ids of a keyframe (merged ids resolved, removed → 0)."""
    lab = np.zeros(shape, np.int32)
    for inst in reader_instances:
        oid = state.resolve(int(inst["object_id"]))
        if oid is None:
            continue
        m = rle.decode(inst["mask"])
        if m.shape == shape:
            lab[m] = oid
    return lab

