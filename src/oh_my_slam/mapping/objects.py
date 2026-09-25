"""Persistent objects: identity across keyframes and updates, evidence accumulation, OBB refits
(through the segmentation API), confirmation, merging and removal on evidence of absence.

Semantics (spec §2.3):

* **One update, one observation.** The order of an update's keyframes never matters. All of its
  instances are grouped at once (``_group``): strongest agreement first, never two instances of
  one keyframe in one group, and an object the map already held absorbs a group only when most of
  the group's instances match it directly. A group without such an object becomes a new object.
* **Evidence is order-free.** An object's evidence — label votes, scores, the keyframes that
  detected it, its mean viewing distance and a canonical point set (``canonical_points``) — is a
  pure function of the instances it received, so its label and OBB do not depend on the order or
  the grouping into updates in which that evidence arrived.
* **Confirmation** is recomputed from all evidence: detected in >= min(3, V) keyframes, where V
  counts the map's keyframes whose view contains the object's unoccluded centroid or that detected
  it (also keyframes of earlier updates, for an object first detected now). Unconfirmed objects are
  kept, never exported, so a later update can still confirm them.
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from oh_my_slam.core import rle
from oh_my_slam.core.geometry import voxel_keys
from oh_my_slam.mapping.store import OBJECTS_JSON, frame_file
from oh_my_slam.mapping.validity import View, keyframe_view, stored_view, tau, well_registered
from oh_my_slam.reconstruction.gravity import floor_candidate_height
from oh_my_slam.segmentation.api import (
    OBB,
    LiftedInstance,
    SceneObject,
    compatible,
    fit_object_obb,
    lift_detections,
    obb_iou_upright,
)

POINT_CAP = 30000
POINT_VOXEL = 0.01
IOU_GATE = 0.3
CENTROID_GATE = 1.5
CONFIRM_DETECTIONS = 3
MERGE_OVERLAP = 0.5
MERGE_BOX_IOU = 0.3  # compatible objects whose boxes overlap this much are one physical object
MERGE_CONTAINMENT = 0.6
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
    views_in_frustum: int = 0
    created_update: int = 0
    last_seen_update: int = 0
    obs_depth: float = 2.0  # mean camera distance of its detections
    pixel_count: int = 0

    @property
    def observations(self) -> int:
        """Keyframes in which the object was detected."""
        return len(self.frames)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "label_votes": self.label_votes,
            "scores": self.scores, "score": self.score,
            "obb": None if self.obb is None else self.obb.to_dict(),
            "observations": self.observations, "frames": self.frames,
            "confirmed": self.confirmed, "strikes": self.strikes,
            "views_in_frustum": self.views_in_frustum, "created_update": self.created_update,
            "last_seen_update": self.last_seen_update, "obs_depth": self.obs_depth,
            "pixel_count": self.pixel_count, "point_file": points_file(self.id),
        }

    @staticmethod
    def from_dict(d: dict[str, Any], points: NDArray[np.float32]) -> MapObject:
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
        )

    def scene_object(self) -> SceneObject:
        assert self.obb is not None
        return SceneObject(
            id=self.id, label=self.label, score=self.score, obb=self.obb,
            pixel_count=self.pixel_count, point_count=len(self.points),
            observations=self.observations, confirmed=self.confirmed, frames=list(self.frames),
        )


@dataclass
class Observation:
    """One lifted instance of a keyframe of this update, in map coordinates."""

    frame: int  # keyframe index
    view: View
    inst: LiftedInstance
    points: NDArray[np.float32]
    centroid: NDArray[np.float64]
    depth: float  # median camera distance
    extent: float
    _tree: cKDTree | None = field(default=None, repr=False)

    @staticmethod
    def of(frame: int, view: View, inst: LiftedInstance) -> Observation:
        pts = np.asarray(inst.lifted.points, np.float64)
        dist = float(np.median(np.linalg.norm(pts - view.T_map_cam.t, axis=1)))
        return Observation(frame, view, inst, pts.astype(np.float32), pts.mean(0), dist,
                           _extent(pts))

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
        return [o.scene_object() for o in sorted(self.objects, key=lambda o: o.id)
                if o.confirmed and o.obb is not None]


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


def centroid_visible(view: View, c: NDArray[Any]) -> bool:
    inside, z, d = view.lookup(c[None])
    return bool(inside[0] and z[0] <= d[0] + tau(z[0]))


def below_floor(obj: MapObject, floor_z: float | None, margin: float = 0.3) -> bool:
    """An object entirely below the map floor is not physical (e.g. seen in a mirror or through
    a reflective screen, where monocular depth places it behind the glass)."""
    if floor_z is None or len(obj.points) < 10:
        return False
    return float(np.percentile(obj.points[:, 2], 90)) < floor_z - margin


def refit(obj: MapObject, floor_z: float | None) -> None:
    if len(obj.points) < 10:
        return
    obj.obb = fit_object_obb(obj.points, obj.label, UP, floor_z)


def confirm(obj: MapObject) -> None:
    """Confirmed when detected in >= min(3, V) keyframes (V: ``views_in_frustum``)."""
    obj.confirmed = obj.observations >= min(CONFIRM_DETECTIONS, max(1, obj.views_in_frustum))


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
    #    the map's count
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
        per_frame[rec.index] = list(range(len(obs), len(obs) + len(insts)))
        obs.extend(Observation.of(rec.index, view, inst) for inst in insts)
        rank = {id(det): j for j, det in enumerate(nf.dets)}
        number.extend(first_number[nf.kf.index] + rank[id(inst.detection)] for inst in insts)

    # 2. group them (order-free) and fold each group into an existing or a new object; new
    #    objects get provisional ids >= first_new (content order) until their final numbering
    owner: list[int] = [0] * len(obs)  # object id per observation
    touched: set[int] = set()
    groups = _group(obs, state.objects)
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

    # 3. merge duplicates (lower id kept; keepers are refitted), then count views and confirm
    alias: dict[int, int] = {}
    merged = _merge(state, touched, alias)
    old_views = _OldViews(ctx)
    for o in state.objects:
        c = o.centroid
        seen = {f for f, (_, v) in new_views.items() if centroid_visible(v, c)}
        seen |= set(o.frames) & set(new_views)
        o.views_in_frustum += len(seen)
        if o.id >= first_new:  # first detected now: earlier keyframes that saw its place count
            o.views_in_frustum += old_views.count_visible(c)
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
        items = [(final_id(owner[i]), obs[i].inst) for i in idxs]
        tx.write_json(frame_file(nf.record.name, "instances.json"), _instances_json(items))
    state.observed = touched
    for o in state.objects:
        if o.id in touched:
            tx.save_npy(points_file(o.id), o.points.astype(np.float32))
    tx.write_json(OBJECTS_JSON, {
        "next_id": state.next_id,
        "floor_z": state.floor_z,
        "merged_into": {str(k): v for k, v in sorted(state.merged_into.items())},
        "objects": [o.to_dict() for o in sorted(state.objects, key=lambda o: o.id)],
    })
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


# ------------------------------------------------------------------------------------------------
# association


def _object_affinity(ob: Observation, o: MapObject, tree: cKDTree) -> float:
    """How well an instance matches an existing object: max(projected-mask IoU, share of the
    instance's points on the object's)."""
    iou = projected_iou(ob.view, ob.inst.mask, o.points)
    return max(iou, overlap_fraction(ob.points, o.points, ob.radius, tree))


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


def _group(obs: list[Observation], objects: list[MapObject]
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
        s = _object_affinity(obs[i], objs[j], trees[j])
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


def _merge_strength(a: MapObject, b: MapObject) -> float:
    """>= 1 when two objects are one physical object: compatible labels and >= 50 % of the
    smaller one's points on the other, or boxes overlapping (IoU >= 0.3, or the smaller padded box
    >= 60 % inside the other). The value orders the merges (strongest first)."""
    if not compatible(a.label, b.label):
        return 0.0
    if np.linalg.norm(a.centroid - b.centroid) > max(CENTROID_GATE * 2,
                                                     _extent(a.points) + _extent(b.points)):
        return 0.0
    small, big = (a, b) if len(a.points) <= len(b.points) else (b, a)
    radius = max(0.05, 0.02 * small.obs_depth)
    s = overlap_fraction(small.points, big.points, radius) / MERGE_OVERLAP
    if a.obb is not None and b.obb is not None:
        s = max(s, obb_iou_upright(a.obb, b.obb, samples=3000) / MERGE_BOX_IOU,
                containment(a.obb, b.obb) / MERGE_CONTAINMENT)
    return s


def _content_key(o: MapObject) -> tuple[Any, ...]:
    return (o.label, *o.centroid.tolist(), len(o.points))


def _merge(state: ObjectState, touched: set[int], alias: dict[int, int]) -> int:
    """Merge duplicates among the objects (at least one of each pair touched by this update),
    strongest pair first; the lower id is kept and ``alias`` maps each merged id to its keeper."""
    strength: dict[tuple[int, int], float] = {}

    def pairs_of(o: MapObject) -> None:
        for p in state.objects:
            if p.id == o.id or not (o.id in touched or p.id in touched):
                continue
            a, b = (o, p) if o.id < p.id else (p, o)
            s = _merge_strength(a, b)
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


class _OldViews:
    """Keyframes of earlier updates, loaded lazily (only where a point projects into them)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.cache: dict[str, View | None] = {}

    def _view(self, rec: Any) -> View | None:
        if rec.name not in self.cache:
            self.cache[rec.name] = stored_view(self.ctx.tx.current, rec)
        return self.cache[rec.name]

    def count_visible(self, c: NDArray[Any]) -> int:
        n = 0
        for rec in self.ctx.old_frames:
            K = rec.K_grid
            pc = rec.T_map_cam.inverse().apply(np.asarray(c, np.float64)[None])[0]
            if pc[2] <= 0:
                continue
            u, v = K.fx * pc[0] / pc[2] + K.cx, K.fy * pc[1] / pc[2] + K.cy
            if not (0 <= u < K.width and 0 <= v < K.height):
                continue
            view = self._view(rec)
            n += int(view is not None and centroid_visible(view, c))
        return n


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

