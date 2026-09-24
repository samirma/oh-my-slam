"""Persistent objects: identity across keyframes and updates, evidence accumulation, OBB refits
(through ``segmentation.obb``), confirmation, merging and removal on evidence of absence.

Ids come from ``next_object_id`` and are never reused; a merged or removed id disappears for
good (merges are recorded in ``merged_into`` so old per-frame instance files still resolve).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from oh_my_slam.core import rle
from oh_my_slam.core.geometry import voxel_downsample_indices
from oh_my_slam.core.images import load_png
from oh_my_slam.mapping.store import OBJECTS_JSON
from oh_my_slam.mapping.validity import View, tau, well_registered
from oh_my_slam.reconstruction.gravity import floor_candidate_height
from oh_my_slam.segmentation.api import SceneObject, lift_detections
from oh_my_slam.segmentation.detect import compatible, floor_gap
from oh_my_slam.segmentation.obb import OBB, fit_obb, obb_iou_upright

POINT_CAP = 30000
POINT_VOXEL = 0.01
IOU_GATE = 0.3
CENTROID_GATE = 1.5
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
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class MapObject:
    id: int
    label: str
    label_votes: dict[str, float]
    scores: list[float]
    points: NDArray[np.float32]
    obb: OBB | None = None
    observations: int = 0
    frames: list[int] = field(default_factory=list)
    confirmed: bool = False
    strikes: int = 0
    views_in_frustum: int = 0
    created_update: int = 0
    last_seen_update: int = 0
    obs_depth: float = 2.0
    pixel_count: int = 0

    @property
    def score(self) -> float:
        top = sorted(self.scores, reverse=True)[:3]
        return float(np.mean(top)) if top else 0.0

    @property
    def centroid(self) -> NDArray[np.float64]:
        return self.points.mean(0).astype(np.float64) if len(self.points) else np.zeros(3)

    def add_points(self, pts: NDArray[Any], seed: int = 0) -> None:
        allp = np.concatenate([self.points, pts.astype(np.float32)]) if len(self.points) else \
            pts.astype(np.float32)
        idx = voxel_downsample_indices(allp, POINT_VOXEL, keep="last")
        allp = allp[idx]
        if len(allp) > POINT_CAP:
            rng = np.random.default_rng(seed + self.id)
            allp = allp[np.sort(rng.choice(len(allp), POINT_CAP, replace=False))]
        self.points = allp

    def vote(self, label: str, score: float) -> None:
        self.label_votes[label] = self.label_votes.get(label, 0.0) + score
        self.label = max(self.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.scores = sorted([*self.scores, float(score)], reverse=True)[:10]

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
            observations=int(d.get("observations", 0)), frames=list(d.get("frames", [])),
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
            observations=self.observations, confirmed=self.confirmed,
            frames=sorted(set(self.frames)),
        )


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


def projected_mask(view: View, pts: NDArray[Any], dilate: int = 3) -> NDArray[np.bool_]:
    """Pixels of ``view`` covered by the visible (unoccluded) points of an object."""
    h, w = view.depth.shape
    mask = np.zeros((h, w), bool)
    if len(pts) == 0:
        return mask
    inside, z, d = view.lookup(pts)
    vis = inside & (z <= d + tau(z))
    if not vis.any():
        return mask
    pc = view.T_map_cam.inverse().apply(pts[vis])
    K = view.K
    u = np.clip(np.rint(K.fx * pc[:, 0] / pc[:, 2] + K.cx).astype(int), 0, w - 1)
    v = np.clip(np.rint(K.fy * pc[:, 1] / pc[:, 2] + K.cy).astype(int), 0, h - 1)
    mask[v, u] = True
    return ndimage.binary_dilation(mask, iterations=dilate) if dilate else mask


def overlap_fraction(a: NDArray[Any], b: NDArray[Any], radius: float) -> float:
    """Fraction of points of ``a`` within ``radius`` of ``b``."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    sa = a if len(a) <= 4000 else a[np.linspace(0, len(a) - 1, 4000).astype(int)]
    d, _ = cKDTree(b).query(sa, k=1, distance_upper_bound=radius)
    return float(np.isfinite(d).mean())


def _indices(ctx: Any) -> set[int]:
    return {nf.kf.index for nf in ctx.new if nf.record is not None}


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
    obj.obb = fit_obb(obj.points, UP, floor_z, floor_gap(obj.label))


# ------------------------------------------------------------------------------------------------
# the update


def map_floor(ctx: Any, per_frame: int | None = None) -> tuple[NDArray[np.float64], float | None]:
    """Map-frame grid points of this update's confidently placed keyframes (about ``per_frame``
    of each, all when None) and the floor height among them: the lowest well-supported
    horizontal surface. Maps accumulate stray points below the floor (see-through shelves,
    reflections, drift), so up to 10 % of the points may lie beneath it (single images: 3 %)."""
    pts = []
    for nf in ctx.new:
        if nf.record is None or nf.depth is None or nf.record.low_confidence:
            continue
        view = View(nf.depth, nf.frame.valid & (nf.depth > 0), nf.record.K_grid,
                    nf.record.T_map_cam)
        p, _, _ = view.grid_points()
        pts.append(p if per_frame is None else p[:: max(1, len(p) // per_frame)])
    if not pts:
        return np.zeros((0, 3)), None
    allp = np.concatenate(pts)
    return allp, floor_candidate_height(allp[:, 2], max_below=MAP_FLOOR_MAX_BELOW)


def _instances_json(frame_items: list[tuple[int, Any]]) -> dict[str, Any]:
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
    touched: set[int] = set()
    new_views: list[tuple[Any, View]] = []
    n_inst = 0
    for nf in ctx.new:
        if nf.record is None or nf.depth is None:
            continue
        rec = nf.record
        valid = nf.frame.valid & (nf.depth > 0)
        view = View(nf.depth, valid, rec.K_grid, rec.T_map_cam)
        new_views.append((nf, view))
        insts = lift_detections(nf.frame, nf.dets, rec.T_map_cam, depth=nf.depth, valid=valid)
        n_inst += len(insts)
        items = _associate(state, insts, view, rec, uid, touched)
        tx.write_json(f"per_frame/{rec.name}/instances.json", _instances_json(items))
    for o in state.objects:  # provisional boxes: the merge test compares boxes of new objects too
        if o.id in touched:
            refit(o, state.floor_z)
    merged = _merge(state, touched)
    before = {o.id: o.observations - sum(1 for f in o.frames if f in _indices(ctx))
              for o in state.objects}
    for o in state.objects:
        # keyframes of this update whose view contains the object's (unoccluded) centroid,
        # counted over the whole update — also frames before the object was first detected
        seen_now = o.observations - before.get(o.id, 0)
        visible = sum(centroid_visible(v, o.centroid) for _, v in new_views)
        o.views_in_frustum += max(visible, seen_now)
        if o.id in touched:
            refit(o, state.floor_z)
            o.confirmed = o.confirmed or o.observations >= min(3, max(1, o.views_in_frustum))
    spurious = [o for o in state.objects
                if (not o.confirmed and o.views_in_frustum - o.observations >= 3)
                or below_floor(o, state.floor_z)]
    removed = _absence(state, [o for o in state.objects if o.id in old_ids and o.id not in
                               touched], new_views)
    drop = {o.id for o in spurious} | set(removed)
    for oid in drop:
        tx.delete(points_file(oid))
    state.objects = [o for o in state.objects if o.id not in drop]
    state.observed = {o.id for o in state.objects if o.id in touched}
    for o in state.objects:
        if o.id in touched:
            tx.save_npy(points_file(o.id), o.points.astype(np.float32))
    tx.write_json(OBJECTS_JSON, {
        "next_id": state.next_id,
        "floor_z": state.floor_z,
        "merged_into": {str(k): v for k, v in state.merged_into.items()},
        "objects": [o.to_dict() for o in sorted(state.objects, key=lambda o: o.id)],
    })
    state.summary = {
        "instances": n_inst, "touched": len(touched), "merged": merged,
        "removed": sorted(removed), "spurious_dropped": len(spurious),
        "total": len(state.objects), "confirmed": sum(o.confirmed for o in state.objects),
    }
    progress(f"objects: {state.summary['confirmed']} confirmed of {len(state.objects)}; "
             f"{len(removed)} removed, {merged} merged")
    return state


def _associate(state: ObjectState, insts: list[Any], view: View, rec: Any, uid: int,
               touched: set[int]) -> list[tuple[int, Any]]:
    """Hungarian assignment of one keyframe's instances to existing objects."""
    objs = state.objects
    items: list[tuple[int, Any]] = []
    if insts and objs:
        cost = np.ones((len(insts), len(objs)))
        cents = [o.centroid for o in objs]
        for i, inst in enumerate(insts):
            ci = inst.lifted.points.mean(0)
            radius = max(0.05, 0.02 * float(np.median(np.linalg.norm(
                inst.lifted.points - rec.T_map_cam.t, axis=1))))
            for j, o in enumerate(objs):
                if not compatible(inst.detection.label, o.label):
                    continue
                if np.linalg.norm(ci - cents[j]) > max(CENTROID_GATE, 0.6 * _extent(o)):
                    continue
                pm = projected_mask(view, o.points)
                inter = np.logical_and(pm, inst.mask).sum()
                iou = inter / max(1, np.logical_or(pm, inst.mask).sum())
                ov = overlap_fraction(inst.lifted.points, o.points, radius)
                s = max(float(iou), ov)
                if s >= IOU_GATE:
                    cost[i, j] = 1.0 - s
        rows, cols = linear_sum_assignment(cost)
        matched = {r: c for r, c in zip(rows, cols, strict=True) if cost[r, c] < 1.0 - IOU_GATE
                   + 1e-9}
    else:
        matched = {}
    for i, inst in enumerate(insts):
        if i in matched:
            o = objs[matched[i]]
        else:
            o = MapObject(state.next_id, inst.detection.label, {}, [],
                          np.zeros((0, 3), np.float32), created_update=uid)
            state.next_id += 1
            state.objects.append(o)
        o.vote(inst.detection.label, inst.detection.score)
        o.add_points(inst.lifted.points)
        o.observations += 1
        o.frames.append(rec.index)
        o.last_seen_update = uid
        o.pixel_count = max(o.pixel_count, int(inst.mask.sum()))
        o.obs_depth = float(np.median(np.linalg.norm(inst.lifted.points - rec.T_map_cam.t,
                                                     axis=1)))
        touched.add(o.id)
        items.append((o.id, inst))
    return items


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


def _extent(o: MapObject) -> float:
    """Horizontal diagonal of the object's points (large objects are seen piecewise)."""
    if len(o.points) < 2:
        return 0.0
    lo, hi = np.percentile(o.points[:, :2], [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def _merge(state: ObjectState, touched: set[int]) -> int:
    """Merge compatible objects sharing >= 50 % of the smaller one's points (lower id kept)."""
    merged = 0
    changed = True
    while changed:
        changed = False
        objs = sorted(state.objects, key=lambda o: o.id)
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                if not (a.id in touched or b.id in touched):
                    continue
                if not compatible(a.label, b.label):
                    continue
                if np.linalg.norm(a.centroid - b.centroid) > max(CENTROID_GATE * 2,
                                                                 _extent(a) + _extent(b)):
                    continue
                small, big = (a, b) if len(a.points) <= len(b.points) else (b, a)
                radius = max(0.05, 0.02 * small.obs_depth)
                boxes_overlap = (a.obb is not None and b.obb is not None
                                 and (obb_iou_upright(a.obb, b.obb, samples=3000) >= MERGE_BOX_IOU
                                      or containment(a.obb, b.obb) >= MERGE_CONTAINMENT))
                if overlap_fraction(small.points, big.points, radius) < MERGE_OVERLAP and \
                        not boxes_overlap:
                    continue
                keep, gone = (a, b)  # a has the lower id
                for lab, v in gone.label_votes.items():
                    keep.label_votes[lab] = keep.label_votes.get(lab, 0.0) + v
                keep.label = max(keep.label_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
                keep.scores = sorted(keep.scores + gone.scores, reverse=True)[:10]
                keep.add_points(gone.points)
                keep.observations += gone.observations
                keep.frames = sorted(set(keep.frames + gone.frames))
                keep.views_in_frustum = max(keep.views_in_frustum, gone.views_in_frustum)
                keep.confirmed = keep.confirmed or gone.confirmed
                keep.created_update = min(keep.created_update, gone.created_update)
                keep.last_seen_update = max(keep.last_seen_update, gone.last_seen_update)
                keep.pixel_count = max(keep.pixel_count, gone.pixel_count)
                state.merged_into[gone.id] = keep.id
                state.objects = [o for o in state.objects if o.id != gone.id]
                touched.add(keep.id)
                touched.discard(gone.id)
                merged += 1
                changed = True
                break
            if changed:
                break
    return merged


def _absence(state: ObjectState, candidates: list[MapObject],
             new_views: list[tuple[Any, View]]) -> list[int]:
    """Objects to remove: their surface is seen through by this update's keyframes."""
    removed = []
    for o in candidates:
        strong_frames, weak = 0, False
        few_ok = []
        for nf, view in new_views:
            if not well_registered(nf.record.stats, nf.record.pose_source):
                continue
            through, consistent = visibility_evidence(view, o.points)
            n = through + consistent
            if n < MIN_VISIBLE_SAMPLES:
                continue  # out of view or occluded: unknown, never free
            frac = through / n
            if frac >= REMOVE_FRACTION:
                strong_frames += 1
                weak = True
                few_ok.append(frac >= REMOVE_FRACTION_FEW and (
                    nf.record.stats.get("observations", FEW_MIN_INLIERS) >= FEW_MIN_INLIERS
                    or nf.record.pose_source in ("identity", "multiview")))
        # 1-2 keyframes may remove only well-established objects; weakly supported ones (seen in
        # <= 3 keyframes, e.g. "ghosts" seen through shelving) only get a strike from so little
        few_rule = bool(few_ok) and all(few_ok) and o.observations >= FEW_MIN_OBSERVATIONS
        if strong_frames >= REMOVE_MIN_FRAMES or few_rule:
            removed.append(o.id)
        elif weak:
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


def load_valid(path: Any, depth: NDArray[Any]) -> NDArray[np.bool_]:
    return (load_png(path) > 0) if path.exists() else depth > 0
