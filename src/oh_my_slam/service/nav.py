"""The navigation and locate reads (O5, O6): ``POST locate`` and ``GET pose`` (api.md §6),
``GET grid`` and ``GET points`` (map.md §6-7), from the versions ``derived`` keeps.

Region, passage and frontier ids are stable across versions and requests: each map keeps a
registry (``derived/ids.json``) of the features it has answered, and a new feature takes the id of
the registered one of its kind that it matches (centre within ``ID_MATCH_M``, and for a region an
area within a factor ``ID_AREA_RATIO``), else the next free id."""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.mapping import navgrid, relocalize
from oh_my_slam.service import checks
from oh_my_slam.service.cameras import Camera, decode_mask, decode_rgb, same_camera
from oh_my_slam.service.derived import Derived, DerivedVersion
from oh_my_slam.service.errors import ApiError, invalid
from oh_my_slam.service.store import Store, utc

DEFAULT_CELL_M = 0.05
MIN_CELL_M = 0.01
MAX_CELL_M = 1.0
MAX_GRID_CELLS = 16_000_000
DEFAULT_Z_BAND = (0.05, 1.8)
DEFAULT_VOXEL_M = 0.01
DEFAULT_MAX_POINTS = 200_000
MAX_POINTS = 2_000_000
ID_MATCH_M = {"regions": 0.75, "passages": 0.3, "frontiers": 0.5}
ID_AREA_RATIO = 2.0


def binary(arr: NDArray[Any]) -> dict[str, Any]:
    """A ``Binary``: little-endian, row-major, base64."""
    a = np.ascontiguousarray(arr)
    dt = a.dtype.newbyteorder("<")
    return {"dtype": str(a.dtype.name), "shape": list(a.shape),
            "data": base64.b64encode(a.astype(dt).tobytes()).decode("ascii")}


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


# -- query parameters ----------------------------------------------------------------------------


def number(params: Any, name: str, default: float | None, lo: float | None = None,
           hi: float | None = None) -> float | None:
    raw = params.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        v = math.nan
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f" between {lo} and {hi}" if lo is not None and hi is not None else ""
        raise invalid(name, f"{name} must be a number{rng}, got {raw!r}")
    return v


def vec3(params: Any, name: str) -> NDArray[np.float64]:
    raw = params.get(name)
    if raw is None:
        raise invalid(name, f"{name} is required: x,y,z in metres (map frame)")
    try:
        v = np.array([float(x) for x in raw.split(",")], np.float64)
    except ValueError:
        v = np.zeros(0)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise invalid(name, f"{name} must be x,y,z in metres, got {raw!r}")
    return v


# -- stable ids ----------------------------------------------------------------------------------


class Ids:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._lock = threading.Lock()

    def assign(self, name: str, kind: str, items: list[tuple[NDArray[np.float64], float]]
               ) -> list[int]:
        """Ids of the features ``items`` (centre, area or 0) of one kind."""
        path = self.store.dir(name) / "derived" / "ids.json"
        with self._lock:
            try:
                reg = json.loads(path.read_text())
            except (OSError, ValueError):
                reg = {}
            known = reg.setdefault(kind, [])
            nxt = int(reg.get(f"next_{kind}", 1))
            used: set[int] = set()
            out = [0] * len(items)
            cands = []
            for i, (c, area) in enumerate(items):
                for e in known:
                    d = math.hypot(c[0] - e["c"][0], c[1] - e["c"][1])
                    if d > ID_MATCH_M[kind]:
                        continue
                    if kind == "regions" and not (1 / ID_AREA_RATIO <= area / max(e["a"], 1e-9)
                                                  <= ID_AREA_RATIO):
                        continue
                    cands.append((d, i, e["id"]))
            for _, i, eid in sorted(cands):
                if out[i] or eid in used:
                    continue
                out[i] = eid
                used.add(eid)
            for i, (c, area) in enumerate(items):
                if not out[i]:
                    out[i] = nxt
                    nxt += 1
                    known.append({"id": out[i], "c": [0.0, 0.0], "a": 0.0})
                for e in known:
                    if e["id"] == out[i]:
                        e["c"], e["a"] = [_r(c[0]), _r(c[1])], _r(area)
            reg[f"next_{kind}"] = nxt
            atomic_write_json(path, reg)
        return out


# -- grid ----------------------------------------------------------------------------------------


@dataclass
class GridAnswer:
    body: dict[str, Any]
    layers: navgrid.Layers
    features: navgrid.Features
    region_ids: list[int]


def grid(dv: DerivedVersion, ids: Ids, cell: float, z_min: float, z_max: float,
         footprint: float | None) -> GridAnswer:
    vol = dv.volume()
    floor = dv.floor_z
    base = navgrid.layers(vol, floor, z_min, z_max)  # the features at the volume's resolution,
    walls = navgrid.inflate_walls(base)  # whatever cell size is asked, on the walls as seen
    navgrid.mark_stands(base, dv.stands, floor, footprint)
    navgrid.mark_stands(walls, dv.stands, floor, footprint)
    lay = navgrid.resample(walls, cell)
    if lay.state.size > MAX_GRID_CELLS:
        raise invalid("cell_m", f"cell_m {cell} gives {lay.state.size} cells (at most "
                      f"{MAX_GRID_CELLS}): ask for larger cells")
    if lay is not walls:
        navgrid.mark_stands(lay, dv.stands, floor, footprint)
    feat = navgrid.features(base, footprint, dv.meta.get("doors", []))
    rid = ids.assign(dv.name, "regions", [(r.centroid, r.area_m2) for r in feat.regions])
    pid = ids.assign(dv.name, "passages", [(p.centre, 0.0) for p in feat.passages])
    fid = ids.assign(dv.name, "frontiers", [(f.centre, 0.0) for f in feat.frontiers])
    regions = [{"id": i, "area_m2": r.area_m2, "coverage_share": r.coverage_share,
                "polygon_m": r.polygon} for i, r in zip(rid, feat.regions, strict=True)]
    passages = [{"id": i, "centre_m": [_r(p.centre[0]), _r(p.centre[1])],
                 "width_m": _r(p.width), "normal_deg": _r(p.normal_deg, 2),
                 "regions": [rid[p.regions[0]], rid[p.regions[1]]], "kind": p.kind,
                 "jambs_m": [[_r(j[0]), _r(j[1])] for j in p.jambs], "leaf": p.leaf}
                for i, p in zip(pid, feat.passages, strict=True)]
    frontiers = [{"id": i, "centre_m": [_r(f.centre[0]), _r(f.centre[1])],
                  "length_m": _r(f.length), "region": None if f.region is None else rid[f.region]}
                 for i, f in zip(fid, feat.frontiers, strict=True)]
    rows, cols = lay.shape
    body = {
        "map": dv.name, "map_version": dv.version,
        "origin_m": [_r(lay.origin[0]), _r(lay.origin[1])], "cell_m": cell,
        "shape": [rows, cols], "z_band_m": [z_min, z_max], "floor_z_m": _r(floor),
        "scale_source": dv.record["scale_source"],
        "scale_sigma_rel": dv.record["scale_sigma_rel"],
        "layers": {"state": binary(lay.state.astype(np.uint8)),
                   "observations": binary(lay.observations.astype(np.uint16)),
                   "height_m": binary(lay.height.astype(np.float32)),
                   "floor_observed": binary(lay.floor_observed.astype(np.uint8))},
        "regions": regions, "passages": passages, "frontiers": frontiers}
    return GridAnswer(body, base, feat, rid)


def object_regions(answer: GridAnswer, objects: list[dict[str, Any]],
                   seen_from: dict[int, list[list[float]]] | None = None) -> dict[int, int | None]:
    """The ``region`` of each object record (map.md §5) on a grid answer: the hook the objects
    read (O8) calls with its records (``id``, ``centre_m``, ``size_m``, ``yaw_deg``) and, per
    object id, the map x, y of the cameras that saw it."""
    out: dict[int, int | None] = {}
    for o in objects:
        k = navgrid.object_region(answer.features, answer.layers, o["centre_m"], o["size_m"],
                                  float(o.get("yaw_deg", 0.0)), (seen_from or {}).get(o["id"], ()))
        out[o["id"]] = None if k is None else answer.region_ids[k]
    return out


# -- points --------------------------------------------------------------------------------------


def points(dv: DerivedVersion, lo: NDArray[np.float64], hi: NDArray[np.float64], voxel: float,
           max_points: int) -> dict[str, Any]:
    c = dv.cloud()
    xyz = c["xyz"]
    sel, truncated = navgrid.thin_points(xyz, lo, hi, voxel, max_points)
    return {"map": dv.name, "map_version": dv.version, "box_min_m": lo.tolist(),
            "box_max_m": hi.tolist(), "voxel_m": voxel, "truncated": truncated,
            "scale_source": dv.record["scale_source"],
            "scale_sigma_rel": dv.record["scale_sigma_rel"],
            "xyz": binary(xyz[sel].astype(np.float32)),
            "object_id": binary(c["object_id"][sel].astype(np.uint32)),
            "rgb": binary(c["rgb"][sel].astype(np.uint8))}


# -- locate --------------------------------------------------------------------------------------


@dataclass
class LocateFrame:
    name: str
    camera: Camera
    rgb: NDArray[np.uint8]
    mask: NDArray[np.bool_] | None
    prior: dict[str, Any] | None
    height_m: float | None
    height_sigma_m: float | None


def locate_frames(store: Store, name: str, req: dict[str, Any], parts: dict[str, bytes]
                  ) -> list[LocateFrame]:
    """The frames of a valid ``LocateRequest``; ``invalid_request`` or ``camera_conflict``."""
    from oh_my_slam.service.app import _rotation_ok

    known = store.cameras(name)
    cams = dict(known)

    def part(field: str, part_name: str) -> bytes:
        if part_name not in parts:
            raise invalid(field, f"{field}: no multipart part named {part_name!r}")
        return parts[part_name]

    cam_masks: dict[str, NDArray[np.bool_] | None] = {}
    for cid, cam in (req.get("cameras") or {}).items():
        if cid in known and not same_camera(known[cid], cam):
            raise ApiError("camera_conflict", f"camera {cid!r} of map {name!r} is already "
                           "defined with other values: omit it, or use a new camera id",
                           {"field": f"cameras.{cid}", "camera_id": cid, "known": known[cid]})
        cams[cid] = cam
        if "ignore_mask" in cam:
            m = decode_mask(part(f"cameras.{cid}.ignore_mask", cam["ignore_mask"]),
                            f"cameras.{cid}.ignore_mask")
            if m.shape != (cam["height"], cam["width"]):
                raise invalid(f"cameras.{cid}.ignore_mask", f"cameras.{cid}.ignore_mask is "
                              f"{m.shape[1]}x{m.shape[0]}, the camera "
                              f"{cam['width']}x{cam['height']}")
            cam_masks[cid] = m
    out, seen = [], set()
    for i, f in enumerate(req["frames"]):
        field = f"frames[{i}]"
        if f["name"] in seen:
            raise invalid(f"{field}.name", f"{field}.name {f['name']!r} is not unique in the "
                          "request")
        seen.add(f["name"])
        cam = cams.get(f["camera_id"])
        if cam is None:
            raise invalid(f"{field}.camera_id", f"{field}.camera_id {f['camera_id']!r} is "
                          "neither in the request's cameras nor known to the map")
        rgb = decode_rgb(part(f"{field}.image", f["image"]), f"{field}.image")
        if rgb.shape[:2] != (cam["height"], cam["width"]):
            raise invalid(f"{field}.image", f"{field}.image is {rgb.shape[1]}x{rgb.shape[0]}; "
                          f"camera {f['camera_id']!r} is {cam['width']}x{cam['height']}")
        mask = cam_masks.get(f["camera_id"])
        if mask is None and f["camera_id"] in known and store.mask_digest(name, f["camera_id"]):
            mask = decode_mask(store.mask_path(name, f["camera_id"]).read_bytes(),
                               f"cameras.{f['camera_id']}.ignore_mask")
        if "ignore_mask" in f:
            own = decode_mask(part(f"{field}.ignore_mask", f["ignore_mask"]),
                              f"{field}.ignore_mask")
            if own.shape != rgb.shape[:2]:
                raise invalid(f"{field}.ignore_mask", f"{field}.ignore_mask is "
                              f"{own.shape[1]}x{own.shape[0]}; camera {f['camera_id']!r} is "
                              f"{cam['width']}x{cam['height']}")
            mask = own if mask is None else (mask | own)
        prior = f.get("prior")
        if prior is not None and "T_map_cam" in prior and not _rotation_ok(prior["T_map_cam"]):
            raise invalid(f"{field}.prior.T_map_cam", f"{field}.prior.T_map_cam is not a rigid "
                          "transform (orthonormal rotation, last row 0 0 0 1)")
        out.append(LocateFrame(f["name"], Camera.from_json(cam), rgb, mask, prior,
                               f.get("height_above_floor_m"), f.get("height_sigma_m")))
    return out


def _query(fr: LocateFrame, floor_z: float) -> tuple[relocalize.Query, checks.Measures]:
    import cv2

    cam = fr.camera
    rgb = cam.rectify(fr.rgb)
    mask = None if fr.mask is None else cam.rectify(fr.mask.astype(np.uint8), nearest=True) > 0
    measures = checks.measure(rgb, mask)
    gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np.uint8)
    uv, desc = relocalize.extract(gray, mask)
    K = cam.canonical()
    q = relocalize.Query(uv, desc, K.K(), K.width, K.height)
    p = fr.prior
    if p is not None and p.get("trust") in ("metric", "drifting") and "T_map_cam" in p:
        q.prior = np.asarray(p["T_map_cam"], np.float64)
        q.prior_sigma_pos_m = float(p.get("sigma_pos_m", relocalize.DEFAULT_PRIOR_SIGMA_POS_M))
        q.prior_sigma_rot_deg = float(p.get("sigma_rot_deg",
                                            relocalize.DEFAULT_PRIOR_SIGMA_ROT_DEG))
    h = fr.height_m if fr.height_m is not None else cam.height_above_floor_m
    if h is not None:
        sig = fr.height_sigma_m if fr.height_m is not None else cam.height_sigma_m
        q.camera_z = floor_z + float(h)
        q.camera_z_sigma = float(sig if sig is not None else relocalize.DEFAULT_HEIGHT_SIGMA_M)
    return q, measures


def locate(derived: Derived, dv: DerivedVersion, frames: list[LocateFrame]
           ) -> list[dict[str, Any]]:
    index = derived.index(dv)
    out = []
    for fr in frames:
        q, measures = _query(fr, dv.floor_z)
        res: dict[str, Any] = {"name": fr.name, "status": "rejected", "reason": None,
                               "T_map_cam": None, "sigma_pos_m": None, "sigma_rot_deg": None,
                               "inlier_ratio": None, "ambiguity": None,
                               "consistent_with_prior": None, "nearest_view": None}
        reason = checks.reason(measures, None)
        if reason is not None:
            res["reason"] = reason
            out.append(res)
            continue
        o = relocalize.locate(q, index)
        if o.located:
            assert o.T is not None
            res.update({"status": "located", "T_map_cam": np.round(o.T, 6).tolist(),
                        "sigma_pos_m": o.sigma_pos_m, "sigma_rot_deg": o.sigma_rot_deg,
                        "inlier_ratio": o.inlier_ratio, "ambiguity": o.ambiguity,
                        "consistent_with_prior": o.consistent_with_prior})
        else:
            res["reason"] = checks.unregistered_reason(measures)
            if res["reason"] == "no_overlap" and o.nearest is not None:
                kf, turn = relocalize.nearest_view(q, index, o.nearest)
                res["nearest_view"] = {"name": kf.name, "T_map_cam": np.round(kf.T, 6).tolist(),
                                       "turn_deg": turn}
        out.append(res)
    return out


def record_pose(store: Store, name: str, version: int, located: dict[str, Any]) -> None:
    atomic_write_json(store.dir(name) / "pose.json", {
        "map_version": version, "name": located["name"], "recorded_at": utc(time.time()),
        "T_map_cam": located["T_map_cam"], "sigma_pos_m": located["sigma_pos_m"],
        "sigma_rot_deg": located["sigma_rot_deg"]})


def recorded_pose(store: Store, name: str) -> dict[str, Any]:
    try:
        out: dict[str, Any] = json.loads((store.dir(name) / "pose.json").read_text())
    except (OSError, ValueError) as exc:
        raise ApiError("no_pose", f"map {name!r} has no recorded pose: locate a frame with "
                       "record: true first", {"map": name}) from exc
    return out
