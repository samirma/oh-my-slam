"""Frame analysis (O7): ``POST /api/frames/analyse`` (api.md §7) [``AnalyseRequest`` →
``Analysis``]. It analyses one frame and changes nothing: no map, camera, version or job.

* **The frame** is taken as an update takes one: rectified to its camera's canonical pinhole
  (``cameras``), its ignored pixels (the camera's mask and its own, and what the camera's image
  does not reach of that pinhole) filled for the detector and giving nothing anywhere. Its
  depth is its depth image (read with the camera's ``depth_scale_m``) when it holds a value,
  else the monocular model's (``Backend.infer``).
* **The frame of the outputs** (``frame``): ``map`` with ``map`` and ``T_map_cam``; ``client``
  with ``T_map_cam`` alone, the pose's own frame; else ``gravity_camera``
  (``mapping.analysis.gravity_pose`` of the gravity the model sees).
* **Scale** (``scale_source``, ``scale_sigma_rel``). In the map frame, the version's: a depth
  image is taken as it is (an anchored map fuses depth images as they are), a monocular depth is
  brought onto the map (``mapping.analysis.map_factor``; when the pose sees too little of the
  map's cloud, by the camera's height above the map's floor, ``height_factor``; else left as the
  model gives it). Elsewhere ``depth`` with a depth image; else ``camera_height`` when the
  camera gives its height and its own floor plane is measurable (``height_factor``); else
  ``monocular``.
* **The floor** of the outputs: the version's in the map frame; elsewhere the frame's own floor
  plane (``mapping.analysis.own_floor_z``), else the camera's height below it, else unknown
  (no object is grounded, and the range scan takes ``DEFAULT_CAMERA_HEIGHT_M`` below the camera).
* **Prompts**: the default vocabulary, the map's (``map`` given) and the request's ``labels``;
  ``labels`` alone with ``labels_only``. A detection's label takes the client's form
  (``client_form``) and ``matches`` names the best object of each client label.
* **Inference** is asked as interactive (``Backend.interactive``): served before a job's fusion
  requests (inference.md, single worker). A depth image with a pose needs no model for points,
  range scan and depth."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import rle
from oh_my_slam.core.constants import FLOOR_SIGMA_DEFAULT_M, floor_band_m
from oh_my_slam.core.images import on_grid, save_jpeg
from oh_my_slam.core.log import get_logger
from oh_my_slam.mapping import analysis
from oh_my_slam.mapping.anchored import (
    DEFAULT_HEIGHT_SIGMA_M,
    DEPTH_SIGMA_REL,
    MIN_SCALE_SIGMA_REL,
    MONOCULAR_SIGMA_REL,
)
from oh_my_slam.mapping.anchored_objects import (
    MONO_DEPTH_REL,
    SENSOR_DEPTH_REL,
    View,
    regrid,
    unignored,
)
from oh_my_slam.segmentation.api import (
    color_for_id,
    normalize_label,
    prompt_of,
    request_floor,
)
from oh_my_slam.segmentation.detect import order
from oh_my_slam.service.backend import Backend
from oh_my_slam.service.cameras import (
    Camera,
    decode_mask,
    decode_png16,
    decode_rgb,
    mapper_depth,
    mapper_image,
)
from oh_my_slam.service.derived import DerivedVersion
from oh_my_slam.service.errors import invalid
from oh_my_slam.service.nav import DEFAULT_Z_BAND, binary
from oh_my_slam.service.prior_first import detection_grid
from oh_my_slam.service.store import Store
from oh_my_slam.service.update import JPEG_QUALITY

log = get_logger("oh_my_slam.service.analyse")

DEFAULT_BEARING_STEP_DEG = 1.0
DEFAULT_VOXEL_M = 0.02
DEFAULT_MIN_SCORE = 0.5
DEFAULT_CAMERA_HEIGHT_M = 1.0  # the range scan's floor below a camera that sees and gives none
MAX_DEPTH_UNITS = 65535

Array = NDArray[np.float64]


@dataclass
class Work:
    """A valid ``AnalyseRequest`` with its parts decoded."""

    camera: Camera
    image: bytes
    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16] | None
    mask: NDArray[np.bool_] | None  # ignored pixels of the image: the camera's and the frame's
    T: Array | None
    outputs: frozenset[str]
    labels: list[str] | None  # the client's labels, trimmed, in request order
    labels_only: bool
    z_band: tuple[float, float]
    step_deg: float
    voxel_m: float
    min_score: float

    @property
    def wants_objects(self) -> bool:
        """``masks`` and ``labels`` imply ``objects`` (api.md §7)."""
        return bool(self.outputs & {"objects", "masks"}) or self.labels is not None


def work(store: Store, req: dict[str, Any], parts: dict[str, bytes]) -> Work:
    """The work of a valid ``AnalyseRequest``; ``invalid_request`` otherwise. The map it names
    exists (the caller's check)."""
    from oh_my_slam.service.app import _rotation_ok

    def part(field: str, part_name: str) -> bytes:
        if part_name not in parts:
            raise invalid(field, f"{field}: no multipart part named {part_name!r}")
        return parts[part_name]

    def sized(field: str, got: NDArray[Any], cam: Camera) -> None:
        if got.shape[:2] != (cam.height, cam.width):
            raise invalid(field, f"{field} is {got.shape[1]}x{got.shape[0]}; the camera is "
                          f"{cam.width}x{cam.height}")

    mask = None
    if "camera" in req:
        body = req["camera"]
        if "ignore_mask" in body:
            mask = decode_mask(part("camera.ignore_mask", body["ignore_mask"]),
                               "camera.ignore_mask")
    else:
        name, cid = req["map"], req["camera_id"]
        known = store.cameras(name)
        if cid not in known:
            raise invalid("camera_id", f"camera_id {cid!r} is not a camera of map {name!r}: "
                          "send the camera inline, or one of " + ", ".join(sorted(known)))
        body = known[cid]
        if store.mask_digest(name, cid):
            mask = decode_mask(store.mask_path(name, cid).read_bytes(), "camera.ignore_mask")
    cam = Camera.from_json(body)
    if mask is not None:
        sized("camera.ignore_mask", mask, cam)
    image = part("image", req["image"])
    rgb = decode_rgb(image, "image")
    sized("image", rgb, cam)
    depth = None
    if "depth" in req:
        depth = decode_png16(part("depth", req["depth"]), "depth")
        sized("depth", depth, cam)
    if "ignore_mask" in req:
        own = decode_mask(part("ignore_mask", req["ignore_mask"]), "ignore_mask")
        sized("ignore_mask", own, cam)
        mask = own if mask is None else (mask | own)
    T = None
    if "T_map_cam" in req:
        if not _rotation_ok(req["T_map_cam"]):
            raise invalid("T_map_cam", "T_map_cam is not a rigid transform (orthonormal "
                          "rotation, last row 0 0 0 1)")
        T = np.asarray(req["T_map_cam"], np.float64)
    labels = [str(lab).strip() for lab in req["labels"]] if "labels" in req else None
    only = bool(req.get("labels_only", False))
    if only and not labels:
        raise invalid("labels_only", "labels_only prompts with labels alone: send labels with "
                      "it, or leave it out")
    lo, hi = (float(v) for v in req.get("z_band_m", DEFAULT_Z_BAND))
    if not hi > lo:
        raise invalid("z_band_m", f"z_band_m [{lo}, {hi}]: its top must be above its bottom")
    step = float(req.get("bearing_step_deg", DEFAULT_BEARING_STEP_DEG))
    if step < analysis.MIN_BEARING_STEP_DEG:
        raise invalid("bearing_step_deg", f"bearing_step_deg {step} gives more than "
                      f"{round(360 / analysis.MIN_BEARING_STEP_DEG)} beams: ask for at least "
                      f"{analysis.MIN_BEARING_STEP_DEG}")
    return Work(cam, image, rgb, depth, mask, T, frozenset(req["outputs"]), labels, only,
                (lo, hi), step, float(req.get("voxel_m", DEFAULT_VOXEL_M)),
                float(req.get("min_score", DEFAULT_MIN_SCORE)))


def client_form(label: str, client: list[str]) -> str:
    """A detection's label in the client's form (api.md §7): the first client label equal
    (map.md §5, ``normalize_label``) to it or to the prompt it was detected as (a default
    label's own wording for the detector, ``prompt_of``); else the label."""
    keys = {normalize_label(label), normalize_label(prompt_of(label))}
    return next((c for c in client if normalize_label(c) in keys), label)


def matches(labels: list[str], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``matches`` (api.md §7): per client label, in request order, the returned object whose
    label equals it with the highest score (ties: the lower id), or nulls."""
    out = []
    for lab in labels:
        key = normalize_label(lab)
        best = min((o for o in objects if normalize_label(o["label"]) == key),
                   key=lambda o: (-o["score"], o["id"]), default=None)
        out.append({"label": lab, "object_id": None if best is None else best["id"],
                    "score": None if best is None else best["score"]})
    return out


def color_of(object_id: int) -> tuple[int, int, int]:
    """An analysis object's display colour, a pure function of its id: a map object's own
    (``segmentation.api.color_for_id``); a local id's is the colour of its rank among the local
    ids (``LOCAL_ID_START`` takes id 1's: the palette is built id by id, so the millionth colour
    is never asked)."""
    local = object_id - analysis.LOCAL_ID_START
    return color_for_id(local + 1 if local >= 0 else object_id)


def _round(v: Any, nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in v]


class Analyser:
    """Frame analysis on a backend's models (the tests' double: ``Backend``)."""

    def __init__(self, store: Store, backend: Backend) -> None:
        self.store = store
        self.backend = backend

    def run(self, w: Work, dv: DerivedVersion | None = None,
            records: Callable[[], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        """The ``Analysis`` body (without ``schema``) of ``w``; ``dv``: the version of the map
        the request names, ``records``: its objects read's records."""
        cam = w.camera
        K = cam.canonical()
        ignore = cam.beyond()
        if w.mask is not None and w.mask.any():
            own = cam.rectify(w.mask.astype(np.uint8), nearest=True) > 0
            ignore = own if ignore is None else (ignore | own)
        sensor = None
        if w.depth is not None:
            sensor = mapper_depth(w.depth, cam, w.mask)
            if ignore is not None:
                sensor[ignore] = 0.0
            if not np.count_nonzero(sensor > 0):
                sensor = None  # a depth image without a value is none
        mono, dets, rgb = self._infer(w, dv, K, sensor is None or w.T is None)
        if sensor is not None:
            depth, K_grid, rel = sensor, K, SENSOR_DEPTH_REL
        else:
            assert mono is not None
            depth, K_grid, rel = mono.depth.astype(np.float32), mono.K_grid, MONO_DEPTH_REL
            m = on_grid(ignore, depth.shape)
            if m is not None:
                depth = np.where(m, 0.0, depth).astype(np.float32)
        if w.T is None:
            assert mono is not None
            T, frame = analysis.gravity_pose(mono.up_cam), "gravity_camera"
        else:
            T, frame = w.T, "map" if dv is not None else "client"
        # the camera's height above the floor, when it gives one that is a height
        height = cam.height_above_floor_m if (cam.height_above_floor_m or 0.0) > 0 else None
        factor, floor_sigma = 1.0, FLOOR_SIGMA_DEFAULT_M
        floor: float | None
        if frame == "map":
            assert dv is not None
            source, sigma = str(dv.record["scale_source"]), float(dv.record["scale_sigma_rel"])
            floor, floor_sigma = dv.floor_z, dv.floor_sigma
            if sensor is None:
                factor = (analysis.map_factor(depth, K_grid, T, dv.cloud()["xyz"])
                          or analysis.height_factor(depth, K_grid, T[:3, :3],
                                                    float(T[2, 3]) - floor) or 1.0)
            depth = (depth * factor).astype(np.float32)
        else:
            source, sigma = "depth", DEPTH_SIGMA_REL
            if sensor is None:
                f = analysis.height_factor(depth, K_grid, T[:3, :3], height)
                if f is None:
                    source, sigma = "monocular", MONOCULAR_SIGMA_REL
                else:
                    assert height is not None
                    factor, source = f, "camera_height"
                    sigma = max((cam.height_sigma_m or DEFAULT_HEIGHT_SIGMA_M) / height,
                                MIN_SCALE_SIGMA_REL)
            depth = (depth * factor).astype(np.float32)
            floor = analysis.own_floor_z(depth, K_grid, T, height)
            if floor is None and height is not None:
                floor = float(T[2, 3]) - height
        shape = (depth.shape[0], depth.shape[1])
        grid_ignore = on_grid(ignore, shape)
        if rgb.shape[:2] != shape:
            rgb = np.asarray(cv2.resize(rgb, (shape[1], shape[0]),
                                        interpolation=cv2.INTER_AREA), np.uint8)
        # the detections in their own order (score first), whatever the backend's: the local
        # ids follow it
        view = View("frame", T, 0.0, K_grid, depth, rgb,
                    unignored(regrid(order(dets), shape), grid_ignore), grid_ignore, rel)
        seen: list[analysis.Seen] = []
        if w.wants_objects:
            known = []
            if frame == "map" and records is not None:
                assert dv is not None
                at = dv
                known = dv.answer(("analysis_objects",),
                                  lambda: analysis.known_objects(records(), at.cloud()))
            seen = analysis.objects(view, floor, floor_sigma, w.min_score, known)
        out: dict[str, Any] = {"frame": frame, "scale_source": source,
                               "scale_sigma_rel": round(sigma, 5)}
        if w.wants_objects:
            out["objects"] = self._objects(w, seen, T)
            returned = {o["id"] for o in out["objects"]}
            seen = [s for s in seen if s.id in returned]  # the points name returned objects only
        if w.outputs & {"points", "range_scan"}:
            pts = analysis.frame_points(view, seen)
            if "points" in w.outputs:
                thin = analysis.thinned(pts, w.voxel_m)
                out["points"] = {"xyz": binary(thin.xyz.astype(np.float32)),
                                 "object_id": binary(thin.object_id.astype(np.uint32)),
                                 "rgb": binary(thin.rgb.astype(np.uint8))}
            if "range_scan" in w.outputs:
                level = floor if floor is not None else float(T[2, 3]) - DEFAULT_CAMERA_HEIGHT_M
                out["range_scan"] = analysis.range_scan(
                    pts.xyz, T[:3, 3], level, floor_band_m(floor_sigma), w.z_band, w.step_deg)
        if "depth" in w.outputs:
            units = np.rint(self._on_image(w, depth) / cam.depth_scale_m)
            out["depth"] = binary(np.clip(units, 0, MAX_DEPTH_UNITS).astype(np.uint16))
        if w.labels is not None:
            out["matches"] = matches(w.labels, out["objects"])
        log.info("analysed a frame: frame %s, scale %s x%.3f, %d objects", frame, source, factor,
                 len(seen))
        return out

    def _infer(self, w: Work, dv: DerivedVersion | None, K: Any, geometry: bool
               ) -> tuple[Any, list[Any], NDArray[np.uint8]]:
        """The frame's monocular depth (with its gravity when it has no pose; None unless
        ``geometry``: it has no depth image or no pose), its detections (none unless objects are
        asked) and its image on the canonical pinhole, the two requests sent together as
        interactive ones."""
        cam = w.camera
        plain = cam.is_canonical and (w.mask is None or not w.mask.any())
        rgb = w.rgb if plain else mapper_image(w.rgb, cam, w.mask)
        if not (geometry or w.wants_objects):
            return None, [], rgb
        backend = self.backend.interactive()
        backend.require_inference()
        shown = [*((dv.record.get("labels") or []) if dv is not None and not w.labels_only
                   else []), *(w.labels or [])]
        workdir = self.store.work / f"analyse-{uuid.uuid4().hex}"
        workdir.mkdir(parents=True)
        try:
            path = workdir / "frame.jpg"
            if plain and w.image[:3] == b"\xff\xd8\xff":
                path.write_bytes(w.image)  # as sent: an update's detections of the same frame
            else:
                save_jpeg(rgb, path, quality=JPEG_QUALITY)
            with ThreadPoolExecutor(2) as pool:
                mono = pool.submit(backend.infer, path, K, w.T is None) if geometry else None
                # asked on an update's detection grid, down to the floor that keeps the objects
                # independent of min_score (``segmentation.detect.request_floor``)
                found = pool.submit(backend.detect, path, detection_grid(K), shown,
                                    request_floor(w.min_score), not w.labels_only
                                    ) if w.wants_objects else None
                return (None if mono is None else mono.result(),
                        [] if found is None else found.result(), rgb)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _on_image(w: Work, grid: NDArray[Any]) -> NDArray[Any]:
        """A grid of the frame's canonical pinhole on the pixels of the image as sent (nearest),
        0 on its ignored pixels."""
        cam = w.camera
        if grid.shape[:2] != (cam.height, cam.width):
            grid = cv2.resize(grid, (cam.width, cam.height), interpolation=cv2.INTER_NEAREST)
        out = cam.unrectify(grid)
        return out if w.mask is None else np.where(w.mask, 0, out)

    def _objects(self, w: Work, seen: list[analysis.Seen], T: Array) -> list[dict[str, Any]]:
        """The ``AnalysisObject`` of each object that has pixels in the image as sent."""
        out = []
        for s in seen:
            mask = self._on_image(w, s.mask.astype(np.uint8)) > 0
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            if not len(rows):
                continue  # all of it beyond what the image shows
            label = s.label if w.labels is None else client_form(s.label, w.labels)
            o: dict[str, Any] = {
                "id": s.id, "label": label, "detected_as": [label],
                "score": round(min(max(s.score, 0.0), 1.0), 4),
                "centre_m": _round(s.box.centre), "size_m": _round(s.box.size),
                "yaw_deg": s.box.yaw_deg, "sigma_centre_m": round(s.sigma_centre_m, 4),
                "extent_seen": s.extent_seen, "mean_rgb": s.mean_rgb,
                "color": list(color_of(s.id)),
                "bbox_px": [float(cols[0]), float(rows[0]), float(cols[-1] + 1),
                            float(rows[-1] + 1)],
                "centre_cam_m": _round((s.box.centre - T[:3, 3]) @ T[:3, :3])}
            if "masks" in w.outputs:
                o["mask"] = rle.encode(mask)
            out.append(o)
        return out
