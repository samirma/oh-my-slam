"""``mapper.sh locate``: the camera pose of images registered against an existing map, which stays
read-only (spec 2.3).

The map is never written, locked or staged: it is read through ``store.MapReader``, and its COLMAP
database is cloned (``cp -c``) into a scratch folder, where the query images get their SIFT features
and their matches to the keyframes. Pairs: every keyframe for a map of at most
``UPDATE_EXHAUSTIVE_MAX`` keyframes (no inference server needed), else the ``RETRIEVAL_TOP_K`` most
similar keyframes by the retrieval descriptor, which the inference server computes for the query
through reconstruction (exit 3 when it is down).

Pose: the verified matches give 2D-3D correspondences — a query keypoint, the keyframe keypoint it
matches, that keypoint's triangulated point in the map's ``sfm/model`` (already in map
coordinates) when SfM posed the keyframe; otherwise — a keypoint without a model point, a
keyframe the model lacks (a one-keyframe map has no ``sfm/`` at all), or one posed by multi-view
(rotation-dominant input, whose model points are unreliable) — the point is the keyframe's stored
metric depth at that keypoint (off depth edges), placed with its stored pose and scaled to the
keyframe's model points (``_MapPoints.depth_scale``). The model points alone are tried first
(``MODEL_FIRST_MIN_INLIERS``), then model and depth points together. LO-RANSAC
absolute pose plus refinement (``pycolmap.estimate_and_refine_absolute_pose``) solves them; the
focal length is refined unless the query shares a map camera. A result with fewer than
``MIN_INLIERS`` inliers, or whose pose contradicts its verified matches to the stored keyframe poses
(median epipolar distance above ``MAX_EPIPOLAR_DEG``, as for the map's own keyframes;
``epipolar_gate``, which refines the pose of a camera seen from the keyframes' spot first), leaves
the image unlocated: it is reported on stderr, naming the image; with no image located the command
fails with an input error.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import paths, timing
from oh_my_slam.core.atomic import clone_file
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import InputError, NotAMapError, UsageError
from oh_my_slam.core.images import IMAGE_SUFFIXES, VIDEO_SUFFIXES, exif_intrinsics, upright_size
from oh_my_slam.core.log import get_logger, json_payload_bytes
from oh_my_slam.core.timing import Stage
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import retrieval, store
from oh_my_slam.mapping.sfm import (
    EPIPOLAR_MIN_MATCHES,
    MAX_EPIPOLAR_DEG,
    MIN_INLIERS,
    CameraPrior,
    Sfm,
    epipolar_deg,
)

log = get_logger("oh_my_slam.locate")

QUERY_PREFIX = "locate_"  # query image names in the scratch database: never fNNNNNN.jpg
LOCATED_CS = "located_"  # coordinate system / stream / PLY comment of input image k: located_<k>
# PLY -t single: map points visible from a located camera — in its frustum, beyond VIS_NEAR, and
# at most VIS_REL_TOL behind the nearest point around their cell (3 x 3 cells, which closes the gaps
# between the points of a near surface) of a coarse z-buffer, VIS_GRID_SIDE cells on the long side:
# the query has no depth of its own
VIS_GRID_SIDE = 96
VIS_REL_TOL = 0.05
VIS_NEAR = 0.05
VIS_CHUNK = 1 << 20
# Epipolar gate of a camera that sees the map from the keyframes' spot (rotation_dominant: most
# matches with multi-view keyframes, or a baseline below ROTATION_BASELINE of the scene depth):
# a pose above MAX_EPIPOLAR_DEG is refined against the keyframe poses (IRLS Cauchy on the
# epipolar distances, REFINE_IRLS_ROUNDS rounds, quadratic centre prior REFINE_CENTRE_PRIOR_M, at
# most REFINE_MAX_PER_KEYFRAME matches each) and judged again. The refined pose is kept only when
# it passes, its centre moved at most REFINE_MAX_MOVE_M (the depth's scale error: centimetres),
# it turned at most the angle that move subtends at the median scene depth, and it keeps
# REFINE_KEEP_INLIERS of the pose's 2D-3D inliers (POSE_MAX_ERROR_PX, as the absolute pose
# estimation counts them).
ROTATION_BASELINE = 0.02
REFINE_CENTRE_PRIOR_M = 0.05
REFINE_MAX_MOVE_M = 2 * REFINE_CENTRE_PRIOR_M
REFINE_MAX_PER_KEYFRAME = 250
REFINE_IRLS_ROUNDS = 5
REFINE_KEEP_INLIERS = 0.9
POSE_MAX_ERROR_PX = 12.0  # pycolmap AbsolutePoseEstimationOptions().ransac.max_error
# A pose from the SfM model points alone is kept when that many of them agree (and it passes the
# epipolar gate); else the model points and the keyframes' depth together
MODEL_FIRST_MIN_INLIERS = 50

Progress = Callable[[str], None]


def _progress(msg: str) -> None:
    log.info(msg)


@dataclass
class Located:
    """The result for one input image (``index``: its position in ``-i``)."""

    image: Path
    index: int
    T_map_cam: Pose | None = None
    K: Intrinsics | None = None
    inliers: int = 0
    reason: str = ""

    @property
    def located(self) -> bool:
        return self.T_map_cam is not None

    @property
    def key(self) -> str:
        return f"{LOCATED_CS}{self.index}"


@dataclass
class LocateResult:
    payload: bytes
    results: list[Located]
    timings: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------
# inputs


def resolve_images(inputs: Sequence[Path]) -> list[Path]:
    """``-i`` of ``locate``: image files only; a video is refused (usage error)."""
    if not inputs:
        raise UsageError("-i needs at least one image")
    out = []
    for p in map(Path, inputs):
        if p.suffix.lower() in VIDEO_SUFFIXES:
            raise UsageError(f"locate takes images, not a video: {p}")
        if not p.exists():
            raise InputError(f"input not found: {p}")
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in IMAGE_SUFFIXES:
            raise InputError(f"not an image file: {p} (locate takes one or more images)")
        out.append(p)
    return out


def open_map(map_dir: Path) -> store.MapReader:
    """The map ``locate`` reads. A missing or empty folder is an input error (and is not
    created); a non-empty folder that is not a map keeps the mapper's not-a-map error."""
    state = store.classify(map_dir)
    if state == "other":
        raise NotAMapError(f"{Path(map_dir).resolve()} is not a map")
    try:
        return store.MapReader(map_dir)
    except NotAMapError:
        raise InputError(f"no map in {map_dir} ({state} folder): locate needs an existing map — "
                         "build one with mapper.sh update") from None


def check_output(map_dir: Path, output: Path | None) -> None:
    """``-o`` must not point into the map folder: the map is read-only for ``locate``."""
    if output is None:
        return
    root = Path(map_dir).resolve()
    if Path(output).resolve().is_relative_to(root):
        raise UsageError(f"-o {output} is inside the map folder, which locate never writes; "
                         "choose a file outside it")


# ------------------------------------------------------------------------------------------------
# features and matches in a scratch copy of the map's database


@dataclass
class _DbImage:
    image_id: int
    camera_id: int


def _db_images(db: Path) -> dict[str, _DbImage]:
    import pycolmap

    if not db.exists():
        return {}
    d = pycolmap.Database.open(str(db))
    try:
        return {im.name: _DbImage(int(im.image_id), int(im.camera_id))
                for im in d.read_all_images()}
    finally:
        d.close()


def _kf_file(fr: store.FrameRecord) -> str:
    return f"{fr.name}.jpg"


@dataclass
class _Query:
    image: Path
    index: int
    name: str  # in the scratch database
    size: tuple[int, int]
    exif: Intrinsics | None


def _extract(reader: store.MapReader, queries: list[_Query], work: Path
             ) -> tuple[Sfm, dict[str, _DbImage]]:
    """Scratch database (the map's, cloned) with the queries' features; keyframes the map's
    database lacks (a one-keyframe map has none) are extracted too, from their stored images."""
    from oh_my_slam.mapping.ingest import write_upright_jpeg

    db, images = work / "database.db", work / "images"
    images.mkdir(parents=True)
    if reader.exists(store.SFM_DB):
        clone_file(reader.path(store.SFM_DB), db)
    sfm = Sfm(db, images, work / "sfm")
    known = _db_images(db)
    missing: dict[tuple[int, int, float], list[str]] = {}
    for fr in reader.frames:
        if _kf_file(fr) not in known:
            clone_file(reader.image_path(fr), images / _kf_file(fr))
            missing.setdefault((fr.width, fr.height, fr.K.fx), []).append(_kf_file(fr))
    for (w, h, f), names in sorted(missing.items()):
        sfm.extract(names, CameraPrior(w, h, focal=f))
    known = _db_images(db)
    for q in queries:
        write_upright_jpeg(q.image, images / q.name)
    groups: dict[tuple[int, int, float | None], list[_Query]] = {}
    for q in queries:
        groups.setdefault((*q.size, None if q.exif is None else q.exif.fx), []).append(q)
    for (w, h, focal), qs in groups.items():
        sfm.extract([q.name for q in qs], _camera_prior(reader, known, w, h, focal))
    return sfm, _db_images(db)


def _camera_prior(reader: store.MapReader, known: dict[str, _DbImage], w: int, h: int,
                  focal: float | None) -> CameraPrior:
    """As for an update's photos: without EXIF, the camera of the latest keyframe of the same size
    (more photos of the device that took the map); with EXIF, a camera of that size with the same
    EXIF focal prior — else a new camera."""
    same = [known[_kf_file(fr)].camera_id for fr in sorted(reader.frames, key=lambda r: r.index)
            if (fr.width, fr.height) == (w, h) and _kf_file(fr) in known]
    if focal is None:
        return CameraPrior(w, h, existing_id=same[-1] if same else None)
    return CameraPrior(w, h, focal=focal, same_focal_ids=tuple(dict.fromkeys(reversed(same))))


def _pairs(reader: store.MapReader, queries: list[_Query], db: dict[str, _DbImage],
           client: Any) -> set[tuple[int, int]]:
    """Query-keyframe pairs (database image ids): all keyframes for a map of at most
    ``UPDATE_EXHAUSTIVE_MAX``, else the ``RETRIEVAL_TOP_K`` most similar ones by the retrieval
    descriptor (inference server, through reconstruction)."""
    from oh_my_slam.core.constants import RETRIEVAL_TOP_K, UPDATE_EXHAUSTIVE_MAX

    kfs = [fr for fr in reader.frames if _kf_file(fr) in db]
    q_ids = [db[q.name].image_id for q in queries]
    if len(reader.frames) <= UPDATE_EXHAUSTIVE_MAX:
        return retrieval.all_pairs(q_ids, [db[_kf_file(fr)].image_id for fr in kfs])
    desc, ids = [], []
    for fr in kfs:
        d = reader.descriptor(fr)
        if d is not None:
            desc.append(d)
            ids.append(db[_kf_file(fr)].image_id)
    q_desc = query_descriptors(queries, client)
    return retrieval.top_k_pairs(q_desc, np.stack(desc), RETRIEVAL_TOP_K, q_ids, ids, 0) \
        if desc else set()


def query_descriptors(queries: Sequence[_Query], client: Any) -> NDArray[np.float32]:
    """Retrieval descriptors of the query images, computed as the keyframes' were (keyframe grid
    and tokens). Needs the inference server: ``ServerUnavailableError`` (exit 3) when it is down."""
    from oh_my_slam.mapping.api import KEYFRAME_GRID_SIDE
    from oh_my_slam.reconstruction.api import KEYFRAME_TOKENS, connect_server, reconstruct_image

    client = client or connect_server()
    out = []
    for q in queries:
        fr = reconstruct_image(q.image, intrinsics=q.exif, max_side=KEYFRAME_GRID_SIDE,
                               num_tokens=KEYFRAME_TOKENS, want_gravity=False,
                               want_descriptor=True, client=client)
        if fr.descriptor is None:
            raise InputError(f"{q.image}: the inference server returned no retrieval descriptor")
        out.append(fr.descriptor)
    return np.stack(out).astype(np.float32)


@dataclass
class _Match:
    """Verified inlier matches of a query with one keyframe (keypoint indices and pixels)."""

    keyframe: str  # database image name
    idx_q: NDArray[np.int64]
    idx_k: NDArray[np.int64]
    uv_q: NDArray[np.float64]
    uv_k: NDArray[np.float64]


def _query_matches(db_path: Path, queries: list[_Query], db: dict[str, _DbImage]
                   ) -> dict[str, list[_Match]]:
    """Per query name, its verified matches (``panorama.verified_matches``' rules) with keyframes,
    keeping the keypoint indices."""
    import pycolmap

    from oh_my_slam.mapping.panorama import USED_CONFIGS

    names = {v.image_id: k for k, v in db.items()}
    qset = {q.name for q in queries}
    out: dict[str, list[_Match]] = {q.name: [] for q in queries}
    d = pycolmap.Database.open(str(db_path))
    try:
        kps: dict[int, NDArray[np.float64]] = {}

        def keypoints(i: int) -> NDArray[np.float64]:
            if i not in kps:
                kps[i] = np.asarray(d.read_keypoints(i), np.float64)[:, :2]
            return kps[i]

        pair_ids, geoms = d.read_two_view_geometries()
        for pid, g in zip(pair_ids, geoms, strict=True):
            if int(g.config) not in USED_CONFIGS or len(g.inlier_matches) < MIN_INLIERS:
                continue
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            m = np.asarray(g.inlier_matches, np.int64)
            if names.get(a) in qset and names.get(b) not in qset and b in names:
                q, k, iq, ik = a, b, m[:, 0], m[:, 1]
            elif names.get(b) in qset and names.get(a) not in qset and a in names:
                q, k, iq, ik = b, a, m[:, 1], m[:, 0]
            else:
                continue
            out[names[q]].append(_Match(names[k], iq, ik, keypoints(q)[iq], keypoints(k)[ik]))
    finally:
        d.close()
    for v in out.values():
        v.sort(key=lambda m: m.keyframe)
    return out


# ------------------------------------------------------------------------------------------------
# 2D-3D correspondences and pose


def model_points_trusted(fr: store.FrameRecord) -> bool:
    """Whether the ``sfm/model`` points of keyframe ``fr`` place its keypoints: only for a pose SfM
    estimated. The points of multi-view (rotation-dominant) poses are triangulated without bundle
    adjustment from near-zero baselines — the mapper itself never fits depth to them — and an
    identity pose has none."""
    return fr.pose_source.startswith("sfm") and "multiview" not in fr.pose_source


# Depth fill of an SfM keyframe: its stored depth is rescaled by the median model/depth z ratio
# over its keypoints that have both (at least DEPTH_SCALE_MIN_POINTS of them); a keyframe whose
# median ratio is off by more than DEPTH_SCALE_MAX_FACTOR (depth seen through a window, say)
# gives no depth points at all.
DEPTH_SCALE_MIN_POINTS = 20
DEPTH_SCALE_MAX_FACTOR = 2.0


class _MapPoints:
    """Map coordinates of keyframe keypoints: their triangulated point in ``sfm/model`` for a
    keyframe SfM posed (``model_points_trusted``), and for every keypoint without one — all of
    them for a multi-view keyframe, or one the model does not hold — the keyframe's stored metric
    depth at the keypoint, placed with its stored pose (and scaled to the model's points,
    ``depth_scale``)."""

    def __init__(self, reader: store.MapReader) -> None:
        import pycolmap

        self.reader = reader
        self.frames = {_kf_file(fr): fr for fr in reader.frames}
        model_dir = reader.path(f"{store.SFM_MODEL}/images.bin").parent  # committed overlay too
        self.rec = pycolmap.Reconstruction(str(model_dir)) \
            if (model_dir / "images.bin").exists() else None
        self._cache: dict[str, tuple[NDArray[np.float64], NDArray[np.bool_],
                                     NDArray[np.float64]] | None] = {}
        self._depth: dict[str, tuple[NDArray[Any], NDArray[Any]]] = {}
        self._scale: dict[str, float | None] = {}

    def _model_points(self, name: str
                      ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]] | None:
        """(xyz per keypoint, has a point, keypoint pixels) of a keyframe the model has posed,
        else None."""
        if name not in self._cache:
            im = None if self.rec is None else self.rec.find_image_with_name(name)
            if im is None or not im.has_pose:
                self._cache[name] = None
            else:
                xyz = np.zeros((len(im.points2D), 3))
                has = np.zeros(len(im.points2D), bool)
                uv = np.zeros((len(im.points2D), 2))
                for i, p in enumerate(im.points2D):
                    uv[i] = p.xy
                    if p.has_point3D():
                        xyz[i] = self.rec.points3D[p.point3D_id].xyz  # type: ignore[union-attr]
                        has[i] = True
                self._cache[name] = (xyz, has, uv)
        return self._cache[name]

    def _depth_points(self, fr: store.FrameRecord, uv: NDArray[np.float64]
                      ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        if fr.name not in self._depth:
            from oh_my_slam.core.geometry import depth_edge_mask

            depth = self.reader.depth(fr)  # depth edges: a keypoint there may be either surface
            self._depth[fr.name] = (depth, self.reader.valid(fr) & ~depth_edge_mask(depth))
        return depth_points(*self._depth[fr.name], fr, uv)

    def depth_scale(self, fr: store.FrameRecord) -> float | None:
        """Factor of keyframe ``fr``'s depth points: the median model/depth z ratio (camera
        frame) over its keypoints with both, 1 without enough of them (a multi-view keyframe,
        whose depth is the map's scale), None when that ratio is off by more than
        ``DEPTH_SCALE_MAX_FACTOR`` (its depth gives no points)."""
        name = _kf_file(fr)
        if name not in self._scale:
            model = self._model_points(name) if model_points_trusted(fr) else None
            ratio = None
            if model is not None and int(model[1].sum()) >= DEPTH_SCALE_MIN_POINTS:
                xyz, has, uv = model
                d_xyz, d_ok = self._depth_points(fr, uv[has])
                z_model = ((xyz[has] - fr.T_map_cam.t) @ fr.T_map_cam.R)[:, 2]
                z_depth = ((d_xyz - fr.T_map_cam.t) @ fr.T_map_cam.R)[:, 2]
                both = d_ok & (z_model > 0) & (z_depth > 0)
                if int(both.sum()) >= DEPTH_SCALE_MIN_POINTS:
                    ratio = float(np.median(z_model[both] / z_depth[both]))
            if ratio is None:
                self._scale[name] = 1.0
                if model is not None:
                    log.info("%s: too few keypoints with both an SfM point and depth to scale its "
                             "depth; using it unscaled", fr.name)
            elif 1.0 / DEPTH_SCALE_MAX_FACTOR <= ratio <= DEPTH_SCALE_MAX_FACTOR:
                self._scale[name] = ratio
            else:
                self._scale[name] = None
                log.info("%s: depth disagrees with the SfM points (scale %.2f); its depth gives "
                         "no points", fr.name, ratio)
        return self._scale[name]

    def lookup(self, m: _Match) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """(xyz (N, 3), valid (N,)) of the keyframe side of the matches ``m``."""
        xyz, model, depth = self.lookup_split(m)
        return xyz, model | depth

    def lookup_split(self, m: _Match
                     ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
        """(xyz (N, 3), from the model (N,), from the depth (N,)) of the keyframe side of the
        matches ``m``: the model point where there is a trusted one, else the scaled depth."""
        n = len(m.idx_k)
        xyz, ok, fill = np.zeros((n, 3)), np.zeros(n, bool), np.zeros(n, bool)
        fr = self.frames.get(m.keyframe)
        if fr is None:
            return xyz, ok, fill
        model = self._model_points(m.keyframe) if model_points_trusted(fr) else None
        if model is not None:
            pts, has, _ = model
            inside = m.idx_k < len(has)
            idx = np.where(inside, m.idx_k, 0)
            ok = inside & has[idx]
            xyz[ok] = pts[idx[ok]]
        scale = self.depth_scale(fr)
        if not ok.all() and scale is not None:
            d_xyz, d_ok = self._depth_points(fr, m.uv_k)
            fill = ~ok & d_ok
            xyz[fill] = fr.T_map_cam.t + scale * (d_xyz[fill] - fr.T_map_cam.t)
        return xyz, ok, fill


def depth_points(depth: NDArray[Any], valid: NDArray[Any], fr: store.FrameRecord,
                 uv: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Map points of full-resolution keypoints ``uv`` of keyframe ``fr`` from its stored depth
    grid (nearest cell), unprojected with its full-resolution intrinsics and stored pose."""
    gh, gw = depth.shape
    col = np.clip((uv[:, 0] * gw / fr.width).astype(int), 0, gw - 1)
    row = np.clip((uv[:, 1] * gh / fr.height).astype(int), 0, gh - 1)
    z = np.asarray(depth, np.float64)[row, col]
    ok = np.asarray(valid, bool)[row, col] & (z > 0)
    K = fr.K
    cam = np.column_stack([(uv[:, 0] - K.cx) / K.fx * z, (uv[:, 1] - K.cy) / K.fy * z, z])
    return fr.T_map_cam.apply(cam), ok


def correspondences(matches: list[_Match], points: _MapPoints, depth: bool = True
                    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(query pixels (N, 2), map points (N, 3)): one per query keypoint — a model point when any
    keyframe it matches gives one, else (with ``depth``) a depth point; among those, the first
    keyframe in name order."""
    seen: set[int] = set()
    uv, xyz = [], []
    looked = [(m, points.lookup_split(m)) for m in matches]
    for kind in (1, 2) if depth else (1,):  # model points first, then depth points
        for m, split in looked:
            pts, ok = split[0], split[kind]
            for j in np.flatnonzero(ok):
                iq = int(m.idx_q[j])
                if iq not in seen:
                    seen.add(iq)
                    uv.append(m.uv_q[j])
                    xyz.append(pts[j])
    return np.asarray(uv, np.float64).reshape(-1, 2), np.asarray(xyz, np.float64).reshape(-1, 3)


def solve_pose(uv: NDArray[Any], xyz: NDArray[Any], K: Intrinsics, refine_focal: bool
               ) -> tuple[Pose, Intrinsics, int] | None:
    """LO-RANSAC absolute pose + refinement: (``T_map_cam``, intrinsics, inliers), or None. With
    ``refine_focal`` the focal length is estimated too (``K`` is only the starting point)."""
    import pycolmap

    if len(uv) < 4:
        return None
    if refine_focal:
        cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=K.width, height=K.height,
                              params=[K.fx, K.cx, K.cy])
    else:
        cam = pycolmap.Camera(model="PINHOLE", width=K.width, height=K.height,
                              params=[K.fx, K.fy, K.cx, K.cy])
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.estimate_focal_length = refine_focal
    est.ransac.random_seed = 0
    ref = pycolmap.AbsolutePoseRefinementOptions()
    ref.refine_focal_length = refine_focal
    ref.refine_extra_params = False
    res = pycolmap.estimate_and_refine_absolute_pose(np.asarray(uv, np.float64),
                                                     np.asarray(xyz, np.float64), cam, est, ref)
    if res is None:
        return None
    T = np.eye(4)
    T[:3, :] = np.asarray(res["cam_from_world"].matrix())
    Kc = np.asarray(cam.calibration_matrix())
    intr = Intrinsics(float(Kc[0, 0]), float(Kc[1, 1]), float(Kc[0, 2]), float(Kc[1, 2]),
                      K.width, K.height, "colmap" if refine_focal else K.source)
    return Pose.from_matrix(np.linalg.inv(T)), intr, int(res["num_inliers"])


def match_residual_deg(T: Pose, K: Intrinsics, matches: list[_Match],
                       frames: dict[str, store.FrameRecord]) -> tuple[float, int]:
    """(median epipolar distance in degrees, matches) of a located pose against its verified
    matches to the stored keyframe poses (``sfm.epipolar_deg``)."""
    errs = [epipolar_deg(T, K, frames[m.keyframe].T_map_cam, frames[m.keyframe].K, m.uv_q, m.uv_k)
            for m in matches if m.keyframe in frames]
    if not errs:
        return 0.0, 0
    e = np.concatenate(errs)
    return float(np.median(e)), len(e)


def _query_intrinsics(reader: store.MapReader, sfm: Sfm, q: _Query, db: dict[str, _DbImage]
                      ) -> tuple[Intrinsics, bool]:
    """(intrinsics, refine the focal length): a map camera's refined intrinsics (the stored ones of
    its latest keyframe) when the query shares it, else the scratch camera's prior."""
    cam = db[q.name].camera_id
    shared = [fr for fr in sorted(reader.frames, key=lambda r: r.index)
              if _kf_file(fr) in db and db[_kf_file(fr)].camera_id == cam]
    if shared:
        return shared[-1].K, False
    _, K = sfm.image_intrinsics({q.name})[q.name]
    return K, True


def _locate_one(q: _Query, K: Intrinsics, refine: bool, matches: list[_Match],
                points: _MapPoints) -> Located:
    out = Located(q.image, q.index)
    # SfM model points alone first: bundle-adjusted with the keyframe poses, they place a camera
    # more precisely than depth points do; the depth fills in when they are too few
    uv, xyz = correspondences(matches, points, depth=False)
    sol = solve_pose(uv, xyz, K, refine) if len(uv) >= MODEL_FIRST_MIN_INLIERS else None
    if sol is not None and sol[2] >= MODEL_FIRST_MIN_INLIERS:
        T, med, ok = epipolar_gate(sol[0], sol[1], matches, points, uv, xyz, sol[2],
                                   name=str(q.image))
        if ok:
            out.T_map_cam, out.K, out.inliers = T, sol[1], sol[2]
            return out
    uv, xyz = correspondences(matches, points)
    if len(uv) < MIN_INLIERS:
        out.reason = (f"only {len(uv)} feature matches with map points (needs {MIN_INLIERS}): "
                      "not enough overlap with the map")
        return out
    sol = solve_pose(uv, xyz, K, refine)
    if sol is None or sol[2] < MIN_INLIERS:
        out.reason = (f"no consistent camera pose ({0 if sol is None else sol[2]} of {len(uv)} "
                      f"matches agree; needs {MIN_INLIERS}): not enough overlap with the map")
        return out
    T, Kq, inliers = sol
    T, med, ok = epipolar_gate(T, Kq, matches, points, uv, xyz, inliers, name=str(q.image))
    if not ok:
        out.reason = (f"its pose contradicts its matches with the map's keyframes (median "
                      f"epipolar distance {med:.2f}° > {MAX_EPIPOLAR_DEG}°)")
        return out
    out.T_map_cam, out.K, out.inliers = T, Kq, inliers
    return out


def epipolar_gate(T: Pose, K: Intrinsics, matches: list[_Match], points: _MapPoints,
                  uv: NDArray[Any], xyz: NDArray[Any], inliers: int, name: str = ""
                  ) -> tuple[Pose, float, bool]:
    """(pose, median epipolar distance, accepted) of a located pose judged against its verified
    matches to the stored keyframe poses (``match_residual_deg``, ``MAX_EPIPOLAR_DEG`` from
    ``EPIPOLAR_MIN_MATCHES`` matches). Seen from the keyframes' spot (``rotation_dominant``), a
    few centimetres of centre error — the depth's scale error — become tenths of a degree of
    epipolar distance: a pose above the limit is refined against the keyframe poses
    (``refine_to_keyframes``), and the refined pose is accepted only within the bounds of that
    explanation — its centre moved at most ``REFINE_MAX_MOVE_M``, it turned at most the angle
    that move subtends at the median scene depth, and it still agrees with
    ``REFINE_KEEP_INLIERS`` of the ``inliers`` of the 2D-3D correspondences ``uv``, ``xyz`` the
    pose was solved from."""
    med, n = match_residual_deg(T, K, matches, points.frames)
    if n < EPIPOLAR_MIN_MATCHES or med <= MAX_EPIPOLAR_DEG:
        return T, med, True
    dominant, depth = _viewpoint(T, matches, points)
    if not dominant:
        return T, med, False
    T2 = refine_to_keyframes(T, K, matches, points.frames)
    med2, _ = match_residual_deg(T2, K, matches, points.frames)
    moved = float(np.linalg.norm(T2.t - T.t))
    turned = _rot_deg(T.R, T2.R)
    max_turn = float(np.degrees(np.arctan2(REFINE_MAX_MOVE_M, depth)))
    kept = count_inliers(T2, K, uv, xyz)
    accept = (med2 <= MAX_EPIPOLAR_DEG and moved <= REFINE_MAX_MOVE_M and turned <= max_turn
              and kept >= REFINE_KEEP_INLIERS * inliers)
    log.info("%s: epipolar %.3f° > %.2f°, refined against the keyframe poses: %.3f°, moved %.1f cm "
             "(max %.0f), turned %.2f° (max %.2f°), %d of %d inliers kept: %s", name or "pose", med,
             MAX_EPIPOLAR_DEG, med2, 100 * moved, 100 * REFINE_MAX_MOVE_M, turned, max_turn, kept,
             inliers, "accepted" if accept else "rejected")
    return (T2, med2, True) if accept else (T, med, False)


def count_inliers(T: Pose, K: Intrinsics, uv: NDArray[Any], xyz: NDArray[Any],
                  max_error_px: float = POSE_MAX_ERROR_PX) -> int:
    """2D-3D correspondences the camera ``T`` (camera-to-map), ``K`` reprojects within
    ``max_error_px``."""
    c = (np.asarray(xyz, np.float64).reshape(-1, 3) - T.t) @ T.R
    z = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * c[:, 0] / z + K.cx
        v = K.fy * c[:, 1] / z + K.cy
    err = np.hypot(u - np.asarray(uv)[:, 0], v - np.asarray(uv)[:, 1])
    return int(np.sum((z > 0) & (err <= max_error_px)))


def _rot_deg(A: NDArray[Any], B: NDArray[Any]) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1.0, 1.0))))


def _viewpoint(T: Pose, matches: list[_Match], points: _MapPoints) -> tuple[bool, float]:
    """(``rotation_dominant``, median scene depth of the matched keyframes in metres, by match
    count)."""
    w, mv, ratio, depth = [], 0, [], []
    for m in matches:
        fr = points.frames.get(m.keyframe)
        if fr is None:
            continue
        xyz, ok = points._depth_points(fr, m.uv_k)
        z = ((xyz[ok] - fr.T_map_cam.t) @ fr.T_map_cam.R)[:, 2]
        if not len(z) or np.median(z) <= 0:
            continue
        w.append(len(m.idx_k))
        mv += 0 if model_points_trusted(fr) else len(m.idx_k)
        depth.append(float(np.median(z)))
        ratio.append(float(np.linalg.norm(T.t - fr.T_map_cam.t)) / depth[-1])
    if not w:
        return False, float("inf")
    z_med = float(np.median(np.repeat(depth, w)))
    return 2 * mv > sum(w) or float(np.median(np.repeat(ratio, w))) < ROTATION_BASELINE, z_med


def rotation_dominant(T: Pose, matches: list[_Match], points: _MapPoints) -> bool:
    """Whether the located camera sees the map from (almost) the same spot as the keyframes it
    matches, by match count: most matches are with multi-view keyframes (a map of a camera
    turning in place), or the median baseline to the matched keyframes is below
    ``ROTATION_BASELINE`` of their median scene depth."""
    return _viewpoint(T, matches, points)[0]


def _signed_epipolar_deg(R: NDArray[Any], C: NDArray[Any], K: Intrinsics, fr: store.FrameRecord,
                         uv_q: NDArray[Any], uv_k: NDArray[Any]) -> NDArray[np.float64]:
    """``sfm.epipolar_deg`` of the query (camera-to-map ``R``, ``C``) and keyframe ``fr``,
    signed (a least-squares residual)."""
    Rk, Ck = fr.T_map_cam.R, fr.T_map_cam.t
    t = Rk.T @ (C - Ck)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    F = np.linalg.inv(fr.K.K()).T @ tx @ (Rk.T @ R) @ np.linalg.inv(K.K())
    xa = np.column_stack([uv_q, np.ones(len(uv_q))])
    xb = np.column_stack([uv_k, np.ones(len(uv_k))])
    la, lb = xa @ F.T, xb @ F
    num = np.sum(xb * la, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * (num / np.hypot(la[:, 0], la[:, 1]) / fr.K.fx
                   + num / np.hypot(lb[:, 0], lb[:, 1]) / K.fx)
    return np.degrees(np.nan_to_num(d, nan=0.0))


def refine_to_keyframes(T: Pose, K: Intrinsics, matches: list[_Match],
                        frames: dict[str, store.FrameRecord]) -> Pose:
    """The located pose refined against its verified matches to the stored keyframe poses:
    iteratively reweighted (Cauchy) least squares of their epipolar distances over the rotation
    and the centre, with a quadratic prior (outside the robust weights) holding the centre near
    its starting point (``REFINE_CENTRE_PRIOR_M``), which the matches of a near-zero baseline
    barely constrain."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    used = [(frames[m.keyframe], m.uv_q, m.uv_k) for m in matches if m.keyframe in frames]
    used = [(fr, uq[sel], uk[sel]) for fr, uq, uk in used
            for sel in [np.linspace(0, len(uq) - 1, min(len(uq), REFINE_MAX_PER_KEYFRAME)).astype(int)]]
    f_scale = MAX_EPIPOLAR_DEG / 2

    def epipolar(p: NDArray[np.float64]) -> NDArray[np.float64]:
        R = Rotation.from_rotvec(p[:3]).as_matrix() @ T.R
        return np.concatenate([_signed_epipolar_deg(R, p[3:], K, fr, uq, uk)
                               for fr, uq, uk in used])

    p = np.concatenate([np.zeros(3), T.t])
    for _ in range(REFINE_IRLS_ROUNDS):
        sw = 1.0 / np.sqrt(1.0 + (epipolar(p) / f_scale) ** 2)  # sqrt of the Cauchy weights

        def residuals(q: NDArray[np.float64], sw: NDArray[np.float64] = sw
                      ) -> NDArray[np.float64]:
            return np.concatenate([sw * epipolar(q), (q[3:] - T.t) / REFINE_CENTRE_PRIOR_M
                                   * f_scale])

        p = least_squares(residuals, p, max_nfev=50).x
    return Pose(Rotation.from_rotvec(p[:3]).as_matrix() @ T.R, p[3:])


# ------------------------------------------------------------------------------------------------
# results


def first_located_key(reader: store.MapReader) -> int:
    """Frame key of input image 0: past every keyframe index the map has used."""
    return max([int(reader.meta.get("next_frame_index", 0))] +
               [fr.index + 1 for fr in reader.frames])


def located_blocks(results: Sequence[Located], base: int
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(coordinate systems, streams, frames) of the located cameras: each its own sensor
    coordinate system and camera stream ``located_<k>`` and frame ``base + k`` (k: position in
    ``-i``), marked ``located``, with the ``located_<k>_to_map`` transform."""
    from oh_my_slam.schema import openlabel as ol

    css, streams, frames = {}, {}, {}
    for r in results:
        if r.T_map_cam is None or r.K is None:
            continue
        css[r.key] = ol.sensor_cs("map")
        streams[r.key] = ol.camera_stream(r.K, uri=str(r.image), description=f"located {r.image}")
        frames[str(base + r.index)] = ol.frame(
            timestamp=float(base + r.index),
            stream_uris={r.key: str(r.image)},
            transforms={f"{r.key}_to_map": ol.transform(r.key, "map", r.T_map_cam)},
            located=True, image=str(r.image), inliers=r.inliers)
    return css, streams, frames


def scene_json(reader: store.MapReader, results: Sequence[Located], mode: str) -> bytes:
    """``-t single``: the located cameras only; ``-t full``: the map's scene (``export.
    scene_bytes``, as ``update -t full`` returns it) plus the located cameras."""
    from oh_my_slam.mapping import export
    from oh_my_slam.schema import openlabel as ol

    css, streams, frames = located_blocks(results, first_located_key(reader))
    if mode == "single":
        md = ol.metadata(reader.root.name, tagged_file=str(reader.root), tool="mapper",
                         map_frame=reader.meta.get("map_frame"), scope="single")
        return json_payload_bytes(ol.document(md, {}, coordinate_systems={
            "map": ol.map_cs(list(css)), **css}, streams=streams, frames=frames))
    doc = json.loads(export.scene_bytes(reader))
    root = doc["openlabel"]
    root["coordinate_systems"]["map"]["children"] += list(css)
    root["coordinate_systems"].update(css)
    root.setdefault("streams", {}).update(streams)
    root.setdefault("frames", {}).update(frames)
    root["frame_intervals"] = ol.frame_intervals([int(k) for k in root["frames"]])
    # keep the key order of a scene document (objects last)
    root["objects"] = root.pop("objects")
    return json_payload_bytes(doc)


def pose_comment(r: Located) -> str:
    """The PLY header line of one input image: ``located_<k> {json}`` with, for a located image,
    the camera-to-map ``transform_src_to_dst`` and the ``stream_properties`` (intrinsics) of its
    OpenLABEL frame and stream; ``"located": false`` for one the map could not localise."""
    from oh_my_slam.schema import openlabel as ol

    d: dict[str, Any] = {"image": str(r.image), "located": r.located}
    if r.T_map_cam is not None and r.K is not None:
        d["transform_src_to_dst"] = ol.transform_data(r.T_map_cam)
        d["stream_properties"] = ol.camera_stream(r.K)["stream_properties"]
    return f"{r.key} {json.dumps(d, separators=(',', ':'), ensure_ascii=True)}"


def visible_points(xyz: NDArray[Any], cams: Sequence[tuple[Pose, Intrinsics]]) -> NDArray[np.bool_]:
    """Points seen by any of the cameras (``T_map_cam``, full-resolution intrinsics): in the
    frustum and not hidden behind nearer points (coarse z-buffer, ``VIS_*``)."""
    xyz = np.asarray(xyz).reshape(-1, 3)
    keep = np.zeros(len(xyz), bool)
    for T, K in cams:
        s = VIS_GRID_SIDE / max(K.width, K.height)
        gw, gh = max(1, int(np.ceil(K.width * s))), max(1, int(np.ceil(K.height * s)))

        def project(sl: slice, T: Pose = T, K: Intrinsics = K, s: float = s, gw: int = gw,
                    gh: int = gh) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
            c = (np.asarray(xyz[sl], np.float64) - T.t) @ T.R  # map -> camera
            z = c[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = K.fx * c[:, 0] / z + K.cx, K.fy * c[:, 1] / z + K.cy
            inside = (z > VIS_NEAR) & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
            cell = np.full(len(z), -1, np.int64)
            cell[inside] = (np.minimum((v[inside] * s).astype(np.int64), gh - 1) * gw
                            + np.minimum((u[inside] * s).astype(np.int64), gw - 1))
            return cell, z

        zmin = np.full(gw * gh, np.inf)
        chunks = [slice(i, i + VIS_CHUNK) for i in range(0, len(xyz), VIS_CHUNK)]
        for sl in chunks:
            cell, z = project(sl)
            ok = cell >= 0
            np.minimum.at(zmin, cell[ok], z[ok])
        zmin = _min3x3(zmin.reshape(gh, gw)).reshape(-1)
        for sl in chunks:
            cell, z = project(sl)
            ok = cell >= 0
            vis = np.zeros(len(z), bool)
            vis[ok] = z[ok] <= zmin[cell[ok]] * (1.0 + VIS_REL_TOL)
            keep[sl] |= vis
    return keep


def _min3x3(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum over each cell's 3 x 3 neighbourhood."""
    p = np.pad(a, 1, constant_values=np.inf)
    h, w = a.shape
    out = p[1:h + 1, 1:w + 1].copy()
    for i in range(3):
        for j in range(3):
            np.minimum(out, p[i:i + h, j:j + w], out=out)
    return out


def ply_payload(reader: store.MapReader, results: Sequence[Located], mode: str,
                attrs: CloudAttrs) -> bytes:
    """``-f ply``: the map points visible from the located cameras (single) or the whole map cloud
    (full), with one header line per input image (``pose_comment``)."""
    from oh_my_slam.mapping import export
    from oh_my_slam.segmentation.cloud import cloud_ply

    _, objs = export.map_objects(reader)
    cloud = export.map_cloud(reader)
    if mode == "single":
        cams = [(r.T_map_cam, r.K) for r in results if r.T_map_cam is not None and r.K is not None]
        cloud = cloud.subset(np.flatnonzero(visible_points(cloud.xyz, cams)))
    return cloud_ply(export.map_source(cloud, objs, reader.frames), attrs,
                     extra_comments=[pose_comment(r) for r in results])


# ------------------------------------------------------------------------------------------------
# entry point


MAX_ATTEMPTS = 3  # a concurrent update committing during locate: start again on the new map


def map_identity(root: Path) -> tuple[bytes, bytes | None]:
    """What changes with every commit of an update: ``map.json`` (written last) and the staged
    commit marker (present while a commit is being applied)."""
    def read(p: Path) -> bytes | None:
        try:
            return p.read_bytes()
        except FileNotFoundError:
            return None

    return read(root / store.MAP_JSON) or b"", read(root / store.STAGING / store.COMMIT)


def locate(reader: store.MapReader, images: Sequence[Path], *, mode: str = "single",
           fmt: str = "json", attrs: CloudAttrs | None = None, client: Any = None,
           progress: Progress = _progress) -> LocateResult:
    """Camera pose of each of ``images`` (``resolve_images``) in the map ``reader`` reads
    (``open_map``); the map is never modified. ``client``: the inference client, needed only for
    retrieval on a map of more than ``UPDATE_EXHAUSTIVE_MAX`` keyframes (connected on demand).

    Every result comes from one state of the map: when a concurrent ``mapper.sh update`` commits
    while it runs (``map_identity`` changed, or a file it read vanished), the map is opened again
    and the images located again, at most ``MAX_ATTEMPTS`` times."""
    if mode not in ("single", "full") or fmt not in ("json", "ply"):
        raise UsageError(f"locate: unknown mode {mode!r} or format {fmt!r}")
    with timing.collect() as tm:
        payload, results = _locate(reader, list(images), mode, fmt, attrs or CloudAttrs(), client,
                                   progress)
        return LocateResult(payload, results, tm.to_dict())


def _locate(reader: store.MapReader, images: list[Path], mode: str, fmt: str, attrs: CloudAttrs,
            client: Any, progress: Progress) -> tuple[bytes, list[Located]]:
    """``_locate_once`` until one attempt saw a single state of the map; only that attempt's
    results are reported (its unlocated images, or the failure when none is located)."""
    root = reader.root
    for attempt in range(MAX_ATTEMPTS):
        timing.count(attempts=attempt + 1)
        ident = map_identity(root)
        if attempt or _stale(reader):
            progress("the map changed during locate (a concurrent update); starting again")
            reader = open_map(root)
        try:
            payload, results = _locate_once(reader, images, mode, fmt, attrs, client, progress)
        except FileNotFoundError:
            if map_identity(root) == ident:
                raise
            continue
        if map_identity(root) != ident:
            continue
        for r in results:
            if not r.located:
                log.warning("%s: not located — %s; take it closer to where the map's images "
                            "were taken, of the same surroundings", r.image, r.reason)
        n = sum(r.located for r in results)
        timing.count(images=len(results), located=n, map_frames=len(reader.frames))
        if payload is None:
            raise InputError("none of the images could be located in the map (not enough "
                             "overlap with it); the map is unchanged")
        progress(f"located {n} of {len(results)} images")
        return payload, results
    raise InputError(f"the map {root} kept changing during locate (a concurrent mapper.sh "
                     "update); retry once the update is done")


def _stale(reader: store.MapReader) -> bool:
    """The map has committed an update since ``reader`` opened it."""
    try:
        return bool(reader.read_json(store.MAP_JSON) != reader.meta)
    except FileNotFoundError:
        return True


def _locate_once(reader: store.MapReader, images: list[Path], mode: str, fmt: str,
                 attrs: CloudAttrs, client: Any, progress: Progress
                 ) -> tuple[bytes | None, list[Located]]:
    """(payload, or None when no image is located; the result of each image) in the map state
    ``reader`` read."""
    stage = timing.stage
    with stage(Stage.SETUP):
        queries = [_Query(p, k, f"{QUERY_PREFIX}{k:06d}.jpg", upright_size(p), exif_intrinsics(p))
                   for k, p in enumerate(images)]
    progress(f"locating {len(images)} images in map {reader.root} ({len(reader.frames)} "
             "keyframes)")
    work = Path(tempfile.mkdtemp(prefix="locate-", dir=paths.scratch_dir()))
    try:
        with stage(Stage.FEATURES_MATCHING):
            sfm, db = _extract(reader, queries, work)
            sfm.match_pairs(_pairs(reader, queries, db, client),
                            {v.image_id: k for k, v in db.items()})
            matches = _query_matches(sfm.db, queries, db)
        with stage(Stage.POSE):
            points = _MapPoints(reader)
            results = []
            for q in queries:
                K, refine = _query_intrinsics(reader, sfm, q, db)
                results.append(_locate_one(q, K, refine, matches[q.name], points))
                timing.progress(len(results), len(queries))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not any(r.located for r in results):
        return None, results
    with stage(Stage.EXPORT):
        payload = ply_payload(reader, results, mode, attrs) if fmt == "ply" else \
            scene_json(reader, results, mode)
    return payload, results
