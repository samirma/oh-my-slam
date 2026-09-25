"""``mapper.sh update``: create or extend a persistent map.

Order (design, "Mapping update"): lock + stage → inputs/keyframes → per-keyframe geometry,
gravity and descriptor (``reconstruction.api``) with detections (``segmentation.api``) → features,
matching, poses → focal re-run rule → metric scale, gravity and map frame (new maps) →
per-keyframe depth alignment → latest wins → objects → fused cloud → scene export → commit.

The keyframes of one update are one observation of the scene: latest wins, object association
and the cloud's colours and labels do not depend on their order (``validity``, ``objects``,
``geometry``); only a later update wins over an earlier one. Capture timestamps are never read.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core import paths, timing
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.errors import RegistrationError
from oh_my_slam.core.images import upright_size
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import frame as mframe
from oh_my_slam.mapping import ingest, retrieval, store, validity
from oh_my_slam.mapping.sfm import (
    MIN_PLACED_FRACTION,
    ROTATION_BASELINE_RATIO,
    ROTATION_PAIR_FRACTION,
    CameraPrior,
    Sfm,
    SfmModel,
    check_versions,
)
from oh_my_slam.reconstruction.api import KEYFRAME_TOKENS, FrameReconstruction, reconstruct_image
from oh_my_slam.reconstruction.depth import fit_frame_scale
from oh_my_slam.reconstruction.gravity import DEFAULT_UP_CAM
from oh_my_slam.segmentation.api import Detection, detect_alongside

if TYPE_CHECKING:
    from oh_my_slam.mapping.geometry import MapGeometry
    from oh_my_slam.mapping.objects import ObjectState

log = get_logger("oh_my_slam.mapper")

KEYFRAME_GRID_SIDE = 768  # depth and segmentation grid of map keyframes
FOCAL_RERUN_REL = 0.03
DEPTH_SCALE_RANGE = (0.8, 1.25)
SEQ_OVERLAP = 12
LOOP_TOP_K = 10
LOOP_MIN_GAP = 30
PHOTO_EXHAUSTIVE_MAX = 200
UPDATE_EXHAUSTIVE_MAX = 150
RETRIEVAL_TOP_K = 30
MV_CHUNK = 24  # gate G6
MV_ANCHORS = 4
REJECT_SCALE = (0.5, 2.0)


@dataclass
class NewFrame:
    kf: ingest.Keyframe
    frame: FrameReconstruction
    dets: list[Detection]
    full_size: tuple[int, int]
    record: store.FrameRecord | None = None
    depth: NDArray[np.float32] | None = None  # aligned metric depth (grid)


@dataclass
class UpdateContext:
    tx: store.MapTransaction
    meta: dict[str, Any]
    old_frames: list[store.FrameRecord]
    new: list[NewFrame]
    update_id: int
    work: Path
    rejected: list[str] = field(default_factory=list)
    model: SfmModel | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    # refined multi-view keyframes: (median match residual in degrees, matches)
    pose_support: dict[str, tuple[float, int]] = field(default_factory=dict)


Progress = Callable[[str], None]


def _progress(msg: str) -> None:
    log.info(msg)


# ------------------------------------------------------------------------------------------------
# per-keyframe inference


def _infer_frames(kfs: list[ingest.Keyframe], work: Path, client: Any, progress: Progress
                  ) -> list[NewFrame]:
    """Geometry, gravity and detections for every keyframe (two frames in flight)."""

    def one(kf: ingest.Keyframe) -> NewFrame:
        frame, dets = reconstruct_and_detect_keyframe(kf, work / kf.name, client)
        return NewFrame(kf, frame, dets, upright_size(kf.path))

    out: list[NewFrame] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as pool:
        for i, nf in enumerate(pool.map(one, kfs), start=1):
            out.append(nf)
            if i % 10 == 0 or i == len(kfs):
                progress(f"inference {i}/{len(kfs)} keyframes ({time.perf_counter() - t0:.0f} s)")
    return out


def _reconstruct_keyframe(path: Path, client: Any, intrinsics: Intrinsics | None,
                          work_dir: Path | None = None, first: bool = True
                          ) -> FrameReconstruction:
    """Keyframe geometry with the mapper's settings (grid, tokens); the first pass also asks for
    gravity and the retrieval descriptor."""
    return reconstruct_image(path, intrinsics=intrinsics, max_side=KEYFRAME_GRID_SIDE,
                             num_tokens=KEYFRAME_TOKENS, want_gravity=first,
                             want_descriptor=first, work_dir=work_dir, client=client)


def reconstruct_and_detect_keyframe(kf: ingest.Keyframe, work: Path, client: Any
                                    ) -> tuple[FrameReconstruction, list[Detection]]:
    """Depth, gravity and descriptor from reconstruction, detections from segmentation (at the
    default threshold, requested while the reconstruction runs)."""
    own = client.clone()
    try:
        return detect_alongside(
            kf.path, own, lambda c: _reconstruct_keyframe(kf.path, c, kf.exif, work),
            max_side=KEYFRAME_GRID_SIDE)
    finally:
        if own is not client:
            own.close()


# ------------------------------------------------------------------------------------------------
# SfM


def _camera_prior(new: list[NewFrame], old: list[store.FrameRecord]) -> CameraPrior:
    w, h = new[0].full_size
    focals = [nf.frame.intrinsics.fx for nf in new if nf.full_size == (w, h)]
    same = [f for f in old if (f.width, f.height) == (w, h)]
    existing = same[-1].camera_id if same and all(nf.kf.exif is None for nf in new) else None
    # same image size as an existing camera (e.g. more frames of the same video): reuse it
    return CameraPrior(w, h, focal=float(np.median(focals)), existing_id=existing)


def _pairs_new_map(new: list[NewFrame], is_video: bool) -> set[tuple[int, int]]:
    ids = [nf.kf.index for nf in new]
    desc = np.stack([nf.frame.descriptor for nf in new]) if new[0].frame.descriptor is not None \
        else None
    if not is_video and len(ids) <= PHOTO_EXHAUSTIVE_MAX:
        return retrieval.all_pairs(ids)
    pairs = retrieval.sequential_pairs(ids, SEQ_OVERLAP)
    if desc is not None:
        if is_video:
            pairs |= retrieval.top_k_pairs(desc, desc, LOOP_TOP_K, ids, ids, LOOP_MIN_GAP)
        else:
            pairs |= retrieval.top_k_pairs(desc, desc, RETRIEVAL_TOP_K, ids, ids, 0)
    return pairs


def _pairs_update(ctx: UpdateContext, is_video: bool) -> set[tuple[int, int]]:
    new_ids = [nf.kf.index for nf in ctx.new]
    old_ids = [f.index for f in ctx.old_frames]
    pairs = retrieval.sequential_pairs(new_ids, SEQ_OVERLAP) if is_video else \
        retrieval.all_pairs(new_ids)
    if len(old_ids) <= UPDATE_EXHAUSTIVE_MAX:
        return pairs | retrieval.all_pairs(new_ids, old_ids)
    reader_desc = []
    keep_ids = []
    for f in ctx.old_frames:
        d = _stored_descriptor(ctx, f)
        if d is not None:
            reader_desc.append(d)
            keep_ids.append(f.index)
    if any(nf.frame.descriptor is None for nf in ctx.new):
        return pairs | retrieval.all_pairs(new_ids, old_ids)
    new_desc = np.stack([nf.frame.descriptor for nf in ctx.new])
    if reader_desc:
        pairs |= retrieval.top_k_pairs(new_desc, np.stack(reader_desc), RETRIEVAL_TOP_K,
                                       new_ids, keep_ids, 0)
    return pairs


@dataclass
class PoolView:
    """A keyframe that can anchor a multi-view chunk (already posed) or is to be posed."""

    name: str
    image: Path
    K: Intrinsics
    descriptor: NDArray[np.float32] | None
    pose: Pose | None = None


def _pick_anchors(chunk: list[PoolView], pool: list[PoolView], k: int) -> list[PoolView]:
    """The ``k`` posed pool views most similar (DINOv2 descriptor) to the chunk."""
    posed = [p for p in pool if p.pose is not None]
    if not posed:
        return []
    if any(p.descriptor is None for p in posed) or any(c.descriptor is None for c in chunk):
        return posed[-k:]
    sim = retrieval.similarity(np.stack([c.descriptor for c in chunk]),
                               np.stack([p.descriptor for p in posed]))
    order = np.argsort(-sim.max(axis=0))
    return [posed[i] for i in order[:k]]


def _multiview_poses(ctx: UpdateContext, todo: list[PoolView], pool: list[PoolView],
                     client: Any) -> dict[str, Pose]:
    """Chunked, pose-anchored MapAnything poses (metric).

    Each chunk carries up to ``MV_ANCHORS`` already-posed views (the most similar ones); the
    model returns poses in its own frame (the first view's), so every chunk is brought into the
    reference frame with the rotation/translation that maps its anchors onto their known poses
    (see ``frame.align_by_poses``; valid for rotation-only rigs). The first chunk of a new map
    has no anchors and defines the frame."""
    from oh_my_slam.reconstruction.multiview import run_multiview

    posed: dict[str, Pose] = {}
    by_name = {p.name: p for p in pool}
    i = 0
    while i < len(todo):
        anchors = _pick_anchors(todo[i:i + MV_CHUNK], list(by_name.values()), MV_ANCHORS)
        chunk = todo[i:i + MV_CHUNK - len(anchors)]
        views = anchors + chunk
        res, _ = run_multiview(
            [v.image for v in views], ctx.work / "mv", intrinsics=[v.K for v in views],
            poses=[v.pose for v in views], client=client,
        )
        out = {v.name: r.pose for v, r in zip(views, res, strict=True)}
        align = Pose.identity()
        if anchors:
            align = mframe.align_by_poses([out[a.name] for a in anchors],
                                          [a.pose for a in anchors if a.pose is not None])
        for v in chunk:
            T = align.compose(out[v.name])
            posed[v.name] = T
            by_name[v.name] = PoolView(v.name, v.image, v.K, v.descriptor, T)
        i += len(chunk)
    return posed


def _new_pool(ctx: UpdateContext) -> list[PoolView]:
    return [PoolView(f"{nf.kf.name}.jpg", nf.kf.path, nf.frame.intrinsics, nf.frame.descriptor)
            for nf in ctx.new]


def _stored_descriptor(ctx: UpdateContext, rec: store.FrameRecord) -> NDArray[np.float32] | None:
    p = ctx.tx.current(store.frame_file(rec.name, "descriptor.npy"))
    return np.load(p) if p.exists() else None


def _old_pool(ctx: UpdateContext) -> list[PoolView]:
    return [PoolView(Path(f.image).name, ctx.tx.current(f.image), f.K,
                     _stored_descriptor(ctx, f), f.T_map_cam) for f in ctx.old_frames]


def _sparse_scale(nf: NewFrame, model: SfmModel, name: str, min_points: int = 50) -> Any:
    """Robust scale of the keyframe's depth to its well-triangulated SfM points (``ScaleFit``),
    None without points."""
    uv, xyz = model.observations(name)
    if not len(xyz):
        return None
    z = model.pose(name).inverse().apply(xyz)[:, 2]
    pred = mframe.sample_depth_at(nf.frame.depth, uv, nf.full_size, nf.frame.K_grid)
    return fit_frame_scale(pred, z, min_points=min_points)


def _depth_consistent(ctx: UpdateContext, model: SfmModel, name: str) -> bool:
    """Registration sanity check: MoGe depth vs SfM depth ratio within [0.5, 2] (fewer than 20
    points cannot contradict the pose)."""
    nf = next(n for n in ctx.new if f"{n.kf.name}.jpg" == name)
    fit = _sparse_scale(nf, model, name, min_points=20)
    return fit is None or not fit.ok or (REJECT_SCALE[0] <= fit.scale <= REJECT_SCALE[1])


def _run_sfm(ctx: UpdateContext, is_video: bool, client: Any, progress: Progress) -> SfmModel:
    check_versions()
    tx = ctx.tx
    db = tx.clone_for_edit(store.SFM_DB)
    frames_dir = tx.stage("frames")
    sfm = Sfm(db, frames_dir, ctx.work / "sfm")
    names = {nf.kf.index: f"{nf.kf.name}.jpg" for nf in ctx.new}
    names.update({f.index: Path(f.image).name for f in ctx.old_frames})
    prior = _camera_prior(ctx.new, ctx.old_frames)
    t0 = time.perf_counter()
    with timing.stage("features_matching"):
        sfm.extract([f"{nf.kf.name}.jpg" for nf in ctx.new], prior)
        pairs = _pairs_new_map(ctx.new, is_video) if not ctx.old_frames else _pairs_update(
            ctx, is_video)
        n = sfm.match_pairs(pairs, names)
    timing.count(matched_pairs=n)
    progress(f"features + matching: {n} pairs in {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    new_names = {f"{nf.kf.name}.jpg" for nf in ctx.new}
    stats = sfm.two_view_stats(new_names if not ctx.old_frames else None)
    ctx.notes["two_view"] = stats
    rotation = stats["rotation_fraction"] > ROTATION_PAIR_FRACTION
    if ctx.old_frames:
        return _extend(ctx, sfm, new_names, rotation, client, progress, t0)
    component = sfm.largest_component(new_names)
    if len(component) < 2:
        component = new_names
    model: SfmModel | None = None
    if not rotation:
        model = sfm.map_global(ctx.work / "sfm_global")
        placed = 0 if model is None else len(set(model.registered) & new_names) / len(new_names)
        if model is None or placed < MIN_PLACED_FRACTION:
            model = sfm.map_incremental(ctx.work / "sfm_incr")
        placed = 0 if model is None else len(set(model.registered) & new_names) / len(new_names)
        if model is not None and placed >= MIN_PLACED_FRACTION:
            ratio = model.baseline_ratio()
            ctx.notes["baseline_ratio"] = ratio
            rotation = ratio < ROTATION_BASELINE_RATIO
            if not rotation:
                model = _complete_registration(sfm, model, new_names, ctx.work)
                progress(f"{model.method}: {len(model.registered)} keyframes posed in "
                         f"{time.perf_counter() - t0:.0f} s")
                return model
    reason = "rotation-dominant input" if rotation else "SfM placed too few frames"
    progress(f"multi-view fallback ({reason}); {len(component)} connected keyframes")
    todo = [v for v in _new_pool(ctx) if v.name in component]
    poses = _multiview_poses(ctx, todo, [], client)
    # the first keyframe holds the gauge; the depth fixes the scale, the matches the focal length
    poses, focal = _refine_multiview(ctx, sfm, poses, set(poses) - {todo[0].name},
                                     refine_focal=True, rotation=rotation)
    mv_model = sfm.triangulate_with_poses(poses, ctx.work / "sfm_mv", bundle=False,
                                          focal_scale=focal)
    mv_model.notes.update(reason=reason, metric=True)
    progress(f"multi-view + refinement: {len(mv_model.registered)} keyframes in "
             f"{time.perf_counter() - t0:.0f} s")
    return mv_model


def _keyframe_intrinsics(ctx: UpdateContext, sfm: Sfm, names: set[str]) -> dict[str, Intrinsics]:
    """Full-resolution intrinsics of new keyframes: the map's camera when they share one with
    stored keyframes (the map holds its refined focal length), else the database's."""
    by_camera = {f.camera_id: f.K for f in ctx.old_frames}
    return {n: by_camera.get(cid, K) for n, (cid, K) in sfm.image_intrinsics(names).items()}


def _refine_multiview(ctx: UpdateContext, sfm: Sfm, poses: dict[str, Pose], free: set[str],
                      refine_focal: bool, rotation: bool) -> tuple[dict[str, Pose], float]:
    """Refine the multi-view poses of the keyframes ``free`` with every verified feature match
    and the keyframes' monocular depth (``mapping.panorama``; staged for ``rotation``-dominant
    input); every other keyframe — this update's other posed keyframes and the map's — holds
    still. Returns all poses and the factor of the shared focal length (1 unless
    ``refine_focal``)."""
    from oh_my_slam.mapping import panorama

    new = {f"{nf.kf.name}.jpg": nf for nf in ctx.new}
    old = {Path(f.image).name: f for f in ctx.old_frames}
    with timing.stage("pose_refinement"):
        pairs = panorama.verified_matches(sfm.db, (set(poses) & set(new)) | set(old))
        linked = {n for p in pairs if p.a in free or p.b in free for n in (p.a, p.b)}
        K = _keyframe_intrinsics(ctx, sfm, linked & set(new))
        views: dict[str, panorama.View] = {}
        for n in sorted(linked):
            if n in new and n in K:
                nf = new[n]
                views[n] = panorama.View(poses[n], K[n], nf.frame.depth, nf.frame.K_grid,
                                         nf.full_size)
            elif n in old:
                f = old[n]
                views[n] = panorama.View(f.T_map_cam, f.K, store.load_depth(ctx.tx.current, f.name),
                                         f.K_grid, (f.width, f.height))
        refine = panorama.refine_turning if rotation else panorama.refine_poses
        fit = refine(pairs, views, free, refine_focal=refine_focal)
    ctx.notes["pose_refinement"] = fit.summary()
    ctx.pose_support.update({n: (fit.per_frame_deg.get(n, float("inf")),
                                 fit.per_frame_matches.get(n, 0)) for n in fit.poses})
    log.info("pose refinement of %d keyframes: median residual %.3f° -> %.3f° (%d pairs)",
             len(fit.poses), fit.median_before_deg, fit.median_after_deg, fit.pairs)
    return {**poses, **fit.poses}, fit.focal_scale


def _complete_registration(sfm: Sfm, model: SfmModel, names: set[str], work: Path) -> SfmModel:
    """Global mapping keeps the largest component only; keyframes it left out but that have
    verified matches to it are registered incrementally with the global poses fixed."""
    missing = names - set(model.registered)
    if not missing or not sfm.connected(set(model.registered), missing):
        return model
    base = work / "sfm_global_model"
    model.write(base)
    more = sfm.map_incremental(work / "sfm_complete", input_path=base, fix_existing=True)
    if more is None or len(more.registered) <= len(model.registered):
        return model
    more.method = model.method + "+incremental"
    return more


def _extend(ctx: UpdateContext, sfm: Sfm, new_names: set[str], rotation: bool, client: Any,
            progress: Progress, t0: float) -> SfmModel:
    """Update: place the new keyframes in the existing map frame (old frames fixed)."""
    old_names = {Path(f.image).name for f in ctx.old_frames}
    model_in = ctx.tx.root / store.SFM_MODEL
    if len(ctx.old_frames) < 3 or not (model_in / "images.bin").exists():
        return _remap_small(ctx, sfm, new_names, client, progress, t0)
    connected = sfm.connected(old_names, new_names)
    ctx.notes["connected_new"] = len(connected)
    if not connected:
        raise RegistrationError("none of the input frames overlaps the map (no verified "
                                "feature matches); the map is unchanged")
    model_in = ctx.tx.root / store.SFM_MODEL
    poses: dict[str, Pose] = {}
    method = "sfm-incremental"
    if not rotation and len(ctx.old_frames) >= 3:
        inc = sfm.map_incremental(ctx.work / "sfm_out", input_path=model_in, fix_existing=True)
        if inc is not None:
            for name in sorted(set(inc.registered) & connected):
                if _depth_consistent(ctx, inc, name):
                    poses[name] = inc.pose(name)
    missing = sorted(connected - set(poses))
    if missing:
        method = "multiview" if not poses else "sfm-incremental+multiview"
        pool = _old_pool(ctx) + [v for v in _new_pool(ctx) if v.name in poses]
        for v in pool:
            if v.name in poses:
                v.pose = poses[v.name]
        todo = [v for v in _new_pool(ctx) if v.name in set(missing)]
        progress(f"anchored multi-view for {len(todo)} keyframes"
                 + (" (rotation-dominant input)" if rotation else ""))
        mv = _multiview_poses(ctx, todo, pool, client)
        poses.update(mv)
        poses, _ = _refine_multiview(ctx, sfm, poses, set(mv), refine_focal=False,
                                     rotation=rotation)
        ctx.notes["mv_names"] = sorted(mv)
    model = sfm.extend_with_poses(model_in, poses, ctx.work / "sfm_ext", method)
    progress(f"registered {len(poses)}/{len(new_names)} new keyframes ({method}) in "
             f"{time.perf_counter() - t0:.0f} s")
    return model


def _remap_small(ctx: UpdateContext, sfm: Sfm, new_names: set[str], client: Any,
                 progress: Progress, t0: float) -> SfmModel:
    """Maps with fewer than 3 keyframes: re-run SfM over old + new keyframes, then bring the
    result into the existing map frame (metric scale, then the rigid transform that maps the
    old keyframes' SfM poses onto their stored map poses)."""
    import pycolmap

    tx = ctx.tx
    db = pycolmap.Database.open(str(sfm.db))
    try:
        in_db = {im.name for im in db.read_all_images()}
    finally:
        db.close()
    frames_dir = tx.stage("frames")
    missing = []
    for f in ctx.old_frames:
        name = Path(f.image).name
        dst = frames_dir / name
        if not dst.exists():
            shutil.copyfile(tx.root / f.image, dst)
        if name not in in_db:
            missing.append(f)
    if missing:
        sfm.extract([Path(f.image).name for f in missing],
                    CameraPrior(missing[0].width, missing[0].height, focal=missing[0].K.fx))
    ids = {nf.kf.index: f"{nf.kf.name}.jpg" for nf in ctx.new}
    ids.update({f.index: Path(f.image).name for f in ctx.old_frames})
    sfm.match_pairs(retrieval.all_pairs(sorted(ids)), ids)
    old_names = {Path(f.image).name for f in ctx.old_frames}
    if not sfm.connected(old_names, new_names):
        raise RegistrationError("none of the input frames overlaps the map (no verified "
                                "feature matches); the map is unchanged")
    model = sfm.map_global(ctx.work / "sfm_small")
    if model is None or not (old_names & set(model.registered)):
        model = sfm.map_incremental(ctx.work / "sfm_small_incr")
    if model is None or not (old_names & set(model.registered)):
        raise RegistrationError("the map's keyframes could not be re-posed with the new input")
    fds = _frame_depths(ctx)
    for f in ctx.old_frames:
        d = store.load_depth(tx.current, f.name)
        fds.append(mframe.FrameDepth(Path(f.image).name, d, f.K_grid, (f.width, f.height)))
    try:
        s = mframe.metric_scale(model, fds).scale
    except ValueError:
        s = 1.0
    model.transform(s, np.eye(3), np.zeros(3))
    pairs = [(model.pose(Path(f.image).name), f.T_map_cam) for f in ctx.old_frames
             if Path(f.image).name in model.registered]
    T = mframe.align_by_poses([p[0] for p in pairs], [p[1] for p in pairs])
    model.transform(1.0, T.R, T.t)
    ctx.notes["remap_small"] = {"old_frames": len(ctx.old_frames), "scale": s}
    progress(f"re-mapped the {len(ctx.old_frames)}-keyframe map with the new input in "
             f"{time.perf_counter() - t0:.0f} s")
    return model


# ------------------------------------------------------------------------------------------------
# frames, scale, alignment


def _rerun_focal(ctx: UpdateContext, model: SfmModel, client: Any, progress: Progress) -> None:
    """Re-run geometry with the COLMAP focal where it differs > 3 % from the one used."""
    redo = []
    for nf in ctx.new:
        name = f"{nf.kf.name}.jpg"
        if name not in model.registered or nf.kf.exif is not None:
            continue
        colmap_K = model.intrinsics(name)
        if abs(colmap_K.fx - nf.frame.intrinsics.fx) / nf.frame.intrinsics.fx > FOCAL_RERUN_REL:
            redo.append((nf, colmap_K))
    if not redo:
        return
    progress(f"re-running geometry for {len(redo)} keyframes with the SfM focal length")

    def one(item: tuple[NewFrame, Intrinsics]) -> None:
        nf, K = item
        own = client.clone()
        try:
            fr = _reconstruct_keyframe(nf.kf.path, own, K, first=False)
        finally:
            if own is not client:
                own.close()
        nf.frame.depth, nf.frame.valid = fr.depth, fr.valid
        nf.frame.intrinsics, nf.frame.K_grid = fr.intrinsics, fr.K_grid

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, redo))


def _frame_depths(ctx: UpdateContext) -> list[mframe.FrameDepth]:
    out = []
    for nf in ctx.new:
        g = nf.frame.gravity
        out.append(mframe.FrameDepth(
            name=f"{nf.kf.name}.jpg", depth=nf.frame.depth, K_grid=nf.frame.K_grid,
            full_size=nf.full_size, up_cam=None if g is None else g.up_cam,
            up_confidence=1.0 if g is None else g.confidence,
        ))
    return out


def _define_map_frame(ctx: UpdateContext, model: SfmModel, progress: Progress) -> None:
    fds = _frame_depths(ctx)
    method = "moge_over_sfm_median"
    if model.notes.get("metric"):
        # refined multi-view poses are in the units of the keyframes' metric depth already
        scale, method = mframe.ScaleResult(1.0, 0.0, {}, {}), "multiview_depth"
    else:
        try:
            scale = mframe.metric_scale(model, fds)
        except ValueError:
            # no frame with enough triangulated points: keep the (metric) multi-view scale
            scale = mframe.ScaleResult(1.0, 0.0, {}, {})
    poses = {n: model.pose(n) for n in model.registered}
    up = mframe.world_up(poses, fds)
    first = next(f"{nf.kf.name}.jpg" for nf in ctx.new if f"{nf.kf.name}.jpg" in poses)
    sim = mframe.map_transform(poses[first], up, scale.scale)
    model.transform(sim.s, sim.R, sim.t)
    ctx.meta["scale"] = {"sfm_to_metric": scale.scale, "spread": scale.spread,
                         "frames": len(scale.per_frame), "method": method}
    ctx.meta["map_frame"] = {
        "units": "m", "axes": "x-forward,y-left,z-up", "gravity_aligned": True,
        "origin": f"camera centre of {first}", "up_source": "geocalib+floor weighted mean",
    }
    progress(f"metric scale {scale.scale:.4g} (spread {scale.spread:.2f}); map frame at {first}")


def _single_image_map(ctx: UpdateContext) -> None:
    """One image: identity pose, map frame from its gravity (C14)."""
    nf = ctx.new[0]
    up = nf.frame.gravity.up_cam if nf.frame.gravity is not None else DEFAULT_UP_CAM
    sim = mframe.map_transform(Pose.identity(), up, 1.0)
    T = mframe.transform_pose(sim, Pose.identity())
    nf.record = _record(ctx, nf, T, "identity", nf.frame.intrinsics, camera_id=1, stats={})
    nf.depth = nf.frame.depth.copy()
    ctx.meta["scale"] = {"sfm_to_metric": 1.0, "spread": 0.0, "frames": 1, "method": "identity"}
    ctx.meta["map_frame"] = {"units": "m", "axes": "x-forward,y-left,z-up",
                             "gravity_aligned": True, "origin": f"camera centre of {nf.kf.name}",
                             "up_source": nf.frame.gravity.source if nf.frame.gravity else "none"}


def _record(ctx: UpdateContext, nf: NewFrame, T: Pose, source: str, K: Intrinsics,
            camera_id: int, stats: dict[str, Any]) -> store.FrameRecord:
    w, h = nf.full_size
    gw, gh = nf.frame.grid_size
    g = nf.frame.gravity
    return store.FrameRecord(
        index=nf.kf.index, name=nf.kf.name, image=f"frames/{nf.kf.name}.jpg",
        source=nf.kf.source, camera_id=camera_id, width=w, height=h,
        K=K, T_map_cam=T, grid_width=gw, grid_height=gh, pose_source=source,
        update_id=ctx.update_id, stats=stats,
        up_cam=None if g is None else [float(v) for v in g.up_cam],
    )


DENSE_REFS = 4
DENSE_MAX_SPREAD = 0.3


def _align_depths(ctx: UpdateContext, model: SfmModel) -> None:
    """Per-keyframe robust scale of MoGe depth to the metric map.

    Sparse: median ratio to the keyframe's well-triangulated SfM points (>= 50). Keyframes with
    too few points (low texture, pure rotation) are aligned densely to overlapping keyframes that
    are already aligned (``reconstruction.depth.dense_scale``). A sparse scale outside [0.5, 2]
    means the pose contradicts the keyframe's own depth: the keyframe is left out. Sparse scales
    outside [0.8, 1.25], inconsistent dense fits, unsupported keyframes and refined multi-view
    poses the matches do not support (``pose_supported``) are low confidence (excluded from
    fusion, latest wins and absence tests)."""
    from oh_my_slam.reconstruction.depth import dense_scale

    lo, hi = DEPTH_SCALE_RANGE
    pool: list[tuple[str, Any, Pose, Intrinsics]] = []  # (name, depth loader, pose, K_grid)
    cache: dict[str, NDArray[np.float32]] = {}

    def old_depth(name: str) -> Any:
        def load() -> NDArray[np.float32]:
            if name not in cache:
                cache[name] = store.load_depth(ctx.tx.current, name)
            return cache[name]
        return load

    for f in ctx.old_frames:
        if not f.low_confidence:
            pool.append((f.name, old_depth(f.name), f.T_map_cam, f.K_grid))
    # Points triangulated from multi-view (rotation-dominant) poses have no reliable depth.
    dense_only: set[str] = set(ctx.notes.get("mv_names", []))
    if model.method == "multiview":
        dense_only |= set(model.registered)
    pending = []
    for nf in ctx.new:
        name = f"{nf.kf.name}.jpg"
        if name not in model.registered:
            continue
        T = model.pose(name)
        K = model.intrinsics(name).with_source("colmap")
        stats: dict[str, Any] = dict(model.image_stats(name))
        if name in ctx.pose_support:
            res, n_matches = ctx.pose_support[name]
            stats.update(pose_residual_deg=round(res, 4) if np.isfinite(res) else None,
                         pose_matches=n_matches)
        fit = None if name in dense_only else _sparse_scale(nf, model, name)
        if fit is not None and fit.ok and fit.spread <= DENSE_MAX_SPREAD:
            s = fit.scale
            if not (REJECT_SCALE[0] <= s <= REJECT_SCALE[1]):
                # sparse points disagree with the keyframe's depth (e.g. thin see-through
                # structures in front): let the dense test decide, reject if it fails too
                stats["sparse_scale_rejected"] = float(s)
                pending.append((nf, T, K, name, stats))
                continue
            stats.update({"depth_scale_method": "sparse", "depth_scale_points": fit.inliers,
                          "depth_scale_spread": fit.spread})
            _set_aligned(ctx, nf, T, K, model, name, stats, s, not (lo <= s <= hi))
            pool.append((nf.kf.name, (lambda d=nf.depth: d), T, nf.record.K_grid))
        else:
            pending.append((nf, T, K, name, stats))
    for nf, T, K, name, stats in pending:
        if not pool:
            # first keyframe of a map without usable sparse depth: MoGe's metric depth as is
            stats.update({"depth_scale_method": "seed"})
            _set_aligned(ctx, nf, T, K, model, name, stats, 1.0, False)
            pool.append((nf.kf.name, (lambda d=nf.depth: d), T, nf.record.K_grid))
            continue
        fwd = T.R[:, 2]
        ranked = sorted(pool, key=lambda r: -float(r[2].R[:, 2] @ fwd)
                        + 0.1 * float(np.linalg.norm(r[2].t - T.t)))[:DENSE_REFS]
        K_grid = K.resized(*nf.frame.grid_size)
        fit = dense_scale(nf.frame.depth, K_grid.K(), T.matrix(),
                          [(ld(), Kg.K(), Tr.matrix()) for _, ld, Tr, Kg in ranked])
        if fit.ok:
            s = fit.scale
            low = fit.spread > DENSE_MAX_SPREAD
            stats.update({"depth_scale_method": "dense", "depth_scale_points": fit.inliers,
                          "depth_scale_spread": fit.spread,
                          "depth_scale_refs": [r[0] for r in ranked]})
        else:
            s, low = 1.0, True
            stats.update({"depth_scale_method": "none"})
        if "sparse_scale_rejected" in stats and (not fit.ok or low or not (
                REJECT_SCALE[0] <= s <= REJECT_SCALE[1])):
            # neither the sparse nor the dense test supports this pose: misregistered
            ctx.rejected.append(nf.kf.name)
            model.rec.deregister_frame(model.rec.find_image_with_name(name).frame_id)
            continue
        _set_aligned(ctx, nf, T, K, model, name, stats, s, low)
        if not nf.record.low_confidence:
            pool.append((nf.kf.name, (lambda d=nf.depth: d), T, nf.record.K_grid))


LEVEL_MAX_DEG = 10.0


def _level_with_floor(ctx: UpdateContext, model: SfmModel, progress: Progress) -> None:
    """Design step 8, second half: refine the map's up axis with the floor plane of the whole
    (depth-aligned) map — the lowest well-supported horizontal surface — and rotate the map about
    its origin so the floor normal is exactly +z."""
    from oh_my_slam.core.geometry import angle_between_deg, ransac_plane, rotation_between
    from oh_my_slam.mapping.objects import map_floor

    allp, fz = map_floor(ctx, per_frame=4000)
    if fz is None:
        return
    band = allp[np.abs(allp[:, 2] - fz) < 0.15]
    if len(band) < 500:
        return
    res = ransac_plane(band, 0.03, iterations=400, normal_prior=np.array([0.0, 0.0, 1.0]),
                       max_angle_deg=LEVEL_MAX_DEG)
    if res is None:
        return
    n = res[0]
    ang = angle_between_deg(n, [0.0, 0.0, 1.0])
    ctx.notes["floor_levelling_deg"] = ang
    if ang < 0.2:
        return
    R = rotation_between(n, np.array([0.0, 0.0, 1.0]))
    model.transform(1.0, R, np.zeros(3))
    for nf in ctx.new:
        if nf.record is not None:
            T = nf.record.T_map_cam
            nf.record.T_map_cam = Pose(R @ T.R, R @ T.t)
    progress(f"map levelled with the floor plane ({ang:.1f}° correction)")


def _set_aligned(ctx: UpdateContext, nf: NewFrame, T: Pose, K: Intrinsics, model: SfmModel,
                 name: str, stats: dict[str, Any], s: float, low: bool) -> None:
    nf.record = _record(ctx, nf, T, model.method, K, model.camera_id(name), stats)
    nf.record.depth_scale = float(s)
    # low confidence: an unreliable depth scale, or a multi-view pose the matches do not support
    nf.record.low_confidence = bool(low) or not validity.pose_supported(stats)
    nf.depth = (nf.frame.depth * s).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# persistence


def _stage_frames(ctx: UpdateContext) -> None:
    from oh_my_slam.core.images import png_bytes

    tx = ctx.tx
    for nf in ctx.new:
        if nf.record is None or nf.depth is None:
            continue
        name = nf.kf.name
        tx.save_npy(store.frame_file(name, "depth.npy"), nf.depth.astype(np.float16))
        tx.write_bytes(store.frame_file(name, "valid.png"), png_bytes(
            (nf.frame.valid & (nf.depth > 0)).astype(np.uint8) * 255))
        if nf.frame.descriptor is not None:
            tx.save_npy(store.frame_file(name, "descriptor.npy"),
                        nf.frame.descriptor.astype(np.float32))
    # keyframe images of rejected frames are not kept
    for name in ctx.rejected:
        p = tx.staging / "frames" / f"{name}.jpg"
        p.unlink(missing_ok=True)


def _frames_json(ctx: UpdateContext) -> list[store.FrameRecord]:
    records = list(ctx.old_frames) + [nf.record for nf in ctx.new if nf.record is not None]
    ctx.tx.write_json(store.FRAMES_JSON, {"frames": [r.to_dict() for r in records]})
    return records


def integrate(ctx: UpdateContext, progress: Progress
              ) -> tuple[list[store.FrameRecord], ObjectState, MapGeometry]:
    """Fold the update's placed keyframes into the map (staged): frames, latest wins, objects and
    the cloud. Returns (all frame records, object state, map geometry)."""
    from oh_my_slam.mapping import objects
    from oh_my_slam.mapping.geometry import build_geometry

    with timing.stage("persist_frames"):
        _stage_frames(ctx)
        records = _frames_json(ctx)
    with timing.stage("validity"):
        validity.apply_latest_wins(ctx, records, progress)
    with timing.stage("objects"):
        objs = objects.update_objects(ctx, records, progress)
    geo = build_geometry(ctx, records, objs, progress)  # stage cloud
    return records, objs, geo


# ------------------------------------------------------------------------------------------------
# entry point


@dataclass
class UpdateResult:
    payload: bytes
    new_frames: list[str]
    rejected: list[str]
    seconds: float
    timings: dict[str, Any] = field(default_factory=dict)


def update(map_dir: Path, inputs: list[Path], *, fps: float = ingest.DEFAULT_FPS,
           mode: str = "full", fmt: str = "json", attrs: CloudAttrs | None = None,
           client: Any = None, progress: Progress = _progress) -> UpdateResult:
    """``attrs``: point-cloud attributes of the ``fmt="ply"`` payload (map scope; default rgb)."""
    with timing.collect() as tm:
        return _update(map_dir, inputs, fps, mode, fmt, attrs or CloudAttrs(), client, progress,
                       tm)


def _update(map_dir: Path, inputs: list[Path], fps: float, mode: str, fmt: str,
            attrs: CloudAttrs, client: Any, progress: Progress, tm: timing.Timings
            ) -> UpdateResult:
    """``update`` with per-stage timings (``core.timing``; stages are exclusive and sequential,
    geometry/gravity/segmentation per keyframe are parts, server time per endpoint)."""
    from oh_my_slam.mapping import export
    from oh_my_slam.reconstruction.api import connect_server as connect

    stage = timing.stage
    t_start = time.perf_counter()
    spec = ingest.resolve_inputs(inputs)
    client = client or connect()
    work = Path(tempfile.mkdtemp(prefix="update-", dir=paths.scratch_dir()))
    try:
        with store.MapTransaction(map_dir) as tx:
            with stage("setup"):
                meta = store.read_meta_or_default(tx)
                old = store.read_frames(tx)
            update_id = int(meta.get("update_count", 0)) + 1
            progress(("creating map " if tx.created else "extending map ") + str(tx.root))
            with stage("ingest"):
                kfs = list(ingest.keyframes(spec, fps, tx.stage("frames"),
                                            int(meta.get("next_frame_index", 0))))
            progress(f"{len(kfs)} keyframes from {spec.kind}")
            timing.count(input_kind=spec.kind, keyframes_sampled=len(kfs), map_frames_before=len(old))
            with stage("inference"):
                new = _infer_frames(kfs, work, client, progress)
            ctx = UpdateContext(tx, meta, old, new, update_id, work)
            if not old and len(new) == 1:
                _single_image_map(ctx)
                model = None
            else:
                with stage("sfm"):
                    model = _run_sfm(ctx, spec.kind == "video", client, progress)
                registered = set(model.registered)
                ctx.rejected = [nf.kf.name for nf in new if f"{nf.kf.name}.jpg" not in registered]
                if len(ctx.rejected) == len(new):
                    raise RegistrationError(
                        "none of the input frames overlaps the map (nothing registered); "
                        "the map is unchanged")
                with stage("focal_rerun"):
                    _rerun_focal(ctx, model, client, progress)
                if not old:
                    with stage("map_frame"):
                        _define_map_frame(ctx, model, progress)
                with stage("depth_alignment"):
                    _align_depths(ctx, model)
                if not old:
                    with stage("map_frame"):
                        _level_with_floor(ctx, model, progress)
                if not any(nf.record is not None for nf in new):
                    raise RegistrationError(
                        "no input frame could be placed consistently in the map; "
                        "the map is unchanged")
                if ctx.rejected:
                    progress(f"left out {len(ctx.rejected)} unplaceable keyframes: "
                             + ", ".join(sorted(ctx.rejected)[:30]))
                with stage("persist_frames"):
                    model.write(tx.stage(store.SFM_MODEL))
            records, objs, geo = integrate(ctx, progress)
            new_names = [nf.kf.name for nf in new if nf.record is not None]
            timing.count(keyframes_registered=len(new_names), keyframes_rejected=len(ctx.rejected),
                         map_frames_after=len(records), objects=len(objs.objects),
                         sfm=None if model is None else model.method, **geo.stats)
            meta.update({
                "update_count": update_id,
                "next_frame_index": max([f.index for f in records] + [-1]) + 1,
                "next_object_id": objs.next_id,
            })
            record: dict[str, Any] = {
                "id": update_id, "at": time.time(), "inputs": [str(i) for i in inputs],
                "kind": spec.kind, "fps": fps if spec.kind == "video" else None,
                "frames_added": new_names, "frames_rejected": ctx.rejected,
                "sfm": None if model is None else model.method, "notes": _jsonable(ctx.notes),
                "objects": objs.summary,
            }
            meta.setdefault("updates", []).append(record)
            with stage("export"):
                scene_full = export.full_scene(tx.root, meta, records, objs.exported())
                tx.write_json(store.SCENE_JSON, scene_full)
                if fmt == "ply":
                    payload = export.ply_payload(geo, mode, records, objs, attrs)
                else:
                    payload = export.scene_payload(scene_full, mode, new_names, records, objs)
            # timings up to the commit (map.json is written by the commit itself)
            record["timings"] = _jsonable(tm.to_dict())
            with stage("commit"):
                tx.commit(meta)
            elapsed = time.perf_counter() - t_start
            progress(f"committed update {update_id}: {len(new_names)} keyframes added, "
                     f"{len(objs.objects)} objects, {elapsed:.0f} s")
            return UpdateResult(payload, new_names, ctx.rejected, elapsed, tm.to_dict())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=float))


