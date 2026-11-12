"""The update job of an anchored map: prior-first registration (``mapping.anchored``; README
"Prior-first registration").

A map whose first update carries a metric or drifting prior on a frame the checks keep is
anchored: its frames are placed by their priors and refined against the stored keyframes, and
only the new keyframes are fused. Its versions record ``engine: anchored`` and the identity
similarity: the mapper's frame is the map frame, and stored poses never move.

1. The checks (``update.Updater._check``) have run: frames they reject never reach the engine,
   so a job whose frames are all rejected ends in the time of the checks.
2. The monocular depth of every kept frame without a depth image is asked of the inference
   server, two frames in flight, while the engine registers the frames over the job's match
   graph (map.md §2: the outcome does not depend on the request order).
3. The engine (``AnchoredMap.update``) places, keeps and fuses the frames in a clone of the
   map's mapper folder; the detections of every frame, asked after the depth, are fused into the
   map's objects for the new keyframes only (``fuse_objects``: ``mapping.anchored_objects``);
   the job reports one ``InputStatus`` per frame and the quality report; if the gate passes, the
   clone is committed with the new version and its object records (``versions/<n>.objects.json``,
   read by ``service.objects``)."""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.images import on_grid, size_at_max_side
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import register_ma, sigmas, standpoint
from oh_my_slam.mapping.anchor import Sim3
from oh_my_slam.mapping.anchored import (
    TOO_CLOSE_M,
    AnchoredMap,
    Features,
    FrameIn,
    Keyframe,
    Placement,
    job_features,
    natural_key,
    pose_residual,
    rot_deg,
)
from oh_my_slam.mapping.anchored_objects import ObjectMap, fuse_objects
from oh_my_slam.mapping.quality import (
    displacement_quality,
    displacements,
    gate,
    reregistered,
    residual_stats,
    too_close,
)
from oh_my_slam.service.backend import MONO_GRID_SIDE, Mono
from oh_my_slam.service.store import POSE_STATES, clone_tree
from oh_my_slam.service.timings import fusion_class

if TYPE_CHECKING:
    from pathlib import Path

    from oh_my_slam.service.jobs import Job
    from oh_my_slam.service.update import Updater, UpdateWork, _Frame

INFERENCE_IN_FLIGHT = 2
# the other engine's keyframes joining the anchor pool mid-job (``_MaRun.more``): the most
# recent, after the stored keyframes and this engine's own frames
ANCHOR_ORDER = 2 * 10 ** 9
LOW_CONFIDENCE_SIGMA_M = 0.10  # as update.LOW_CONFIDENCE_SIGMA_M
NONE_ENGINE_ENV = "OH_MY_SLAM_NONE_ENGINE"
DEFAULT_NONE_ENGINE = "mapanything"


def none_engine() -> str:
    """The engine of the frames without a prior (``register_ma``): ``OH_MY_SLAM_NONE_ENGINE``,
    else ``DEFAULT_NONE_ENGINE``."""
    v = os.environ.get(NONE_ENGINE_ENV, DEFAULT_NONE_ENGINE).strip().lower()
    if v not in register_ma.ENGINES:
        raise ValueError(f"{NONE_ENGINE_ENV}={v!r}: expected one of {register_ma.ENGINES}")
    return v


def camera_only(w: UpdateWork) -> int:
    """The frames of a request MapAnything registers (``none_engine``): the monocular ones
    without a metric or drifting prior (map.md §2 step 8: frames with depth images go to the
    geometric engine first)."""
    if none_engine() != "mapanything":
        return 0
    return sum(f.trust not in ("metric", "drifting") and f.depth is None for f in w.frames)


def _trusted(fr: _Frame) -> bool:
    return fr.work.trust in ("metric", "drifting")


def _register_none(up: Updater, am: AnchoredMap, frames: list[_Frame], workdir: Path,
                   progress: Callable[[str], None], pairs: standpoint.Pairs | None = None,
                   feats: dict[int, Features] | None = None
                   ) -> tuple[list[Placement], float, float]:
    """The camera-only frames by MapAnything (``_MaRun``) against the map's confirmed
    keyframes. Returns the placements (frames' order) and the registration and fusion
    seconds."""
    run = _MaRun(up, am, frames, workdir, pairs or standpoint.Pairs(), feats or {})
    run.first()
    for fr in frames:
        pl = run.got[fr.work.index]
        progress(f"{fr.work.name}: {pl.status} ({pl.reason or pl.method})")
    return [run.got[fr.work.index] for fr in frames], run.reg_s, run.fuse_s


class _MaRun:
    """MapAnything over a job's camera-only frames (``register_ma``), each registered one added
    as a keyframe; its depth scale is the map's, and its camera height feeds the map's depth
    scale (``ScaleState``) as a monocular keyframe's does. Every frame is inferred once
    (``Registration.run``); order independence (map.md §2) re-tests the frames no reference
    confirmed (``no_overlap``) against what registered since from their output
    (``Registration.recheck``), and infers one again only while the pool of anchors grew
    since its last call (``Registration.retry``). The frames go in canonical order
    (name, camera). ``feats``: the frames' features by request index, the job's static content
    left out (``job_features``)."""

    def __init__(self, up: Updater, am: AnchoredMap, frames: list[_Frame], workdir: Path,
                 pairs: standpoint.Pairs, feats: dict[int, Features]) -> None:
        self.up, self.am, self.workdir, self.pairs, self.feats = up, am, workdir, pairs, feats
        self.frames = sorted(frames, key=lambda fr: (natural_key(fr.work.name), fr.work.index))
        self.got: dict[int, Placement] = {}
        self.kf_of: dict[str, str] = {}  # ``new<index>`` anchor key -> its keyframe's id
        self.reg: register_ma.Registration | None = None
        self.reg_s = self.fuse_s = 0.0

    def first(self) -> list[Keyframe]:
        """Infer every frame once and register what the map confirms; the keyframes added."""
        t = time.perf_counter()
        am = self.am
        am.save()  # this update's keyframes get their image and depth on disk: anchors
        stored = [register_ma.Stored(k.id, img, k.K, k.T, i, functools.partial(am.depth_of,
                                                                                 k.id),
                                     k.depth_kind == "sensor")
                  for i, k in enumerate(am.keyframes)
                  if k.state == "fused" for img in [am.image_path(k.id)] if img is not None]
        if not stored:  # nothing confirmed to register against (a camera-only map, no origin)
            for fr in self.frames:
                self.got[fr.work.index] = Placement(fr.work.name, "rejected", "no_overlap")
            self.reg_s += time.perf_counter() - t
            return []
        new = []
        for fr in self.frames:
            assert fr.path is not None and fr.measures is not None
            new.append(register_ma.New(fr.work.index, fr.path, fr.cam.canonical(),
                                       fr.depth_m if _has_depth(fr) else None,
                                       fr.measures.features, fr.ignore))
        calls = [0]

        def multiview(images: list[Path], Ks: list[Intrinsics], poses: list[Any]) -> Any:
            calls[0] += 1
            return self.up.backend.geometry_views(images, Ks, poses,
                                                  self.workdir / f"mv{calls[0]}")

        self.reg = register_ma.Registration(multiview, new, stored, self._ambiguous)
        self.reg.run()
        self.reg.recheck()  # across the calls: what later calls registered
        # a frame still unconfirmed is inferred again, once each time the pool grew since its
        # call (its first call's anchors may not have seen its view)
        self.reg.retry()
        self.reg_s += time.perf_counter() - t
        return self._place(self.frames)

    def _ambiguous(self, f: register_ma.New, key: str) -> bool:
        """The registration gate's rule (map.md §1 "Symmetric scenes", §2 step 2) for
        MapAnything (``register_ma.confirms``): whether the pair of new frame ``f`` and the view
        of anchor ``key`` (a keyframe id, or ``new<index>`` for a frame of this job) is
        ambiguous, through the job's two-view fits (``standpoint.ambiguous``)."""
        def view(fr: _Frame) -> standpoint.View | None:
            got = self.feats.get(fr.work.index)
            d = _depth_image(fr)
            return None if got is None else standpoint.View(
                fr.work.name, fr.cam.canonical(), got, np.eye(3), np.zeros(3),
                depth=lambda: d)

        by_index = {fr.work.index: fr for fr in self.frames}
        a = view(by_index[f.index])
        if key.startswith("new"):
            b = view(by_index[int(key[3:])])
        else:
            k = self.am.keyframe(key)
            b = None if k is None else standpoint.keyframe_view(self.am, k)
        return a is not None and b is not None and standpoint.ambiguous(a, b, self.pairs)

    def more(self, extra: list[Keyframe]) -> list[Keyframe]:
        """The frames still unconfirmed tested again against ``extra`` (keyframes the other
        engine added since: anchors and references from now on) and what they bring, then
        inferred again, once per growth of the anchor pool (``Registration.retry``); the
        keyframes added."""
        assert self.reg is not None  # only after ``first`` added keyframes
        t = time.perf_counter()
        am = self.am
        am.save()  # their image and depth on disk: anchors
        self.reg.add_anchors([register_ma.Stored(k.id, img, k.K, k.T, ANCHOR_ORDER + i,
                                                 functools.partial(am.depth_of, k.id),
                                                 k.depth_kind == "sensor")
                              for i, k in enumerate(extra)
                              for img in [am.image_path(k.id)] if img is not None])
        got = set(self.reg.recheck())
        got |= set(self.reg.retry())
        self.reg_s += time.perf_counter() - t
        return self._place([fr for fr in self.frames if fr.work.index in got])

    def _place(self, frames: list[_Frame]) -> list[Keyframe]:
        """``frames``' placements (``Registration.placements``) into ``got``: the distance
        limit, one standpoint (map.md §1), each registered one a keyframe (the pose duplicates
        are the job's last pass: ``AnchoredMap.end_job``)."""
        assert self.reg is not None
        t = time.perf_counter()
        am, feats, res = self.am, self.feats, self.reg.placements
        # frames of one standpoint: a pure rotation about its centre (map.md §1). MapAnything
        # matches its views internally (no features of ours reach it): the static image content
        # is left out of the features the service matches, those of standpoints and duplicates
        turned = standpoint.standpoints(
            am, [standpoint.NewFrame(fr.work.name, fr.cam.canonical(), fr.work.index,
                                     _has_depth(fr), p.T, feats[fr.work.index],
                                     _depth_image(fr)) for fr in frames
                 for p in [res[fr.work.index]] if p.status == "registered" and p.T is not None],
            self.pairs)
        self.reg_s += time.perf_counter() - t
        t = time.perf_counter()
        added = []
        for fr in frames:
            p = res[fr.work.index]
            scene = fr.scene_m if fr.scene_m is not None else p.scene_m
            # MapAnything gives no image feature inliers: depth agreement never counts toward
            # the 300-inlier exemption (api.md §4, depth never overrides too_close)
            if too_close(scene, 0):
                self.reg.waiting.pop(fr.work.index, None)
                self.got[fr.work.index] = Placement(fr.work.name, "rejected", "too_close",
                                                    inliers=p.inliers, scene_m=scene)
                continue
            if p.status == "rejected":
                self.got[fr.work.index] = Placement(fr.work.name, "rejected", p.reason)
                continue
            assert p.T is not None and p.depth is not None and p.K_depth is not None
            T, sp, sr = turned.get(fr.work.index) or (p.T, *sigmas.calibrated(
                p.sigma_pos_m or 0.0, p.sigma_rot_deg or 0.0, _has_depth(fr), None))
            # a call of a single anchor: never tighter than its floor (map.md §2 step 8)
            sr = max(sr, p.sigma_rot_floor)
            # its sigmas calibrated as every image pose's (``mapping.sigmas``; no prior); its
            # displacement (map.md §1): the committed pose against MapAnything's image-only
            # one, 0 unless a standpoint turned it
            pl = Placement(fr.work.name, "registered", None, T, round(sp, 4), round(sr, 3),
                           p.inliers, "rotation" if fr.work.index in turned else "mapanything",
                           align_T=p.T,
                           map_residual=(0.0, round(rot_deg(T[:3, :3].T @ p.T[:3, :3]), 3)),
                           # its call's anchors (a seeded component's floor, ``_seed_floor``)
                           refs=tuple(self.kf_of.get(a, a) for a in p.anchors))
            f = _frame_in(fr, None)
            h, w = p.depth.shape
            Kg = Intrinsics(float(p.K_depth[0, 0]), float(p.K_depth[1, 1]),
                            float(p.K_depth[0, 2]), float(p.K_depth[1, 2]), w, h)
            depth = p.depth
            kind = "sensor" if _has_depth(fr) else "mono"
            if kind == "sensor":  # its depth image's residual against the map (scale_sigma_rel)
                am.depth_residual(f, T, depth, Kg)
            if kind == "mono":
                am.add_height_evidence(f, T, depth, Kg)
                k = am.scale.k
                am.scale.estimate(am.prior_span(), am.prior_span("drifting"))
                depth = (depth * (am.scale.k / k)).astype(np.float32)
            assert fr.rgb is not None
            added.append(am.add(f, feats[fr.work.index], pl, depth, Kg, kind, None))
            self.kf_of[f"new{fr.work.index}"] = added[-1].id
            self.got[fr.work.index] = pl
        self.fuse_s += time.perf_counter() - t
        return added


def _canonical(fr: _Frame) -> tuple[Any, ...]:
    """A frame's canonical order (map.md §2): its name in natural order, its camera."""
    return natural_key(fr.work.name), fr.work.camera_id, fr.work.index


def _late(up: Updater, am: AnchoredMap, geo: list[_Frame], ins: list[FrameIn],
          late: list[_Frame], feats: dict[int, Features], placements: dict[int, Placement],
          stored: list[Keyframe], workdir: Path, progress: Callable[[str], None],
          pairs: standpoint.Pairs) -> tuple[float, float]:
    """The frames without a prior by MapAnything after the others (``_MaRun``), then one match
    graph over both engines (map.md §2: the same fixed point whatever the request order): the
    job's frames with a prior still kept for want of an image pose are placed again against the
    frames MapAnything registered (``AnchoredMap.retry_kept``), and the frames MapAnything left
    unplaced are tested again against those (never inferred again), until neither adds a frame.
    The stored kept keyframes the new frames overlap are re-registered (map.md §2). Returns the
    registration and fusion seconds."""
    run = _MaRun(up, am, late, workdir, pairs, feats)
    added = run.first()
    new = list(added)
    while added:
        t = time.perf_counter()
        kept = [(f, k) for fr, f in zip(geo, ins, strict=True)
                for pl in [placements[fr.work.index]]
                if pl.status == "kept_at_prior" and pl.keyframe is not None
                for k in [am.keyframe(pl.keyframe)] if k is not None]
        again = am.retry_kept(kept, added)
        placements.update({fr.work.index: again[fr.work.name] for fr in geo
                           if fr.work.name in again})
        confirmed = [k for got in again.values() if got.status == "registered"
                     for k in [am.keyframe(got.keyframe or "")] if k is not None]
        new += confirmed
        run.reg_s += time.perf_counter() - t
        added = run.more(confirmed) if confirmed else []
        new += added
    for fr in late:
        pl = placements[fr.work.index] = run.got[fr.work.index]
        progress(f"{fr.work.name}: {pl.status} ({pl.reason or pl.method})")
    am.reregistered += am.reregister([k for k in stored if not k.usable()], new)
    return run.reg_s, run.fuse_s


def frame_class(fr: _Frame) -> str:
    """A frame's fusion class (``timings.fusion_class``): its prior's trust, its depth kind."""
    return fusion_class(_trusted(fr), _has_depth(fr))


def _fusion_by_class(up: Updater, geo: list[_Frame], late: list[_Frame],
                     placements: dict[int, Placement], fused: dict[str, float], shared: float
                     ) -> None:
    """The fusion seconds per kept frame by fusion class (``Timings``): each engine's fusion
    over its kept frames, plus the objects and the save (``shared``) over every kept frame."""
    kept = {g: [fr for fr in frs if placements[fr.work.index].status != "rejected"]
            for g, frs in (("geo", geo), ("late", late))}
    total = sum(len(v) for v in kept.values())
    per: dict[str, list[float]] = {}
    for g, frs in kept.items():
        for fr in frs:
            per.setdefault(frame_class(fr), []).append(fused[g] / len(frs) + shared / total)
    for cls, v in per.items():
        up.timings.add(f"fusion_per_frame_{cls}", float(np.mean(v)), len(v))


def classes(w: UpdateWork) -> dict[str, int]:
    """The request's frames by fusion class (the job's ETA, ``Timings.predict_phases``)."""
    out: dict[str, int] = {}
    for f in w.frames:
        c = fusion_class(f.trust in ("metric", "drifting"), f.depth is not None)
        out[c] = out.get(c, 0) + 1
    return out


def detection_grid(K: Intrinsics) -> tuple[int, int]:
    """The grid (width, height) a frame's detections are asked on: its monocular depth grid."""
    return size_at_max_side(K.width, K.height, MONO_GRID_SIDE)


def _waiter(fut: Future[Mono], ignore: NDArray[np.bool_] | None = None
            ) -> Callable[[], tuple[Any, Any, Any, Any]]:
    """A frame's monocular depth when it is ready, none on its ignored pixels (map.md §2: no
    depth point, scale evidence or scene distance from them)."""
    def get() -> tuple[Any, Any, Any, Any]:
        d, K, desc, up = fut.result().as_tuple()
        m = on_grid(ignore, d.shape)
        return (d if m is None else np.where(m, 0.0, d).astype(np.float32)), K, desc, up

    return get


def _frame_in(fr: _Frame, mono: Future[Mono] | None) -> FrameIn:
    f = fr.work
    assert fr.rgb is not None and fr.measures is not None
    prior = f.prior if f.trust in ("metric", "drifting") else None
    height, sigma = (f.height_m, f.height_sigma_m) if f.height_m is not None else (
        fr.cam.height_above_floor_m, fr.cam.height_sigma_m)
    return FrameIn(f.name, f.camera_id, fr.rgb, fr.cam.canonical(), prior, fr.depth_m, fr.ignore,
                   height, sigma, fr.measures.features, None if mono is None else _waiter(mono, fr.ignore),
                   check_close=fr.deferred, unconfirmed=fr.unconfirmed)


def _depth_kind(am: AnchoredMap | None, pl: Placement) -> str:
    """The depth kind (``sensor``, ``mono``, ``none``) of a placed frame's keyframe."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return "none" if k is None else k.depth_kind


def _contributes(am: AnchoredMap | None, pl: Placement) -> bool:
    """Whether a placed frame's keyframe carries content (``Keyframe.contributes``)."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return k is not None and k.contributes()


def _height_off(am: AnchoredMap | None, pl: Placement) -> bool:
    """Whether the height gate left a placed frame's keyframe out of content (map.md §2)."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return k is not None and not k.height_ok


def _height_sigma(am: AnchoredMap | None, pl: Placement) -> float | None:
    """The camera height sigma a placed frame's keyframe carries (its own, else its
    camera's), or None."""
    k = None if am is None or pl.keyframe is None else am.keyframe(pl.keyframe)
    return None if k is None else k.height_sigma_m


def _has_depth(fr: _Frame) -> bool:
    return fr.depth_m is not None and bool(np.count_nonzero(fr.depth_m > 0))


def _depth_image(fr: _Frame) -> tuple[NDArray[np.float32], Intrinsics] | None:
    """A frame's depth image for its two-view fits (``standpoint.NewFrame``): on its canonical
    pixels, its ignored pixels holding none (``anchored.depth_image``); None without one."""
    if not _has_depth(fr):
        return None
    assert fr.depth_m is not None
    K = fr.cam.canonical()
    m = on_grid(fr.ignore, fr.depth_m.shape)
    d = fr.depth_m.astype(np.float32) if m is None else np.where(m, 0.0, fr.depth_m).astype(
        np.float32)
    return d, K


def _input(fr: _Frame, pl: Placement | None) -> dict[str, Any]:
    """The ``InputStatus`` of a frame: rejected by the checks (``pl`` None) or as placed."""
    if pl is None:  # rejected by the checks
        return input_row(fr.work.name, "rejected", fr.reason, 0, None, None, None, None,
                         fr.scene_m, fr.duplicate_of)
    res = pl.prior_residual if fr.work.trust in ("metric", "drifting") else None
    return input_row(fr.work.name, pl.status, pl.reason, pl.inliers, pl.T, pl.sigma_pos_m,
                     pl.sigma_rot_deg, res, pl.scene_m, pl.duplicate_of, pl.detail)


def input_row(name: str, status: str, reason: str | None, inliers: int, T: Any,
              sigma_pos_m: float | None, sigma_rot_deg: float | None,
              residual: tuple[float, float] | None, scene_m: float | None,
              duplicate_of: str | None = None, detail: str | None = None) -> dict[str, Any]:
    """An ``InputStatus`` (api.md §4): pose and sigmas null when rejected, the prior residuals
    null without a prior, with reason ``too_close`` the distance limit and the frame's median
    scene depth, with reason ``duplicate`` the frame it duplicates, and ``detail`` (why the
    images did not confirm a kept frame) only on a ``kept_at_prior`` frame not kept for a
    failed check."""
    kept = status != "rejected"
    row = {"name": name, "status": status, "reason": reason, "inliers": inliers,
           "T_map_cam": np.asarray(T).round(6).tolist() if kept and T is not None else None,
           "sigma_pos_m": sigma_pos_m if kept else None,
           "sigma_rot_deg": sigma_rot_deg if kept else None,
           "prior_residual_m": None if residual is None else round(residual[0], 4),
           "prior_residual_deg": None if residual is None else round(residual[1], 3),
           "detail": detail if status == "kept_at_prior" else None}
    if reason == "too_close":
        row["min_distance_m"] = TOO_CLOSE_M
        row["scene_distance_m"] = round(scene_m if scene_m is not None else 0.0, 3)
    if reason == "duplicate":
        row["duplicate_of"] = duplicate_of or name
    return row


def prior_frames_rejected(frames: list[_Frame], status: dict[int, str]) -> bool:
    """Whether every prior-carrying frame of a map's first update is rejected (map.md §1):
    ``status`` holds each frame's status by index (absent: rejected by the checks)."""
    trusted = [fr for fr in frames if fr.work.trust in ("metric", "drifting")]
    return bool(trusted) and all(status.get(fr.work.index, "rejected") == "rejected"
                                 for fr in trusted)


def run_anchored(up: Updater, job: Job, w: UpdateWork, frames: list[_Frame],
                 prev: dict[str, Any] | None, workdir: Path,
                 changed: Callable[[], None] = lambda: None) -> None:
    tm = job.timings_s
    todo = [fr for fr in frames if fr.reason is None and fr.path is not None]
    placements: dict[int, Placement] = {}
    am: AnchoredMap | None = None
    stored: list[Keyframe] = []
    k_prev: float | None = None
    mapper = workdir / "anchored"
    om: ObjectMap | None = None
    n_objects = 0
    pairs = standpoint.Pairs()  # the two-view fits of this update (standpoints, the hard limit)
    rotations_bad = False
    rotation_pairs: list[dict[str, Any]] = []  # the rotation check's pairs (map.md §4)
    # the map's label vocabulary with this update's labels (map.md §5): every detection of the
    # job prompts with it and the defaults
    from oh_my_slam.service.update import map_labels

    labels = map_labels(prev, w.labels)
    if todo:
        t = time.perf_counter()
        if prev is not None and up.store.mapper_dir(w.map).exists():
            clone_tree(up.store.mapper_dir(w.map), mapper)
            am = AnchoredMap.load(mapper)
            k_prev = am.scale.k
            stored = list(am.keyframes)
        else:
            mapper.mkdir(parents=True, exist_ok=True)
            # a new mapper map numbers its objects and surfaces past every id the map gave
            # (a revert to a version without a stored mapper map: map.md §2 Revert)
            up.store.seed_ids(w.map, mapper, anchored=True)
            am = AnchoredMap(mapper)
        am.begin_job()
        tm["clone"] = round(time.perf_counter() - t, 3)
        pool = ThreadPoolExecutor(INFERENCE_IN_FLIGHT)
        # a new map's origin frame (its first with a prior, else its first registered) also
        # gives gravity: the angle to its prior's up axis, or the camera-only map's up
        origin = None
        if prev is None:  # in canonical name order (map.md §1, §2)
            origin = min([fr for fr in todo if _trusted(fr)] or todo, key=_canonical)
        camera_only_origin = origin is not None and not _trusted(origin)
        # the engine order (map.md §2 step 8): frames with depth images go to the geometric
        # engine first, as a dry run registers them; monocular frames without a prior go to
        # MapAnything after the others (a new camera-only map's origin is the prior-first
        # engine's), and so do the frames without a prior the geometric engine leaves unplaced
        ma = none_engine() == "mapanything"
        late = [fr for fr in todo if not _trusted(fr) and fr is not origin
                and not _has_depth(fr)] if ma else []
        geo = [fr for fr in todo if all(fr is not x for x in late)]
        # MapAnything may take some of them: the job's static content is measured over all
        spill = ma and any(not _trusted(fr) for fr in geo)
        ins = []
        for fr in geo:
            fut = None
            if not _has_depth(fr) or fr is origin:
                assert fr.path is not None
                fut = pool.submit(up.backend.infer, fr.path, fr.cam.canonical(), fr is origin)
            ins.append(_frame_in(fr, fut))
        # the job's static image content, over every frame whichever engine registers it
        late_ins = [_frame_in(fr, None) for fr in late]
        shared = bool(late) or spill  # else the engine measures it (``AnchoredMap.update``)
        late_feats = dict(zip([fr.work.index for fr in late], job_features(ins + late_ins)[len(ins):],
                              strict=True)) if shared else {}
        # the detections queue behind the monocular depth registration waits for
        dets = {fr.work.index: pool.submit(up.backend.detect, fr.path,
                                           detection_grid(fr.cam.canonical()), labels)
                for fr in todo if fr.path is not None}
        done = [0]

        def progress(msg: str) -> None:
            done[0] += 1
            job.progress = round(0.1 + 0.8 * min(1.0, done[0] / len(todo)), 3)
            changed()

        # a frame without a prior registered by the prior-first engine joins a standpoint
        am.standpoint = functools.partial(standpoint.join_standpoint, am, pairs)
        try:
            # the job as it starts: a second registration starts again from it (map.md §2
            # match graph: a pose duplicate that served the job, a cross-engine drift fit)
            start = am.job_start()
            geo0, ins0, late0, late_feats0 = list(geo), list(ins), list(late), dict(late_feats)
            reg_s = fuse_s = late_r = 0.0
            spilled: list[_Frame] = []
            while True:
                # the job stays open until both engines registered (``AnchoredMap.end_job``:
                # the pose-duplicate pass over all of them, then the fusion)
                pls = am.update(ins, progress, static=not shared, hold=True) if ins else []
                reg_s += am.timings.get("register", 0.0)
                fuse_s += am.timings.get("fuse", 0.0)
                # a new camera-only map whose origin was rejected: the next frame in canonical
                # order is offered as the origin (map.md §1: the first *registered* frame)
                while camera_only_origin and late and not am.confirmed():
                    origin = min(late, key=_canonical)
                    late.remove(origin)
                    geo.append(origin)
                    assert origin.path is not None
                    fin = _frame_in(origin, pool.submit(up.backend.infer, origin.path,
                                                        origin.cam.canonical(), True))
                    fin.feats = late_feats.pop(origin.work.index)
                    ins.append(fin)
                    pls += am.update([fin], progress, static=False, hold=True)
                    reg_s += am.timings.get("register", 0.0)
                    fuse_s += am.timings.get("fuse", 0.0)
                for fr, pl in zip(geo, pls, strict=True):
                    placements[fr.work.index] = pl
                # the frames without a prior it left unplaced: MapAnything's too (map.md §2)
                spilled = []
                for fr, f in zip(geo, ins, strict=True):
                    pl = placements[fr.work.index]
                    if ma and not _trusted(fr) and pl.status == "rejected" \
                            and pl.reason == "no_overlap" and f.feats is not None:
                        spilled.append(fr)
                        late_feats[fr.work.index] = f.feats
                late = [*late, *spilled]
                geo_s = reg_s - late_r
                fused = {"geo": fuse_s, "late": 0.0}  # each engine's fusion seconds
                if late:  # their scene distance is MapAnything's depth (``register_ma``)
                    r, fu = _late(up, am, geo, ins, late, late_feats, placements, stored,
                                  workdir, progress, pairs)
                    late_r += r
                    reg_s, fuse_s = reg_s + r, fuse_s + fu
                    fused["late"] = fu
                job_pls = [placements[fr.work.index] for fr in todo
                           if fr.work.index in placements]
                if not am.plan_drift_rerun(ins, [placements[fr.work.index] for fr in geo]) \
                        and not am.plan_rerun(job_pls):
                    break
                # once more from the job's start, without what the plan leaves out
                am.job_restart(start)
                placements.clear()
                geo, ins, late = list(geo0), list(ins0), []
                late_feats = dict(late_feats0)
                for fr in late0:
                    if fr.work.name in am.excluded():
                        placements[fr.work.index] = am.excluded()[fr.work.name]
                    else:
                        late.append(fr)
            if geo:
                up.timings.add("registration_per_frame", geo_s / len(geo), len(geo))
            if late:
                up.timings.add("registration_mapanything_per_frame", late_r / len(late),
                               len(late))
            # the drift takes the frames the cross-engine retries confirmed too
            am.close_job(ins, [placements[fr.work.index] for fr in geo])
            dups, end_s = am.end_job([placements[fr.work.index] for fr in todo
                                      if fr.work.index in placements], prev is None)
            for fr in todo:
                if fr.work.name in dups:
                    placements[fr.work.index] = dups[fr.work.name]
            fuse_s += end_s
            fused["geo"] += end_s
            if not any(_trusted(fr) for fr in frames):
                # an update without priors: committed relative rotations against the images'
                # (map.md §4 rotation_inconsistent; registered frames only)
                t = time.perf_counter()
                rotations_bad, rotation_pairs = standpoint.rotation_check(
                    am, list(placements.values()), pairs)
                reg_s += time.perf_counter() - t
            t = time.perf_counter()
            version = (prev["version"] + 1) if prev else 1
            again = []  # the re-registered keyframes' objects: their stored image's detections
            for e in am.reregistered:  # confirmed from their stored image: it exists
                if e.get("objects"):  # the too-close exception: its detections are in already
                    continue
                kf, path = am.keyframe(e["keyframe"]), am.image_path(e["keyframe"])
                assert kf is not None and path is not None
                again.append((kf, pool.submit(up.backend.detect, path, detection_grid(kf.K),
                                              labels)))
            if am.keyframes:  # else no origin: every frame with a prior was rejected
                om, n_objects = fuse_objects(am, [(placements[fr.work.index], fr.ignore,
                                                   fr.path, dets[fr.work.index]) for fr in todo],
                                             version, functools.partial(
                                                 up.backend.look_closer, labels=labels), again)
            tm["objects"] = round(time.perf_counter() - t, 3)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        t = time.perf_counter()
        am.save()
        if om is not None:
            om.save()
        tm["register"] = round(reg_s, 3)
        tm["fusion"] = round(fuse_s + tm["objects"] + time.perf_counter() - t, 3)
        n_kept = sum(pl.status != "rejected" for pl in placements.values())
        if n_kept:
            up.timings.add("fusion_per_frame", tm["fusion"] / n_kept, n_kept)
            _fusion_by_class(up, [fr for fr in geo if all(fr is not x for x in spilled)],
                             late, placements, fused, tm["fusion"] - fuse_s)
    job.progress = 0.95
    changed()
    t = time.perf_counter()
    job.inputs = [_input(fr, placements.get(fr.work.index)) for fr in frames]
    kept = [(fr, placements[fr.work.index]) for fr in todo
            if placements[fr.work.index].status != "rejected"]
    version = (prev["version"] + 1) if prev else 1
    # the stored frames at the poses this version would commit (their keyframes'), and how far
    # each moved from the previous version: the gauge shift and ``previous_residuals``
    by_id = {k.id: k for k in am.keyframes} if am is not None else {}
    again_ids = {e["keyframe"] for e in am.reregistered} if am is not None else set()
    rows, shifts = [], []
    for r in prev["frames"] if prev else []:
        k = by_id.get(r.get("internal") or "")
        if k is not None and k.id in again_ids:
            # confirmed by this job's images (map.md §2): registered now; the movement of its
            # confirmation is in ``reregistered`` only, not in the gauge shift or the residuals
            rows.append({**r, "T_map_cam": k.T.tolist(), "sigma_pos_m": k.sigma_pos_m,
                         "sigma_rot_deg": k.sigma_rot_deg, "pose_state": "fused",
                         "inliers": k.inliers, "method": k.method})
            continue
        row = r if k is None else {**r, "T_map_cam": k.T.tolist()}
        shifts.append(pose_residual(np.asarray(row["T_map_cam"]), np.asarray(r["T_map_cam"])))
        rows.append(row)
    for fr, pl in kept:
        assert pl.T is not None
        p = fr.work.prior
        rows.append({"name": fr.work.name, "camera_id": fr.work.camera_id,
                     "internal": pl.keyframe, "T_map_cam": pl.T.tolist(),
                     "sigma_pos_m": pl.sigma_pos_m, "sigma_rot_deg": pl.sigma_rot_deg,
                     "pose_state": "fused" if pl.status == "registered" else "kept_at_prior",
                     "added_version": version, "sha256": fr.sha, "inliers": pl.inliers,
                     "method": pl.method, "depth_kind": _depth_kind(am, pl),
                     "prior": p if p is not None and p["trust"] == "metric" else None,
                     # the too-close exception (api.md §4): a kept frame whose in-range depth
                     # feeds occupancy and heights (``derived``)
                     "content": _contributes(am, pl),
                     # the height gate (map.md §2): a reference, but no content
                     "height_inconsistent": _height_off(am, pl),
                     # its camera height sigma (locate's height_sigma, api.md §6)
                     "height_sigma_m": _height_sigma(am, pl)})
    prior_res = [pose_residual(np.asarray(r["T_map_cam"]), np.asarray(r["prior"]["T_map_cam"]))
                 for r in rows if r.get("prior")]
    source, sigma = am.scale_summary() if am is not None else ("monocular", 0.1)
    ratio = None
    if prev is not None and am is not None:
        # the monocular keyframes' depth follows k; positions and depth images do not move
        mono = any(k.depth_kind == "mono" for k in am.keyframes)
        ratio = round(am.scale.k / k_prev, 5) if mono and k_prev else 1.0
    # registered inputs whose pose sigma exceeds the threshold (map.md §4)
    registered = [pl for _, pl in kept if pl.status == "registered"]
    low = sum((pl.sigma_pos_m or 0.0) > LOW_CONFIDENCE_SIGMA_M for pl in registered)
    gauge = None
    if prev is not None:
        gauge = {"translation_m": round(float(np.median([m for m, _ in shifts])), 4)
                 if shifts else 0.0,
                 "rotation_deg": round(float(np.median([d for _, d in shifts])), 3)
                 if shifts else 0.0, "scale_ratio": 1.0}
    # the new frames against their image-only poses (map.md §1, ``max_displacement_*``)
    moved = displacements([pl for _, pl in kept])
    q: dict[str, Any] = {
        "scale_ratio_vs_prev": ratio,
        "gauge_shift": gauge,
        **displacement_quality(moved),
        "prior_disagreement_deg": 0.0 if am is None else am.prior_disagreement_deg,
        "reregistered": [] if am is None else reregistered(am.reregistered),
        "height_inconsistent": [] if am is None else list(am.height_inconsistent),
        "rotation_pairs": rotation_pairs,
        "prior_residuals": residual_stats(prior_res),
        "previous_residuals": residual_stats(shifts) if prev else None,
        "rejected_share": round(1.0 - len(kept) / len(frames), 4),
        "low_confidence_share": round(low / len(registered), 4) if registered else 0.0,
        "rebuilt": False,
        "prior_gravity_deg": None if prev is not None or am is None else am.up_check_deg,
    }
    status = {i: pl.status for i, pl in placements.items()}
    job.quality = gate(q, bool(kept), w.accept_if,
                       prev is None and prior_frames_rejected(frames, status), rotations_bad)
    if am is not None:
        job.scale_source, job.scale_sigma_rel = source, round(sigma, 5)
    if not job.quality["accepted"]:
        job.state, job.progress = "rejected", 1.0
        return
    assert am is not None and om is not None
    floor, floor_sigma, pts = om.floor_z, om.floor_sigma, am.bounds()
    record = {"kind": "update", "engine": "anchored", "job_id": job.id,
              "sim3": Sim3().to_dict(), "scale_source": source,
              "scale_sigma_rel": round(sigma, 5), "frames": rows, "objects": n_objects,
              "floor_z_m": None if floor is None else round(floor, 4),
              "floor_sigma_m": round(floor_sigma, 4), "pose_states": POSE_STATES,
              "bounds_m": {"min": pts.min(0).round(4).tolist(),
                           "max": pts.max(0).round(4).tolist()},
              "points": len(am.cloud.keys), "labels": labels}
    job.result_version = up.store.commit(w.map, record, mapper,
                                         {"objects.json": om.records()}, w.cameras, w.masks)
    job.state, job.progress = "committed", 1.0
    tm["commit"] = round(time.perf_counter() - t, 3)
