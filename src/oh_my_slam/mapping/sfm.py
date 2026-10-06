"""Structure from motion with COLMAP 4.2.

SIFT features and matching run through the Homebrew ``colmap`` CLI; mapping, triangulation and
bundle adjustment run through pycolmap on the same database. Both must be 4.2.x.

New map: global mapping (GLOMAP) → incremental if < 60 % placed → multi-view (MapAnything)
poses refined with the verified matches and monocular depth (``panorama``) + triangulation.
Rotation-dominant input goes straight to the multi-view path. SfM poses without triangulated
support are not accepted (``vet``), nor are poses that contradict their own verified matches
(``contradicted``); the mapper then joins what the main reconstruction lacks or got wrong in
scale or tilt (``mapping.trajectory``).
Update: photos of the device that took the map's share its camera (``Sfm.existing_camera``);
incremental mapping continues the stored model with its frames and cameras fixed (``_back_onto``
maps it back); keyframes it cannot place (all of them for rotation-dominant input) get anchored,
refined multi-view poses.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.errors import OhMySlamError
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics, Pose

log = get_logger("oh_my_slam.sfm")

# Gate G4 (measured on the user's inputs): SIFT registers as many frames as ALIKED+LightGlue and is
# ~30x faster on this CPU/CoreML build, so SIFT is the only feature type. LightGlue matches SIFT's
# own keypoints again where a video walk hangs together by next to nothing (``weak_link_pairs``).
MIN_PLACED_FRACTION = 0.6
ROTATION_PAIR_FRACTION = 0.5
ROTATION_BASELINE_RATIO = 0.02
MAX_FEATURES = 4096
SIFT_UNDOUBLED_VIDEO_SIDE = 1600  # Sfm.extract

# COLMAP TwoViewGeometry configuration codes
_PLANAR, _PANORAMIC, _PLANAR_OR_PANORAMIC = 4, 5, 6


class SfmError(OhMySlamError):
    pass


def colmap_bin() -> str:
    path = shutil.which("colmap")
    if path is None:
        raise SfmError("COLMAP not found — install it with: brew install colmap (4.2.x, as the "
                       "pycolmap in .venv)")
    return path


def check_versions() -> str:
    import pycolmap

    out = subprocess.run([colmap_bin(), "version"], capture_output=True, text=True).stdout
    cli = out.split()[1] if out.startswith("COLMAP") else "?"
    if not cli.startswith("4.2") or not pycolmap.__version__.startswith("4.2"):
        raise SfmError(f"COLMAP CLI {cli} and pycolmap {pycolmap.__version__} must both be 4.2.x "
                       "— run: brew upgrade colmap")
    return cli


def _run(args: list[str], log_path: Path) -> None:
    with log_path.open("a") as f:
        res = subprocess.run([colmap_bin(), *args], stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        lines = log_path.read_text(errors="replace").splitlines()
        fatal = [ln for ln in lines if ln.startswith(("F2", "E2")) or "Check failed" in ln]
        tail = (fatal or lines)[-6:]
        raise SfmError(f"colmap {args[0]} failed ({res.returncode}): " + " | ".join(tail))


FOCAL_MATCH_REL = 0.01  # EXIF focal priors this close are the same camera (device and zoom)


@dataclass
class CameraPrior:
    width: int
    height: int
    focal: float | None = None  # full-resolution pixels
    existing_id: int | None = None  # reuse this database camera (same image size)
    # else the first of these database cameras of the same image size whose focal prior is
    # ``focal`` (within ``FOCAL_MATCH_REL``): photos of the same device, with EXIF, in a later update
    same_focal_ids: tuple[int, ...] = ()


def _camera_params(prior: CameraPrior) -> str:
    """``--ImageReader.camera_params`` of a new SIMPLE_PINHOLE camera for ``prior``."""
    return f"{prior.focal:.4f},{prior.width / 2:.4f},{prior.height / 2:.4f}"


@dataclass
class SfmModel:
    rec: Any  # pycolmap.Reconstruction
    method: str
    notes: dict[str, Any] = field(default_factory=dict)
    # the mapper's other reconstructions (its own frames each; they may share keyframes)
    others: list[Any] = field(default_factory=list)

    @property
    def registered(self) -> list[str]:
        return sorted(im.name for im in self.rec.images.values() if im.has_pose)

    def supported(self, min_points: int) -> set[str]:
        """Registered keyframes with at least ``min_points`` triangulated observations."""
        return {im.name for im in self.rec.images.values()
                if im.has_pose and im.num_points3D >= min_points}

    def point_counts(self) -> dict[str, int]:
        """Triangulated observations of each registered keyframe."""
        return {im.name: int(im.num_points3D) for im in self.rec.images.values() if im.has_pose}

    def covisibility(self) -> dict[frozenset[str], int]:
        """Triangulated points shared by each pair of registered keyframes."""
        name = {im.image_id: im.name for im in self.rec.images.values() if im.has_pose}
        out: Counter[frozenset[str]] = Counter()
        for p in self.rec.points3D.values():
            ids = sorted({el.image_id for el in p.track.elements if el.image_id in name})
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    out[frozenset((name[a], name[b]))] += 1
        return dict(out)

    def deregister(self, names: set[str]) -> None:
        for n in sorted(names):
            if self.rec.find_image_with_name(n) is not None:
                self.rec.deregister_frame(self.rec.find_image_with_name(n).frame_id)

    def pose(self, name: str) -> Pose:
        """Camera-to-world (model frame)."""
        im = self.rec.find_image_with_name(name)
        T = np.eye(4)
        T[:3, :] = np.asarray(im.cam_from_world().matrix())
        return Pose.from_matrix(np.linalg.inv(T))

    def intrinsics(self, name: str) -> Intrinsics:
        im = self.rec.find_image_with_name(name)
        cam = self.rec.cameras[im.camera_id]
        K = np.asarray(cam.calibration_matrix())
        return Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], cam.width, cam.height, "colmap")

    def camera_id(self, name: str) -> int:
        return int(self.rec.find_image_with_name(name).camera_id)

    def observations(self, name: str, max_error: float = 2.0, min_track: int = 3
                     ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(uv (M, 2) in full-res pixels, world xyz (M, 3)) of well-triangulated points."""
        im = self.rec.find_image_with_name(name)
        uv, xyz = [], []
        for p2 in im.points2D:
            if not p2.has_point3D():
                continue
            p3 = self.rec.points3D[p2.point3D_id]
            if p3.error > max_error or p3.track.length() < min_track:
                continue
            uv.append(p2.xy)
            xyz.append(p3.xyz)
        return np.asarray(uv, np.float64).reshape(-1, 2), np.asarray(xyz, np.float64).reshape(-1, 3)

    def image_stats(self, name: str) -> dict[str, float]:
        im = self.rec.find_image_with_name(name)
        errs = [self.rec.points3D[p.point3D_id].error for p in im.points2D if p.has_point3D()]
        return {"observations": float(len(errs)),
                "reproj_error": float(np.mean(errs)) if errs else 0.0}

    def points(self) -> NDArray[np.float64]:
        return np.asarray([p.xyz for p in self.rec.points3D.values()], np.float64).reshape(-1, 3)

    def baseline_ratio(self) -> float:
        """Median distance between capture-order neighbours / median point depth."""
        names = self.registered
        if len(names) < 3:
            return 0.0
        C = np.array([self.pose(n).t for n in names])
        base = np.median(np.linalg.norm(np.diff(C, axis=0), axis=1))
        depths = []
        for n in names[:: max(1, len(names) // 20)]:
            _, xyz = self.observations(n, max_error=4.0, min_track=2)
            if len(xyz):
                T = self.pose(n).inverse()
                depths.append(np.median(T.apply(xyz)[:, 2]))
        d = float(np.median(depths)) if depths else 0.0
        return float(base / d) if d > 0 else 0.0

    def transform(self, s: float, R: NDArray[Any], t: NDArray[Any]) -> None:
        import pycolmap

        self.rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(np.asarray(R)),
                                          np.asarray(t, np.float64)))

    def write(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.rec.write(str(path))


# A cut in a video's capture order that fewer than WEAK_CUT_PAIRS verified sequential pairs span
# (at most one: what lies beyond hangs on one link) is a weak link. On the office walk the white
# wardrobe doors and the door (f000101-f000109, 67-1300 SIFT keypoints each) had no verified pair
# to the rest at one end and 0-1 at the other; the global mapper left these nine keyframes a
# reconstruction of their own in 4 of 5 builds, and the fifth joined them on one 15-inlier pair,
# 5 % off in depth against their neighbours. The listed pairs that span a weak cut, at most
# WEAK_CUT_REACH keyframes apart, are matched again with LightGlue on the same SIFT keypoints:
# there 15-16 of 41-42 verified with 15-67 inliers (SIFT: 0-22 raw matches), and every build mapped
# the walk as one reconstruction, the nine keyframes 1.6-1.8 % off in depth against their
# neighbours. LightGlue costs about 1 s per pair and CPU thread (COLMAP's CoreML provider cannot
# compile its dynamic shapes and was 6x slower), so it is kept to these pairs: a normal cut is
# spanned by ~30 verified pairs (5th percentile 6-8 on officev, lv and livingroom), and the office
# walk's next weakest cut (2 pairs) cost 4 s more and moved no pose.
WEAK_CUT_PAIRS = 2
WEAK_CUT_REACH = 6
MIN_INLIERS = 15  # COLMAP's TwoViewGeometry.min_num_inliers: a verified pair


def weak_link_pairs(order: list[str], listed: set[frozenset[str]],
                    verified: set[frozenset[str]], window: int,
                    min_pairs: int = WEAK_CUT_PAIRS, reach: int = WEAK_CUT_REACH
                    ) -> tuple[list[str], set[frozenset[str]]]:
    """The weak cuts of a video's capture ``order`` (named by the keyframe before them) and the
    ``listed`` pairs not ``verified`` that span one at most ``reach`` keyframes apart. A cut is weak
    when fewer than ``min_pairs`` verified pairs at most ``window`` apart (the sequential pairs)
    span it; loop closures do not count: they hold the walk together, not the stretch."""
    pos = {n: i for i, n in enumerate(order)}
    span = np.zeros(max(len(order) - 1, 0), int)
    for p in verified:
        if p <= pos.keys():
            a, b = sorted(pos[n] for n in p)
            if b - a <= window:
                span[a:b] += 1
    weak = np.flatnonzero(span < min_pairs)
    out = set()
    for p in listed - verified:
        if len(p) == 2 and p <= pos.keys():
            a, b = sorted(pos[n] for n in p)
            k = int(np.searchsorted(weak, a))  # the first weak cut at or after a
            if b - a <= reach and k < len(weak) and weak[k] < b:
                out.add(p)
    return [order[c] for c in weak], out


FIXED_ROT_TOL_DEG = 2.0
FIXED_POS_TOL = 0.05  # of the fixed frames' spread
EXTEND_SEED = 0  # random seed of incremental extensions and the global mapper (COLMAP: -1, by time)
# threads of the SfM pipelines (-1: all cores; seeded, they still differ between runs only by
# the order threads finish in, which tests pin with 1)
SFM_THREADS = -1


def vet(model: SfmModel, rotation_pairs: set[frozenset[str]]) -> dict[str, list[str]]:
    """Deregister the keyframes whose SfM pose nothing supports: fewer than
    ``trajectory.SUPPORT_MIN_POINTS`` triangulated observations (a part of the view graph that the
    global mapper shrank onto one centre, or a keyframe it registered with next to no points). The
    unsupported keyframes that also collapsed onto another keyframe's centre (``trajectory.
    collapsed_keyframes``) are listed apart. Returns both lists (names)."""
    from oh_my_slam.mapping import trajectory as traj

    names = model.registered
    supported = model.supported(traj.SUPPORT_MIN_POINTS)
    unsupported = set(names) - supported
    if not unsupported:
        return {"unsupported": [], "collapsed": []}
    poses = {n: model.pose(n) for n in names}
    radius = traj.collapse_radius(poses, names, supported)
    collapsed = traj.collapsed_keyframes(poses, supported, radius, rotation_pairs)
    model.deregister(unsupported)
    log.warning("%s: %d keyframes registered without support (< %d points) are not accepted as "
                "posed%s", model.method, len(unsupported), traj.SUPPORT_MIN_POINTS,
                f", {len(collapsed)} of them collapsed onto one camera centre" if collapsed else "")
    return {"unsupported": sorted(unsupported), "collapsed": sorted(collapsed)}


# A keyframe's poses agree with its verified matches to a few hundredths of a degree (median
# symmetric epipolar distance; livingroom and lv at most 0.12°, the office in one update 0.08°);
# the global mapper's 6-photo office map placed one 0.43° off (0.26° with its neighbours' pairs).
MAX_EPIPOLAR_DEG = 0.25
EPIPOLAR_MIN_MATCHES = 50


def epipolar_deg(Ta: Pose, Ka: Intrinsics, Tb: Pose, Kb: Intrinsics, uv_a: NDArray[Any],
                 uv_b: NDArray[Any]) -> NDArray[np.float64]:
    """Symmetric epipolar distance of matched keypoints (full-resolution pixels) under the
    camera-to-world poses ``Ta``, ``Tb``, in degrees (pixels over the focal length). It measures
    rotation errors and translation-direction errors alike; for two views with no baseline it is
    zero for any translation direction when the rotation is right."""
    R = Tb.R.T @ Ta.R
    t = Tb.R.T @ (Ta.t - Tb.t)
    n = float(np.linalg.norm(t))
    if n <= 0:
        return np.zeros(len(uv_a))
    t = t / n
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    F = np.linalg.inv(Kb.K()).T @ tx @ R @ np.linalg.inv(Ka.K())
    xa = np.column_stack([uv_a, np.ones(len(uv_a))])
    xb = np.column_stack([uv_b, np.ones(len(uv_b))])
    la, lb = xa @ F.T, xb @ F  # epipolar lines in b and in a
    num = np.abs(np.sum(xb * la, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 0.5 * (num / np.hypot(la[:, 0], la[:, 1]) / Kb.fx
                   + num / np.hypot(lb[:, 0], lb[:, 1]) / Ka.fx)
    return np.degrees(np.nan_to_num(d, nan=0.0))


def contradicted(poses: dict[str, Pose], K: dict[str, Intrinsics], matches: list[Any],
                 max_deg: float = MAX_EPIPOLAR_DEG, min_matches: int = EPIPOLAR_MIN_MATCHES
                 ) -> dict[str, float]:
    """Keyframes whose pose contradicts their own verified matches (``panorama.PairMatches``):
    the median epipolar distance (``epipolar_deg``) over all their matches to the other posed
    keyframes is above ``max_deg``. A wrong pose also spoils its partners' pairs, so the worst
    keyframe is taken out first and the rest judged again without it. Returns name → median."""
    errs = [(p.a, p.b, epipolar_deg(poses[p.a], K[p.a], poses[p.b], K[p.b], p.uv_a, p.uv_b))
            for p in matches if p.a in poses and p.b in poses and p.a in K and p.b in K]
    out: dict[str, float] = {}
    while True:
        per: dict[str, list[NDArray[Any]]] = {}
        for a, b, e in errs:
            if a not in out and b not in out:
                per.setdefault(a, []).append(e)
                per.setdefault(b, []).append(e)
        med = {n: float(np.median(np.concatenate(v))) for n, v in per.items()
               if sum(len(x) for x in v) >= min_matches}
        bad = {n: m for n, m in med.items() if m > max_deg}
        if not bad:
            return out
        worst = max(sorted(bad), key=lambda n: bad[n])
        out[worst] = bad[worst]


def _back_onto(model: SfmModel, base: SfmModel) -> SfmModel | None:
    """COLMAP re-normalises an extended reconstruction — a similarity away from its input, the
    fixed frames included — so map it back onto the input with the similarity that takes the
    frames of both onto their input poses.

    COLMAP can also drop a fixed frame whose observations the new images' points contradict (a
    weakly supported stored pose: a few dozen points, each over the reprojection limit once the
    new tracks triangulate) and register it again, elsewhere: the other fixed frames stay exactly
    where they were, and so does everything registered on them. Such re-placed frames are left out
    of the similarity (the worst one at a time, refitting on the rest) and listed in
    ``model.notes["moved_fixed"]``; the map keeps their stored poses. None when fewer than two
    frames, or not more than half of them, stayed rigid (then the extension moved the map)."""
    from oh_my_slam.mapping.frame import similarity_by_poses

    common = sorted(set(model.registered) & set(base.registered))
    if len(common) < 2:
        return None
    ref = {n: base.pose(n) for n in common}
    out = {n: model.pose(n) for n in common}
    keep = list(common)
    centres = np.array([r.t for r in ref.values()])
    tol = max(1e-4, FIXED_POS_TOL * float(np.linalg.norm(centres - centres.mean(0), axis=1).max()))
    moved: dict[str, tuple[float, float]] = {}
    while True:
        sim = similarity_by_poses([out[n] for n in keep], [ref[n] for n in keep])
        dev = {}
        for n in keep:
            M = sim.transform_pose(out[n].matrix())
            rot = float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3].T @ ref[n].R) - 1) / 2,
                                                     -1.0, 1.0))))
            dev[n] = (rot, float(np.linalg.norm(M[:3, 3] - ref[n].t)))
        off = [n for n in keep if dev[n][0] > FIXED_ROT_TOL_DEG or dev[n][1] > tol]
        if not off:
            break
        worst = max(off, key=lambda n: max(dev[n][0] / FIXED_ROT_TOL_DEG, dev[n][1] / tol))
        moved[worst] = dev[worst]
        keep.remove(worst)
        if len(keep) < 2 or 2 * len(keep) <= len(common):
            log.warning("incremental extension moved the fixed frames (%s); its result is not "
                        "used", ", ".join(f"{n} {r:.2f}° {d:.3g}" for n, (r, d) in
                                           sorted(moved.items())))
            return None
    model.transform(sim.s, sim.R, sim.t)
    if moved:
        model.notes["moved_fixed"] = sorted(moved)
        log.warning("incremental extension re-placed %d weakly supported fixed frames (%s); they "
                    "keep their stored poses", len(moved), ", ".join(
                        f"{n} {r:.2f}° {d:.3g}" for n, (r, d) in sorted(moved.items())))
    return model


class Sfm:
    def __init__(self, db_path: Path, image_dir: Path, work_dir: Path) -> None:
        self.db = Path(db_path)
        self.image_dir = Path(image_dir)
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work / "colmap.log"

    # -- features & matching ---------------------------------------------------------------------

    def extract(self, names: list[str], prior: CameraPrior, video: bool = False) -> int:
        """Extract features for ``names`` (relative to ``image_dir``) sharing one camera.

        SIFT doubles the image for its finest octave (COLMAP's default), except for the keyframes
        of a ``video`` of at least ``SIFT_UNDOUBLED_VIDEO_SIDE`` px: they overlap densely and carry
        compression and motion blur, and their doubled octave was most of the COLMAP stage (the
        lv walk, 124 keyframes of 1920 x 1080: features and matching 36 s at 8.7 GB doubled, 10 s
        at 2.4 GB not; global mapping 13 s -> 7 s; every keyframe still posed by SfM, overlapping
        keyframes' depth agreeing as well or better). Smaller frames would keep few features
        without it. Photos keep it: downscaled to 2000-2800 px or not doubled, the 13-photo office
        map fell into a wrong global solution in 2 to 8 of 8 trials, never with COLMAP's
        features."""
        camera = self._register(names, prior)
        names = self._without_features(names)  # a rebuild keeps the stored keyframes' features
        if not names:
            return camera
        lst = self.work / "extract_list.txt"
        lst.write_text("\n".join(names) + "\n")
        doubled = not video or max(prior.width, prior.height) < SIFT_UNDOUBLED_VIDEO_SIDE
        args = [
            "feature_extractor", "--database_path", str(self.db), "--image_path",
            str(self.image_dir), "--image_list_path", str(lst),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--FeatureExtraction.use_gpu", "0", "--log_level", "1",
            "--FeatureExtraction.num_threads", str(SFM_THREADS),
            "--FeatureExtraction.type", "SIFT", "--SiftExtraction.max_num_features",
            str(MAX_FEATURES), "--SiftExtraction.first_octave", "-1" if doubled else "0",
        ]
        args += ["--ImageReader.existing_camera_id", str(camera)]
        _run(args, self.log_path)
        return camera

    def _without_features(self, names: list[str]) -> list[str]:
        """``names`` whose features the database does not hold yet."""
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            ids = {im.name: im.image_id for im in db.read_all_images()}
            return [n for n in names if n not in ids or not db.exists_keypoints(ids[n])]
        finally:
            db.close()

    def _register(self, names: list[str], prior: CameraPrior) -> int:
        """Write ``names`` into the database in their order, before their features, with the
        camera they share (``existing_camera``, else a new one for ``prior`` as COLMAP's
        ``single_camera`` creates it: the prior focal, or 1.2 times the larger side without one).
        COLMAP's threaded extraction writes the images it registers itself in the order they
        finish, and the global mapper's result follows the image ids: registered first, the ids
        follow the keyframes' order and the same input maps the same way."""
        import pycolmap

        camera = self.existing_camera(prior)
        db = pycolmap.Database.open(str(self.db))
        try:
            if camera is None:
                focal = prior.focal if prior.focal is not None else 1.2 * max(prior.width,
                                                                           prior.height)
                cam = pycolmap.Camera.create(0, pycolmap.CameraModelId.SIMPLE_PINHOLE, focal,
                                             prior.width, prior.height)
                # the parameters as COLMAP parses them from text (``set_prior`` does the same)
                cam.set_params_from_string(_camera_params(replace(prior, focal=focal)))
                cam.has_prior_focal_length = prior.focal is not None
                camera = int(db.write_camera(cam))
            known = {im.name for im in db.read_all_images()}
            for name in names:
                if name not in known:
                    db.write_image(pycolmap.Image(name=name, camera_id=camera))
        finally:
            db.close()
        return camera

    def set_prior(self, camera_id: int, prior: CameraPrior) -> None:
        """Give the new camera that ``extract`` created for a provisional prior (with a focal)
        the parameters it gives one for ``prior``, parsed from the same text as the CLI parses
        them: the database is then the one ``extract`` with ``prior`` writes (the features do
        not depend on the camera), so the features can be extracted before the prior is known."""
        import pycolmap

        assert prior.focal is not None
        db = pycolmap.Database.open(str(self.db))
        try:
            cam = db.read_camera(camera_id)
            cam.set_params_from_string(_camera_params(prior))
            db.update_camera(cam)
        finally:
            db.close()

    def existing_camera(self, prior: CameraPrior) -> int | None:
        """The database camera the new images share (``CameraPrior``): the map then keeps one
        camera, with the intrinsics its earlier updates refined, instead of starting another one
        at the uncalibrated prior."""
        if prior.existing_id is not None and self._has_camera(prior.existing_id, prior):
            return prior.existing_id
        for cid in prior.same_focal_ids:
            if self._has_camera(cid, prior, match_focal=True):
                return cid
        return None

    def _has_camera(self, camera_id: int, prior: CameraPrior, match_focal: bool = False) -> bool:
        import pycolmap

        if not self.db.exists():
            return False
        db = pycolmap.Database.open(str(self.db))
        try:
            if not db.exists_camera(camera_id):
                return False
            cam = db.read_camera(camera_id)
            if (cam.width, cam.height) != (prior.width, prior.height):
                return False
            if not match_focal:
                return True
            if prior.focal is None:
                return False
            f = float(np.mean(np.asarray(cam.params)[list(cam.focal_length_idxs())]))
            return abs(f - prior.focal) <= FOCAL_MATCH_REL * prior.focal
        finally:
            db.close()

    def match_pairs(self, pairs: set[tuple[int, int]], names: dict[int, str]) -> int:
        from oh_my_slam.mapping.retrieval import write_pair_list

        if not pairs:
            return 0
        lst = self.work / "pairs.txt"
        n = write_pair_list(lst, pairs, names)
        args = ["matches_importer", "--database_path", str(self.db), "--match_list_path",
                str(lst), "--match_type", "pairs", "--FeatureMatching.use_gpu", "0",
                "--log_level", "1", "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                "--TwoViewGeometry.random_seed", str(EXTEND_SEED),
                "--FeatureMatching.num_threads", str(SFM_THREADS)]
        _run(args, self.log_path)
        return n

    def rematch_lightglue(self, pairs: set[frozenset[str]]) -> dict[str, int]:
        """Match ``pairs`` (image names) again with LightGlue on their SIFT keypoints. COLMAP
        skips a pair the database already holds, so SIFT's matches and two-view geometry are
        taken out first; they are put back where LightGlue verifies fewer inliers than SIFT did,
        and for every pair when LightGlue fails (no model: COLMAP downloads it, 46 MB, into
        ``~/.cache/colmap`` on first use). Runs on the CPU: the CoreML provider cannot compile
        LightGlue's dynamic shapes and was 6x slower. Returns the pairs matched and verified
        before and after."""
        import pycolmap

        from oh_my_slam.mapping.retrieval import write_pair_list

        if not pairs:
            return {"pairs": 0, "verified_before": 0, "verified_after": 0}
        db = pycolmap.Database.open(str(self.db))
        try:
            ids = {im.name: im.image_id for im in db.read_all_images()}
            todo = sorted((ids[a], ids[b]) for a, b in (sorted(p) for p in pairs)
                          if a in ids and b in ids)
            kept: dict[tuple[int, int], tuple[Any, Any]] = {}
            for i, j in todo:
                m: Any = db.read_matches(i, j) if db.exists_matches(i, j) else None
                g: Any = db.read_two_view_geometry(i, j) if db.exists_two_view_geometry(i, j) \
                    else None
                kept[i, j] = (m, g)
                if m is not None:
                    db.delete_matches(i, j)
                if g is not None:
                    db.delete_two_view_geometry(i, j)
        finally:
            db.close()
        before = sum(g is not None and len(g.inlier_matches) >= MIN_INLIERS
                     for _, g in kept.values())
        lst = self.work / "pairs_lightglue.txt"
        names = {v: k for k, v in ids.items()}
        write_pair_list(lst, set(todo), names)
        args = ["matches_importer", "--database_path", str(self.db), "--match_list_path",
                str(lst), "--match_type", "pairs", "--FeatureMatching.use_gpu", "0",
                "--log_level", "1", "--FeatureMatching.type", "SIFT_LIGHTGLUE",
                "--TwoViewGeometry.random_seed", str(EXTEND_SEED),
                "--FeatureMatching.num_threads", str(SFM_THREADS)]
        try:
            _run(args, self.log_path)
            failed = False
        except SfmError as e:
            log.warning("LightGlue matching failed, the weak pairs keep SIFT's matches: %s", e)
            failed = True
        db = pycolmap.Database.open(str(self.db))
        try:
            after = 0
            for (i, j), (m, g) in kept.items():
                new = db.read_two_view_geometry(i, j) if db.exists_two_view_geometry(i, j) \
                    else None
                n_new = len(new.inlier_matches) if new is not None else 0
                n_old = len(g.inlier_matches) if g is not None else 0
                if failed or n_old > n_new:
                    if db.exists_matches(i, j):
                        db.delete_matches(i, j)
                    if new is not None:
                        db.delete_two_view_geometry(i, j)
                    if m is not None:
                        db.write_matches(i, j, m)
                    if g is not None:
                        db.write_two_view_geometry(i, j, g)
                    n_new = n_old
                after += n_new >= MIN_INLIERS
        finally:
            db.close()
        return {"pairs": len(todo), "verified_before": int(before), "verified_after": int(after)}

    def two_view_stats(self, names: set[str] | None = None) -> dict[str, float]:
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            id_name = {im.image_id: im.name for im in db.read_all_images()}
            pair_ids, geoms = db.read_two_view_geometries()
        finally:
            db.close()
        configs: Counter[int] = Counter()
        for pid, g in zip(pair_ids, geoms, strict=True):
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            if names is not None and not (id_name.get(a) in names and id_name.get(b) in names):
                continue
            if len(g.inlier_matches) >= 15:
                configs[int(g.config)] += 1
        total = sum(configs.values())
        rot = configs[_PANORAMIC] + configs[_PLANAR_OR_PANORAMIC]
        return {"verified_pairs": float(total),
                "rotation_fraction": rot / total if total else 0.0,
                "planar_fraction": configs[_PLANAR] / total if total else 0.0}

    def verified_pairs(self, min_inliers: int = 15) -> dict[frozenset[str], tuple[int, int]]:
        """(inlier matches, two-view configuration) of every verified image pair."""
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            id_name = {im.image_id: im.name for im in db.read_all_images()}
            pair_ids, geoms = db.read_two_view_geometries()
        finally:
            db.close()
        out = {}
        for pid, g in zip(pair_ids, geoms, strict=True):
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            if len(g.inlier_matches) >= min_inliers and a in id_name and b in id_name:
                out[frozenset((id_name[a], id_name[b]))] = (len(g.inlier_matches), int(g.config))
        return out

    def rotation_pairs(self, min_inliers: int = 15) -> set[frozenset[str]]:
        """Verified pairs whose two-view geometry is a pure rotation (panoramic)."""
        return {p for p, (_, c) in self.verified_pairs(min_inliers).items()
                if c in (_PANORAMIC, _PLANAR_OR_PANORAMIC)}

    def match_graph(self, min_inliers: int = 15) -> dict[str, set[str]]:
        """Adjacency of verified image pairs (by image name)."""
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            id_name = {im.image_id: im.name for im in db.read_all_images()}
            pair_ids, geoms = db.read_two_view_geometries()
        finally:
            db.close()
        adj: dict[str, set[str]] = {n: set() for n in id_name.values()}
        for pid, g in zip(pair_ids, geoms, strict=True):
            if len(g.inlier_matches) < min_inliers:
                continue
            a, b = pycolmap.pair_id_to_image_pair(int(pid))
            na, nb = id_name.get(a), id_name.get(b)
            if na and nb:
                adj[na].add(nb)
                adj[nb].add(na)
        return adj

    def connected(self, seeds: set[str], candidates: set[str], min_inliers: int = 15
                  ) -> set[str]:
        """Candidates reachable from ``seeds`` through verified pairs (via other candidates)."""
        adj = self.match_graph(min_inliers)
        seen = set(seeds)
        todo = list(seeds)
        while todo:
            n = todo.pop()
            for m in adj.get(n, ()):
                if m not in seen and (m in candidates or m in seeds):
                    seen.add(m)
                    todo.append(m)
        return seen & candidates

    def largest_component(self, names: set[str], min_inliers: int = 15) -> set[str]:
        adj = self.match_graph(min_inliers)
        left = set(names)
        best: set[str] = set()
        while left:
            start = left.pop()
            comp = {start}
            todo = [start]
            while todo:
                n = todo.pop()
                for m in adj.get(n, ()):
                    if m in names and m not in comp:
                        comp.add(m)
                        todo.append(m)
            left -= comp
            if len(comp) > len(best):
                best = comp
        return best

    # -- mapping ---------------------------------------------------------------------------------

    def _largest(self, recs: dict[int, Any], anchors: set[str] | None = None) -> Any | None:
        """The reconstruction with the most registered images; with ``anchors`` (the images of
        an input model being extended), the one that continues that model — COLMAP starts a
        separate reconstruction, in an unrelated frame, for images it cannot attach to it."""
        if not recs:
            return None
        if anchors:
            def held(r: Any) -> int:
                return sum(im.name in anchors for im in r.images.values() if im.has_pose)
            recs = {k: r for k, r in recs.items() if held(r) > 0}
            if not recs:
                return None
            return max(recs.values(), key=lambda r: (held(r), r.num_reg_images()))
        return max(recs.values(), key=lambda r: r.num_reg_images())

    def map_global(self, out: Path) -> SfmModel | None:
        import pycolmap

        opts = pycolmap.GlobalPipelineOptions()
        opts.num_threads = SFM_THREADS
        # seeded like the incremental extensions: the global positioner starts from random
        # positions, and COLMAP's default seed (-1) is time-seeded
        mapper = opts.mapper
        for o in (opts, mapper, mapper.rotation_averaging, mapper.global_positioning):
            o.random_seed = EXTEND_SEED
        mapper.num_threads = SFM_THREADS
        recs = pycolmap.global_mapping(str(self.db), str(self.image_dir), str(out), opts)
        rec = self._largest(recs)
        return None if rec is None else self._with_others(SfmModel(rec, "sfm-global"), recs)

    @staticmethod
    def _with_others(model: SfmModel, recs: dict[int, Any]) -> SfmModel:
        model.others = [r for r in recs.values() if r is not model.rec]
        model.notes["reconstructions"] = sorted((r.num_reg_images() for r in recs.values()),
                                                reverse=True)
        return model

    def map_incremental(self, out: Path, input_path: Path | None = None,
                        fix_existing: bool = False, constant_cameras: set[int] | None = None
                        ) -> SfmModel | None:
        """Incremental mapping; with ``input_path``, an extension of that reconstruction (its
        frames fixed with ``fix_existing``, the intrinsics of ``constant_cameras`` held), with a
        fixed random seed so that the same input extends a map the same way."""
        import pycolmap

        opts = pycolmap.IncrementalPipelineOptions()
        opts.extract_colors = False
        opts.fix_existing_frames = fix_existing
        opts.structure_less_registration_fallback = True
        opts.num_threads = SFM_THREADS
        if input_path:
            # only the input's continuation: with several models COLMAP goes on to start fresh
            # ones from the images left, the input's own images included, in an unrelated frame —
            # one of those holding as many of the input's images, and more images, was taken for
            # the extension and rejected as having moved its fixed frames
            opts.multiple_models = False
            opts.random_seed = EXTEND_SEED
            if constant_cameras:
                opts.constant_cameras = set(constant_cameras)
        recs = pycolmap.incremental_mapping(str(self.db), str(self.image_dir), str(out), opts,
                                            input_path=str(input_path) if input_path else "")
        if not input_path:
            rec = self._largest(recs)
            return None if rec is None else self._with_others(SfmModel(rec, "sfm-incremental"),
                                                              recs)
        base = SfmModel(pycolmap.Reconstruction(str(input_path)), "input")
        rec = self._largest(recs, set(base.registered))
        return None if rec is None else _back_onto(SfmModel(rec, "sfm-incremental"), base)

    @staticmethod
    def model_cameras(path: Path) -> set[int]:
        """Camera ids of a stored reconstruction."""
        import pycolmap

        return {int(c) for c in pycolmap.Reconstruction(str(path)).cameras}

    def image_intrinsics(self, names: set[str]) -> dict[str, tuple[int, Intrinsics]]:
        """(camera id, full-resolution intrinsics) in the database of each of ``names``."""
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            cams = {c.camera_id: c for c in db.read_all_cameras()}
            out = {}
            for im in db.read_all_images():
                if im.name in names:
                    cam = cams[im.camera_id]
                    K = np.asarray(cam.calibration_matrix())
                    out[im.name] = (int(im.camera_id), Intrinsics(
                        K[0, 0], K[1, 1], K[0, 2], K[1, 2], cam.width, cam.height, "colmap"))
            return out
        finally:
            db.close()

    def _posed_reconstruction(self, poses: dict[str, Pose], base: Any = None,
                              focal_scale: float = 1.0, cameras: dict[int, Any] | None = None
                              ) -> Any:
        """``base`` (or an empty reconstruction) plus images at the given camera-to-world poses.
        Images of ``base`` that are in ``poses`` keep their stored pose. Cameras not yet in the
        reconstruction come from ``cameras`` (the ones the poses were estimated with), else from
        the database, their focal length times ``focal_scale``."""
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            cams = {c.camera_id: c for c in db.read_all_cameras()}
            images = {im.name: im for im in db.read_all_images()}
        finally:
            db.close()
        rec = pycolmap.Reconstruction() if base is None else base
        for name, T in poses.items():
            if name not in images:
                continue
            im = images[name]
            if rec.exists_image(im.image_id):
                continue
            if not rec.exists_camera(im.camera_id):
                cam = cams[im.camera_id]
                if cameras and im.camera_id in cameras:
                    cam.params = np.asarray(cameras[im.camera_id].params, np.float64).tolist()
                if focal_scale != 1.0:
                    params = np.asarray(cam.params, np.float64).copy()
                    params[list(cam.focal_length_idxs())] *= focal_scale
                    cam.params = params.tolist()
                rec.add_camera_with_trivial_rig(cam)
            Tcw = T.inverse()
            img = pycolmap.Image(name=name, camera_id=im.camera_id, image_id=im.image_id)
            rec.add_image_with_trivial_frame(
                img, pycolmap.Rigid3d(pycolmap.Rotation3d(Tcw.R), Tcw.t))
        return rec

    def triangulate_with_poses(self, poses: dict[str, Pose], out: Path,
                               focal_scale: float = 1.0) -> SfmModel:
        """Reconstruction from known camera-to-world poses: the database matches triangulated,
        poses and intrinsics held (no bundle adjustment: the refined multi-view poses are kept).
        The cameras' focal lengths are the database's times ``focal_scale``."""
        import pycolmap

        rec = self._posed_reconstruction(poses, focal_scale=focal_scale)
        out.mkdir(parents=True, exist_ok=True)
        rec = pycolmap.triangulate_points(rec, str(self.db), str(self.image_dir), str(out),
                                          clear_points=True, refine_intrinsics=False)
        return SfmModel(rec, "multiview")

    def extend_with_poses(self, base_path: Path, poses: dict[str, Pose], out: Path,
                          method: str, cameras: dict[int, Any] | None = None) -> SfmModel:
        """The stored map model plus new images at known map poses; points re-triangulated
        (existing frames untouched). New cameras take the intrinsics of ``cameras`` where given
        (those the poses were estimated with: an incremental extension refines them)."""
        import pycolmap

        base = pycolmap.Reconstruction(str(base_path))
        rec = self._posed_reconstruction(poses, base, cameras=cameras)
        out.mkdir(parents=True, exist_ok=True)
        rec = pycolmap.triangulate_points(rec, str(self.db), str(self.image_dir), str(out),
                                          clear_points=False, refine_intrinsics=False)
        return SfmModel(rec, method)
