"""Structure from motion with COLMAP 4.2.

Features and matching run through the Homebrew ``colmap`` CLI (its ONNX/CoreML build is needed for
ALIKED/LightGlue; the PyPI pycolmap wheel has no ONNX); mapping, triangulation and bundle
adjustment run through pycolmap on the same database. Both must be 4.2.x.

New map: global mapping (GLOMAP) → incremental if < 60 % placed → multi-view (MapAnything)
poses refined with the verified matches and monocular depth (``panorama``) + triangulation.
Rotation-dominant input goes straight to the multi-view path. SfM poses without triangulated
support are not accepted (``vet``); the mapper then joins what the main reconstruction lacks or
got wrong in scale or tilt (``mapping.trajectory``).
Update: incremental mapping with the existing frames fixed; keyframes it cannot place (all of
them for rotation-dominant input) get anchored, refined multi-view poses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oh_my_slam.core.errors import OhMySlamError
from oh_my_slam.core.log import get_logger
from oh_my_slam.core.types import Intrinsics, Pose

log = get_logger("oh_my_slam.sfm")

# Gate G4 (measured on the user's inputs, TUM data blocked on U2): SIFT registers as many frames
# as ALIKED+LightGlue and is ~30x faster on this CPU/CoreML build.
FEATURES = os.environ.get("OH_MY_SLAM_FEATURES", "sift").lower()
MIN_PLACED_FRACTION = 0.6
ROTATION_PAIR_FRACTION = 0.5
ROTATION_BASELINE_RATIO = 0.02
MAX_FEATURES = 4096

# COLMAP TwoViewGeometry configuration codes
_PLANAR, _PANORAMIC, _PLANAR_OR_PANORAMIC = 4, 5, 6


class SfmError(OhMySlamError):
    pass


def colmap_bin() -> str:
    exe = os.environ.get("OH_MY_SLAM_COLMAP", "colmap")
    path = shutil.which(exe)
    if path is None:
        raise SfmError("COLMAP not found — install it with: brew install colmap "
                       "(then ./scripts/install_tools.sh)")
    return path


def check_versions() -> str:
    import pycolmap

    out = subprocess.run([colmap_bin(), "version"], capture_output=True, text=True).stdout
    cli = out.split()[1] if out.startswith("COLMAP") else "?"
    if not cli.startswith("4.2") or not pycolmap.__version__.startswith("4.2"):
        raise SfmError(f"COLMAP CLI {cli} and pycolmap {pycolmap.__version__} must both be 4.2.x")
    return cli


def _run(args: list[str], log_path: Path) -> None:
    with log_path.open("a") as f:
        res = subprocess.run([colmap_bin(), *args], stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        lines = log_path.read_text(errors="replace").splitlines()
        fatal = [ln for ln in lines if ln.startswith(("F2", "E2")) or "Check failed" in ln]
        tail = (fatal or lines)[-6:]
        raise SfmError(f"colmap {args[0]} failed ({res.returncode}): " + " | ".join(tail))


@dataclass
class CameraPrior:
    width: int
    height: int
    focal: float | None = None  # full-resolution pixels
    existing_id: int | None = None


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


FIXED_ROT_TOL_DEG = 2.0
FIXED_POS_TOL = 0.05  # of the fixed frames' spread


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


def _back_onto(model: SfmModel, base: SfmModel) -> SfmModel | None:
    """COLMAP re-normalises an extended reconstruction — a similarity away from its input, the
    fixed frames included — so map it back onto the input with the similarity that takes the
    frames of both onto their input poses. None when those frames did not stay rigid (or are too
    few to tell)."""
    from oh_my_slam.mapping.frame import similarity_by_poses

    common = sorted(set(model.registered) & set(base.registered))
    if len(common) < 2:
        return None
    ref = [base.pose(n) for n in common]
    sim = similarity_by_poses([model.pose(n) for n in common], ref)
    model.transform(sim.s, sim.R, sim.t)
    centres = np.array([r.t for r in ref])
    tol = max(1e-4, FIXED_POS_TOL * float(np.linalg.norm(centres - centres.mean(0), axis=1).max()))
    for n, r in zip(common, ref, strict=True):
        p = model.pose(n)
        rot = np.degrees(np.arccos(np.clip((np.trace(p.R.T @ r.R) - 1) / 2, -1.0, 1.0)))
        if rot > FIXED_ROT_TOL_DEG or np.linalg.norm(p.t - r.t) > tol:
            log.warning("incremental extension moved the fixed frame %s (%.2f°, %.3g); "
                        "its result is not used", n, rot, float(np.linalg.norm(p.t - r.t)))
            return None
    return model


class Sfm:
    def __init__(self, db_path: Path, image_dir: Path, work_dir: Path,
                 features: str = FEATURES) -> None:
        self.db = Path(db_path)
        self.image_dir = Path(image_dir)
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.log_path = self.work / "colmap.log"

    # -- features & matching ---------------------------------------------------------------------

    def extract(self, names: list[str], prior: CameraPrior) -> int:
        """Extract features for ``names`` (relative to ``image_dir``) sharing one camera."""
        lst = self.work / "extract_list.txt"
        lst.write_text("\n".join(names) + "\n")
        args = [
            "feature_extractor", "--database_path", str(self.db), "--image_path",
            str(self.image_dir), "--image_list_path", str(lst),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--FeatureExtraction.use_gpu", "0", "--log_level", "1",
        ]
        if self.features == "aliked":
            args += ["--FeatureExtraction.type", "ALIKED_N16ROT"]
        else:
            args += ["--FeatureExtraction.type", "SIFT", "--SiftExtraction.max_num_features",
                     str(MAX_FEATURES)]
        if prior.existing_id is not None and self._has_camera(prior.existing_id, prior):
            args += ["--ImageReader.existing_camera_id", str(prior.existing_id)]
        else:
            args += ["--ImageReader.single_camera", "1"]
            if prior.focal is not None:
                args += ["--ImageReader.camera_params",
                         f"{prior.focal:.4f},{prior.width / 2:.4f},{prior.height / 2:.4f}"]
        _run(args, self.log_path)
        return self._camera_of(names[0])

    def _has_camera(self, camera_id: int, prior: CameraPrior) -> bool:
        import pycolmap

        if not self.db.exists():
            return False
        db = pycolmap.Database.open(str(self.db))
        try:
            if not db.exists_camera(camera_id):
                return False
            cam = db.read_camera(camera_id)
            return (cam.width, cam.height) == (prior.width, prior.height)
        finally:
            db.close()

    def _camera_of(self, name: str) -> int:
        import pycolmap

        db = pycolmap.Database.open(str(self.db))
        try:
            return int(db.read_image_with_name(name).camera_id)
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
                "--log_level", "1"]
        args += ["--FeatureMatching.type",
                 "ALIKED_LIGHTGLUE" if self.features == "aliked" else "SIFT_BRUTEFORCE"]
        _run(args, self.log_path)
        return n

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
        opts.num_threads = -1
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
                        fix_existing: bool = False) -> SfmModel | None:
        import pycolmap

        opts = pycolmap.IncrementalPipelineOptions()
        opts.extract_colors = False
        opts.fix_existing_frames = fix_existing
        opts.structure_less_registration_fallback = True
        opts.num_threads = -1
        recs = pycolmap.incremental_mapping(str(self.db), str(self.image_dir), str(out), opts,
                                            input_path=str(input_path) if input_path else "")
        if not input_path:
            rec = self._largest(recs)
            return None if rec is None else self._with_others(SfmModel(rec, "sfm-incremental"),
                                                              recs)
        base = SfmModel(pycolmap.Reconstruction(str(input_path)), "input")
        rec = self._largest(recs, set(base.registered))
        return None if rec is None else _back_onto(SfmModel(rec, "sfm-incremental"), base)

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
                              focal_scale: float = 1.0) -> Any:
        """``base`` (or an empty reconstruction) plus images at the given camera-to-world poses.
        Images of ``base`` that are in ``poses`` keep their stored pose. Cameras not yet in the
        reconstruction come from the database, their focal length times ``focal_scale``."""
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
                               refine_intrinsics: bool = True, bundle: bool = True,
                               focal_scale: float = 1.0) -> SfmModel:
        """Reconstruction from known camera-to-world poses: triangulate the database matches,
        then (``bundle``) bundle-adjust (intrinsics shared per camera). The cameras' focal lengths
        are the database's times ``focal_scale``."""
        import pycolmap

        rec = self._posed_reconstruction(poses, focal_scale=focal_scale)
        out.mkdir(parents=True, exist_ok=True)
        rec = pycolmap.triangulate_points(rec, str(self.db), str(self.image_dir), str(out),
                                          clear_points=True, refine_intrinsics=False)
        if bundle:
            ba = pycolmap.BundleAdjustmentOptions()
            ba.refine_focal_length = refine_intrinsics
            ba.refine_principal_point = False
            ba.refine_extra_params = False
            pycolmap.bundle_adjustment(rec, ba)
        return SfmModel(rec, "multiview")

    def extend_with_poses(self, base_path: Path, poses: dict[str, Pose], out: Path,
                          method: str) -> SfmModel:
        """The stored map model plus new images at known map poses; points re-triangulated
        (existing frames untouched)."""
        import pycolmap

        base = pycolmap.Reconstruction(str(base_path))
        rec = self._posed_reconstruction(poses, base)
        out.mkdir(parents=True, exist_ok=True)
        rec = pycolmap.triangulate_points(rec, str(self.db), str(self.image_dir), str(out),
                                          clear_points=False, refine_intrinsics=False)
        return SfmModel(rec, method)
