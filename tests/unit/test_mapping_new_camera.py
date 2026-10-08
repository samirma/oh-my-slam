"""An update whose keyframes bring a camera of their own (a wide-angle pan-tilt camera added to a
phone video's map) fits that camera's lens from its own matches before they meet the map's, and
the multi-view refinement of an update levels its keyframes with their gravity estimates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, panorama
from oh_my_slam.mapping.panorama import PoseFit
from oh_my_slam.reconstruction.gravity import GravityEstimate
from tests.unit.test_mapping_cov_api import FakeSfm, ctx_of, jpg, nf, quiet

LENS = Intrinsics(1390.0, 1390.0, 960.0, 540.0, 1920, 1080, "colmap", -0.51)
WIDE = Intrinsics(1500.0, 1500.0, 960.0, 540.0, 1920, 1080, "colmap")


class _Sfm:
    """The database of an update: each new keyframe's camera, the matched component."""

    def __init__(self, cams: dict[str, tuple[int, Intrinsics]], component: set[str]) -> None:
        self.cams = dict(cams)
        self.component = component
        self.db = Path("/nonexistent/db")
        self.distortion: tuple[set[str], float, float] | None = None

    def image_intrinsics(self, names: set[str]) -> dict[str, tuple[int, Intrinsics]]:
        return {n: c for n, c in self.cams.items() if n in names}

    def largest_component(self, names: set[str]) -> set[str]:
        return self.component & names

    def set_distortion(self, names: set[str], focal_scale: float, distortion: float) -> None:
        self.distortion = (names, focal_scale, distortion)
        for n in names:
            cid, c = self.cams[n]
            self.cams[n] = (cid, Intrinsics(c.fx * focal_scale, c.fy * focal_scale, c.cx, c.cy,
                                            c.width, c.height, "colmap", distortion))


def _world(monkeypatch: pytest.MonkeyPatch, n: int = 5, fit: PoseFit | None = None,
           turn_deg: float = 0.3) -> SimpleNamespace:
    new = [nf(100 + k) for k in range(n)]
    for f in new:
        f.full_size = (1920, 1080)
    old = [SimpleNamespace(camera_id=1)]
    ctx = ctx_of(new, old)
    seen: dict[str, Any] = {"refined": []}

    def multiview(ctx_: Any, todo: list[Any], pool: list[Any], client: Any,
                  temporal: bool = False) -> dict[str, Pose]:
        seen["todo"], seen["pool"] = [v.name for v in todo], pool
        return {v.name: Pose.identity() for v in todo}

    def refine(name: str, out: PoseFit) -> Any:
        def run(pairs: Any, views: Any, free: Any, **kw: Any) -> PoseFit:
            seen["refined"].append((name, kw))
            return out
        return run

    turn = PoseFit({}, 1.0, pairs=4, median_after_deg=turn_deg)
    final = fit or PoseFit({}, 0.927, pairs=4, median_after_deg=0.43, distortion=-0.51)
    monkeypatch.setattr(api, "_multiview_poses", multiview)
    monkeypatch.setattr(panorama, "verified_matches", lambda db, names: [])
    monkeypatch.setattr(panorama, "refine_poses", lambda *a, **kw: (
        refine("poses", turn)(*a, **kw) if kw.get("use_depth") is False
        else refine("poses", final)(*a, **kw)))
    monkeypatch.setattr(panorama, "refine_turning", refine("turning", final))
    monkeypatch.setattr(api, "_rerun_undistorted", lambda ctx_, todo, client, progress:
                        seen.__setitem__("rerun", todo))
    sfm = _Sfm({jpg(100 + k): (2, WIDE) for k in range(n)}, {jpg(100 + k) for k in range(n)})
    return SimpleNamespace(ctx=ctx, sfm=sfm, seen=seen, new=new)


def test_a_new_camera_turning_in_place_gets_its_lens(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _world(monkeypatch)
    msgs: list[str] = []
    assert api._calibrate_new_camera(w.ctx, w.sfm, None, msgs.append)
    assert w.seen["pool"] == []  # the new keyframes alone, in a frame of their own
    assert [r[0] for r in w.seen["refined"]] == ["poses", "turning"]
    assert w.seen["refined"][1][1]["refine_focal"]
    names, scale, k = w.sfm.distortion
    assert names == {jpg(100 + i) for i in range(5)} and (scale, k) == (0.927, -0.51)
    assert w.ctx.lens_camera == 2
    assert [K_.k for _, K_ in w.seen["rerun"]] == [-0.51] * 5  # inferred again undistorted
    note = w.ctx.notes["new_camera"]
    assert note["turning"] and note["camera"] == 2 and note["corner_shift"] > 0.15
    assert any("turning in place" in m for m in msgs)


def test_a_moving_new_camera_is_refined_with_its_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _world(monkeypatch, turn_deg=2.0)
    assert api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    assert [r[0] for r in w.seen["refined"]] == ["poses", "poses"]
    assert not w.ctx.notes["new_camera"]["turning"]


@pytest.mark.parametrize("fit", [
    PoseFit({}, 1.02, pairs=4, median_after_deg=0.2),  # a pinhole: SfM refines its focal length
    PoseFit({}, 1.0, pairs=4, median_after_deg=1.6, distortion=-0.4),  # a fit too poor to take
])
def test_no_lens_is_taken_from_a_pinhole_or_a_poor_fit(monkeypatch: pytest.MonkeyPatch,
                                                        fit: PoseFit) -> None:
    w = _world(monkeypatch, fit=fit)
    assert not api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    assert w.sfm.distortion is None and w.ctx.lens_camera is None and "rerun" not in w.seen
    assert "new_camera" in w.ctx.notes


def test_only_a_camera_of_its_own_without_a_lens_is_fitted(monkeypatch: pytest.MonkeyPatch
                                                          ) -> None:
    w = _world(monkeypatch)
    w.ctx.old_frames = [SimpleNamespace(camera_id=2)]  # the map's camera
    assert not api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    w = _world(monkeypatch)
    w.sfm.cams[jpg(100)] = (3, WIDE)  # two cameras
    assert not api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    w = _world(monkeypatch)
    w.sfm.cams = {n: (2, LENS) for n in w.sfm.cams}  # its lens is known
    assert not api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    w = _world(monkeypatch)
    w.sfm.component = {jpg(100), jpg(101)}  # too few matched keyframes
    assert not api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    assert "todo" not in w.seen and "new_camera" not in w.ctx.notes


def test_many_keyframes_are_fitted_evenly_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    n = api.CALIBRATION_MAX_KEYFRAMES + 20
    w = _world(monkeypatch, n=n)
    assert api._calibrate_new_camera(w.ctx, w.sfm, None, quiet)
    todo = w.seen["todo"]
    assert len(todo) == api.CALIBRATION_MAX_KEYFRAMES
    assert todo[0] == jpg(100) and todo[-1] == jpg(100 + n - 1)
    assert len(w.seen["rerun"]) == n  # every keyframe of the camera is inferred again


class _UpdateSfm(FakeSfm):
    calls_log: ClassVar[list[Any]] = []

    def match_pairs(self, pairs: set[tuple[int, int]], names: dict[int, str]) -> int:
        self.calls_log.append(("match", set(pairs)))
        return len(pairs)

    def unmatch(self, pairs: set[tuple[int, int]], names: dict[int, str]) -> None:
        self.calls_log.append(("unmatch", set(pairs)))


@pytest.mark.parametrize("calibrated", [True, False])
def test_an_update_matches_its_own_pairs_before_the_lens_and_the_rest_with_it(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calibrated: bool) -> None:
    monkeypatch.setattr(api, "check_versions", lambda: "4.2.1")
    monkeypatch.setattr(api, "Sfm", _UpdateSfm)
    _UpdateSfm.calls_log = []
    monkeypatch.setattr(api, "_calibrate_new_camera",
                        lambda ctx, sfm, client, progress: calibrated)
    own, cross = {(100, 101), (101, 102)}, {(1, 100), (2, 101)}
    monkeypatch.setattr(api, "_pairs_update", lambda ctx, is_video: own | cross)
    monkeypatch.setattr(api, "_extend", lambda *a: "extended")
    old = [SimpleNamespace(index=i, image=f"frames/{jpg(i)}", width=640, height=480,
                           camera_id=1, source="a.jpg") for i in (1, 2)]
    tx = SimpleNamespace(clone_for_edit=lambda rel: tmp_path / rel,
                         stage=lambda rel: tmp_path / "staging" / rel)
    ctx = ctx_of([nf(i) for i in (100, 101, 102)], old, tmp=tmp_path, tx=tx)
    assert api._run_sfm(ctx, False, None, quiet) == "extended"
    expected = [("match", own), ("unmatch", own), ("match", own | cross)] if calibrated \
        else [("match", own), ("match", cross)]
    assert _UpdateSfm.calls_log == expected


def _grav(up: list[float], unc: float = 0.3, source: str = "geocalib") -> GravityEstimate:
    return GravityEstimate(np.asarray(up, float), source, unc, unc)


def test_an_update_of_a_levelled_map_takes_its_keyframes_gravity_as_priors() -> None:
    new = [nf(1, up_cam=[0.0, -1.0, 0.0]), nf(2, up_cam=[0.0, -1.0, 0.1]), nf(3)]
    new[0].frame.gravity = _grav([0.0, -1.0, 0.0])
    new[1].frame.gravity = _grav([0.0, -1.0, 0.1], unc=3.0)
    new[2].frame.gravity = _grav([0.0, -1.0, 0.0], source="default")
    old = [SimpleNamespace(camera_id=1)]
    ctx = ctx_of(new, old, meta={"map_frame": {"gravity_aligned": True}})
    pri = api._gravity_priors(ctx, {jpg(1), jpg(2), jpg(3)})
    assert pri is not None and set(pri) == {jpg(1), jpg(2)}  # no estimate: no prior
    assert pri[jpg(1)][1] == api.GRAVITY_PRIOR_MIN_DEG  # GeoCalib's 0.4° is not taken
    assert pri[jpg(2)][1] == pytest.approx(np.hypot(3.0, 3.0))
    assert set(api._gravity_priors(ctx, {jpg(2)}) or {}) == {jpg(2)}
    assert api._gravity_priors(ctx_of(new, [], meta={"map_frame": {"gravity_aligned": True}}),
                               {jpg(1)}) is None  # a new map: no levelled frame yet
    assert api._gravity_priors(ctx_of(new, old, meta={}), {jpg(1)}) is None
