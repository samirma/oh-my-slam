"""Body links in the mapper (oms/2.15; map.md §1 "Body links", "Keyframe depth scale", "Scale"
item 3, §2 "Match graph" linked edge and linked pose; api.md §4 ``body_link``) on AiNex-like
views (``tests.synth.stereo``): a head panned about a vertical neck axis 0.057 m behind the
camera with the body still (motion source ``none``), and a body turning in place."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import body as bl
from oh_my_slam.mapping import motion
from oh_my_slam.mapping.anchored import AnchoredMap, FrameIn, Placement
from tests.synth import stereo as syn

K = Intrinsics(syn.F, syn.F, syn.CX, syn.CY, syn.W, syn.H, "model")
NECK = 0.057
T_WB = np.eye(4)
T_WB[:3, 3] = [-NECK, 0.0, 0.0]  # the body frame: the neck axis, z up, x forward


@pytest.fixture(scope="module")
def pans() -> dict[int, syn.Shot]:
    room = syn.ainex_room()
    return {p: syn.render(room, syn.panned(float(p), NECK), i)
            for i, p in enumerate((0, 25, 50, -25))}


def mount_of(T_cam: np.ndarray) -> dict[str, Any]:
    return {"T_body_cam": (np.linalg.inv(T_WB) @ T_cam).tolist(), "source": "measured"}


def body(name: str, s: syn.Shot, frm: str | None, stamp: float = 0.0, standing: bool = True,
         M: np.ndarray | None = None, sources: tuple[str, str] = ("none", "none")) -> bl.Body:
    b: dict[str, Any] = {"image_stamp_s": stamp, "standing": standing, "mount": mount_of(s.T),
                         "mount_stamp_s": stamp}
    if frm is not None:
        b["motion"] = {"from": frm, "T_from_body": (np.eye(4) if M is None else M).tolist(),
                       "yaw_source": sources[0], "sigma_yaw_deg": 0.0,
                       "translation_source": sources[1], "sigma_pos_m": 0.0}
    return bl.body_of(name, "cam", b, None)


def prior(T: np.ndarray, trust: str = "metric", sp: float = 0.05, sr: float = 2.0
          ) -> dict[str, Any]:
    return {"trust": trust, "T_map_cam": T.tolist(), "sigma_pos_m": sp, "sigma_rot_deg": sr}


def frame(name: str, s: syn.Shot, b: bl.Body | None, pr: dict[str, Any] | None = None,
          **kw: Any) -> FrameIn:
    d = syn.model_depth(s.depth)
    return FrameIn(name, "cam", s.rgb, K, pr, None, None, syn.HEAD_M, 0.01, 1000,
                   lambda d=d: (d, K, None, None), body=b, **kw)


def origin(tmp_path: Path, pans: dict[int, syn.Shot]) -> AnchoredMap:
    m = AnchoredMap(tmp_path / "m")
    m.update([frame("p0", pans[0], body("p0", pans[0], None), prior(pans[0].T))],
             lambda _: None)
    return m


def err(T: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    return an.pose_residual(T, truth)


# -- a head pan with the body still --------------------------------------------------------------


def test_a_head_pan_registers_and_measures_its_scale(tmp_path: Path,
                                                     pans: dict[int, syn.Shot]) -> None:
    """The pan look sent one view per job, no prior after the origin: each view is placed by its
    images (a points fit the linked pose confirms) or by its linked edge, its link used; its
    keyframe's depth scale is measured on body-link pairs (link_neighbours), the model's 0.8 x
    metres (about 1.25)."""
    m = origin(tmp_path, pans)
    pls = []
    for p, frm in ((25, "p0"), (50, "p25")):
        pls += m.update([frame(f"p{p}", pans[p], body(f"p{p}", pans[p], frm))], lambda _: None)
    for pl, p in zip(pls, (25, 50), strict=True):
        assert pl.status == "registered", pl
        assert pl.link is not None and pl.link["state"] == "used" and pl.link["reason"] is None
        assert pl.link["baseline_m"] == pytest.approx(2 * NECK * math.sin(math.radians(12.5)),
                                                      rel=1e-3)
        assert pl.link["sigma_m"] / pl.link["baseline_m"] < 0.20
        assert pl.link["rotation_residual_deg"] < 1.0
        assert pl.T is not None
        dp, dr = err(pl.T, pans[p].T)
        assert dp < 0.03 and dr < 1.0
    k = m.confirmed()[-1]
    assert k.stereo_links and set(k.stereo_links) <= set(k.stereo_pairs or [])
    assert k.link_measured()
    assert k.stereo_scale == pytest.approx(1.25, rel=0.15)
    m.save()
    again = AnchoredMap.load(m.root)
    k2 = again.keyframe(k.id)
    assert k2 is not None and k2.body == k.body and k2.stereo_links == k.stereo_links
    # the link-measured scales enter k only where the map's scale is not the metric priors'
    # (the priors here span nothing: allowed) once three agree
    assert again.scale.links_ok


def test_a_linked_edge_places_a_view_without_a_points_fit(tmp_path: Path,
                                                          pans: dict[int, syn.Shot],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any points fit (as on references lifted at a wrong scale), the view is placed by
    its linked edge: the images' rotation and the link's offset (method ``linked``), without a
    prior and with one (fused, its residual against the prior)."""
    m = origin(tmp_path, pans)
    monkeypatch.setattr(AnchoredMap, "_image_pose", lambda *a, **k: None)
    snap = m._snapshot()
    pl = m.update([frame("p25", pans[25], body("p25", pans[25], "p0"))], lambda _: None)[0]
    assert pl.status == "registered" and pl.method == "linked" and pl.T is not None
    dp, dr = err(pl.T, pans[25].T)
    assert dp < 0.02 and dr < 0.5
    assert pl.refs and pl.align_T is not None
    m._restore(snap)
    m.keyframes = [k for k in m.keyframes if k.name == "p0"]
    pl = m.update([frame("p25b", pans[25], body("p25b", pans[25], "p0"),
                         prior(pans[25].T, "drifting", 0.2, 5.0))], lambda _: None)[0]
    assert pl.status == "registered" and pl.method == "linked"
    assert pl.prior_residual is not None and pl.prior_residual[0] < 0.05
    # a linked pose is no image-only pose for the drift estimate (map.md §1)
    assert m.drift.support == [] or all(
        not np.allclose(c, pans[25].T[:3, 3]) for c in m.drift.support)


def test_a_points_fit_that_conflicts_with_the_linked_pose_is_no_placement(
        tmp_path: Path, pans: dict[int, syn.Shot], monkeypatch: pytest.MonkeyPatch) -> None:
    """A points fit 0.5 m along the view (a reference lifted at a wrong scale) beside a linked
    pose: the linked pose places the frame."""
    m = origin(tmp_path, pans)
    real = AnchoredMap._image_pose

    def slid(self: AnchoredMap, *a: Any, **k: Any) -> Any:
        got = real(self, *a, **k)
        if got is None:
            return None
        img, corr, share = got
        s = copy.copy(img[0])
        s.c = s.c + 0.5 * s.R[:, 2]
        return (s, *img[1:]), corr, share

    monkeypatch.setattr(AnchoredMap, "_image_pose", slid)
    pl = m.update([frame("p25", pans[25], body("p25", pans[25], "p0"))], lambda _: None)[0]
    assert pl.status == "registered" and pl.method == "linked"
    assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.02


def test_link_verdicts_with_a_prior(tmp_path: Path, pans: dict[int, syn.Shot],
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """The prior checks with the linked pose as the hypothesis: a prior far off (conflict: kept
    ``conflict_unsupported`` below the measurable-pair threshold, rejected ``prior_conflict``
    above it), a loose linked pose (kept ``loose_image_pose``), a failed image check (kept), each
    kept frame at the linked pose."""
    m = origin(tmp_path, pans)
    monkeypatch.setattr(AnchoredMap, "_image_pose", lambda *a, **k: None)
    far = pans[25].T.copy()
    far[:3, 3] += [1.0, 0.0, 0.0]
    snap = m._snapshot()

    def run(name: str, pr: dict[str, Any], **kw: Any) -> Placement:
        m._restore(snap)
        return m.update([frame(name, pans[25], body(name, pans[25], "p0"), pr, **kw)],
                        lambda _: None)[0]

    pl = run("a", prior(far, "metric", 0.05, 2.0))
    assert pl.link is not None
    if pl.status == "rejected":
        assert pl.reason == "prior_conflict"
    else:
        assert pl.status == "kept_at_prior" and pl.detail == "conflict_unsupported"
        assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.03  # at the linked pose
        assert pl.link["state"] == "used"
    real_linked = AnchoredMap._linked

    def measurable(self: AnchoredMap, f: FrameIn, feats: Any) -> Any:
        ln = real_linked(self, f, feats)
        if ln is not None and ln.pose is not None:
            ln.measurable = not ln.measurable
        return ln

    with monkeypatch.context() as mp:
        mp.setattr(AnchoredMap, "_linked", measurable)
        pl2 = run("a2", prior(far, "metric", 0.05, 2.0))
    assert {pl.status, pl2.status} == {"rejected", "kept_at_prior"}
    with monkeypatch.context() as mp:
        mp.setattr(an, "CONFIRM_SIGMA_ROT_DEG", 0.0)  # a linked pose too loose to confirm
        pl = run("b", prior(pans[25].T))
    assert pl.status == "kept_at_prior" and pl.detail == "loose_image_pose"
    assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.03  # at the linked pose
    pl = run("c", prior(pans[25].T), unconfirmed="blur")
    assert pl.status == "kept_at_prior" and pl.reason == "blur" and pl.detail is None
    assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.03  # at the linked pose
    pl = run("d", prior(far), unconfirmed="blur")
    assert pl.status == "kept_at_prior" and pl.detail == "conflict_unsupported"
    assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.03  # at the linked pose


def test_link_reasons(tmp_path: Path, pans: dict[int, syn.Shot],
                      monkeypatch: pytest.MonkeyPatch) -> None:
    m = origin(tmp_path, pans)

    def rep(b: bl.Body | None) -> str | None:
        row = m.link_report(b)[0]
        return None if row is None else row["reason"]

    assert m.link_report(None) == (None, None, None)
    assert rep(body("x", pans[25], None)) is None
    m.set_bodies([])
    assert rep(body("x", pans[25], "nowhere")) == "from_unknown"
    gone = m._linked(frame("x", pans[25], body("x", pans[25], "nowhere")), an.Features(
        np.zeros((0, 2)), np.zeros((0, 128), np.float32)))
    assert gone is not None and gone.raw is None and gone.absolute() is None
    p0 = m.confirmed()[0]
    body0, p0.body = p0.body, None
    m.set_bodies([])
    assert rep(body("x", pans[25], "p0")) == "from_no_mount"
    p0.body = body0
    m.set_bodies([])
    assert rep(body("x", pans[25], "p0", stamp=0.0, standing=False)) is not None
    stale = body("x", pans[25], "p0")
    assert stale.mount is not None
    stale.mount.fresh = False
    assert rep(stale) == "stale_mount"
    m._bodies["p0"].mount.fresh = False  # type: ignore[union-attr]
    assert rep(body("x", pans[25], "p0")) == "from_stale_mount"
    m.set_bodies([])
    m._bodies["p0"].standing = False
    assert rep(body("x", pans[25], "p0")) == "from_moving"
    assert rep(body("x", pans[25], "p0", sources=("imu", "lidar_match"))) == "no_match"
    m.set_bodies([])
    p0.state = "provisional"
    assert rep(body("x", pans[25], "p0")) == "from_unconfirmed"
    p0.state = "fused"
    # no matches: the link's raw pose only; the images contradicting the link: inconsistent
    f = frame("x", pans[25], body("x", pans[25], "p0"))
    feats = an.features(f.rgb)
    with monkeypatch.context() as mp:
        mp.setattr(motion, "pair", lambda *a, **k: motion.Pair(0, "no_overlap"))
        ln = m._linked(f, feats)
        assert ln is not None and ln.raw is not None and ln.pose is None
        assert ln.row["reason"] == "no_match"
    assert m._linked(f, an.Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32))
                     ).pose is None  # type: ignore[union-attr]
    twisted = body("x", pans[25], "p0", M=np.eye(4), sources=("imu", "imu"))
    assert twisted.motion is not None
    R = np.eye(4)
    a = math.radians(20.0)
    R[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    twisted.motion.T = R
    twisted.motion.sigma_yaw_deg = 1.0
    ln = m._linked(frame("x", pans[25], twisted), feats)
    assert ln is not None and ln.inconsistent and ln.row["reason"] == "inconsistent"
    assert ln.row["rotation_residual_deg"] > 10.0
    # a kept frame whose link the images contradict keeps its prior
    pl = an.AnchoredMap._finish_link(Placement("x", "kept_at_prior", "no_overlap", np.eye(4),
                                               0.1, 1.0), ln)
    assert np.allclose(pl.T, np.eye(4)) and pl.link is not None  # type: ignore[arg-type]
    assert m._link_key(frame("y", pans[25], None)) == ()
    assert m._link_key(frame("y", pans[25], body("y", pans[25], "nowhere"))) == ()
    assert m._link_key(frame("y", pans[25], body("y", pans[25], "p0")))[0] == p0.id


def test_a_kept_frame_takes_the_raw_linked_pose(tmp_path: Path, pans: dict[int, syn.Shot],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """No matches with its ``from`` frame (a turn past the field of view) and no image pose: the
    frame is kept, at the link's raw pose with its composed sigmas."""
    m = origin(tmp_path, pans)
    monkeypatch.setattr(AnchoredMap, "_image_pose", lambda *a, **k: None)
    monkeypatch.setattr(motion, "pair", lambda *a, **k: motion.Pair(0, "no_overlap"))
    pl = m.update([frame("p25", pans[25], body("p25", pans[25], "p0"),
                         prior(pans[25].T, "drifting", 0.3, 8.0))], lambda _: None)[0]
    assert pl.status == "kept_at_prior" and pl.T is not None
    assert pl.link is not None and pl.link["state"] == "used"
    assert err(pl.T, pans[25].T)[0] < 0.01
    assert pl.sigma_pos_m is not None and pl.sigma_pos_m < 0.1


# -- the scale vote ------------------------------------------------------------------------------


def test_link_votes_pool_with_camera_heights() -> None:
    s = an.ScaleState()
    s.links = [[1.2, 0.1], [1.25, 0.1], [1.3, 0.1]]
    assert s.link_votes() is None  # not allowed
    s.links_ok = True
    got = s.link_votes()
    assert got is not None and got[0] == pytest.approx(1.25, rel=0.01)
    s.estimate(0.0)
    assert s.source == "baseline" and s.k == pytest.approx(got[0])
    s.links = [[1.2, 0.1], [1.25, 0.1]]
    assert s.link_votes() is None  # fewer than LINK_SCALE_MIN
    s.links = [[1.2, 0.1], [2.5, 0.05], [0.5, 0.05], [1.0, 0.05]]
    assert s.link_votes() is None  # no majority agrees
    s.links = [[1.2, 0.05], [1.25, 0.05], [1.3, 0.05]]
    s.heights = [[0.02, 1.0]]  # a camera height: k 1.0 with sigma about 0.03
    s.estimate(0.0)
    assert s.source == "camera_height" and 1.0 < s.k < 1.25
    s.links = [[1.2, 0.002], [1.2, 0.002], [1.2, 0.002]]
    s.estimate(0.0)
    assert s.source == "baseline" and s.k == pytest.approx(1.2, rel=0.02)
    back = an.ScaleState.from_json(s.to_json())
    assert back.links == s.links and back.links_ok


def test_links_never_enter_a_metric_or_depth_scale(tmp_path: Path,
                                                   pans: dict[int, syn.Shot]) -> None:
    m = origin(tmp_path, pans)
    assert m._links_allowed()
    m.keyframes[0].depth_kind = "sensor"
    assert not m._links_allowed()
    m.keyframes[0].depth_kind = "mono"
    k2 = an.Keyframe.from_json({**m.keyframes[0].to_json(), "id": "k9", "name": "q"})
    assert k2.prior is not None
    k2.T = k2.T.copy()
    k2.prior = {**k2.prior, "T": (np.asarray(k2.prior["T"]) + np.array(
        [[0, 0, 0, 3.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])).tolist()}
    m.keyframes.append(k2)
    assert not m._links_allowed()


def test_scale_neighbours_by_path(tmp_path: Path, pans: dict[int, syn.Shot]) -> None:
    m = origin(tmp_path, pans)
    m.update([frame("p25", pans[25], body("p25", pans[25], "p0"))], lambda _: None)
    a, b = m.confirmed()
    assert m._path(a, b) is not None
    b.camera_id = "other"
    assert m._path(a, b) is None
    b.camera_id = a.camera_id
    nb = m._scale_neighbours(b, [a, b])
    assert nb and nb[0][4]
    a.prior = None  # no prior and a path: still a link pair
    assert m._scale_neighbours(b, [a, b])[0][0] is a
    m._bodies.pop("p25")
    assert m._scale_neighbours(b, [a, b]) == []  # neither a path nor a prior pair
    assert not m._linkable(b)


def test_more_linked_paths(tmp_path: Path, pans: dict[int, syn.Shot],
                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A linked view without a prior beside a standpoint grouping (never grouped), one with a
    prior and no sent height, one too close (kept, at the linked pose), and the report of its
    keyframe's measured scale naming its link pairs."""
    m = origin(tmp_path, pans)
    monkeypatch.setattr(AnchoredMap, "_image_pose", lambda *a, **k: None)
    grouped: list[str] = []
    m.standpoint = lambda f, feats, pl: grouped.append(f.name) or pl  # type: ignore[assignment]
    snap = m._snapshot()
    pl = m.update([frame("a", pans[25], body("a", pans[25], "p0"))], lambda _: None)[0]
    assert pl.method == "linked" and grouped == []
    m._restore(snap)
    m.keyframes = [k for k in m.keyframes if k.name == "p0"]
    f = frame("b", pans[25], body("b", pans[25], "p0"), prior(pans[25].T, "drifting", 0.2, 5.0))
    f.height_m = None
    pl = m.update([f], lambda _: None)[0]
    assert pl.status == "registered" and pl.method == "linked" and not pl.height_term
    m._restore(snap)
    m.keyframes = [k for k in m.keyframes if k.name == "p0"]
    real = AnchoredMap._prepare

    def close(self: AnchoredMap, f: FrameIn) -> Any:
        got = real(self, f)
        return an._Prep(got.feats, 0.1, True, got.descriptor)

    monkeypatch.setattr(AnchoredMap, "_prepare", close)
    monkeypatch.setattr(an, "TOO_CLOSE_INLIERS", 10 ** 6)
    pl = m.update([frame("c", pans[25], body("c", pans[25], "p0"), prior(pans[25].T,
                                                                       "drifting", 0.3, 8.0))],
                  lambda _: None)[0]
    assert pl.status == "kept_at_prior" and pl.reason == "too_close"
    assert pl.link is not None and pl.link["state"] == "used"
    assert pl.T is not None and err(pl.T, pans[25].T)[0] < 0.02


def test_link_pairs_need_the_angle_and_the_baseline(tmp_path: Path,
                                                    pans: dict[int, syn.Shot]) -> None:
    from oh_my_slam.service.prior_first import stereo_scale

    m = origin(tmp_path, pans)
    m.update([frame("p25", pans[25], body("p25", pans[25], "p0"))], lambda _: None)
    a, b = m.confirmed()
    row = stereo_scale(b)
    assert row is not None and row["link_neighbours"] == b.stereo_links
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(an, "KEYFRAME_PAIR_MAX_DEG", 10.0)  # the pan's 25 deg is too far
        assert m._scale_neighbours(b, [a, b]) == []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(motion, "KEYFRAME_BASELINE_REL_MAX", 0.01)  # its baseline too short
        assert m._scale_neighbours(b, [a, b]) == []
