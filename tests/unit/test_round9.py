"""Round 9: the drift closed over the cross-engine phase (R-16), the supporting frames (SI-19,
P17) and locate's height_sigma over registered references (SI-20, P17)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import relocalize
from oh_my_slam.mapping.anchor import Sim3
from oh_my_slam.mapping.anchored import AnchoredMap, Drift
from oh_my_slam.service import prior_first
from oh_my_slam.service.update import Updater, _Anchored
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, frame, new_map, prior
from tests.unit.test_round7 import _drifting
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import CAM, frames_for
from tests.unit.test_service_prior_first import client as _client  # noqa: F401


def _moved(T: np.ndarray, dx: float) -> np.ndarray:
    W = T.copy()
    W[:3, 3] += [dx, 0.0, 0.0]
    return W


# -- R-16: a drifting frame only the cross-engine retries confirmed -----------------------------


def test_a_drifting_frame_only_a_retry_confirmed_closes_the_drift(tmp_path: Path) -> None:
    """R-16 (map.md §1 "Supporting frames", §2): a held job's drifting frame the prior-first
    engine kept for want of an image pose, confirmed by ``retry_kept``, becomes a supporting
    frame and the drift reference at the job's close (``close_job``), like a frame the job's
    own registration confirmed."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), p="drifting")])
    p = frame("p", base_pose(95), p="drifting")
    pls = m.update([p], hold=True)
    assert (pls[0].status, pls[0].detail) == ("kept_at_prior", "no_image_pose")
    kp = next(k for k in m.keyframes if k.name == "p")
    n = frame("n", base_pose(90))
    nk = m.add(n, an.features(n.rgb),
               an.Placement("n", "registered", None, base_pose(90), 0.01, 0.1, 500, "pnp"),
               n.depth, K, "sensor", None)  # type: ignore[arg-type]
    m.candidates_for = lambda f, *a, **k: [nk]  # type: ignore[method-assign]
    got = m.retry_kept([(p, kp)], [nk])
    assert got["p"].status == "registered" and got["p"].T is not None
    centre = np.asarray(p.prior["T_map_cam"])[:3, 3].tolist()  # type: ignore[index]
    assert centre not in m.drift.support  # the job's own close did not see it
    m.close_job([p], [got["p"]])
    assert centre in m.drift.support
    assert m.drift.ref_T is not None and np.allclose(m.drift.ref_T, got["p"].T)
    m.end_job([got["p"]], False)
    assert m._close_base is None


def test_the_job_s_close_is_canonical_and_confirms_each_frame_once(tmp_path: Path) -> None:
    """R-16: ``close_job`` closes from the estimate the job's registration closed from, every
    registered drifting frame once, in canonical order: the reference, the supporting frames
    and the odometry-scale evidence do not depend on the request order, and a frame the job's
    registration confirmed is not confirmed twice."""
    base = Drift(base_pose(0), base_pose(0))
    Ts = {"d1": _moved(base_pose(0), 1.0), "d2": _moved(base_pose(0), 2.1),
          "d10": _moved(base_pose(0), 3.0)}
    fs = {n: frame(n, base_pose(0), p=prior(_moved(T, 0.0), "drifting")) for n, T in Ts.items()}
    fs["m"] = frame("m", base_pose(0))  # a metric prior: not the drift's
    fs["k"] = frame("k", base_pose(0), p=prior(_moved(base_pose(0), 5.0), "drifting"))
    pl = {n: an.Placement(n, "registered", None, T, 0.01, 0.1, 500, "pnp")
          for n, T in Ts.items()}
    pl["m"] = an.Placement("m", "registered", None, base_pose(0), 0.01, 0.1, 500, "pnp")
    pl["k"] = an.Placement("k", "kept_at_prior", "no_overlap", base_pose(0), 0.02, 1.0, 0)
    states = []
    for order in (["d10", "m", "d2", "k", "d1"], ["d1", "k", "d2", "m", "d10"]):
        m = new_map(tmp_path / order[0])
        m.close_job([fs[n] for n in order], [pl[n] for n in order])  # no held job: nothing
        assert m.drift.ref_T is None
        m._close_base = copy.deepcopy(base)
        # the job's registration confirmed d1 already: closing again starts from the base
        m.drift = copy.deepcopy(base)
        m.drift.confirm(Ts["d1"], np.asarray(fs["d1"].prior["T_map_cam"]))  # type: ignore[index]
        m.close_job([fs[n] for n in order], [pl[n] for n in order])
        states.append(m.drift.to_json())
    assert states[0] == states[1]
    d = states[0]
    assert np.allclose(d["ref_T"], Ts["d10"])  # natural order: d1, d2, d10
    assert len(d["evidence"]) == 3 and len(d["support"]) == 3
    assert [c[0] for c in d["support"]] == pytest.approx(
        [float(Ts[n][0, 3]) for n in ("d1", "d2", "d10")])


def test_a_mapanything_job_closes_its_drift_after_the_retries(
        _client: Client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """R-16 through ``run_anchored``: the held job's drift is closed after the cross-engine
    phase, before ``end_job``, over the prior-first engine's frames."""
    monkeypatch.setenv(prior_first.NONE_ENGINE_ENV, "mapanything")
    calls: list[tuple[list[str], list[str]]] = []
    real = AnchoredMap.close_job

    def spy(self: AnchoredMap, frames: list[an.FrameIn], pls: list[an.Placement]) -> None:
        calls.append(([f.name for f in frames], [p.name for p in pls]))
        assert self._job_used is not None or not frames  # before ``end_job``
        real(self, frames, pls)

    monkeypatch.setattr(AnchoredMap, "close_job", spy)
    fr, files = frames_for(_client, [base_pose(0, 0.0)], tag="a", trust="drifting")
    for f, T in zip(fr, [base_pose(0, 0.0)], strict=True):
        _client.backend.mv_add(files[f["image"]], T)
    assert _client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    assert calls == [(["a0"], ["a0"])]


# -- SI-19: the supporting frames ---------------------------------------------------------------


def test_the_supporting_frames_have_no_count_bound_and_each_frame_once() -> None:
    """map.md §1 "Supporting frames" (P17): no count bound; a frame confirmed again is one
    supporting frame."""
    d = Drift()
    for k in range(an.MAX_EVIDENCE + 10):
        d.confirm(_moved(np.eye(4), 0.01 * k), _moved(np.eye(4), 0.01 * k))
    assert len(d.support) == an.MAX_EVIDENCE + 10
    d.confirm(np.eye(4), np.eye(4))
    assert len(d.support) == an.MAX_EVIDENCE + 10


def test_a_changed_estimate_resets_the_supporting_frames(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """P17 on the prior-first engine: when a fit changes the estimate, its supporting frames
    are the new fit's inliers (and the frames confirmed under it), none of the old ones."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), p="drifting")])
    old = [[99.0, 99.0, 99.0]]
    m.drift.support = list(old)
    fitted = Drift(base_pose(0), base_pose(0), support=[[1.0, 2.0, 3.0]])
    calls: list[int] = []

    def fit(self: AnchoredMap, *a: Any) -> tuple[Drift | None, dict[int, an.Placement]]:
        calls.append(1)
        return (copy.deepcopy(fitted) if len(calls) == 1 else None), {}

    monkeypatch.setattr(AnchoredMap, "_fit_drift", fit)
    m.update([frame("d1", base_pose(20), p="drifting")])
    assert old[0] not in m.drift.support and [1.0, 2.0, 3.0] in m.drift.support


def _older(name: str, T: np.ndarray, index: int, trust: str = "drifting",
           internal: bool = True) -> Any:
    fr = _drifting(name, T, T, index, 500)
    fr.work.trust = fr.work.prior["trust"] = trust
    fr.work.camera_id = "cam"
    if not internal:
        fr.internal = None
    return fr


def test_the_older_mapper_s_supporting_frames() -> None:
    """P17 on the older mapper: a changed estimate's supporting frames are its fit's inliers
    only; under the estimate a job kept, its registered drifting frames join (canonical order,
    each once, no count bound), never a kept, a metric or a prior-less one."""
    shift = np.eye(4)
    shift[0, 3] = 0.5
    poses = [base_pose(20 * k) for k in range(3)]
    frames = [_drifting(f"d{k}", T, shift @ T, k, 500) for k, T in enumerate(poses)]
    prev = {"drift_sim": Sim3().to_dict(), "drift_sigma": [0.01, 0.2],
            "drift_support": [[99.0, 99.0, 99.0]]}
    sim, _, support, kept = Updater._drift_estimate(frames, Sim3(), prev)  # type: ignore[arg-type]
    assert sim is not None and not kept and support is not None
    assert [99.0, 99.0, 99.0] not in support and len(support) == 3
    # refused: the previous estimate stays
    weak = [_drifting(f"w{k}", T, shift @ T, k, 10) for k, T in enumerate(poses)]
    got = Updater._drift_estimate(weak, Sim3(), prev)  # type: ignore[arg-type]
    assert got[2] == prev["drift_support"] and got[3]
    assert Updater._drift_estimate([], Sim3(), prev)[3]  # type: ignore[arg-type]
    assert not Updater._drift_estimate([], Sim3(), None)[3]  # type: ignore[arg-type]
    an_ = _Anchored(Sim3(), "priors", 0.01, 0.0, drift_sim=Sim3(),
                    drift_support=[[99.0, 99.0, 99.0]], drift_kept=True)
    regs = [_older("r10", _moved(np.eye(4), 3.0), 0), _older("r2", _moved(np.eye(4), 2.0), 1),
            _older("r2b", _moved(np.eye(4), 2.0), 2),
            _older("k", _moved(np.eye(4), 5.0), 3, internal=False),
            _older("m", _moved(np.eye(4), 6.0), 4, trust="metric")]
    none = _older("n", _moved(np.eye(4), 7.0), 5)
    none.work.prior = None
    Updater._confirmed_support(an_, [*regs, none])  # type: ignore[arg-type]
    assert an_.drift_support == [[99.0, 99.0, 99.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    changed = _Anchored(Sim3(), "priors", 0.01, 0.0, drift_sim=Sim3(),
                        drift_support=[[1.0, 0.0, 0.0]])
    Updater._confirmed_support(changed, regs)  # type: ignore[arg-type]
    assert changed.drift_support == [[1.0, 0.0, 0.0]]
    empty = _Anchored(Sim3(), "priors", 0.01, 0.0, drift_kept=True)
    Updater._confirmed_support(empty, regs[:1])  # type: ignore[arg-type]
    assert empty.drift_support == [[3.0, 0.0, 0.0]]


# -- SI-20: locate's height_sigma over registered references ------------------------------------


def _kf(name: str, sp: float, registered: bool, hs: float | None = None) -> relocalize.Keyframe:
    return relocalize.Keyframe(name, np.eye(4), K.K(),
                               np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8),
                               np.zeros((0, 3), np.float32), sp, 0.1, False, hs, registered)


def test_height_sigma_step_3_counts_registered_references_only() -> None:
    """api.md §6 height_sigma step 3 (P17, SI-20): the median sigma_pos_m / √3 of the
    registered references, kept ones skipped; none left: the default (step 4)."""
    q = SimpleNamespace(camera_z_sigma_given=False, camera_z_sigma=relocalize.DEFAULT_HEIGHT_SIGMA_M)
    idx = relocalize.Index([_kf("r", 0.03, True), _kf("k", 0.9, False), _kf("r2", 0.05, True)])
    got = relocalize._ref_z_sigma(q, np.array([0, 1, 2]), idx)  # type: ignore[arg-type]
    assert got == pytest.approx(0.04 / np.sqrt(3.0))
    only_kept = relocalize._ref_z_sigma(q, np.array([1]), idx)  # type: ignore[arg-type]
    assert only_kept == relocalize.DEFAULT_HEIGHT_SIGMA_M
    # step 2 is unchanged: the sigma every reference carried when sent
    idx2 = relocalize.Index([_kf("k", 0.9, False, 0.07)])
    assert relocalize._ref_z_sigma(q, np.array([0]), idx2) == 0.07  # type: ignore[arg-type]


def test_the_locate_index_carries_each_reference_s_status(tmp_path: Path) -> None:
    """SI-20: ``derived._build_index`` sets ``Keyframe.registered`` from the row's
    ``pose_state``."""
    from oh_my_slam.service import derived

    names = ["f", "k"]
    z_f = {"uv": np.zeros((0, 2), np.float32), "desc": np.zeros((0, 32), np.uint8),
           "size": np.array([64, 48])}
    feats = tmp_path / "m" / "features"
    feats.mkdir(parents=True)
    for n in names:
        np.savez(feats / f"{n}.npz", **z_f)
    vdir = tmp_path / "v"
    vdir.mkdir()
    np.savez(vdir / "locate.npz", p0=np.zeros((0, 3), np.float32),
             p1=np.zeros((0, 3), np.float32))
    rows = [{"name": "f", "camera_id": "cam", "T_map_cam": np.eye(4).tolist(),
             "sigma_pos_m": 0.01, "pose_state": "fused"},
            {"name": "k", "camera_id": "cam", "T_map_cam": np.eye(4).tolist(),
             "sigma_pos_m": 0.02, "pose_state": "kept_at_prior"}]
    dv = SimpleNamespace(name="m", dir=vdir, record={"frames": rows},
                         meta={"keyframes": names, "keys": {n: n for n in names}})
    self = SimpleNamespace(store=None, dir=lambda name: tmp_path / name)
    cams = {"cam": CAM}
    idx = derived.Derived._build_index(self, dv, cams)  # type: ignore[arg-type]
    assert [k.registered for k in idx.keyframes] == [True, False]
