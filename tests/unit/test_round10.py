"""Round 10: the supporting frames' timing (P19), versions and revert (O2), ignored pixels and
the label vocabulary (P18, oms/2.3), and object extents over many views."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import Drift
from tests.synth.room import base_pose
from tests.unit.test_anchored import frame, new_map

# -- P19: a frame joins the drift support at its version's commit -------------------------------


def _far(T: np.ndarray, dx: float) -> np.ndarray:
    W = T.copy()
    W[:3, 3] += [dx, 0.0, 0.0]
    return W


def test_a_confirmed_frame_supports_only_from_its_commit() -> None:
    """map.md §1 "Supporting frames" (P19): a drifting frame the job confirms is a supporting
    frame of the committed state, but the job's later predictions (a retry, a rerun, the close)
    still grow the sigma from the support as it was; the next job counts it."""
    d = Drift(np.eye(4), np.eye(4), fit_sigma=(0.01, 0.1), support=[[0.0, 0.0, 0.0]])
    P = _far(np.eye(4), 4.0)
    before = d.predict(P, 0.02, 1.0)
    d.confirm(P, P)
    assert P[:3, 3].tolist() in d.support and d.to_json()["support"] == d.support
    assert "fresh" not in d.to_json()
    assert d.predict(P, 0.02, 1.0)[1:] == before[1:]  # its own bound is not loosened
    d.fresh = []  # the next job (``begin_job``)
    assert d.predict(P, 0.02, 1.0)[1] < before[1]


def test_the_prior_first_job_never_reads_its_own_confirmations(tmp_path: Path) -> None:
    """P19 on the prior-first engine: after a held job's registration closed its drift
    (``_close_drift``), its confirmed frame is in the support but not in what the job's later
    passes predict with; ``begin_job`` of the next job lets it support."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0), p="drifting")])
    m.drift = Drift(m.keyframes[0].T, m.keyframes[0].T, fit_sigma=(0.01, 0.1),
                    support=[[50.0, 50.0, 50.0]])
    f = frame("d1", base_pose(10), p="drifting")
    pls = m.update([f], hold=True)
    assert pls[0].status == "registered"
    c = np.asarray(f.prior["T_map_cam"])[:3, 3].tolist()  # type: ignore[index]
    assert c in m.drift.support and c in m.drift.fresh
    in_job = m._predict(f)
    assert in_job is not None
    m.end_job(pls, False)
    m.begin_job()
    after = m._predict(f)
    assert after is not None and after[1] < in_job[1]
    assert an.MAX_EVIDENCE == 500
