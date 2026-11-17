"""oms/2.19: a depth pair fails only on a supported contradiction (map.md §4 "Depth
consistency"), and a rotation-only fit far from its metric prior on a thin depth confirmation is
too loose to confirm it (map.md §2 "Thin rotation-only fits")."""
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import standpoint as sp
from tests.unit.test_anchored import base_pose, frame, new_map


def _pair(agree: int, through: int) -> sp.DepthPair:
    ok = an.DepthCounts(3071, 500, 0, 0, 2571)
    return sp.DepthPair("a", "b", an.DepthCounts(3071, agree, through, 0, 3071 - agree - through),
                        ok, 0.1)


def test_a_contradiction_needs_support() -> None:
    """Below the agreement threshold, a direction fails when fewer than the clear share agree
    (a few points suffice) or with at least the minimum of see-through points; a few scattered
    see-through points between the two are the background of a correct pose."""
    assert sp.CONTRADICTION_CLEAR < an.DEPTH_CONFIRM_AGREEMENT
    # the correct views' background (README "Contradiction support"): 87 of 530, 22 of 177
    assert not _pair(443, 87).fails and not _pair(155, 22).fails
    assert _pair(443, 87).agrees  # a measured agreement for the suspect test
    # as many points as the minimum fail it; so does a clear contradiction on a few points
    assert _pair(800, sp.CONTRADICTION_MIN_POINTS).fails
    assert not _pair(800, sp.CONTRADICTION_MIN_POINTS - 1).fails
    assert _pair(30, 20).fails and _pair(0, 55).fails
    # the lattice turn's (README "Depth-consistency resolution"): hundreds of points
    assert _pair(2136, 534).fails and _pair(690, 496).fails
    # under the minimum count nothing is measured; at the threshold nothing fails
    assert not _pair(20, 19).fails and not _pair(900, 100).fails


def test_a_thin_rotation_only_fit_does_not_confirm_a_metric_prior(tmp_path: Path) -> None:
    """A rotation-only image pose more than ``ROTATION_PRIOR_K`` x the metric prior's rotation
    sigma from it, depth-confirmed on fewer than ``ROTATION_CONFIRM_MIN`` points one way, is
    too loose to confirm the frame (``_image_conflict`` None: ``loose_image_pose``)."""
    m = new_map(tmp_path)
    T0 = base_pose(0.0)
    f = frame("r", T0)  # metric prior, rotation sigma 1 deg
    assert f.prior is not None and f.prior["trust"] == "metric"
    R = an.exp_so3(np.array([0.0, np.radians(2.5), 0.0])) @ T0[:3, :3]
    s = an.Solve(R, T0[:3, 3].copy(), np.zeros((6, 6)), np.ones(20, bool), 0.5,
                 confirmed=an.ROTATION_CONFIRM_MIN - 1)
    img = (s, 20, 0.5, "rotation", 0.02, 0.5, None, None)
    assert m._image_conflict(f, img, [], T0, 0.02, 1.0) is None


def _conflict(tmp_path: Path, p: str, deg: float, confirmed: int | None,
              method: str = "rotation") -> Any:
    """``_image_conflict`` of a frame at ``base_pose(0)`` with a prior of trust ``p`` (rotation
    sigma 1 deg) for an image pose turned ``deg`` about its centre, confirmed on ``confirmed``
    agreeing points."""
    m = new_map(tmp_path / f"{p}-{deg}-{confirmed}-{method}")
    T0 = base_pose(0.0)
    f = frame("r", T0, p=p)
    R = an.exp_so3(np.array([0.0, np.radians(deg), 0.0])) @ T0[:3, :3]
    s = an.Solve(R, T0[:3, 3].copy(), np.eye(6) * 1e-6, np.ones(20, bool), 0.5,
                 confirmed=confirmed)
    sub = (None, None, None, np.zeros(0, np.intp))
    refs = SimpleNamespace(median=np.array([2.0]))
    img = (s, 20, 0.5, method, 0.02, 0.5, sub, refs)
    return m._image_conflict(f, img, [], T0, 0.02, 1.0)


def test_the_thin_rotation_only_rule_fires_only_where_it_applies(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The rule does not fire for a drifting prior, for a rotation within ``ROTATION_PRIOR_K``
    x the prior's rotation sigma, for a confirmation of ``ROTATION_CONFIRM_MIN`` agreeing points
    or more, for a pose without a depth confirmation, nor for a points fit: each is tested as
    before (a residual and the conflict bound), never loose by this rule."""
    thin = an.ROTATION_CONFIRM_MIN - 1
    caplog.set_level(logging.INFO, logger=an.log.name)
    for p, deg, confirmed, method in (
            ("drifting", 2.5, thin, "rotation"),
            ("metric", 1.5, thin, "rotation"),  # within 2 sigma of the prior
            ("metric", 2.5, an.ROTATION_CONFIRM_MIN, "rotation"),
            ("metric", 2.5, None, "rotation"),
            ("metric", 2.5, thin, "pnp")):
        got = _conflict(tmp_path, p, deg, confirmed, method)
        assert got is not None, (p, deg, confirmed, method)
        dp, dr, _, _ = got
        assert dr == pytest.approx(deg, abs=1e-6)
    assert not any("too loose to confirm it" in r.getMessage() for r in caplog.records)
    # and the case it applies to, through the same helper
    assert _conflict(tmp_path, "metric", 2.5, thin) is None
    assert any("too loose to confirm it" in r.getMessage() for r in caplog.records)
