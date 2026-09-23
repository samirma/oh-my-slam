"""Offline parts of the developer tools (metrics, isotonic calibration)."""

from __future__ import annotations

import numpy as np

from oh_my_slam.tools.calibrate_scores import isotonic, precision_at
from oh_my_slam.tools.evaluate import ate, depth_metrics


def test_isotonic_is_monotone_and_fits() -> None:
    rng = np.random.default_rng(0)
    x = rng.random(2000)
    y = (rng.random(2000) < x**2).astype(float)
    kx, ky = isotonic(x, y)
    assert np.all(np.diff(ky) >= -1e-12) and kx[0] == 0.0 and kx[-1] == 1.0
    assert abs(np.interp(0.9, kx, ky) - 0.81) < 0.1
    assert precision_at(np.array([0.2, 0.6, 0.7]), np.array([0, 1, 0]), 0.5) == 0.5
    assert np.isnan(precision_at(np.array([0.1]), np.array([1]), 0.5))


def test_depth_and_trajectory_metrics() -> None:
    gt = np.full((10, 10), 2.0)
    m = depth_metrics(gt * 1.1, gt)
    assert abs(m["absrel"] - 0.1) < 1e-9 and m["delta1"] == 1.0
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(50, 3))
    est = 0.5 * ref + 1.0
    e_sim, s = ate(est, ref, True)
    assert e_sim < 1e-9 and abs(s - 2.0) < 1e-9
    e_se3, _ = ate(est, ref, False)
    assert e_se3 > 0.1
