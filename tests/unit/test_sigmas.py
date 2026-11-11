"""Calibrated sigmas and the combined conflict bound (``mapping.sigmas``, api.md §4, §6)."""

from __future__ import annotations

import math

import pytest

from oh_my_slam.mapping import sigmas


def test_calibration_adds_the_noise_floor_of_the_data_kind(monkeypatch: pytest.MonkeyPatch
                                                            ) -> None:
    assert sigmas.kind(True, "metric") == "depth/metric"
    assert sigmas.kind(False, None) == sigmas.kind(False, "none") == "nodepth/none"
    fp, fr = sigmas.NOISE_FLOORS["depth/metric"]
    assert sigmas.calibrated(0.0017, 0.24, True, "metric") == pytest.approx(
        (math.hypot(0.0017, fp), math.hypot(0.24, fr)))
    # fused with a default metric prior (2 deg) the floor gives the measured 1.06 deg
    assert 1.0 / math.sqrt(1 / fr ** 2 + 1 / 2.0 ** 2) == pytest.approx(1.06, abs=0.02)
    monkeypatch.setenv(sigmas.SIGMA_FLOORS_ENV, '{"depth/metric": [0, 0]}')
    monkeypatch.setenv(sigmas.SIGMA_SCALE_ENV, "2")
    assert sigmas.calibrated(0.01, 0.5, True, "metric") == pytest.approx((0.02, 1.0))
    assert sigmas.floor_of(False, "drifting") == sigmas.NOISE_FLOORS["nodepth/drifting"]
    for bad in ('{"other": [1, 1]}', '{"depth/none": [1]}', '{"depth/none": [-1, 0]}'):
        monkeypatch.setenv(sigmas.SIGMA_FLOORS_ENV, bad)
        with pytest.raises(ValueError, match="expected one of"):
            sigmas.floors()
    monkeypatch.delenv(sigmas.SIGMA_FLOORS_ENV)
    monkeypatch.setenv(sigmas.SIGMA_SCALE_ENV, "0")
    with pytest.raises(ValueError, match="positive"):
        sigmas.scale()
    monkeypatch.delenv(sigmas.SIGMA_SCALE_ENV)
    assert sigmas.scale() == sigmas.SIGMA_SCALE


def test_the_conflict_bound_combines_prior_and_hypothesis() -> None:
    prior, hyp = (0.002, 0.05), (0.008, 1.25)
    assert sigmas.bounds(prior, hyp) == pytest.approx((3 * math.hypot(0.002, 0.008),
                                                       3 * math.hypot(0.05, 1.25)))
    # a correct frame's 7 mm / 1 deg image noise is no conflict with its own tight prior
    assert not sigmas.conflicts(0.007, 1.0, prior, hyp)
    assert sigmas.conflicts(1.0, 0.0, prior, hyp)  # a prior 1 m off is
    assert sigmas.conflicts(0.0, 4.0, prior, hyp)
    assert not sigmas.conflicts(1.0, 0.0, prior, hyp, position=False)
    assert sigmas.distance(0.01, 0.0, (0.01, 1.0), (0.0, 1.0)) == pytest.approx(1.0)
    assert sigmas.distance(0.0, 0.0, (0.0, 0.0), (0.0, 0.0)) == 0.0
