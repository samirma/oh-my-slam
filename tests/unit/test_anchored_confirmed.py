"""oms/2.2 rules of the prior-first engine (``mapping.anchored``): only image-confirmed frames
are references, cloud and scale evidence; stored kept frames re-registered by a later update;
pose duplicates; the camera-height floor of a camera-only map; the direction spread of the
camera-height scale."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.anchored import AnchoredMap, ScaleState
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, err, frame, new_map, prior


def test_duplicate_limits_default_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(an.DUPLICATE_POS_ENV, raising=False)
    monkeypatch.delenv(an.DUPLICATE_ROT_ENV, raising=False)
    assert an.duplicate_limits() == (0.02, 2.0)
    monkeypatch.setenv(an.DUPLICATE_POS_ENV, "0.049")
    monkeypatch.setenv(an.DUPLICATE_ROT_ENV, "4.5")
    assert an.duplicate_limits() == (0.049, 4.5)
    monkeypatch.setenv(an.DUPLICATE_POS_ENV, "0.05")  # never reaching the cap
    with pytest.raises(ValueError, match=an.DUPLICATE_POS_ENV):
        an.duplicate_limits()
    monkeypatch.setenv(an.DUPLICATE_POS_ENV, "0.02")
    monkeypatch.setenv(an.DUPLICATE_ROT_ENV, "-1")
    with pytest.raises(ValueError, match=an.DUPLICATE_ROT_ENV):
        an.duplicate_limits()


def test_kept_frames_are_stored_but_never_fused_or_references(tmp_path: Path) -> None:
    """A kept_at_prior frame (map.md §2) adds nothing to the cloud and is no reference for a
    later frame of the update; a frame of another camera at a confirmed pose is no duplicate."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    n = len(m.cloud.keys)
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    pl = m.update([frame("blank", base_pose(10), rgb=blank, depth=False, mono=1.0)])[0]
    assert pl.status == "kept_at_prior" and len(m.cloud.keys) == n
    kept = m.keyframes[-1]
    assert not kept.usable() and m.confirmed() == m.keyframes[:1]
    assert m.prior_span() == 0.0
    assert m._candidates(frame("x", base_pose(10)), base_pose(10), None) == m.keyframes[:1]
    f = frame("other", base_pose(0))
    f.camera_id = "cam2"
    assert m.update([f])[0].status == "registered"
    assert m.duplicate_of("cam", base_pose(10)) is None  # the kept frame does not count


def test_an_origin_failing_a_check_is_kept_then_the_next_prior_registers(tmp_path: Path
                                                                          ) -> None:
    m = new_map(tmp_path)
    bad = frame("bad", base_pose(0))
    bad.unconfirmed = "plain"
    pls = m.update([bad, frame("a1", base_pose(30))])
    assert (pls[0].status, pls[0].reason) == ("kept_at_prior", "plain")
    assert np.allclose(pls[0].T, base_pose(0))  # in the priors' frame all the same
    assert (pls[1].status, pls[1].method) == ("registered", "origin")
    assert m.created_with == "metric" and np.allclose(m.origin_c, base_pose(0)[:3, 3])


def test_a_stored_kept_frame_is_re_registered_by_a_later_update(tmp_path: Path) -> None:
    """map.md §2: a later job registering frames that overlap a stored kept frame places it
    again; confirmed, it is fused and reported with its displacement."""
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    T = base_pose(20)
    off = T.copy()
    off[:3, 3] += [0.06, 0.0, 0.0]
    k = frame("k", T, p=prior(off, sp=0.2, sr=2.0), depth=False, mono=1.0)
    k.unconfirmed = "blur"  # its image failed a check: kept at its prior, 6 cm off
    pl = m.update([k])[0]
    assert pl.status == "kept_at_prior"
    m.save()
    m = AnchoredMap.load(m.root)
    n = len(m.cloud.keys)
    pls = m.update([frame("b", base_pose(25))])
    assert pls[0].status == "registered"
    assert [e["name"] for e in m.reregistered] == ["k"]
    e = m.reregistered[0]
    kf = m.keyframe(e["keyframe"])
    assert kf is not None and kf.usable() and err(kf.T, T)[0] < 0.03
    assert e["displacement_m"] == pytest.approx(err(off, kf.T)[0], abs=1e-3)
    assert len(m.cloud.keys) > n and kf.rgb is not None
    # nothing kept left: the next update re-registers nothing
    m.update([frame("c", base_pose(35))])
    assert m.reregistered == []
    assert m.keyframe("nope") is None


def test_re_registration_edges(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    m.update([frame("blank", base_pose(10), rgb=blank, depth=False, mono=1.0)])
    kept = m.keyframes[-1]
    m.save()
    new = m.keyframes[0]
    # still unconfirmed (no features): stays kept
    assert m.reregister([kept], [new]) == [] and not kept.usable()
    # no overlap with any new frame: not tried
    far = an.Keyframe("z", "z", "cam", K, base_pose(180), 0.01, 0.1, "fused", None, "none")
    assert not m.overlaps(kept, far)
    assert m.reregister([kept], [far]) == []
    # without its image or a prior: nothing to place
    (m.root / an.KF_DIR / f"{kept.id}.jpg").unlink()
    assert m._place_again(kept) is None
    kept.prior = None
    assert m.reregister([kept], [new]) == []
    # a confirmed keyframe is not re-registered
    assert m.reregister([new], [new]) == []


def test_place_again_unreadable_image(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    blank = np.full((K.height, K.width, 3), 120, np.uint8)
    m.update([frame("blank", base_pose(10), rgb=blank, depth=False, mono=1.0)])
    m.save()
    kept = m.keyframes[-1]
    (m.root / an.KF_DIR / f"{kept.id}.jpg").write_bytes(b"not an image")
    assert m._place_again(kept) is None


def test_camera_only_map_with_a_camera_height_stands_on_its_floor(tmp_path: Path) -> None:
    """map.md §1, §6: without priors but with a camera height, the origin is the floor below
    the first registered frame, which stands at that height; floor_z_m is about 0."""
    m = new_map(tmp_path)
    T0 = base_pose(0, pitch_deg=-20.0)
    m.update([frame("a", T0, p=None, mono=1.0, height=0.5)])
    assert m.origin_height_m == 0.5
    c = m.keyframes[0].centre
    assert abs(c[0]) < 1e-9 and abs(c[1]) < 1e-9 and c[2] == pytest.approx(0.5, abs=0.02)
    floor, sigma = m.floor()
    assert floor is not None and abs(floor) < 0.03 and sigma > 0
    m.save()
    again = AnchoredMap.load(m.root)
    assert again.origin_height_m == 0.5
    # without points the floor is the height's: z 0
    m.cloud = an.Cloud()
    assert m.floor()[0] == 0.0
    # a plane outside the band is not taken: z 0 stays
    g = np.mgrid[-2.0:2.0:0.02, -2.0:2.0:0.02].reshape(2, -1).T
    m.cloud.add(np.column_stack([g, np.full(len(g), -0.4)]), np.zeros((len(g), 3)))
    assert m.floor()[0] == 0.0


def test_cloud_rotated_with_a_translation() -> None:
    c = an.Cloud()
    c.add(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)))
    R = an.exp_so3(np.array([0.0, 0.0, math.pi / 2]))
    got = c.rotated(R, np.array([0.0, 0.0, 1.0])).points()
    assert np.allclose(got, [[0.0, 1.0, 1.0]], atol=1e-6)


def test_camera_height_scale_counts_the_spread_between_directions() -> None:
    s = ScaleState()
    s.heights = [[0.01, 1.0], [0.01, 1.0], [0.01, 1.1], [0.01, 1.1]]
    s.height_yaws = [0.0, 10.0, 90.0, 100.0]
    spread = s.direction_spread()
    assert spread == pytest.approx(abs(math.log(1.1)) / 2, rel=0.05)
    s.estimate(0.0)
    assert s.source == "camera_height" and s.sigma_rel >= spread
    s.height_yaws = [0.0, 1.0, 2.0, 3.0]  # one direction: nothing to compare
    assert s.direction_spread() == 0.0
    s.height_yaws = []  # recorded before oms/2.2
    assert s.direction_spread() == 0.0
    back = ScaleState.from_json({**s.to_json(), "height_yaws": [5.0]})
    assert back.height_yaws == [5.0]
    old = s.to_json()
    del old["height_yaws"]
    assert ScaleState.from_json(old).height_yaws == []


def test_small_edges_of_the_confirmed_only_rules(tmp_path: Path) -> None:
    # an agreement without points or references
    assert an.agreement(np.zeros((0, 3)), np.eye(4), [], 0.1).points == 0
    # a cloud turned without a move
    c = an.Cloud()
    c.add(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)))
    assert np.allclose(c.rotated(np.eye(3)).points(), [[1.0, 0.0, 0.0]])
    # the same keyframe twice among the candidates: its matches are taken once
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    f = frame("b", base_pose(10))
    feats = an.features(f.rgb)
    once = m._correspondences(feats, m.keyframes[:1])
    twice = m._correspondences(feats, m.keyframes[:1] * 2)
    assert len(twice.uv) == len(once.uv) > 0
    # height evidence stored before oms/2.2 (no yaws): no yaw recorded from then on
    m.scale.heights, m.scale.height_yaws = [[0.04, 1.0]], []
    T = base_pose(0)
    h = frame("h", T, height=0.5)
    assert h.depth is not None
    m.add_height_evidence(h, T, h.depth, K)
    assert len(m.scale.heights) == 2 and m.scale.height_yaws == []


def test_a_too_close_camera_only_origin_is_rejected(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    near = frame("near", base_pose(0), p=None, depth=False, mono=0.05)
    near.check_close = True
    p = m.update([near])[0]
    assert (p.status, p.reason) == ("rejected", "too_close") and not m.keyframes


def test_a_re_registered_keyframe_without_depth_is_not_fused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    k = frame("k", base_pose(20), depth=False)  # no depth at all
    k.unconfirmed = "blur"
    m.update([k])
    kept = m.keyframes[-1]
    assert kept.depth_kind == "none"
    m.save()
    n = len(m.cloud.keys)
    pl = an.Placement("k", "registered", None, base_pose(20), 0.01, 0.1, 50, "pnp")
    monkeypatch.setattr(m, "_place_again",
                        lambda kf: (pl, np.zeros((K.height, K.width, 3), np.uint8)))
    got = m.reregister([kept], [m.keyframes[0]])
    assert [e["name"] for e in got] == ["k"] and kept.usable() and len(m.cloud.keys) == n
