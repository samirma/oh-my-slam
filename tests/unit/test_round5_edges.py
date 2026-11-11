"""Round-5 rules on their edges: the one overlap measure in the older mapper's prior check, its
rotation hard limit (map.md §1 "Every engine"), the MapAnything depth residual and the
re-registration overlap without features."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from oh_my_slam.mapping import anchored as an
from oh_my_slam.service.backend import InternalFrame
from oh_my_slam.service.update import Updater
from tests.synth.room import base_pose
from tests.unit.test_anchored import K, frame, new_map, shot


def _fr(name: str, T: np.ndarray, index: int, trust: str | None = None,
        prior: np.ndarray | None = None) -> Any:
    rgb, _ = shot(T)
    p = None if trust is None else {"trust": trust, "T_map_cam": (T if prior is None
                                                                  else prior).tolist()}
    return SimpleNamespace(work=SimpleNamespace(name=name, index=index, trust=trust, prior=p),
                           cam=SimpleNamespace(canonical=lambda: K), rgb=rgb, ignore=None,
                           pose=T, internal=object())


def test_the_mapper_prior_disagreement_uses_the_coverage() -> None:
    """Two overlapping registered frames whose priors disagree by 6 deg: reported; a pair of
    frames that share nothing is not."""
    A, B = base_pose(0), base_pose(10)
    turned = an.exp_so3(np.array([0.0, 0.0, np.radians(6.0)])) @ B[:3, :3]
    PB = B.copy()
    PB[:3, :3] = turned
    got = Updater._prior_disagreement([_fr("a", A, 0, "metric"), _fr("b", B, 1, "metric", PB),
                                       _fr("n", B, 2)], None, None)  # type: ignore[arg-type]
    assert abs(got - 6.0) < 1.5
    far = Updater._prior_disagreement([_fr("a", A, 0, "metric"),
                                       _fr("w", base_pose(180), 1, "metric")],
                                      None, None)  # type: ignore[arg-type]
    assert far == 0.0


def test_the_mapper_rotation_check(tmp_path: Path) -> None:
    """The older mapper's registered frames against its other fused frames (stored images);
    frames it cannot read are skipped; committed rotations 30 deg off fail the limit."""
    poses = [base_pose(y) for y in (0, 10, 20)]
    imgs = {}
    for i, T in enumerate(poses):
        p = tmp_path / f"k{i}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(shot(T)[0], cv2.COLOR_RGB2BGR))
        imgs[f"k{i}"] = p
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    imgs["bad"] = tmp_path / "bad.jpg"
    internal: Any = SimpleNamespace(
        frames=[InternalFrame(n, n, np.eye(4), K.fx) for n in (*imgs, "none")],
        image=lambda n: imgs.get(n))
    rows = [{"name": f"s{i}", "internal": f"k{i}", "pose_state": "fused",
             "T_map_cam": T.tolist()} for i, T in enumerate(poses)]
    rows += [{"name": "b", "internal": "bad", "pose_state": "fused", "T_map_cam": poses[0].tolist()},
             {"name": "n", "internal": "none", "pose_state": "fused",
              "T_map_cam": poses[0].tolist()},
             {"name": "kept", "internal": "k0", "pose_state": "kept_at_prior",
              "T_map_cam": poses[0].tolist()},
             {"name": "x", "internal": "unknown", "pose_state": "fused",
              "T_map_cam": poses[0].tolist()},
             {"name": "new", "internal": "k1", "pose_state": "fused",
              "T_map_cam": poses[1].tolist()}]
    T = base_pose(15)
    good = _fr("new", T, 0)
    assert not Updater._rotations_bad([good], {"frames": rows}, internal)
    off = _fr("new", T, 0)
    off.pose = T.copy()
    off.pose[:3, :3] = an.exp_so3(np.array([0.0, 0.0, np.radians(30.0)])) @ T[:3, :3]
    assert Updater._rotations_bad([off], {"frames": rows}, internal)


def test_depth_residual_needs_seen_points_and_overlaps_need_features(tmp_path: Path) -> None:
    m = new_map(tmp_path)
    m.update([frame("a0", base_pose(0))])
    before = list(m.scale.depth_residuals.get("sensor", []))
    f = frame("far", base_pose(180))
    assert f.depth is not None
    m.depth_residual(f, base_pose(180), f.depth, K)  # looking away: nothing seen
    assert m.scale.depth_residuals.get("sensor", []) == before
    g = frame("near", base_pose(5))
    assert g.depth is not None
    m.depth_residual(g, base_pose(5), g.depth, K)
    assert len(m.scale.depth_residuals["sensor"]) == len(before) + 1
    ghost = an.Keyframe("g", "g", "cam", K, base_pose(0), 0.01, 0.1, "provisional", None, "none")
    assert not m.overlaps(ghost, m.keyframes[0])  # its features are nowhere
