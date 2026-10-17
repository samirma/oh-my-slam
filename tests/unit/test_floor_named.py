"""The floor named as an object (``segmentation.api.on_the_floor``): an open-vocabulary detector
names bare floor at the bottom of a low camera's views after furniture ("bed"). A detection of a
floor-standing class whose points lie within the floor's own thickness (``reconstruction.gravity.
floor_band``, measured from the floor's points) is the floor; floor coverings are kept.
Randomised over the camera's height, image size, focal length and the floor's noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.api import FrameReconstruction
from oh_my_slam.reconstruction.gravity import (
    FLOOR_BAND_SIGMAS,
    GravityEstimate,
    floor_band,
    floor_candidate_height,
)
from oh_my_slam.segmentation.api import (
    FLOOR_SHARE,
    Detection,
    LiftedInstance,
    frame_floor,
    on_the_floor,
    segment_frame,
)
from oh_my_slam.segmentation.lift import Lifted


@pytest.mark.parametrize("seed", range(10))
def test_the_floors_thickness_is_measured_from_its_points(seed: int) -> None:
    rng = np.random.default_rng(seed)
    level = rng.uniform(-2.0, 0.5)
    sd = rng.uniform(0.003, 0.04)
    floor = rng.normal(level, sd, 20000)
    # furniture tops, a wall and a few strays below: the floor is the lowest well-supported peak
    clutter = np.r_[rng.normal(level + 0.45, 0.01, 4000), rng.normal(level + 0.75, 0.01, 3000),
                    rng.uniform(level, level + 2.5, 15000), rng.uniform(level - 0.5, level, 200)]
    h = np.r_[floor, clutter]
    found = floor_candidate_height(h)
    assert found is not None
    at, band = floor_band(h, found)
    assert at == pytest.approx(level, abs=max(0.005, sd / 2))
    # the band is FLOOR_BAND_SIGMAS of the floor's spread (its half width at half maximum on
    # bins of a sixth of the finder's), within the bins' resolution
    assert band == pytest.approx(FLOOR_BAND_SIGMAS * sd, rel=0.3, abs=0.008)


def test_a_floor_without_points_near_the_finders_level_has_the_thinnest_band() -> None:
    at, band = floor_band(np.array([5.0, 6.0]), 0.0)
    assert abs(at) <= 0.03 and 0.0 < band < 0.01


def instance(label: str, pts: np.ndarray) -> LiftedInstance:
    det = Detection(label, 0.6, "yoloe", np.ones((2, 2), bool), (0.0, 0.0, 1.0, 1.0))
    return LiftedInstance(det, det.mask, Lifted(pts, np.arange(len(pts))))


@pytest.mark.parametrize("seed", range(6))
def test_a_floor_standing_class_lying_flat_on_the_floor_is_the_floor(seed: int) -> None:
    rng = np.random.default_rng(10 + seed)
    level, sd = rng.uniform(-1.5, 0.0), rng.uniform(0.005, 0.03)
    band = FLOOR_BAND_SIGMAS * sd
    up = np.array([0.0, 0.0, 1.0])
    flat = np.c_[rng.uniform(-1, 1, (3000, 2)), rng.normal(level, sd, 3000)]
    assert on_the_floor(instance("bed", flat), up, level, band)
    # a floor covering, or what may lie flat on the floor, is kept
    assert not on_the_floor(instance("rug", flat), up, level, band)
    assert not on_the_floor(instance("towel", flat), up, level, band)
    # a bed with a body above the floor, its mask bleeding onto the floor in front of it
    body = np.c_[rng.uniform(-1, 1, (3000, 2)), rng.uniform(level + 0.1, level + 0.6, 3000)]
    bed = np.r_[body, flat[:int(3000 * (FLOOR_SHARE - 0.1) / (1 - FLOOR_SHARE + 0.1))]]
    assert not on_the_floor(instance("bed", bed), up, level, band)
    assert not on_the_floor(instance("bed", np.zeros((0, 3))), up, level, band)


def level_camera(rng: np.random.Generator) -> tuple[Intrinsics, float]:
    w = int(rng.choice([320, 640, 768, 1024]))
    h = w * 3 // 4
    f = (w / 2) / np.tan(np.radians(rng.uniform(50.0, 90.0)) / 2)
    return Intrinsics(f, f, w / 2, h / 2, w, h), rng.uniform(0.3, 1.8)


@pytest.mark.parametrize("seed", range(6))
def test_a_single_image_drops_the_floor_named_as_a_bed(seed: int) -> None:
    """A level camera ``height`` m above the floor: the floor strip at the bottom of the image
    named a bed is no object; the bed standing on the floor ahead stays."""
    rng = np.random.default_rng(20 + seed)
    K, height = level_camera(rng)
    v, u = np.mgrid[0:K.height, 0:K.width]
    with np.errstate(divide="ignore"):
        depth = np.where(v + 0.5 > K.cy + 1, height * K.fy / (v + 0.5 - K.cy), 0.0)
    depth *= np.exp(rng.normal(0.0, 0.01, depth.shape))  # monocular noise
    dist, top = rng.uniform(2.0, 4.0), rng.uniform(0.4, 0.7)  # the bed's face, its height
    r0 = int(K.cy + K.fy * (height - top) / dist)
    r1 = int(K.cy + K.fy * height / dist)
    c0, c1 = int(K.width * 0.3), int(K.width * 0.7)
    depth[r0:r1, c0:c1] = dist
    depth[:int(K.cy) + 2] = 6.0  # a wall above the horizon
    valid = depth > 0
    bed = np.zeros(depth.shape, bool)
    bed[r0:r1, c0:c1] = True
    strip = np.zeros(depth.shape, bool)
    strip[int(K.height * 0.85):, :] = True
    gravity = GravityEstimate(np.array([0.0, -1.0, 0.0]), "geocalib+floor", floor_height=height)
    frame = FrameReconstruction(Path("x.jpg"), np.zeros((K.height, K.width, 3), np.uint8),
                                depth.astype(np.float32), valid, K, K, gravity=gravity)
    found = frame_floor(frame)
    assert found is not None
    _, level, band = found
    assert level == pytest.approx(-height, abs=0.05) and 0.0 < band < 0.1
    dets = [Detection("bed", 0.8, "yoloe", bed, (c0, r0, c1, r1)),
            Detection("bed", 0.6, "yoloe", strip, (0, int(K.height * 0.85), K.width, K.height))]
    seg = segment_frame(frame, detections=dets)
    assert [o.label for o in seg.objects] == ["bed"]
    # the bed's face, not the floor before it (camera frame: z is the distance)
    assert seg.objects[0].obb.center[2] == pytest.approx(dist, rel=0.05)
    # an image without a floor has nothing to measure it against
    assert frame_floor(FrameReconstruction(Path("x.jpg"), frame.rgb, frame.depth, valid, K, K)
                       ) is None
