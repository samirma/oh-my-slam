"""Geometry-first candidates (``segmentation.regions``), the closer looks at them
(``detect.detect_regions``), their labelling (``segmentation.api.label_regions``) and the size
priors of labels (``detect.size_plausible``, ``MapObject.plausible_votes``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping.anchored_objects import Box, MapObject, Sighting
from oh_my_slam.segmentation import api, detect, regions
from oh_my_slam.segmentation.api import Detection

K = Intrinsics(300.0, 300.0, 160.0, 120.0, 320, 240)
DOWN = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0.35], [0, 0, 0, 1]])


def worktop() -> np.ndarray:
    """A worktop 0.35 m below a camera looking straight down: a plate's 2 cm rim (its inside at
    the worktop's level), a 6 cm item, a gripper under the camera with a farther part joined to
    it, and a 1 mm sheet (flat: no candidate)."""
    d = np.full((240, 320), 0.35, np.float32)
    yy, xx = np.mgrid[:240, :320]
    r = np.hypot(xx - 80, yy - 80)
    d[(r >= 40) & (r <= 50)] = 0.33
    d[150:190, 200:240] = 0.29
    d[200:240, 140:180] = 0.10  # the gripper
    d[215:240, 180:200] = 0.30  # its farther part
    d[20:60, 250:300] = 0.349
    d[0:2, :] = 0.0
    return d


def test_raised_regions_of_a_view_from_above() -> None:
    found = regions.raised_regions(worktop(), K, DOWN, 0.01)
    assert len(found) == 2
    plate = next(r for r in found if r.box[0] < 100)
    assert plate.mask[80, 80]  # the rim's inside is filled
    assert abs(plate.height_m - 0.02) < 1e-3
    assert abs(next(r for r in found if r is not plate).height_m - 0.06) < 1e-3


def test_no_regions_without_a_view_down_or_a_support() -> None:
    side = DOWN.copy()
    side[:3, :3] = [[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]
    assert regions.raised_regions(worktop(), K, side, 0.01) == []
    assert regions.raised_regions(np.zeros((240, 320), np.float32), K, DOWN, 0.01) == []
    rough = np.random.default_rng(0).uniform(0.2, 0.34, (240, 320)).astype(np.float32)
    assert regions.raised_regions(rough, K, DOWN, 0.01) == []
    assert regions.support_level(np.zeros(0)) is None
    # a candidate wider than an object is none; monocular noise hides a 2 cm rim
    big = np.full((240, 320), 0.35, np.float32)
    big[60:180, 20:300] = 0.30
    big[10:20, 10:20] = 0.30  # too small to be an object
    wide = Intrinsics(100.0, 100.0, 160.0, 120.0, 320, 240)
    assert regions.raised_regions(big, wide, DOWN, 0.01) == []
    valid = np.ones((240, 320), bool)
    assert len(regions.raised_regions(worktop(), K, DOWN, 0.05, valid)) == 1


class CropDetector:
    """Reports one detection over the middle of whatever crop it is shown, and records the
    crops' sizes."""

    def __init__(self, label: str, score: float) -> None:
        self.label, self.score, self.sizes = label, score, []

    def segment_image(self, req):  # type: ignore[no-untyped-def]
        from oh_my_slam.client import protocol as p
        from oh_my_slam.core import rle

        w, h = Image.open(req.image_path).size
        self.sizes.append((w, h))
        m = np.zeros((h, w), bool)
        m[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        inst = [p.Instance(label=self.label, score=self.score, source="yoloe",
                           box_xyxy=[w / 4, h / 4, 3 * w / 4, 3 * h / 4], mask=rle.encode(m)),
                p.Instance(label="wall", score=0.9, source="yoloe", box_xyxy=[0, 0, w, h],
                           mask=rle.encode(np.ones((h, w), bool))),
                p.Instance(label="pen", score=0.9, source="yoloe", box_xyxy=[0, 0, 2, 2],
                           mask=rle.encode(np.pad(np.ones((2, 2), bool),
                                                  ((0, h - 2), (0, w - 2)))))]
        return p.SegmentResponse(width=w, height=h, instances=inst)


def test_closer_looks_are_labelled_on_the_raised_area(tmp_path: Path) -> None:
    img = tmp_path / "top.png"
    Image.new("RGB", (640, 480), (200, 200, 200)).save(img)
    found = regions.raised_regions(worktop(), K, DOWN, 0.01)
    client = CropDetector("plate", 0.3)
    boxes = [r.box for r in found]
    looks = detect.detect_regions(img, boxes, (320, 240), client, 0.5)  # type: ignore[arg-type]
    assert [[d.label for d in ds] for ds in looks] == [[], []]  # under min_score
    looks = detect.detect_regions(img, boxes, (320, 240), client, 0.2)  # type: ignore[arg-type]
    assert [[d.label for d in ds] for ds in looks] == [["plate"], ["plate"]]
    assert min(min(s) for s in client.sizes) >= detect.CROP_MIN_PX
    d = looks[0][0]
    assert d.mask.shape == (240, 320) and d.mask[80, 80]
    assert 60 <= (d.box[0] + d.box[2]) / 2 <= 100
    assert detect.detect_regions(img, [], (320, 240), client, 0.2) == []  # type: ignore[arg-type]
    # kept on the raised area, unless the frame already has it; covered regions are skipped
    labelled = api.label_regions(found, looks, [])
    assert sorted(x.label for x in labelled) == ["plate", "plate"]
    assert len(api.label_regions(found, looks, [looks[0][0]])) == 1
    off = Detection("plate", 0.3, "yoloe", np.roll(d.mask, 120, axis=1), d.box)
    assert api.label_regions(found[:1], [[off]], []) == []
    assert api.uncovered_regions(found, []) == found
    assert api.uncovered_regions(found, [Detection("plate", 0.9, "yoloe", found[0].mask,
                                                   (0, 0, 1, 1))]) == found[1:]


class ScoreDetector(CropDetector):
    """A crop detector whose class-wise pass (``per_label``) also scores the middle square as
    a bowl and a wall."""

    def segment_image(self, req):  # type: ignore[no-untyped-def]
        res = super().segment_image(req)
        if req.per_label:
            mid = res.instances[0]
            res.instances += [mid.model_copy(update={"label": "bowl", "score": 0.12}),
                              mid.model_copy(update={"label": "bowl", "score": 0.08}),
                              mid.model_copy(update={"label": "wall", "score": 0.5})]
        return res


def test_look_closer_adds_the_regions_and_the_label_scores(tmp_path: Path) -> None:
    img = tmp_path / "top.png"
    Image.new("RGB", (320, 240), (200, 200, 200)).save(img)
    found = regions.raised_regions(worktop(), K, DOWN, 0.01)
    client = ScoreDetector("mirror", 0.4)
    m = np.zeros((240, 320), bool)
    m[60:180, 80:240] = True  # the full image's middle square, as the detector reports it
    first = Detection("mirror", 0.6, "yoloe", m, (80, 60, 240, 180))
    out = api.look_closer(img, found, [first], (320, 240), client,  # type: ignore[arg-type]
                          0.2)
    assert out[0].alternatives == {"bowl": 0.12}  # its own label and the background are not
    # the plate's raised area looked at closer (the detection covers the other item's)
    assert len(out) == 2 and out[1].mask[80, 80]
    covered = Detection("mirror", 0.6, "yoloe", np.ones((240, 320), bool), (0, 0, 320, 240))
    assert len(api.look_closer(img, found, [covered], (320, 240), client)) == 1  # type: ignore[arg-type]
    assert detect.label_scores(img, [], (320, 240), client) == []  # type: ignore[arg-type]


def test_label_fit() -> None:
    fit = detect.label_fit
    assert fit("mug", (0.09, 0.08, 0.1), True) == 1.0
    assert fit("satellite dish", (0.1, 0.09, 0.1), False) < 0.5
    assert fit("satellite dish", (0.5, 0.4, 0.1), True) == 1.0
    assert fit("plate", (0.06, 0.06, 0.02), True) < 0.5
    assert fit("plate", (0.08, 0.06, 0.02), False) == 1.0  # a partial view may be smaller
    assert fit("wire", (0.01, 0.01, 0.01), True) == 1.0  # no typical shape: any
    assert fit("mirror", (0.3, 0.3, 0.1), True) < 0.5  # a mirror is thin
    assert fit("bowl", (0.2, 0.2, 0.08), True, "open") == 1.0
    assert fit("bowl", (0.2, 0.2, 0.08), True, "closed") == detect.SHAPE_MISS
    assert fit("globe", (0.2, 0.2, 0.08), True, "open") == detect.SHAPE_MISS


def _object(votes: dict[str, float], size: tuple[float, float, float],
            others: dict[str, float] | None = None) -> MapObject:
    s = Sighting("k", 1, [0, 0, 0], 0.3, 500, [False] * 4, [0, 0, -1.0], [0, 0, 0.3], 0.01,
                 [0, 0, 0], "mug", 0.6)
    return MapObject(1, votes, [s], box=Box(np.zeros(3), np.array(size), 0.0),
                     others=others or {})


def test_labels_ranked_by_evidence_and_fit() -> None:
    o = _object({"satellite dish": 0.67, "computer mouse": 0.65}, (0.09, 0.08, 0.04))
    assert o.label == "computer mouse"
    assert o.evidence()["satellite dish"] < 0.1
    o.box = Box(np.zeros(3), np.array([1.0, 0.8, 0.1]), 0.0)
    assert o.label == "satellite dish"
    # a label only the label scores gave wins when the detected ones do not fit the box
    o = _object({"mirror": 0.28}, (0.18, 0.16, 0.08), {"bowl": 0.12, "cup": 0.13})
    assert o.label == "mirror"  # its relief unknown
    o.box.centre = np.array([0.0, 0.0, 0.04])
    o.points = _points(lambda r: 0.01 + 0.07 * r ** 2)  # seen from above: open
    assert o.label == "bowl"  # a cup is smaller, a mirror thin and closed
    assert o.detected_as() == ["bowl", "mirror"]
    assert MapObject(2, {"mug": 1.0}).evidence() == {"mug": 1.0}
    back = MapObject.from_json(o.to_json())
    assert back.others == o.others  # its points are kept apart (objects.npz)


def _points(top: callable, n: int = 40) -> np.ndarray:  # type: ignore[valid-type]
    xy = np.random.default_rng(1).uniform(-0.1, 0.1, (n * n, 2))
    return np.c_[xy, [top(np.hypot(*p) / 0.1) for p in xy]].astype(np.float32)


def test_relief_from_above() -> None:
    from oh_my_slam.mapping.anchored_objects import relief

    o = _object({"bowl": 1.0}, (0.2, 0.2, 0.08))
    o.box.centre = np.array([0.0, 0.0, 0.04])
    o.points = _points(lambda r: 0.01 + 0.07 * r ** 2)  # a bowl: low inside, high at the rim
    assert relief(o) == "open"
    o.points = _points(lambda r: 0.08 - 0.05 * r ** 2)  # a dome
    assert relief(o) == "closed"
    o.points = _points(lambda r: 0.04)  # flat
    assert relief(o) is None
    o.points = o.points[:5]  # too few points inside
    assert relief(o) is None
    o.sightings[0].ray = [1.0, 0.0, 0.0]  # never seen from above
    assert relief(o) is None
    assert relief(MapObject(3, {"bowl": 1.0})) is None
