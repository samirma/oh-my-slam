"""Segmentation on synthetic frames with the fake client: detection filters, exclusive masks,
lifting, OBBs, ids/colours, catalogue, segmented.png, artefacts and the OpenLABEL scene."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.images import load_png
from oh_my_slam.core.log import json_payload_bytes
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.api import reconstruct_image
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation import detect
from oh_my_slam.segmentation.api import (
    pixel_owners,
    reconstruct_and_detect,
    segment_frame,
)
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES, write_artifacts
from oh_my_slam.segmentation.catalog import CSV_HEADER, catalog_csv, catalog_md
from oh_my_slam.segmentation.cloud import (
    derive_cloud,
    image_cloud_source,
    map_cloud_source,
)
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, segment_colors
from oh_my_slam.segmentation.render import segmented_image, segmented_png
from oh_my_slam.segmentation.scene import single_image_scene
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


@pytest.fixture
def synth(tmp_path: Path) -> tuple[FakeClient, Path]:
    room = default_room()
    pose = look_at(np.array([2.6, 2.2, 1.6]), np.array([0.0, 0.0, 0.4]))
    r = render(room, pose, K)
    inst = []
    for k, box in enumerate(room.boxes):
        m = r.ids == k + 2
        if m.sum() > 100:
            inst.append(FakeInstance(box.label, 0.9 - 0.1 * k, m))
    # a low-score detection and a duplicate (same mask, other label)
    inst.append(FakeInstance("cabinet", 0.3, r.ids == 2))
    inst.append(FakeInstance("cupboard", 0.6, r.ids == 2))
    inst.append(FakeInstance("floor", 0.95, r.ids == 0))
    client = FakeClient()
    up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    img = client.add(tmp_path / "room.png", r.rgb, FakeFrame(r.depth, K, up, inst, pose=pose))
    return client, img


def test_detect_filters_and_orders(synth) -> None:  # type: ignore[no-untyped-def]
    client, img = synth
    dets = detect.detect(img, client=client)
    labels = [d.label for d in dets]
    assert "floor" not in labels  # background concept never reported
    assert labels.count("cabinet") == 1 and "cupboard" not in labels  # duplicate suppressed
    assert all(d.score >= 0.5 for d in dets)
    assert [d.score for d in dets] == sorted([d.score for d in dets], reverse=True)
    hi = detect.detect(img, client=client, min_score=0.85)
    assert {d.label for d in hi} <= {"cabinet"}


def test_vocabulary_and_label_rules() -> None:
    vocab = detect.default_vocabulary()
    assert 250 <= len(vocab) <= 320 and len(set(vocab)) == len(vocab)
    for needed in ("chair", "dining table", "person", "potted plant", "bottle", "television",
                   "sofa", "coffee table", "door", "window", "balcony", "street light"):
        assert needed in vocab
    assert detect.normalize_label("  Dining_Table ") == "dining table"
    assert detect.compatible("sofa", "couch") and not detect.compatible("sofa", "person")
    # a pile of magazines is a book in some keyframes and a magazine in others
    assert detect.compatible("book", "magazine") and not detect.compatible("book", "bookcase")
    # floor grounding: (largest gap closed, smallest visible share of the grounded height)
    assert detect.grounding("Chair") == (detect.FLOOR_STANDING["chair"], detect.GROUND_MIN_VISIBLE)
    assert detect.grounding("dining table") == (detect.FLOOR_STANDING["dining table"], 0.0)
    assert detect.grounding("bottle") == (0.0, 0.0)
    assert detect.grounding("door") == (detect.FLOOR_STANDING["door"], detect.GROUND_MIN_VISIBLE)
    # pieces of one horizontal surface: surface classes with compatible labels only
    assert detect.split_surface("desk", "kitchen island") and detect.split_surface("rug", "carpet")
    assert not detect.split_surface("desk", "bed") and not detect.split_surface("cabinet", "cabinet")


def test_segment_frame_objects(synth) -> None:  # type: ignore[no-untyped-def]
    client, img = synth
    frame = reconstruct_image(img, client=client)
    seg = segment_frame(frame, client=client)
    assert [o.id for o in seg.objects] == list(range(1, len(seg.objects) + 1))
    by_label = {o.label: o for o in seg.objects}
    assert set(by_label) == {"cabinet", "box", "sofa"}
    sofa = by_label["sofa"]
    # visible-surface box of a 1.6 x 0.4 x 0.9 sofa: width and height close to truth
    assert sofa.obb.size[0] == pytest.approx(1.6, abs=0.15)
    assert sofa.obb.size[2] == pytest.approx(0.9, abs=0.1)
    # exclusive masks: every labelled pixel belongs to exactly one object
    for o in seg.objects:
        assert o.pixel_count == int((seg.label_map == o.id).sum())
        assert o.point_count == len(seg.points[o.id])
    cloud = derive_cloud(image_cloud_source(frame, seg), CloudAttrs(color="segment", label=True))
    assert cloud.label is not None and cloud.rgb is not None
    for o in seg.objects:  # colour contract: the object colour on exactly its lifted points
        np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == o.id], axis=0),
                                      [o.color])
        assert int((cloud.label == o.id).sum()) == o.point_count
    np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == 0], axis=0), [UNSEGMENTED])
    f2, dets = reconstruct_and_detect(img, client)
    assert len(dets) == 3 and f2.gravity is not None


def _rect(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros((10, 10), bool)
    m[r0:r1, c0:c1] = True
    return m


def exclusive_masks(dets: list[detect.Detection], shape: tuple[int, int]) -> list[np.ndarray]:
    """Per-detection masks after ``pixel_owners`` gave every pixel to at most one detection."""
    owner = pixel_owners(dets, shape)
    return [owner == i for i in range(len(dets))]


@pytest.mark.parametrize("cup_score", [0.55, 0.95])
def test_exclusive_masks_nested_object_keeps_its_pixels(cup_score: float) -> None:
    """A plate on a table: the smaller mask keeps its pixels whether it scores lower or higher
    than the larger one around it, and every pixel has at most one owner."""
    dets = [detect.Detection("table", 0.9, "yoloe", _rect(0, 10, 0, 10), (0, 0, 10, 10)),
            detect.Detection("cup", cup_score, "yoloe", _rect(2, 4, 2, 4), (2, 2, 4, 4))]
    table, cup = exclusive_masks(dets, (10, 10))
    assert cup.sum() == 4 and table.sum() == 96 and not (cup & table).any()
    assert [m.sum() for m in exclusive_masks(dets[::-1], (10, 10))] == [4, 96]  # order-free


def test_exclusive_masks_low_score_detections_get_only_leftover_pixels() -> None:
    """Below the default threshold a detection never takes pixels from one at or above it (such
    detections are mostly fragments of the object around them), but nests among its peers."""
    trusted = detect.Detection("chair", 0.6, "yoloe", _rect(0, 6, 0, 10), (0, 0, 10, 6))
    fragment = detect.Detection("chair", 0.3, "yoloe", _rect(2, 8, 2, 6), (2, 2, 6, 8))
    chair, frag = exclusive_masks([trusted, fragment], (10, 10))
    assert chair.sum() == 60 and frag.sum() == 8 and not (chair & frag).any()
    low_table = detect.Detection("table", 0.45, "yoloe", _rect(0, 10, 0, 10), (0, 0, 10, 10))
    low_cup = detect.Detection("cup", 0.3, "yoloe", _rect(2, 4, 2, 4), (2, 2, 4, 4))
    table, cup = exclusive_masks([low_table, low_cup], (10, 10))
    assert cup.sum() == 4 and table.sum() == 96


@pytest.fixture
def overlapping(tmp_path: Path) -> tuple[FakeClient, Path]:
    """Three boxes (cabinet 0.9, box 0.8, sofa 0.7), a cushion (0.55) nested in the sofa, and
    low-score detections that overlap them (and each other)."""
    room = default_room()
    pose = look_at(np.array([2.6, 2.2, 1.6]), np.array([0.0, 0.0, 0.4]))
    r = render(room, pose, K)
    inst = [FakeInstance(b.label, 0.9 - 0.1 * k, r.ids == k + 2) for k, b in enumerate(room.boxes)]
    sofa = r.ids == 4
    rows, cols = np.nonzero(sofa)
    cushion = np.zeros_like(sofa)
    cushion[85:115, 50:90] = True
    inst.append(FakeInstance("cushion", 0.55, cushion & sofa))
    blob = np.zeros_like(sofa)  # covers the left half of the sofa and the floor next to it
    blob[rows.min() - 10: rows.max() + 10, cols.min() - 20: (cols.min() + cols.max()) // 2] = True
    inst.append(FakeInstance("chair", 0.45, blob))
    cab_rows, cab_cols = np.nonzero(r.ids == 2)
    lamp = np.zeros_like(sofa)  # the cabinet's top and the wall above it
    lamp[cab_rows.min() - 25: cab_rows.min() + 15, cab_cols.min(): cab_cols.max()] = True
    inst.append(FakeInstance("lamp", 0.35, lamp))
    client = FakeClient()
    up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    img = client.add(tmp_path / "over.png", r.rgb, FakeFrame(r.depth, K, up, inst, pose=pose))
    return client, img


def _objects(client: FakeClient, img: Path, min_score: float, via_cli_path: bool = False
             ) -> dict[int, tuple]:
    if via_cli_path:  # segment.sh / reconstruct.sh / view.sh: floor-level detections, then filter
        frame, dets = reconstruct_and_detect(img, client)
        seg = segment_frame(frame, client=client, detections=dets, min_score=min_score)
    else:
        frame = reconstruct_image(img, client=client)
        seg = segment_frame(frame, client=client, min_score=min_score)
    pixels = {o.id: np.flatnonzero(seg.label_map == o.id).tolist() for o in seg.objects}
    return {o.id: (o.label, o.score, o.color, o.obb.center.tolist(), o.obb.size.tolist(),
                   o.obb.R.tolist(), o.pixel_count, o.point_count, pixels[o.id],
                   seg.points[o.id].tolist()) for o in seg.objects}


def test_min_score_only_adds_or_removes_objects(overlapping) -> None:  # type: ignore[no-untyped-def]
    client, img = overlapping
    confs: list[float] = []
    real = client.segment_image

    def spy(req):  # type: ignore[no-untyped-def]
        confs.append(req.conf)
        return real(req)

    client.segment_image = spy  # type: ignore[method-assign]
    runs = {t: _objects(client, img, t) for t in (0.3, 0.5, 0.6, 0.85)}
    labels = {t: [v[0] for v in objs.values()] for t, objs in runs.items()}
    assert labels[0.85] == ["cabinet"]
    assert labels[0.6] == ["cabinet", "box", "sofa"]
    assert labels[0.5] == ["cabinet", "box", "sofa", "cushion"]  # nested in the sofa, kept
    assert set(labels[0.3]) == {"cabinet", "box", "sofa", "cushion", "chair", "lamp"}
    for t in runs:
        assert list(runs[t]) == list(range(1, len(runs[t]) + 1))
    for lo, hi in ((0.3, 0.5), (0.5, 0.6), (0.6, 0.85), (0.3, 0.85)):
        # same id, colour, OBB, pixels and points at both thresholds; new ones appended
        assert {oid: runs[lo][oid] for oid in runs[hi]} == runs[hi]
    by_label = {v[0]: v for v in runs[0.5].values()}
    sofa, cushion = by_label["sofa"], by_label["cushion"]
    assert not set(sofa[8]) & set(cushion[8]) and cushion[6] > 500  # the cushion's own pixels
    assert _objects(client, img, 0.5) == runs[0.5]  # re-running gives the same ids and colours
    assert _objects(client, img, 0.5, via_cli_path=True) == runs[0.5]  # same code, same objects
    # the server request depends on --min-score only by tier (request_floor): below the default
    # everything down to DETECTION_FLOOR, from the default up nothing below it (the objects at
    # 0.5 are the same either way: the equality above for 0.3 vs 0.5)
    assert {round(c, 4) for c in confs} == {detect.DETECTION_FLOOR, detect.TRUSTED_SCORE}
    assert detect.request_floor(0.3) == detect.DETECTION_FLOOR
    assert detect.request_floor(0.5) == detect.request_floor(0.85) == detect.TRUSTED_SCORE
    with pytest.raises(ValueError):
        detect.detect(img, client=client, min_score=detect.DETECTION_FLOOR / 2)
    # the spec bounds --min-score nowhere: below the detector's floor it keeps what the floor keeps
    assert (_objects(client, img, detect.DETECTION_FLOOR / 2)
            == _objects(client, img, detect.DETECTION_FLOOR))


def test_low_scoring_detections_never_change_the_default_objects(overlapping) -> None:  # type: ignore[no-untyped-def]
    """The spec sets no lower bound on --min-score, so the detector is asked for scores down to
    ``DETECTION_FLOOR`` (0.05). Detections below the default 0.5 — however many, whatever they
    cover — leave the objects at 0.5 with the same ids, colours, masks, points and OBBs."""
    client, img = overlapping
    frame = reconstruct_image(img, client=client)
    dets = detect.detect(img, client=client, min_score=detect.DETECTION_FLOOR)
    h, w = frame.depth.shape
    everywhere = np.ones((h, w), bool)
    middle = np.zeros((h, w), bool)
    middle[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    low = [detect.Detection("chair", 0.2, "yoloe", everywhere, (0.0, 0.0, float(w), float(h))),
           detect.Detection("lamp", 0.08, "yoloe", middle,
                            (w / 4, h / 4, 3 * w / 4, 3 * h / 4)),
           detect.Detection("cup", 0.06, "yoloe", middle[::-1], (0.0, 0.0, 1.0, 1.0))]

    def signature(detections: list[detect.Detection], min_score: float) -> dict[int, tuple]:
        seg = segment_frame(frame, client=client, detections=detections, min_score=min_score)
        return {o.id: (o.label, o.score, o.color, o.obb.center.tolist(), o.obb.size.tolist(),
                       np.flatnonzero(seg.label_map == o.id).tolist(),
                       seg.points[o.id].tolist()) for o in seg.objects}

    plain = signature(dets, 0.5)
    assert plain and signature(dets + low, 0.5) == plain
    assert signature(dets, 0.5) == signature(dets + low[:1], 0.5)
    everything = signature(dets + low, detect.DETECTION_FLOOR)  # the lowest --min-score works
    assert {oid: everything[oid] for oid in plain} == plain and len(everything) > len(plain)


def test_segment_frame_does_not_depend_on_the_detection_order(overlapping) -> None:  # type: ignore[no-untyped-def]
    client, img = overlapping
    ref = _objects(client, img, 0.3)
    real = client.segment_image

    def reversed_order(req):  # type: ignore[no-untyped-def]
        res = real(req)
        return res.model_copy(update={"instances": res.instances[::-1]})

    client.segment_image = reversed_order  # type: ignore[method-assign]
    assert _objects(client, img, 0.3) == ref


class PassDetector:
    """A detector whose output depends on the image it is shown, like YOLOE: on the fine pass
    (the image at 1024 px) a dark screen is a 'television' at 0.13 and a window with its frame
    scores 0.51; on the coarse pass (768 px, upsampled by the detector) the screen is a 'computer
    monitor' at 0.56, the window without its frame (IoU 0.67) scores 0.7, and a wire appears
    (small: the coarse pass's small detections are dropped). Regions are fractions of the image."""

    FINE = (("television", 0.13, (0.3, 0.1, 0.8, 0.5)), ("cup", 0.9, (0.05, 0.5, 0.12, 0.6)),
            ("window", 0.51, (0.0, 0.0, 0.3, 0.45)))
    COARSE = (("computer monitor", 0.56, (0.3, 0.1, 0.8, 0.5)),
              ("window", 0.7, (0.0, 0.0, 0.3, 0.3)), ("wire", 0.6, (0.5, 0.6, 0.55, 0.62)))

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.requests: list[tuple[int, int, float]] = []

    def clone(self) -> PassDetector:
        return self

    def close(self) -> None:
        pass

    def segment_image(self, req):  # type: ignore[no-untyped-def]
        from oh_my_slam.client import protocol as p
        from oh_my_slam.core import rle
        from oh_my_slam.core.images import size_at_max_side

        self.requests.append((req.max_side, req.imgsz, req.conf))
        w, h = size_at_max_side(*self.size, req.max_side)
        fine = req.max_side >= max(self.size) or req.max_side == detect.DETECT_SIDE
        inst = []
        for label, score, (x0, y0, x1, y1) in self.FINE if fine else self.COARSE:
            if score <= req.conf:
                continue
            box = [x0 * w, y0 * h, x1 * w, y1 * h]
            m = np.zeros((h, w), bool)
            m[round(box[1]):round(box[3]), round(box[0]):round(box[2])] = True
            inst.append(p.Instance(label=label, score=score, source="yoloe", box_xyxy=box,
                                   mask=rle.encode(m)))
        return p.SegmentResponse(width=w, height=h, instances=inst)


def _signature(dets: list[detect.Detection]) -> list[tuple[str, float, float]]:
    return [(d.label, d.score, round(d.area / d.mask.size, 2)) for d in dets]


def test_detections_do_not_depend_on_the_callers_grid(tmp_path: Path) -> None:
    """``segment.sh -i`` (1024 px grid) and the mapper (768 px keyframe grid, via
    ``detect_alongside``) get the same detections of one image: the detector is shown the same
    fine and coarse inputs, and only the masks are resampled onto each grid (critic OI-1: the
    monitor the mapper found and segment.sh -i missed)."""
    from oh_my_slam.segmentation.api import detect_alongside

    img = tmp_path / "desk.png"
    Image.new("RGB", (1600, 1200)).save(img)
    client = PassDetector((1600, 1200))
    single = detect.detect(img, client=client, min_score=0.5, floor=0.5)  # type: ignore[arg-type]
    assert {r[:2] for r in client.requests} == {(1024, 1024), (768, 1024)}
    client.requests.clear()
    _, keyframe = detect_alongside(img, client, lambda c: None,  # type: ignore[arg-type,return-value]
                                   max_side=768)
    assert {r[:2] for r in client.requests} == {(1024, 1024), (768, 1024)}
    assert [d.mask.shape for d in single] == [(768, 1024)] * 3
    assert [d.mask.shape for d in keyframe] == [(576, 768)] * 3
    # one window (the passes' detections of it are one object), the monitor, no wire
    assert _signature(single) == _signature(keyframe) == [
        ("cup", 0.9, 0.01), ("window", 0.7, 0.09), ("computer monitor", 0.56, 0.2)]
    x0, y0, x1, y1 = keyframe[2].box
    assert (x0, y0, x1, y1) == pytest.approx((0.3 * 768, 0.1 * 576, 0.8 * 768, 0.5 * 576))
    # down to the floor: the fine pass's television and 0.51 window are the coarse pass's objects
    low = detect.detect(img, client=client, min_score=detect.DETECTION_FLOOR,  # type: ignore[arg-type]
                        max_side=768)
    assert _signature(low) == _signature(keyframe)


def test_small_images_take_one_detection_pass(tmp_path: Path) -> None:
    img = tmp_path / "small.png"
    Image.new("RGB", (640, 480)).save(img)
    client = PassDetector((640, 480))
    dets = detect.detect(img, client=client, max_side=768)  # type: ignore[arg-type]
    assert [r[0] for r in client.requests] == [detect.DETECT_SIDE]
    assert [d.label for d in dets] == ["cup", "window"] and dets[0].mask.shape == (480, 640)


def test_fuse_passes_and_resample_mask() -> None:
    a = detect.Detection("window", 0.51, "yoloe", _rect(0, 6, 0, 10), (0, 0, 10, 6))
    b = detect.Detection("window", 0.7, "yoloe", _rect(0, 4, 0, 10), (0, 0, 10, 4))  # IoU 0.67
    c = detect.Detection("cup", 0.6, "yoloe", _rect(0, 4, 0, 10), (0, 0, 10, 4))
    assert detect.fuse_passes([[a], [b]]) == [b]
    assert detect.fuse_passes([[a, c], []]) == [c, a]  # one pass: nothing is fused
    assert detect.fuse_passes([[b], [a]]) == detect.fuse_passes([[a], [b]])  # order-free
    m = _rect(2, 6, 4, 8)
    np.testing.assert_array_equal(detect.resample_mask(m, (10, 10)), m)
    half = detect.resample_mask(m, (5, 5))
    assert half.shape == (5, 5) and half.sum() == 4 and half[1:3, 2:4].all()


def test_scene_catalogue_render_and_artifacts(synth, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client, img = synth
    frame = reconstruct_image(img, client=client)
    seg = segment_frame(frame, client=client)
    scene = single_image_scene(seg, tool="segment")
    assert validation_errors(scene) == []
    objs = scene["openlabel"]["objects"]
    for o in seg.objects:
        entry = objs[str(o.id)]
        assert entry["type"] == o.label and entry["name"] == f"{o.label} {o.id}"
        assert entry["object_data"]["text"][0]["val"] == o.color_hex
        assert entry["object_data"]["vec"][0]["val"] == list(o.color)
        assert "boolean" not in entry["object_data"]  # every listed object is an object
    md = scene["openlabel"]["metadata"]
    assert md["intrinsics_source"] == "model" and md["gravity"]["source"].startswith("geocalib")

    csv_text = catalog_csv(seg.objects)
    assert csv_text.splitlines()[0] == ",".join(CSV_HEADER)
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert [int(r["id"]) for r in rows] == [o.id for o in seg.objects]
    md_text = catalog_md(seg.objects)
    vols = [float(line.split("|")[9]) for line in md_text.splitlines() if line.startswith("| <span")]
    assert vols == sorted(vols, reverse=True)

    png = segmented_image(frame.rgb, seg.label_map)
    for o in seg.objects:
        np.testing.assert_array_equal(np.unique(png[seg.label_map == o.id], axis=0), [o.color])
    bg = seg.label_map == 0
    assert (png[bg].astype(int) <= frame.rgb[bg].astype(int)).all()

    payload = json_payload_bytes(scene)
    out = tmp_path / "out"
    png_payload = segmented_png(frame.rgb, seg.label_map)
    files = write_artifacts(out, payload, png_payload, seg.objects, "t")
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)
    assert [f.name for f in files] == list(ARTIFACT_NAMES) == [
        "segmentation.json", "segmented.png", "catalog.csv", "catalog.md"]
    assert (out / "segmentation.json").read_bytes() == payload
    assert json.loads(payload)["openlabel"]["objects"]
    assert (out / "segmented.png").read_bytes() == png_payload  # the bytes of -f png
    img_back = load_png(out / "segmented.png")
    np.testing.assert_array_equal(img_back, png)
    assert "icc_profile" not in Image.open(out / "segmented.png").info


def test_map_source_keeps_only_exported_objects() -> None:
    src = map_cloud_source(np.zeros((4, 3)), np.zeros((4, 3), np.uint8), np.array([0, 3, 3, 9]),
                           {3}, np.zeros((0, 3)))
    out = derive_cloud(src, CloudAttrs(color="segment", label=True))
    assert out.label is not None and out.label.tolist() == [0, 3, 3, 0]
    assert out.rgb is not None
    assert tuple(out.rgb[1]) == color_for_id(3) and tuple(out.rgb[3]) == UNSEGMENTED
    assert segment_colors(np.array([], np.int32)).shape == (0, 3)

