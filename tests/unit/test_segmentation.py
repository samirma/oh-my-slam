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
from oh_my_slam.core.ply import read_ply
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
    cloud_ply,
    derive_cloud,
    image_cloud_source,
    map_cloud_source,
)
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, segment_colors
from oh_my_slam.segmentation.render import (
    choose_contact_frames,
    contact_sheet,
    segmented_image,
)
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
    assert detect.floor_gap("chair") > 0 and detect.floor_gap("bottle") == 0


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
    assert len(set(confs)) == 1  # the server request does not depend on --min-score
    with pytest.raises(ValueError):
        detect.detect(img, client=client, min_score=detect.DETECTION_FLOOR / 2)
    frame = reconstruct_image(img, client=client)
    with pytest.raises(ValueError):
        segment_frame(frame, client=client, min_score=detect.DETECTION_FLOOR / 2)


def test_segment_frame_does_not_depend_on_the_detection_order(overlapping) -> None:  # type: ignore[no-untyped-def]
    client, img = overlapping
    ref = _objects(client, img, 0.3)
    real = client.segment_image

    def reversed_order(req):  # type: ignore[no-untyped-def]
        res = real(req)
        return res.model_copy(update={"instances": res.instances[::-1]})

    client.segment_image = reversed_order  # type: ignore[method-assign]
    assert _objects(client, img, 0.3) == ref


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
    ply_payload = cloud_ply(image_cloud_source(frame, seg), CloudAttrs(color="segment"))
    files = write_artifacts(out, payload, png, seg.objects, ply_payload, "t")
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)
    assert [f.name for f in files] == list(ARTIFACT_NAMES)
    assert (out / "segmentation.json").read_bytes() == payload
    assert json.loads(payload)["openlabel"]["objects"]
    img_back = load_png(out / "segmented.png")
    np.testing.assert_array_equal(img_back, png)
    assert "icc_profile" not in Image.open(out / "segmented.png").info
    assert (out / "segments.ply").read_bytes() == ply_payload
    ply = read_ply(out / "segments.ply")
    assert ply.label is None and ply.rgb is not None  # label is off by default


def test_contact_sheet_and_set_cover() -> None:
    sets = [{1, 2}, {2, 3}, {4}, {1, 2, 3}, set()]
    chosen = choose_contact_frames(sets, max_tiles=6)
    covered = set().union(*(sets[i] for i in chosen))
    assert covered == {1, 2, 3, 4} and len(chosen) == 2
    assert choose_contact_frames([], 6) == []
    lab = np.zeros((20, 30), np.int32)
    lab[5:10, 5:10] = 7
    rgb = np.full((20, 30, 3), 200, np.uint8)
    sheet = contact_sheet([("f000001", rgb, lab), ("f000002", rgb, lab)], tile_width=60)
    assert sheet.shape[1] == 120
    colors = {tuple(c) for c in sheet.reshape(-1, 3)}
    assert color_for_id(7) in colors
    assert contact_sheet([]).shape[0] > 0


def test_map_source_keeps_only_exported_objects() -> None:
    src = map_cloud_source(np.zeros((4, 3)), np.zeros((4, 3), np.uint8), np.array([0, 3, 3, 9]),
                           {3}, np.zeros((0, 3)))
    out = derive_cloud(src, CloudAttrs(color="segment", label=True))
    assert out.label is not None and out.label.tolist() == [0, 3, 3, 0]
    assert out.rgb is not None
    assert tuple(out.rgb[1]) == color_for_id(3) and tuple(out.rgb[3]) == UNSEGMENTED
    assert segment_colors(np.array([], np.int32)).shape == (0, 3)


def test_export_map_contact_sheet() -> None:
    from oh_my_slam.segmentation.api import KeyframeLabels, SceneObject, export_map
    from oh_my_slam.segmentation.obb import OBB

    objs = [SceneObject(i, "chair", 0.9, OBB(np.zeros(3), np.eye(3), np.ones(3)), 10, 10)
            for i in (3, 5, 8)]
    kfs = []
    for name, ids in (("f000000", [3]), ("f000001", [3, 5]), ("f000002", [8, 99])):
        lab = np.zeros((40, 60), np.int32)
        for j, oid in enumerate(ids):
            lab[5:15, 5 + 15 * j: 15 + 15 * j] = oid
        kfs.append(KeyframeLabels(name, np.full((40, 60, 3), 100, np.uint8), lab))
    out = export_map(objs, kfs)
    assert out.tiles == ["f000001", "f000002"]
    sheet_colors = {tuple(c) for c in out.segmented.reshape(-1, 3)}
    assert {color_for_id(3), color_for_id(5), color_for_id(8)} <= sheet_colors
    assert color_for_id(99) not in sheet_colors  # unknown id never painted
    empty = export_map([], kfs[:1])
    assert empty.tiles == ["f000000"]
