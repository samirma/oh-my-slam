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

from oh_my_slam.core.images import load_png
from oh_my_slam.core.log import json_payload_bytes
from oh_my_slam.core.ply import PointCloud, read_ply
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.api import reconstruct_image
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation import calibration, detect
from oh_my_slam.segmentation.api import (
    colorize_cloud,
    exclusive_masks,
    reconstruct_and_detect,
    segment_frame,
)
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES, write_artifacts
from oh_my_slam.segmentation.catalog import CSV_HEADER, catalog_csv, catalog_md
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id
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
    only = detect.detect(img, client=client, labels=["Sofa"])
    assert [d.label for d in only] == ["sofa"]
    floors = detect.detect(img, client=client, labels=["floor"])
    assert [d.label for d in floors] == ["floor"]  # explicitly asked for
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


def test_calibration_maps() -> None:
    ident = calibration.load_map("yoloe")
    assert ident.is_identity and ident(0.42) == pytest.approx(0.42)
    m = calibration.CalibrationMap("x", (0.0, 0.2, 1.0), (0.0, 0.5, 0.9))
    assert m(0.2) == pytest.approx(0.5) and m(1.0) == pytest.approx(0.9)
    assert m.to_dict()["knots_raw"] == [0.0, 0.2, 1.0]
    with pytest.raises(ValueError):
        calibration.CalibrationMap("x", (0.0, 1.0), (1.0, 0.0))
    with pytest.raises(ValueError):
        calibration.CalibrationMap("x", (0.0,), (0.0,))
    with pytest.raises(ValueError):
        calibration.CalibrationMap("x", (1.0, 0.0), (0.0, 1.0))
    assert calibration.calibrate(0.7, "sam3") == pytest.approx(0.7)
    assert calibration.load_map("unknown-path").is_identity


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
    cloud = seg.segments_cloud()
    assert cloud.label is not None
    for o in seg.objects:
        np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == o.id], axis=0),
                                      [o.color])
    np.testing.assert_array_equal(np.unique(cloud.rgb[cloud.label == 0], axis=0), [UNSEGMENTED])
    f2, dets = reconstruct_and_detect(img, client)
    assert len(dets) == 3 and f2.gravity is not None


def test_exclusive_masks_small_on_top() -> None:
    from oh_my_slam.segmentation.detect import Detection

    big = np.zeros((10, 10), bool)
    big[:, :] = True
    small = np.zeros((10, 10), bool)
    small[2:4, 2:4] = True
    dets = [Detection("table", 0.9, 0.9, "yoloe", big, (0, 0, 10, 10)),
            Detection("cup", 0.8, 0.8, "yoloe", small, (2, 2, 4, 4))]
    a, b = exclusive_masks(dets, (10, 10))
    assert b.sum() == 4 and a.sum() == 96 and not (a & b).any()


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
    files = write_artifacts(out, payload, png, seg.objects, seg.segments_cloud(), "t")
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)
    assert [f.name for f in files] == list(ARTIFACT_NAMES)
    assert (out / "segmentation.json").read_bytes() == payload
    assert json.loads(payload)["openlabel"]["objects"]
    img_back = load_png(out / "segmented.png")
    np.testing.assert_array_equal(img_back, png)
    assert "icc_profile" not in Image.open(out / "segmented.png").info
    ply = read_ply(out / "segments.ply")
    assert ply.label is not None and len(ply) == len(seg.segments_cloud())


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


def test_colorize_cloud() -> None:
    c = PointCloud(np.zeros((4, 3)), np.zeros((4, 3)), np.array([0, 3, 3, 9]))
    out = colorize_cloud(c, {3})
    assert out.label is not None and out.label.tolist() == [0, 3, 3, 0]
    assert tuple(out.rgb[1]) == color_for_id(3) and tuple(out.rgb[3]) == UNSEGMENTED


def test_export_map_contact_sheet_and_cloud() -> None:
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
    cloud = PointCloud(np.zeros((5, 3)), np.zeros((5, 3)), np.array([3, 5, 8, 99, 0]))
    out = export_map(objs, kfs, cloud)
    assert out.tiles == ["f000001", "f000002"]
    sheet_colors = {tuple(c) for c in out.segmented.reshape(-1, 3)}
    assert {color_for_id(3), color_for_id(5), color_for_id(8)} <= sheet_colors
    assert color_for_id(99) not in sheet_colors  # unknown id never painted
    assert out.segments.label is not None and out.segments.label.tolist() == [3, 5, 8, 0, 0]
    empty = export_map([], kfs[:1], cloud)
    assert empty.tiles == ["f000000"]
