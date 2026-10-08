"""Contract checkers of the evaluator on synthetic artefacts: stdout purity (JSON, PLY, PNG, the
depth image), OpenLABEL validity and the colour contract (JSON, segmented image, catalogue,
PLY)."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.images import png_bytes
from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.artifacts import write_artifacts
from oh_my_slam.segmentation.catalog import catalog_csv, catalog_md
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, color_hex_for_id
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.segmentation.render import segmented_image, segmented_png
from oh_my_slam.segmentation.scene import objects_block
from oh_my_slam.tools.evaluate.contracts import (
    ContractLog,
    artifact_problems,
    catalog_csv_problems,
    catalog_md_problems,
    cloud_colour_problems,
    depth_image_problems,
    parse_scene,
    payload_problems,
    png_colour_problems,
    same_objects_problems,
    scene_colour_problems,
    served_cloud_problems,
    subject_of,
)
from oh_my_slam.tools.evaluate.scene import doc_objects
from oh_my_slam.viewer.bundle import DisplayCloud
from oh_my_slam.viewer.routes import cloud_document

IDS = (1, 2, 7, 23)  # 23: a hue-rotated palette colour


def objects() -> list[SceneObject]:
    return [SceneObject(i, "chair", 0.8, OBB(np.array([i, 0.0, 2.0]), np.eye(3),
                                             np.array([0.5, 0.4, 0.9])), 100, 50)
            for i in IDS]


def scene_doc() -> dict:
    return ol.document(ol.metadata("x"), objects_block(objects(), "camera"),
                       coordinate_systems={"camera": ol.sensor_cs()})


def scene_bytes() -> bytes:
    return (json.dumps(scene_doc()) + "\n").encode()


def labelled_cloud(n: int = 400) -> PointCloud:
    rng = np.random.default_rng(0)
    lab = rng.choice([0, *IDS], n).astype(np.int32)
    rgb = np.array([UNSEGMENTED if i == 0 else color_for_id(int(i)) for i in lab], np.uint8)
    return PointCloud(rng.random((n, 3)), rgb, lab)


# -- stdout purity ------------------------------------------------------------------------------------


def test_one_json_document() -> None:
    assert payload_problems(scene_bytes(), "json") == []
    assert payload_problems(b"loading models\n" + scene_bytes(), "json")
    assert payload_problems(scene_bytes() + scene_bytes(), "json")
    assert payload_problems(b"", "json")
    assert payload_problems(b"[1, 2]", "json")


def test_one_ply_binary_or_ascii() -> None:
    cloud = labelled_cloud()
    binary = ply_bytes(cloud, comments=["attributes color=segment"])
    ascii_ = ply_bytes(cloud, encoding="ascii")
    assert payload_problems(binary, "ply") == []
    assert payload_problems(ascii_, "ply") == []
    assert payload_problems(binary + b"done\n", "ply")  # trailing progress text
    assert payload_problems(binary[:-5], "ply")  # truncated
    assert payload_problems(ascii_ + b"1 2 3 0 0 0 0\n", "ply")  # one row too many
    assert payload_problems(b"[info] start\n" + binary, "ply")  # banner before the header


def chunk(kind: bytes, data: bytes, crc: int | None = None) -> bytes:
    """One PNG chunk (its CRC, unless one is given)."""
    crc = zlib.crc32(kind + data) if crc is None else crc
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def test_one_png() -> None:
    png = png_bytes(np.zeros((4, 5, 3), np.uint8))
    assert payload_problems(png, "png") == []
    assert payload_problems(png_bytes(np.zeros((4, 5), np.uint16)), "png") == []
    assert payload_problems(b"", "png") == ["expected one PNG payload, got nothing"]
    assert "signature" in payload_problems(b"[info] start\n" + png, "png")[0]  # a banner first
    assert "after the PNG's IEND" in payload_problems(png + b"done\n", "png")[0]  # progress after
    assert "after the PNG's IEND" in payload_problems(png + png, "png")[0]  # two images
    assert "cut short" in payload_problems(png[:-6], "png")[0]  # truncated in the IEND chunk
    assert "cut short" in payload_problems(png[:20], "png")[0]  # truncated in the IHDR chunk
    signature, ihdr = png[:8], png[8:33]
    assert payload_problems(signature, "png") == [
        "the chunks run [] … [], not IHDR … IEND (cut short?)"]
    assert "not IHDR … IEND" in payload_problems(signature + ihdr, "png")[0]  # no IEND at all
    iend = chunk(b"IEND", b"")
    assert "not IHDR … IEND" in payload_problems(signature + iend, "png")[0]  # no IHDR
    bad = bytearray(png)
    bad[30] ^= 0xFF  # a byte of the IHDR chunk's data
    assert "fails its CRC" in payload_problems(bytes(bad), "png")[0]
    garbage = chunk(b"IDAT", b"not zlib data")  # a well-formed chunk that does not decode
    assert "does not decode" in payload_problems(signature + ihdr + garbage + iend, "png")[0]


def test_the_depth_image_is_16_bit_greyscale_of_the_inputs_size() -> None:
    depth = png_bytes(np.zeros((3, 4), np.uint16))
    assert depth_image_problems(depth, (4, 3)) == []
    assert depth_image_problems(depth, (8, 6)) == ["4x3 pixels; the input has 8x6"]
    assert depth_image_problems(png_bytes(np.zeros((3, 4, 3), np.uint8)), (4, 3)) == [
        "bit depth 8 and colour type 2: not a 16-bit single-channel (greyscale) image"]
    assert "signature" in depth_image_problems(b"not a png", (4, 3))[0]


def test_empty_stdout_with_an_output_file() -> None:
    assert payload_problems(b"", "empty") == []
    assert payload_problems(b"\n", "empty")
    with pytest.raises(ValueError):
        payload_problems(b"", "xml")


def test_openlabel_validity() -> None:
    doc, problems = parse_scene(scene_bytes())
    assert doc is not None and problems == []
    bad = scene_doc()
    bad["$schema"] = ol.SCHEMA_URL  # the root allows only "openlabel"
    assert parse_scene(json.dumps(bad).encode())[1]
    assert parse_scene(b"not json")[0] is None


def test_subjects_of_entry_points() -> None:
    assert subject_of("start_inference_server.sh") == "server"
    assert subject_of("/repo/segment.sh") == "segment"


# -- colour contract ---------------------------------------------------------------------------------


def test_scene_colours_are_a_function_of_the_id() -> None:
    objs = doc_objects(scene_doc())
    assert scene_colour_problems(objs) == []
    doc = scene_doc()
    doc["openlabel"]["objects"]["7"]["object_data"]["text"][0]["val"] = color_hex_for_id(8)
    assert scene_colour_problems(doc_objects(doc)) == [
        f"id 7: color_hex {color_hex_for_id(8)} != {color_hex_for_id(7)}"]


def test_cloud_colours_with_labels() -> None:
    cloud = labelled_cloud()
    assert cloud_colour_problems(cloud, set(IDS)) == []
    assert cloud.rgb is not None and cloud.label is not None
    i = int(np.flatnonzero(cloud.label == 2)[0])
    cloud.rgb[i] = (np.asarray(color_for_id(2)) * 0.5 + 64).astype(np.uint8)  # blended
    assert "1 points not in their object's colour" in cloud_colour_problems(cloud, set(IDS))[0]
    grey_wrong = labelled_cloud()
    assert grey_wrong.rgb is not None and grey_wrong.label is not None
    grey_wrong.rgb[grey_wrong.label == 0] = (127, 127, 127)
    assert cloud_colour_problems(grey_wrong, None)
    assert cloud_colour_problems(labelled_cloud(), {1, 2}) == ["labels not in the scene: [7, 23]"]


def test_cloud_colours_without_labels() -> None:
    cloud = labelled_cloud()
    plain = PointCloud(cloud.xyz, cloud.rgb)
    assert cloud_colour_problems(plain, set(IDS)) == []
    assert cloud_colour_problems(plain, {1, 2})  # colours of ids 7, 23 are foreign
    assert cloud_colour_problems(plain, None)
    assert cloud_colour_problems(PointCloud(cloud.xyz), set(IDS)) == ["no colour properties"]


def test_viewer_cloud_colours() -> None:
    """The viewer's /api/cloud?color=segment payload, decoded with the viewer's own parser."""
    def served(cloud: PointCloud) -> bytes:
        return cloud_document(DisplayCloud(cloud, len(cloud), 0.0, 0), "color=segment").tobytes()

    cloud = labelled_cloud()
    assert served_cloud_problems(served(cloud), set(IDS)) == []
    assert served_cloud_problems(served(PointCloud(cloud.xyz, cloud.rgb)), set(IDS)) == []
    assert served_cloud_problems(served(cloud), {1}) == ["labels not in the scene: [2, 7, 23]"]
    assert cloud.rgb is not None
    cloud.rgb[0] = (1, 2, 3)
    assert "1 points not in their object's colour" in served_cloud_problems(served(cloud),
                                                                            set(IDS))[0]
    assert served_cloud_problems(None, set(IDS)) == ["the viewer served no color=segment cloud"]
    assert served_cloud_problems(b"\x01", set(IDS))[0].startswith("unreadable cloud payload")


def label_map(h: int = 60, w: int = 80) -> np.ndarray:
    lab = np.zeros((h, w), np.int32)
    for k, oid in enumerate(IDS):
        lab[5:25, 5 + 18 * k: 20 + 18 * k] = oid
    return lab


def test_segmented_png_exact_masks() -> None:
    rgb = np.random.default_rng(1).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    objs = doc_objects(scene_doc())
    good = segmented_image(rgb, label_map())
    assert png_colour_problems(good, objs) == []
    blended = good.copy()
    mask = label_map() == 7
    blended[mask] = (0.6 * np.asarray(color_for_id(7)) + 0.4 * rgb[mask]).astype(np.uint8)
    problems = png_colour_problems(blended, objs)
    assert any("not object colours" in p for p in problems)
    missing = segmented_image(rgb, np.where(label_map() >= 7, 0, label_map()))
    assert png_colour_problems(missing, objs) == [
        "only 2 of 4 object colours appear in the image"]
    grey = good.copy()
    grey[mask] = (200, 200, 200)  # a grey is no object colour either
    assert any("not object colours" in p for p in png_colour_problems(grey, objs))


def test_catalogue_colours() -> None:
    objs = doc_objects(scene_doc())
    csv_text, md_text = catalog_csv(objects()), catalog_md(objects(), "t")
    assert catalog_csv_problems(csv_text, objs) == []
    assert catalog_md_problems(md_text, objs) == []
    wrong = color_hex_for_id(3)
    assert catalog_csv_problems(csv_text.replace(color_hex_for_id(2), wrong), objs)
    assert catalog_md_problems(md_text.replace(f"`{color_hex_for_id(2)}`", f"`{wrong}`"), objs)
    assert catalog_md_problems(md_text.replace(f"color:{color_hex_for_id(2)}", f"color:{wrong}"),
                               objs)
    assert catalog_csv_problems(csv_text.replace("id,label", "id,name"), objs)
    assert catalog_md_problems(md_text, objs[:2])  # rows for objects the scene does not have


def test_same_objects() -> None:
    a = doc_objects(scene_doc())
    assert same_objects_problems(a, a, geometry=True) == []
    doc = scene_doc()
    doc["openlabel"]["objects"]["2"]["type"] = "table"
    del doc["openlabel"]["objects"]["23"]
    problems = same_objects_problems(a, doc_objects(doc), geometry=False)
    assert problems and problems[0].startswith("2 objects differ (ids [2, 23])")


def test_artefact_folder(tmp_path: Path) -> None:
    data = scene_bytes()
    png = segmented_png(np.zeros((60, 80, 3), np.uint8), label_map())
    write_artifacts(tmp_path, data, png, objects(), title="t")
    assert artifact_problems(tmp_path, data) == []
    assert artifact_problems(tmp_path, png, "png") == []  # the segmented image of -f png
    assert artifact_problems(tmp_path, None) == []
    assert artifact_problems(tmp_path, data + b" ") == [
        "segmentation.json differs from the -f json result on stdout"]
    assert artifact_problems(tmp_path, data, "png") == [
        "segmented.png differs from the -f png result on stdout"]
    (tmp_path / "segments.ply").write_bytes(ply_bytes(labelled_cloud()))  # no longer an artefact
    assert artifact_problems(tmp_path, data)[0].startswith("artefacts ['catalog.csv'")


def test_contract_log_counts_failed_checks() -> None:
    log = ContractLog()
    log.check("stdout", "segment", "a", [])
    log.check("stdout", "segment", "b", ["banner"])
    with pytest.raises(ValueError):
        log.check("stdout", "nobody", "c", [])
    res = {mid: (value, detail, error) for mid, value, detail, error in log.results()}
    assert res["contract.stdout.segment"][0] == 1
    assert res["contract.stdout.segment"][1] == {"checked": 2, "failed": {"b": ["banner"]}}
    value, _, error = res["contract.colour.view"]
    assert value is None and error is not None and "nothing was checked" in error
