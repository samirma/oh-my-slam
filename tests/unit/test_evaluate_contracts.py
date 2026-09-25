"""Contract checkers of the evaluator on synthetic artefacts: stdout purity, OpenLABEL validity and
the colour contract (JSON, segmented.png, catalogue, PLY)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.ply import PointCloud, ply_bytes
from oh_my_slam.schema import openlabel as ol
from oh_my_slam.segmentation.api import SceneObject
from oh_my_slam.segmentation.artifacts import write_artifacts
from oh_my_slam.segmentation.catalog import catalog_csv, catalog_md
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, color_hex_for_id
from oh_my_slam.segmentation.obb import OBB
from oh_my_slam.segmentation.render import contact_sheet, segmented_image
from oh_my_slam.segmentation.scene import objects_block
from oh_my_slam.tools.evaluate.contracts import (
    ContractLog,
    artifact_problems,
    catalog_csv_problems,
    catalog_md_problems,
    cloud_colour_problems,
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
from oh_my_slam.viewer.server import cloud_payload

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
        return cloud_payload(DisplayCloud(cloud, len(cloud), 1, 0.0), "color=segment")

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
    assert png_colour_problems(good, objs, sheet=False) == []
    blended = good.copy()
    mask = label_map() == 7
    blended[mask] = (0.6 * np.asarray(color_for_id(7)) + 0.4 * rgb[mask]).astype(np.uint8)
    problems = png_colour_problems(blended, objs, sheet=False)
    assert any("not object colours" in p for p in problems)
    missing = segmented_image(rgb, np.where(label_map() >= 7, 0, label_map()))
    assert png_colour_problems(missing, objs, sheet=False) == [
        "only 2 of 4 object colours appear in the image"]


def test_map_contact_sheet_headers_are_not_mask_colours() -> None:
    rgb = np.random.default_rng(2).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    lab = np.where(label_map() == 1, 1, 0)
    sheet = contact_sheet([("f000001", rgb, lab), ("f000002", rgb, np.zeros_like(lab))])
    assert png_colour_problems(sheet, doc_objects(scene_doc()), sheet=True) == []


def test_catalogue_colours() -> None:
    objs = doc_objects(scene_doc())
    csv_text, md_text = catalog_csv(objects()), catalog_md(objects())
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
    ply = ply_bytes(labelled_cloud())
    write_artifacts(tmp_path, data, segmented_image(np.zeros((60, 80, 3), np.uint8),
                                                    label_map()), objects(), ply, title="t")
    assert artifact_problems(tmp_path, data) == []
    assert artifact_problems(tmp_path, data + b" ")
    (tmp_path / "extra.txt").write_text("x")
    assert artifact_problems(tmp_path, data)


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
