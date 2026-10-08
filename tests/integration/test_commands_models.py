"""Real-model runs of reconstruct.sh / segment.sh on the repository's reference images
(``-m models``): ``examples/restaurant.jpg`` (spec §5), the first ``examples/ainex-captures``
frame and a 1920x888 ``examples/camera`` frame; and mapper.sh update on overlapping captures of
both sequences. Timings are printed (``-s`` to see them).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from oh_my_slam.core.images import size_at_max_side, upright_size
from oh_my_slam.core.ply import parse_ply
from oh_my_slam.mapping.store import MapReader
from oh_my_slam.schema.validate import validation_errors
from oh_my_slam.segmentation.artifacts import ARTIFACT_NAMES
from oh_my_slam.segmentation.colors import hex_to_rgb

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(os.environ.get("OH_MY_SLAM_TEST_REAL_SERVER") != "1",
                       reason="set OH_MY_SLAM_TEST_REAL_SERVER=1 with a running server"),
]

REPO = Path(__file__).resolve().parents[2]
IMAGES = (REPO / "examples" / "restaurant.jpg",
          REPO / "examples" / "ainex-captures" / "001_bootstrap_level.jpg",
          REPO / "examples" / "camera" / "img_023_p08_mid.jpg")
# Overlapping captures of the two §5 sequences, two headings each, every heading with its level
# frame and its up and down tilts: (frames, name grammar → heading and tilt, the level tilt).
# Headings grow to the left: the pan position PP of examples/camera (p08, p09), and the commanded
# yaw of examples/ainex-captures (004–010: bootstrap_left015 and bootstrap_left030).
SEQUENCES = {
    "camera": (sorted((REPO / "examples" / "camera").glob("img_*_p0[89]_*.jpg")),
               r"_p(\d+)_(up|mid|down)\.jpg$", "mid"),
    "ainex": ([f for f in sorted((REPO / "examples" / "ainex-captures").glob("*.jpg"))
               if 4 <= int(f.name[:3]) <= 10],
              r"_left(\d+)(?:_side)?_(up|level|down)\.jpg$", "level"),
}


def run(script: str, *args: str) -> tuple[subprocess.CompletedProcess[bytes], float]:
    t0 = time.perf_counter()
    res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=300)
    return res, time.perf_counter() - t0


def rotations(doc: dict[str, Any], map_dir: Path) -> dict[str, np.ndarray]:
    """Camera-to-map rotation of each keyframe, by the file name of its input (``frames.json``
    source): its ``camera_<id>_to_map`` quaternion, scalar last (README)."""
    sources = {str(r.index): Path(r.source).name for r in MapReader(map_dir).frames}
    return {sources[k]: Rotation.from_quat(t["transform_src_to_dst"]["quaternion"]).as_matrix()
            for k, f in doc["openlabel"]["frames"].items()
            for t in f["frame_properties"]["transforms"].values()}


def direction(ref: np.ndarray, rot: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees of camera ``rot``'s optical axis in the axes of camera ``ref``
    (OpenCV: x right, y down, z forward): yaw positive to the left, pitch positive up. Relative
    to another camera, so neither the map's gauge nor its gravity alignment matters."""
    x, y, z = ref.T @ rot[:, 2]
    return float(np.degrees(np.arctan2(-x, z))), float(np.degrees(np.arctan2(-y, np.hypot(x, z))))


@pytest.mark.parametrize("img", IMAGES, ids=lambda p: p.name)
def test_reconstruct_and_segment(img: Path, tmp_path: Path) -> None:
    res, t_json = run("reconstruct.sh", "-i", str(img))
    assert res.returncode == 0, res.stderr.decode()
    doc = json.loads(res.stdout)
    assert validation_errors(doc) == []
    objs = doc["openlabel"]["objects"]
    assert objs, img
    # -f depth (spec §2.2): one 16-bit single-channel PNG with the pixel size of the input
    res, t_depth = run("reconstruct.sh", "-i", str(img), "-f", "depth")
    assert res.returncode == 0, res.stderr.decode()
    with Image.open(io.BytesIO(res.stdout)) as im:
        assert im.format == "PNG" and im.mode.startswith("I;16")  # 16-bit greyscale
        assert im.size == upright_size(img)
        depth = np.asarray(im)
    assert depth.dtype == np.uint16 and depth.ndim == 2 and depth.any()
    res, t_ply = run("reconstruct.sh", "-i", str(img), "-f", "ply")
    assert res.returncode == 0, res.stderr.decode()
    cloud = parse_ply(res.stdout)
    assert len(cloud) > 0
    out = tmp_path / img.stem
    res, t_seg = run("segment.sh", "-i", str(img), "-d", str(out))
    assert res.returncode == 0, res.stderr.decode()
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_NAMES)  # the four of §2.4
    assert (out / "segmentation.json").read_bytes() == res.stdout
    scene = json.loads(res.stdout)
    assert validation_errors(scene) == []
    seg = scene["openlabel"]["objects"]
    # colour contract (spec §2.4): one sRGB triple per object in the JSON, catalog.csv and its
    # opaque mask in segmented.png, the image on the reconstruction's grid (README: long side 1024)
    colours = {int(k): tuple(o["object_data"]["vec"][0]["val"]) for k, o in seg.items()}
    assert colours == {int(k): hex_to_rgb(o["object_data"]["text"][0]["val"])
                       for k, o in seg.items()}
    with (out / "catalog.csv").open() as f:
        assert {int(r["id"]): hex_to_rgb(r["color_hex"]) for r in csv.DictReader(f)} == colours
    with Image.open(out / "segmented.png") as im:
        assert im.mode == "RGB" and im.size == size_at_max_side(*upright_size(img), 1024)
        painted = np.asarray(im)
    painted = painted[painted.max(axis=-1) > 0.35 * 255]  # brighter than the dimmed image (README)
    assert {tuple(c) for c in np.unique(painted, axis=0).tolist()} == set(colours.values())
    print(f"{img.name}: json {t_json:.2f}s ({len(objs)} objects), depth {t_depth:.2f}s, "
          f"ply {t_ply:.2f}s ({len(cloud)} pts), segment -d {t_seg:.2f}s ({len(seg)} objects)")


@pytest.mark.parametrize("seq", SEQUENCES)
def test_mapper_update(seq: str, tmp_path: Path) -> None:
    """mapper.sh update (spec §2.3) creates a map from overlapping captures (camera: 1920x888
    frames of a real pan-tilt camera through a wide-angle lens with visible distortion; ainex:
    rendered robot head), poses every one of them, and in the directions their names encode (§5
    pose accuracy)."""
    frames, grammar, level = SEQUENCES[seq]
    out = tmp_path / "map"
    res, t_map = run("mapper.sh", "update", "-i", *(str(f) for f in frames), "-m", str(out))
    assert res.returncode == 0, res.stderr.decode()
    doc = json.loads(res.stdout)
    assert validation_errors(doc) == []
    objs = doc["openlabel"]["objects"]
    assert objs
    rot = rotations(doc, out)
    assert len(frames) >= 6 and sorted(rot) == [f.name for f in frames]  # -t full: every pose
    names: dict[str, tuple[int, str]] = {}
    for f in frames:
        m = re.search(grammar, f.name)
        assert m, f.name
        names[f.name] = int(m[1]), m[2]
    ref: dict[int, str] = {}  # each heading's (first) level frame
    for n, (h, tilt) in names.items():
        if tilt == level:
            ref.setdefault(h, n)
    lo, hi = sorted(ref)
    turn = {n: direction(rot[ref[lo]], rot[n])[0] for n, (h, _) in names.items() if h == hi}
    step = turn[ref[hi]]
    own = {n: direction(rot[ref[h]], rot[n]) for n, (h, _) in names.items() if n != ref[h]}
    pitch = {n: own[n][1] for n, (_, tilt) in names.items() if tilt in ("up", "down")}
    print(f"{seq}: update {t_map:.2f}s ({len(rot)}/{len(frames)} frames posed, {len(objs)} "
          f"objects), step {step:.1f}°, yaw/pitch off the level frame "
          + ", ".join(f"{n} {y:+.1f}°/{p:+.1f}°" for n, (y, p) in own.items()))
    # the higher heading is to the left of the lower one, whatever the tilt
    assert all(y > 0 for y in turn.values()), turn
    # every frame of a heading shares its yaw: less than half the step between the two headings
    # off its level frame, i.e. nearer in yaw to its own heading than to the other one
    assert all(abs(y) < step / 2 for y, _ in own.values()), (step, own)
    # pitch direction: up looks image-up of its level frame, down image-down
    assert all(p > 0 if names[n][1] == "up" else p < 0 for n, p in pitch.items()), pitch
