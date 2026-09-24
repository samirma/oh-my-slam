"""Real-model runs of reconstruct.sh / segment.sh on three images (``-m models``).

The images come from the user's validation inputs (no CC0 download needed): restaurant.jpg, one
ainex capture and one living-room video frame. Timings are printed (``-s`` to see them).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from oh_my_slam.core.ply import parse_ply
from oh_my_slam.schema.validate import validation_errors

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(os.environ.get("OH_MY_SLAM_TEST_REAL_SERVER") != "1",
                       reason="set OH_MY_SLAM_TEST_REAL_SERVER=1 with a running server"),
]

REPO = Path(__file__).resolve().parents[2]
INPUTS = Path("/Users/U124317/robot_view")


def _frame_from_video(tmp: Path) -> Path:
    out = tmp / "livingroom_t20.jpg"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "20", "-i",
                    str(INPUTS / "livingroom.mp4"), "-frames:v", "1", "-q:v", "2", str(out)],
                   check=True)
    return out


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    if not INPUTS.exists():
        pytest.skip("validation inputs missing")
    tmp = tmp_path_factory.mktemp("frames")
    return [INPUTS / "restaurant.jpg",
            INPUTS / "ainex-captures" / "001_bootstrap_level.jpg",
            _frame_from_video(tmp)]


def run(script: str, *args: str) -> tuple[subprocess.CompletedProcess[bytes], float]:
    t0 = time.perf_counter()
    res = subprocess.run([str(REPO / script), *args], capture_output=True, timeout=300)
    return res, time.perf_counter() - t0


def test_reconstruct_and_segment_on_three_images(images: list[Path], tmp_path: Path) -> None:
    for img in images:
        res, t_json = run("reconstruct.sh", "-i", str(img))
        assert res.returncode == 0, res.stderr.decode()
        doc = json.loads(res.stdout)
        assert validation_errors(doc) == []
        objs = doc["openlabel"]["objects"]
        assert objs, img
        res, t_ply = run("reconstruct.sh", "-i", str(img), "-f", "ply")
        assert res.returncode == 0
        cloud = parse_ply(res.stdout)
        out = tmp_path / img.stem
        res, t_seg = run("segment.sh", "-i", str(img), "-d", str(out))
        assert res.returncode == 0
        assert len(list(out.iterdir())) == 5
        assert (out / "segmentation.json").read_bytes() == res.stdout
        print(f"{img.name}: json {t_json:.2f}s ({len(objs)} objects), ply {t_ply:.2f}s "
              f"({len(cloud)} pts), segment -d {t_seg:.2f}s")
