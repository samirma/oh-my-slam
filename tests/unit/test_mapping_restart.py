"""§2.3 "the result is the same whether a sequence is mapped in one update or split across several in
the same order", for a split whose first update is weakly posed: real COLMAP on rendered views,
fake inference.

The first update is four views panned from one spot towards a wall whose middle is a window: the
scenery seen through it carries the texture of the far wall, while monocular depth places it on
the pane, 30 % nearer (as `office_sequence`'s trees behind the glass). Panned from one spot, the
four views triangulate no metric scale and are posed by the multi-view fallback, without SfM
support. An update that only extended them would freeze those poses: the map is rebuilt with the
next update's views instead, as one update of them all (``mapping.api._restart_weak_map``), and
ends as the one-update map does: the same objects, ids, labels and boxes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.types import Pose
from oh_my_slam.mapping import api
from oh_my_slam.mapping.api import update
from oh_my_slam.segmentation.obb import obb_iou_upright
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.mapping import K, mapping_room, ring
from tests.synth.scene import Room, look_at, render
from tests.unit.test_mapping_e2e_synth import cuboid_obb, objects_by_label

pytestmark = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs Homebrew colmap")

PANE = 0.7  # monocular depth of the scenery behind the window: on the pane, 30 % nearer


def _scenery_on_the_pane(depth: np.ndarray, pose: Pose) -> np.ndarray:
    """``depth`` with the window in the wall at x = -3 m (|y| < 1 m, 0.6-1.9 m high) placed
    ``PANE`` times nearer."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(depth)], axis=-1)
    pts = (rays * depth[..., None]) @ pose.R.T + pose.t
    window = ((np.abs(pts[..., 0] + 3.0) < 0.03) & (np.abs(pts[..., 1]) < 1.0)
              & (pts[..., 2] > 0.6) & (pts[..., 2] < 1.9) & (depth > 0))
    out = depth.copy()
    out[window] *= PANE
    return out


def _add(client: FakeClient, room: Room, poses: list[Pose], folder: Path, prefix: str
         ) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, pose in enumerate(poses):
        r = render(room, pose, K)
        inst = [FakeInstance(box.label, 0.85, r.ids == k + 2)
                for k, box in enumerate(room.boxes) if (r.ids == k + 2).sum() > 150]
        up = pose.R.T @ np.array([0.0, 0.0, 1.0])
        path = folder / f"{prefix}_{i:03d}.png"
        depth = _scenery_on_the_pane(r.depth, pose).astype(np.float32)
        client.add(path, r.rgb, FakeFrame(depth, K, up, inst, pose=pose))
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def views(tmp_path_factory: pytest.TempPathFactory) -> tuple[FakeClient, list[Path], list[Path]]:
    base = tmp_path_factory.mktemp("restart")
    client = FakeClient(mv_noise=(1.0, 0.03))
    room = mapping_room()
    eye = np.array([2.3, 0.2, 1.5])
    pan = [look_at(eye, np.array([-3.0, y, 0.9])) for y in (-0.9, -0.3, 0.3, 0.9)]
    first = _add(client, room, pan, base / "a", "a")
    rest = _add(client, room, ring(12, start=0.2), base / "b", "b")
    return client, first, rest


def _quiet(msg: str) -> None:
    pass


def test_a_weak_first_update_is_rebuilt_with_the_next(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path) -> None:
    client, first, rest = views
    one = json.loads(update(tmp_path / "one", first + rest, client=client,
                            progress=_quiet).payload)
    split = tmp_path / "split"
    update(split, first, client=client, progress=_quiet)
    frames = json.loads((split / "frames.json").read_text())["frames"]
    assert all(api._weak_keyframe(api.store.FrameRecord.from_dict(f)) for f in frames)
    two = json.loads(update(split, rest, client=client, progress=_quiet).payload)
    meta = json.loads((split / "map.json").read_text())
    assert meta["updates"][-1]["notes"]["restarted"] == {"stored_keyframes": len(first)}
    assert meta["updates"][0]["frames_added"] == [f"f{k:06d}" for k in range(4)]
    a, b = objects_by_label(one), objects_by_label(two)
    assert sorted(a) == sorted(b) == ["box", "cabinet", "sofa"], (a.keys(), b.keys())
    for label in a:
        (oa,), (ob,) = a[label], b[label]
        assert oa["id"] == ob["id"], label
        iou = obb_iou_upright(cuboid_obb(oa), cuboid_obb(ob))
        centre = np.linalg.norm(cuboid_obb(oa).center - cuboid_obb(ob).center)
        assert iou > 0.7 and centre < 0.05, (label, iou, centre)
    # nothing of the first map's derived files is left behind
    ids = {o["id"] for o in json.loads((split / "objects.json").read_text())["objects"]}
    files = {int(p.stem.split("_")[1]) for p in (split / "objects").glob("*.npy")}
    assert files <= ids


def test_a_map_posed_by_sfm_is_extended(
        views: tuple[FakeClient, list[Path], list[Path]], tmp_path: Path) -> None:
    """The ring's views are posed by SfM: the next update extends them, held fixed."""
    client, first, rest = views
    m = tmp_path / "m"
    update(m, rest, client=client, progress=_quiet)
    update(m, first, client=client, progress=_quiet)
    notes = json.loads((m / "map.json").read_text())["updates"][-1]["notes"]
    assert "restarted" not in notes
