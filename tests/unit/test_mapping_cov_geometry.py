"""Map geometry (``mapping.geometry``) at its edges: empty point sets and keyframes without
pixels to fuse, keyframes that cannot see a region, places of removed objects that nothing
witnesses, objects too small to take support, and the fused surface queried box by box."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oh_my_slam.core import rle
from oh_my_slam.core.images import save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import geometry as g
from oh_my_slam.mapping import store
from oh_my_slam.mapping.objects import MapObject, ObjectState, Vacated
from oh_my_slam.segmentation.api import OBB
from tests.synth.scene import Room, default_room, look_at, orbit_poses, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def _frame(i: int, pose: Pose, room: Room, valid: bool = True, update: int = 1) -> g.FrameData:
    r = render(room, pose, K)
    rec = store.FrameRecord(i, store.frame_name(i), f"frames/{store.frame_name(i)}.jpg", "", 1,
                            320, 240, K, pose, 320, 240, update_id=update)
    ok = (r.depth > 0) if valid else np.zeros(r.depth.shape, bool)
    return g.FrameData(rec, r.depth.astype(np.float32), ok, r.rgb,
                       np.where(r.ids >= 2, r.ids - 1, 0).astype(np.int32), False)


def _walls() -> list[g.FrameData]:
    """Keyframe 0 looks at the +x wall; keyframes 1-3 at the -x wall."""
    room = Room()
    poses = [look_at(np.array([0.5, 0.0, 1.3]), np.array([3.0, 0.0, 1.0]))]
    poses += [look_at(np.array([0.5, dy, 1.3]), np.array([-3.0, dy, 1.0])) for dy in (-0.1, 0, 0.1)]
    return [_frame(i, p, room) for i, p in enumerate(poses)]


def _blank(i: int, like: g.FrameData) -> g.FrameData:
    return g.FrameData(store.FrameRecord(i, store.frame_name(i), "", "", 1, 320, 240, K,
                                         like.rec.T_map_cam, 320, 240),
                       np.zeros_like(like.depth), np.zeros(like.depth.shape, bool), like.rgb,
                       np.zeros(like.depth.shape, np.int32), False)


def test_empty_cells_select_and_split_nothing() -> None:
    cells = g._Cells(np.zeros((0, 3)))
    assert cells.split(4) == []
    assert len(cells.select(_walls()[0])) == 0


def test_keyframes_that_no_other_keyframe_faces_keep_their_border_depth() -> None:
    frames = _walls()[:2]  # back to back: neither sees the other's surfaces
    before = [fd.depth.copy() for fd in frames]
    assert g.correct_borders(frames) == 0
    for fd, d in zip(frames, before, strict=True):
        np.testing.assert_array_equal(fd.depth, d)


def test_a_keyframe_without_valid_pixels_takes_no_consensus() -> None:
    a = _walls()[1]
    blank = _blank(9, a)
    depth = a.depth.copy()
    assert g.consensus_depths([a, blank]) == (0, 0)  # nothing vouched for by the other
    assert blank.drop is None
    np.testing.assert_array_equal(a.depth, depth)


def test_a_region_only_some_keyframes_reach_is_their_fusion_there() -> None:
    frames = _walls()
    whole = g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0)
    lo, hi = np.array([-3.2, -0.6, 0.6]), np.array([-2.6, 0.6, 1.6])  # the -x wall
    inside = whole[np.all((whole >= lo) & (whole <= hi), axis=1)]
    assert len(inside) > 500
    # keyframe 0 looks the other way: it cannot reach the box, and still counts as fused
    assert np.array_equal(g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0,
                                               region=(lo, hi)), inside)
    far = (np.array([20.0, 20.0, 20.0]), np.array([21.0, 21.0, 21.0]))  # nothing there
    assert g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0, region=far).shape == (0, 3)


def test_keyframes_without_pixels_to_fuse_change_nothing() -> None:
    frames = _walls()
    whole = g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0)
    with_blank = g.fused_cloud_points(frames + [_blank(7, frames[1])], voxel=0.02, depth_max=6.0)
    assert np.array_equal(whole, with_blank)
    assert g.fused_cloud_points([_blank(7, frames[1])], voxel=0.02, depth_max=6.0).shape == (0, 3)


def test_vacated_places_of_keyframes_not_held_or_masks_of_another_grid() -> None:
    frames = _walls()
    pts = g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0)
    by_name = {fd.rec.name: fd for fd in frames}
    full = np.ones((240, 320), bool)
    place = Vacated(1, 5, {"f000099": rle.encode(full),  # a keyframe it does not hold
                           frames[1].rec.name: rle.encode(np.ones((10, 10), bool)),  # other grid
                           frames[2].rec.name: rle.encode(np.zeros((240, 320), bool))},  # empty
                    [frames[3].rec.name])
    assert not g._vacated_region(pts, place, by_name).any()


def test_witnesses_that_see_none_of_the_points_draw_nothing() -> None:
    frames = _walls()
    behind = np.array([[3.0, 0.0, 1.0], [2.9, 0.2, 1.2]])  # behind keyframes 1-3
    drawn, left = g._witnessed(frames[1:], behind)
    assert not drawn.any() and not left.any()


def test_places_without_witnesses_in_the_fusion_are_left_as_fused() -> None:
    frames = _walls()
    pts = g.fused_cloud_points(frames, voxel=0.02, depth_max=6.0)
    mask = np.zeros((240, 320), bool)
    mask[80:160, 120:200] = True
    place = Vacated(1, 5, {frames[1].rec.name: rle.encode(mask)}, ["f000042"])
    keep, add = g._vacated(pts, pts[:10], frames, [place], frames)
    assert keep.all() and not add.any()
    plain = g.attribute_points(pts, frames)
    with_place = g.attribute_points(pts, frames, vacated=[place])
    for a, b in zip(plain, with_place, strict=True):
        np.testing.assert_array_equal(a, b)


def _plane(n: int, x0: float = 0.0) -> np.ndarray:
    gx, gy = np.meshgrid(np.linspace(x0, x0 + 0.4, n), np.linspace(0.0, 0.4, n))
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 1.0)])


def test_support_needs_enough_own_points_and_works_on_an_unsorted_cloud() -> None:
    box = OBB(np.array([0.2, 0.2, 1.0]), np.eye(3), np.array([0.5, 0.5, 0.1]))
    own = _plane(10).astype(np.float32)
    big = MapObject(1, "table", {"table": 1.0}, [0.9], own, box, confirmed=True, obs_depth=1.0)
    small = MapObject(2, "table", {"table": 1.0}, [0.9], own[:5], box, confirmed=True,
                      obs_depth=1.0)
    cloud = _plane(20)[::-1].copy()  # not sorted by x
    for objs, taken in (([big], 1), ([small], 0)):
        label = np.zeros(len(cloud), np.int32)
        assert g.support_labels(cloud, label, ObjectState(objs, 3)) == taken
        assert (label > 0).any() == bool(taken)
        assert set(np.unique(label)) <= {0, 1}


def test_the_surface_queried_in_a_box_is_the_whole_fusion_there(tmp_path: Path) -> None:
    room = default_room()
    records = []
    for i, pose in enumerate(orbit_poses(6)):
        fd = _frame(i, pose, room)
        (tmp_path / "per_frame" / fd.rec.name).mkdir(parents=True)
        np.save(tmp_path / store.frame_file(fd.rec.name, "depth.npy"), fd.depth)
        (tmp_path / "frames").mkdir(exist_ok=True)
        save_jpeg(fd.rgb, tmp_path / fd.rec.image)
        records.append(fd.rec)
    ctx = SimpleNamespace(new=[], tx=SimpleNamespace(current=lambda rel: tmp_path / rel))
    whole = g.fuse_map(ctx, records)
    assert len(whole.frames) == 6 and whole.vacated == []
    query = g.SurfaceQuery(ctx, records)
    for lo, hi in ((np.array([-0.5, -1.0, 0.2]), np.array([1.5, 0.7, 1.4])),
                   (np.array([-1.6, -1.6, -0.1]), np.array([0.0, 0.0, 0.6]))):
        inside = whole.xyz[np.all((whole.xyz >= lo) & (whole.xyz <= hi), axis=1)]
        assert len(inside) > 500
        assert np.array_equal(query(lo, hi), inside)
    assert query.calls == 2 and query.seconds > 0
    query.release()
    assert query._setup is None and query._blocks is None
