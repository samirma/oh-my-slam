"""The shared cloud derivation (segmentation.cloud): the effect of every point-cloud attribute on
an image source and a map source, determinism, the colour contract and the PLY header."""

from __future__ import annotations

import numpy as np
import pytest

from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope, parse_cloud_attrs
from oh_my_slam.core.geometry import voxel_downsample_indices, voxel_keys
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.core.types import Intrinsics
from oh_my_slam.reconstruction.pointcloud import PointNormals, pixel_mask
from oh_my_slam.segmentation.cloud import (
    ImageCloudSource,
    MapCloudSource,
    cloud_ply,
    derive_cloud,
    derive_thinned,
    map_cloud_source,
)
from oh_my_slam.segmentation.colors import UNSEGMENTED, color_for_id, height_colors
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


@pytest.fixture(scope="module")
def image() -> ImageCloudSource:
    """The synthetic room; boxes k = 0..2 are objects 1..3, and a 40 x 30 hole of invalid depth."""
    pose = look_at(np.array([2.6, 2.2, 1.6]), np.array([0.0, 0.0, 0.4]))
    r = render(default_room(), pose, K)
    depth = r.depth.copy()
    depth[100:130, 40:80] = 0.0
    labels = np.where(r.ids >= 2, r.ids - 1, 0).astype(np.int32)
    return ImageCloudSource(depth, depth > 0, r.rgb, K, labels,
                            pose.R.T @ np.array([0.0, 0.0, 1.0]))


def _pixels(cloud_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of camera-frame points on the grid."""
    p = cloud_xyz.astype(np.float64)
    return (np.rint(K.fx * p[:, 0] / p[:, 2] + K.cx).astype(int),
            np.rint(K.fy * p[:, 1] / p[:, 2] + K.cy).astype(int))


def test_default_cloud_is_the_default_pixel_selection(image: ImageCloudSource) -> None:
    cloud = derive_cloud(image, CloudAttrs())
    mask = pixel_mask(image.depth, image.valid)
    assert len(cloud) == int(mask.sum()) and cloud.label is None and cloud.normals is None
    u, v = _pixels(cloud.xyz)
    assert mask[v, u].all()
    np.testing.assert_array_equal(cloud.rgb, image.rgb[v, u])  # colour = the image colour
    np.testing.assert_allclose(cloud.xyz[:, 2], image.depth[v, u], rtol=1e-6)


def test_stride_keeps_every_nth_pixel(image: ImageCloudSource) -> None:
    full = pixel_mask(image.depth, image.valid)
    for n in (2, 5):
        cloud = derive_cloud(image, CloudAttrs(stride=n))
        u, v = _pixels(cloud.xyz)
        assert (u % n == 0).all() and (v % n == 0).all()
        assert len(cloud) == int(full[::n, ::n].sum())


def test_depth_range(image: ImageCloudSource) -> None:
    cloud = derive_cloud(image, CloudAttrs(min_depth=2.5, max_depth=3.5))
    z = cloud.xyz[:, 2]
    assert len(cloud) > 1000 and z.min() >= 2.5 - 1e-6 and z.max() <= 3.5 + 1e-6
    full = derive_cloud(image, CloudAttrs()).xyz[:, 2]
    assert len(cloud) == int(((full >= 2.5) & (full <= 3.5)).sum())


def test_edge_filter_and_zero_disables_it(image: ImageCloudSource) -> None:
    valid = int(image.valid.sum())
    off = derive_cloud(image, CloudAttrs(edge=0))
    assert len(off) == valid  # every valid pixel, also those next to the invalid hole
    u, v = _pixels(off.xyz)
    assert ((v == 99) & (u >= 40) & (u < 80)).sum() == 40  # the row touching the hole is kept
    default = len(derive_cloud(image, CloudAttrs()))
    loose = len(derive_cloud(image, CloudAttrs(edge=0.5)))
    strict = len(derive_cloud(image, CloudAttrs(edge=0.005)))
    assert strict < default < loose < valid  # loose still drops pixels next to the hole


def test_voxel_one_exact_point_per_voxel_deterministic(image: ImageCloudSource) -> None:
    attrs = CloudAttrs(color="segment", label=True, voxel=0.05)
    a = derive_cloud(image, attrs)
    b = derive_cloud(image, attrs)
    np.testing.assert_array_equal(a.xyz, b.xyz)
    np.testing.assert_array_equal(a.rgb, b.rgb)
    full = derive_cloud(image, CloudAttrs(color="segment", label=True))
    assert 0 < len(a) < len(full) / 5
    keys = voxel_keys(a.xyz.astype(np.float64), 0.05)
    assert len(np.unique(keys, axis=0)) == len(a)  # one point per voxel
    assert len(np.unique(voxel_keys(full.xyz.astype(np.float64), 0.05), axis=0)) == len(a)
    # every kept point is an original point with its own colour and label (nothing averaged)
    rows = {tuple(r) for r in np.concatenate([full.xyz.view(np.int32), full.rgb.astype(np.int32),
                                              full.label[:, None]], axis=1)}
    kept = np.concatenate([a.xyz.view(np.int32), a.rgb.astype(np.int32), a.label[:, None]], 1)
    assert all(tuple(r) in rows for r in kept)
    rgb_image = derive_cloud(image, CloudAttrs(voxel=0.05)).rgb
    u, v = _pixels(a.xyz)
    np.testing.assert_array_equal(rgb_image, image.rgb[v, u])


def test_label_property(image: ImageCloudSource) -> None:
    on = derive_cloud(image, CloudAttrs(label=True))
    assert on.label is not None
    u, v = _pixels(on.xyz)
    np.testing.assert_array_equal(on.label, image.labels[v, u])
    assert set(np.unique(on.label)) == {0, 1, 2, 3}
    assert derive_cloud(image, CloudAttrs()).label is None


def test_colour_modes(image: ImageCloudSource) -> None:
    seg = derive_cloud(image, CloudAttrs(color="segment", label=True))
    assert seg.rgb is not None and seg.label is not None
    for oid in (1, 2, 3):  # the colour contract: exactly the object's sRGB triple
        np.testing.assert_array_equal(np.unique(seg.rgb[seg.label == oid], axis=0),
                                      [color_for_id(oid)])
    np.testing.assert_array_equal(np.unique(seg.rgb[seg.label == 0], axis=0), [UNSEGMENTED])
    height = derive_cloud(image, CloudAttrs(color="height"))
    h = height.xyz.astype(np.float64) @ image.up
    np.testing.assert_array_equal(height.rgb, height_colors(h))
    low, high = height.rgb[np.argmin(h)].astype(int), height.rgb[np.argmax(h)].astype(int)
    assert tuple(low) == (0x44, 0x01, 0x54) and tuple(high) == (0xfd, 0xe7, 0x25)  # viridis ends
    none = derive_cloud(image, CloudAttrs(color="none"))
    assert none.rgb is None and len(none) == len(height)


def test_missing_inputs_are_programming_errors(image: ImageCloudSource) -> None:
    bare = ImageCloudSource(image.depth, image.valid, image.rgb, K)
    with pytest.raises(ValueError):
        derive_cloud(bare, CloudAttrs(color="segment"))
    with pytest.raises(ValueError):
        derive_cloud(bare, CloudAttrs(label=True))
    with pytest.raises(ValueError):
        derive_cloud(bare, CloudAttrs(color="height"))
    assert len(derive_cloud(bare, CloudAttrs())) == len(derive_cloud(image, CloudAttrs()))


def test_image_normals_from_depth() -> None:
    """A tilted plane: every normal is the plane normal, unit length, facing the camera."""
    n_true = np.array([0.0, -0.6, -0.8])  # plane n·X = -2 (in front of the camera, tilted)
    v, u = np.mgrid[0:K.height, 0:K.width]
    rays = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones(u.shape)], -1)
    depth = (-2.0 / (rays @ n_true)).astype(np.float32)
    src = ImageCloudSource(depth, depth > 0, np.zeros((*depth.shape, 3), np.uint8), K)
    cloud = derive_cloud(src, CloudAttrs(normals=True, edge=0))
    assert cloud.normals is not None and len(cloud) == depth.size
    np.testing.assert_allclose(np.linalg.norm(cloud.normals, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(cloud.normals, np.tile(n_true, (len(cloud), 1)), atol=2e-3)
    assert ((cloud.normals * cloud.xyz).sum(1) < 0).all()


def test_image_normals_on_the_room(image: ImageCloudSource) -> None:
    cloud = derive_cloud(image, CloudAttrs(normals=True, label=True, stride=2))
    assert cloud.normals is not None and cloud.label is not None
    np.testing.assert_allclose(np.linalg.norm(cloud.normals, axis=1), 1.0, atol=1e-5)
    assert ((cloud.normals * cloud.xyz).sum(1) <= 0).all()  # towards the camera
    floor = (cloud.label == 0) & (np.abs(cloud.xyz.astype(np.float64) @ image.up + 1.6) < 0.01)
    assert floor.sum() > 3000
    assert (cloud.normals[floor] @ image.up > 0.99).mean() > 0.95  # the floor faces up
    again = derive_cloud(image, CloudAttrs(normals=True, label=True, stride=2))
    np.testing.assert_array_equal(cloud.normals, again.normals)


@pytest.fixture(scope="module")
def plane_map() -> MapCloudSource:
    rng = np.random.default_rng(5)
    xy = rng.uniform(-1, 1, (4000, 2))
    xyz = np.concatenate([xy, rng.normal(0, 0.002, (4000, 1))], axis=1)
    labels = np.where(xy[:, 0] > 0.5, 7, np.where(xy[:, 0] < -0.5, 99, 0))
    rgb = rng.integers(0, 256, (4000, 3)).astype(np.uint8)
    return map_cloud_source(xyz, rgb, labels, {7}, np.array([[0.0, 0.0, 1.5]]))


def test_map_source_voxel_colour_label_and_normals(plane_map: MapCloudSource) -> None:
    cloud = derive_cloud(plane_map, CloudAttrs(color="segment", label=True, normals=True))
    assert len(cloud) == 4000 and cloud.label is not None and cloud.normals is not None
    assert set(np.unique(cloud.label)) == {0, 7}  # 99 is not an exported object
    np.testing.assert_array_equal(cloud.rgb[cloud.label == 7], [color_for_id(7)] * int(
        (cloud.label == 7).sum()))
    assert (cloud.normals[:, 2] > 0.95).mean() > 0.99  # towards the camera above the plane
    below = map_cloud_source(plane_map.xyz, plane_map.rgb, plane_map.labels, {7},
                             np.array([[0.0, 0.0, -1.5]]))
    assert (derive_cloud(below, CloudAttrs(normals=True)).normals[:, 2] < -0.95).mean() > 0.99
    thin = derive_cloud(plane_map, CloudAttrs(voxel=0.2))
    assert len(thin) == len(np.unique(voxel_keys(plane_map.xyz, 0.2), axis=0))
    idx = np.array([np.flatnonzero((plane_map.xyz.astype(np.float32) == p).all(1))[0]
                    for p in thin.xyz])
    assert (np.diff(idx) > 0).all()  # the first point of each voxel, in storage order
    np.testing.assert_array_equal(thin.rgb, plane_map.rgb[idx])  # exact colours
    np.testing.assert_array_equal(derive_cloud(plane_map, CloudAttrs(voxel=0.2)).xyz, thin.xyz)
    h = derive_cloud(plane_map, CloudAttrs(color="height"))
    np.testing.assert_array_equal(h.rgb, height_colors(plane_map.xyz[:, 2]))  # map up is z


def test_ply_header_records_the_effective_attributes(image: ImageCloudSource,
                                                     plane_map: MapCloudSource) -> None:
    attrs = parse_cloud_attrs("color=segment,voxel=0.05,normals=on,encoding=ascii",
                              CloudScope.IMAGE)
    data = cloud_ply(image, attrs)
    head = parse_header(data)
    assert head.encoding == "ascii" and head.comments[0].startswith("oh-my-slam camera frame")
    assert head.comments[1] == ("attributes color=segment,stride=1,min-depth=0,max-depth=inf,"
                                "edge=0.04,voxel=0.05,normals=on,label=off,encoding=ascii")
    back = parse_ply(data)
    ref = derive_cloud(image, attrs)
    np.testing.assert_array_equal(back.xyz, ref.xyz)
    np.testing.assert_array_equal(back.rgb, ref.rgb)
    np.testing.assert_array_equal(back.normals, ref.normals)
    assert back.label is None
    mhead = parse_header(cloud_ply(plane_map, CloudAttrs(color="none")))
    assert mhead.comments == ["oh-my-slam map frame (z up), metres",
                              "attributes color=none,voxel=0,normals=off,label=off,"
                              "encoding=binary"]
    assert [n for n, _ in mhead.fields] == ["x", "y", "z"]
    assert cloud_ply(image, attrs) == data  # byte-identical on every run


@pytest.fixture()
def room_map() -> MapCloudSource:
    """Three walls and a floor, 60 000 noisy points (several normal chunks), with objects."""
    rng = np.random.default_rng(11)
    n = 15_000
    floor = np.c_[rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.normal(0, 0.003, n)]
    wall_x = np.c_[np.full(n, 2.0) + rng.normal(0, 0.003, n), rng.uniform(-2, 2, n),
                   rng.uniform(0, 2.5, n)]
    wall_y = np.c_[rng.uniform(-2, 2, n), np.full(n, -2.0) + rng.normal(0, 0.003, n),
                   rng.uniform(0, 2.5, n)]
    box = np.c_[rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n), rng.uniform(0, 0.6, n)]
    xyz = np.vstack([floor, wall_x, wall_y, box])
    labels = np.r_[np.zeros(3 * n, np.int32), np.full(n, 4, np.int32)]
    rgb = rng.integers(0, 256, (len(xyz), 3)).astype(np.uint8)
    return map_cloud_source(xyz, rgb, labels, {4}, np.array([[0.0, 0.5, 1.2], [0.2, 0.0, 1.2]]))


def all_normals(xyz: np.ndarray, viewpoints: np.ndarray) -> np.ndarray:
    """Normals of every point of the cloud, asked for at once, in order."""
    return PointNormals(xyz, viewpoints).at(np.arange(len(xyz)))


def test_map_normals_only_for_the_emitted_points(room_map: MapCloudSource) -> None:
    """Normals are computed for the points a derivation emits, and a point's normal does not
    depend on ``voxel``, on which other points are asked for, or on their order (the same values
    ``segment.sh -m`` / ``mapper.sh -f ply`` / the viewer emit)."""
    thin = derive_cloud(room_map, CloudAttrs(voxel=0.1, normals=True))
    assert room_map.normals._done.sum() == len(thin) < len(room_map.xyz) / 5
    full = derive_cloud(room_map, CloudAttrs(normals=True))
    assert room_map.normals._done.all()
    keep = voxel_downsample_indices(np.asarray(room_map.xyz, np.float64), 0.1, keep="first")
    np.testing.assert_array_equal(thin.normals, full.normals[keep])
    # a fresh source asked in another order, and the whole-cloud function, agree bit for bit
    rows = np.random.default_rng(3).permutation(len(room_map.xyz))[:5000]
    fresh = PointNormals(room_map.xyz, room_map.viewpoints)
    np.testing.assert_array_equal(fresh.at(rows[::-1])[::-1], full.normals[rows])
    np.testing.assert_array_equal(all_normals(room_map.xyz, room_map.viewpoints), full.normals)
    # correct geometry away from edges and the box: floor up, walls facing the cameras
    x, y, z = np.asarray(room_map.xyz).T
    part = np.repeat(np.arange(4), 15_000)
    floor = (part == 0) & (np.abs(x) < 1.8) & (np.abs(y) < 1.8) & (np.hypot(x, y) > 0.7)
    wall_x = (part == 1) & (np.abs(y) < 1.8) & (z > 0.2) & (z < 2.3)
    wall_y = (part == 2) & (np.abs(x) < 1.8) & (z > 0.2) & (z < 2.3)
    for m, axis, sign in ((floor, 2, 1), (wall_x, 0, -1), (wall_y, 1, 1)):
        assert m.sum() > 5000 and (sign * full.normals[m, axis] > 0.98).mean() > 0.99
    np.testing.assert_allclose(np.linalg.norm(full.normals, axis=1), 1.0, atol=1e-5)


def test_derive_thinned_is_a_subset_of_derive_cloud(room_map: MapCloudSource) -> None:
    for attrs in (CloudAttrs(color="height", normals=True, label=True),
                  CloudAttrs(color="segment", voxel=0.05, normals=True, label=True)):
        src = map_cloud_source(room_map.xyz, room_map.rgb, room_map.labels, {4},
                               room_map.viewpoints)
        t = derive_thinned(src, attrs, 7_000)
        assert src.normals._done.sum() == len(t.cloud)  # normals only for the kept points
        full = derive_cloud(src, attrs)
        assert t.total == len(full) and t.step == -(-len(full) // 7_000) > 1
        sel = np.arange(0, len(full), t.step)
        assert len(t.cloud) == len(sel) <= 7_000
        for name in ("xyz", "rgb", "label", "normals"):
            np.testing.assert_array_equal(getattr(t.cloud, name), getattr(full, name)[sel])
    t = derive_thinned(room_map, CloudAttrs(), None)
    assert (t.total, t.step, len(t.cloud)) == (len(room_map.xyz), 1, len(room_map.xyz))


def test_complete_map_cloud_shares_the_source_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """A map cloud with every point costs no copy of the map: positions, colours and labels are
    read-only views of the source's arrays; any selection of points copies (and colours are
    derived per chunk with exactly the values of a single pass)."""
    import oh_my_slam.segmentation.cloud as sc
    import oh_my_slam.segmentation.colors as colors

    monkeypatch.setattr(sc, "CHUNK_POINTS", 7)
    monkeypatch.setattr(colors, "SEGMENT_CHUNK", 5)
    rng = np.random.default_rng(2)
    xyz = rng.normal(size=(40, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (40, 3)).astype(np.uint8)
    given = rng.integers(0, 6, 40)
    src = map_cloud_source(xyz, rgb, given, {2, 5}, np.zeros((1, 3)))
    assert src.labels is not None
    assert src.labels.tolist() == np.where(np.isin(given, [2, 5]), given, 0).tolist()
    cloud = derive_cloud(src, CloudAttrs(label=True))
    assert cloud.rgb is not None and cloud.label is not None
    for mine, theirs in ((cloud.xyz, xyz), (cloud.rgb, rgb), (cloud.label, src.labels)):
        assert np.shares_memory(mine, theirs) and not mine.flags.writeable
    assert xyz.flags.writeable and rgb.flags.writeable  # the caller's arrays are untouched
    seg = derive_cloud(src, CloudAttrs(color="segment", label=True))
    assert seg.rgb is not None and not np.shares_memory(seg.rgb, rgb)
    expected = [UNSEGMENTED if i == 0 else color_for_id(int(i)) for i in src.labels]
    assert [tuple(c) for c in seg.rgb] == expected
    thin = derive_cloud(src, CloudAttrs(voxel=0.5, label=True))
    assert not np.shares_memory(thin.xyz, xyz) and thin.xyz.flags.writeable
    h = derive_cloud(src, CloudAttrs(color="height"))
    np.testing.assert_array_equal(h.rgb, height_colors(xyz[:, 2].astype(np.float64)))
