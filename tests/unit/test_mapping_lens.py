"""The keyframes of a camera with lens distortion are reconstructed on their undistorted images (the
lens known from the map before inference, or found by a new map's multi-view fit and then
re-run) and detected on their own images, their detections moved onto the undistorted grid;
they are stored with the lens in their intrinsics and their depth on the pinhole's grid; densely
aligned depth scales never store a failed fit, and are fitted again against neighbours on either
side."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.images import save_jpeg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.mapping import api, ingest, store
from oh_my_slam.reconstruction import depth as rdepth
from oh_my_slam.reconstruction.depth import ScaleFit
from oh_my_slam.segmentation.api import Detection
from tests.unit.test_mapping_cov_api import FakeModel, K, ctx_of, jpg, line, nf

LENS = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap", -0.3)


def _image(path: Path, size: tuple[int, int] = (640, 480)) -> Path:
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    rgb[:, ::20] = 255
    save_jpeg(rgb, path)
    return path


def _stored(index: int, K_: Intrinsics, size: tuple[int, int] = (640, 480)) -> store.FrameRecord:
    return store.FrameRecord(index, f"f{index:06d}", f"frames/f{index:06d}.jpg", "", 1, *size,
                             K_, Pose.identity(), 64, 48)


def test_an_update_knows_the_lens_of_the_camera_its_keyframes_share(tmp_path: Path) -> None:
    lens_of = api._lens_of([_stored(0, K), _stored(1, LENS), _stored(2, K, (800, 600))])
    kf = ingest.Keyframe("f000003", 3, _image(tmp_path / "a.jpg"), "a.jpg", None)
    assert lens_of(kf) == LENS  # the latest stored keyframe of its size
    assert lens_of(replace(kf, exif=K)) is None  # photos with EXIF get a camera of their own
    assert lens_of(replace(kf, path=_image(tmp_path / "b.jpg", (800, 600)))) is None  # pinhole
    assert lens_of(replace(kf, path=_image(tmp_path / "c.jpg", (320, 240)))) is None  # no camera
    assert api._lens_of([])(kf) is None


class _Client:
    """An inference client whose clones are connections of their own, closed after use."""

    def __init__(self) -> None:
        self.closed = 0

    def clone(self) -> _Client:
        return _Client()

    def close(self) -> None:
        self.closed += 1


def test_keyframes_of_a_known_lens_are_inferred_undistorted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, tuple[Path, Any]] = {}

    def infer(kf: ingest.Keyframe, work: Path, client: Any, lens: Any = None
              ) -> tuple[Any, list[Any]]:
        seen[kf.name] = (kf.path, lens)
        return SimpleNamespace(), []

    monkeypatch.setattr(api, "reconstruct_and_detect_keyframe", infer)
    kfs = [ingest.Keyframe(f"f{i:06d}", i, _image(tmp_path / f"in{i}.jpg"), f"in{i}.jpg", None)
           for i in range(2)]
    new = api._infer_frames(iter(kfs), "images", tmp_path / "work", None, lambda m: None,
                            lambda written: None, lambda kf: LENS if kf.index == 0 else None)
    assert seen == {"f000000": (kfs[0].path, LENS), "f000001": (kfs[1].path, None)}
    assert [n.lens for n in new] == [LENS, None] and new[0].kf is kfs[0]


def _grid_frame(**kw: Any) -> SimpleNamespace:
    """A keyframe's reconstruction on a 64 x 48 grid: a plain image, depth 2 m everywhere."""
    return SimpleNamespace(grid_size=(64, 48), depth=np.full((48, 64), 2.0, np.float32),
                           valid=np.ones((48, 64), bool), rgb=np.zeros((48, 64, 3), np.uint8),
                           **kw)


def _raw_detection(shape: tuple[int, int] = (48, 64)) -> Detection:
    """A detection on a grid over the keyframe's own image: a block around its centre."""
    mask = np.zeros(shape, bool)
    h, w = shape
    mask[h // 2 - 4:h // 2 + 4, w // 2 - 6:w // 2 + 6] = True
    return Detection("box", 0.8, "fine", mask, (w / 2 - 6, h / 2 - 4, w / 2 + 6, h / 2 + 4))


def test_a_lens_keyframe_is_reconstructed_undistorted_and_detected_as_the_camera_saw_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def reconstruct(path: Path, client: Any, intrinsics: Any, work: Path) -> Any:
        calls["reconstruct"] = (path, intrinsics)
        return _grid_frame()

    def alongside(path: Path, client: Any, rec: Any, *, max_side: int) -> Any:
        calls["detect"] = path
        return rec(client), [_raw_detection()]

    monkeypatch.setattr(api, "_reconstruct_keyframe", reconstruct)
    monkeypatch.setattr(api, "detect_alongside", alongside)
    kf = ingest.Keyframe("f000000", 0, _image(tmp_path / "in.jpg"), "in.jpg", None)
    frame, dets = api.reconstruct_and_detect_keyframe(kf, tmp_path / "w", _Client(), LENS)
    assert calls["detect"] == kf.path  # the detector sees the camera's own image
    assert calls["reconstruct"][0].name == "undistorted.jpg"
    assert calls["reconstruct"][1] == LENS.pinhole()
    # the undistorted grid's pixels beyond the lens (above the centre, left of it) have no depth;
    # the grid holds the whole lens, its corners too
    assert not frame.valid[0, 32] and frame.depth[0, 32] == 0 and not frame.valid[24, 0]
    assert frame.valid[24, 32] and frame.valid[0, 0]
    (det,) = dets
    assert det.mask.shape == (48, 64) and det.mask[24, 32] and det.label == "box"
    # the undistorted image shows the block smaller (a shorter focal length), still centred
    x0, y0, x1, y1 = det.box
    assert 26 < x0 < 32 < x1 < 38 and 20 < y0 < 24 < y1 < 28
    assert det.mask.sum() < _raw_detection().mask.sum()
    plain, same = api.reconstruct_and_detect_keyframe(kf, tmp_path / "w", _Client())
    assert calls["detect"] == calls["reconstruct"][0] == kf.path
    assert np.array_equal(same[0].mask, _raw_detection().mask)


def test_detections_move_between_undistorted_grids_and_boxes_beyond_the_lens_vanish() -> None:
    frame = SimpleNamespace(grid_size=(64, 48), depth=np.ones((48, 64), np.float32),
                            valid=np.ones((48, 64), bool))
    moved = api._onto_undistorted(frame, [_raw_detection(), _raw_detection()], LENS, None)
    back = api._onto_undistorted(frame, moved, LENS, LENS)  # the same grid again
    np.testing.assert_array_equal(back[0].mask, moved[0].mask)
    np.testing.assert_allclose(back[1].box, moved[1].box, atol=1e-6)
    # a box where a pincushion lens shows nothing: no outline left
    edge = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, "colmap", 0.5)
    assert api._box_onto((0.0, 0.0, 1.0, 1.0), (32, 24), LENS, (64, 48), edge) == (0, 0, 0, 0)


def test_a_lens_the_fit_found_re_runs_inference_on_the_undistorted_images(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """f0: inferred on its own image, re-run undistorted, its detections moved; f1: already
    inferred with the lens; f2: a pinhole photo with EXIF, left as it is; f3 not registered."""
    model = FakeModel(line([jpg(0), jpg(1), jpg(2)]))
    monkeypatch.setattr(model, "intrinsics", lambda n: K if n == jpg(2) else LENS)
    monkeypatch.setattr(api, "_undistorted", lambda kf, lens, work: SimpleNamespace(
        **{**vars(kf), "exif": lens.pinhole()}))
    done: list[tuple[str, Any]] = []

    def reconstruct(path: Path, client: Any, intrinsics: Any, work: Path) -> Any:
        done.append((path.name, intrinsics))
        return _grid_frame(intrinsics=intrinsics)

    monkeypatch.setattr(api, "_reconstruct_keyframe", reconstruct)
    monkeypatch.setattr(api, "_onto_undistorted",
                        lambda frame, dets, lens, seen: [("moved", tuple(dets), lens, seen)])
    new = [nf(0), nf(1), nf(2, exif=K), nf(3)]
    new[0].dets = ["det"]
    new[1].lens = replace(LENS, fx=LENS.fx * 1.01)  # within FOCAL_RERUN_REL: the same lens
    msgs: list[str] = []
    api._rerun_focal(ctx_of(new), model, _Client(), msgs.append)  # type: ignore[arg-type]
    assert done == [("f000000.jpg", LENS.pinhole())]
    assert new[0].lens == LENS and new[0].dets == [("moved", ("det",), LENS, None)]
    assert new[0].frame.intrinsics == LENS.pinhole() and new[1].dets == []
    assert msgs == ["re-running inference for 1 keyframes on their undistorted images "
                    "(lens distortion -0.300)"]


def test_a_rebuild_keeps_its_published_ids_on_the_moved_detections(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuilt keyframe's stored detections, moved onto the new undistorted grid, carry the ids
    the map published with them (``Rebuild.prior`` / ``first``, by detection)."""
    monkeypatch.setattr(api, "_undistorted", lambda kf, lens, work: kf)
    monkeypatch.setattr(api, "_reconstruct_keyframe", lambda *a: _grid_frame())
    old = [_raw_detection(), _raw_detection()]
    rb = SimpleNamespace(prior={id(old[0]): 7}, first={id(old[0]): 5, id(old[1]): 9})
    new = nf(0)
    new.dets = list(old)
    ctx = ctx_of([new])
    ctx.rebuild = rb
    api._rerun_undistorted(ctx, [(new, LENS)], _Client(), lambda m: None)  # type: ignore[arg-type]
    moved = new.dets
    assert moved[0] is not old[0] and rb.prior == {id(moved[0]): 7}
    assert rb.first == {id(moved[0]): 5, id(moved[1]): 9}
    api._carry_ids(None, old, moved)  # no rebuild: nothing to carry


def test_a_keyframe_whose_camera_ends_without_a_lens_is_inferred_on_its_own_image(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """f0 was inferred undistorted with the map's lens, but its SfM camera ends a pinhole (a
    rebuild's accepted SfM, a photo sharing another camera): inferred again on its own image at
    the camera's focal length, its detections moved back from the undistorted grid."""
    model = FakeModel(line([jpg(0)]))
    monkeypatch.setattr(model, "intrinsics", lambda n: K)
    done: list[tuple[Path, Any]] = []

    def reconstruct(path: Path, client: Any, intrinsics: Any, work: Path) -> Any:
        done.append((path, intrinsics))
        return _grid_frame(intrinsics=intrinsics)

    monkeypatch.setattr(api, "_reconstruct_keyframe", reconstruct)
    monkeypatch.setattr(api, "_onto_undistorted",
                        lambda frame, dets, lens, seen: [("moved", tuple(dets), lens, seen)])
    new = nf(0)
    new.lens, new.dets = LENS, ["det"]
    msgs: list[str] = []
    api._rerun_focal(ctx_of([new]), model, _Client(), msgs.append)  # type: ignore[arg-type]
    assert done == [(new.kf.path, K)] and new.lens is None
    assert new.dets == [("moved", ("det",), K, LENS)]
    assert msgs == ["re-running inference for 1 keyframes on their own images (their camera has "
                    "no lens distortion)"]


def test_detections_move_back_from_an_undistorted_grid_onto_the_image_s() -> None:
    frame = _grid_frame()
    moved = api._onto_undistorted(frame, [_raw_detection()], LENS, None)
    back = api._onto_undistorted(_grid_frame(), moved, K, LENS)
    assert back[0].mask[24, 32] and back[0].mask.sum() >= 0.8 * _raw_detection().mask.sum()


def test_a_client_that_is_its_own_clone_is_not_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_rerun_undistorted`` with a client whose clone is itself (one connection): kept open."""
    monkeypatch.setattr(api, "_undistorted", lambda kf, lens, work: kf)
    monkeypatch.setattr(api, "_reconstruct_keyframe", lambda *a: _grid_frame())
    monkeypatch.setattr(api, "_onto_undistorted", lambda frame, dets, lens, seen: ["moved"])

    class Same(_Client):
        def clone(self) -> _Client:
            return self

    client, new = Same(), nf(0)
    api._rerun_undistorted(ctx_of([new]), [(new, LENS)], client, lambda m: None)  # type: ignore[arg-type]
    assert client.closed == 0 and new.dets == ["moved"] and new.lens == LENS


def test_lenses_differ_by_focal_length_or_distortion() -> None:
    assert api._same_lens(LENS, LENS) and not api._same_lens(None, LENS)
    assert not api._same_lens(replace(LENS, fx=600.0), LENS)
    assert not api._same_lens(replace(LENS, k=-0.2), LENS)


def test_the_undistorted_keyframe_is_the_pinholes(tmp_path: Path) -> None:
    kf = ingest.Keyframe("f000000", 0, _image(tmp_path / "in.jpg"), "in.jpg", None)
    out = api._undistorted(kf, LENS, tmp_path / "work")
    assert out.path == tmp_path / "work" / "f000000" / "undistorted.jpg" and out.path.exists()
    assert out.exif == LENS.pinhole() and out.name == kf.name and kf.path.exists()


def test_a_stored_keyframes_image_is_on_its_depths_grid(tmp_path: Path) -> None:
    path = _image(tmp_path / "k.jpg")
    pin = store.keyframe_rgb(path, _stored(0, K), 64)
    lens = store.keyframe_rgb(path, _stored(0, LENS), 64)
    assert pin.shape == lens.shape == (48, 64, 3) and not np.array_equal(pin, lens)
    assert _stored(0, LENS).K_grid == LENS.pinhole().resized(64, 48)
    assert _stored(0, LENS).K_grid.fx < K.resized(64, 48).fx  # the lens's field of view


# --- depth scales -------------------------------------------------------------------------------


def _dense_world(monkeypatch: pytest.MonkeyPatch, fits: list[ScaleFit]) -> tuple[Any, Any]:
    """Three keyframes of a multi-view map (densely aligned), the dense fits in call order."""
    names = [jpg(i) for i in range(3)]
    ctx = ctx_of([nf(i) for i in range(len(names))])
    model = FakeModel(line(names, 0.05), method="multiview")
    for n in ctx.new:
        n.lens = None
    queue = list(fits)
    monkeypatch.setattr(rdepth, "dense_scale", lambda *a, **k: queue.pop(0))
    return ctx, model


def test_a_failed_dense_fit_takes_its_neighbours_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """f0 seeds the map; f1's fit fails, f2's is inconsistent: neither is stored, both take the
    scale of the keyframes they were fitted against (low confidence); no keyframe is dense, so
    nothing is fitted again."""
    ctx, model = _dense_world(monkeypatch, [ScaleFit(float("nan"), 0, float("inf")),
                                            ScaleFit(9.3, 5000, 1.5)])
    api._align_depths(ctx, model)  # type: ignore[arg-type]
    f0, f1, f2 = (n.record for n in ctx.new)
    assert f0.stats["depth_scale_method"] == "seed" and f0.depth_scale == 1.0
    assert f1.stats["depth_scale_method"] == "neighbours" and f1.stats["dense_fit"] is None
    assert f2.stats["dense_fit"] == {"scale": 9.3, "spread": 1.5}
    assert f1.depth_scale == f2.depth_scale == 1.0 and f1.low_confidence and f2.low_confidence


def test_dense_scales_are_fitted_again_on_either_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """f1 and f2 fit 1.2 and 1.1 against the keyframes before them; fitted again, f1 takes 1.05
    (then a failed fit keeps it) and f2 1.08 (then an inconsistent one keeps it)."""
    ctx, model = _dense_world(monkeypatch, [
        ScaleFit(1.2, 5000, 0.05), ScaleFit(1.1, 5000, 0.05),  # the first pass
        ScaleFit(1.05, 5000, 0.04), ScaleFit(1.08, 5000, 0.03),  # sweep 1
        ScaleFit(float("nan"), 0, float("inf")), ScaleFit(2.0, 5000, 0.9)])  # sweep 2
    api._align_depths(ctx, model)  # type: ignore[arg-type]
    f1, f2 = ctx.new[1].record, ctx.new[2].record
    assert (f1.depth_scale, f2.depth_scale) == (1.05, 1.08)
    assert f1.stats["depth_scale_spread"] == 0.04 and f2.stats["depth_scale_points"] == 5000
    np.testing.assert_allclose(ctx.new[1].depth, ctx.new[1].frame.depth * 1.05)
    assert len(f1.stats["depth_scale_refs"]) == 2  # f0 and f2, on either side


def test_no_densely_aligned_keyframe_is_nothing_to_fit_again() -> None:
    api._realign_dense(ctx_of([nf(0)]), [])  # type: ignore[arg-type]


def test_confident_pairs_of_sightings_skip_low_confidence_keyframes_on_either_side() -> None:
    """``objects._nearest_pairs``: a sighting from a low-confidence keyframe takes part in no
    confident pair, whichever object it belongs to."""
    from oh_my_slam.mapping import objects as mo

    def sighting(frame: int) -> mo.Sighting:
        return mo.Sighting(frame, 100, 0.0, (0.0, 0.0, 2.0), (-0.1, -0.1, 1.9), (0.1, 0.1, 2.1))

    def obj(oid: int, frames: list[int]) -> mo.MapObject:
        return mo.MapObject(oid, "box", {"box": 1.0}, [0.9], np.zeros((0, 3), np.float32),
                            frames=frames, sightings=[sighting(f) for f in frames])

    records = [replace(_stored(i, K), low_confidence=i == 2) for i in range(3)]
    views = mo._Views(None, {}, records)
    a, b = obj(1, [0, 2]), obj(2, [1, 2])
    pairs = mo._nearest_pairs(a, b, views, confident=True)
    assert [(sa.frame, sb.frame) for _, sa, sb in pairs] == [(0, 1)]
    assert len(mo._nearest_pairs(a, b, views)) == min(4, mo.DEPTH_PAIRS)  # all otherwise


# --- the out-of-focus near field ----------------------------------------------------------------


def _blurred_frame(**kw: Any) -> SimpleNamespace:
    """A keyframe's reconstruction whose left band is a blurred object 0.25 m from the lens
    (``tests.unit.test_defocus``)."""
    from tests.unit.test_defocus import _scene

    rgb, depth, valid = _scene(4.0)
    h, w = depth.shape
    return SimpleNamespace(grid_size=(w, h), depth=depth, valid=valid, rgb=rgb,
                           K_grid=K.resized(w, h), **kw)


def test_every_inference_path_drops_the_out_of_focus_near_field(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first inference, the focal re-run and the undistorted re-run of a keyframe: its
    blurred near band has no depth afterwards (``_without_defocus``), the rest keeps it."""

    def check(frame: Any) -> None:
        assert not frame.valid[:, 20:60].any() and not frame.depth[:, 20:60].any()
        assert frame.valid[:, 100:].all() and (frame.depth[:, 100:] == 2.0).all()

    monkeypatch.setattr(api, "detect_alongside",
                        lambda path, client, rec, *, max_side: (rec(client), []))
    monkeypatch.setattr(api, "_reconstruct_keyframe",
                        lambda *a, **k: _blurred_frame(intrinsics=K))
    kf = ingest.Keyframe("f000000", 0, _image(tmp_path / "in.jpg"), "in.jpg", None)
    frame, _ = api.reconstruct_and_detect_keyframe(kf, tmp_path / "w", _Client())
    check(frame)
    # the focal re-run: SfM's focal length 10 % off the model's
    model = FakeModel(line([jpg(0), jpg(1)]))
    monkeypatch.setattr(model, "intrinsics", lambda n: replace(K, fx=550.0, fy=550.0))
    redo = nf(0)
    redo.frame = _blurred_frame(intrinsics=K)  # the pixels the first pass read
    api._rerun_focal(ctx_of([redo]), model, _Client(), lambda m: None)  # type: ignore[arg-type]
    check(redo.frame)
    # the undistorted re-run
    monkeypatch.setattr(api, "_undistorted", lambda kf, lens, work: SimpleNamespace(
        **{**vars(kf), "exif": lens.pinhole()}))
    monkeypatch.setattr(api, "_onto_undistorted", lambda frame, dets, lens, seen: dets)
    again = nf(1)
    api._rerun_undistorted(ctx_of([again]), [(again, LENS)], _Client(),  # type: ignore[arg-type]
                           lambda m: None)
    check(again.frame)
