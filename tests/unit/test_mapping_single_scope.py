"""``mapper.sh update -t single`` (spec §2.3): "only what the newly added input covers: the poses of
the new frames and the objects observed in them (PLY: the new frames' points)" — after an update
that extends the map and after one that rebuilds it with the stored keyframes (``api.Rebuild``),
whose keyframes are mapped again but are not the update's input.

Rendered scene with known poses (no SfM, no server): the map's first keyframes see a cup and a
cabinet; the update's keyframes look at the cabinet only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.geometry import project
from oh_my_slam.core.images import load_rgb, save_jpeg
from oh_my_slam.core.ply import parse_ply
from oh_my_slam.mapping import api, export, ingest, store
from oh_my_slam.mapping.geometry import MapGeometry
from oh_my_slam.mapping.objects import ObjectState
from oh_my_slam.reconstruction.api import FrameReconstruction
from tests.synth.scene import look_at
from tests.unit.test_mapping_latest_wins import AT, CABINET, CUP, WITH_CUP
from tests.unit.test_mapping_semantics import K, Shot, _detection, shoot

# looking at the cabinet from beside it: the cup is far outside the image
CAB = [look_at(np.array([-0.3 + dx, 1.2, 1.1]), CABINET.center) for dx in (0.0, 0.06, -0.06)]


def _update(mdir: Path, shots: list[Shot], work: Path, stored: list[Shot] | None = None
            ) -> tuple[ObjectState, MapGeometry, list[store.FrameRecord], list[str]]:
    """One update of ``shots`` with their true poses (``test_mapping_semantics.known_pose_update``)
    returning its geometry and the names of its input keyframes. With ``stored``: a rebuild of a
    map whose first update held those keyframes, mapped again with ``shots`` as one update (their
    records keep update 1, ``Rebuild.uids``)."""
    work.mkdir(parents=True, exist_ok=True)
    with store.MapTransaction(mdir) as tx:
        meta = store.read_meta_or_default(tx)
        old = store.read_frames(tx)
        uid = int(meta.get("update_count", 0)) + 1 + (1 if stored else 0)
        start = int(meta.get("next_frame_index", 0))
        batch = [(sh, 1) for sh in stored or []] + [(sh, uid) for sh in shots]
        new = []
        for k, (sh, u) in enumerate(batch):
            idx = start + k
            name = store.frame_name(idx)
            img = tx.stage("frames") / f"{name}.jpg"
            save_jpeg(sh.rgb, img, quality=95)
            h, w = sh.depth.shape
            frame = FrameReconstruction(img, load_rgb(img, max_side=max(w, h)), sh.depth,
                                        sh.depth > 0, K, K)
            nf = api.NewFrame(ingest.Keyframe(name, idx, img, f"shot {k}", None), frame,
                              [_detection(*d) for d in sh.dets], (w, h))
            nf.record = store.FrameRecord(
                idx, name, f"frames/{name}.jpg", f"shot {k}", 1, w, h, K, sh.pose, w, h,
                pose_source="sfm-global", update_id=u,
                stats={"observations": 500.0, "reproj_error": 0.5})
            nf.depth = sh.depth
            new.append(nf)
        rb = (api.Rebuild({nf.kf.name: 1 for nf in new[:len(stored)]}, {}, {}, 1, {}, {}, [])
              if stored else None)
        ctx = api.UpdateContext(tx, meta, old, new, uid, work, rebuild=rb)
        records, objs, geo = api.integrate(ctx, lambda m: None)
        meta.update(update_count=uid, next_frame_index=start + len(batch),
                    next_object_id=objs.next_id)
        tx.commit(meta)
    return objs, geo, records, [nf.kf.name for nf in new[len(stored or []):]]


def _observed_in(mdir: Path, names: list[str]) -> set[int]:
    """The objects the keyframes ``names`` observed: the ids of their stored instances."""
    out = set()
    for n in names:
        doc = json.loads((mdir / store.frame_file(n, "instances.json")).read_text())
        out |= {int(it["object_id"]) for it in doc["instances"]}
    return out - {0}


def _seen_by(xyz: np.ndarray, records: list[store.FrameRecord], mdir: Path, names: list[str],
             rel: float) -> np.ndarray:
    """Whether each point lies on a surface one of the keyframes ``names`` sees: it projects into
    the keyframe's image in front of it, onto a pixel whose stored depth agrees within ``rel``."""
    out = np.zeros(len(xyz), bool)
    for r in records:
        if r.name not in names:
            continue
        depth = np.load(mdir / store.frame_file(r.name, "depth.npy")).astype(np.float64)
        cam = r.T_map_cam.inverse()
        uv, z = project(cam.apply(xyz), r.K_grid.K())
        h, w = depth.shape
        with np.errstate(invalid="ignore"):
            u, v = np.floor(uv[:, 0] + 0.5), np.floor(uv[:, 1] + 0.5)
            inside = np.flatnonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        d = depth[v[inside].astype(int), u[inside].astype(int)]
        out[inside[np.abs(d - z[inside]) <= rel * z[inside]]] = True
    return out


def _on_cup(xyz: np.ndarray) -> np.ndarray:
    return np.asarray((np.abs(xyz[:, :2] - CUP.center[:2]).max(axis=1) < 0.15)
                      & (xyz[:, 2] > 0.03))


def _check_single(mdir: Path, objs: ObjectState, geo: MapGeometry,
                  records: list[store.FrameRecord], names: list[str]) -> None:
    exported = {o.id: o for o in objs.exported()}
    (cup,) = [o.id for o in exported.values() if o.label == "cup"]  # the map still holds it
    (cab,) = [o.id for o in exported.values() if o.label == "cabinet"]
    # objects: exactly those the input observed — the cabinet, not the cup
    assert _observed_in(mdir, names) == {cab}
    single = {o.id for o in export.payload_objects(objs, "single")}
    assert single == {cab} and {o.id for o in export.payload_objects(objs, "full")} == {cab, cup}
    doc = json.loads(export.scene_payload(
        export.full_scene(mdir, {}, records, objs.exported()), "single", names, records, objs))
    assert {int(k) for k in doc["openlabel"]["objects"]} == {cab}
    assert {f["frame_properties"]["keyframe"] for f in doc["openlabel"]["frames"].values()} \
        == set(names)
    # PLY: the input's points only, and every one of them — not the cup the stored keyframes see
    full = geo.cloud.xyz.astype(np.float64)
    ply = parse_ply(export.ply_payload(geo, "single", records, objs, CloudAttrs()))
    pts = ply.xyz.astype(np.float64)
    assert 0 < len(pts) < len(full)
    assert _on_cup(full).sum() > 100 and not _on_cup(pts).any()
    assert _seen_by(pts, records, mdir, names, 0.1).all()
    seen = _seen_by(full, records, mdir, names, 0.01)
    kept = {tuple(p) for p in np.round(pts, 5)}
    assert np.mean([tuple(p) in kept for p in np.round(full[seen], 5)]) > 0.95


def test_single_scope_of_an_update_that_extends_the_map(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    _update(mdir, shoot(WITH_CUP, AT), tmp_path / "w1")
    objs, geo, records, names = _update(mdir, shoot(WITH_CUP, CAB), tmp_path / "w2")
    assert {r.update_id for r in records if r.name in names} == {2}
    _check_single(mdir, objs, geo, records, names)


def test_single_scope_of_an_update_that_rebuilds_the_map(tmp_path: Path) -> None:
    """The stored keyframes (``AT``: the cup and the cabinet) are mapped again with the input
    (``CAB``): ``-t single`` still covers the input only."""
    mdir = tmp_path / "m"
    objs, geo, records, names = _update(mdir, shoot(WITH_CUP, CAB), tmp_path / "w",
                                        stored=shoot(WITH_CUP, AT))
    assert len(records) == len(AT) + len(CAB)
    assert {r.update_id for r in records if r.name not in names} == {1}
    _check_single(mdir, objs, geo, records, names)
