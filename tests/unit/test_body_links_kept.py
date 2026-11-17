"""Body links (oms/2.15), the audit's paths: a points fit within the bound beside a held link,
every kept path at the linked pose, and the records of frames the checks reject (map.md §1
"Body links", §2 "Match graph")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping import body as bl
from oh_my_slam.mapping.anchored import AnchoredMap
from tests.synth import stereo as syn
from tests.unit.test_body_links_map import body, err, frame, origin, pans, prior  # noqa: F401


def test_a_points_fit_within_the_bound_stands_beside_a_held_link(
        tmp_path: Path, pans: dict[int, syn.Shot], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Without a prior, a points fit the linked pose confirms is registered below the no-prior
    minimums (without the link it is rejected); with a prior, a fit the checks would find loose
    or in an unsupported conflict gives way to the linked pose."""
    m = origin(tmp_path, pans)
    snap = m._snapshot()
    with monkeypatch.context() as mp:
        mp.setattr(an, "NO_PRIOR_MIN_INLIERS", 10 ** 6)
        pl = m.update([frame("a", pans[25], body("a", pans[25], "p0"))], lambda _: None)[0]
        assert pl.status == "registered" and pl.method == "pnp"
        m._restore(snap)
        m.keyframes = [k for k in m.keyframes if k.name == "p0"]
        pl = m.update([frame("b", pans[25], None)], lambda _: None)[0]
        assert pl.status == "rejected"
    for name, patch in (("c", lambda *a, **k: None),
                        ("d", lambda self, f, img, *a: (0.5, 1.0, (0.01, 0.2), True))):
        m._restore(snap)
        m.keyframes = [k for k in m.keyframes if k.name == "p0"]
        with monkeypatch.context() as mp:
            mp.setattr(AnchoredMap, "_image_conflict", patch)
            mp.setattr(an, "measurable_pose", lambda *a, **k: False)
            pl = m.update([frame(name, pans[25], body(name, pans[25], "p0"),
                                 prior(pans[25].T, "drifting", 0.2, 5.0))], lambda _: None)[0]
        assert pl.status == "registered" and pl.method == "linked", name


def test_a_frame_kept_before_the_origin_takes_the_linked_pose(
        tmp_path: Path, pans: dict[int, syn.Shot]) -> None:  # noqa: F811
    """One job: ``a`` (a failed image check, before ``p0`` in canonical order) is kept before
    the origin exists; once ``p0`` registers, ``a`` is kept at its linked pose, its keyframe
    with it."""
    off = pans[25].T.copy()
    off[0, 3] += 0.2
    m = AnchoredMap(tmp_path / "m")
    pls = m.update([frame("p0", pans[0], body("p0", pans[0], None), prior(pans[0].T)),
                    frame("a", pans[25], body("a", pans[25], "p0"),
                          prior(off, "drifting", 0.3, 8.0), unconfirmed="blur")],
                   lambda _: None)
    a = pls[1]
    assert a.status == "kept_at_prior" and a.reason == "blur" and a.T is not None
    assert err(a.T, pans[25].T)[0] < 0.02
    assert a.link is not None and a.link["state"] == "used"
    k = m.keyframe(a.keyframe or "")
    assert k is not None and np.allclose(k.T, a.T)


def test_a_kept_frame_without_a_confirmed_from_keeps_its_prior(
        tmp_path: Path, pans: dict[int, syn.Shot]) -> None:  # noqa: F811
    m = AnchoredMap(tmp_path / "m")
    pls = m.update([frame("p0", pans[0], body("p0", pans[0], None), prior(pans[0].T),
                          unconfirmed="blur"),
                    frame("p1", pans[25], body("p1", pans[25], "p0"), prior(pans[25].T),
                          unconfirmed="blur")], lambda _: None)
    assert [p.status for p in pls] == ["kept_at_prior", "kept_at_prior"]
    assert pls[1].link is not None and pls[1].link["reason"] == "from_unconfirmed"
    assert pls[1].T is not None and np.allclose(pls[1].T, pans[25].T)


def test_a_rejected_frame_s_record_composes_paths_in_its_job(
        tmp_path: Path, pans: dict[int, syn.Shot]) -> None:  # noqa: F811
    m = origin(tmp_path, pans)
    m.job_bodies = {"r": body("r", pans[25], "p0")}
    m.set_bodies([])
    row = m.link_report(body("s", pans[50], "r"))[0]
    assert row is not None and row["reason"] == "from_unconfirmed"
    m._bodies["s"] = body("s", pans[50], "r")
    assert bl.path_link(m._bodies, "p0", "s") is not None


def test_a_map_whose_scale_becomes_metric_mid_job_stops_pooling_links(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A map holding a link vote (three tight agreeing link-measured scales) and one metric
    keyframe; a job whose first frame in canonical order lies 0.5 m away (the metric priors then
    span past ``PRIOR_SPAN_M``): every estimate of k from then on, within the job, leaves the
    link vote out (map.md §1 "Scale", item 3)."""
    from tests.unit.test_keyframe_scale import frame as walk_frame

    room = syn.ainex_room()
    shots = [syn.render(room, syn.head(x), i) for i, x in enumerate((0.0, 0.25, 0.5))]
    m = AnchoredMap(tmp_path / "m")
    m.update([walk_frame("f0", shots[0], "metric", 0.01)], lambda _: None)
    m.scale.links = [[1.2, 0.002], [1.2, 0.002], [1.2, 0.002]]
    m._estimate()
    assert m.scale.links_ok and m.scale.source == "baseline"
    calls: list[tuple[float, bool, str]] = []
    real = AnchoredMap._estimate

    def spy(self: AnchoredMap) -> None:
        real(self)
        calls.append((self.prior_span(), self.scale.links_ok, self.scale.source))

    monkeypatch.setattr(AnchoredMap, "_estimate", spy)
    pls = m.update([walk_frame("a", shots[2], "metric", 0.01),
                    walk_frame("b", shots[1], "metric", 0.01)], lambda _: None)
    assert [p.status for p in pls] == ["registered", "registered"]
    wide = [c for c in calls if c[0] >= an.PRIOR_SPAN_M]
    assert len(wide) >= 2  # within the job (``b``'s scale evidence), not only at its end
    assert all(not ok and source != "baseline" for _, ok, source in wide)
    assert any(ok for span, ok, _ in calls if span < an.PRIOR_SPAN_M)
