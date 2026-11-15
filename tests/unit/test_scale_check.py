"""``quality.scale_check`` (api.md §4, oms/2.8): the camera heights a job's frames sent against
the frames' own floor heights in their raw depth, on the prior-first engine (jobs and dry runs)
and the older mapper, and the helpers behind it and behind frame analysis's ``scale_check``
(api.md §7: ``tests/unit/test_service_analyse.py``). Every body is checked against the schema."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import analysis
from oh_my_slam.mapping import anchored as an
from oh_my_slam.mapping.quality import scale_check
from tests.synth.room import base_pose
from tests.unit.test_round14 import _floor_points, old  # noqa: F401
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import (
    CAM,
    client,  # noqa: F401
    dry,
    frames_for,
)

H = 0.5  # the synthetic camera's height above the room's floor
CAM_H = {**CAM, "height_above_floor_m": H, "height_sigma_m": 0.01}


def _check(job: dict[str, Any]) -> dict[str, Any] | None:
    got: dict[str, Any] | None = job["quality"]["scale_check"]
    return got


def test_the_summary_is_the_median_the_spread_and_the_count() -> None:
    assert scale_check([]) is None
    assert scale_check([1.2]) == {"median_ratio": 1.2, "spread": 0.0, "frames": 1}
    got = scale_check([1.0, 1.0, 2.0, 1.0, 1.0])
    assert got == {"median_ratio": 1.0, "spread": 0.0, "frames": 5}
    got = scale_check([0.9, 1.0, 1.1, 1.0, 3.0])  # MAD 0.1 over the median 1
    assert got is not None and got["spread"] == pytest.approx(1.4826 * 0.1)
    assert got["median_ratio"] == 1.0 and got["frames"] == 5


def test_a_job_and_its_dry_run_report_the_frames_sent_height_against_their_floor(
        client: Client) -> None:  # noqa: F811
    """Sensor depth: a camera height of 0.5 m, every frame's floor 0.5 m below it. A frame that
    sends its own height overrides the camera's, and counts whether the floor check kept its
    content or not (the wrong heights exclude it)."""
    fr, files = frames_for(client, [base_pose(0)], tag="a")
    first = client.update("m", fr, files, cameras={"cam": CAM_H})
    assert first["state"] == "committed"
    got = _check(first)
    assert got is not None and got["frames"] == 1
    assert got["median_ratio"] == pytest.approx(1.0, abs=0.03) and got["spread"] == 0.0
    poses = [base_pose(15.0 + 30.0 * k) for k in range(5)]
    fr, files = frames_for(client, poses, tag="s")
    d = dry(client, "m", fr, files)
    job = client.update("m", fr, files)
    got = _check(job)
    assert got is not None and got["frames"] == 5
    assert got["median_ratio"] == pytest.approx(1.0, abs=0.03) and got["spread"] < 0.05
    assert d["quality"]["scale_check"] == got  # the dry run measures what the job does
    # per-frame heights over the camera's: three frames send 1 m, the other two fall back on
    # the camera's 0.5 m. The floor check excludes the three sensor-depth frames (listed), and
    # their ratio of 2 still counts: the median is theirs
    fr, files = frames_for(client, [base_pose(5.0 + 30.0 * k) for k in range(3)], tag="o",
                           height_above_floor_m=1.0)
    fr2, files2 = frames_for(client, [base_pose(95.0 + 30.0 * k) for k in range(2)], tag="p")
    d = dry(client, "m", fr + fr2, {**files, **files2})
    job = client.update("m", fr + fr2, {**files, **files2})
    assert job["state"] == "committed"
    assert sorted(job["quality"]["floor_inconsistent"]) == ["o0", "o1", "o2"]
    got = _check(job)
    assert got is not None and got["frames"] == 5
    assert got["median_ratio"] == pytest.approx(2.0, abs=0.06)
    assert got["spread"] == pytest.approx(0.0, abs=0.06)
    assert d["quality"]["scale_check"] == got


def test_a_job_without_a_sent_height_or_without_a_floor_has_no_scale_check(
        client: Client) -> None:  # noqa: F811
    fr, files = frames_for(client, [base_pose(0), base_pose(30.0)], tag="a")
    job = client.update("m", fr, files, cameras={"cam": CAM})
    assert job["state"] == "committed" and _check(job) is None
    assert dry(client, "m", fr[:1], files)["quality"]["scale_check"] is None
    # a height, but a depth with no floor plane in it: nothing to compare with
    fr, files = frames_for(client, [base_pose(60.0)], tag="b", height_above_floor_m=H)
    for name, blob in list(files.items()):
        if name.endswith("d"):
            files[name] = _flat(blob)
    job = client.update("m", fr, files)
    assert job["state"] == "committed" and _check(job) is None


def _flat(depth_png: bytes) -> bytes:
    """A depth image of the same size holding a frontal wall 2 m ahead: no floor plane."""
    import io

    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(depth_png)))
    out = io.BytesIO()
    Image.fromarray(np.full_like(arr, 2000)).save(out, format="PNG")
    return out.getvalue()


def _mono_job(c: Client, name: str, scale: float, odd: float = 1.0
              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """A monocular map with a camera height (frame ``a0``), then a job of five frames whose
    monocular depth is ``scale`` times the true one, the third (``s2``) ``odd`` times that more.
    Returns the dry run of the job and the job."""
    fr, files = frames_for(c, [base_pose(0)], tag="a", depth=False)
    key = hashlib.sha256(files["a0"]).hexdigest()
    c.backend.mono_depth[key] = c.backend.mono_depth[key] * scale
    assert c.update(name, fr, files, cameras={"cam": CAM_H})["state"] == "committed"
    fr, files = frames_for(c, [base_pose(15.0 + 30.0 * k) for k in range(5)], tag="s",
                           depth=False)
    for k in range(5):
        key = hashlib.sha256(files[f"s{k}"]).hexdigest()
        c.backend.mono_depth[key] = c.backend.mono_depth[key] * scale * (odd if k == 2 else 1.0)
    return dry(c, name, fr, files), c.update(name, fr, files)


def test_monocular_frames_are_checked_in_the_models_own_metres(client: Client) -> None:  # noqa: F811
    """The model's depth is 0.8 of the truth in every frame: the map takes its scale from the
    height (k = 1.25), and the ratio still reads the model's own metres, before that scale."""
    d, job = _mono_job(client, "m", 0.8)
    assert job["state"] == "committed" and job["quality"]["floor_inconsistent"] == []
    got = _check(job)
    assert got is not None and got["frames"] == 5
    assert got["median_ratio"] == pytest.approx(1.25, abs=0.05) and got["spread"] < 0.05
    assert d["quality"]["scale_check"] == got


def test_a_frame_the_floor_check_did_not_judge_counts_in_the_scale_check(
        client: Client) -> None:  # noqa: F811
    """A monocular frame whose own depth scale lies beyond the bound is ``floor_not_judged``
    (its content is withheld); its sent height against its own floor is measured all the same:
    its ratio of 2 is the outlier of five, so the median stays 1 and the spread 0."""
    d, job = _mono_job(client, "m", 1.0, odd=0.5)
    rows = {i["name"]: i for i in job["inputs"]}
    assert rows["s2"]["detail"] == "floor_not_judged"
    got = _check(job)
    assert got is not None and got["frames"] == 5
    assert got["median_ratio"] == pytest.approx(1.0, abs=0.05) and got["spread"] < 0.05
    assert d["quality"]["scale_check"] == got
    # the same job with all five frames' depth halved: every ratio is 2, none judged
    _, job = _mono_job(client, "n", 1.0, odd=1.0)
    assert _check(job)["frames"] == 5  # type: ignore[index]


def test_floor_ratios_skip_a_frame_without_a_height_a_depth_or_a_floor_plane(
        tmp_path: Any) -> None:
    from tests.unit.test_round14 import frame

    m = an.AnchoredMap(tmp_path / "m")
    assert m.floor_ratios() == []
    # a depth image, a monocular depth 0.8 of the truth (its raw metres: the ratio 1.25 whatever
    # scale the map gives it), a frame without a height and one without a depth: only the
    # first two count
    m.update([frame("a0", base_pose(0)), frame("a1", base_pose(30.0), depth=None, mono=0.8),
              frame("a2", base_pose(60.0), height=None),
              frame("a3", base_pose(90.0), depth=None, mono=None)])
    got = m.floor_ratios()
    assert got == [pytest.approx(1.0, abs=0.03), pytest.approx(1.25, abs=0.05)]
    assert len(m._added) == 4


def test_the_older_mapper_measures_a_registered_frames_floor() -> None:
    from oh_my_slam.service.update import Updater

    z, Kg, T = _floor_points(0.5)

    class Cam:
        height_above_floor_m = 0.5
        height_sigma_m = None

        @staticmethod
        def canonical() -> Intrinsics:
            return Kg

    class Work:
        height_m: float | None = None
        height_sigma_m: float | None = None

    class Fr:
        internal = type("I", (), {"name": "k"})()
        pose: Any = T
        cam, work = Cam(), Work()
        depth_m: Any = None
        ignore = None

    mapper_depth: Any = z * 0.8  # the mapper's monocular depth
    internal = type("M", (), {})()
    internal.depth = lambda name: mapper_depth
    fr = Fr()
    # a monocular frame: the mapper's own metres, before the similarity to the map
    got = Updater._floor_ratio(fr, internal)  # type: ignore[arg-type]
    assert got == pytest.approx(0.5 / (0.8 * 0.5), abs=0.03)
    # a depth image: its own sensor metres, whatever the mapper's depth says
    fr.depth_m = z * 2.0
    assert Updater._floor_ratio(fr, internal) == pytest.approx(0.5, abs=0.02)  # type: ignore[arg-type]
    fr.depth_m = None
    # the frame's own height over the camera's
    fr.work = type("W", (), {"height_m": 1.0, "height_sigma_m": None})()
    assert Updater._floor_ratio(fr, internal) == pytest.approx(2.5, abs=0.1)  # type: ignore[arg-type]
    fr.work = Work()
    # none: no depth in the mapper, no floor plane, no height, no pose
    mapper_depth = None
    assert Updater._floor_ratio(fr, internal) is None  # type: ignore[arg-type]
    mapper_depth = np.full_like(z, 2.0)
    assert Updater._floor_ratio(fr, internal) is None  # type: ignore[arg-type]
    fr.cam = type("C", (), {"height_above_floor_m": None})()
    assert Updater._floor_ratio(fr, internal) is None  # type: ignore[arg-type]
    fr.cam, fr.pose = Cam(), None
    assert Updater._floor_ratio(fr, internal) is None  # type: ignore[arg-type]


def test_an_own_floor_plane_and_its_inlier_share() -> None:
    z, Kg, T = _floor_points(0.5)
    R = T[:3, :3]
    got = an.own_floor_plane(z, Kg, R, 0.5, 0.06)
    assert got is not None and got[0] == pytest.approx(0.5, abs=0.01)
    assert got[1] == pytest.approx([0.0, 0.0, 1.0], abs=0.05)
    share = an.floor_share(z, Kg, R, got[0], got[1])
    assert 0.0 < share < 1.0
    assert an.floor_share(np.zeros_like(z), Kg, R, 0.5, got[1]) == 0.0
    assert an.own_floor_plane(np.zeros_like(z), Kg, R, 0.5, 0.06) is None
    assert analysis.scale_check(z, Kg, R, 0.5, "sensor")["ratio"] == pytest.approx(1.0, abs=0.03)
