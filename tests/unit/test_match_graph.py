"""The match-graph rule (map.md §2, api.md §4 "Choosing a status"): a job's registration is
order-independent; each connected component of new frames registers as a whole; a frame the
prior checks fail is kept without breaking the chain; a dry run predicts the same statuses.

Capture A u0 of the round-5 replay, synthetic: 640 x 480, fx 655.8 (52° horizontal field of
view), depth images, 8 frames turning about 27° per step (about 48 % overlap), the camera 3.7 cm
off the rotation axis, metric priors of σ 0.02 m / 1°."""

from __future__ import annotations

import functools
import hashlib
import io
import math
from typing import Any

import numpy as np

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.mapping import anchored as an
from tests.synth.room import Room, base_pose, jpeg, png_depth, render
from tests.unit.test_service_api import Client
from tests.unit.test_service_prior_first import client  # noqa: F401

K_A = Intrinsics(655.8, 655.8, 320.0, 240.0, 640, 480)
CAM_A = {"fx": 655.8, "fy": 655.8, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
ROOM = Room()
HEIGHT = 0.49555
GATE = {"max_residual_m": 0.10, "max_gauge_shift_m": 0.05, "max_displacement_m": 0.10,
        "max_displacement_deg": 3.0, "max_rejected_share": 0.5}


@functools.lru_cache(maxsize=64)
def _shot(key: tuple[float, ...]) -> tuple[bytes, bytes, bytes]:
    T = np.array(key).reshape(4, 4)
    rgb, z = render(ROOM, T, K_A, supersample=1)
    buf = io.BytesIO()
    np.save(buf, z)
    return jpeg(rgb), png_depth(z), buf.getvalue()


def capture_a(step: float = 27.0, n: int = 8) -> list[np.ndarray]:
    """The true poses: a turn about a base 3.7 cm behind the camera, level."""
    return [base_pose(step * k, x=0.3, y=-0.2, height=HEIGHT, offset=0.037, pitch_deg=-5.0)
            for k in range(n)]


def noisy(T: np.ndarray, k: int, pos_m: float = 0.005, rot_deg: float = 0.6) -> np.ndarray:
    """A prior a few millimetres and a fraction of a degree off the truth (deterministic)."""
    rng = np.random.default_rng(1000 + k)
    out = T.copy()
    out[:3, 3] += rng.normal(0.0, pos_m / math.sqrt(3), 3)
    w = rng.normal(0.0, math.radians(rot_deg) / math.sqrt(3), 3)
    th = float(np.linalg.norm(w))
    if th > 0:
        a = w / th
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx
        out[:3, :3] = R @ T[:3, :3]
    return out


def request(c: Client, poses: list[np.ndarray], priors: list[np.ndarray] | None = None,
            order: list[int] | None = None, tag: str = "a", sp: float = 0.02, sr: float = 1.0
            ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    frames, files = [], {}
    for i in order if order is not None else range(len(poses)):
        img, dep, z = _shot(tuple(np.round(poses[i], 9).ravel().tolist()))
        c.backend.mono_depth[hashlib.sha256(img).hexdigest()] = np.load(io.BytesIO(z))
        P = (priors or poses)[i]
        files[f"{tag}{i}"], files[f"{tag}{i}d"] = img, dep
        frames.append({"name": f"{tag}{i}", "image": f"{tag}{i}", "depth": f"{tag}{i}d",
                       "camera_id": "cam", "height_above_floor_m": HEIGHT, "height_sigma_m": 0.02,
                       "prior": {"trust": "metric", "T_map_cam": P.tolist(), "sigma_pos_m": sp,
                                 "sigma_rot_deg": sr}})
    return frames, files


def by_name(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {i["name"]: i for i in job["inputs"]}


def test_capture_a_u0_registers_every_frame(client: Client) -> None:  # noqa: F811
    poses = capture_a()
    priors = [noisy(T, k) for k, T in enumerate(poses)]
    fr, files = request(client, poses, priors)
    job = client.update("m", fr, files, cameras={"cam": CAM_A}, accept_if=GATE)
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 8, job["inputs"]
    assert job["state"] == "committed", job["quality"]


def _poses(job: dict[str, Any]) -> dict[str, np.ndarray]:
    return {i["name"]: np.array(i["T_map_cam"]) for i in job["inputs"]
            if i["T_map_cam"] is not None}


def test_a_kept_frame_does_not_break_the_chain(client: Client) -> None:  # noqa: F811
    """Frame a3 is plain (kept at its prior); a4..a7 share no view with a0..a2 (54° and more at a
    52° field of view): their component is placed by its own priors and registers."""
    from tests.fakes.service import plain

    poses = capture_a()
    fr, files = request(client, poses, [noisy(T, k) for k, T in enumerate(poses)])
    files["a3"] = plain(640, 480)
    job = client.update("m", fr, files, cameras={"cam": CAM_A}, accept_if=GATE)
    got = by_name(job)
    assert (got["a3"]["status"], got["a3"]["reason"], got["a3"]["detail"]) == \
        ("kept_at_prior", "plain", None)
    assert [got[f"a{k}"]["status"] for k in (0, 1, 2, 4, 5, 6, 7)] == ["registered"] * 7, got
    assert job["state"] == "committed", job["quality"]
    for k in (4, 5, 6, 7):
        assert np.linalg.norm(np.array(got[f"a{k}"]["T_map_cam"])[:3, 3]
                              - poses[k][:3, 3]) < 0.03


def test_the_outcome_does_not_depend_on_the_request_order(client: Client) -> None:  # noqa: F811
    """A shuffled request: the same statuses, the same poses within noise; its dry run predicts
    the update's statuses."""
    from tests.fakes.service import plain
    from tests.unit.test_service_prior_first import dry

    poses = capture_a()
    priors = [noisy(T, k) for k, T in enumerate(poses)]
    out = []
    for name, order in (("m1", list(range(8))), ("m2", [5, 2, 7, 0, 3, 6, 1, 4])):
        fr, files = request(client, poses, priors, order)
        files["a3"] = plain(640, 480)
        pred = dry(client, name, fr, files, cameras={"cam": CAM_A}, accept_if=GATE)
        job = client.update(name, fr, files, cameras={"cam": CAM_A}, accept_if=GATE)
        assert {r["name"]: (r["predicted_status"], r["predicted_reason"])
                for r in pred["frames"]} == {i["name"]: (i["status"], i["reason"])
                                             for i in job["inputs"]}
        assert pred["quality"]["accepted"] == job["quality"]["accepted"]
        out.append(job)
    a, b = by_name(out[0]), by_name(out[1])
    assert {n: (r["status"], r["reason"]) for n, r in a.items()} == \
        {n: (r["status"], r["reason"]) for n, r in b.items()}
    pa, pb = _poses(out[0]), _poses(out[1])
    for n, T in pa.items():
        dp, dr = an.pose_residual(T, pb[n])
        assert dp < 0.01 and dr < 0.3, (n, dp, dr)


def test_the_dry_run_predicts_an_update_on_a_stored_map(client: Client) -> None:  # noqa: F811
    from tests.unit.test_service_prior_first import dry

    poses = capture_a()
    fr, files = request(client, poses[:4], [noisy(T, k) for k, T in enumerate(poses[:4])])
    assert client.update("m", fr, files, cameras={"cam": CAM_A})["state"] == "committed"
    more = capture_a(step=27.0, n=8)[4:]
    fr, files = request(client, more, [noisy(T, k + 4) for k, T in enumerate(more)], tag="b")
    pred = dry(client, "m", fr, files)
    job = client.update("m", fr, files)
    assert [r["predicted_status"] for r in pred["frames"]] == \
        [i["status"] for i in job["inputs"]] == ["registered"] * 4


def test_the_displacement_of_a_frame_with_a_prior_is_against_that_prior(
        client: Client) -> None:  # noqa: F811
    """map.md §1 "New-frame displacement": a metric prior of 2 cm / 1° and an image-only pose
    0.12 m / 5° off it register (monocular depth: the bound is 3 × sqrt(σ_prior² + σ_image²));
    the job commits, its displacement is the committed pose against the prior, not against the
    image-only pose."""
    from tests.unit.test_service_prior_first import CAM, frames_for

    fr, files = frames_for(client, [base_pose(0), base_pose(30)], tag="a", depth=False)
    assert client.update("m", fr, files, cameras={"cam": CAM})["state"] == "committed"
    T = base_pose(15)
    P = T.copy()
    a = math.radians(5.0)
    P[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                          [0.0, 0.0, 1.0]]) @ T[:3, :3]
    P[:3, 3] += [0.0, 0.12, 0.0]
    fr, files = frames_for(client, [T], tag="b", depth=False, priors=[P])
    job = client.update("m", fr, files, accept_if={"max_displacement_m": 0.10,
                                                   "max_displacement_deg": 3.0})
    i = job["inputs"][0]
    assert i["status"] == "registered" and job["state"] == "committed", job
    q = job["quality"]
    assert (q["max_displacement_m"], q["max_displacement_deg"]) == (
        i["prior_residual_m"], i["prior_residual_deg"])
    assert q["max_displacement_m"] < 0.05


def test_the_height_gate_in_the_job_and_the_version(client: Client) -> None:  # noqa: F811
    """map.md §2: a frame whose committed camera height disagrees with the one sent stays
    registered, is listed in ``quality.height_inconsistent`` and its version row gives no
    occupancy (``derived._occupies``)."""
    from oh_my_slam.service.derived import _occupies

    poses = capture_a(n=3)
    fr, files = request(client, poses)
    fr[2]["height_above_floor_m"] = 0.80
    job = client.update("m", fr, files, cameras={"cam": CAM_A})
    assert [i["status"] for i in job["inputs"]] == ["registered"] * 3
    assert job["quality"]["height_inconsistent"] == ["a2"]
    rows = {r["name"]: r for r in client.service.store.version("m")["frames"]}
    assert rows["a2"]["height_inconsistent"] and not _occupies(rows["a2"])
    assert not rows["a0"]["height_inconsistent"] and _occupies(rows["a0"])
