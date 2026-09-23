from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oh_my_slam.core.geometry import angle_between_deg
from oh_my_slam.core.types import Intrinsics, Pose
from oh_my_slam.reconstruction.api import reconstruct_image
from oh_my_slam.reconstruction.multiview import run_multiview
from tests.fakes.client import FakeClient, FakeFrame
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


@pytest.fixture
def scene(tmp_path: Path) -> tuple[FakeClient, Path, Pose]:
    room = default_room()
    pose = look_at(np.array([2.5, 2.0, 1.5]), np.array([0.0, 0.0, 0.4]))
    r = render(room, pose, K)
    client = FakeClient()
    up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    img = client.add(tmp_path / "a.png", r.rgb, FakeFrame(r.depth, K, up, pose=pose))
    return client, img, pose


def test_reconstruct_image_model_intrinsics_and_gravity(scene) -> None:  # type: ignore[no-untyped-def]
    client, img, pose = scene
    f = reconstruct_image(img, client=client, want_descriptor=True, want_normals=True)
    assert f.intrinsics.source == "model" and f.intrinsics.fx == pytest.approx(260.0)
    assert f.grid_size == (320, 240)
    assert f.gravity is not None and f.gravity.source == "geocalib+floor"
    true_up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    assert angle_between_deg(f.gravity.up_cam, true_up) < 1.0
    assert f.descriptor is not None and f.normals is not None
    cloud, idx = f.camera_cloud()
    np.testing.assert_array_equal(cloud.rgb, f.rgb.reshape(-1, 3)[idx])
    assert len(f.cloud_in(pose)) == len(cloud)
    assert client.calls["geometry"] == 1 and client.calls["gravity"] == 1


def test_given_intrinsics_are_passed_as_fov_and_kept(scene) -> None:  # type: ignore[no-untyped-def]
    client, img, _ = scene
    given = Intrinsics(300.0, 300.0, 160.0, 120.0, 320, 240, "colmap")
    f = reconstruct_image(img, client=client, intrinsics=given, want_gravity=False)
    assert f.intrinsics == given and f.gravity is None
    assert f.K_grid.fx == pytest.approx(300.0)


def test_work_dir_keeps_server_files(scene, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client, img, _ = scene
    work = tmp_path / "work"
    reconstruct_image(img, client=client, work_dir=work, want_gravity=False)
    assert (work / "depth.npy").exists()


def test_multiview_wrapper(scene, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    client, img, pose = scene
    res, scale = run_multiview([img, img], tmp_path, intrinsics=[K, None], poses=[pose, None],
                               client=client)
    assert len(res) == 2 and scale == 1.0
    np.testing.assert_allclose(res[0].pose.matrix(), pose.matrix(), atol=1e-9)
    assert res[0].K.shape == (3, 3)
