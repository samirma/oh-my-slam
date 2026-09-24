"""Synthetic inputs for the viewer tests (unit and browser): one rendered image answered by the fake
inference client, a small map built by the real mapper (COLMAP) from rendered frames, and a
running viewer server."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from oh_my_slam.core.types import Intrinsics
from oh_my_slam.viewer.bundle import ViewBundle
from oh_my_slam.viewer.server import serve, url_of
from tests.fakes.client import FakeClient, FakeFrame, FakeInstance
from tests.synth.scene import default_room, look_at, render

K = Intrinsics(260.0, 260.0, 160.0, 120.0, 320, 240)


def synthetic_image(folder: Path) -> tuple[Path, FakeClient]:
    """The default room seen from one camera; its three boxes are the detections."""
    room = default_room()
    pose = look_at(np.array([2.6, 2.2, 1.6]), np.array([0.0, 0.0, 0.4]))
    r = render(room, pose, K)
    inst = [FakeInstance(b.label, 0.9 - 0.15 * k, r.ids == k + 2)
            for k, b in enumerate(room.boxes)]
    client = FakeClient()
    up = pose.R.T @ np.array([0.0, 0.0, 1.0])
    img = client.add(folder / "room.png", r.rgb, FakeFrame(r.depth, K, up, inst, pose=pose))
    return img, client


def synthetic_map(folder: Path) -> Path:
    """A 12-keyframe map of the mapping room (needs Homebrew colmap)."""
    from oh_my_slam.mapping.api import update
    from tests.synth.mapping import add_frames, mapping_room, ring

    client = FakeClient()
    add_frames(client, mapping_room(), ring(12), folder / "in", "v", seed=11)
    update(folder / "map", [folder / "in"], client=client, progress=lambda m: None)
    return folder / "map"


@contextmanager
def running(bundle: ViewBundle) -> Iterator[str]:
    """Serve ``bundle`` on a free port; yields the URL."""
    httpd = serve(bundle, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield url_of(httpd)
    finally:
        httpd.shutdown()
        httpd.server_close()
