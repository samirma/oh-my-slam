"""Record and replay of inference responses, per process, set by the environment.

* ``OH_MY_SLAM_INFERENCE_RECORD=<dir>``: every response the client receives is appended to
  ``<dir>/responses.jsonl``, with the files it names (``*_path`` fields: depth, validity masks)
  copied into ``<dir>``.
* ``OH_MY_SLAM_INFERENCE_REPLAY=<dir>``: the client answers each request from that recording
  instead of the server (which may then be down): the next recorded response to the same route
  and the same request (path fields aside), its files copied out afresh. A request the recording
  does not hold is an :class:`~oh_my_slam.core.errors.InferenceError` naming it.

The web service (spec §2.6) records a job's command step and replays it for the job's viewer step,
so the viewer shows exactly the detections and ids of the result, with no second inference pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from oh_my_slam.core.errors import InferenceError

ENV_RECORD = "OH_MY_SLAM_INFERENCE_RECORD"
ENV_REPLAY = "OH_MY_SLAM_INFERENCE_REPLAY"
INDEX = "responses.jsonl"
_lock = threading.Lock()
_replay: dict[str, dict[str, list[Any]]] = {}  # recording dir → key → responses left


def _pathless(obj: Any) -> Any:
    """``obj`` without its path fields (temporary files differ between runs)."""
    if isinstance(obj, dict):
        return {k: _pathless(v) for k, v in sorted(obj.items())
                if not (k.endswith("_path") or k.endswith("_paths") or k == "out_dir")}
    if isinstance(obj, list):
        return [_pathless(v) for v in obj]
    return obj


def request_key(route: str, request: dict[str, Any]) -> str:
    text = json.dumps([route, _pathless(request)], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _files(obj: Any, fn: Any) -> Any:
    """``obj`` with every ``*_path`` string value mapped through ``fn``."""
    if isinstance(obj, dict):
        return {k: fn(v) if k.endswith("_path") and isinstance(v, str) else _files(v, fn)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_files(v, fn) for v in obj]
    return obj


def replaying() -> Path | None:
    v = os.environ.get(ENV_REPLAY)
    return Path(v) if v else None


def record(route: str, request: dict[str, Any], response: dict[str, Any]) -> None:
    """Append ``response`` to the recording named by ``OH_MY_SLAM_INFERENCE_RECORD``, if any."""
    target = os.environ.get(ENV_RECORD)
    if not target:
        return
    folder = Path(target)
    folder.mkdir(parents=True, exist_ok=True)

    def keep(path: str) -> str:
        src = Path(path)
        if not src.is_file():
            return path
        fd, name = tempfile.mkstemp(dir=folder, prefix="f", suffix=src.suffix)
        os.close(fd)
        shutil.copyfile(src, name)
        return "@" + Path(name).name

    line = json.dumps({"route": route, "key": request_key(route, request),
                       "response": _files(response, keep)})
    with _lock, (folder / INDEX).open("a") as f:
        f.write(line + "\n")


def replay(route: str, request: dict[str, Any], out_dir: str | None) -> dict[str, Any]:
    """The recorded response to this request (files copied into ``out_dir``, else a temporary
    folder), consumed once."""
    folder = replaying()
    assert folder is not None
    with _lock:
        if str(folder) not in _replay:
            index: dict[str, list[Any]] = {}
            try:
                lines = (folder / INDEX).read_text().splitlines()
            except OSError as exc:
                raise InferenceError(f"no inference recording in {folder} ({exc.strerror})"
                                     ) from exc
            for line in lines:
                rec = json.loads(line)
                index.setdefault(rec["key"], []).append(rec["response"])
            _replay[str(folder)] = index
        left = _replay[str(folder)].get(request_key(route, request))
        if not left:
            raise InferenceError(f"{route}: the inference recording in {folder} holds no response "
                                 "to this request (it records another run)")
        response = left.pop(0)
    dest = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="oms-replay-"))
    dest.mkdir(parents=True, exist_ok=True)

    def restore(value: str) -> str:
        if not value.startswith("@"):
            return value
        fd, name = tempfile.mkstemp(dir=dest, suffix=Path(value).suffix)
        os.close(fd)
        shutil.copyfile(folder / value[1:], name)
        return name

    return _files(response, restore)
