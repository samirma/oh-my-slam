"""The mapper process of an update job (``backend.MapperBackend``):

    python -m oh_my_slam.service.mapper_proc <args.json>

``args.json`` names the mapper map, the input images in request order and their intrinsics. It
runs ``mapping.api.update`` and writes the result JSON the args name: the inputs the mapper did
not register and the object count, or the error. Its exit code is the mapper's
(``core.errors.ExitCode``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from oh_my_slam.core.atomic import atomic_write_json
from oh_my_slam.core.errors import DeviceMemoryError, ExitCode, OhMySlamError
from oh_my_slam.core.types import Intrinsics


def _next_index(map_dir: Path) -> int:
    try:
        return int(json.loads((map_dir / "map.json").read_text()).get("next_frame_index", 0))
    except FileNotFoundError:
        return 0


def run(args: dict[str, Any]) -> int:
    from oh_my_slam.mapping.api import update
    from oh_my_slam.mapping.store import frame_name

    map_dir = Path(args["map_dir"])
    inputs = [Path(p) for p in args["inputs"]]
    intrinsics = {Path(p): Intrinsics.from_dict(d) for p, d in args["intrinsics"].items()}
    out = Path(args["result"])
    start = _next_index(map_dir)
    try:
        res = update(map_dir, inputs, intrinsics=intrinsics)
    except OhMySlamError as exc:
        atomic_write_json(out, {"error": {"message": str(exc), "exit_code": int(exc.exit_code),
                                          "out_of_memory": isinstance(exc, DeviceMemoryError)}})
        return int(exc.exit_code)
    source = {frame_name(start + i): str(p) for i, p in enumerate(inputs)}
    scene = json.loads(res.payload)
    atomic_write_json(out, {
        "rejected": [source.get(n, n) for n in res.rejected],
        "objects": len(scene.get("openlabel", {}).get("objects", {})),
        "seconds": res.seconds,
    })
    return int(ExitCode.OK)


def main(argv: list[str]) -> int:
    return run(json.loads(Path(argv[0]).read_text()))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
