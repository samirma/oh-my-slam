"""The ``server.sh`` web service's viewer step: save the viewer of a command's request.

    python -m oh_my_slam.cli.view_save <folder> <prog> <argv…>

``<prog> <argv…>`` is the command line of ``view.sh`` (a ``view-image`` / ``view-map`` job) or of
an image command whose job asked for its viewer (``reconstruct.sh``, ``segment.sh -i``). It is
parsed and validated by that command's own definitions (``commands.spec``), and the bundle is the
one ``view.sh`` would serve (``cli.view.make_bundle``), built with the command's own segmentation
options; it is saved into ``<folder>`` (``viewer.bundle.save_bundle``), a map as a reference. Run
after the command step with ``OH_MY_SLAM_INFERENCE_REPLAY`` set, it reuses the command's recorded
inference (``client.replay``), so the viewer shows the result's own detections and ids.
"""

from __future__ import annotations

from pathlib import Path

from oh_my_slam.cli.common import run_main
from oh_my_slam.cli.view import make_bundle
from oh_my_slam.commands import spec
from oh_my_slam.core.errors import UsageError
from oh_my_slam.core.log import claim_stdout, get_logger

log = get_logger("oh_my_slam.cli.view_save")


def main(argv: list[str]) -> int:
    from oh_my_slam.viewer.bundle import save_bundle, save_map_reference

    if len(argv) < 2:
        raise UsageError("usage: view_save <folder> <prog> <argv…>")
    folder, prog, rest = Path(argv[0]), argv[1], argv[2:]
    program = next((p for p in spec.PROGRAMS if p.prog == prog), None)
    if program is None:
        raise UsageError(f"unknown command {prog}")
    args = spec.build_parser(program).parse_args(rest)
    command = program.command(getattr(args, "command", None))
    values = spec.validate(command, args, log.warning)
    claim_stdout()  # nothing reaches stdout
    if getattr(values, "map", None) is not None:
        save_map_reference(values.map, folder)  # a persisted map is its own bundle
        return 0
    save_bundle(make_bundle(values), folder)
    return 0


if __name__ == "__main__":
    run_main("view.sh", main)
