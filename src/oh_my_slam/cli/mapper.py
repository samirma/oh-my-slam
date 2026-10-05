"""``mapper.sh update`` — build and update a persistent map; ``mapper.sh locate`` — the camera pose
of images in an existing map, which it never modifies.

    mapper.sh update -i <image(s)|video> -m <folder> [-f json|ply] [-o <file>]
                     [-p <attrs>] [-t full|single] [-fps <n>]

Creates the map if <folder> is missing or empty, extends it if it is a map, refuses any other
non-empty folder (exit 4). Only -i and -m are required. The result goes to stdout, or to
``-o <file>`` (stdout then stays empty): the OpenLABEL scene (json, default) of the whole map
(-t full, default, with every keyframe pose) or of the new input only (-t single); -f ply writes
the map cloud (full) or the new frames' points (single), map coordinates, shaped by the ``-p``
point-cloud attributes (map scope: the pixel-level keys are refused). -fps (default 2) samples
video input and is ignored for images. Options are validated before the server is contacted.

    mapper.sh locate -i <image(s)> -m <map> [-f json|ply] [-o <file>] [-p <attrs>] [-t full|single]

Registers each image against the whole map (``mapping.locate``). -t single (default): the located
camera poses only (PLY: the map points visible from them); -t full: the map's scene as ``update -t
full`` returns it plus the located poses (PLY: the whole map cloud). A video, a missing or empty
-m folder, and an -o inside the map are refused (exit 2); a non-empty folder that is not a map exit
4; no image located exit 2. The inference server is needed only for retrieval in maps of more than
``UPDATE_EXHAUSTIVE_MAX`` keyframes (exit 3 when it is down).
"""

from __future__ import annotations

import argparse

from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.commands import spec
from oh_my_slam.core.log import claim_stdout, get_logger

PROGRAM = spec.MAPPER
UPDATE, LOCATE = PROGRAM.command("update"), PROGRAM.command("locate")
PROG = PROGRAM.prog
log = get_logger("oh_my_slam.cli.mapper")


def build_parser() -> ArgumentParser:
    return spec.build_parser(PROGRAM)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "locate":
        return _locate(args)
    # -p, -o, -fps, the inputs and the map are checked before the server is contacted
    v = spec.validate(UPDATE, args, log.warning)
    out = claim_stdout(args.output)
    from oh_my_slam.core import timing
    from oh_my_slam.mapping.api import update

    res = update(args.map, args.inputs, fps=v.fps, mode=args.mode, fmt=args.format, attrs=v.attrs)
    out.write_bytes(res.payload)
    timing.report(res.timings, log, command=UPDATE.label(UPDATE.mode(None)), mode=args.mode,
                  format=args.format, fps=v.fps if v.is_video else None, map=str(args.map))
    return 0


def _locate(args: argparse.Namespace) -> int:
    from oh_my_slam.core import timing
    from oh_my_slam.mapping.locate import locate

    # -p, the inputs, the map and -o are checked before -o is prepared (which creates its folder)
    v = spec.validate(LOCATE, args, log.warning)
    out = claim_stdout(args.output)
    res = locate(v.reader, v.images, mode=args.mode, fmt=args.format, attrs=v.attrs)
    out.write_bytes(res.payload)
    timing.report(res.timings, log, command=LOCATE.label(LOCATE.mode(None)), mode=args.mode,
                  format=args.format, map=str(args.map))
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
