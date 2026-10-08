"""``segment.sh`` — instance segmentation of an image.

    segment.sh -i <image> [-f json|png] [-o <file>] [-d <folder>] [--min-score <s>]

The result — the OpenLABEL scene (json, default) or the segmented image (png) — goes to stdout, or
to ``-o <file>`` (stdout then stays empty). ``-d <folder>`` also writes the four artefacts:
``segmentation.json`` and ``segmented.png`` are byte-identical to what ``-f json`` and ``-f png``
output for the same run. Without ``-o`` and ``-d`` no file is written.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from oh_my_slam.commands import spec
from oh_my_slam.commands.parser import ArgumentParser, run_main
from oh_my_slam.core import timing
from oh_my_slam.core.log import claim_stdout, get_logger, json_payload_bytes
from oh_my_slam.core.timing import Stage

PROGRAM = spec.SEGMENT
COMMAND = PROGRAM.command()
PROG = PROGRAM.prog
log = get_logger("oh_my_slam.cli.segment")


def build_parser() -> ArgumentParser:
    return spec.build_parser(PROGRAM)


def _segment_image(args: argparse.Namespace, min_score: float, prepare: Callable[[], None]
                   ) -> bytes:
    """The result (``-f``) of segmenting ``args.image``; the ``-d`` artefacts written."""
    from oh_my_slam.reconstruction.api import connect_server
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.artifacts import write_artifacts
    from oh_my_slam.segmentation.render import segmented_png
    from oh_my_slam.segmentation.scene import single_image_scene

    image: Path = args.image  # validated: it exists
    folder: Path | None = args.artifacts
    stage = timing.stage
    with stage(Stage.CONNECT):
        client = connect_server()  # exit 3 when the server is down: nothing written, no -d folder
        prepare()  # creates the -d folder
    with stage(Stage.INFERENCE):
        frame, dets = reconstruct_and_detect(image, client, min_score)
    with stage(Stage.SEGMENT):
        seg = segment_frame(frame, client=client, detections=dets, min_score=min_score)
    with stage(Stage.EXPORT):
        scene = json_payload_bytes(single_image_scene(seg, tool="segment")) \
            if args.format == "json" or folder is not None else b""
        png = segmented_png(frame.rgb, seg.label_map) \
            if args.format == "png" or folder is not None else b""
    if folder is not None:
        with stage(Stage.ARTIFACTS):
            write_artifacts(folder, scene, png, seg.objects, title=f"Objects in {image.name}")
    timing.count(objects=len(seg.objects), detections=len(dets))
    log.info("%d objects", len(seg.objects))
    return scene if args.format == "json" else png


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    # --min-score, -o, -d and -i are checked before any work; the -d folder is created once the
    # inference server is known to be up
    v, prepare = spec.validate_deferring(COMMAND, args, log.warning, defer=(spec.ARTIFACTS_RULE,))
    out = claim_stdout(args.output)
    t0 = time.perf_counter()
    with timing.collect() as tm:
        result = _segment_image(args, v.min_score, prepare)
        with timing.stage(Stage.WRITE):
            out.write_bytes(result)
    log.info("done in %.2f s", time.perf_counter() - t0)
    timing.report(tm, log, command=COMMAND.label(COMMAND.mode_of(args)), format=args.format,
                  image=str(args.image), artifacts=args.artifacts is not None)
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
