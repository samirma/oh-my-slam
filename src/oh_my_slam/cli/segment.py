"""``segment.sh`` — instance segmentation of an image or of a persisted map.

    segment.sh -i <image> [-f json|ply] [-o <file>] [-d <folder>] [-p <attrs>] [--min-score <s>]
    segment.sh -m <map-folder> [-f json|ply] [-o <file>] [-d <folder>] [-p <attrs>]

The result — the OpenLABEL scene (json, default) or the object-coloured PLY — goes to stdout, or
to ``-o <file>`` (stdout then stays empty). ``-d <folder>`` also writes the five artefacts:
``segmentation.json`` and ``segments.ply`` are byte-identical to what ``-f json`` and ``-f ply``
output for the same run. Without ``-o`` and ``-d`` no file is written. ``-p`` shapes the PLY
(``color`` is fixed to ``segment``) and needs ``-f ply`` or ``-d``; it never changes objects, ids
or colours. ``-m`` runs no inference, needs no inference server and never modifies the map.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from oh_my_slam.cli import spec
from oh_my_slam.cli.common import ArgumentParser, run_main
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import CloudAttrs
from oh_my_slam.core.log import claim_stdout, get_logger, json_payload_bytes
from oh_my_slam.core.timing import Stage

PROGRAM = spec.SEGMENT
COMMAND = PROGRAM.command()
PROG = PROGRAM.prog
log = get_logger("oh_my_slam.cli.segment")


def build_parser() -> ArgumentParser:
    return spec.build_parser(PROGRAM)


def _segment_image(args: argparse.Namespace, attrs: CloudAttrs, min_score: float
                   ) -> tuple[bytes, bytes | None]:
    from oh_my_slam.client.client import connect
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.artifacts import write_artifacts
    from oh_my_slam.segmentation.cloud import cloud_ply, image_cloud_source
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    image: Path = args.image  # validated: it exists
    stage = timing.stage
    with stage(Stage.CONNECT):
        client = connect()
    with stage(Stage.INFERENCE):
        frame, dets = reconstruct_and_detect(image, client, min_score)
    with stage(Stage.SEGMENT):
        seg = segment_frame(frame, client=client, detections=dets, min_score=min_score)
    with stage(Stage.EXPORT):
        scene = json_payload_bytes(single_image_scene(seg, tool="segment"))
        ply = cloud_ply(image_cloud_source(frame, seg), attrs) \
            if args.format == "ply" or args.artifacts is not None else None
    if args.artifacts is not None:
        assert ply is not None
        with stage(Stage.ARTIFACTS):
            write_artifacts(args.artifacts, scene, segmented_image(frame.rgb, seg.label_map),
                            seg.objects, ply, title=f"Objects in {image.name}")
    timing.count(objects=len(seg.objects), detections=len(dets))
    log.info("%d objects", len(seg.objects))
    return scene, ply


def _result(fmt: str, scene: bytes, ply: bytes | None) -> bytes:
    if fmt == "json":
        return scene
    assert ply is not None
    return ply


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    mode = COMMAND.mode_of(args)
    on_map = mode.name == "map"
    # --min-score, -p, -o, -d and -i are checked (and -d created) before any work
    v = spec.validate(COMMAND, args, log.warning)
    attrs: CloudAttrs = v.attrs
    out = claim_stdout(args.output)
    t0 = time.perf_counter()
    if on_map:
        from oh_my_slam.mapping.export import map_segment_outputs

        with timing.collect() as tm:
            with timing.stage(Stage.EXPORT):
                scene, ply = map_segment_outputs(args.map, args.artifacts, attrs,
                                                 want_ply=args.format == "ply")
            with timing.stage(Stage.WRITE):
                out.write_bytes(_result(args.format, scene, ply))
        log.info("done in %.2f s", time.perf_counter() - t0)
        timing.report(tm, log, command=COMMAND.label(mode), format=args.format, map=str(args.map),
                      artifacts=args.artifacts is not None)
        return 0
    with timing.collect() as tm:
        scene, ply = _segment_image(args, attrs, v.min_score)
        with timing.stage(Stage.WRITE):
            out.write_bytes(_result(args.format, scene, ply))
    log.info("done in %.2f s", time.perf_counter() - t0)
    timing.report(tm, log, command=COMMAND.label(mode), format=args.format, image=str(args.image),
                  artifacts=args.artifacts is not None)
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
