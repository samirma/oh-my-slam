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

from oh_my_slam.cli.common import (
    ArgumentParser,
    add_result_options,
    attrs_help,
    cloud_attrs_arg,
    run_main,
)
from oh_my_slam.core import timing
from oh_my_slam.core.cloud_attrs import CloudAttrs, CloudScope
from oh_my_slam.core.errors import InputError, UsageError
from oh_my_slam.core.log import claim_stdout, get_logger, json_payload_bytes

PROG = "segment.sh"
log = get_logger("oh_my_slam.cli.segment")


def _score(value: str) -> float:
    from oh_my_slam.segmentation.detect import DETECTION_FLOOR

    try:
        s = float(value)
    except ValueError as exc:
        raise UsageError(f"--min-score must be a number, got {value!r}") from exc
    if not DETECTION_FLOOR <= s <= 1.0:
        raise UsageError(f"--min-score must be within [{DETECTION_FLOOR}, 1] (the detector "
                         f"reports no scores below {DETECTION_FLOOR}), got {value}")
    return s


def build_parser() -> ArgumentParser:
    ap = ArgumentParser(prog=PROG, description="Instance segmentation → JSON + OBBs, artefacts.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", dest="image", type=Path, help="input RGB image")
    src.add_argument("-m", dest="map", type=Path, help="existing map folder (read-only)")
    ap.add_argument("-f", dest="format", choices=("json", "ply"), default="json",
                    help="output format (default: json)")
    add_result_options(ap, attrs_help(
        CloudScope.IMAGE | CloudScope.SEGMENT,
        "shapes the -f ply output and segments.ply, so it needs -f ply or -d; with -m the "
        "pixel-level keys (stride, min-depth, max-depth, edge) are refused"))
    ap.add_argument("-d", dest="artifacts", type=Path, metavar="FOLDER",
                    help="also write segmentation.json, segmented.png, catalog.csv, catalog.md "
                         "and segments.ply into FOLDER")
    ap.add_argument("--min-score", dest="min_score", default=None,
                    help="drop detections below this score (default 0.5; -i only)")
    return ap


def _segment_image(args: argparse.Namespace, attrs: CloudAttrs, min_score: float
                   ) -> tuple[bytes, bytes | None]:
    from oh_my_slam.client.client import connect
    from oh_my_slam.segmentation.api import reconstruct_and_detect, segment_frame
    from oh_my_slam.segmentation.artifacts import write_artifacts
    from oh_my_slam.segmentation.cloud import cloud_ply, image_cloud_source
    from oh_my_slam.segmentation.render import segmented_image
    from oh_my_slam.segmentation.scene import single_image_scene

    image: Path = args.image
    if not image.is_file():
        raise InputError(f"image not found: {image}")
    stage = timing.stage
    with stage("connect"):
        client = connect()
    with stage("inference"):
        frame, dets = reconstruct_and_detect(image, client)
    with stage("segment"):
        seg = segment_frame(frame, client=client, detections=dets, min_score=min_score)
    with stage("export"):
        scene = json_payload_bytes(single_image_scene(seg, tool="segment"))
        ply = cloud_ply(image_cloud_source(frame, seg), attrs) \
            if args.format == "ply" or args.artifacts is not None else None
    if args.artifacts is not None:
        assert ply is not None
        with stage("artifacts"):
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
    on_map = args.map is not None
    if on_map and args.min_score is not None:
        raise UsageError("-m exports the map's persistent objects; --min-score applies to -i only")
    min_score = 0.5 if args.min_score is None else _score(args.min_score)
    scope = (CloudScope.MAP if on_map else CloudScope.IMAGE) | CloudScope.SEGMENT
    attrs = cloud_attrs_arg(args.attrs, scope,
                            writes_ply=args.format == "ply" or args.artifacts is not None,
                            requires="shape the PLY output: use -f ply or -d <folder>")
    out = claim_stdout(args.output)
    t0 = time.perf_counter()
    if on_map:
        from oh_my_slam.mapping.export import map_segment_outputs

        with timing.collect() as tm:
            with timing.stage("export"):
                scene, ply = map_segment_outputs(args.map, args.artifacts, attrs,
                                                 want_ply=args.format == "ply")
            with timing.stage("write"):
                out.write_bytes(_result(args.format, scene, ply))
        log.info("done in %.2f s", time.perf_counter() - t0)
        timing.report(tm, log, command="segment.sh -m", format=args.format, map=str(args.map),
                      artifacts=args.artifacts is not None)
        return 0
    with timing.collect() as tm:
        scene, ply = _segment_image(args, attrs, min_score)
        with timing.stage("write"):
            out.write_bytes(_result(args.format, scene, ply))
    log.info("done in %.2f s", time.perf_counter() - t0)
    timing.report(tm, log, command="segment.sh -i", format=args.format, image=str(args.image),
                  artifacts=args.artifacts is not None)
    return 0


def entry() -> None:
    run_main(PROG, main)


if __name__ == "__main__":
    entry()
