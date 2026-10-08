"""The evaluation plan (high_level_spec.md §5): every entry point on the files in ``examples/``,
strictly one command at a time.

1. ``start_inference_server.sh``: stop, cold start, resident memory (the server's initial state is
   restored at the end).
2. ``restaurant.jpg``: ``reconstruct.sh`` (JSON, PLY, ``-o`` PLY coloured by segment, the depth
   image), ``segment.sh -i`` (JSON with the artefacts, the segmented image with the artefacts)
   and ``view.sh -i``.
3. Each capture sequence (``SEQUENCES``: ``ainex-captures``, then ``camera``), its names and
   metrics as ``CaptureSequence`` gives them:

   a. every frame: ``segment.sh -i``;
   b. ``mapper.sh update``: the sequence in one update (default options: the whole map as JSON),
      and split across ``SPLITS`` updates into a second map (first update ``-o`` JSON, middle
      ones ``-t single -f ply``, last ``-t full``). Between the first and the second update,
      ``mapper.sh locate`` of the second update's images (held out of the map):
      ``pose.<key>locate.*``. What each earlier update published is kept for the stability
      metrics: the first's objects, and the objects of the ids a middle update's points carry
      (read from the map as that update left it);
   c. on the one-update map: ``view.sh -m`` and ``mapper.sh locate`` of three of its captures
      (``locate_picks``: ``-t single`` JSON, ``-t full`` JSON, ``-f ply -o``), which must leave
      it unchanged, hidden entries included.
4. ``office_sequence``: ``mapper.sh update`` of the whole sequence in one update, and split as
   its annotation says (4+4+5 and 6+7), judged by ``mapupdate`` (the ``map_update.*`` metrics).
5. Ground truth: ``segment.sh -i`` on every other example image an ``objects`` annotation names.
6. ``street2.mp4`` (outside the repository, ``--street2``): ``mapper.sh update`` of the video at
   the default sampling rate (performance, contracts, ``pose.street2.registered_fraction``), into
   ``server.sh``'s workspace and through the recording inference proxy, so that it is also the
   command's result its ``server.sh`` request is compared with.
7. ``server.sh`` over a scratch workspace (``service``): performance, parity with the commands on
   every reference input, and the UI (``server_sh.*``).

Every output is checked against the contracts; the metrics are computed from the outputs.
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oh_my_slam.commands import spec
from oh_my_slam.core import paths
from oh_my_slam.core.errors import OhMySlamError
from oh_my_slam.core.images import load_rgb, upright_size
from oh_my_slam.core.ply import parse_header, parse_ply
from oh_my_slam.core.types import Pose
from oh_my_slam.tools.evaluate import groundtruth as gt
from oh_my_slam.tools.evaluate import mapupdate, service
from oh_my_slam.tools.evaluate.contracts import (
    ContractLog,
    artifact_problems,
    catalog_csv_problems,
    catalog_md_problems,
    cloud_colour_problems,
    depth_image_problems,
    parse_scene,
    payload_problems,
    png_colour_problems,
    same_objects_problems,
    scene_colour_problems,
    served_cloud_problems,
    subject_of,
    tree_digest,
)
from oh_my_slam.tools.evaluate.locate import (
    LOCATE_METRICS,
    held_out_metrics,
    located_poses,
    located_problems,
)
from oh_my_slam.tools.evaluate.mapquality import (
    AGREEMENT_METRICS,
    STABILITY_METRICS,
    TILT_AGREEMENT_METRICS,
    Published,
    agreement_metrics,
    split_alignment,
    stability_metrics,
    update_alignment,
)
from oh_my_slam.tools.evaluate.memory import phys_footprint_gb, server_pid
from oh_my_slam.tools.evaluate.metrics import Metrics
from oh_my_slam.tools.evaluate.names import (
    AnyCapture,
    Capture,
    PanCapture,
    captures_in,
    pan_captures_in,
    same_heading_pairs,
    tilt_pairs,
)
from oh_my_slam.tools.evaluate.performance import perf_ids, perf_metrics
from oh_my_slam.tools.evaluate.poses import (
    PAN_POSE_METRICS,
    POSE_METRICS,
    capture_poses,
    pan_pose_metrics,
    pose_metrics,
)
from oh_my_slam.tools.evaluate.runner import REPO, Runner, RunRecord, RunSpec
from oh_my_slam.tools.evaluate.scene import DocObject, Json, doc_objects
from oh_my_slam.tools.evaluate.segmentation import detection_row
from oh_my_slam.tools.evaluate.viewer import BrowserProbe

EXAMPLES = REPO / "examples"
RESTAURANT = "restaurant.jpg"
SEQUENCE = "ainex-captures"
OFFICE = gt.MAP_UPDATE_SEQUENCE  # examples/office_sequence
OFFICE_MAP = "office"  # its one-update map, under maps/
GROUND_TRUTH = "ground_truth"
SERVER = "start_inference_server.sh"
# street2.mp4: every benchmark maps it (the user's rule, 2026-10-02); it lives outside examples/
STREET2 = Path.home() / "oh-my-slam-data" / "loop" / "inputs" / "street2.mp4"
STREET2_MAP = "street2"  # its map, in server.sh's workspace (service.workspace_root)
LABELLED_SEGMENTS = "color=segment,label=on"  # per-point object ids: exact colour checks
SPLITS = 3  # updates of the split map (spec §5: one update versus several)
SERVER_METRICS = ("perf.server.cold_start_s", "perf.server.resident_gb", "perf.server.peak_gb")
SEG_METRICS = ("seg.restaurant.objects", "seg.min_score")
PARITY_SEQUENCE = 3  # the first captures, mapped by the parity cases of mapper.sh update
STREET2_METRIC = "pose.street2.registered_fraction"


@dataclass(frozen=True)
class CaptureSequence[C: (Capture, PanCapture)]:
    """A capture sequence of spec §5 (a folder of ``examples/``): its file-name grammar, its pose
    metrics, its same-heading pairs and how their depth agreement is judged (``tilts``: like the
    overlapping pairs, ``frame_agreement_tilt_*``; else by the worst pair), and the names of its
    runs, maps, metrics and details.
    ``ainex-captures`` has the bare names (``segment_frames``, ``mapper_single``,
    ``pose.single.*``, ``map.stability.*``, ``view_map``, ``locate``, ``pose.locate.*``); another
    sequence puts its ``key`` in front of the part that names its data (``camera_``:
    ``segment_camera_frames``, ``mapper_camera_single``, ``pose.camera_single.*``,
    ``map.camera_stability.*``, ``view_camera_map``, ``camera_locate``,
    ``pose.camera_locate.*``)."""

    folder: str  # under examples/
    key: str
    read: Callable[[Path], list[C]]  # the folder's captures in capture order (names)
    pose_ids: tuple[str, ...]
    judge_poses: Callable[[Metrics, str, dict[str, Pose], list[C]], list[dict[str, Any]]]
    same_heading: Callable[[list[C]], list[tuple[C, C]]]  # pairs the frame agreement compares
    tilts: bool = False  # the same-heading pairs are tilt pairs (mapquality.agreement_metrics)

    @property
    def agreement_ids(self) -> tuple[str, ...]:
        """The ``map.<map>.*`` frame-agreement metrics."""
        return TILT_AGREEMENT_METRICS if self.tilts else AGREEMENT_METRICS

    @property
    def maps(self) -> tuple[str, str]:
        """The one-update and the split map: their folders under ``maps/``, their ``mapper_<map>``
        runs and their ``pose.<map>.*`` / ``map.<map>.*`` metrics."""
        return f"{self.key}single", f"{self.key}split"

    @property
    def frames(self) -> str:
        """``segment.sh -i`` per frame: the ``segment_<frames>`` group, ``seg.<frames>.*``."""
        return f"{self.key}frames"

    @property
    def seg_metric(self) -> str:
        return f"seg.{self.frames}.with_detections_fraction"

    @property
    def stability(self) -> str:
        """The one-update map against the split map: the metrics' prefix, the detail's key."""
        return f"map.{self.key}stability"

    @property
    def view(self) -> str:
        """``view.sh -m`` on the one-update map: its run and group."""
        return f"view_{self.key}map"

    @property
    def located(self) -> str:
        """``mapper.sh locate``: its runs' prefix, the group on the one-update map (performance)
        and the ``pose.<located>.*`` metrics of the held-out captures."""
        return f"{self.key}locate"

    def named(self, what: str) -> str:
        """The name of a contract check about this sequence's maps."""
        return f"{what} ({self.folder})" if self.key else what

    def metric_ids(self) -> list[str]:
        """Every metric the sequence's runs record (besides performance and contracts)."""
        ids = [self.seg_metric]
        ids += [f"pose.{mp}.{k}" for mp in self.maps for k in self.pose_ids]
        ids += [f"pose.{self.located}.{k}" for k in LOCATE_METRICS]
        ids += [f"map.{mp}.{k}" for mp in self.maps for k in self.agreement_ids]
        ids += [f"{self.stability}.{k}" for k in STABILITY_METRICS]
        return ids


AINEX = CaptureSequence(SEQUENCE, "", captures_in, POSE_METRICS, pose_metrics,
                        same_heading_pairs)
CAMERA = CaptureSequence("camera", "camera_", pan_captures_in, PAN_POSE_METRICS,
                         pan_pose_metrics, tilt_pairs, tilts=True)  # user ruling 2026-10-08
SEQUENCES: tuple[CaptureSequence[Any], ...] = (AINEX, CAMERA)


def locate_picks[T](captures: Sequence[T]) -> list[T]:
    """The captures ``mapper.sh locate`` finds in a sequence's one-update map: the first, the one
    halfway and the one three quarters through (``ainex-captures``: 001, 040, 060)."""
    n = len(captures)
    return [captures[i] for i in sorted({0, n // 2, 3 * n // 4})] if n else []


def sequence_input(name: str, files: Sequence[Path], map_dir: Path | None) -> dict[str, Any]:
    """An example sequence as a reference input of ``server.sh``'s parity cases
    (``service.Inputs``): its first image (the one-image operations), its first
    ``PARITY_SEQUENCE`` images (to map), the images it locates in its one-update map but the first
    (``locate_picks``) and that map (when it was built)."""
    return {"name": name, "image": files[0] if files else None,
            "images": locate_picks(list(files))[1:], "sequence": list(files[:PARITY_SEQUENCE]),
            "map": map_dir if map_dir is not None and (map_dir / "map.json").is_file() else None}


def office_splits(examples: Path = EXAMPLES) -> list[str]:
    """The names of the office sequence's splits its annotation asks for."""
    files, skipped = gt.discover(examples / GROUND_TRUTH)
    plan = gt.map_update_plan(files, skipped)
    if plan is None or not (examples / plan.sequence).is_dir():
        return []
    images = mapupdate.sequence_images(examples / plan.sequence)
    return [mapupdate.split_name(s) for s in plan.split_sizes(images)]


def expected_ids(examples: Path = EXAMPLES) -> list[str]:
    """Every metric a run records (ground-truth metrics come on top when annotations exist, and
    per-stage metrics for every stage the commands record)."""
    ids = [*SERVER_METRICS, *perf_ids(), *SEG_METRICS]
    for seq in SEQUENCES:
        ids += seq.metric_ids()
    ids += [STREET2_METRIC]
    ids += mapupdate.metric_ids(office_splits(examples))
    ids += service.metric_ids()
    ids += [mid for mid, *_ in ContractLog().results()]
    return ids


def _problems_reading(fn: Any, *args: Any) -> list[str]:
    """``fn(*args)``, with a missing or unreadable file reported as a problem."""
    try:
        return list(fn(*args))
    except (OSError, ValueError, KeyError) as exc:
        return [f"{type(exc).__name__}: {exc}"]


def persisted_scene(map_dir: Path) -> Json:
    """The map's whole scene as ``mapper.sh update -t full`` returns it, read from the map as it
    is now (read-only: the code of ``mapper.sh locate -t full``)."""
    from oh_my_slam.mapping import export, store

    doc: Json = json.loads(export.scene_bytes(store.MapReader(map_dir)))
    return doc


def _point_labels(rec: RunRecord) -> set[int]:
    """The object ids a successful ``-f ply`` run's points carry (none without a label)."""
    try:
        labels = parse_ply(rec.stdout_bytes()).label if rec.ok else None
    except ValueError:
        return set()
    return set() if labels is None else {int(v) for v in np.unique(labels) if v > 0}


def _pose_lines(ply: bytes, asked: int) -> list[str]:
    """``mapper.sh locate -f ply``: one ``located_<k> {json}`` header line per input image."""
    lines = [c for c in parse_header(ply).comments if c.startswith("located_")]
    out = [] if len(lines) == asked else [f"{len(lines)} located_ header lines for {asked} "
                                          "input images"]
    for line in lines:
        try:
            d = json.loads(line.split(" ", 1)[1])
        except (IndexError, ValueError):
            out.append(f"not a pose line: {line[:80]}")
            continue
        if d.get("located") and "transform_src_to_dst" not in d:
            out.append(f"{line.split(' ', 1)[0]}: located but no transform")
    return out


@dataclass
class Evaluation:
    out: Path
    runner: Runner
    probe: BrowserProbe
    examples: Path = EXAMPLES
    metrics: Metrics = field(default_factory=Metrics)
    contracts: ContractLog = field(default_factory=ContractLog)
    details: dict[str, Any] = field(default_factory=dict)
    images: dict[str, list[DocObject]] = field(default_factory=dict)  # segment.sh -i objects
    # each sequence's one-update map (None: not built): its poses by capture name
    single_poses: dict[str, dict[str, Pose] | None] = field(default_factory=dict)
    was_running: bool = False
    street2: Path | None = STREET2
    # what each split map's earlier updates published: their objects, with that map's poses
    published: dict[str, list[mapupdate.MapView]] = field(default_factory=dict)
    # each sequence's held-out locate (by folder): its located poses, reference pose, captures
    held_out: dict[str, dict[str, Any]] = field(default_factory=dict)
    ui: Any = None  # the web application's browser tests (None: service.run_ui_tests)
    # the recording inference proxy's environment (``recording_proxy``; empty: none runs)
    proxy_env: dict[str, str] = field(default_factory=dict)
    # street2.mp4 in server.sh's workspace, and its run there, as server.sh's parity reuses it
    street2_placed: Path | None = None
    street2_ran: service.Ran | None = None

    # -- running and checking one command ------------------------------------------------------------

    def run(self, tag: str, group: str, entry: str, *args: str | Path, stdout: str = "json",
            output: Path | None = None, ok_exit: tuple[int, ...] = (0,),
            timeout_s: float = 3600.0, env: dict[str, str] | None = None) -> RunRecord:
        kind = None if output is None else output.suffix.lstrip(".")
        rec = self.runner.run(RunSpec(tag, group, entry, tuple(str(a) for a in args), stdout,
                                      output, kind, ok_exit, timeout_s,
                                      tuple((env or {}).items())))
        self.check_payloads(rec)
        return rec

    def check_payloads(self, rec: RunRecord) -> None:
        """stdout purity: one payload on success, nothing on failure (and nothing with ``-o``,
        whose file then holds exactly one payload)."""
        spec = rec.spec
        succeeded = rec.error is None and rec.exit_code == 0
        problems = payload_problems(rec.stdout_bytes(), spec.stdout if succeeded else "empty")
        if succeeded and spec.output is not None and spec.output_kind is not None:
            data = spec.output.read_bytes() if spec.output.exists() else b""
            problems += [f"-o file: {p}" for p in payload_problems(data, spec.output_kind)]
        self.contracts.check("stdout", subject_of(spec.entry), rec.tag, problems)

    def result_bytes(self, rec: RunRecord) -> bytes:
        """The run's result: its ``-o`` file (empty if missing) or its stdout."""
        out = rec.spec.output
        if out is None:
            return rec.stdout_bytes()
        return out.read_bytes() if out.exists() else b""

    def scene(self, rec: RunRecord) -> Json | None:
        """The OpenLABEL result of a successful run, checked for validity and colours."""
        if not rec.ok:
            return None
        subject = subject_of(rec.spec.entry)
        doc, problems = parse_scene(self.result_bytes(rec))
        self.contracts.check("openlabel", subject, rec.tag, problems)
        if doc is None:
            return None
        self.contracts.check("colour", subject, rec.tag, scene_colour_problems(doc_objects(doc)))
        return doc

    def cloud_colours(self, rec: RunRecord, ids: set[int] | None) -> None:
        """Colour contract of a ``color=segment`` PLY result."""
        if rec.ok:
            self.contracts.check("colour", subject_of(rec.spec.entry), rec.tag, _problems_reading(
                lambda: cloud_colour_problems(parse_ply(self.result_bytes(rec)), ids)))

    def artefacts(self, rec: RunRecord, folder: Path, doc: Json) -> None:
        """The four ``-d`` artefacts of a ``-f json`` run: exactly those files, the JSON identical
        to stdout, and every colour in them the object colours of ``doc``."""
        objs = doc_objects(doc)
        self.contracts.check("artifacts", "segment", rec.tag,
                             _problems_reading(artifact_problems, folder, rec.stdout_bytes()))
        problems = _problems_reading(
            lambda: png_colour_problems(load_rgb(folder / "segmented.png"), objs))
        problems += _problems_reading(
            lambda: catalog_csv_problems((folder / "catalog.csv").read_text(), objs))
        problems += _problems_reading(
            lambda: catalog_md_problems((folder / "catalog.md").read_text(), objs))
        self.contracts.check("colour", "segment", f"{rec.tag}/artefacts", problems)

    def segmented_image(self, rec: RunRecord, folder: Path) -> None:
        """A ``segment.sh -f png -d`` run: its ``segmented.png`` is the PNG on stdout, byte for
        byte, and that image's mask colours are those of the objects its ``segmentation.json``
        lists (the same run)."""
        if not rec.ok:
            return
        data = rec.stdout_bytes()
        self.contracts.check("artifacts", "segment", rec.tag,
                             _problems_reading(artifact_problems, folder, data, "png"))

        def colours() -> list[str]:
            doc = json.loads((folder / "segmentation.json").read_bytes())
            with Image.open(io.BytesIO(data)) as img:
                rgb = np.asarray(img.convert("RGB"))
            return png_colour_problems(rgb, doc_objects(doc))

        self.contracts.check("colour", "segment", f"{rec.tag}/segmented image",
                             _problems_reading(colours))

    def view(self, tag: str, source: tuple[str, str, Json | None], *args: str | Path) -> None:
        """``view.sh`` with ``args``: time to the rendered page, and the scene it draws (OBB
        colours) checked against the contracts and against ``source`` = (same-objects subject,
        what the scene is compared with, that output's scene)."""
        spec = RunSpec(tag, tag, "view.sh", (*(str(a) for a in args), "--no-browser"),
                       stdout="empty", ok_exit=(0, 130), timeout_s=900.0)
        seen = self.probe.measure(self.runner, spec)
        rec = seen.record
        if seen.error is not None:
            rec.notes["render_error"] = seen.error
        self.check_payloads(rec)
        self.details[f"viewer.{tag}"] = {"url": seen.url, "render_s": seen.render_s,
                                         "error": seen.error,
                                         "console_errors": (seen.console_errors or [])[:10]}
        if seen.scene is None:
            return
        doc, problems = parse_scene(seen.scene)
        self.contracts.check("openlabel", "view", tag, problems)
        if doc is None:
            return
        objs = doc_objects(doc)
        self.contracts.check("colour", "view", tag, scene_colour_problems(objs))
        self.contracts.check("colour", "view", f"{tag}/cloud color=segment",
                             served_cloud_problems(seen.cloud, {o.id for o in objs}))
        subject, name, other = source
        if other is not None:
            self.contracts.check("same_objects", subject, f"view.sh vs {name}",
                                 same_objects_problems(doc_objects(other), objs,
                                                       geometry=subject == "map"))

    # -- sections --------------------------------------------------------------------------------------

    def server_start(self) -> None:
        first = self.run("server_status_initial", "server", SERVER, "--status", ok_exit=(0, 3))
        self.was_running = first.exit_code == 0
        self.run("server_stop", "server", SERVER, "--stop", stdout="empty")
        cold = self.run("server_cold_start", "server", SERVER, stdout="empty", timeout_s=1800.0)
        if cold.ok:
            pid = server_pid()
            fp = phys_footprint_gb(pid) if pid is not None else None
            self.metrics.add(SERVER_METRICS[0], cold.wall_s)
            self.metrics.add(SERVER_METRICS[1], None if fp is None else fp[0], {"pid": pid},
                             error="server footprint unavailable")
        else:
            self.metrics.fail(SERVER_METRICS[:2], cold.failure())
        status = self.run("server_status", "server", SERVER, "--status")
        if status.ok:
            self.details["server_health"] = json.loads(status.stdout_bytes())

    def server_restore(self) -> None:
        pid = server_pid()
        fp = phys_footprint_gb(pid) if pid is not None else None
        self.metrics.add(SERVER_METRICS[2], None if fp is None else fp[1], {"pid": pid},
                         error="the server is not running at the end of the evaluation")
        if self.was_running and pid is None:
            self.run("server_restart", "server", SERVER, stdout="empty", timeout_s=1800.0)
        elif not self.was_running:
            self.run("server_stop_final", "server", SERVER, "--stop", stdout="empty")

    def restaurant(self) -> None:
        img = self.examples / RESTAURANT
        outputs = self.out / "outputs"
        self.scene(self.run("reconstruct_warmup", "warmup", "reconstruct.sh", "-i", img))
        recon = self.scene(self.run("reconstruct_json", "reconstruct_json", "reconstruct.sh",
                                    "-i", img))
        self.run("reconstruct_ply", "reconstruct_ply", "reconstruct.sh", "-i", img, "-f", "ply",
                 stdout="ply")
        rec = self.run("reconstruct_segment_ply", "contracts", "reconstruct.sh", "-i", img,
                       "-f", "ply", "-p", LABELLED_SEGMENTS, "-o",
                       outputs / "reconstruct_segment.ply", stdout="empty",
                       output=outputs / "reconstruct_segment.ply")
        self.cloud_colours(rec, None if recon is None else {o.id for o in doc_objects(recon)})
        rec = self.run("reconstruct_depth", "reconstruct_depth", "reconstruct.sh", "-i", img,
                       "-f", "depth", stdout="png")
        if rec.ok:
            self.contracts.check("stdout", "reconstruct", f"{rec.tag}/depth image",
                                 _problems_reading(depth_image_problems, rec.stdout_bytes(),
                                                   upright_size(img)))
        folder = outputs / "segment_image"
        rec = self.run("segment_image", "segment_image", "segment.sh", "-i", img, "-d", folder)
        seg = self.scene(rec)
        with self.metrics.expect(SEG_METRICS[0]):
            if seg is None:
                self.metrics.fail(SEG_METRICS[:1], rec.failure())
            else:
                self.artefacts(rec, folder, seg)
                objs = self.images[RESTAURANT] = doc_objects(seg)
                self.details["segmentation.restaurant"] = detection_row(RESTAURANT, objs)
                self.metrics.add(SEG_METRICS[0], len(objs))
        if recon is not None and seg is not None:
            self.contracts.check("same_objects", "image", "reconstruct.sh vs segment.sh -i",
                                 same_objects_problems(doc_objects(recon), doc_objects(seg),
                                                       geometry=False))
        folder = outputs / "segment_image_png"
        self.segmented_image(self.run("segment_image_png", "segment_image_png", "segment.sh",
                                      "-i", img, "-f", "png", "-d", folder, stdout="png"), folder)
        self.view("view_image", ("image", "segment.sh -i", seg), "-i", img)

    def frames(self, seq: CaptureSequence[Any], captures: Sequence[AnyCapture]) -> None:
        rows: list[dict[str, Any]] = []
        for c in captures:
            rec = self.run(f"segment_{seq.key}frame_{c.index:03d}", f"segment_{seq.frames}",
                           "segment.sh", "-i", self.examples / seq.folder / c.name,
                           timeout_s=600.0)
            doc = self.scene(rec)
            if doc is None:
                rows.append({"image": c.name, "error": rec.failure()})
                continue
            objs = self.images[f"{seq.folder}/{c.name}"] = doc_objects(doc)
            rows.append({**detection_row(c.name, objs), "wall_s": round(rec.wall_s, 3)})
        self.details[f"segmentation.{seq.frames}"] = rows
        done = [r for r in rows if "error" not in r]
        failed = [r["error"] for r in rows if "error" in r]
        self.metrics.add(seq.seg_metric, sum(r["objects"] > 0 for r in done) / len(done)
                         if done else None, {"segmented": len(done), "frames": len(rows)},
                         error=f"no frame was segmented ({failed[0] if failed else 'no frames'})")

    def build_maps(self, seq: CaptureSequence[Any], captures: list[Capture] | list[PanCapture]
                   ) -> tuple[Json | None, Json | None]:
        """The one-update map and the split map; their ``-t full`` scenes."""
        name, split_name = seq.maps
        folder = self.examples / seq.folder
        single_dir, split_dir = self.out / "maps" / name, self.out / "maps" / split_name
        single = self.scene(self.run(f"mapper_{name}", f"mapper_{name}", "mapper.sh", "update",
                                     "-i", *(folder / c.name for c in captures),
                                     "-m", single_dir))
        split = None
        published = self.published[split_name] = []
        group = f"mapper_{split_name}"
        parts = np.array_split(np.arange(len(captures)), SPLITS)  # always SPLITS (3) parts
        for k, idx in enumerate(parts, start=1):
            tag = f"{group}_{k}"
            args: tuple[str | Path, ...] = (
                "update", "-i", *(folder / captures[i].name for i in idx), "-m", split_dir)
            if k == 1:
                target = self.out / "outputs" / f"{tag}.json"
                first = self.scene(self.run(tag, group, "mapper.sh", *args, "-o", target,
                                            stdout="empty", output=target))
                if first is not None:
                    published.append(mapupdate.MapView.of(first, split_dir))
                    self.locate_held_out(seq, first, split_dir, [captures[i] for i in parts[1]],
                                         captures)
            elif k < len(parts):
                rec = self.run(tag, group, "mapper.sh", *args, "-t", "single", "-f", "ply", "-p",
                               LABELLED_SEGMENTS, stdout="ply")
                self.cloud_colours(rec, None)
                self.publish_labelled(tag, split_dir, _point_labels(rec), published)
            else:
                split = self.scene(self.run(tag, group, "mapper.sh", *args, "-t", "full"))
        return single, split

    def publish_labelled(self, tag: str, split_dir: Path, ids: set[int],
                         published: list[mapupdate.MapView]) -> None:
        """A ``-t single -f ply`` update published the objects whose ids its points carry: those
        objects of the map as the update left it, with its poses (``persisted_scene``)."""
        if not ids:
            return
        try:
            view = mapupdate.MapView.of(persisted_scene(split_dir), split_dir)
        except (OSError, ValueError, KeyError, OhMySlamError) as exc:
            self.details.setdefault("map.stability.unread", {})[tag] = f"{type(exc).__name__}: {exc}"
            return
        view.objects = [o for o in view.objects if o.id in ids]
        published.append(view)

    def locate_held_out(self, seq: CaptureSequence[Any], first: Json, split_dir: Path,
                        held_out: Sequence[AnyCapture], captures: Sequence[AnyCapture]) -> None:
        """``mapper.sh locate`` of captures the split map has not seen yet (read-only)."""
        before = tree_digest(split_dir)
        tag = f"{seq.located}_held_out"
        rec = self.run(tag, tag, "mapper.sh", "locate", "-i",
                       *(self.examples / seq.folder / c.name for c in held_out), "-m", split_dir)
        doc = self.scene(rec)
        self.contracts.check("readonly", "map", seq.named("mapper.sh locate (held out)"),
                             [] if tree_digest(split_dir) == before else
                             ["mapper.sh locate changed the map folder"])
        reference = captures[0].name if captures else ""
        self.held_out[seq.folder] = {
            "located": None if doc is None else located_poses(doc),
            "reference": capture_poses(first, split_dir).get(reference), "captures": held_out,
            "reference_name": reference}
        if doc is not None:
            self.contracts.check("openlabel", "mapper", f"{tag}/located frames",
                                 located_problems(doc, len(held_out)))

    def held_out_metrics(self, seq: CaptureSequence[Any], split: Json | None) -> None:
        later = None if split is None else capture_poses(split, self.out / "maps" / seq.maps[1])
        h = self.held_out.get(seq.folder, {})
        self.details[f"poses.{seq.located}"] = held_out_metrics(
            self.metrics, f"pose.{seq.located}", h.get("located"), h.get("reference"),
            h.get("captures") or [], later, h.get("reference_name"))

    def map_metrics[C: (Capture, PanCapture)](self, seq: CaptureSequence[C], captures: list[C],
                                              single: Json | None, split: Json | None) -> None:
        name, split_name = seq.maps
        dirs = {mp: self.out / "maps" / mp for mp in seq.maps}
        poses: dict[str, dict[str, Pose]] = {}
        for mp, doc in ((name, single), (split_name, split)):
            ids = [f"pose.{mp}.{k}" for k in seq.pose_ids]
            with self.metrics.expect(*ids):
                if doc is None:
                    self.metrics.fail(ids, f"the {mp} map was not built")
                else:
                    poses[mp] = capture_poses(doc, dirs[mp])
                    self.details[f"poses.{mp}"] = seq.judge_poses(
                        self.metrics, f"pose.{mp}", poses[mp], captures)
            ids = [f"map.{mp}.{k}" for k in seq.agreement_ids]
            with self.metrics.expect(*ids):
                if doc is None:
                    self.metrics.fail(ids, f"the {mp} map was not built")
                else:
                    self.details[f"map.{mp}.pairs"] = agreement_metrics(
                        self.metrics, f"map.{mp}", dirs[mp], seq.same_heading(captures),
                        seq.tilts)
        self.single_poses[seq.folder] = poses.get(name)
        ids = [f"{seq.stability}.{k}" for k in STABILITY_METRICS]
        with self.metrics.expect(*ids):
            if single is None or split is None:
                self.metrics.fail(ids, "both maps are needed")
            else:
                published = [Published(v.objects, update_alignment(poses[split_name], v.poses))
                             for v in self.published.get(split_name, [])]
                self.details[seq.stability] = stability_metrics(
                    self.metrics, seq.stability, doc_objects(single), doc_objects(split),
                    split_alignment(poses[name], poses[split_name]), published)
        with self.metrics.expect(*(f"pose.{seq.located}.{k}" for k in LOCATE_METRICS)):
            self.held_out_metrics(seq, split)

    def map_commands(self, seq: CaptureSequence[Any], single: Json | None) -> None:
        single_dir = self.out / "maps" / seq.maps[0]
        before = tree_digest(single_dir) if single is not None else None
        self.view(seq.view, ("map", seq.named("mapper.sh -t full"), single), "-m", single_dir)
        self.locate_reference(seq, single, single_dir)
        if before is not None:
            same = tree_digest(single_dir) == before
            self.contracts.check("readonly", "map", seq.named("view.sh -m, mapper.sh locate"),
                                 [] if same else ["the map folder changed"])

    def locate_reference(self, seq: CaptureSequence[Any], single: Json | None,
                         single_dir: Path) -> None:
        """``mapper.sh locate`` of three of the sequence's captures (``locate_picks``) on its
        one-update map: ``-t single`` JSON (the default), ``-t full`` JSON (the map exactly as
        ``update -t full`` gave it, plus the located cameras) and ``-f ply -o`` (map points, one
        header line per input image)."""
        if single is None:
            return
        folder = self.examples / seq.folder
        images = [folder / c.name for c in locate_picks(seq.read(folder))]
        args: tuple[str | Path, ...] = ("locate", "-i", *images, "-m", single_dir)
        name = seq.located
        doc = self.scene(self.run(f"{name}_single", name, "mapper.sh", *args))
        if doc is not None:
            self.contracts.check("openlabel", "mapper", f"{name}_single/located frames",
                                 located_problems(doc, len(images)))
        doc = self.scene(self.run(f"{name}_full", f"{name}_full", "mapper.sh", *args, "-t",
                                  "full"))
        if doc is not None:
            self.contracts.check("openlabel", "mapper", f"{name}_full/located frames",
                                 located_problems(doc, len(images)))
            self.contracts.check("same_objects", "map",
                                 seq.named("mapper.sh locate -t full vs update -t full"),
                                 same_objects_problems(doc_objects(single), doc_objects(doc),
                                                       geometry=True))
        target = self.out / "outputs" / f"{name}.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        rec = self.run(f"{name}_ply", f"{name}_ply", "mapper.sh", *args, "-f", "ply", "-o",
                       target, stdout="empty", output=target)
        if rec.ok:
            problems = _problems_reading(lambda: _pose_lines(target.read_bytes(), len(images)))
            self.contracts.check("stdout", "mapper", f"{name}_ply/pose header lines", problems)

    def map_update(self) -> None:
        """The office sequence mapped whole in one update, and split across updates as its
        annotation says; the ``map_update`` files of the ground truth say what changed (no file:
        the metrics fail, nothing is mapped)."""
        files, skipped = gt.discover(self.examples / GROUND_TRUTH)
        plan = gt.map_update_plan(files, skipped)
        self.single_poses[OFFICE] = None  # its poses' ground truth: not built until it is
        if plan is None:
            self.metrics.fail(mapupdate.metric_ids(), "no 'map_update' file in "
                              "examples/ground_truth/ (see its README.md): what changed in the "
                              "sequence is not annotated")
            return
        folder = self.examples / plan.sequence
        images = mapupdate.sequence_images(folder)
        early = plan.before_images(images)
        sizes = plan.split_sizes(images)
        ids = mapupdate.metric_ids([mapupdate.split_name(s) for s in sizes])
        self.details["map_update"] = {"annotation": [str(p) for p in plan.files],
                                      "images": images, "before_images": early,
                                      "split_sizes": [list(s) for s in sizes]}
        if not early:
            self.metrics.fail(ids, f"no annotated image is in {folder}")
            return
        maps = self.out / "maps"
        single_dir = maps / OFFICE_MAP
        single = self.scene(self.run("mapper_office", "mapper_office", "mapper.sh", "update",
                                     "-i", *(folder / n for n in images), "-m", single_dir))
        if single is not None:
            self.single_poses[plan.sequence] = capture_poses(single, single_dir)
        splits: list[mapupdate.SplitMaps] = []
        failed: dict[str, str] = {}
        for s in sizes:
            name = mapupdate.split_name(s)
            try:
                parts = mapupdate.parts_of(images, s)
            except ValueError as exc:
                failed[name] = str(exc)
                continue
            views, d = [], maps / f"office_{name}"
            for k, part in enumerate(parts, start=1):
                doc = self.scene(self.run(f"mapper_office_{name}_{k}", "mapper_office_split",
                                          "mapper.sh", "update", "-i", *(folder / n for n in part),
                                          "-m", d))
                if doc is None:
                    failed[name] = f"update {k} of {name} failed"
                    break
                # each update's view is read now: the map's frame records change with the next
                views.append(mapupdate.MapView.of(doc, d, cloud=k == len(parts)))
            else:
                splits.append(mapupdate.SplitMaps(s, parts, views))
        for name, why in failed.items():
            self.metrics.fail(mapupdate.split_metric_ids(name), why)
        with self.metrics.expect(*ids):
            if single is None:
                self.metrics.fail(ids, "the map of the whole sequence was not built")
            else:
                self.details["map_update"].update(mapupdate.map_update_metrics(
                    self.metrics, plan, images, mapupdate.MapView.of(single, single_dir, True),
                    splits, failed))

    def street2_video(self) -> None:
        """``mapper.sh update`` of the street2 video (the user's benchmark rule): performance,
        contracts and the share of sampled frames the map registered. The video (hard-linked)
        and its map are in ``server.sh``'s workspace and the run goes through the recording
        inference proxy, so that it is also the command's result its ``server.sh`` request is
        compared with (``service.Ran``): the long video is mapped once by the commands."""
        video = self.street2
        if video is None or not Path(video).is_file():
            self.metrics.fail([STREET2_METRIC], f"street2.mp4 not found at {video} (pass "
                              "--street2 PATH)")
            self.details["street2"] = {"video": str(video), "error": "not found"}
            return
        ws = service.workspace_root(self.out)
        placed = service.place(Path(video), service.input_folder(ws, Path(video).name))
        update = spec.MAPPER.command("update")
        argv = spec.argv_of(update, update.modes[0], {"inputs": [str(placed)],
                                                      "map": str(ws / "maps" / STREET2_MAP)})
        rec = self.run("mapper_street2", "mapper_street2", "mapper.sh", *argv, timeout_s=5400.0,
                       env=self.proxy_env)
        self.scene(rec)
        if self.proxy_env:
            self.street2_placed, self.street2_ran = placed, service.Ran(tuple(argv), rec)
        counts = (rec.timings or {}).get("counts") or {}
        sampled, registered = counts.get("keyframes_sampled"), counts.get("keyframes_registered")
        self.details["street2"] = {"video": str(video), "argv": argv, "counts": counts,
                                   "through_proxy": bool(self.proxy_env)}
        self.metrics.add(STREET2_METRIC, registered / sampled if rec.ok and sampled else None,
                         {"sampled": sampled, "registered": registered},
                         error=rec.failure() if not rec.ok else "the run recorded no counts")

    def parity_inputs(self) -> list[dict[str, Any]]:
        """``server.sh``'s reference inputs (``service.Inputs``), every one of spec §5:
        restaurant.jpg, each example sequence with its one-update map (``sequence_input``), and
        street2.mp4 as its section mapped it (when through the proxy: ``street2_ran``)."""
        out: list[dict[str, Any]] = [{"name": RESTAURANT, "image": self.examples / RESTAURANT}]
        for seq in SEQUENCES:
            folder = self.examples / seq.folder
            try:
                files = [folder / c.name for c in seq.read(folder)]
            except (OSError, ValueError):
                files = []
            out.append(sequence_input(seq.folder, files, self.out / "maps" / seq.maps[0]))
        office = self.examples / OFFICE
        files = [office / n for n in mapupdate.sequence_images(office)] if office.is_dir() else []
        out.append(sequence_input(OFFICE, files, self.out / "maps" / OFFICE_MAP))
        if self.street2_placed is not None and self.street2_ran is not None:
            out.append({"name": self.street2_placed.name, "sequence": [self.street2_placed],
                        "ran": self.street2_ran, "writes": STREET2_MAP})
        return out

    def server_sh(self) -> None:
        """``server.sh`` (http_server.md "Evaluation"), through the recording inference proxy."""
        if not self.proxy_env:
            raise RuntimeError("the recording inference proxy is not running")
        kw = {} if self.ui is None else {"ui": self.ui}
        service.ServiceEvaluation(self, self.parity_inputs(), self.proxy_env, **kw).run()

    @contextmanager
    def recording_proxy(self) -> Iterator[None]:
        """The record-or-replay inference proxy (``proxy``) that street2.mp4's run and
        ``server.sh`` share (``proxy_env``; empty if it did not start: an error of its own
        section); its records are deleted when it stops."""
        import shutil

        from oh_my_slam.tools.evaluate.proxy import InferenceProxy, short_runtime

        runtime = short_runtime()
        proxy = self.section("inference proxy", lambda: InferenceProxy(
            runtime, paths.socket_path()).start())
        self.proxy_env = {} if proxy is None else proxy.env()
        try:
            yield
        finally:
            self.proxy_env = {}
            if proxy is not None:
                proxy.stop()
                self.details.setdefault("server_sh", {})["proxy"] = proxy.stats()
            shutil.rmtree(runtime, ignore_errors=True)

    def annotated_images(self) -> None:
        """``segment.sh -i`` on every example image an ``objects`` annotation names that no
        section above segmented (an ``office_sequence`` image, say): ground truth added later for
        any example file is judged without code changes (``ground_truth``)."""
        files, _ = gt.discover(self.examples / GROUND_TRUTH)
        root = self.examples.resolve()
        todo = [n for n in gt.annotated_images(files) if n not in self.images]
        for k, name in enumerate(todo, start=1):
            path = (self.examples / name).resolve()
            if root not in path.parents or not path.is_file():
                continue  # not an example image: object_metrics says so
            doc = self.scene(self.run(f"segment_annotated_{k:02d}", "segment_annotated",
                                      "segment.sh", "-i", path, timeout_s=600.0))
            if doc is not None:
                self.images[name] = doc_objects(doc)

    def ground_truth(self) -> None:
        files, skipped = gt.discover(self.examples / GROUND_TRUTH)
        gt.object_metrics(self.metrics, [f for f in files if f.kind == "objects"], self.images,
                          skipped)
        gt.pose_metrics(self.metrics, [f for f in files if f.kind == "poses"], self.single_poses)
        self.details["ground_truth"] = {"files": [str(f.path) for f in files],
                                        "skipped": skipped}

    # -- all -------------------------------------------------------------------------------------------

    def section(self, name: str, fn: Any, *args: Any) -> Any:
        """``fn(*args)``; an unexpected error is logged and ends only this section (its metrics
        then fail as not computed)."""
        print(f"== {name}", file=sys.stderr, flush=True)
        try:
            return fn(*args)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.details.setdefault("errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    def capture_sequence(self, seq: CaptureSequence[Any]) -> None:
        """One capture sequence: ``segment.sh -i`` per frame, its two maps, their pose accuracy
        and quality, and the commands on its one-update map. A file name outside the sequence's
        grammar ends only this sequence (its metrics then fail as not computed)."""
        folder = seq.folder
        captures = self.section(f"{folder}: capture names", seq.read, self.examples / folder)
        if captures is None:
            return
        self.section(f"{folder}: segment.sh -i per frame", self.frames, seq, captures)
        maps = self.section(f"{folder}: mapper.sh update", self.build_maps, seq, captures)
        single, split = maps or (None, None)
        self.section(f"{folder}: pose accuracy, map quality", self.map_metrics, seq, captures,
                     single, split)
        self.section(f"{folder}: view.sh -m, mapper.sh locate",
                     self.map_commands, seq, single)

    def run_all(self) -> None:
        self.section("inference server", self.server_start)
        try:
            self.section("restaurant.jpg", self.restaurant)
            for seq in SEQUENCES:
                self.capture_sequence(seq)
            self.section("office_sequence: mapper.sh update (map update)", self.map_update)
            self.section("ground truth: segment.sh -i on the other annotated images",
                         self.annotated_images)
            with self.recording_proxy():
                self.section("street2.mp4: mapper.sh update", self.street2_video)
                self.section("server.sh", self.server_sh)
        finally:
            self.section("restore the inference server", self.server_restore)
        self.section("summary", self.summarise)
        self.metrics.fail(expected_ids(self.examples),
                          "not computed (an earlier step failed; see errors)")

    def summarise(self) -> None:
        m = self.metrics
        with m.expect(*perf_ids()):
            perf_metrics(m, self.runner.records)
        with m.expect(SEG_METRICS[1]):
            scores = [o.score for objs in self.images.values() for o in objs
                      if o.score is not None]
            m.add(SEG_METRICS[1], min(scores) if scores else None, error="no detections")
        for mid, value, detail, error in self.contracts.results():
            m.add(mid, value, detail, error)
        self.ground_truth()
