"""Renders, viewer screenshots and the report document for ``tools.perf_report`` (R44).

    renders(out)  PLY outputs drawn with a small NumPy point splatter (z-buffered), segmented.png
                  copies                                              → screenshots/formats/
    shots(out)    ``view.sh -i restaurant.jpg`` and ``view.sh -m`` per map in headless Edge: initial
                  view, every layer alone, catalogue selection, hover tooltip, image/display tabs
                                                                      → screenshots/<view>/
    write(out)    report.md + report.html (charts inlined as SVG in the HTML)

Nothing here recomputes geometry, colours or objects: it only draws the files the commands wrote.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
VALIDATION = Path.home() / "oh-my-slam-data" / "validation" / "20260923T001638Z"
MAP_NAMES = ("ainex", "church", "livingroom")
TITLES = {"ainex": "ainex-captures (79 images)", "church": "church.mp4 (-fps 2)",
          "livingroom": "livingroom.mp4 (-fps 2)", "restaurant": "restaurant.jpg"}
BG = (21, 23, 28)  # viewer background #15171c


# ------------------------------------------------------------------------------------------------
# point renderer


def render_points(xyz: np.ndarray, rgb: np.ndarray, eye: np.ndarray, target: np.ndarray,
                  up: np.ndarray, size: tuple[int, int] = (1280, 800), fov_deg: float = 55.0,
                  px: int = 2) -> np.ndarray:
    """Perspective splat of coloured points (nearest point wins per pixel)."""
    w, h = size
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    pc = (xyz - eye) @ np.stack([right, down, fwd]).T
    keep = pc[:, 2] > 0.05
    pc, col = pc[keep], rgb[keep]
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    u = np.floor(f * pc[:, 0] / pc[:, 2] + w / 2).astype(np.int64)
    v = np.floor(f * pc[:, 1] / pc[:, 2] + h / 2).astype(np.int64)
    offs = [(dx, dy) for dx in range(px) for dy in range(px)]
    uu = np.concatenate([u + dx for dx, _ in offs])
    vv = np.concatenate([v + dy for _, dy in offs])
    zz = np.tile(pc[:, 2], len(offs))
    cc = np.tile(col, (len(offs), 1))
    ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    lin = vv[ok] * w + uu[ok]
    order = np.lexsort((zz[ok], lin))
    lin_s = lin[order]
    first = np.ones(len(lin_s), bool)
    first[1:] = lin_s[1:] != lin_s[:-1]
    img = np.empty((h * w, 3), np.uint8)
    img[:] = BG
    img[lin_s[first]] = cc[ok][order][first]
    return img.reshape(h, w, 3)


def _crop(xyz: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0, pad: float = 0.1) -> np.ndarray:
    lo, hi = np.percentile(xyz, lo_q, axis=0), np.percentile(xyz, hi_q, axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def map_views(xyz: np.ndarray, floor_z: float | None) -> dict[str, dict[str, Any]]:
    """Oblique and top views of a z-up map cloud (top view: ceiling cropped 2.2 m above floor)."""
    m = _crop(xyz)
    c = np.median(xyz[m], axis=0)
    span = np.percentile(xyz[m], 97, axis=0) - np.percentile(xyz[m], 3, axis=0)
    e = float(max(span[0], span[1], 1.0))
    views = {
        "oblique": {"eye": c + np.array([-0.55 * e, -0.75 * e, 0.6 * e]), "target": c,
                    "up": np.array([0.0, 0.0, 1.0]), "mask": m, "caption": "oblique view"},
    }
    top = m.copy()
    note = "top view"
    if floor_z is not None:
        top &= xyz[:, 2] < floor_z + 2.2
        note = "top view (points more than 2.2 m above the floor hidden)"
    views["top"] = {"eye": c + np.array([0.0, 0.0, 1.25 * e]), "target": c,
                    "up": np.array([1.0, 0.0, 0.0]), "mask": top, "caption": note}
    return views


def camera_views(xyz: np.ndarray) -> dict[str, dict[str, Any]]:
    """Camera-frame cloud (OpenCV axes): the photo's own viewpoint and an oblique one."""
    m = _crop(xyz)
    c = np.median(xyz[m], axis=0)
    d = float(np.linalg.norm(c))
    up = np.array([0.0, -1.0, 0.0])
    return {
        "camera": {"eye": np.zeros(3), "target": np.array([0.0, 0.0, 1.0]), "up": up, "mask": m,
                   "caption": "from the photo's viewpoint"},
        "oblique": {"eye": np.array([-0.55 * d, -0.45 * d, 0.1 * d]), "target": c, "up": up,
                    "mask": m, "caption": "oblique view (camera moved left and up)"},
    }


def _save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=True)


def _copy_scaled(src: Path, dst: Path, max_w: int = 1600) -> tuple[int, int]:
    from PIL import Image

    im = Image.open(src).convert("RGB")
    size = im.size
    if im.width > max_w:
        # nearest keeps mask colours exact (no blended edge pixels)
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.Resampling.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    return size


def renders(out: Path) -> None:
    from oh_my_slam.core.ply import read_ply

    fmt = out / "screenshots" / "formats"
    manifest: dict[str, list[dict[str, Any]]] = {}

    def add(name: str, kind: str, src: Path, path: Path, caption: str, **extra: Any) -> None:
        manifest.setdefault(name, []).append({"kind": kind, "src": str(src.relative_to(out)),
                                              "png": str(path.relative_to(out)),
                                              "caption": caption, **extra})

    def draw(name: str, kind: str, ply: Path, views: Any, label: str) -> None:
        cloud = read_ply(ply)
        for vname, v in (views(cloud.xyz)).items():
            t0 = time.perf_counter()
            img = render_points(cloud.xyz[v["mask"]], cloud.rgb[v["mask"]], v["eye"],
                                v["target"], v["up"])
            path = fmt / f"{name}_{kind}_{vname}.png"
            _save_png(img, path)
            add(name, kind, ply, path, f"{label} — {v['caption']}", points=len(cloud),
                render_s=round(time.perf_counter() - t0, 2))

    rest = out / "outputs" / "restaurant"
    if (rest / "reconstruct.ply").exists():
        draw("restaurant", "ply", rest / "reconstruct.ply", camera_views,
             "`reconstruct.sh -f ply` (camera frame)")
    if (rest / "segment_o" / "segments.ply").exists():
        draw("restaurant", "segments", rest / "segment_o" / "segments.ply", camera_views,
             "`segment.sh -o` segments.ply")
        size = _copy_scaled(rest / "segment_o" / "segmented.png", fmt / "restaurant_segmented.png")
        add("restaurant", "segmented", rest / "segment_o" / "segmented.png",
            fmt / "restaurant_segmented.png", "`segment.sh -o` segmented.png", size=size)
    for name in MAP_NAMES:
        o = out / "outputs" / name
        meta_p = out / "maps" / name / "map.json"
        floor = json.loads(meta_p.read_text()).get("floor_z") if meta_p.exists() else None

        def views(xyz: np.ndarray, floor: float | None = floor) -> Any:
            return map_views(xyz, floor)

        if (o / "update_full.ply").exists():
            draw(name, "ply", o / "update_full.ply", views,
                 "`mapper.sh update -f ply -t full` (after a one-image update)")
        if (o / "segment_m" / "segments.ply").exists():
            draw(name, "segments", o / "segment_m" / "segments.ply", views,
                 "`segment.sh -m -o` segments.ply")
            size = _copy_scaled(o / "segment_m" / "segmented.png", fmt / f"{name}_segmented.png")
            add(name, "segmented", o / "segment_m" / "segmented.png",
                fmt / f"{name}_segmented.png", "`segment.sh -m -o` segmented.png (contact sheet)",
                size=size)
    (out / "renders.json").write_text(json.dumps(manifest, indent=1))


# ------------------------------------------------------------------------------------------------
# viewer screenshots

LAYERS = ("points", "mesh", "cameras", "segments", "labels", "obbs")
LAYER_TITLES = {"points": "Point cloud (RGB)", "mesh": "Textured mesh (mesh.glb)",
                "cameras": "Camera poses", "segments": "Segmentation (object colours)",
                "labels": "Labels", "obbs": "Oriented boxes"}

_HOVER_JS = """() => {
  const v = window.__viewer; const o = v.objects.find(x => x.id === v.selected);
  if (!o) return null;
  const g = window.__viewerGroups.pick.parent; const p = o.center.clone();
  p.applyMatrix4(g.matrixWorld);
  const cam = window.__viewerCamera; p.project(cam);
  const r = document.querySelector('#canvas-host').getBoundingClientRect();
  return [r.left + (p.x + 1) / 2 * r.width, r.top + (1 - p.y) / 2 * r.height];
}"""


def _set_layers(pg: Any, on: set[str]) -> dict[str, bool]:
    pg.click('#tabs button[data-tab="layers"]')
    state = {}
    for k in LAYERS:
        loc = pg.locator(f'[data-layer="{k}"] input')
        if not loc.count() or not loc.is_enabled():
            state[k] = False
            continue
        if loc.is_checked() != (k in on):
            loc.click()
        state[k] = True
    pg.wait_for_timeout(600)
    return state


def _browse(out: Path, view: str, args: list[str]) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    shots = out / "screenshots" / view
    shots.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"view": view, "cmd": "view.sh " + " ".join(args), "shots": [],
                           "errors": []}
    proc = subprocess.Popen([str(REPO / "view.sh"), *args, "--no-browser"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t_start = time.perf_counter()
    try:
        assert proc.stderr is not None
        url = None
        deadline = time.time() + 300
        while time.time() < deadline:
            line = proc.stderr.readline().decode()
            m = re.search(r"(http://127\.0\.0\.1:\d+/)", line)
            if m:
                url = m.group(1)
                break
            if not line and proc.poll() is not None:
                break
        if url is None:
            rec["errors"].append("viewer printed no URL")
            return rec
        rec["server_start_s"] = round(time.perf_counter() - t_start, 2)

        def shot(pg: Any, name: str, caption: str, full: bool = True) -> None:
            path = shots / f"{name}.png"
            if full:
                pg.screenshot(path=str(path))
            else:
                pg.locator("#viewport").screenshot(path=str(path))
            rec["shots"].append({"name": name, "png": str(path.relative_to(out)),
                                 "caption": caption})

        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            ctx = b.new_context(viewport={"width": 1440, "height": 900})
            pg = ctx.new_page()
            pg.on("console", lambda msg: rec["errors"].append(msg.text)
                  if msg.type == "error" else None)
            pg.on("pageerror", lambda e: rec["errors"].append(str(e)))
            t0 = time.perf_counter()
            pg.goto(url)
            pg.wait_for_timeout(250)
            if not pg.evaluate("() => !!(window.__viewer && window.__viewer.ready)"):
                shot(pg, "loading", "Loading overlay with per-asset progress")
            pg.wait_for_function("window.__viewer && window.__viewer.ready === true",
                                 timeout=300000)
            rec["page_ready_s"] = round(time.perf_counter() - t0, 2)
            pg.wait_for_timeout(1200)
            rec["stats_line"] = pg.inner_text("#stats")
            shot(pg, "initial", "Initial view: auto-framed, default layers (maps open with the "
                 "textured mesh instead of the point cloud)")
            defaults = {k for k in LAYERS if pg.locator(f'[data-layer="{k}"] input').count()
                        and pg.locator(f'[data-layer="{k}"] input').is_checked()}
            rec["default_layers"] = sorted(defaults)
            for layer in LAYERS:
                enabled = _set_layers(pg, {layer})
                if not enabled[layer]:
                    rec.setdefault("unavailable_layers", []).append(layer)
                    continue
                shot(pg, f"layer_{layer}", f"{LAYER_TITLES[layer]} only")
            _set_layers(pg, defaults)
            pg.click('#tabs button[data-tab="catalogue"]')
            rows = pg.locator("#catalogue tbody tr")
            if rows.count():
                rows.nth(0).click()
                pg.wait_for_timeout(1200)
                sel = pg.evaluate("() => window.__viewer.selected")
                rec["selected"] = sel
                shot(pg, "selected", "Catalogue row clicked: the largest object is selected, "
                     "highlighted and framed")
                xy = pg.evaluate(_HOVER_JS)
                if xy:
                    pg.mouse.move(xy[0], xy[1])
                    pg.wait_for_timeout(500)
                    tip = pg.locator("#tooltip")
                    rec["tooltip"] = tip.inner_text().replace("\n", " ") if tip.is_visible() \
                        else None
                    shot(pg, "tooltip", "Hover tooltip for the box under the cursor, placed at "
                         "the selected object's centre (label, id, score, W×D×H, volume)")
            pg.keyboard.press("Escape")
            pg.keyboard.press("r")
            pg.wait_for_timeout(600)
            if pg.locator("#tab-image-btn").is_visible():
                pg.click("#tab-image-btn")
                pg.wait_for_timeout(500)
                shot(pg, "image_tab", "Image tab: segmented.png next to the 3D view")
            pg.click('#tabs button[data-tab="display"]')
            pg.wait_for_timeout(300)
            shot(pg, "display_tab", "Display tab: point size, label density, background, reset")
            ctx.close()
            b.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rec


def _reuse_validation(out: Path) -> dict[str, list[dict[str, str]]]:
    """GIFs and 1920×1080 screenshots from the final validation run (same viewer code)."""
    reused: dict[str, list[dict[str, str]]] = {}
    for view in ("restaurant", *MAP_NAMES):
        src = VALIDATION / "ui" / f"view_{view}"
        if not src.is_dir():
            continue
        dst = out / "screenshots" / "validation_run" / view
        dst.mkdir(parents=True, exist_ok=True)
        for fname, caption in (("orbit.gif", "Orbit (16 frames)"),
                               ("1920x1080_initial.png", "1920×1080 layout"),
                               ("1280x800_initial.png", "1280×800 layout")):
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
                reused.setdefault(view, []).append(
                    {"png": str((dst / fname).relative_to(out)), "caption": caption})
    return reused


def shots(out: Path) -> None:
    recs = [_browse(out, "restaurant", ["-i", "/Users/U124317/robot_view/restaurant.jpg"])]
    for name in MAP_NAMES:
        mdir = out / "maps" / name
        if (mdir / "map.json").exists():
            before = _tree_hash(mdir)
            rec = _browse(out, name, ["-m", str(mdir)])
            rec["map_unchanged"] = _tree_hash(mdir) == before
            recs.append(rec)
    data = {"views": recs, "reused": _reuse_validation(out)}
    (out / "shots.json").write_text(json.dumps(data, indent=1))


def _tree_hash(root: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# report


STAGES = {
    "setup": "Lock + read map",
    "ingest": "Ingest / keyframe sampling",
    "inference": "Inference (geometry, gravity, segmentation)",
    "features_matching": "Features + matching (COLMAP)",
    "sfm": "Poses (SfM / multi-view)",
    "focal_rerun": "Geometry re-run with SfM focal",
    "map_frame": "Map frame, metric scale, floor levelling",
    "depth_alignment": "Per-keyframe depth alignment",
    "persist_frames": "Persist keyframes + SfM model",
    "validity": "Latest wins (validity masks)",
    "objects": "Objects (lift, associate, OBBs)",
    "cloud": "Map cloud",
    "fusion": "TSDF fusion",
    "mesh": "Mesh extraction + clean-up",
    "texture": "Mesh texture (OpenMVS)",
    "export": "Scene export (JSON / PLY)",
    "commit": "Commit (atomic swap)",
    "connect": "Server check",
    "segment": "Lift masks + fit OBBs",
    "artifacts": "Write 5 artefacts",
    "write": "Write stdout",
    "_startup": "Start-up, imports, other",
}
GROUPS = [
    ("Inference", ("inference", "focal_rerun")),
    ("Features + matching", ("features_matching",)),
    ("Poses", ("sfm",)),
    ("Frame, depth, latest wins", ("map_frame", "depth_alignment", "persist_frames", "validity")),
    ("Objects", ("objects",)),
    ("Cloud, fusion, mesh", ("cloud", "fusion", "mesh")),
    ("Texture", ("texture",)),
    ("Other", ("setup", "ingest", "export", "commit", "write", "_startup")),
]
PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                 "#9a9892"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
                "#77766f"]


def _runs(data: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return [r for r in data["runs"] if r["group"] == group and r["exit"] == 0 and r["timings"]]


def _stages(r: dict[str, Any]) -> dict[str, float]:
    """Exclusive stage times plus the rest of the wall-clock (interpreter start-up, imports, the
    few calls outside any stage)."""
    st = dict(r["timings"]["stages_s"])
    st["_startup"] = max(0.0, r["wall_s"] - sum(st.values()))
    return st


def _fmt_s(v: float | None) -> str:
    if v is None:
        return "–"
    if v >= 120:
        return f"{v / 60:.1f} min"
    if v >= 10:
        return f"{v:.0f} s"
    return f"{v:.2f} s" if v < 1 else f"{v:.1f} s"


def _mb(n: float) -> str:
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def _table(head: list[str], rows: list[list[str]], align: str | None = None) -> str:
    al = align or ("l" + "r" * (len(head) - 1))
    sep = ["---:" if a == "r" else "---" for a in al]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _group_totals(st: dict[str, float]) -> list[float]:
    return [sum(st.get(k, 0.0) for k in keys) for _, keys in GROUPS]


def _chart_svg(bars: list[tuple[str, list[float]]], unit: str, title: str,
               totals: list[float] | None = None) -> str:
    """Horizontal stacked bars (one per map), 2 px surface gaps, rounded data end, legend,
    total at the tip, <title> tooltips. Light/dark via prefers-color-scheme."""
    scale_div = 60.0 if unit == "min" else 1.0
    width, left, right, bar_h, gap = 860, 190, 90, 22, 18
    legend_h = 56
    top = legend_h + 10
    height = top + len(bars) * (bar_h + gap) + 34
    vmax = max(sum(v) for _, v in bars) / scale_div
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300) if vmax / s <= 6)
    xmax = step * np.ceil(vmax / step)
    px = (width - left - right) / xmax
    css_l = "".join(f".s{i}{{fill:{c}}}" for i, c in enumerate(PALETTE_LIGHT))
    css_d = "".join(f".s{i}{{fill:{c}}}" for i, c in enumerate(PALETTE_DARK))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="100%" role="img" aria-label="{html.escape(title)}" '
           'style="max-width:860px;font-family:-apple-system,system-ui,sans-serif">',
           "<style>.surf{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e6e5e0}"
           f".gap{{stroke:#fcfcfb}}{css_l}"
           "@media (prefers-color-scheme: dark){.surf{fill:#1a1a19}.t1{fill:#fff}"
           f".t2{{fill:#c3c2b7}}.grid{{stroke:#383835}}.gap{{stroke:#1a1a19}}{css_d}}}</style>",
           f'<rect class="surf" width="{width}" height="{height}" rx="8"/>']
    # legend (two rows)
    lx, ly = left, 18
    for i, (name, _) in enumerate(GROUPS):
        if i == 4:
            lx, ly = left, 40
        out.append(f'<rect class="s{i}" x="{lx}" y="{ly - 9}" width="10" height="10" rx="2"/>')
        out.append(f'<text class="t2" x="{lx + 14}" y="{ly}" font-size="12">'
                   f"{html.escape(name)}</text>")
        lx += 14 + 6.8 * len(name) + 22
    # grid + axis
    t = 0.0
    while t <= xmax + 1e-9:
        x = left + t * px
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" '
                   f'y2="{height - 26}" stroke-width="1"/>')
        out.append(f'<text class="t2" x="{x:.1f}" y="{height - 10}" font-size="11" '
                   f'text-anchor="middle">{t:g} {unit}</text>')
        t += step
    for bi, (label, vals) in enumerate(bars):
        y = top + bi * (bar_h + gap)
        out.append(f'<text class="t1" x="{left - 10}" y="{y + bar_h / 2 + 4}" font-size="13" '
                   f'text-anchor="end">{html.escape(label)}</text>')
        x = left
        nz = [i for i, v in enumerate(vals) if v > 0]
        for i, v in enumerate(vals):
            if v <= 0:
                continue
            w = v / scale_div * px
            last = i == nz[-1]
            tip = f"{label} — {GROUPS[i][0]}: {_fmt_s(v)} ({100 * v / sum(vals):.0f} %)"
            if last and w > 4:
                d = (f"M{x:.2f},{y} h{w - 4:.2f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 "
                     f"h{-(w - 4):.2f} z")
                out.append(f'<path class="s{i}" d="{d}"><title>{html.escape(tip)}</title></path>')
            else:
                out.append(f'<rect class="s{i}" x="{x:.2f}" y="{y}" width="{w:.2f}" '
                           f'height="{bar_h}"><title>{html.escape(tip)}</title></rect>')
            x += w
            if not last:
                out.append(f'<line class="gap" x1="{x:.2f}" x2="{x:.2f}" y1="{y}" '
                           f'y2="{y + bar_h}" stroke-width="2"/>')
        total = totals[bi] if totals else sum(vals)
        out.append(f'<text class="t1" x="{x + 8:.1f}" y="{y + bar_h / 2 + 4}" font-size="12">'
                   f"{_fmt_s(total)}</text>")
    out.append("</svg>")
    return "\n".join(out)


def _objects_log(out: Path, tag: str) -> dict[str, int] | None:
    """The mapper's "objects: N confirmed of M; K removed, J merged" stderr line of a run."""
    p = out / "logs" / f"{tag}.stderr.txt"
    if not p.exists():
        return None
    found = re.findall(r"objects: (\d+) confirmed of (\d+); (\d+) removed, (\d+) merged",
                       p.read_text(errors="replace"))
    if not found:
        return None
    return dict(zip(("confirmed", "stored", "removed", "merged"), map(int, found[-1]),
                    strict=True))


def _json_excerpt(path: Path, n_objects: int = 2, n_frames: int = 1) -> str:
    doc = json.loads(path.read_text())["openlabel"]
    ex: dict[str, Any] = {"metadata": doc.get("metadata")}
    for key in ("coordinate_systems", "streams"):
        if key in doc:
            items = list(doc[key].items())
            ex[key] = dict(items[:2])
            if len(items) > 2:
                ex[key][f"… {len(items) - 2} more"] = "…"
    if "frames" in doc:
        fr = list(doc["frames"].items())
        ex["frames"] = dict(fr[:n_frames])
        if len(fr) > n_frames:
            ex["frames"][f"… {len(fr) - n_frames} more frames"] = "…"
    if "frame_intervals" in doc:
        ex["frame_intervals"] = doc["frame_intervals"]
    objs = list(doc.get("objects", {}).items())
    ex["objects"] = dict(objs[:n_objects])
    if len(objs) > n_objects:
        ex["objects"][f"… {len(objs) - n_objects} more objects"] = "…"
    if "ontologies" in doc:
        ex["ontologies"] = doc["ontologies"]
    text = json.dumps({"openlabel": ex}, indent=1)
    # keep long numeric arrays on one line
    text = re.sub(r"\[\s+([-\d.e,\s]+?)\s+\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    text = re.sub(r'("boundary_list": )\[[^\]]*\]', r'\1["…"]', text)
    return text


def _catalog_head(path: Path, rows: int = 10) -> str:
    lines = path.read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("|")]
    extra = len(head) - 2 - rows
    body = head[: 2 + rows]
    txt = "\n".join(body)
    if extra > 0:
        txt += f"\n\n*… {extra} more rows in* `{path.name}`"
    return txt


def write(out: Path) -> None:
    data = json.loads((out / "measurements.json").read_text())
    rend = json.loads((out / "renders.json").read_text()) if (out / "renders.json").exists() \
        else {}
    shots_d = json.loads((out / "shots.json").read_text()) if (out / "shots.json").exists() \
        else {"views": [], "reused": {}}
    cold = json.loads((out / "cold_start.json").read_text()) if (out / "cold_start.json").exists() \
        else {}
    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    mach = data.get("machine", {})
    md: list[str] = []
    add = md.append
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    add("# oh-my-slam — development & performance report\n")
    add(f"Generated {now} from `{out}`. Machine: {mach.get('cpu')} ({mach.get('cores')} cores, "
        f"{mach.get('memory_gb')} GB), macOS {mach.get('tools', {}).get('macos')}, inference "
        f"server on **{mach.get('server', {}).get('device')}** "
        f"({mach.get('server', {}).get('precision')}, status "
        f"`{mach.get('server', {}).get('status')}` — SAM 3 unavailable, YOLOE-only). Warm "
        "server, every command run on its own, one after another.\n")
    add(_summary(out, data, shots_d, cold))
    add("## Contents\n\n1. [Performance per map](#performance-per-map)\n"
        "2. [restaurant.jpg (single image)](#restaurantjpg-single-image)\n"
        "3. [Per image updated](#per-image-updated)\n4. [Development report](#development-report)\n"
        "5. [Viewer screenshots](#viewer-screenshots)\n"
        "6. [Output formats](#output-formats)\n7. [Method and raw data](#method-and-raw-data)\n")

    # -------------------------------------------------------------- whole maps
    add("## Performance per map\n")
    add("Whole-map build: `mapper.sh update -a <input> -m <new map> -t full` (videos with "
        "`-fps 2`). *Wall-clock* is the process time seen by the caller; the stage times come from "
        "the command's own instrumentation (`OH_MY_SLAM_TIMINGS`); *server peak* is the inference "
        "server's physical footprint (incl. MPS memory) sampled every 0.25 s during the run; "
        "*client peak* is the command's peak RSS (children = largest COLMAP/OpenMVS process).\n")
    rows = []
    bars = []
    builds = {}
    for name in MAP_NAMES:
        rs = _runs(data, f"{name}_build")
        if not rs:
            continue
        r = rs[-1]
        builds[name] = r
        c = r["timings"]["counts"]
        mp = data["maps"].get(name, {})
        reg = c.get("keyframes_registered", 0)
        rows.append([
            TITLES[name], str(c.get("keyframes_sampled")),
            f"{reg} ({100 * reg / max(1, c.get('keyframes_sampled', 1)):.0f} %)",
            f"**{_fmt_s(r['wall_s'])}**", f"{r['wall_s'] / max(1, reg):.2f} s",
            f"{r['server_mem']['peak_gb']:.1f} GB",
            f"{r['timings']['peak_rss_mb']['self'] / 1e3:.1f} GB / "
            f"{r['timings']['peak_rss_mb']['children'] / 1e3:.1f} GB",
            _mb(mp.get("size_bytes", 0)), str(mp.get("objects", "")),
            str(c.get("sfm", "")),
        ])
        bars.append((TITLES[name].split(" (")[0], _group_totals(_stages(r))))
    add(_table(["map", "keyframes sampled", "registered", "wall-clock", "amortised / keyframe",
                "server peak", "client peak (self / children)", "map on disk", "objects",
                "pose method"], rows, "lrrrrrrrrl"))
    if bars:
        svg = _chart_svg([(b[0], b[1]) for b in bars], "min", "Whole-map build time by stage")
        (charts / "build_stages.svg").write_text(svg)
        add("\n![Whole-map build time by stage](charts/build_stages.svg)\n")
    # stage table
    if builds:
        head = ["stage"] + [TITLES[n].split(" (")[0] for n in builds]
        srows = []
        keys = [k for k in STAGES if any(_stages(r).get(k, 0.0) >= 0.005
                                         for r in builds.values())]
        for k in keys:
            row = [STAGES[k]]
            for r in builds.values():
                st = _stages(r)
                v = st.get(k)
                row.append("–" if v is None else f"{_fmt_s(v)} ({100 * v / r['wall_s']:.0f} %)")
            srows.append(row)
        srows.append(["**total (wall-clock)**"] + [f"**{_fmt_s(r['wall_s'])}**"
                                                   for r in builds.values()])
        add("\n### Stage breakdown (whole-map build)\n")
        add(_table(head, srows))
        add("\nStages under 5 ms in every run are omitted. Chart groups: *Inference* = "
            "per-keyframe inference + geometry re-run with the SfM focal; *Frame, depth, latest "
            "wins* = map frame/scale/floor levelling, depth alignment, persisting keyframes, "
            "validity masks; *Other* = lock/read map, ingest, export, commit, start-up.\n")
        # per keyframe inference split
        add("\n### Per-keyframe inference split\n")
        add("GPU compute per request as reported by the server (mean over the build), the "
            "client-observed time per call (includes queueing: two keyframes are in flight and "
            "detection runs concurrently with geometry), and the multi-view (MapAnything) calls "
            "made by the pose stage.\n")
        irows = []
        for name, r in builds.items():
            sv = r["timings"]["server"]
            pa = r["timings"]["parts"]
            n = r["timings"]["counts"].get("keyframes_sampled", 1)

            def per(k: str, sv: dict[str, Any] = sv) -> str:
                e = sv.get(k)
                return "–" if not e else f"{1000 * e['compute_s'] / e['count']:.0f} ms"

            def cl(k: str, pa: dict[str, Any] = pa) -> str:
                e = pa.get(k)
                return "–" if not e else f"{1000 * e['seconds'] / e['count']:.0f} ms"

            mv = sv.get("multiview")
            inf = r["timings"]["stages_s"].get("inference", 0.0)
            irows.append([TITLES[name].split(" (")[0], str(n), _fmt_s(inf),
                          f"{inf / max(1, n):.2f} s", per("geometry"), per("gravity"),
                          per("segment"), f"{cl('geometry')} / {cl('gravity')} / "
                          f"{cl('segmentation')}",
                          "–" if not mv else f"{mv['count']} calls, {_fmt_s(mv['compute_s'])}"])
        add(_table(["map", "keyframes", "inference stage", "per keyframe (wall)",
                    "geometry (MoGe-2) GPU", "gravity (GeoCalib) GPU", "segmentation (YOLOE) GPU",
                    "client per call geo / grav / seg", "multi-view (MapAnything)"], irows,
                   "lrrrrrrrr"))
        # map size breakdown
        add("\n### Map size on disk\n")
        zrows = []
        for name in builds:
            mp = data["maps"].get(name, {})
            br = mp.get("size_breakdown", {})
            top = sorted(br.items(), key=lambda kv: -kv[1])[:5]
            zrows.append([TITLES[name].split(" (")[0], _mb(mp.get("size_bytes", 0)),
                          str(mp.get("keyframes")),
                          f"{mp.get('size_bytes', 0) / max(1, mp.get('keyframes', 1)) / 1e6:.1f} MB",
                          ", ".join(f"`{k}` {_mb(v)}" for k, v in top)])
        add(_table(["map", "total", "keyframes", "per keyframe", "largest parts"], zrows,
                   "lrrrl"))

    # -------------------------------------------------------------- restaurant
    add("\n## restaurant.jpg (single image)\n")
    add("12 MP photo (4000×3000), EXIF focal. Each command run 5× after one warm-up run; the "
        "process wall-clock is what the caller sees (≈ 0.35 s of it is Python start-up and "
        "imports).\n")
    rrows = []
    rest_groups = [("restaurant_json", "`reconstruct.sh -i` (JSON)"),
                   ("restaurant_ply", "`reconstruct.sh -i -f ply`"),
                   ("restaurant_segment", "`segment.sh -i -o` (5 artefacts)")]
    for g, label in rest_groups:
        rs = _runs(data, g)
        if not rs:
            continue
        w = [r["wall_s"] for r in rs]
        stg = {k: statistics.median(_stages(r).get(k, 0.0) for r in rs)
               for k in _stages(rs[0])}
        sv = rs[0]["timings"]["server"]
        gpu = ", ".join(f"{k} {1000 * statistics.median(r['timings']['server'][k]['compute_s'] for r in rs):.0f} ms"
                        for k in sv)
        main = ", ".join(f"{STAGES.get(k, k).lower()} {_fmt_s(v)}"
                         for k, v in sorted(stg.items(), key=lambda kv: -kv[1]) if v >= 0.01)
        extra = rs[-1]["timings"].get("objects") or rs[-1]["timings"].get("points") or \
            rs[-1]["timings"]["counts"].get("objects")
        rrows.append([label, _fmt_s(min(w)), f"**{_fmt_s(statistics.median(w))}**",
                      _fmt_s(max(w)), main, gpu,
                      f"{statistics.median(r['server_mem']['peak_gb'] for r in rs):.1f} GB",
                      f"{rs[-1]['timings']['peak_rss_mb']['self'] / 1e3:.2f} GB",
                      f"{extra:,}" if isinstance(extra, int) else str(extra)])
    add(_table(["command", "min", "median", "max", "median stages", "server GPU compute (median)",
                "server peak", "client peak RSS", "objects / points"], rrows, "lrrrllrrr"))
    add(f"\nServer cold start with cached weights: **{cold.get('cold_start_s', '–')} s** "
        f"(`./start_inference_server.sh` after `--stop`); footprint after start "
        f"{mach.get('server_footprint_gb_at_start')} GB.\n")

    # -------------------------------------------------------------- per image
    add("\n## Per image updated\n")
    inp = data.get("update_inputs", {})
    add("One new image added to each existing map, 5 times, each time to a fresh APFS clone of "
        "the whole-map build (`cp -c -R`), with `mapper.sh update -a <image> -m <copy> -t single`. "
        "Images: " + "; ".join(f"**{n}** `{Path(p).name}`" for n, p in inp.items()) + ". "
        "The ainex image is a capture that is already in the map (the robot revisiting a "
        "heading — there is no held-out capture); the video frames are extracted with ffmpeg at "
        "the given time and are not identical to a sampled keyframe.\n")
    urows = []
    ubars = []
    upd = {}
    for name in MAP_NAMES:
        rs = _runs(data, f"{name}_update")
        if not rs:
            continue
        upd[name] = rs
        w = [r["wall_s"] for r in rs]
        c = rs[-1]["timings"]["counts"]
        reg = [r["timings"]["counts"].get("keyframes_registered", 0) for r in rs]
        removed = [(_objects_log(out, r["tag"]) or {}).get("removed", "?") for r in rs]
        urows.append([TITLES[name].split(" (")[0], str(c.get("map_frames_before")), str(len(rs)),
                      _fmt_s(min(w)), f"**{_fmt_s(statistics.median(w))}**", _fmt_s(max(w)),
                      f"{sum(reg)}/{len(rs)}", ", ".join(map(str, removed)),
                      f"{statistics.median(r['server_mem']['peak_gb'] for r in rs):.1f} GB",
                      f"{statistics.median(r['timings']['peak_rss_mb']['self'] for r in rs) / 1e3:.1f} GB",
                      str(c.get("sfm", ""))])
        med = [statistics.median(_group_totals(_stages(r))[i] for r in rs)
               for i in range(len(GROUPS))]
        ubars.append((TITLES[name].split(" (")[0], med))
    add(_table(["map", "keyframes in map", "runs", "min", "median", "max", "registered",
                "objects removed (per run)", "server peak (median)", "client peak (median)",
                "pose method"], urows, "lrrrrrrrrrl"))
    if ubars:
        svg = _chart_svg(ubars, "s", "One-image update time by stage (median of 5)",
                         [statistics.median(r["wall_s"] for r in rs) for rs in upd.values()])
        (charts / "update_stages.svg").write_text(svg)
        add("\n![One-image update time by stage (median of 5)](charts/update_stages.svg)\n")
    if upd:
        head = ["stage"] + [f"{TITLES[n].split(' (')[0]} median [min–max]" for n in upd]
        keys = [k for k in STAGES if any(_stages(r).get(k, 0.0) >= 0.005
                                         for rs in upd.values() for r in rs)]
        srows = []
        for k in keys:
            row = [STAGES[k]]
            for rs in upd.values():
                vals = [_stages(r).get(k, 0.0) for r in rs]
                row.append(f"{_fmt_s(statistics.median(vals))} [{_fmt_s(min(vals))}–"
                           f"{_fmt_s(max(vals))}]")
            srows.append(row)
        srows.append(["**total (wall-clock)**"] + [
            f"**{_fmt_s(statistics.median(r['wall_s'] for r in rs))}**" for rs in upd.values()])
        add("\n### Stage breakdown (one-image update)\n")
        add(_table(head, srows))
        ply = {n: _runs(data, f"{n}_update_ply") for n in upd}
        if any(ply.values()):
            add("\nThe same update with `-f ply -t full` (the whole updated map cloud on stdout, "
                "rendered under *Output formats*): " + "; ".join(
                    f"{TITLES[n].split(' (')[0]} {_fmt_s(v[0]['wall_s'])} "
                    f"({v[0]['stdout_bytes'] / 1e6:.0f} MB PLY)" for n, v in ply.items() if v)
                + ".\n")

    # -------------------------------------------------------------- development report
    add("\n## Development report\n")
    if (out / "development.md").exists():
        add((out / "development.md").read_text().strip() + "\n")

    # -------------------------------------------------------------- screenshots
    add("\n## Viewer screenshots\n")
    add("Captured with headless Microsoft Edge (Playwright) at 1440×900 against the live "
        "`view.sh` server for the whole-map builds of this report and for "
        "`view.sh -i /Users/U124317/robot_view/restaurant.jpg`. Each layer is shown alone with "
        "the other five switched off.\n")
    for rec in shots_d.get("views", []):
        v = rec["view"]
        add(f"### {TITLES.get(v, v)} — `{rec['cmd']}`\n")
        facts = [f"page ready in {rec.get('page_ready_s', '–')} s",
                 f"stats line: *{rec.get('stats_line', '')}*",
                 f"console errors: {len(rec.get('errors', []))}"]
        if "map_unchanged" in rec:
            facts.append("map unchanged after viewing: " + ("yes" if rec["map_unchanged"]
                                                              else "**no**"))
        if rec.get("unavailable_layers"):
            facts.append("not available in this mode: " + ", ".join(
                LAYER_TITLES[k] for k in rec["unavailable_layers"]))
        if rec.get("tooltip"):
            facts.append(f"tooltip: “{rec['tooltip']}”")
        add("- " + "\n- ".join(facts) + "\n")
        for s in rec.get("shots", []):
            add(f"![{s['caption']}]({s['png']})\n*{s['caption']}*\n")
        for s in shots_d.get("reused", {}).get(v, []):
            add(f"![{s['caption']}]({s['png']})\n*{s['caption']} — from the validation run "
                f"`{VALIDATION.name}`*\n")

    # -------------------------------------------------------------- formats
    add("\n## Output formats\n")
    add("Every supported output, per map and for restaurant.jpg. PLY files are drawn with a small "
        "z-buffered point splatter (no re-colouring: the pixels are the PLY's own colours on the "
        "viewer's background); `mesh.glb` is shown by the viewer's mesh-only layer above (the "
        "viewer loads the map's `mesh/mesh.glb` unchanged).\n")
    fmt_sources = {"restaurant": ("outputs/restaurant/reconstruct.json",
                                  "outputs/restaurant/segment_o/catalog.md")}
    for n in MAP_NAMES:
        fmt_sources[n] = (f"outputs/{n}/full.json", f"outputs/{n}/segment_m/catalog.md")
    for n, (js, cat) in fmt_sources.items():
        if not (out / js).exists():
            continue
        add(f"### {TITLES[n]}\n")
        cmd = "reconstruct.sh -i restaurant.jpg" if n == "restaurant" else \
            "mapper.sh update … -t full"
        size = (out / js).stat().st_size
        add(f"**OpenLABEL JSON** (`{cmd}`, {size / 1e3:.0f} kB; excerpt — first "
            f"{'object' if n == 'restaurant' else 'frame and objects'}, arrays shortened):\n")
        add("```json\n" + _json_excerpt(out / js) + "\n```\n")
        for item in rend.get(n, []):
            if item["kind"] in ("ply", "segments"):
                add(f"![{item['caption']}]({item['png']})\n*{item['caption']} — "
                    f"{item['points']:,} points*\n")
        for item in rend.get(n, []):
            if item["kind"] == "segmented":
                add(f"![{item['caption']}]({item['png']})\n*{item['caption']} "
                    f"({item['size'][0]}×{item['size'][1]} px)*\n")
        mesh_shot = next((s for rec in shots_d.get("views", []) if rec["view"] == n
                          for s in rec.get("shots", []) if s["name"] == "layer_mesh"), None)
        if mesh_shot:
            glb = out / "maps" / n / "mesh" / "mesh.glb"
            add(f"![mesh.glb]({mesh_shot['png']})\n*`mesh/mesh.glb` "
                f"({glb.stat().st_size / 1e6:.0f} MB) in the viewer, mesh layer only*\n")
        if (out / cat).exists():
            add(f"**catalog.md** (`{cat}`, first rows):\n")
            add(_catalog_head(out / cat) + "\n")

    # -------------------------------------------------------------- method
    add("\n## Method and raw data\n")
    add("- Reproduce: `uv run python -m oh_my_slam.tools.perf_report measure|renders|shots|write "
        "--out <dir>` (warm server; `measure` is strictly sequential and never runs two commands "
        "at once).\n"
        "- `measurements.json` — every run: command, exit code, wall-clock, server memory "
        "(start/peak/end), the full timing record.\n"
        "- `timings/<run>.json` — the `OH_MY_SLAM_TIMINGS` record of each run; `logs/` — stderr "
        "of each run (the `timings:` summary line is its last line); `outputs/` — stdout "
        "payloads and `segment.sh -o` artefacts; `maps/` — the whole-map builds and one updated "
        "copy per map (`<map>_updated`).\n"
        "- Stage times are exclusive (a nested stage is not counted in its parent); "
        "*start-up, imports, other* is the wall-clock minus the sum of the stages. Parts "
        "(geometry, gravity, segmentation, lift) overlap in time and are reported per call.\n")
    text = "\n".join(md).rstrip() + "\n"
    (out / "report.md").write_text(text)
    (out / "report.html").write_text(_to_html(text, out))


def _summary(out: Path, data: dict[str, Any], shots_d: dict[str, Any], cold: dict[str, Any]) -> str:
    """Headline table and findings, computed from the measurements."""
    md = ["## Summary\n"]
    rows = []
    geo_keys = ("cloud", "fusion", "mesh", "texture")
    facts: dict[str, dict[str, float]] = {}
    for name in MAP_NAMES:
        b = _runs(data, f"{name}_build")
        u = _runs(data, f"{name}_update")
        if not b:
            continue
        r = b[-1]
        c = r["timings"]["counts"]
        mp = data["maps"].get(name, {})
        uw = [x["wall_s"] for x in u]
        st = _stages(r)
        n = c.get("keyframes_sampled", 1)
        sv = r["timings"]["server"]
        facts[name] = {
            "build": r["wall_s"], "objects_share": st.get("objects", 0) / r["wall_s"],
            "inference_share": st.get("inference", 0) / r["wall_s"],
            "inf_per_kf": st.get("inference", 0) / max(1, n),
            **{f"gpu_{k}": 1000 * sv[k]["compute_s"] / sv[k]["count"] for k in
               ("geometry", "gravity", "segment") if k in sv},
            "client_peak": r["timings"]["peak_rss_mb"]["self"],
            "child_peak": r["timings"]["peak_rss_mb"]["children"],
            "server_peak": r["server_mem"]["peak_gb"],
        }
        if u:
            med = statistics.median(uw)
            facts[name]["update"] = med
            facts[name]["update_geo_share"] = statistics.median(
                sum(_stages(x).get(k, 0) for k in geo_keys) / x["wall_s"] for x in u)
            facts[name]["update_image"] = statistics.median(
                sum(_stages(x).get(k, 0) for k in ("inference", "features_matching", "sfm"))
                for x in u)
        rows.append([
            f"**{TITLES[name]}**", f"**{_fmt_s(r['wall_s'])}** ({r['wall_s']:.0f} s)",
            f"{n} ({c.get('keyframes_registered')})",
            f"{r['wall_s'] / max(1, c.get('keyframes_registered', 1)):.1f} s",
            f"**{_fmt_s(statistics.median(uw))}** ({_fmt_s(min(uw))}–{_fmt_s(max(uw))})"
            if uw else "–",
            f"{r['server_mem']['peak_gb']:.1f} GB", _mb(mp.get("size_bytes", 0))])
    rest = {g: _runs(data, g) for g in ("restaurant_json", "restaurant_ply", "restaurant_segment")}
    med = {g: statistics.median(x["wall_s"] for x in rs) if rs else None for g, rs in rest.items()}
    if rest["restaurant_json"]:
        rows.append(["**restaurant.jpg**",
                     f"`reconstruct.sh` JSON **{_fmt_s(med['restaurant_json'])}**, `-f ply` "
                     f"**{_fmt_s(med['restaurant_ply'])}**, `segment.sh -o` "
                     f"**{_fmt_s(med['restaurant_segment'])}** (median of 5)", "1", "–", "–",
                     f"{rest['restaurant_json'][-1]['server_mem']['peak_gb']:.1f} GB", "–"])
    md.append(_table(["", "whole-map build", "keyframes (registered)", "per registered keyframe",
                      "one-image update: median (min–max)", "server peak", "map on disk"],
                     rows, "lrrrrrr"))
    checks = []
    if med.get("restaurant_json") is not None:
        checks.append(f"single-image JSON {_fmt_s(med['restaurant_json'])} ≤ 3.0 s and PLY "
                      f"{_fmt_s(med['restaurant_ply'])} ≤ 1.5 s (AC22)")
    if "church" in facts:
        checks.append(f"church {_fmt_s(facts['church']['build'])} ≤ 6 min (AC27)")
    if "livingroom" in facts:
        checks.append(f"livingroom {_fmt_s(facts['livingroom']['build'])} ≤ 18 min (AC28)")
        if "update" in facts["livingroom"]:
            checks.append(f"one-image update of the 184-keyframe map "
                          f"{_fmt_s(facts['livingroom']['update'])} ≤ 3 min (AC22)")
    if cold:
        checks.append(f"server cold start {cold.get('cold_start_s')} s ≤ 120 s")
    peaks = [f["server_peak"] for f in facts.values()]
    if peaks:
        checks.append(f"server footprint {data.get('machine', {}).get('server_footprint_gb_at_start')}"
                      f" GB after start, ≤ {max(peaks):.1f} GB peak (≤ 22 GB multi-view budget)")
    all_reg = all(_runs(data, f"{n}_build")[-1]["timings"]["counts"].get("keyframes_registered")
                  == _runs(data, f"{n}_build")[-1]["timings"]["counts"].get("keyframes_sampled")
                  for n in facts)
    bullets = ["**Targets that apply, measured here:** " + "; ".join(checks) + "."
               + (" Every sampled keyframe of the three inputs was registered." if all_reg else "")]

    def rng(key: str, fmt: str = "{:.0f}", scale: float = 1.0) -> str:
        vals = [f[key] * scale for f in facts.values() if key in f]
        lo, hi = min(vals), max(vals)
        return fmt.format(lo) if abs(hi - lo) < 1e-9 else f"{fmt.format(lo)}–{fmt.format(hi)}"

    if facts:
        share = ", ".join(f"{100 * f['objects_share']:.0f} % of {n}" for n, f in facts.items())
        bullets.append(
            "**Where the build time goes:** the *objects* stage (association, merging and "
            f"visibility tests over keyframe pairs) is the largest — {share}. Per-keyframe GPU "
            f"inference is {rng('inference_share', '{:.0f}', 100)} % of the build "
            f"({rng('inf_per_kf', '{:.2f}')} s per keyframe wall; GPU compute per keyframe: "
            f"MoGe-2 {rng('gpu_geometry')} ms, GeoCalib {rng('gpu_gravity')} ms, YOLOE "
            f"{rng('gpu_segment')} ms).")
        if all("update_geo_share" in f for f in facts.values()):
            ushare = ", ".join(f"{100 * f['update_geo_share']:.0f} % ({n})"
                               for n, f in facts.items())
            bullets.append(
                "**Where the update time goes:** rebuilding the *whole map's* cloud, TSDF mesh "
                f"and texture — {ushare} of the one-image update; the new image itself costs "
                f"{rng('update_image', '{:.1f}')} s (inference, matching, registration).")
        bullets.append(
            f"**Memory:** mapper process peak {rng('client_peak', '{:.1f}', 1e-3)} GB, "
            f"COLMAP/OpenMVS children up to {max(f['child_peak'] for f in facts.values()) / 1e3:.1f}"
            f" GB; server peak {rng('server_peak', '{:.1f}')} GB (highest with MapAnything).")
    views = shots_d.get("views", [])
    if views:
        ready = [v.get("page_ready_s", 0) for v in views]
        errs = sum(len(v.get("errors", [])) for v in views)
        unchanged = all(v.get("map_unchanged", True) for v in views)
        n_new = sum(len(v.get("shots", [])) for v in views)
        n_old = sum(len(x) for x in shots_d.get("reused", {}).values())
        bullets.append(f"**Viewer:** the four views are ready {min(ready):.1f}–{max(ready):.1f} s "
                       f"after page load, {errs} console errors, maps "
                       f"{'unchanged' if unchanged else '**changed**'} by viewing; {n_new} new "
                       f"screenshots and {n_old} reused from the validation run.")
    bullets.append("Known issues and blocked items (AC26/AC27 data-limited — C31/C30, small "
                   "\"television\" false positives, SAM 3 inaccessible — U1, dataset-based "
                   "accuracy targets not run — U2) are in the development report.")
    md.append("\n" + "\n".join(f"- {b}" for b in bullets) + "\n")
    return "\n".join(md)


_CSS = """
:root{--bg:#fcfcfb;--fg:#1d1d1b;--muted:#5f5e5a;--line:#e3e2dd;--code:#f3f2ee;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#ecebe6;
--muted:#b3b2aa;--line:#383835;--code:#242422;--accent:#6da7ec}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#ecebe6;--muted:#b3b2aa;--line:#383835;--code:#242422;
--accent:#6da7ec}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,system-ui,
"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:28px;margin:0 0 8px}h2{font-size:22px;margin:44px 0 12px;padding-top:12px;
border-top:1px solid var(--line)}h3{font-size:17px;margin:28px 0 8px}
a{color:var(--accent)}
p,li{max-width:900px}
table{border-collapse:collapse;margin:12px 0;font-size:13px;display:block;overflow-x:auto;
max-width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top;
font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:600;white-space:nowrap}
td[style*="right"],th[style*="right"]{white-space:nowrap}
code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:.92em}
pre{background:var(--code);padding:12px;border-radius:8px;overflow:auto;max-height:420px;
font-size:12px;line-height:1.4}pre code{padding:0;background:none}
img{max-width:100%;height:auto;border-radius:6px;border:1px solid var(--line);display:block;
margin:10px 0 2px}
em{color:var(--muted)}
svg{display:block;margin:14px 0}
"""


def _to_html(text: str, out: Path) -> str:
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    except ImportError:  # pragma: no cover
        from markdown_it import MarkdownIt

        body = MarkdownIt("commonmark").enable("table").render(text)

    def inline_svg(m: re.Match[str]) -> str:
        p = out / m.group(2)
        return p.read_text() if p.exists() else m.group(0)

    body = re.sub(r'<img alt="([^"]*)" src="(charts/[^"]+\.svg)" ?/?>', inline_svg, body)
    body = body.replace("<img ", '<img loading="lazy" ')
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>oh-my-slam report</title>"
            f"<style>{_CSS}</style></head><body><main>\n{body}\n</main></body></html>\n")
