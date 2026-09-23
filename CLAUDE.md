# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repository

Implementation of `high_level_spec.md` (devtask `~/.dev-workflow/oh-my-slam.md` is the plan and
the record of decisions). Package `src/oh_my_slam` (uv, Python 3.12); see `README.md`.

Commands:

- Environment: `uv sync`; external tools: `brew install colmap` + `./scripts/install_tools.sh`.
- Server: `./start_inference_server.sh [--status|--stop|--foreground]` (log in
  `~/Library/Caches/oh-my-slam/server.log`).
- Offline tests: `uv run pytest -m "not models and not browser and not eval" -q`
  (coverage: add `--cov=oh_my_slam`). Real models: `OH_MY_SLAM_TEST_REAL_SERVER=1 uv run pytest -m models`
  with the server running. Browser: `uv run pytest -m browser` (Playwright, Edge channel).
- Lint/types/ownership: `uv run ruff check . && uv run mypy src && uv run lint-imports`.
- Validation on the user's inputs: `uv run python -m oh_my_slam.tools.validate_inputs --inputs
  /Users/U124317/robot_view` (outputs in `~/oh-my-slam-data/validation/<UTC>/`).
- Performance/development report (R44): `uv run python -m oh_my_slam.tools.perf_report
  measure|renders|shots|write --out DIR` (`~/oh-my-slam-data/reports/<UTC>/`). Per-stage timings of
  any `mapper.sh update` / `reconstruct.sh` / `segment.sh -i`: `timings:` line on stderr, JSON with
  `OH_MY_SLAM_TIMINGS=path`, and `map.json → updates[].timings`.

Never import torch/open3d in the same process (duplicate libomp aborts): torch lives only in the
server process. Never edit files in `.staging/` of a map; `view.sh -m` / `segment.sh -m` are read-only.

## Spec invariants that span several sections

- **Five entry points:** `start_inference_server.sh`, `reconstruct.sh`, `mapper.sh`, `segment.sh`, `view.sh`.
- **Ownership:**
  - `segment.sh` is the only owner of segmentation, OBB fitting and colour assignment.
  - `mapper.sh` delegates depth/reconstruction to `reconstruct.sh`.
  - `view.sh` owns only the web server and UI.
  - Shared logic lives in one common Python package. Taken literally at the shell level these rules
    form a cycle (`reconstruct -f json` needs objects, `segment` needs depth), so enforce ownership at
    the Python-module level.
- **Inference server:** every inference-requiring operation must fail with an actionable error when
  the server is down. `view.sh -m` (viewing a persisted map) must work without it.
- **stdout:** exactly one JSON document or one PLY. Everything else goes to stderr.
- **Colour contract:** an object's colour is a pure function of its `id`. The same sRGB triple must
  appear in all of these:
  - `segmentation.json`
  - the `segmented.png` mask pixels, which rules out alpha blending
  - `catalog.csv` and `catalog.md`
  - `segments.ply`, where unsegmented points are mid-grey
  - the OBB colour rendered by `view.sh`
- **Scene JSON is ASAM OpenLABEL 1.0.0** (verified online 2026-09-22):
  - The canonical schema URL is `https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json`.
    The `…-v1.0.0.json` variant returns 404.
  - The schema is Draft-07 and its root allows only `openlabel`, so a root `$schema` fails
    validation. Put the URL in `metadata.schema_url`.
  - A 10-value cuboid is `x,y,z,qx,qy,qz,qw,sx,sy,sz`, with the quaternion scalar last.
  - The schema never validates stream intrinsics or the top-level `frame_intervals`; check those in code.
- **Input and platform:** RGB only (no depth sensor, stereo or IMU). Must run on the Apple M4 Max.
  MPS is preferred.

## Machine and tooling facts (verified locally)

- Apple M4 Max, 36 GB unified memory, macOS 26, no CUDA.
- **COLMAP:**
  - Homebrew `colmap` 4.2.0 includes GLOMAP as `global_mapper`, and ALIKED/LightGlue feature
    extraction and matching through ONNX Runtime with the CoreML provider.
  - The **PyPI pycolmap 4.2 wheel is built without ONNX**. It lists the ALIKED/LightGlue enums but
    fails when they are used. Run learned features through the Homebrew CLI and mapping through
    pycolmap, and keep both on the same 4.2.x version.
  - `patch_match_stereo` (dense MVS) requires CUDA, so it is unavailable on this machine.
- **PyTorch MPS:**
  - Two threads touching MPS in one process abort the whole process (pytorch#197805; its fix was
    reverted). Route *all* device work — load, `.to()`, forward, readback — through a single thread.
  - torch 2.13 produces corrupted attention output on macOS 26; use 2.14+.

## Working rules for this repo

- **Never name a directory `benchmark/` or `eval_results/`.** An external AI IDE (Antigravity)
  deletes directories with those names across checkouts. Keep evaluation data and anything
  irreplaceable outside the repo.
- **Never bind a fixed port.** Orphaned processes from earlier work hold ports such as 8080, 8081
  and 8088.
- Run benchmarks one at a time; concurrent GPU work invalidates timings.
- Research subagents verify claims online only: no installs, weight downloads or local benchmarks.
  This machine is the user's daily driver.
