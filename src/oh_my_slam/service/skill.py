"""``SKILL.md``: a short API reference for the robot-guide client, generated from the endpoint
definitions (``routes.ENDPOINTS``), the error table and the schema, as ``/api/openapi.json`` is.

    python -m oh_my_slam.service.skill      # rewrite SKILL.md at the checkout's root

``tests/unit/test_service_skill.py`` fails while the file is stale."""

from __future__ import annotations

import sys
from pathlib import Path

from oh_my_slam.service.app import CAPABILITIES
from oh_my_slam.service.errors import ERRORS
from oh_my_slam.service.routes import ENDPOINTS
from oh_my_slam.service.schema import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[3]
FILE = ROOT / "SKILL.md"


def render() -> str:
    rows = []
    for e in ENDPOINTS:
        body = "" if not e.request else f"[`{e.request}`] (multipart: part `request` + file " \
            "parts)" if e.multipart else f"[`{e.request}`] (JSON)"
        ok = ", ".join(f"{s} [`{n}`]" if n else f"{s} `{e.media}`" if e.media else str(s)
                       for s, n in e.responses.items())
        params = ", ".join(f"`{p.name}`" for p in e.params if p.where == "query")
        rows.append(f"| `{e.method} {e.path}` | {e.summary} | {body} | {ok} | {params} |")
    errors = [f"| `{code}` | {status} | {'yes' if retry else 'no'} |"
              for code, (status, retry) in ERRORS.items()]
    return "\n".join([
        "---",
        "name: oh-my-slam",
        f'description: "Short reference of the oh-my-slam HTTP API (schema {SCHEMA_VERSION}) for '
        "the robot-guide skill's client: update jobs (and dry runs) that build an anchored map "
        "from camera frames with pose priors, per-frame status and the quality gate; the map "
        "summary, keyframes, objects and support surfaces, navigation grid, points near a place "
        "and bird's-eye image of any kept version; locating frames with uncertainty and "
        'ambiguity, the recorded pose; health and timings."',
        "---",
        "",
        "# oh-my-slam API",
        "",
        "Generated from the API definitions (`python -m oh_my_slam.service.skill`); never edit it "
        "by hand. The full contract is `GET /api/openapi.json` (OpenAPI 3.1, its schemas those of "
        "`specs/api.schema.json`) and `specs/api.md`.",
        "",
        "* Service: `server.sh` on port 9091, every route under `/api/`, reachable from the LAN.",
        f"* Every JSON body carries `schema` (`{SCHEMA_VERSION}`); units are metres, degrees and "
        "seconds; `T_a_b` is the 4x4 row-major pose of b in a; the map frame has z up.",
        f"* Implemented groups (`GET /api/health` → `capabilities`): {', '.join(CAPABILITIES)}. "
        "Refuse what is missing instead of calling it.",
        "* Errors answer `{schema, error: {code, message, retryable, details}}`; a frame that "
        "cannot be used is an input status, never an error.",
        "",
        "## Endpoints",
        "",
        "| Route | Purpose | Body | Response | Query |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        "An update answers `202` with a job (`Location: /api/jobs/{id}`); poll it with "
        "`?wait_s=30` until `state` is `committed`, `rejected` or `failed`. `inputs[]` gives each "
        "frame's `status`, `reason`, `T_map_cam` and sigmas; `quality` the residuals and the gate.",
        "",
        "```sh",
        "curl -s http://HOST:9091/api/maps/home/updates \\",
        "  -F 'request={\"cameras\":{\"head\":{\"fx\":525,\"fy\":525,\"cx\":320,\"cy\":240,"
        "\"width\":640,\"height\":480}},\"frames\":[{\"name\":\"f0\",\"image\":\"f0\","
        "\"camera_id\":\"head\",\"prior\":{\"trust\":\"metric\",\"T_map_cam\":[[0,0,1,0],"
        "[-1,0,0,0],[0,-1,0,0.5],[0,0,0,1]],\"sigma_pos_m\":0.02,\"sigma_rot_deg\":1}}]};"
        "type=application/json' \\",
        "  -F f0=@f0.jpg",
        "curl -s 'http://HOST:9091/api/jobs/JOB_ID?wait_s=30'",
        "```",
        "",
        "## Errors",
        "",
        "| `code` | HTTP | Retryable |",
        "| --- | --- | --- |",
        *errors,
        "",
    ])


def main(argv: list[str]) -> int:
    FILE.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
