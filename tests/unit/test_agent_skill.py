"""The agent skill ``SKILL.md`` (spec §2.7): the committed file is the generator's output, it
describes every API route of the app with a ``curl`` command and a sample, every code the service
answers with, a registry change (a new option, output, error or mode) reaches it with no hand
edit, and its address snippet is POSIX ``sh`` that finds a running ``server.sh`` in the spec's
order."""

from __future__ import annotations

import dataclasses
import inspect
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from oh_my_slam.commands import spec
from oh_my_slam.core.errors import ExitCode, OhMySlamError, error_code
from oh_my_slam.web import skill
from oh_my_slam.web.app import Service, create_app
from oh_my_slam.web.jobs import Runner
from oh_my_slam.web.workspace import Workspace
from tests.fakes import slow_command
from tests.unit.test_web_server_sh import server, start

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "src" / "oh_my_slam" / "web"
REGENERATE = "regenerate it: uv run python -m oh_my_slam.web.skill"


def committed() -> str:
    return (REPO / "SKILL.md").read_text("utf-8")


def snippet() -> str:
    """The address snippet as SKILL.md gives it (inside ``sh <<'EOF'`` … ``EOF``)."""
    m = re.search(r"^sh <<'EOF'\n(.*?)^EOF$", committed(), re.S | re.M)
    assert m, "no address snippet in SKILL.md"
    return m.group(1)


def test_the_committed_skill_is_the_generated_one() -> None:
    assert committed() == skill.render(), f"SKILL.md is out of date; {REGENERATE}"
    assert skill.DEFAULT_PATH == REPO / "SKILL.md"


def test_front_matter() -> None:
    _, front, body = committed().split("---\n", 2)
    meta = yaml.safe_load(front)
    assert meta["name"] == "oh-my-slam-api" and set(meta) == {"name", "description"}
    assert 0 < len(meta["description"]) <= skill.DESCRIPTION_MAX == 1024
    assert "openapi.json" in body and "the service's document wins" in body


def test_the_description_states_every_capability_and_when_to_use_it(tmp_path: Path) -> None:
    """Spec §2.7 "Description": every operation with what it produces, every other feature of
    the API, a running service, and the operations that need the inference server."""
    text = yaml.safe_load(committed().split("---\n", 2)[1])["description"]
    for p in spec.PROGRAMS:  # generated from the commands' own descriptions
        assert p.prog in text and p.description.rstrip(".") in text
    ops = skill.operations()
    for op in ops.values():
        assert re.search(rf"\b{re.escape(op.id)}\b", text), op.id
        for out in op.mode.outputs:
            assert skill.FORMATS.get(out.format, out.format) in text, (op.id, out.format)
    never = [op.id for op in ops.values() if op.mode.inference == "never"]
    sometimes = [op.id for op in ops.values() if op.mode.inference == "conditional"]
    assert f"Inference server needed except for {', '.join(never)} " \
           f"({', '.join(sometimes)}: at times)." in text
    assert "a server.sh is running" in text and "Use when the user asks" in text
    # every route outside the operations is one of the service's features the description names
    paths = {r.split(" ", 1)[1] for r in api_routes(tmp_path)
             if r.split(" ", 1)[1].startswith(("/api/", "/viewer/"))}
    unnamed = [p for p in paths if not p.startswith("/api/ops/")
               and not any(part in p for parts, _ in skill.FEATURES for part in parts)]
    assert not unnamed, f"routes without words in skill.FEATURES: {unnamed}"
    assert all(words in text for _, words in skill.FEATURES)


def api_routes(tmp_path: Path) -> list[str]:
    """``METHOD /path`` of every route of the real app (HEAD aside), path parameters as in the
    OpenAPI document."""
    ws = Workspace(tmp_path / "data")
    ws.create()
    service = Service(ws, Runner(ws), url="http://0.0.0.0:0/")
    app = inspect.getclosurevars(create_app(service)).nonlocals["app"]  # inside the guard
    return sorted(f"{m} " + re.sub(r"\{(\w+):\w+\}", r"{\1}", r.path)
                  for r in app.routes for m in r.methods - {"HEAD"})


def test_every_route_is_described_with_curl_and_a_sample(tmp_path: Path) -> None:
    text = committed()
    routes = api_routes(tmp_path)
    api = [r for r in routes if r.split(" ", 1)[1].startswith(("/api/", "/viewer/"))]
    assert len([r for r in api if " /api/" in r]) >= 26, api
    missing = [r for r in api if f"`{r}`" not in text]
    assert not missing, f"routes missing from SKILL.md ({REGENERATE}): {missing}"
    for op in skill.operations():
        assert f"`POST /api/ops/{op}` · `POST /api/ops/{op}/validate`" in text, op
    # each endpoint and operation: a ready-to-run curl command and a sample answer
    endpoints = text.split("\n## Endpoints\n", 1)[1].split("\n#### ")[1:]
    operations = text.split("\n## Operations\n", 1)[1].split("\n## Endpoints\n")[0] \
        .split("\n### `")[2:]
    assert len(endpoints) >= 20 and len(operations) == len(skill.operations())
    for section in [*endpoints, *operations]:
        assert "```sh\ncurl " in section, section[:80]
        assert "\n→ " in section or "\n```text\n" in section, section[:80]
    for section in operations:
        assert "→ validate: `" in section and "→ submit: 202" in section, section[:80]
    by_route = {s.split("\n", 1)[0]: s for s in endpoints}
    assert "curl -sS -N " in by_route["`GET /api/jobs/{id}/events`"]
    assert "-X POST -T " in by_route["`POST /api/uploads`"]
    for route in ("`GET /api/jobs/{id}/result`", "`GET /api/jobs/{id}/files/{path}`",
                  "`GET /api/maps/{name}/files/{path}`"):
        assert "curl -sS -o " in by_route[route] or "--create-dirs -o " in by_route[route]


def test_every_code_the_service_answers_with_is_described() -> None:
    """The service's own refusals (``_error``, ``JobError``, a job's ``error``) are in SKILL.md,
    besides the commands' exit codes, which come from the registry."""
    app, jobs = (WEB / "app.py").read_text("utf-8"), (WEB / "jobs.py").read_text("utf-8")
    codes = {c for m in re.finditer(r'_error\(\w+, "(\w+)"(?: if [^"]* else "(\w+)")?', app)
             for c in m.groups() if c}
    codes |= set(re.findall(r'JobError\(\d+,.*?, "(\w+)"\)', app + jobs, re.S))
    codes |= set(re.findall(r'"code": "(\w+)"', jobs))
    assert {"forbidden", "unsupported_media_type", "too_large", "gone", "not_cancellable",
            "interrupted", "cancelled"} <= codes
    text = committed()
    assert not [c for c in codes if f"`{c}`" not in text], codes
    assert set(skill.SERVICE_ERRORS) >= codes - {error_code(c) for c in ExitCode} - {"cancelled"}
    assert all(f"| `{error_code(c)}` | {int(c)} |" in text for c in ExitCode)


class ShadeError(OhMySlamError):
    exit_code = ExitCode.MAP_LOCKED


def test_a_registry_change_reaches_the_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new option, output and error on segment.sh -i, and a whole new command, change the
    skill with no edit to the generator."""
    seg = spec.SEGMENT.command()
    shade = spec.Option("--shade", "shade", spec.Kind.ENUM, "shade of the overlay",
                        default="dark", choices=("dark", "light"))
    rule = spec.Rule("shade", ("shade",), "--shade suits the overlay", lambda ctx: None,
                     (ShadeError,))
    overlay = spec.Output("overlay.png", "-d", "png", "the overlay in its shade",
                          (spec.When("artifacts"),))
    image = dataclasses.replace(spec.SEGMENT_IMAGE, rules=(*spec.SEGMENT_IMAGE.rules, rule),
                                outputs=(*spec.SEGMENT_IMAGE.outputs, overlay))
    seg2 = dataclasses.replace(seg, options=(*seg.options, shade), modes=(image, *seg.modes[1:]))
    monkeypatch.setattr(spec, "PROGRAMS", (
        *[p for p in spec.PROGRAMS if p is not spec.SEGMENT],
        dataclasses.replace(spec.SEGMENT, commands=(seg2,)), slow_command.registry_program()))

    text = skill.render()
    assert text != committed()
    section = text.split("### `segment-image`", 1)[1].split("\n### ", 1)[0]
    assert "| `shade` | `--shade` | one of `dark`, `light` | `dark` | shade of the overlay |" \
        in section
    assert "* `<artifacts>/overlay.png` when `artifacts` is given" in section
    assert "--shade suits the overlay" in section and "`map_locked` (409)" in section
    assert "art/overlay.png" in text  # the workflow example's files
    assert "### `slow` — `slow.sh`" in text and "`POST /api/ops/slow`" in text
    front = text.split("---\n", 2)[1]
    assert "slow.sh " in front and ": slow → " in front


SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


@pytest.mark.parametrize("shell", SHELLS)
def test_address_snippet_is_posix_and_finds_the_service_in_order(tmp_path: Path,
                                                                  shell: str) -> None:
    """User URL → cached URL while it answers → server.json of the workspace (0.0.0.0 →
    127.0.0.1) → ask the user; the URL found is cached, and nothing is scanned."""
    code = snippet()
    assert subprocess.run([shell, "-n"], input=code, text=True).returncode == 0
    home, data = tmp_path / "home", tmp_path / "home" / "ws"
    cache = tmp_path / "cache" / "oh-my-slam-api" / "server_url"
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "XDG_CACHE_HOME": str(tmp_path / "cache")}

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([shell], input=code, env={**env, **extra}, capture_output=True,
                              text=True, timeout=60)

    none = run()
    assert none.returncode == 1 and none.stdout == "" and not cache.exists()
    assert "Ask the user" in none.stderr and "server.sh --status" in none.stderr
    proc, url = start(data)
    try:
        base = url.rstrip("/")
        port = base.rsplit(":", 1)[1]
        assert '"url":"http://0.0.0.0:' in (data / "server.json").read_text()
        found = run(OMS_DATA=str(data))  # 3. the workspace's server.json
        assert (found.returncode, found.stdout) == (0, base + "\n"), found.stderr
        assert cache.read_text() == base + "\n"
        assert run().stdout == base + "\n"  # 2. the cache, while it answers
        cache.write_text("http://127.0.0.1:9\n")  # a stale cache: searched again
        assert run(OMS_DATA="~/ws").stdout == base + "\n"
        assert cache.read_text() == base + "\n"
        # 1. the user's URL wins (its path dropped, 0.0.0.0 is this machine), and must answer
        assert run(OMS_URL=f" http://0.0.0.0:{port}/#/maps ").stdout == base + "\n"
        assert run(OMS_URL=f"localhost:{port}").stdout == f"http://localhost:{port}\n"
        dead = run(OMS_URL="http://127.0.0.1:9")
        assert dead.returncode == 1 and "check the URL with the user" in dead.stderr
        assert run(OMS_URL=f"http://127.0.0.1:{port}$(touch {tmp_path}/x)").returncode == 1
        assert not (tmp_path / "x").exists()
    finally:
        server("--data", str(data), "--stop")
        proc.wait(60)
    gone = run(OMS_DATA=str(data))  # the cached URL stopped answering and server.json is gone
    assert gone.returncode == 1 and "Ask the user" in gone.stderr
