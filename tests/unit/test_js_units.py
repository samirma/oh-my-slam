"""The browser modules' unit tests (spec §4: every component is unit-tested offline, and the build
fails below 100 % coverage). The first-party JavaScript of ``viewer/static`` (view.sh's page) and
``web/static`` (server.sh's web application) is tested with ``bun test`` (``tests/js``, bun's own
test runner, no package installed), offline: fetch, workers, the location and the upload request
are test doubles, and three.js is the vendored copy the pages load.

Each first-party module is classified below:

* ``UNIT``: unit-tested. With no name listed, every line and every function of the module must run
  (100 % of bun's line and function coverage). With names, these top-level functions and classes
  are DOM- or WebGL-bound (they build the page's elements, or a WebGL renderer) and are covered by
  the ``-m browser`` tests; every other line of the module, its pure logic, must run. bun's lcov
  report names no functions, so in such a module only its lines are checked.
* ``BROWSER``: covered by the ``-m browser`` tests only (a page script, or DOM through and
  through).

bun measures lines and functions; it has no branch coverage, so branches are checked through the
lines they run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESTS_JS = REPO / "tests" / "js"
VIEWER = "src/oh_my_slam/viewer/static"
WEB = "src/oh_my_slam/web/static"

UNIT: dict[str, tuple[str, ...]] = {
    f"{VIEWER}/lib/cloud.js": (),
    f"{VIEWER}/lib/data.js": (),
    f"{VIEWER}/lib/obbs.js": (),
    f"{VIEWER}/lib/ply.js": (),
    f"{VIEWER}/lib/plyworker.js": (),
    f"{VIEWER}/lib/cameras.js": ("fillCameraTable",),
    f"{VIEWER}/lib/cloudview.js": ("CloudView",),
    f"{VIEWER}/lib/controls.js": ("buildAttributeControls",),
    f"{VIEWER}/lib/labels.js": ("LabelLayer",),
    f"{VIEWER}/lib/layers.js": ("buildLayerControls",),
    f"{VIEWER}/lib/viewer.js": ("Viewer",),
    f"{WEB}/js/api.js": (),
    f"{WEB}/js/store.js": (),
    f"{WEB}/js/url.js": (),
    f"{WEB}/js/cloudresult.js": ("cloudSection",),
    f"{WEB}/js/dom.js": ("el", "clear", "notice", "facts"),
    f"{WEB}/js/form.js": ("Field", "EnumField", "NumberField", "FlagField", "TextField",
                          "AttrsField", "MapField", "FilesField", "makeField", "OpForm"),
    f"{WEB}/js/pages/image.js": ("imagePage",),
    f"{WEB}/js/pages/maps.js": ("card", "mapsPage", "history", "mapPage"),
    f"{WEB}/js/request.js": ("onBeforeUnload", "RequestView"),
    f"{WEB}/js/result.js": ("objectsTable", "stagesTable", "fullText", "resultView"),
    f"{WEB}/js/runpanel.js": ("runPanel",),
}
BROWSER: dict[str, str] = {
    f"{VIEWER}/app.js": "view.sh's page script: it runs against the page as it loads",
    f"{VIEWER}/lib/dom.js": "creates the page's elements",
    f"{WEB}/js/app.js": "the web application's page script: routing, top bar, leaving a page",
    f"{WEB}/js/dialog.js": "the modal confirmation (a <dialog> element)",
    f"{WEB}/js/pages/mapflow.js": "builds the map flow's page",
}


@dataclass
class Coverage:
    lines: dict[int, int]  # line -> hits
    functions: int
    functions_hit: int


@dataclass
class Run:
    returncode: int
    output: str
    coverage: dict[str, Coverage]


def first_party_js() -> set[str]:
    out = set()
    for folder in (VIEWER, WEB):
        for p in (REPO / folder).rglob("*.js"):
            if "vendor" not in p.relative_to(REPO / folder).parts:
                out.add(p.relative_to(REPO).as_posix())
    return out


def bun() -> str:
    found = shutil.which("bun") or next(
        (p for p in ("/opt/homebrew/bin/bun", "/usr/local/bin/bun", str(Path.home() / ".bun/bin/bun"))
         if os.access(p, os.X_OK)), None)
    if found is None:
        pytest.fail("bun is not installed: the browser modules' unit tests (tests/js) run with "
                    "`bun test`. Install it with `brew install bun`, then run the tests again.")
    return found


def parse_lcov(path: Path, base: Path) -> dict[str, Coverage]:
    """Per module (path relative to the repository): its lines' hits and its functions."""
    out = {}
    for record in path.read_text().split("end_of_record"):
        name, lines, fnf, fnh = None, {}, 0, 0
        for row in record.splitlines():
            key, _, value = row.partition(":")
            if key == "SF":
                name = (base / value).resolve().relative_to(REPO).as_posix()
            elif key == "DA":
                line, hits = value.split(",")[:2]
                lines[int(line)] = int(hits)
            elif key == "FNF":
                fnf = int(value)
            elif key == "FNH":
                fnh = int(value)
        if name:
            out[name] = Coverage(lines, fnf, fnh)
    return out


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """``bun test`` with coverage, once for this file (about a second)."""
    out = tmp_path_factory.mktemp("bun-coverage")
    env = {**os.environ, "TZ": "UTC", "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8",
           "NO_COLOR": "1", "DO_NOT_TRACK": "1"}
    done = subprocess.run([bun(), "test", "--coverage", "--coverage-reporter=lcov",
                           "--coverage-reporter=text", f"--coverage-dir={out}"],
                          cwd=TESTS_JS, env=env, capture_output=True, text=True, timeout=120)
    lcov = out / "lcov.info"
    return Run(done.returncode, done.stdout + done.stderr,
               parse_lcov(lcov, TESTS_JS) if lcov.is_file() else {})


def span(lines: list[str], name: str) -> range:
    """The lines of the top-level function or class ``name``: from its declaration to the closing
    brace at the start of a line (the modules' style), or its one line."""
    decl = re.compile(rf"^(?:export\s+)?(?:async\s+)?(?:function\*?|class)\s+{re.escape(name)}\b")
    start = next((i for i, text in enumerate(lines, 1) if decl.match(text)), None)
    if start is None:
        pytest.fail(f"no top-level function or class {name}: update UNIT in {Path(__file__).name}")
    first = lines[start - 1].rstrip()
    if first.endswith("}") and first.count("{") == first.count("}"):
        return range(start, start + 1)
    end = next(i for i in range(start + 1, len(lines) + 1) if lines[i - 1].startswith("}"))
    return range(start, end + 1)


def test_every_module_is_classified() -> None:
    assert not set(UNIT) & set(BROWSER)
    assert set(UNIT) | set(BROWSER) == first_party_js()


def test_the_js_unit_tests_pass(run: Run) -> None:
    assert run.returncode == 0, f"bun test failed:\n{run.output[-6000:]}"
    assert re.search(r"^\s*0 fail$", run.output, re.MULTILINE), run.output[-6000:]


@pytest.mark.parametrize("module", sorted(UNIT))
def test_coverage(run: Run, module: str) -> None:
    assert module in run.coverage, f"{module} is not imported by tests/js:\n{run.output[-3000:]}"
    cov = run.coverage[module]
    missed = sorted(n for n, hits in cov.lines.items() if hits == 0)
    source = (REPO / module).read_text().splitlines()
    dom_bound = {n for name in UNIT[module] for n in span(source, name)}
    outside = [n for n in missed if n not in dom_bound]
    assert not outside, f"{module}: lines no unit test runs:\n" + "\n".join(
        f"{n:5d}  {source[n - 1]}" for n in outside)
    if not UNIT[module]:
        assert cov.functions_hit == cov.functions, (
            f"{module}: {cov.functions - cov.functions_hit} of {cov.functions} functions never run")
