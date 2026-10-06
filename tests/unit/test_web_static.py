"""The web application's files (http_server.md "Web application", "Self-contained"): served by the
service itself, nothing loaded from elsewhere, built on the public API only, with no command or
option named in its code. It has no map or scene viewer; a point-cloud result is drawn by the §2.5
viewer's own modules, which the service serves from the viewer's package (``lib/``, ``vendor/``),
not copied: the app itself draws nothing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from oh_my_slam.commands import spec
from oh_my_slam.web.app import VIEWER_STATIC, WEB_STATIC, Service, create_app
from oh_my_slam.web.runner import Runner
from oh_my_slam.web.workspace import Workspace

JS = sorted(WEB_STATIC.rglob("*.js"))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    ws = Workspace(tmp_path / "data")
    ws.create()
    return TestClient(create_app(Service(ws, Runner(ws), url="http://0.0.0.0:0/",
                                         extra_hosts={"testserver"})))


def test_the_app_and_its_files_are_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'src="/static/js/app.js"' in r.text
    for path in ["/static/js/app.js", "/static/style.css", "/static/js/pages/maps.js"]:
        r = client.get(path)
        assert r.status_code == 200, path
    assert client.get("/static/js/app.js").headers["content-type"].startswith("text/javascript")
    for bad in ["/static/../app.py", "/static/nope.js", "/static/%2e%2e/app.py",
                # no scene viewer: neither the scene schema it read nor view.sh's own page
                "/static/openlabel_json_schema.json", "/static/viewer/index.html",
                "/static/viewer/app.js", "/static/viewer/style.css", "/static/viewer/lib/../app.js",
                "/static/viewer/lib/%2e%2e/app.js", "/static/viewer/nope/viewer.js"]:
        assert client.get(bad).status_code == 404, bad


def test_the_viewers_modules_are_served_from_its_package(client: TestClient) -> None:
    """The point-cloud drawing is the §2.5 viewer's own: its modules and vendored three.js are the
    viewer package's files, byte for byte, and no copy of them is in the app's folder."""
    for rel in ["lib/cloudview.js", "lib/viewer.js", "lib/cloud.js", "lib/ply.js", "lib/plyworker.js",
                "lib/controls.js", "vendor/three/build/three.module.js",
                "vendor/three/addons/controls/OrbitControls.js"]:
        r = client.get(f"/static/viewer/{rel}")
        assert r.status_code == 200, rel
        assert r.headers["content-type"].startswith("text/javascript"), rel
        assert r.content == (VIEWER_STATIC / rel).read_bytes(), rel
    viewer = {f.read_bytes() for f in VIEWER_STATIC.rglob("*") if f.is_file()}
    assert not [f.name for f in WEB_STATIC.rglob("*") if f.is_file() and f.read_bytes() in viewer]
    # the app imports nothing of three.js itself: it embeds the viewer's cloud view
    for f in JS:
        assert not re.search(r"""from\s+['"]three""", f.read_text()), f.name


def test_nothing_is_loaded_from_elsewhere() -> None:
    """Self-contained: no URL to another host in a script, style or page of the app, and every
    module it imports is its own."""
    for f in [*JS, *WEB_STATIC.rglob("*.css"), *WEB_STATIC.rglob("*.html")]:
        text = f.read_text()
        for url in re.findall(r"""(?:src|href|import|url)\s*\(?\s*['"](https?:)?//[^'"]+""", text):
            pytest.fail(f"{f.name} loads {url}")
        for m in re.findall(r"""from\s+['"]([^'"]+)['"]""", text):
            assert m.startswith(("./", "../")), (f.name, m)
            assert (f.parent / m).resolve().is_file(), (f.name, m)
        # loaded on demand: only the viewer's own modules, served by this service
        for m in re.findall(r"""import\(\s*['"`]?([^'"`)]+)""", text):
            if not m.startswith("/static/viewer/"):
                pytest.fail(f"{f.name} imports {m}")
            assert (VIEWER_STATIC / m.removeprefix("/static/viewer/")).is_file(), (f.name, m)


def test_the_app_uses_the_public_api_and_names_no_command() -> None:
    """Forms and results come from the API description: no command, mode or option of the registry
    is named in the app's code, and it requests only the API and its own files."""
    names: set[str] = set()
    for prog in spec.PROGRAMS:
        names.add(prog.prog)
        for c in prog.commands:
            for o in c.options:
                names.add(o.flag)
    code = "\n".join(re.sub(r"//.*", "", f.read_text()) for f in JS)
    for n in names:
        assert not re.search(rf"""['"`]{re.escape(n)}['"`]""", code), n
    for url in re.findall(r"""['"`](/[a-z][^'"`$]*)""", code):
        assert url.startswith(("/api/", "/static/")), url
    routes = set(re.findall(r"""['"`]/api/([a-z]+)""", code))
    assert routes <= {"health", "openapi", "ops", "uploads", "maps"}, routes
