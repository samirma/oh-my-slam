"""The web application's files (http_server.md "Web application", "Self-contained"): served by the
service itself, nothing loaded from elsewhere, built on the public API only, with no command or
option named in its code; and the constants it shares with the Python side agree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from oh_my_slam.commands import spec
from oh_my_slam.schema import openlabel
from oh_my_slam.viewer import bundle
from oh_my_slam.web.app import VIEWER_STATIC, WEB_STATIC, Service, create_app
from oh_my_slam.web.jobs import Runner
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
    for path in ["/static/js/app.js", "/static/style.css", "/static/viewer/lib/viewer.js",
                 "/static/viewer/lib/labels.css",
                 "/static/viewer/vendor/three/build/three.module.js"]:
        r = client.get(path)
        assert r.status_code == 200, path
    assert client.get("/static/js/app.js").headers["content-type"].startswith("text/javascript")
    schema = client.get("/static/openlabel_json_schema.json")
    assert schema.status_code == 200 and schema.json()["$schema"].startswith("http://json-schema.org/draft-07")
    for bad in ["/static/../app.py", "/static/viewer/../routes.py", "/static/nope.js",
                "/static/%2e%2e/app.py"]:
        assert client.get(bad).status_code == 404, bad


def test_nothing_is_loaded_from_elsewhere() -> None:
    """Self-contained: no URL to another host in a script, style or page of the app."""
    for f in [*JS, *WEB_STATIC.rglob("*.css"), *WEB_STATIC.rglob("*.html")]:
        text = f.read_text()
        for url in re.findall(r"""(?:src|href|import|url)\s*\(?\s*['"](https?:)?//[^'"]+""", text):
            pytest.fail(f"{f.name} loads {url}")
        for m in re.findall(r"""from\s+['"]([^'"]+)['"]""", text):
            assert m.startswith(("./", "../", "/static/", "three")), (f.name, m)


def test_the_app_uses_the_public_api_and_names_no_command() -> None:
    """Forms, results and downloads come from the API description: no command, mode or option of
    the registry is named in the app's code."""
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
        assert url.startswith(("/api/", "/static/", "/viewer/")), url


def test_constants_shared_with_python_agree() -> None:
    controls = (VIEWER_STATIC / "lib" / "controls.js").read_text()
    m = re.search(r"DISPLAY_POINT_BUDGET = ([\d_]+);", controls)
    assert m and int(m[1].replace("_", "")) == bundle.DISPLAY_POINT_BUDGET
    ol = (WEB_STATIC / "js" / "scene" / "openlabel.js").read_text()
    assert f"SCHEMA_VERSION = '{openlabel.SCHEMA_VERSION}'" in ol
    assert f"SCHEMA_URL = '{openlabel.SCHEMA_URL}'" in ol
    # the job page picks the command's timings line out of its stderr by its opening words
    from oh_my_slam.core.timing import summary_line

    line = summary_line({"stages_s": {"x": 1.0}, "total_s": 1.0, "server": {},
                         "peak_rss_mb": {"self": 1.0}})
    jobview = (WEB_STATIC / "js" / "jobview.js").read_text()
    assert "const TIMINGS_LINE = /\\btimings: total /;" in jobview
    assert line.startswith("timings: total ")
