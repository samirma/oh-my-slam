"""The web application with the inference server down (``-m browser``; http_server.md "Actionable
errors"): the top bar says so with the command that starts it; the actions that need it are
disabled with the explanation; everything else stays available."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.browser.test_viewer_modules import scene_doc
from tests.browser.test_webapp_browser import Tab
from tests.browser.webapp import running_service
from tests.unit.test_view_cli import minimal_map, sh

pytestmark = [pytest.mark.browser]


@pytest.fixture(scope="module")
def down(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, str]]:
    sh("start_inference_server.sh", "--stop")  # none runs in the tests' runtime folder
    with running_service(tmp_path_factory.mktemp("down") / "ws") as (service, url):
        minimal_map(service.workspace.maps / "m")
        yield service, url


@pytest.fixture
def tab(browser: Any, down: tuple[Any, str]) -> Iterator[Tab]:
    t = Tab(browser, down[1])
    yield t
    t.close()


def test_the_top_bar_says_how_to_start_it(tab: Tab) -> None:
    pg = tab.go("#/image")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    assert "down" in pg.inner_text("#inference")
    assert pg.inner_text("[data-testid=start-command]") == "./start_inference_server.sh"


def test_actions_that_need_it_are_disabled_with_the_reason(tab: Tab) -> None:
    pg = tab.go("#/image")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    for mode in ("reconstruct", "segment-image", "view-image"):
        pg.check(f"#mode-{mode}")
        btn = pg.locator("form.image-form button[type=submit]")
        assert btn.is_disabled()
        warn = pg.inner_text("form.image-form .notice.warn")
        assert "inference server is down" in warn and "./start_inference_server.sh" in warn
    tab.a11y()
    pg = tab.go("#/maps/new")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    pg.wait_for_selector("form.flow .notice.warn")
    assert pg.locator("form.flow button[type=submit]").is_disabled()
    tab.a11y()
    assert tab.errors == []


def test_the_rest_stays_available(tab: Tab, tmp_path: Path) -> None:
    pg = tab.go("#/maps/m")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    card = pg.locator("details.export[data-op=segment-map]")
    card.locator("summary").click()
    pg.wait_for_function("() => document.querySelector('details.export[data-op=segment-map] .command')?.textContent.includes('segment.sh')")
    assert card.locator("button[type=submit]").is_enabled()
    assert card.locator(".notice.warn").count() == 0
    # mapper.sh locate needs the server only for a large map: the service's own check decides
    assert pg.locator("details.export[data-op=mapper-locate] button[type=submit]").is_enabled()
    pg.wait_for_selector(".viewer-box[data-ready=true]", timeout=60000)  # view -m needs no server
    tab.a11y()
    pg = tab.go("#/scene")
    js = tmp_path / "scene.json"
    js.write_text(json.dumps(scene_doc()))
    pg.set_input_files("#scene-files", str(js))
    pg.wait_for_selector("[data-testid=scene-view][data-loaded=true]")
    pg = tab.go("#/jobs")
    pg.wait_for_selector("[data-testid=jobs]")
    assert tab.errors == []
