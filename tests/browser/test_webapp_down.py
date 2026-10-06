"""The web application with the inference server down (``-m browser``; http_server.md "Actionable
errors"): the top bar says so with the command that starts it; the actions that need it are
disabled with the explanation; everything else stays available — images are located in a small
map without it, answered by the command itself."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.browser.webapp import Tab, operation_response, running_service
from tests.unit.test_view_cli import minimal_map, sh
from tests.unit.test_web_api import jpeg

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


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_actions_that_need_it_are_disabled_with_the_reason(browser: Any, down: tuple[Any, str],
                                                            scheme: str) -> None:
    tab = Tab(browser, down[1], scheme=scheme)
    try:
        pg = tab.go("#/image")
        pg.wait_for_function("() => document.body.dataset.inference === 'down'")
        for op in ("reconstruct", "segment-image"):
            pg.check(f"#op-{op}")
            btn = pg.locator("button[data-action=run]")
            assert btn.is_disabled()
            warn = pg.inner_text(".run-panel .notice.warn")
            assert "inference server is down" in warn and "./start_inference_server.sh" in warn
            assert btn.get_attribute("aria-describedby") == f"{op}-blocked"
        tab.a11y()
        pg = tab.go("#/maps/new")
        pg.wait_for_selector(".run-panel .notice.warn")
        assert pg.locator("button[data-action=run]").is_disabled()
        tab.a11y()
        assert tab.errors == []
    finally:
        tab.close()


def test_the_rest_stays_available(tab: Tab, tmp_path: Path) -> None:
    pg = tab.go("#/maps")
    pg.wait_for_selector("li.map-card[data-map=m]")
    # mapper.sh locate needs the server only for a large map: the service's own check decides
    pg = tab.go("#/maps/m?op=mapper-locate")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    assert pg.locator("button[data-action=run]").is_enabled()
    assert pg.locator(".run-panel .notice.warn").count() == 0
    pg.set_input_files(".field[data-param=inputs] input[type=file]", str(jpeg(tmp_path / "q.jpg")))
    pg.wait_for_selector(".field[data-param=inputs] li.ready")
    pg.wait_for_function("() => document.querySelector('[data-testid=command]').textContent.includes('-m=maps/m')")
    with operation_response(pg, "mapper-locate") as answer:
        pg.click("button[data-action=run]")
    tab.wait_request("failed")
    # it ran: the answer is the command's own (an empty map locates nothing), not the server's 503
    text = pg.inner_text("[data-testid=request] .notice.error")
    assert answer.status != 503 and "HTTP 503" not in text and "server_unavailable" not in text
    if shutil.which("colmap"):  # without COLMAP the command fails before matching
        assert answer.status == 400 and "none of the images could be located" in text
    tab.a11y()
    # its refusal is the only failed response
    assert [e for e in tab.errors
            if "/api/ops/mapper-locate" not in e and "Failed to load resource" not in e] == []
