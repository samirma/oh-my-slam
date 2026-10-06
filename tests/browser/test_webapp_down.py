"""The web application with the inference server down (``-m browser``; http_server.md "Actionable
errors"): the top bar says so with the command that starts it; the actions that need it are
disabled with the explanation; everything else stays available — a map is segmented without it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from tests.browser.webapp import Tab, operation_response, running_service
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


def test_the_rest_stays_available(tab: Tab) -> None:
    pg = tab.go("#/maps")
    pg.wait_for_selector("li.map-card[data-map=m]")
    pg = tab.go("#/maps/m?op=segment-map")
    pg.wait_for_function("() => document.body.dataset.inference === 'down'")
    assert pg.locator("button[data-action=run]").is_enabled()
    assert pg.locator(".run-panel .notice.warn").count() == 0
    with operation_response(pg, "segment-map") as answer:
        pg.click("button[data-action=run]")
    tab.wait_request()
    assert json.loads(answer.body)["openlabel"]
    # mapper.sh locate needs the server only for a large map: the service's own check decides
    pg.check("#op-mapper-locate")
    assert pg.locator("button[data-action=run]").is_enabled()
    tab.a11y()
    assert tab.errors == []
