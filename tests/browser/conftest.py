"""Shared browser-test fixtures: a headless Chromium-based browser (Edge, else Chrome) and a page
showing one viewer bundle."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from oh_my_slam.viewer.bundle import ViewBundle

CHANNELS = ("msedge", "chrome")


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = None
        for ch in CHANNELS:
            try:
                b = p.chromium.launch(channel=ch, headless=True)
                break
            except Exception:
                continue
        if b is None:
            pytest.skip("no Chromium-based browser (Edge/Chrome) available")
        yield b
        b.close()


class View:
    """A page showing one bundle, with the console errors it logged."""

    def __init__(self, browser: Any, bundle: ViewBundle, url: str) -> None:
        self.bundle, self.url, self.errors = bundle, url, []
        self.pg = browser.new_page(viewport={"width": 1280, "height": 800})
        self.pg.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.goto(url)
        self.pg.wait_for_selector('body[data-rendered="true"]', timeout=120000)

    def js(self, expr: str, arg: Any = None) -> Any:
        return self.pg.evaluate(expr, arg)

    def settle(self) -> None:
        """Wait for any pending re-derivation and two drawn frames."""
        self.pg.wait_for_function("() => !window.__viewer.busy", timeout=60000)
        self.js("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
