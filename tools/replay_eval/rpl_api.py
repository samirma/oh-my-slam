"""A client of the oh-my-slam HTTP API (specs/api.md) that times, validates and records every call."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from rpl_validate import Validator

Part = tuple[str, str, bytes, str]  # (part name, file name, content, media type)


class ApiError(Exception):
    """An error answer of the API (specs/api.md §9), or a connection failure (status 0)."""

    def __init__(self, status: int, code: str, message: str, retryable: bool,
                 body: Any = None) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code, self.message, self.retryable = status, code, message, retryable
        self.body = body


@dataclass
class Call:
    route: str
    status: int
    seconds: float
    server_timing: bool
    schema: str | None
    schema_errors: list[str]


@dataclass
class Log:
    """The calls of one replay, shared by the threads that make them. With ``record_dir``, every
    request is also written as a request set (``requests.jsonl`` and its file parts under
    ``files/``) that ``tools/evaluate/evaluate.py --requests`` replays."""

    calls: list[Call] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    record_dir: Path | None = None
    recorded: int = 0

    def add(self, call: Call) -> None:
        with self.lock:
            self.calls.append(call)

    def record(self, method: str, path: str, route: str, schema: str | None,
               params: dict[str, Any] | None, body: Any, parts: list[Part] | None,
               ok: tuple[int, ...]) -> None:
        if self.record_dir is None:
            return
        with self.lock:
            n = self.recorded
            self.recorded += 1
            files = self.record_dir / "files"
            files.mkdir(parents=True, exist_ok=True)
            saved = []
            for name, file, data, media in parts or []:
                rel = f"files/{n:05d}_{name}_{Path(file).name}"
                (self.record_dir / rel).write_bytes(data)
                saved.append({"name": name, "file": file, "path": rel, "media": media})
            line = {"method": method, "path": path, "route": route, "schema": schema,
                    "params": params, "body": body, "parts": saved if parts is not None
                    else None, "ok": list(ok)}
            with (self.record_dir / "requests.jsonl").open("a") as fh:
                fh.write(json.dumps(line) + "\n")


class Api:
    def __init__(self, base_url: str, validator: Validator, log: Log | None = None,
                 timeout_s: float = 300.0, retries: int = 3, retry_wait_s: float = 2.0) -> None:
        self.base = base_url.rstrip("/")
        self.validator = validator
        self.log = log if log is not None else Log()
        self.timeout_s, self.retries, self.retry_wait_s = timeout_s, retries, retry_wait_s
        self.session = requests.Session()
        self.session.trust_env = False  # a LAN service: never through a proxy

    def clone(self) -> Api:
        """Another client (own session, for another thread) writing to the same log."""
        return Api(self.base, self.validator, self.log, self.timeout_s, self.retries,
                   self.retry_wait_s)

    def call(self, method: str, path: str, route: str, schema: str | None = None, *,
             params: dict[str, Any] | None = None, body: Any = None,
             parts: list[Part] | None = None, ok: tuple[int, ...] = (200,)) -> tuple[Any, float]:
        """(decoded body, seconds) of one call; JSON bodies are validated against ``schema``.
        Retryable errors are retried ``retries`` times. Raises ``ApiError``."""
        self.log.record(method, path, route, schema, params, body, parts, ok)
        attempt = 0
        while True:
            try:
                return self._once(method, path, route, schema, params, body, parts, ok)
            except ApiError as e:
                if not e.retryable or attempt >= self.retries:
                    raise
                attempt += 1
                time.sleep(self.retry_wait_s)

    def _once(self, method: str, path: str, route: str, schema: str | None,
              params: dict[str, Any] | None, body: Any, parts: list[Part] | None,
              ok: tuple[int, ...]) -> tuple[Any, float]:
        kwargs: dict[str, Any] = {"params": params, "timeout": self.timeout_s}
        if parts is not None:
            kwargs["files"] = [("request", (None, json.dumps(body), "application/json"))] + [
                (name, (file, data, media)) for name, file, data, media in parts]
        elif body is not None:
            kwargs["json"] = body
        t0 = time.monotonic()
        try:
            r = self.session.request(method, self.base + path, **kwargs)
        except requests.RequestException as e:
            raise ApiError(0, "connection", str(e), True) from e
        seconds = time.monotonic() - t0
        is_json = r.headers.get("Content-Type", "").startswith("application/json")
        decoded: Any = r.json() if is_json and r.content else r.content
        name = schema if r.status_code in ok else "ErrorBody"
        errors: list[str] = []
        if is_json and name is not None:
            errors = self.validator.errors(name, decoded)
        self.log.add(Call(route, r.status_code, seconds, "Server-Timing" in r.headers,
                          name if is_json else None, errors))
        if r.status_code in ok:
            return decoded, seconds
        err = decoded.get("error", {}) if isinstance(decoded, dict) else {}
        raise ApiError(r.status_code, str(err.get("code", "unknown")),
                       str(err.get("message", r.text[:200])), bool(err.get("retryable", False)),
                       decoded)
