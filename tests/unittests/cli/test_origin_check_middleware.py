# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Host/Origin validation in _OriginCheckMiddleware."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.cli.api_server import _build_allowed_hosts
from google.adk.cli.api_server import _OriginCheckMiddleware
from google.adk.cli.fast_api import get_fast_api_app
import pytest
from starlette.websockets import WebSocketDisconnect

_ALLOWED_HOSTS = _build_allowed_hosts("127.0.0.1", 8000)
_BASE_URL = "http://127.0.0.1:8000"


class _RecordingApp:
  """Minimal ASGI app that records the scopes it was called with."""

  def __init__(self) -> None:
    self.scopes: list[dict[str, Any]] = []

  async def __call__(self, scope, receive, send) -> None:
    self.scopes.append(scope)
    if scope["type"] == "websocket":
      await send({"type": "websocket.accept"})
      return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _make_scope(
    scope_type: str = "http",
    method: str = "POST",
    host: str | None = "127.0.0.1:8000",
    origin: str | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
  """Build a minimal ASGI scope."""
  headers: list[tuple[bytes, bytes]] = []
  if host is not None:
    headers.append((b"host", host.encode()))
  if origin is not None:
    headers.append((b"origin", origin.encode()))
  headers.extend(extra_headers or [])
  return {
      "type": scope_type,
      "method": method,
      "server": ("127.0.0.1", 8000),
      "headers": headers,
  }


async def _call(
    middleware: _OriginCheckMiddleware, scope: dict[str, Any]
) -> list[dict[str, Any]]:
  """Run the middleware over a scope and return the messages it sent."""
  sent: list[dict[str, Any]] = []

  async def receive() -> dict[str, Any]:
    return {"type": "websocket.connect"}

  async def send(message: dict[str, Any]) -> None:
    sent.append(message)

  await middleware(scope, receive, send)
  return sent


def _make_middleware(
    inner_app: _RecordingApp,
    allowed_hosts: frozenset[str] | None = _ALLOWED_HOSTS,
) -> _OriginCheckMiddleware:
  return _OriginCheckMiddleware(
      inner_app,
      has_configured_allowed_origins=False,
      allowed_origins=[],
      allowed_origin_regex=None,
      allowed_hosts=allowed_hosts,
  )


def _assert_forbidden(sent: list[dict[str, Any]], body: bytes) -> None:
  assert sent[0]["type"] == "http.response.start"
  assert sent[0]["status"] == 403
  assert sent[1]["body"] == body


class TestHostValidation:
  """The Host header is validated on every method and every scope type."""

  async def test_disallowed_host_rejected(self):
    """A safe method is no longer exempt: the Host is checked first."""
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(method="GET", host="evil.example:8000"),
    )

    _assert_forbidden(sent, b"Forbidden: host not allowed")
    assert not inner_app.scopes

  async def test_missing_host_rejected(self):
    inner_app = _RecordingApp()
    sent = await _call(_make_middleware(inner_app), _make_scope(host=None))

    _assert_forbidden(sent, b"Forbidden: host not allowed")
    assert not inner_app.scopes

  async def test_allowed_host_forwarded(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(method="GET", host="LOCALHOST:8000"),
    )

    assert sent[0]["status"] == 200
    assert len(inner_app.scopes) == 1

  async def test_no_host_check_without_declared_bind_address(self):
    """The library-embedding path does not declare a bind address."""
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app, allowed_hosts=None),
        _make_scope(method="GET", host="evil.example:8000"),
    )

    assert sent[0]["status"] == 200
    assert len(inner_app.scopes) == 1


class TestOriginValidation:
  """The Origin header is validated on state-changing HTTP methods."""

  async def test_cross_origin_post_rejected(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(origin="http://evil.example:8000"),
    )

    _assert_forbidden(sent, b"Forbidden: origin not allowed")
    assert not inner_app.scopes

  async def test_same_origin_post_forwarded(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app), _make_scope(origin=_BASE_URL)
    )

    assert sent[0]["status"] == 200
    assert len(inner_app.scopes) == 1

  async def test_post_without_origin_forwarded(self):
    """Non-browser clients (curl, SDKs, health checks) send no Origin."""
    inner_app = _RecordingApp()
    sent = await _call(_make_middleware(inner_app), _make_scope())

    assert sent[0]["status"] == 200
    assert len(inner_app.scopes) == 1

  async def test_preflight_with_foreign_origin_reaches_cors_middleware(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(method="OPTIONS", origin="http://evil.example:8000"),
    )

    assert sent[0]["status"] == 200
    assert len(inner_app.scopes) == 1

  async def test_forwarded_host_spoof_rejected(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(
            host="localhost:8000",
            origin="http://evil.example",
            extra_headers=[(b"x-forwarded-host", b"evil.example")],
        ),
    )

    _assert_forbidden(sent, b"Forbidden: origin not allowed")
    assert not inner_app.scopes


class TestRejectionLogging:
  """A rejection logs the offending value and no other request data."""

  async def test_single_warning_naming_the_offending_value(
      self, caplog: pytest.LogCaptureFixture
  ):
    with caplog.at_level(logging.WARNING):
      await _call(
          _make_middleware(_RecordingApp()),
          _make_scope(host="evil.example:8000"),
      )

    assert len(caplog.messages) == 1
    assert "evil.example:8000" in caplog.messages[0]


class TestNonHttpScopes:
  """WebSocket upgrades are checked; other scope types pass through."""

  @pytest.mark.parametrize(
      "host,origin",
      [
          ("evil.example:8000", None),
          ("127.0.0.1:8000", "http://evil.example:8000"),
      ],
      ids=["disallowed_host", "disallowed_origin"],
  )
  async def test_websocket_rejected(self, host: str, origin: str | None):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(scope_type="websocket", host=host, origin=origin),
    )

    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert not inner_app.scopes

  async def test_websocket_forwarded(self):
    inner_app = _RecordingApp()
    sent = await _call(
        _make_middleware(inner_app),
        _make_scope(scope_type="websocket", origin=_BASE_URL),
    )

    assert sent == [{"type": "websocket.accept"}]
    assert len(inner_app.scopes) == 1

  async def test_lifespan_passes_through(self):
    inner_app = _RecordingApp()
    scope = {"type": "lifespan", "headers": []}
    await _call(_make_middleware(inner_app), scope)

    assert inner_app.scopes == [scope]


def _build_app(tmp_path, host: str = "127.0.0.1", **kwargs) -> FastAPI:
  """Build a real app over an empty agents directory."""
  return get_fast_api_app(
      agents_dir=str(tmp_path),
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      a2a=False,
      host=host,
      port=8000,
      **kwargs,
  )


class TestWildcardBindWarning:
  """A wildcard bind disables Host validation, so it has to say so."""

  @pytest.mark.parametrize(
      "app_kwargs,expected",
      [
          (dict(host="0.0.0.0"), True),
          (dict(host="0.0.0.0", allow_origins=["https://ok.test"]), False),
      ],
      ids=["wildcard", "wildcard_with_allow_origins"],
  )
  def test_warning_emitted_only_for_an_undeclarable_bind(
      self,
      tmp_path,
      caplog: pytest.LogCaptureFixture,
      app_kwargs: dict[str, Any],
      expected: bool,
  ):
    with caplog.at_level(logging.WARNING):
      _build_app(tmp_path, **app_kwargs)

    assert (
        any("Host header validation is disabled" in m for m in caplog.messages)
        is expected
    )


class TestServedRequests:
  """End-to-end checks over the real ASGI stack, no mocks."""

  def test_rebound_host_cannot_read_dev_server_state(self, tmp_path):
    client = TestClient(
        _build_app(tmp_path), base_url="http://evil.example:8000"
    )

    response = client.get("/list-apps")

    assert response.status_code == 403
    assert response.text == "Forbidden: host not allowed"

  def test_expected_host_is_served(self, tmp_path):
    client = TestClient(_build_app(tmp_path), base_url=_BASE_URL)

    response = client.get("/list-apps")

    assert response.status_code == 200
    assert response.json() == []

  def test_allow_origins_is_the_documented_escape_hatch(self, tmp_path):
    client = TestClient(
        _build_app(tmp_path, allow_origins=["http://tunnel.test:8000"]),
        base_url="http://tunnel.test:8000",
    )

    response = client.get("/list-apps")

    assert response.status_code == 200


# TestClient.websocket_connect ignores base_url, so the Host under test has to
# be spelled out in the URL.
_RUN_LIVE_PATH = "/run_live?app_name=test_app&user_id=user&session_id=session"


class TestRunLiveWebSocket:
  """The /run_live upgrade is covered by the same Host allowlist."""

  def test_disallowed_origin_is_closed(self, tmp_path):
    client = TestClient(_build_app(tmp_path))

    with pytest.raises(WebSocketDisconnect) as exc_info:
      with client.websocket_connect(
          f"ws://127.0.0.1:8000{_RUN_LIVE_PATH}",
          headers={"origin": "http://evil.example"},
      ):
        pass

    assert exc_info.value.code == 1008
