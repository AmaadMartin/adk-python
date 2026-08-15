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

"""Tests for the calling convention of ApiServer.get_fast_api_app.

DevServer overrides the method with a keyword-forwarding signature, so the two
classes only stay substitutable while every parameter is keyword-only.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import inspect
import json
from pathlib import Path
from typing import AsyncIterator
from unittest import mock

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
from google.adk.cli.adk_web_server import AdkWebServer
from google.adk.cli.api_server import ApiServer
from google.adk.cli.dev_server import DevServer
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService
import pytest


def _create_server(server_cls: type[ApiServer], agents_dir: Path) -> ApiServer:
  """Builds a real server of the given class over stubbed-out services."""
  return server_cls(
      agent_loader=mock.create_autospec(BaseAgentLoader, instance=True),
      session_service=mock.create_autospec(BaseSessionService, instance=True),
      memory_service=mock.create_autospec(BaseMemoryService, instance=True),
      artifact_service=mock.create_autospec(BaseArtifactService, instance=True),
      credential_service=mock.create_autospec(
          BaseCredentialService, instance=True
      ),
      eval_sets_manager=mock.create_autospec(EvalSetsManager, instance=True),
      eval_set_results_manager=mock.create_autospec(
          EvalSetResultsManager, instance=True
      ),
      agents_dir=str(agents_dir),
  )


@pytest.mark.parametrize("server_cls", [ApiServer, DevServer, AdkWebServer])
def test_positional_argument_is_rejected(server_cls, tmp_path):
  """Every class in the hierarchy rejects a positional call identically."""
  server = _create_server(server_cls, tmp_path)

  with pytest.raises(TypeError, match="positional"):
    server.get_fast_api_app(None)


@pytest.mark.parametrize("server_cls", [ApiServer, DevServer])
def test_keyword_call_builds_an_app(server_cls, tmp_path):
  server = _create_server(server_cls, tmp_path)

  app = server.get_fast_api_app(
      allow_origins=["*"],
      setup_observer=lambda _observer, _server: None,
      tear_down_observer=lambda _observer, _server: None,
  )

  assert isinstance(app, FastAPI)


def test_dev_server_forwards_every_api_server_parameter(tmp_path):
  """Guards against a parameter DevServer's **kwargs override cannot forward."""
  web_assets_dir = tmp_path / "assets_root"
  web_assets_dir.mkdir()
  server = _create_server(DevServer, tmp_path)
  calls: list[str] = []

  @asynccontextmanager
  async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    calls.append("lifespan")
    yield

  app = server.get_fast_api_app(
      lifespan=lifespan,
      allow_origins=["https://example.com"],
      web_assets_dir=str(web_assets_dir),
      setup_observer=lambda _observer, _server: calls.append("setup"),
      tear_down_observer=lambda _observer, _server: calls.append("tear_down"),
      register_processors=lambda _provider: calls.append("register"),
      otel_to_cloud=False,
      with_ui=False,
  )

  assert isinstance(app, FastAPI)
  # The dev endpoints only exist when DevServer ran its own registration.
  assert any(
      isinstance(route, APIRoute) and route.path.startswith("/dev/")
      for route in app.routes
  )
  # web_assets_dir reached the parent, which writes the UI runtime config.
  runtime_config = web_assets_dir / "assets" / "config" / "runtime-config.json"
  assert json.loads(runtime_config.read_text())["backendUrl"] == ""

  with TestClient(app) as client:
    assert client.get("/health").status_code == 200

  assert calls == ["setup", "register", "lifespan", "tear_down"]


def test_every_parameter_is_keyword_only():
  parameters = inspect.signature(ApiServer.get_fast_api_app).parameters

  positional = [
      name
      for name, parameter in parameters.items()
      if name != "self" and parameter.kind is not parameter.KEYWORD_ONLY
  ]
  assert not positional
