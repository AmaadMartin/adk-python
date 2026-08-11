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

"""Tests for the dev UI runtime config served by ApiServer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from unittest import mock

from fastapi.testclient import TestClient
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
from google.adk.cli import fast_api
from google.adk.cli.api_server import ApiServer
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService
import pytest

from .conftest import _PACKAGED_RUNTIME_CONFIG

# The path the packaged Angular bundle fetches at bootstrap, relative to the
# /dev-ui/ document root it is mounted on.
_URL = "/dev-ui/assets/config/runtime-config.json"


def _make_api_server(agents_dir: Path, **kwargs) -> ApiServer:
  """Builds an ApiServer with autospec'd service doubles."""
  return ApiServer(
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
      **kwargs,
  )


def _make_client(agents_dir: Path, **kwargs) -> TestClient:
  """Builds a dev UI server rooted in ``agents_dir`` and a client for it."""
  api_server = _make_api_server(agents_dir, **kwargs)
  return TestClient(api_server.get_fast_api_app(web_assets_dir=str(agents_dir)))


def _runtime_config_path(assets_dir: Path) -> Path:
  return assets_dir / "assets" / "config" / "runtime-config.json"


def _seed(assets_dir: Path, contents: str) -> Path:
  """Writes a starting runtime-config.json into a temp assets dir."""
  config_path = _runtime_config_path(assets_dir)
  config_path.parent.mkdir(parents=True, exist_ok=True)
  config_path.write_text(contents)
  return config_path


@pytest.fixture(autouse=True)
def no_telemetry_consent(monkeypatch):
  """Pins telemetry consent so results do not depend on ~/.adk/config.json."""
  monkeypatch.setattr(
      "google.adk.cli.api_server.read_telemetry_consent", lambda: None
  )


def test_endpoint_serves_backend_url_and_telemetry(tmp_path):
  _seed(tmp_path, '{\n  "backendUrl": ""\n}')

  response = _make_client(tmp_path).get(_URL)

  assert response.status_code == 200
  assert response.json() == {"backendUrl": "", "telemetry": None}


def test_packaged_config_is_never_written(tmp_path):
  seeded = '{\n  "backendUrl": "packaged"\n}\n'
  config_path = _seed(tmp_path, seeded)

  assert _make_client(tmp_path).get(_URL).status_code == 200

  assert config_path.read_bytes() == seeded.encode()


def test_no_directories_are_created_under_web_assets_dir(tmp_path, caplog):
  with caplog.at_level(logging.INFO, logger="google_adk"):
    response = _make_client(tmp_path).get(_URL)

  assert response.json() == {"backendUrl": "", "telemetry": None}
  assert not (tmp_path / "assets").exists()
  assert "Runtime config file not found" in caplog.text


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="A read-only directory does not stop the Windows or root user.",
)
def test_read_only_web_assets_dir_serves_the_config(tmp_path, caplog):
  config_path = _seed(tmp_path, '{"backendUrl": ""}')
  # The file mode blocks a rewrite in place; the directory modes block a
  # replacement and a mkdir. A read-only install denies all three.
  os.chmod(config_path, 0o444)
  read_only_dirs = [config_path.parent, config_path.parent.parent, tmp_path]
  for directory in read_only_dirs:
    os.chmod(directory, 0o555)

  try:
    with caplog.at_level(logging.ERROR, logger="google_adk"):
      response = _make_client(tmp_path, url_prefix="/proxy").get(_URL)
  finally:
    for directory in read_only_dirs:
      os.chmod(directory, 0o755)
    os.chmod(config_path, 0o644)

  assert response.status_code == 200
  assert response.json() == {"backendUrl": "/proxy", "telemetry": None}
  assert not caplog.records


def test_unparsable_packaged_config_falls_back_to_defaults(tmp_path, caplog):
  config_path = _seed(tmp_path, "not json")

  with caplog.at_level(logging.WARNING, logger="google_adk"):
    response = _make_client(tmp_path).get(_URL)

  assert response.json() == {"backendUrl": "", "telemetry": None}
  assert "Failed to decode JSON" in caplog.text
  assert config_path.read_text() == "not json"


def test_unrelated_keys_are_preserved(tmp_path):
  _seed(tmp_path, '{"backendUrl": "", "customKey": 1}')

  assert _make_client(tmp_path).get(_URL).json()["customKey"] == 1


def test_url_prefix_becomes_the_backend_url(tmp_path):
  response = _make_client(tmp_path, url_prefix="/proxy").get(_URL)

  assert response.json()["backendUrl"] == "/proxy"


@pytest.mark.parametrize("consent", [True, False, None])
def test_telemetry_consent_is_injected(tmp_path, monkeypatch, consent):
  monkeypatch.setattr(
      "google.adk.cli.api_server.read_telemetry_consent", lambda: consent
  )

  response = _make_client(tmp_path).get(_URL)

  assert response.json()["telemetry"] == consent


def test_logo_options_are_served(tmp_path):
  response = _make_client(
      tmp_path,
      logo_text="ACME",
      logo_image_url="https://example.com/logo.png",
  ).get(_URL)

  assert response.json()["logo"] == {
      "text": "ACME",
      "imageUrl": "https://example.com/logo.png",
  }


def test_stale_logo_in_packaged_config_is_dropped(tmp_path):
  _seed(
      tmp_path, '{"backendUrl": "", "logo": {"text": "old", "imageUrl": "x"}}'
  )

  assert "logo" not in _make_client(tmp_path).get(_URL).json()


@pytest.mark.parametrize(
    "logo_kwargs",
    [
        {"logo_text": "ACME"},
        {"logo_image_url": "https://example.com/logo.png"},
    ],
)
def test_partial_logo_configuration_is_rejected(tmp_path, logo_kwargs):
  api_server = _make_api_server(tmp_path, **logo_kwargs)

  with pytest.raises(ValueError, match="Both --logo-text and --logo-image-url"):
    api_server.get_fast_api_app(web_assets_dir=str(tmp_path))


def test_route_wins_over_the_static_mount(tmp_path):
  config_path = _seed(tmp_path, '{"backendUrl": "STALE"}')

  response = _make_client(tmp_path, url_prefix="/proxy").get(_URL)

  assert response.json()["backendUrl"] == "/proxy"
  assert config_path.read_text() == '{"backendUrl": "STALE"}'


def test_response_is_not_cached(tmp_path):
  response = _make_client(tmp_path).get(_URL)

  assert response.headers["cache-control"] == "no-store"


def test_endpoint_absent_without_web_assets_dir(tmp_path):
  app = _make_api_server(tmp_path).get_fast_api_app()

  assert TestClient(app).get(_URL).status_code == 404


def test_concurrent_servers_do_not_share_state(tmp_path):
  config_path = _seed(tmp_path, '{"backendUrl": "packaged"}')

  first = _make_client(tmp_path, url_prefix="/a")
  second = _make_client(tmp_path, url_prefix="/b")

  assert first.get(_URL).json()["backendUrl"] == "/a"
  assert second.get(_URL).json()["backendUrl"] == "/b"
  assert config_path.read_text() == '{"backendUrl": "packaged"}'


def test_web_ui_app_does_not_touch_the_packaged_runtime_config(tmp_path):
  packaged_before = _PACKAGED_RUNTIME_CONFIG.read_bytes()

  app = fast_api.get_fast_api_app(
      agents_dir=str(tmp_path),
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )

  assert TestClient(app).get(_URL).status_code == 200
  assert _PACKAGED_RUNTIME_CONFIG.read_bytes() == packaged_before
