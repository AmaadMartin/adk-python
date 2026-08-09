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

"""Tests for the dev UI runtime config written by ApiServer."""

from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path
from unittest import mock

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


def test_web_assets_dir_receives_the_rewritten_runtime_config(tmp_path):
  config_path = _seed(tmp_path, '{\n  "backendUrl": ""\n}')

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert config_path.read_text() == (
      '{\n  "backendUrl": "",\n  "telemetry": null\n}\n'
  )


def test_unrelated_keys_in_runtime_config_are_preserved(tmp_path):
  config_path = _seed(tmp_path, '{"backendUrl": "", "customKey": 1}')

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert json.loads(config_path.read_text())["customKey"] == 1


def test_missing_runtime_config_is_created(tmp_path):
  config_path = _runtime_config_path(tmp_path)

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert json.loads(config_path.read_text()) == {
      "backendUrl": "",
      "telemetry": None,
  }


def test_unparsable_runtime_config_is_overwritten(tmp_path):
  config_path = _seed(tmp_path, "not json")

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert json.loads(config_path.read_text()) == {
      "backendUrl": "",
      "telemetry": None,
  }


def test_url_prefix_becomes_the_backend_url(tmp_path):
  config_path = _runtime_config_path(tmp_path)

  _make_api_server(tmp_path, url_prefix="/proxy").get_fast_api_app(
      web_assets_dir=str(tmp_path)
  )

  assert json.loads(config_path.read_text())["backendUrl"] == "/proxy"


@pytest.mark.parametrize("consent", [True, False, None])
def test_telemetry_consent_is_injected(tmp_path, monkeypatch, consent):
  monkeypatch.setattr(
      "google.adk.cli.api_server.read_telemetry_consent", lambda: consent
  )
  config_path = _runtime_config_path(tmp_path)

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert json.loads(config_path.read_text())["telemetry"] == consent


def test_logo_options_are_written_to_the_runtime_config(tmp_path):
  config_path = _runtime_config_path(tmp_path)

  _make_api_server(
      tmp_path,
      logo_text="ACME",
      logo_image_url="https://example.com/logo.png",
  ).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert json.loads(config_path.read_text())["logo"] == {
      "text": "ACME",
      "imageUrl": "https://example.com/logo.png",
  }


def test_stale_logo_is_removed_when_no_logo_is_configured(tmp_path):
  config_path = _seed(
      tmp_path, '{"backendUrl": "", "logo": {"text": "old", "imageUrl": "x"}}'
  )

  _make_api_server(tmp_path).get_fast_api_app(web_assets_dir=str(tmp_path))

  assert "logo" not in json.loads(config_path.read_text())


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


def test_unwritable_runtime_config_is_logged_and_does_not_raise(
    tmp_path, monkeypatch, caplog
):
  _seed(tmp_path, '{"backendUrl": ""}')
  real_open = builtins.open

  def _fail_on_write(file, mode="r", *args, **kwargs):
    if "w" in mode:
      raise IOError("disk full")
    return real_open(file, mode, *args, **kwargs)

  monkeypatch.setattr(builtins, "open", _fail_on_write)

  with caplog.at_level(logging.ERROR, logger="google_adk"):
    app = _make_api_server(tmp_path).get_fast_api_app(
        web_assets_dir=str(tmp_path)
    )

  assert app is not None
  assert "Failed to write runtime config file" in caplog.text
  assert str(_runtime_config_path(tmp_path)) in caplog.text


def test_web_ui_app_does_not_touch_the_packaged_runtime_config(
    tmp_path, isolated_web_assets_dir
):
  packaged_before = _PACKAGED_RUNTIME_CONFIG.read_bytes()

  fast_api.get_fast_api_app(
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

  assert _runtime_config_path(isolated_web_assets_dir).exists()
  assert _PACKAGED_RUNTIME_CONFIG.read_bytes() == packaged_before
