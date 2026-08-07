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

"""Tests for telemetry bootstrap branch selection in api_server."""

from __future__ import annotations

from typing import NamedTuple
from unittest import mock

from google.adk.cli.api_server import _otel_env_vars_enabled
from google.adk.cli.api_server import _setup_telemetry
import opentelemetry.sdk.environment_variables as otel_env
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
import pytest

_ENDPOINT = "http://localhost:4318"

_OTLP_ENV_VARS = (
    otel_env.OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
    otel_env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
    otel_env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
)


def _set_otlp_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
  """Clears every OTLP endpoint var, then sets only the given ones.

  Starting from a known-empty state keeps each parametrized case hermetic
  regardless of what the host environment happens to export.
  """
  for name in _OTLP_ENV_VARS:
    monkeypatch.delenv(name, raising=False)
  for name, value in env.items():
    monkeypatch.setenv(name, value)


def _span_processor() -> SpanProcessor:
  """Returns an inert SpanProcessor stand-in that is only ever forwarded."""
  return mock.create_autospec(SpanProcessor, instance=True)


class _TelemetryMocks(NamedTuple):
  """Every global side effect `_setup_telemetry` can reach, patched out."""

  setup_gcp: mock.MagicMock
  setup_from_env: mock.MagicMock
  tracer_provider_cls: mock.MagicMock
  tracer_provider: mock.MagicMock
  set_tracer_provider: mock.MagicMock


@pytest.fixture
def telemetry_mocks(monkeypatch: pytest.MonkeyPatch) -> _TelemetryMocks:
  """Patches the two delegates, the provider class and the global setter.

  Keeping all three boundaries mocked means no exporter is built, no network
  I/O happens and no process-global TracerProvider leaks into later tests.
  """
  tracer_provider = mock.MagicMock(spec=TracerProvider)
  mocks = _TelemetryMocks(
      setup_gcp=mock.MagicMock(),
      setup_from_env=mock.MagicMock(),
      tracer_provider_cls=mock.MagicMock(return_value=tracer_provider),
      tracer_provider=tracer_provider,
      set_tracer_provider=mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.cli.api_server._setup_gcp_telemetry", mocks.setup_gcp
  )
  monkeypatch.setattr(
      "google.adk.cli.api_server._setup_telemetry_from_env",
      mocks.setup_from_env,
  )
  monkeypatch.setattr(
      "google.adk.cli.api_server.TracerProvider", mocks.tracer_provider_cls
  )
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", mocks.set_tracer_provider
  )
  return mocks


@pytest.mark.parametrize("var_name", _OTLP_ENV_VARS)
def test_otel_env_vars_enabled_true_for_each_endpoint_var(
    monkeypatch: pytest.MonkeyPatch, var_name: str
):
  """Any single OTLP endpoint variable is enough to enable the env branch."""
  _set_otlp_env(monkeypatch, **{var_name: _ENDPOINT})

  assert _otel_env_vars_enabled() is True


def test_otel_env_vars_enabled_true_when_all_endpoint_vars_set(
    monkeypatch: pytest.MonkeyPatch,
):
  _set_otlp_env(monkeypatch, **{name: _ENDPOINT for name in _OTLP_ENV_VARS})

  assert _otel_env_vars_enabled() is True


def test_otel_env_vars_enabled_false_when_no_endpoint_var_set(
    monkeypatch: pytest.MonkeyPatch,
):
  _set_otlp_env(monkeypatch)

  assert _otel_env_vars_enabled() is False


@pytest.mark.parametrize("var_name", _OTLP_ENV_VARS)
def test_otel_env_vars_enabled_false_for_empty_endpoint_var(
    monkeypatch: pytest.MonkeyPatch, var_name: str
):
  """An empty value is falsy, so it counts as unset (current behaviour)."""
  _set_otlp_env(monkeypatch, **{var_name: ""})

  assert _otel_env_vars_enabled() is False


@pytest.mark.parametrize("var_name", _OTLP_ENV_VARS)
def test_setup_telemetry_uses_env_branch_for_each_endpoint_var(
    monkeypatch: pytest.MonkeyPatch,
    telemetry_mocks: _TelemetryMocks,
    var_name: str,
):
  _set_otlp_env(monkeypatch, **{var_name: _ENDPOINT})
  exporters = [_span_processor()]

  _setup_telemetry(internal_exporters=exporters)

  telemetry_mocks.setup_from_env.assert_called_once_with(
      internal_exporters=exporters
  )
  assert not telemetry_mocks.setup_gcp.called
  assert not telemetry_mocks.set_tracer_provider.called


def test_setup_telemetry_legacy_branch_registers_exporters_in_order(
    monkeypatch: pytest.MonkeyPatch, telemetry_mocks: _TelemetryMocks
):
  _set_otlp_env(monkeypatch)
  first, second = _span_processor(), _span_processor()

  _setup_telemetry(otel_to_cloud=False, internal_exporters=[first, second])

  telemetry_mocks.tracer_provider_cls.assert_called_once_with()
  assert telemetry_mocks.tracer_provider.add_span_processor.call_args_list == [
      mock.call(first),
      mock.call(second),
  ]
  telemetry_mocks.set_tracer_provider.assert_called_once_with(
      tracer_provider=telemetry_mocks.tracer_provider
  )
  assert not telemetry_mocks.setup_gcp.called
  assert not telemetry_mocks.setup_from_env.called


def test_setup_telemetry_legacy_branch_without_exporters(
    monkeypatch: pytest.MonkeyPatch, telemetry_mocks: _TelemetryMocks
):
  """The default `internal_exporters=None` still installs a bare provider."""
  _set_otlp_env(monkeypatch)

  _setup_telemetry()

  assert not telemetry_mocks.tracer_provider.add_span_processor.called
  telemetry_mocks.set_tracer_provider.assert_called_once_with(
      tracer_provider=telemetry_mocks.tracer_provider
  )
  assert not telemetry_mocks.setup_gcp.called
  assert not telemetry_mocks.setup_from_env.called


def test_setup_telemetry_uses_gcp_branch(
    monkeypatch: pytest.MonkeyPatch, telemetry_mocks: _TelemetryMocks
):
  _set_otlp_env(monkeypatch)
  exporters = [_span_processor()]

  _setup_telemetry(otel_to_cloud=True, internal_exporters=exporters)

  telemetry_mocks.setup_gcp.assert_called_once_with(
      internal_exporters=exporters
  )
  assert not telemetry_mocks.setup_from_env.called
  assert not telemetry_mocks.set_tracer_provider.called


def test_setup_telemetry_gcp_branch_wins_over_endpoint_vars(
    monkeypatch: pytest.MonkeyPatch, telemetry_mocks: _TelemetryMocks
):
  """`otel_to_cloud` takes precedence even with every endpoint var set."""
  _set_otlp_env(monkeypatch, **{name: _ENDPOINT for name in _OTLP_ENV_VARS})
  exporters = [_span_processor()]

  _setup_telemetry(otel_to_cloud=True, internal_exporters=exporters)

  telemetry_mocks.setup_gcp.assert_called_once_with(
      internal_exporters=exporters
  )
  assert not telemetry_mocks.setup_from_env.called
  assert not telemetry_mocks.set_tracer_provider.called
