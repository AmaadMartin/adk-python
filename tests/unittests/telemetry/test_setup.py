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

import builtins
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import logging
import os
import sys
from types import ModuleType
from unittest import mock

from google.adk.telemetry import setup as telemetry_setup
from google.adk.telemetry.setup import maybe_set_otel_providers
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import pytest

_OtlpExporter = SpanProcessor | MetricReader | LogRecordProcessor
_ExporterFactory = Callable[[], _OtlpExporter | None]

_TRACE_EXPORTER_MODULE = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
_METRIC_EXPORTER_MODULE = (
    "opentelemetry.exporter.otlp.proto.http.metric_exporter"
)
_LOG_EXPORTER_MODULE = "opentelemetry.exporter.otlp.proto.http._log_exporter"

_OTLP_EXPORTER_MODULES = (
    _TRACE_EXPORTER_MODULE,
    _METRIC_EXPORTER_MODULE,
    _LOG_EXPORTER_MODULE,
)

_OTLP_ENV_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
)

_EXPORTER_FACTORIES = (
    (
        telemetry_setup._get_otel_span_exporter,
        _TRACE_EXPORTER_MODULE,
        BatchSpanProcessor,
    ),
    (
        telemetry_setup._get_otel_metrics_exporter,
        _METRIC_EXPORTER_MODULE,
        PeriodicExportingMetricReader,
    ),
    (
        telemetry_setup._get_otel_logs_exporter,
        _LOG_EXPORTER_MODULE,
        BatchLogRecordProcessor,
    ),
)

_MISSING_PACKAGE = "opentelemetry-exporter-otlp-proto-http"


@pytest.fixture
def mock_os_environ():
  initial_env = os.environ.copy()
  with mock.patch.dict(os.environ, initial_env, clear=False) as m:
    yield m


@pytest.mark.parametrize(
    "env_vars, should_setup_trace, should_setup_metrics, should_setup_logs",
    [
        (
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "some-endpoint"},
            True,
            False,
            False,
        ),
        (
            {"OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": "some-endpoint"},
            False,
            True,
            False,
        ),
        (
            {"OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "some-endpoint"},
            False,
            False,
            True,
        ),
        (
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "some-endpoint",
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": "some-endpoint",
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": "some-endpoint",
            },
            True,
            True,
            True,
        ),
        (
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "some-endpoint"},
            True,
            True,
            True,
        ),
    ],
)
def test_maybe_set_otel_providers(
    env_vars: dict[str, str],
    should_setup_trace: bool,
    should_setup_metrics: bool,
    should_setup_logs: bool,
    monkeypatch: pytest.MonkeyPatch,
    mock_os_environ,  # pylint: disable=unused-argument,redefined-outer-name
):
  """
  Test initializing correct providers in setup_otel
  when providing OTel env variables.
  """
  # Arrange.
  for k, v in env_vars.items():
    monkeypatch.setenv(k, v)
  trace_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider",
      trace_provider_mock,
  )
  meter_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.metrics.set_meter_provider",
      meter_provider_mock,
  )
  logs_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry._logs.set_logger_provider",
      logs_provider_mock,
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_span_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_metrics_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_logs_exporter",
      lambda: mock.MagicMock(),
  )

  # Act.
  maybe_set_otel_providers()

  # Assert.
  # If given telemetry type was enabled,
  # the corresponding provider should be set.
  assert trace_provider_mock.call_count == (1 if should_setup_trace else 0)
  assert meter_provider_mock.call_count == (1 if should_setup_metrics else 0)
  assert logs_provider_mock.call_count == (1 if should_setup_logs else 0)


def _clear_otlp_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """Removes every OTLP endpoint variable inherited from the host."""
  for env_var in _OTLP_ENV_VARS:
    monkeypatch.delenv(env_var, raising=False)


def _otlp_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
  return [
      record.getMessage()
      for record in caplog.records
      if record.levelno == logging.WARNING
      and _MISSING_PACKAGE in record.getMessage()
  ]


@pytest.mark.parametrize(
    "exporter_factory, exporter_module",
    [(factory, module) for factory, module, _ in _EXPORTER_FACTORIES],
)
def test_exporter_factory_returns_none_when_package_is_missing(
    exporter_factory: _ExporterFactory,
    exporter_module: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Each factory degrades to None when the exporter package is absent."""
  caplog.set_level(logging.WARNING)

  with mock.patch.dict(sys.modules, {exporter_module: None}):
    exporter = exporter_factory()

  assert exporter is None
  warnings = _otlp_warnings(caplog)
  assert len(warnings) == 1
  assert exporter_module in warnings[0]


@pytest.mark.parametrize(
    "exporter_factory, exporter_module",
    [(factory, module) for factory, module, _ in _EXPORTER_FACTORIES],
)
def test_exporter_factory_returns_none_when_import_raises_attribute_error(
    exporter_factory: _ExporterFactory,
    exporter_module: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """An installed but broken exporter package disables the signal too."""
  caplog.set_level(logging.WARNING)
  real_import = builtins.__import__

  def raising_import(
      name: str,
      module_globals: Mapping[str, object] | None = None,
      module_locals: Mapping[str, object] | None = None,
      fromlist: Sequence[str] = (),
      level: int = 0,
  ) -> ModuleType:
    if name == exporter_module:
      raise AttributeError("simulated broken exporter package")
    return real_import(name, module_globals, module_locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", raising_import)

  exporter = exporter_factory()

  assert exporter is None
  warnings = _otlp_warnings(caplog)
  assert len(warnings) == 1
  assert "simulated broken exporter package" in warnings[0]


@pytest.mark.parametrize(
    "exporter_factory, expected_type",
    [(factory, expected) for factory, _, expected in _EXPORTER_FACTORIES],
)
def test_exporter_factory_returns_exporter_when_package_is_installed(
    exporter_factory: _ExporterFactory,
    expected_type: type[_OtlpExporter],
    caplog: pytest.LogCaptureFixture,
) -> None:
  """The guard does not disable export for users who have the package."""
  caplog.set_level(logging.WARNING)

  exporter = exporter_factory()

  try:
    assert isinstance(exporter, expected_type)
    assert not _otlp_warnings(caplog)
  finally:
    if exporter is not None:
      # Real exporters own a background worker thread; stop it.
      exporter.shutdown()


def test_get_otel_exporters_skips_missing_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """OTelHooks never carries a None exporter, which OTel cannot consume."""
  _clear_otlp_env(monkeypatch)
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "some-endpoint")

  with mock.patch.dict(
      sys.modules, {module: None for module in _OTLP_EXPORTER_MODULES}
  ):
    otel_hooks = telemetry_setup._get_otel_exporters()

  assert otel_hooks.span_processors == []
  assert otel_hooks.metric_readers == []
  assert otel_hooks.log_record_processors == []


class _FalsySpanProcessor(SpanProcessor):
  """A valid span processor that evaluates as false, like an empty container."""

  def __bool__(self) -> bool:
    return False


def test_get_otel_exporters_keeps_a_falsy_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Only a None result disables a signal, never a falsy exporter."""
  _clear_otlp_env(monkeypatch)
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "some-endpoint")
  span_processor = _FalsySpanProcessor()
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_span_exporter",
      lambda: span_processor,
  )

  otel_hooks = telemetry_setup._get_otel_exporters()

  assert otel_hooks.span_processors == [span_processor]


def test_maybe_set_otel_providers_without_exporter_package(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Startup survives an OTLP endpoint without the exporter package."""
  caplog.set_level(logging.WARNING)
  _clear_otlp_env(monkeypatch)
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "some-endpoint")
  trace_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", trace_provider_mock
  )
  meter_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.metrics.set_meter_provider", meter_provider_mock
  )
  logs_provider_mock = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry._logs.set_logger_provider", logs_provider_mock
  )

  with mock.patch.dict(
      sys.modules, {module: None for module in _OTLP_EXPORTER_MODULES}
  ):
    maybe_set_otel_providers()

  assert trace_provider_mock.call_count == 0
  assert meter_provider_mock.call_count == 0
  assert logs_provider_mock.call_count == 0
  warnings = _otlp_warnings(caplog)
  assert len(warnings) == 3
  for signal in ("trace", "metric", "log"):
    assert sum(f"OTLP {signal} export" in warning for warning in warnings) == 1
