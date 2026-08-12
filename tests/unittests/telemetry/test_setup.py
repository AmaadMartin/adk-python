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

import logging
import os
import sys
import types
from unittest import mock

from google.adk.telemetry.setup import flush_telemetry
from google.adk.telemetry.setup import maybe_set_otel_providers
from google.adk.telemetry.setup import OTelHooks
from google.adk.telemetry.setup import setup_telemetry
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


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


class _RecordingSpanProcessor(SpanProcessor):
  """Records the name of every span started on the provider it is added to."""

  def __init__(self) -> None:
    self.started: list[str] = []

  def on_start(self, span, parent_context=None) -> None:
    self.started.append(span.name)


def _fake_instrumentor_module(
    class_name: str, recorded: list[str]
) -> types.ModuleType:
  """Builds a stand-in for one of the optional OTel instrumentation packages."""

  class _Instrumentor:

    def instrument(self) -> None:
      recorded.append(class_name)

  module = types.ModuleType(f"fake_{class_name}")
  setattr(module, class_name, _Instrumentor)
  return module


@pytest.fixture
def no_otel_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
  """Removes every OTel endpoint variable the ambient environment may set."""
  for name in (
      "OTEL_EXPORTER_OTLP_ENDPOINT",
      "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
      "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
      "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
      "GOOGLE_CLOUD_AGENT_ENGINE_ID",
  ):
    monkeypatch.delenv(name, raising=False)


@pytest.fixture(name="instrumented")
def instrumented_fixture(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """Replaces the optional instrumentation packages with recording fakes."""
  recorded: list[str] = []
  for module_name, class_name in (
      (
          "opentelemetry.instrumentation.google_genai",
          "GoogleGenAiSdkInstrumentor",
      ),
      ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
      ("opentelemetry.instrumentation.grpc", "GrpcInstrumentorClient"),
  ):
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_instrumentor_module(class_name, recorded),
    )
  return recorded


@pytest.fixture
def uninstrumentable(monkeypatch: pytest.MonkeyPatch) -> None:
  """Makes every optional instrumentation package fail to import."""
  for module_name in (
      "opentelemetry.instrumentation.google_genai",
      "opentelemetry.instrumentation.httpx",
      "opentelemetry.instrumentation.grpc",
  ):
    monkeypatch.setitem(sys.modules, module_name, None)


def _patch_provider_getters(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider,
    meter_provider,
    logger_provider,
) -> None:
  monkeypatch.setattr(
      "opentelemetry.trace.get_tracer_provider", lambda: tracer_provider
  )
  monkeypatch.setattr(
      "opentelemetry.metrics.get_meter_provider", lambda: meter_provider
  )
  monkeypatch.setattr(
      "opentelemetry._logs.get_logger_provider", lambda: logger_provider
  )


def test_flush_telemetry_flushes_every_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """flush_telemetry should flush the tracer, meter and logger providers."""
  # Arrange.
  providers = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
  _patch_provider_getters(monkeypatch, *providers)

  # Act.
  flush_telemetry()

  # Assert.
  for provider in providers:
    provider.force_flush.assert_called_once_with(30_000)


def test_flush_telemetry_honors_a_custom_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """flush_telemetry should pass its timeout on to every provider."""
  # Arrange.
  providers = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
  _patch_provider_getters(monkeypatch, *providers)

  # Act.
  flush_telemetry(timeout_millis=1234)

  # Assert.
  for provider in providers:
    provider.force_flush.assert_called_once_with(1234)


def test_flush_telemetry_skips_providers_that_cannot_flush(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  """The API-level no-op providers have no force_flush and must be skipped."""
  # Arrange.
  _patch_provider_getters(monkeypatch, object(), object(), object())
  caplog.set_level(logging.WARNING)

  # Act.
  flush_telemetry()

  # Assert.
  assert caplog.records == []


def test_flush_telemetry_warns_when_a_provider_does_not_flush(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  """An unsuccessful flush is reported as a warning and does not raise."""
  # Arrange.
  tracer_provider = mock.MagicMock()
  tracer_provider.force_flush.return_value = False
  _patch_provider_getters(
      monkeypatch, tracer_provider, mock.MagicMock(), mock.MagicMock()
  )
  caplog.set_level(logging.WARNING)

  # Act.
  flush_telemetry(timeout_millis=500)

  # Assert.
  assert len(caplog.records) == 1
  assert (
      "OTel tracer provider did not flush within 500 ms"
      in caplog.records[0].getMessage()
  )


@pytest.mark.usefixtures("no_otel_env_vars")
@pytest.mark.parametrize("pass_internal_exporters", [True, False])
def test_setup_telemetry_falls_back_to_a_bare_tracer_provider(
    monkeypatch: pytest.MonkeyPatch,
    pass_internal_exporters: bool,
) -> None:
  """Without the flag and without env vars, only ADK exporters are wired."""
  # Arrange.
  set_tracer_provider = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", set_tracer_provider
  )
  internal_exporter = _RecordingSpanProcessor()

  # Act.
  setup_telemetry(
      internal_exporters=[internal_exporter]
      if pass_internal_exporters
      else None
  )

  # Assert.
  tracer_provider = set_tracer_provider.call_args.kwargs["tracer_provider"]
  tracer_provider.get_tracer("test").start_span("a-span")
  assert internal_exporter.started == (
      ["a-span"] if pass_internal_exporters else []
  )


@pytest.mark.usefixtures("no_otel_env_vars")
def test_setup_telemetry_uses_the_otlp_env_exporters(
    monkeypatch: pytest.MonkeyPatch,
    instrumented: list[str],
) -> None:
  """An OTLP endpoint in the environment selects the env exporters."""
  # Arrange.
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
  for getter in (
      "_get_otel_span_exporter",
      "_get_otel_metrics_exporter",
      "_get_otel_logs_exporter",
  ):
    monkeypatch.setattr(
        f"google.adk.telemetry.setup.{getter}", lambda: mock.MagicMock()
    )
  set_tracer_provider = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", set_tracer_provider
  )
  monkeypatch.setattr(
      "opentelemetry.metrics.set_meter_provider", mock.MagicMock()
  )
  monkeypatch.setattr(
      "opentelemetry._logs.set_logger_provider", mock.MagicMock()
  )
  internal_exporter = _RecordingSpanProcessor()

  # Act.
  setup_telemetry(internal_exporters=[internal_exporter])

  # Assert.
  tracer_provider = set_tracer_provider.call_args.args[0]
  tracer_provider.get_tracer("test").start_span("a-span")
  assert internal_exporter.started == ["a-span"]
  assert instrumented == ["GoogleGenAiSdkInstrumentor"]


@pytest.mark.usefixtures("no_otel_env_vars")
@pytest.mark.parametrize("pass_internal_exporters", [True, False])
def test_setup_telemetry_uses_the_gcp_exporters(
    monkeypatch: pytest.MonkeyPatch,
    pass_internal_exporters: bool,
    instrumented: list[str],
) -> None:
  """The flag selects the Cloud exporters, resolved against ADC."""
  # Arrange.
  credentials = object()
  monkeypatch.setattr("google.auth.default", lambda: (credentials, "a-project"))
  gcp_exporter = _RecordingSpanProcessor()
  get_gcp_exporters = mock.MagicMock(
      return_value=OTelHooks(span_processors=[gcp_exporter])
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud.get_gcp_exporters", get_gcp_exporters
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud.get_gcp_resource",
      lambda project_id: Resource.create({"gcp.project_id": project_id}),
  )
  set_tracer_provider = mock.MagicMock()
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", set_tracer_provider
  )
  internal_exporter = _RecordingSpanProcessor()

  # Act.
  setup_telemetry(
      otel_to_cloud=True,
      internal_exporters=[internal_exporter]
      if pass_internal_exporters
      else None,
  )

  # Assert.
  get_gcp_exporters.assert_called_once_with(
      enable_cloud_tracing=True,
      enable_cloud_metrics=True,
      enable_cloud_logging=True,
      google_auth=(credentials, "a-project"),
  )
  tracer_provider = set_tracer_provider.call_args.args[0]
  tracer_provider.get_tracer("test").start_span("a-span")
  assert internal_exporter.started == (
      ["a-span"] if pass_internal_exporters else []
  )
  assert gcp_exporter.started == ["a-span"]
  assert tracer_provider.resource.attributes["gcp.project_id"] == "a-project"
  assert instrumented == ["GoogleGenAiSdkInstrumentor"]


@pytest.mark.usefixtures("no_otel_env_vars")
def test_setup_telemetry_instruments_a2a_clients_on_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
    instrumented: list[str],
) -> None:
  """On Agent Engine, the HTTPX and gRPC clients are instrumented too."""
  # Arrange.
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "an-engine")
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://c:4318")
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_span_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", mock.MagicMock()
  )

  # Act.
  setup_telemetry()

  # Assert.
  assert instrumented == [
      "GoogleGenAiSdkInstrumentor",
      "HTTPXClientInstrumentor",
      "GrpcInstrumentorClient",
  ]


@pytest.mark.usefixtures("no_otel_env_vars", "uninstrumentable")
def test_setup_telemetry_warns_when_the_instrumentation_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
  """Missing instrumentation packages warn once each and never raise."""
  # Arrange.
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "an-engine")
  monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://c:4318")
  monkeypatch.setattr(
      "google.adk.telemetry.setup._get_otel_span_exporter",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "opentelemetry.trace.set_tracer_provider", mock.MagicMock()
  )
  caplog.set_level(logging.WARNING)

  # Act.
  setup_telemetry()

  # Assert.
  messages = [record.message for record in caplog.records]
  assert len(messages) == 3
  assert "Unable to import GoogleGenAiSdkInstrumentor" in messages[0]
  assert "without HTTPX instrumentation" in messages[1]
  assert "without gRPC instrumentation" in messages[2]


def test_flush_telemetry_exports_buffered_telemetry() -> None:
  """Real SDK providers hand their buffered data to the exporters.

  The batch processor is given a 60 second schedule, so the span reaches the
  exporter only because flush_telemetry drains it.
  """
  # Arrange.
  span_exporter = InMemorySpanExporter()
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(
      BatchSpanProcessor(span_exporter, schedule_delay_millis=60_000)
  )
  metric_reader = InMemoryMetricReader()
  meter_provider = MeterProvider(
      metric_readers=[metric_reader], shutdown_on_exit=False
  )

  # `monkeypatch` is not used here: the providers must be torn down in a
  # `finally`, and their batch worker threads outlive fixture finalization.
  with (
      mock.patch(
          "opentelemetry.trace.get_tracer_provider",
          return_value=tracer_provider,
      ),
      mock.patch(
          "opentelemetry.metrics.get_meter_provider",
          return_value=meter_provider,
      ),
  ):
    try:
      tracer_provider.get_tracer("test").start_span("a-span").end()
      meter_provider.get_meter("test").create_counter("a-counter").add(1)
      assert span_exporter.get_finished_spans() == ()

      # Act.
      flush_telemetry()

      # Assert.
      assert [span.name for span in span_exporter.get_finished_spans()] == [
          "a-span"
      ]
      metrics_data = metric_reader.get_metrics_data()
      assert (
          metrics_data.resource_metrics[0].scope_metrics[0].metrics[0].name
          == "a-counter"
      )
    finally:
      tracer_provider.shutdown()
      meter_provider.shutdown()
