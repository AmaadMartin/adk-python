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

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
import os

import google.auth
from opentelemetry import _logs
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
import opentelemetry.sdk.environment_variables as otel_env
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import OTELResourceDetector
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger("google_adk." + __name__)


@dataclass
class OTelHooks:
  span_processors: list[SpanProcessor] = field(default_factory=list)
  metric_readers: list[MetricReader] = field(default_factory=list)
  log_record_processors: list[LogRecordProcessor] = field(default_factory=list)


def maybe_set_otel_providers(
    otel_hooks_to_setup: list[OTelHooks] | None = None,
    otel_resource: Resource | None = None,
) -> None:
  """Sets up OTel providers if hooks for a given telemetry type were
  passed.

  Additionally adds generic OTLP exporters based on following env variables:
  OTEL_EXPORTER_OTLP_ENDPOINT
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
  OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
  OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
  See https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/
  for how they are used.

  If a provider for a specific telemetry type was already globally set -
  this function will not override it or register more exporters.

  Args:
    otel_hooks_to_setup: per-telemetry-type processors and readers to be added
    to OTel providers. If no hooks for a specific telemetry type are passed -
    provider will not be set.
    otel_resource: OTel resource to use in providers.
    If empty - default OTel resource detection will be used.
  """
  hooks_to_setup = list(otel_hooks_to_setup or ())
  otel_resource = otel_resource or _get_otel_resource()

  # Add generic OTel exporters based on OTel env variables.
  hooks_to_setup.append(_get_otel_exporters())

  span_processors: list[SpanProcessor] = []
  metric_readers: list[MetricReader] = []
  log_record_processors: list[LogRecordProcessor] = []
  for otel_hooks in hooks_to_setup:
    span_processors.extend(otel_hooks.span_processors)
    metric_readers.extend(otel_hooks.metric_readers)
    log_record_processors.extend(otel_hooks.log_record_processors)

  # Try to set up OTel tracing.
  # If the TracerProvider was already set outside of ADK, this would be a no-op
  # and results in a warning. In such case we rely on user setup.
  if span_processors:
    new_tracer_provider = TracerProvider(resource=otel_resource)
    for span_processor in span_processors:
      new_tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(new_tracer_provider)

  # Try to set up OTel metrics.
  # If the MeterProvider was already set outside of ADK, this would be a no-op
  # and results in a warning. In such case we rely on user setup.
  if metric_readers:
    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=metric_readers,
            resource=otel_resource,
            # Not collecting on exit to avoid points being collected too close together.
            shutdown_on_exit=False,
        )
    )

  # Try to set up OTel logging.
  # If the LoggerProvider was already set outside of ADK, this would be a no-op
  # and results in a warning. In such case we rely on user setup.
  if log_record_processors:
    new_logger_provider = LoggerProvider(
        resource=otel_resource,
    )
    for log_record_processor in log_record_processors:
      new_logger_provider.add_log_record_processor(log_record_processor)
    _logs.set_logger_provider(new_logger_provider)


def setup_telemetry(
    otel_to_cloud: bool = False,
    internal_exporters: list[SpanProcessor] | None = None,
) -> None:
  """Installs the global OTel providers for this process.

  Args:
    otel_to_cloud: whether to export to Google Cloud Observability.
    internal_exporters: ADK-specific span processors to register on the tracer
      provider in addition to the exporters selected by `otel_to_cloud` and by
      the OTel environment variables.
  """
  # TODO - remove the else branch here once maybe_set_otel_providers is no
  # longer experimental.
  if otel_to_cloud:
    _setup_gcp_telemetry(internal_exporters=internal_exporters)
  elif _otel_env_vars_enabled():
    _setup_telemetry_from_env(internal_exporters=internal_exporters)
  else:
    # Old logic - to be removed when above leaves experimental.
    tracer_provider = TracerProvider()
    if internal_exporters is not None:
      for exporter in internal_exporters:
        tracer_provider.add_span_processor(exporter)
    trace.set_tracer_provider(tracer_provider=tracer_provider)


def flush_telemetry(timeout_millis: int = 30_000) -> None:
  """Force-flushes the globally installed OTel providers.

  Short-lived processes such as `adk run` must call this before exiting: the
  Cloud exporters buffer through a BatchSpanProcessor, a
  PeriodicExportingMetricReader and a BatchLogRecordProcessor, and the
  MeterProvider installed by `maybe_set_otel_providers` is built with
  `shutdown_on_exit=False`, so anything still buffered at exit is dropped.

  Providers that are not SDK providers - the API's no-op defaults when no
  provider was installed - have no `force_flush` and are skipped.

  Args:
    timeout_millis: per-provider flush budget, in milliseconds.
  """
  providers = {
      "tracer": trace.get_tracer_provider(),
      "meter": metrics.get_meter_provider(),
      "logger": _logs.get_logger_provider(),
  }
  for name, provider in providers.items():
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is None:
      continue
    if not force_flush(timeout_millis):
      logger.warning(
          "OTel %s provider did not flush within %d ms - some telemetry was"
          " dropped.",
          name,
          timeout_millis,
      )


def _get_otel_resource() -> Resource:
  # The OTELResourceDetector populates resource labels from
  # environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
  return OTELResourceDetector().detect()


def _get_otel_exporters() -> OTelHooks:
  span_processors = []
  if os.getenv(otel_env.OTEL_EXPORTER_OTLP_ENDPOINT) or os.getenv(
      otel_env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
  ):
    span_processors.append(_get_otel_span_exporter())

  metric_readers = []
  if os.getenv(otel_env.OTEL_EXPORTER_OTLP_ENDPOINT) or os.getenv(
      otel_env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
  ):
    metric_readers.append(_get_otel_metrics_exporter())

  log_record_processors = []
  if os.getenv(otel_env.OTEL_EXPORTER_OTLP_ENDPOINT) or os.getenv(
      otel_env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT
  ):
    log_record_processors.append(_get_otel_logs_exporter())

  return OTelHooks(
      span_processors=span_processors,
      metric_readers=metric_readers,
      log_record_processors=log_record_processors,
  )


def _get_otel_span_exporter() -> SpanProcessor:
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

  return BatchSpanProcessor(OTLPSpanExporter())


def _get_otel_metrics_exporter() -> MetricReader:
  from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

  return PeriodicExportingMetricReader(OTLPMetricExporter())


def _get_otel_logs_exporter() -> LogRecordProcessor:
  from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

  return BatchLogRecordProcessor(OTLPLogExporter())


def _otel_env_vars_enabled() -> bool:
  return any([
      os.getenv(endpoint_var)
      for endpoint_var in [
          otel_env.OTEL_EXPORTER_OTLP_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
      ]
  ])


def _setup_gcp_telemetry(
    internal_exporters: list[SpanProcessor] | None = None,
) -> None:
  # Imported here to break the import cycle: google_cloud imports OTelHooks
  # from this module.
  from .google_cloud import get_gcp_exporters
  from .google_cloud import get_gcp_resource

  otel_hooks_to_add: list[OTelHooks] = []

  if internal_exporters:
    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  credentials, project_id = google.auth.default()

  otel_hooks_to_add.append(
      get_gcp_exporters(
          # TODO - use trace_to_cloud here as well once otel_to_cloud is no
          # longer experimental.
          enable_cloud_tracing=True,
          enable_cloud_metrics=True,
          enable_cloud_logging=True,
          google_auth=(credentials, project_id),
      )
  )
  otel_resource = get_gcp_resource(project_id)

  maybe_set_otel_providers(
      otel_hooks_to_setup=otel_hooks_to_add,
      otel_resource=otel_resource,
  )
  _setup_instrumentation_lib_if_installed()


def _setup_telemetry_from_env(
    internal_exporters: list[SpanProcessor] | None = None,
) -> None:
  otel_hooks_to_add: list[OTelHooks] = []

  if internal_exporters:
    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  maybe_set_otel_providers(otel_hooks_to_setup=otel_hooks_to_add)
  _setup_instrumentation_lib_if_installed()


def _setup_instrumentation_lib_if_installed() -> None:
  # Set instrumentation to enable emitting OTel data from GenAISDK
  # Currently the instrumentation lib is in extras dependencies, make sure to
  # warn the user if it's not installed.
  try:
    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

    GoogleGenAiSdkInstrumentor().instrument()
  except ImportError:
    logger.warning(
        "Unable to import GoogleGenAiSdkInstrumentor - some"
        " telemetry will be disabled. Make sure to install google-adk[otel-gcp]"
    )
  if os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
    # Set up HTTPX and gRPC instrumentation for A2A multi-agent observability.
    try:
      from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

      HTTPXClientInstrumentor().instrument()
    except (ImportError, AttributeError):
      logger.warning(
          "telemetry enabled but proceeding without HTTPX instrumentation,"
          " because google-adk[otel-gcp] has not been installed"
      )
    try:
      from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

      GrpcInstrumentorClient().instrument()
    except (ImportError, AttributeError):
      logger.warning(
          "telemetry enabled but proceeding without gRPC instrumentation,"
          " because google-adk[otel-gcp] has not been installed"
      )
