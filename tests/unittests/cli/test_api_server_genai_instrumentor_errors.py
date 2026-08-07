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

"""Tests that a failing GenAI SDK instrumentor cannot stop api_server startup.

These tests call the private `_setup_instrumentation_lib_if_installed()`
because no public entry point reaches it without real telemetry setup:
`_setup_gcp_telemetry()` calls `google.auth.default()` and builds GCP
exporters, and `_setup_telemetry_from_env()` replaces the global OpenTelemetry
providers. Both are worse test seams than the private function.

The tests replace each instrumentation module with a fake.
`opentelemetry-instrumentation-google-genai` is in the `test` extra, so the
real `GoogleGenAiSdkInstrumentor().instrument()` runs otherwise. It patches the
`google.genai` SDK process-globally and nothing undoes it, so the patch would
leak into the other tests in the same pytest-xdist worker.
"""

import logging
import sys
import types

from google.adk.cli import api_server
import pytest

_GENAI_MODULE = "opentelemetry.instrumentation.google_genai"
_GENAI_CLASS = "GoogleGenAiSdkInstrumentor"
_HTTPX_MODULE = "opentelemetry.instrumentation.httpx"
_HTTPX_CLASS = "HTTPXClientInstrumentor"
_GRPC_MODULE = "opentelemetry.instrumentation.grpc"
_GRPC_CLASS = "GrpcInstrumentorClient"
_AGENT_ENGINE_ID_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ID"
_LOGGER_NAME = api_server.logger.name


def _fake_instrumentation_module(
    module_name: str,
    class_name: str,
    calls: list[str],
    error: Exception | None = None,
) -> types.ModuleType:
  """Builds a stand-in for an optional instrumentation module."""

  class _FakeInstrumentor:

    def instrument(self) -> None:
      if error is not None:
        raise error
      calls.append(class_name)

  module = types.ModuleType(module_name)
  setattr(module, class_name, _FakeInstrumentor)
  return module


@pytest.fixture
def calls() -> list[str]:
  """Records each instrumentor that took effect, in order."""
  return []


@pytest.fixture(autouse=True)
def working_instrumentation_modules(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
  """Installs a working fake for every optional instrumentation module.

  Putting the full dotted name in `sys.modules` is enough: the import machinery
  finds it there and never imports the real package or its parents. That also
  covers HTTPX and gRPC, which the `test` extra does not install. `monkeypatch`
  restores `sys.modules` after each test.
  """
  for module_name, class_name in (
      (_GENAI_MODULE, _GENAI_CLASS),
      (_HTTPX_MODULE, _HTTPX_CLASS),
      (_GRPC_MODULE, _GRPC_CLASS),
  ):
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_instrumentation_module(module_name, class_name, calls),
    )
  monkeypatch.delenv(_AGENT_ENGINE_ID_ENV, raising=False)


def _fail_genai_instrumentor(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
  """Makes the GenAI instrumentor raise the version-skew AttributeError."""
  monkeypatch.setitem(
      sys.modules,
      _GENAI_MODULE,
      _fake_instrumentation_module(
          _GENAI_MODULE,
          _GENAI_CLASS,
          calls,
          AttributeError("module 'google.genai' has no attribute 'Models'"),
      ),
  )


def test_genai_instrument_attribute_error_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
  _fail_genai_instrumentor(monkeypatch, calls)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert "Unable to import GoogleGenAiSdkInstrumentor" in caplog.text
  assert "google-adk[otel-gcp]" in caplog.text


def test_genai_attribute_error_does_not_block_agent_engine_instrumentors(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, "test-agent-engine-id")
  _fail_genai_instrumentor(monkeypatch, calls)

  api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_HTTPX_CLASS, _GRPC_CLASS]
