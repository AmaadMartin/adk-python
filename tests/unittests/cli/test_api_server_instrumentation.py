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

"""Unit tests for optional OTel instrumentation setup in api_server.

These tests call the private `_setup_instrumentation_lib_if_installed()`
directly. No public entry point reaches it without real telemetry setup: both
callers first build GCP exporters or mutate the global OTel providers.

The optional instrumentation packages are replaced with fakes through
`sys.modules`, so the tests behave the same with and without the
`google-adk[otel-gcp]` extra installed. The fakes are installed by an autouse
fixture because the real `GoogleGenAiSdkInstrumentor().instrument()` patches
the `google.genai` SDK process-globally and is never undone.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Protocol

from google.adk.cli import api_server
import pytest

_GENAI_MODULE = "opentelemetry.instrumentation.google_genai"
_GENAI_CLASS = "GoogleGenAiSdkInstrumentor"
_HTTPX_MODULE = "opentelemetry.instrumentation.httpx"
_HTTPX_CLASS = "HTTPXClientInstrumentor"
_GRPC_MODULE = "opentelemetry.instrumentation.grpc"
_GRPC_CLASS = "GrpcInstrumentorClient"

_INSTRUMENTOR_CLASSES = {
    _GENAI_MODULE: _GENAI_CLASS,
    _HTTPX_MODULE: _HTTPX_CLASS,
    _GRPC_MODULE: _GRPC_CLASS,
}

_AGENT_ENGINE_ID_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ID"
_AGENT_ENGINE_ID = "test-engine-id"

_LOGGER_NAME = api_server.logger.name


class _FailInstrumentor(Protocol):
  """Makes one optional instrumentation module fail for the current test."""

  def __call__(self, module_name: str, error: Exception | None = None) -> None:
    ...


def _fake_instrumentation_module(
    module_name: str,
    calls: list[str],
    error: Exception | None = None,
) -> types.ModuleType:
  """Builds a stand-in for an optional opentelemetry instrumentation module.

  Args:
    module_name: The dotted name of the module to stand in for.
    calls: The recorder that each successful `instrument()` appends to.
    error: Raised by `instrument()` instead of recording the call.

  Returns:
    A module exposing the instrumentor class that `api_server` imports from
    `module_name`.
  """
  class_name = _INSTRUMENTOR_CLASSES[module_name]

  class _FakeInstrumentor:

    def instrument(self) -> None:
      if error is not None:
        raise error
      calls.append(class_name)

  module = types.ModuleType(module_name)
  setattr(module, class_name, _FakeInstrumentor)
  return module


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
  """Returns the messages of the warning-or-worse records in `caplog`."""
  return [
      record.getMessage()
      for record in caplog.records
      if record.levelno >= logging.WARNING
  ]


def _only_warning(caplog: pytest.LogCaptureFixture) -> str:
  """Returns the one warning-or-worse message that `caplog` captured."""
  messages = _warnings(caplog)
  assert len(messages) == 1, messages
  return messages[0]


@pytest.fixture
def calls() -> list[str]:
  """Records the name of every instrumentor that took effect, in order."""
  return []


@pytest.fixture(autouse=True)
def fail_instrumentor(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> _FailInstrumentor:
  """Installs a working fake for every optional instrumentation module.

  Also clears the Agent Engine id, so every test starts off Agent Engine.

  Returns:
    A helper that makes one module unimportable, or - when given an `error` -
    makes its `instrument()` raise.
  """
  monkeypatch.delenv(_AGENT_ENGINE_ID_ENV, raising=False)
  for module_name in _INSTRUMENTOR_CLASSES:
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_instrumentation_module(module_name, calls),
    )

  def fail(module_name: str, error: Exception | None = None) -> None:
    # `None` in sys.modules makes the import raise ModuleNotFoundError.
    replacement = (
        None
        if error is None
        else _fake_instrumentation_module(module_name, calls, error)
    )
    monkeypatch.setitem(sys.modules, module_name, replacement)

  return fail


def test_instruments_genai_sdk_when_available(
    calls: list[str], caplog: pytest.LogCaptureFixture
):
  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS]
  assert _warnings(caplog) == []


def test_warns_when_genai_instrumentor_missing(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    fail_instrumentor: _FailInstrumentor,
):
  fail_instrumentor(_GENAI_MODULE)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == []
  message = _only_warning(caplog)
  assert "Unable to import GoogleGenAiSdkInstrumentor" in message
  assert "ModuleNotFoundError" in message
  assert "Make sure to install google-adk[otel-gcp]" in message


def test_instruments_httpx_and_grpc_on_agent_engine(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS, _HTTPX_CLASS, _GRPC_CLASS]
  assert _warnings(caplog) == []


def test_warns_when_httpx_and_grpc_missing_on_agent_engine(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fail_instrumentor: _FailInstrumentor,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)
  fail_instrumentor(_HTTPX_MODULE)
  fail_instrumentor(_GRPC_MODULE)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS]
  messages = _warnings(caplog)
  assert len(messages) == 2, messages
  httpx_message, grpc_message = messages
  assert "without HTTPX instrumentation" in httpx_message
  assert "without gRPC instrumentation" in grpc_message
  for message in messages:
    assert "ModuleNotFoundError" in message
    assert "Make sure to install google-adk[otel-gcp]" in message
    assert "has not been installed" not in message


def test_httpx_failure_does_not_block_grpc(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fail_instrumentor: _FailInstrumentor,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)
  fail_instrumentor(_HTTPX_MODULE)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS, _GRPC_CLASS]
  message = _only_warning(caplog)
  assert "without HTTPX instrumentation" in message
  assert "without gRPC instrumentation" not in message
  assert "ModuleNotFoundError" in message
  assert "Make sure to install google-adk[otel-gcp]" in message
  assert "has not been installed" not in message


def test_genai_failure_does_not_block_agent_engine_instrumentors(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fail_instrumentor: _FailInstrumentor,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)
  fail_instrumentor(_GENAI_MODULE)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_HTTPX_CLASS, _GRPC_CLASS]
  assert "Unable to import GoogleGenAiSdkInstrumentor" in caplog.text
  assert "without HTTPX instrumentation" not in caplog.text
  assert "without gRPC instrumentation" not in caplog.text


def test_httpx_attribute_error_is_tolerated(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fail_instrumentor: _FailInstrumentor,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)
  error = AttributeError(
      f"'{_HTTPX_CLASS}' object has no attribute 'instrument'"
  )
  fail_instrumentor(_HTTPX_MODULE, error)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS, _GRPC_CLASS]
  message = _only_warning(caplog)
  assert "without HTTPX instrumentation" in message
  assert f"AttributeError: {error}" in message
  assert "google-adk[otel-gcp]" not in message


def test_grpc_attribute_error_is_tolerated(
    calls: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fail_instrumentor: _FailInstrumentor,
):
  monkeypatch.setenv(_AGENT_ENGINE_ID_ENV, _AGENT_ENGINE_ID)
  error = AttributeError(
      f"'{_GRPC_CLASS}' object has no attribute 'instrument'"
  )
  fail_instrumentor(_GRPC_MODULE, error)

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    api_server._setup_instrumentation_lib_if_installed()

  assert calls == [_GENAI_CLASS, _HTTPX_CLASS]
  message = _only_warning(caplog)
  assert "without gRPC instrumentation" in message
  assert f"AttributeError: {error}" in message
  assert "google-adk[otel-gcp]" not in message
