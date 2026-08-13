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

"""Tests for the HITL prompt rendered by `adk run`."""

from __future__ import annotations

import json

import google.adk.cli.cli as cli
import pytest

_COUNT_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}}


def test_prompt_prints_legacy_camel_case_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A schema recorded by an older adk-js is still shown to the user."""
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: "42")

  cli._prompt_for_function_call(
      "interrupt-1",
      "adk_request_input",
      {"message": "how many?", "responseSchema": _COUNT_SCHEMA},
  )

  assert f"Schema: {json.dumps(_COUNT_SCHEMA)}" in capsys.readouterr().out


def test_prompt_prefers_canonical_schema_over_legacy_camel_case_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """With both spellings present, only the canonical schema is shown."""
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: "42")
  legacy_schema = {"type": "string"}

  cli._prompt_for_function_call(
      "interrupt-1",
      "adk_request_input",
      {
          "message": "how many?",
          "response_schema": _COUNT_SCHEMA,
          "responseSchema": legacy_schema,
      },
  )

  out = capsys.readouterr().out
  assert f"Schema: {json.dumps(_COUNT_SCHEMA)}" in out
  assert json.dumps(legacy_schema) not in out
