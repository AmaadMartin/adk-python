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

"""Shared fixtures for the ADK CLI unit tests."""

from __future__ import annotations

from google.adk.cli import fast_api
import pytest

_PACKAGED_RUNTIME_CONFIG = (
    fast_api._WEB_ASSETS_DIR / "assets" / "config" / "runtime-config.json"
)


# Snapshot taken once, so every test that rewrites the packaged file fails and
# not only the first one.
_ORIGINAL_RUNTIME_CONFIG_BYTES = _PACKAGED_RUNTIME_CONFIG.read_bytes()


@pytest.fixture(autouse=True)
def packaged_runtime_config_stays_untouched():
  """Fails any CLI test that writes the packaged dev UI runtime config."""
  yield

  assert (
      _PACKAGED_RUNTIME_CONFIG.read_bytes() == _ORIGINAL_RUNTIME_CONFIG_BYTES
  ), (
      f"{_PACKAGED_RUNTIME_CONFIG} was modified by this test. The server must"
      " never write under its packaged web assets directory."
  )
