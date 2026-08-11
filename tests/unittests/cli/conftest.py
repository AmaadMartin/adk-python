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

# Read from the server's own constant, so this guard follows the directory the
# server actually writes to. Module scope runs at collection, before the
# fixture below repoints it.
_PACKAGED_RUNTIME_CONFIG = (
    fast_api._WEB_ASSETS_DIR / "assets" / "config" / "runtime-config.json"
)


# Snapshot taken once, so every test that rewrites the packaged file fails and
# not only the first one.
_ORIGINAL_RUNTIME_CONFIG_BYTES = _PACKAGED_RUNTIME_CONFIG.read_bytes()


@pytest.fixture(autouse=True)
def isolated_web_assets_dir(tmp_path_factory, monkeypatch):
  """Keeps CLI tests from rewriting the packaged dev UI runtime config.

  ``ApiServer._setup_runtime_config`` rewrites
  ``<web_assets_dir>/assets/config/runtime-config.json`` every time an app is
  built with ``web=True``. Unpatched, that target is the copy checked into
  ``src/google/adk/cli/browser``, so running the suite dirties the working tree
  and lets xdist workers race on one file. Point the server at a throwaway
  directory, and fail the test that manages to touch the packaged copy anyway.

  Yields:
    The temporary directory the server treats as its web assets root.
  """
  assets_dir = tmp_path_factory.mktemp("web_assets")
  monkeypatch.setattr(fast_api, "_WEB_ASSETS_DIR", assets_dir)

  yield assets_dir

  assert (
      _PACKAGED_RUNTIME_CONFIG.read_bytes() == _ORIGINAL_RUNTIME_CONFIG_BYTES
  ), (
      f"{_PACKAGED_RUNTIME_CONFIG} was modified by this test. The server must"
      " only write runtime-config.json into a temporary web assets directory."
  )
