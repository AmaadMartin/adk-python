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

import click
import pytest


@pytest.fixture(autouse=True)
def _mute_click(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Suppresses click output so test runs stay readable.

  Applies to every test in this directory. A test that needs to assert on
  click output can opt out with the ``unmute_click`` marker::

      @pytest.mark.unmute_click
      def test_prints_summary() -> None:
        ...
  """
  if "unmute_click" in request.keywords:
    return
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)
  monkeypatch.setattr(click, "secho", lambda *a, **k: None)
