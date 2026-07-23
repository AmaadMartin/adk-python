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

"""Tests for state delta isolation."""

from __future__ import annotations

from google.adk.sessions.state import State


def test_state_setitem_updates_delta_not_value() -> None:
  """State.__setitem__ only updates delta, not value."""
  state = State(value={"a": 1}, delta={})
  state["a"] = 2
  state["b"] = 3
  
  # Delta contains the uncommitted changes
  assert state._delta == {"a": 2, "b": 3}
  # Value remains unchanged
  assert state._value == {"a": 1}
  # getitem retrieves the updated value
  assert state["a"] == 2
  assert state["b"] == 3


def test_state_update_updates_delta_not_value() -> None:
  """State.update only updates delta, not value."""
  state = State(value={"a": 1}, delta={})
  state.update({"a": 2, "b": 3})
  
  # Delta contains the uncommitted changes
  assert state._delta == {"a": 2, "b": 3}
  # Value remains unchanged
  assert state._value == {"a": 1}
  # getitem retrieves the updated value
  assert state["a"] == 2
  assert state["b"] == 3


def test_state_setdefault_updates_delta_not_value() -> None:
  """State.setdefault only updates delta, not value."""
  state = State(value={"a": 1}, delta={})
  
  val1 = state.setdefault("a", 2)
  assert val1 == 1
  assert state._delta == {}
  
  val2 = state.setdefault("b", 3)
  assert val2 == 3
  assert state._delta == {"b": 3}
  assert state._value == {"a": 1}
