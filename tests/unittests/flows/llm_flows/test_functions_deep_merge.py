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

"""Pins the recursive merge semantics that parallel tool calls depend on."""

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.functions import deep_merge_dicts
from google.adk.flows.llm_flows.functions import merge_parallel_function_response_events

# --- deep_merge_dicts ---


def test_nested_dicts_under_the_same_key_are_merged_not_replaced():
  """Two sources writing different keys under one key keep both values."""
  merged = deep_merge_dicts({'user': {'name': 'a'}}, {'user': {'age': 2}})

  assert merged == {'user': {'name': 'a', 'age': 2}}


def test_nesting_is_merged_at_every_level():
  """The merge recurses below the first level of nesting."""
  d1 = {'a': {'b': {'c': 1}, 'sibling': {'kept': True}}}
  d2 = {'a': {'b': {'d': 2}}}

  merged = deep_merge_dicts(d1, d2)

  assert merged == {'a': {'b': {'c': 1, 'd': 2}, 'sibling': {'kept': True}}}


def test_scalar_overwrites_a_nested_dict():
  """A scalar in the later source replaces the dict it lands on."""
  merged = deep_merge_dicts({'user': {'name': 'a'}}, {'user': 'anonymous'})

  assert merged == {'user': 'anonymous'}


def test_nested_dict_overwrites_a_scalar():
  """A dict in the later source replaces the scalar it lands on."""
  merged = deep_merge_dicts({'user': 'anonymous'}, {'user': {'name': 'a'}})

  assert merged == {'user': {'name': 'a'}}


def test_list_values_are_replaced_not_concatenated():
  """A list in the later source replaces the earlier list at every depth."""
  d1 = {'items': ['a'], 'section': {'items': ['a']}}
  d2 = {'items': ['b'], 'section': {'items': ['b']}}

  merged = deep_merge_dicts(d1, d2)

  assert merged == {'items': ['b'], 'section': {'items': ['b']}}


def test_keys_present_in_only_one_source_are_kept():
  """Keys that only one source defines survive at every depth."""
  d1 = {'only_in_d1': 1, 'shared': {'only_in_d1': 1}}
  d2 = {'only_in_d2': 2, 'shared': {'only_in_d2': 2}}

  merged = deep_merge_dicts(d1, d2)

  assert merged == {
      'only_in_d1': 1,
      'only_in_d2': 2,
      'shared': {'only_in_d1': 1, 'only_in_d2': 2},
  }


def test_merging_an_empty_dict_changes_nothing():
  """An empty later source leaves the earlier dict untouched."""
  assert deep_merge_dicts({'a': 1}, {}) == {'a': 1}


def test_merge_mutates_and_returns_the_first_dict():
  """The merge happens in place, so callers get the first dict back."""
  d1 = {'user': {'name': 'a'}}

  merged = deep_merge_dicts(d1, {'user': {'age': 2}})

  assert merged is d1
  assert d1 == {'user': {'name': 'a', 'age': 2}}


# --- merge_parallel_function_response_events ---


def _event(actions: EventActions) -> Event:
  """Builds the minimal function response event the merge accepts."""
  return Event(
      invocation_id='test_invocation',
      author='test_agent',
      actions=actions,
  )


def test_parallel_events_merge_nested_state_under_the_same_key():
  """Parallel events writing different nested state keys keep both values."""
  event1 = _event(EventActions(state_delta={'profile': {'name': 'ana'}}))
  event2 = _event(EventActions(state_delta={'profile': {'age': 42}}))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.state_delta == {
      'profile': {'name': 'ana', 'age': 42}
  }


def test_parallel_events_replace_state_delta_lists():
  """A list in a later parallel event replaces the earlier list."""
  event1 = _event(EventActions(state_delta={'items': ['a']}))
  event2 = _event(EventActions(state_delta={'items': ['b']}))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.state_delta == {'items': ['b']}


def test_parallel_events_keep_the_last_artifact_version_for_the_same_file():
  """The last parallel event decides the version of a shared artifact."""
  event1 = _event(EventActions(artifact_delta={'report.pdf': 1}))
  event2 = _event(EventActions(artifact_delta={'report.pdf': 2}))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.artifact_delta == {'report.pdf': 2}


def test_parallel_events_keep_artifact_versions_for_distinct_files():
  """Artifacts written by different parallel events all survive."""
  event1 = _event(EventActions(artifact_delta={'a.txt': 1}))
  event2 = _event(EventActions(artifact_delta={'b.txt': 2}))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.artifact_delta == {'a.txt': 1, 'b.txt': 2}


def test_parallel_events_keep_the_last_transfer_to_agent():
  """The last parallel event that sets a transfer target wins."""
  event1 = _event(EventActions(transfer_to_agent='agent_one'))
  event2 = _event(EventActions(transfer_to_agent='agent_two'))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.transfer_to_agent == 'agent_two'


def test_transfer_to_agent_survives_a_later_event_that_does_not_set_it():
  """An unset transfer target does not clear one an earlier event set."""
  event1 = _event(EventActions(transfer_to_agent='agent_one'))
  event2 = _event(EventActions(state_delta={'key': 'value'}))

  merged_event = merge_parallel_function_response_events([event1, event2])

  assert merged_event.actions.transfer_to_agent == 'agent_one'
