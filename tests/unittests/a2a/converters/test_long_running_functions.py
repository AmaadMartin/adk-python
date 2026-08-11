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

from google.adk.a2a import _compat
from google.adk.a2a.converters.long_running_functions import LongRunningFunctions
from google.adk.a2a.converters.part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.genai import types
import pytest


def _make_long_running_call_event(*calls: tuple[str, str]) -> Event:
  """Builds an event whose every function call is long running."""
  parts = [
      types.Part(
          function_call=types.FunctionCall(id=call_id, name=name, args={})
      )
      for call_id, name in calls
  ]
  return Event(
      invocation_id="invocation-1",
      author="agent",
      content=types.Content(role="model", parts=parts),
      long_running_tool_ids={call_id for call_id, _ in calls},
  )


def test_default_converter_returns_a2a_long_running_function_call():
  """The default converter must translate GenAI parts into A2A parts."""
  function_call = types.Part(
      function_call=types.FunctionCall(
          id="call-1", name="request_approval", args={}
      )
  )
  event = Event(
      invocation_id="invocation-1",
      author="agent",
      content=types.Content(role="model", parts=[function_call]),
      long_running_tool_ids={"call-1"},
  )
  long_running_functions = LongRunningFunctions()

  processed_event = long_running_functions.process_event(event)
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )

  assert processed_event.content is not None
  assert processed_event.content.parts == []
  assert result is not None
  assert result.status.state == _compat.TS_INPUT_REQUIRED
  assert result.status.message is not None
  result_part = result.status.message.parts[0]
  assert _compat.is_data_part(result_part)
  assert (
      _compat.part_metadata(result_part)[
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
      ]
      is True
  )


def test_euc_function_call_sets_auth_required_state():
  """A request for end-user credentials must pause the task in auth_required."""
  long_running_functions = LongRunningFunctions()

  long_running_functions.process_event(
      _make_long_running_call_event(("call-1", REQUEST_EUC_FUNCTION_CALL_NAME))
  )
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )

  assert result is not None
  assert result.status.state == _compat.TS_AUTH_REQUIRED
  assert result.status.message is not None
  result_part = result.status.message.parts[0]
  # The function name travels in the data payload, never in the part metadata.
  assert (
      _compat.data_part_dict(result_part)["name"]
      == REQUEST_EUC_FUNCTION_CALL_NAME
  )
  assert "name" not in _compat.part_metadata(result_part)


@pytest.mark.parametrize(
    "names",
    [
        (REQUEST_EUC_FUNCTION_CALL_NAME, "request_approval"),
        ("request_approval", REQUEST_EUC_FUNCTION_CALL_NAME),
    ],
)
def test_euc_call_takes_priority_over_other_long_running_calls(names):
  """auth_required must win over input_required, whatever the call order."""
  long_running_functions = LongRunningFunctions()

  long_running_functions.process_event(
      _make_long_running_call_event(
          ("call-1", names[0]),
          ("call-2", names[1]),
      )
  )
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )

  assert result is not None
  assert result.status.state == _compat.TS_AUTH_REQUIRED


def test_function_call_name_in_part_metadata_does_not_set_auth_required():
  """A name in the part metadata must not decide the task state."""
  event = Event(
      invocation_id="invocation-1",
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="call-1", name="request_approval", args={}
                  ),
                  part_metadata={"name": REQUEST_EUC_FUNCTION_CALL_NAME},
              )
          ],
      ),
      long_running_tool_ids={"call-1"},
  )
  long_running_functions = LongRunningFunctions()

  long_running_functions.process_event(event)
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )

  assert result is not None
  assert result.status.state == _compat.TS_INPUT_REQUIRED
  assert result.status.message is not None
  assert (
      _compat.part_metadata(result.status.message.parts[0])["name"]
      == REQUEST_EUC_FUNCTION_CALL_NAME
  )


def test_function_response_does_not_change_task_state():
  """A function response must not change the state an EUC call decided."""
  response_event = Event(
      invocation_id="invocation-1",
      author="agent",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="call-1",
                      name=REQUEST_EUC_FUNCTION_CALL_NAME,
                      response={"status": "ok"},
                  )
              )
          ],
      ),
  )
  long_running_functions = LongRunningFunctions()

  long_running_functions.process_event(
      _make_long_running_call_event(("call-1", REQUEST_EUC_FUNCTION_CALL_NAME))
  )
  processed_response_event = long_running_functions.process_event(
      response_event
  )
  result = long_running_functions.create_long_running_function_call_event(
      "task-1", "context-1"
  )
  repeated_result = (
      long_running_functions.create_long_running_function_call_event(
          "task-1", "context-1"
      )
  )

  assert processed_response_event.content is not None
  assert processed_response_event.content.parts == []
  assert result is not None
  assert result.status.message is not None
  assert len(result.status.message.parts) == 2
  assert result.status.state == _compat.TS_AUTH_REQUIRED
  assert repeated_result is not None
  assert repeated_result.status.state == _compat.TS_AUTH_REQUIRED
