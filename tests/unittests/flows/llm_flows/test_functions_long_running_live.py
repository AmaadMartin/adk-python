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

"""Pins the live/bidi flow's skip-the-auto-FunctionResponse guard.

`_execute_single_function_call_live` skips building the automatic
FunctionResponse when a long-running (or response-deferring) tool returns a
*falsy* value -- not merely `None`. These tests hold the live copy of that
guard to the same contract as its non-live twin: falsy results from a
long-running tool are suppressed, while truthy results and ordinary tools are
left alone.
"""

from typing import Callable
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_live
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
import pytest

from ... import testing_utils

_FALSY_RESPONSES = [
    pytest.param({}, id='empty_dict'),
    pytest.param([], id='empty_list'),
    pytest.param('', id='empty_string'),
    pytest.param(0, id='zero'),
    pytest.param(False, id='false'),
]


def _make_tool_func(tool_result: object) -> Callable[[], object]:
  """Builds a zero-argument tool function returning `tool_result`."""

  def report_status() -> object:
    return tool_result

  return report_status


async def _run_live_single_call(tool: BaseTool) -> Optional[Event]:
  """Drives one function call for `tool` through the live flow."""
  model = testing_utils.MockModel.create(responses=[])
  agent = Agent(name='agent', model=model, tools=[tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  function_call = types.FunctionCall(name=tool.name, args={})
  content = types.Content(parts=[types.Part(function_call=function_call)])
  event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=content,
  )
  return await handle_function_calls_live(
      invocation_context, event, {tool.name: tool}
  )


@pytest.mark.parametrize('tool_result', _FALSY_RESPONSES)
@pytest.mark.asyncio
async def test_live_long_running_tool_with_falsy_result_emits_no_event(
    tool_result: object,
):
  """A long-running tool returning a falsy value emits no function response."""
  tool = LongRunningFunctionTool(func=_make_tool_func(tool_result))

  result = await _run_live_single_call(tool)

  assert result is None


@pytest.mark.parametrize(
    'tool_result, expected_response',
    [
        pytest.param({}, {}, id='empty_dict'),
        pytest.param([], {'result': []}, id='empty_list'),
        pytest.param('', {'result': ''}, id='empty_string'),
        pytest.param(0, {'result': 0}, id='zero'),
        pytest.param(False, {'result': False}, id='false'),
    ],
)
@pytest.mark.asyncio
async def test_live_regular_tool_with_falsy_result_still_emits_event(
    tool_result: object, expected_response: dict[str, object]
):
  """A non-long-running tool returning a falsy value still emits a response."""
  tool = FunctionTool(_make_tool_func(tool_result))

  result = await _run_live_single_call(tool)

  assert result is not None
  assert [r.response for r in result.get_function_responses()] == [
      expected_response
  ]


@pytest.mark.asyncio
async def test_live_long_running_tool_with_truthy_result_emits_event():
  """A long-running tool returning a value still emits a function response."""
  tool = LongRunningFunctionTool(func=_make_tool_func({'status': 'pending'}))

  result = await _run_live_single_call(tool)

  assert result is not None
  assert [r.response for r in result.get_function_responses()] == [
      {'status': 'pending'}
  ]
