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

from typing import Any
from typing import Optional

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.get_user_choice_tool import get_user_choice_tool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
import pytest

from ... import testing_utils


def test_async_function():
  responses = [
      Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def increase_by_one(x: int, tool_context: ToolContext) -> int:
    nonlocal function_called

    function_called += 1
    return {'status': 'pending'}

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[LongRunningFunctionTool(func=increase_by_one)],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 2
  # 1 item: user content
  assert mockModel.requests[0].contents == [
      testing_utils.UserContent('test1'),
  ]
  increase_by_one_call = Part.from_function_call(
      name='increase_by_one', args={'x': 1}
  )
  pending_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'pending'}
  )

  assert testing_utils.simplify_contents(mockModel.requests[1].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', pending_response),
  ]

  # Asserts the function calls.
  assert function_called == 1

  # Asserts the responses.
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='increase_by_one', args={'x': 1}),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='increase_by_one', response={'status': 'pending'}
          ),
      ),
      ('root_agent', 'response1'),
  ]
  assert events[0].long_running_tool_ids

  # Updates with another pending progress.
  still_waiting_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'still waiting'}
  )
  events = runner.run(testing_utils.UserContent(still_waiting_response))
  # We have one new request.
  assert len(mockModel.requests) == 3
  assert testing_utils.simplify_contents(mockModel.requests[2].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', still_waiting_response),
  ]

  assert testing_utils.simplify_events(events) == [('root_agent', 'response2')]

  # Calls when the result is ready.
  result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 2}
  )
  events = runner.run(testing_utils.UserContent(result_response))
  # We have one new request.
  assert len(mockModel.requests) == 4
  assert testing_utils.simplify_contents(mockModel.requests[3].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response3')]

  # Calls when the result is ready. Here we still accept the result and do
  # another summarization. Whether this is the right behavior is TBD.
  another_result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 3}
  )
  events = runner.run(testing_utils.UserContent(another_result_response))
  # We have one new request.
  assert len(mockModel.requests) == 5
  assert testing_utils.simplify_contents(mockModel.requests[4].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', another_result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response4')]

  # At the end, function_called should still be 1.
  assert function_called == 1


def test_async_function_with_none_response():
  responses = [
      Part.from_function_call(name='increase_by_one', args={'x': 1}),
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def increase_by_one(x: int, tool_context: ToolContext) -> int:
    nonlocal function_called
    function_called += 1
    return 'pending'

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[LongRunningFunctionTool(func=increase_by_one)],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 2
  # 1 item: user content
  assert mockModel.requests[0].contents == [
      testing_utils.UserContent('test1'),
  ]
  increase_by_one_call = Part.from_function_call(
      name='increase_by_one', args={'x': 1}
  )

  assert testing_utils.simplify_contents(mockModel.requests[1].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      (
          'user',
          Part.from_function_response(
              name='increase_by_one', response={'result': 'pending'}
          ),
      ),
  ]

  # Asserts the function calls.
  assert function_called == 1

  # Asserts the responses.
  assert testing_utils.simplify_events(events) == [
      (
          'root_agent',
          Part.from_function_call(name='increase_by_one', args={'x': 1}),
      ),
      (
          'root_agent',
          Part.from_function_response(
              name='increase_by_one', response={'result': 'pending'}
          ),
      ),
      ('root_agent', 'response1'),
  ]

  # Updates with another pending progress.
  still_waiting_response = Part.from_function_response(
      name='increase_by_one', response={'status': 'still waiting'}
  )
  events = runner.run(testing_utils.UserContent(still_waiting_response))
  # We have one new request.
  assert len(mockModel.requests) == 3
  assert testing_utils.simplify_contents(mockModel.requests[2].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', still_waiting_response),
  ]

  assert testing_utils.simplify_events(events) == [('root_agent', 'response2')]

  # Calls when the result is ready.
  result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 2}
  )
  events = runner.run(testing_utils.UserContent(result_response))
  # We have one new request.
  assert len(mockModel.requests) == 4
  assert testing_utils.simplify_contents(mockModel.requests[3].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response3')]

  # Calls when the result is ready. Here we still accept the result and do
  # another summarization. Whether this is the right behavior is TBD.
  another_result_response = Part.from_function_response(
      name='increase_by_one', response={'result': 3}
  )
  events = runner.run(testing_utils.UserContent(another_result_response))
  # We have one new request.
  assert len(mockModel.requests) == 5
  assert testing_utils.simplify_contents(mockModel.requests[4].contents) == [
      ('user', 'test1'),
      ('model', increase_by_one_call),
      ('user', another_result_response),
  ]
  assert testing_utils.simplify_events(events) == [('root_agent', 'response4')]

  # At the end, function_called should still be 1.
  assert function_called == 1


def _oauth_auth_config(client_id: str) -> AuthConfig:
  """Builds a minimal OAuth2 authorization-code AuthConfig."""
  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                  tokenUrl='https://oauth2.googleapis.com/token',
                  scopes={
                      'https://www.googleapis.com/auth/calendar': (
                          'See and edit your calendars'
                      )
                  },
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id=client_id,
              client_secret='oauth_client_secret',
          ),
      ),
  )


def _actions_only_events(events: list[Event]) -> list[Event]:
  """Returns the content-less events, i.e. the actions-only ones."""
  return [event for event in events if event.content is None]


def test_skip_summarization_survives_a_none_returning_long_running_tool():
  """`get_user_choice` sets skip_summarization and returns None."""
  mock_model = testing_utils.MockModel.create(
      responses=[
          Part.from_function_call(
              name='get_user_choice', args={'options': ['a', 'b']}
          ),
      ]
  )
  agent = Agent(
      name='root_agent', model=mock_model, tools=[get_user_choice_tool]
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('pick one')

  actions_only_events = _actions_only_events(events)
  assert len(actions_only_events) == 1
  assert actions_only_events[0].actions.skip_summarization is True
  assert not actions_only_events[0].long_running_tool_ids
  # The actions-only event must not trigger another model turn: the model is
  # configured with a single response and would raise if asked for a second.
  assert len(mock_model.requests) == 1


def test_state_delta_survives_a_none_returning_long_running_tool():
  mock_model = testing_utils.MockModel.create(
      responses=[Part.from_function_call(name='start_job', args={})]
  )

  def start_job(tool_context: ToolContext) -> None:
    tool_context.state['job_status'] = 'pending'
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=start_job)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('go')

  actions_only_events = _actions_only_events(events)
  assert len(actions_only_events) == 1
  assert actions_only_events[0].actions.state_delta == {'job_status': 'pending'}
  # The real session service applied the delta.
  assert runner.session.state['job_status'] == 'pending'


def test_actions_only_event_is_not_visible_to_the_model():
  mock_model = testing_utils.MockModel.create(
      responses=[
          Part.from_function_call(name='start_job', args={}),
          'done',
      ]
  )

  def start_job(tool_context: ToolContext) -> None:
    tool_context.state['job_status'] = 'pending'
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=start_job)],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('go')

  # Deliver the long-running tool's real result.
  job_done_response = Part.from_function_response(
      name='start_job', response={'status': 'done'}
  )
  job_done_response.function_response.id = (
      events[0].content.parts[0].function_call.id
  )
  runner.run(testing_utils.UserContent(job_done_response))

  # The actions-only event is content-less, so it never reaches the model.
  assert testing_utils.simplify_contents(mock_model.requests[-1].contents) == [
      ('user', 'go'),
      ('model', Part.from_function_call(name='start_job', args={})),
      (
          'user',
          Part.from_function_response(
              name='start_job', response={'status': 'done'}
          ),
      ),
  ]


def test_artifact_delta_survives_a_none_returning_long_running_tool():
  mock_model = testing_utils.MockModel.create(
      responses=[Part.from_function_call(name='save_report', args={})]
  )

  async def save_report(tool_context: ToolContext) -> None:
    await tool_context.save_artifact('f.txt', types.Part.from_text(text='x'))
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=save_report)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('go')

  actions_only_events = _actions_only_events(events)
  assert len(actions_only_events) == 1
  assert actions_only_events[0].actions.artifact_delta == {'f.txt': 0}


def test_untouched_actions_still_produce_no_event():
  mock_model = testing_utils.MockModel.create(
      responses=[Part.from_function_call(name='start_job', args={})]
  )

  def start_job() -> None:
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=start_job)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('go')

  assert testing_utils.simplify_events(events) == [
      ('root_agent', Part.from_function_call(name='start_job', args={})),
  ]
  assert not _actions_only_events(events)
  assert not _actions_only_events(runner.session.events)


def test_mixed_parallel_batch_merges_long_running_actions():
  mock_model = testing_utils.MockModel.create(
      responses=[
          [
              Part.from_function_call(name='start_job', args={}),
              Part.from_function_call(name='echo', args={'text': 'hi'}),
          ],
          'done',
      ]
  )

  def start_job(tool_context: ToolContext) -> None:
    tool_context.state['k'] = 'v'
    return None

  def echo(text: str) -> str:
    return text

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=start_job), echo],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('go')

  merged_event = events[1]
  assert testing_utils.simplify_content(
      merged_event.content
  ) == Part.from_function_response(name='echo', response={'result': 'hi'})
  assert merged_event.actions.state_delta == {'k': 'v'}


def test_all_long_running_parallel_batch_merges_into_content_less_event():
  mock_model = testing_utils.MockModel.create(
      responses=[[
          Part.from_function_call(name='start_job_a', args={}),
          Part.from_function_call(name='start_job_b', args={}),
      ]]
  )

  def start_job_a(tool_context: ToolContext) -> None:
    tool_context.state['a'] = 'a_pending'
    return None

  def start_job_b(tool_context: ToolContext) -> None:
    tool_context.state['b'] = 'b_pending'
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[
          LongRunningFunctionTool(func=start_job_a),
          LongRunningFunctionTool(func=start_job_b),
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('go')

  actions_only_events = _actions_only_events(events)
  assert len(actions_only_events) == 1
  assert actions_only_events[0].actions.state_delta == {
      'a': 'a_pending',
      'b': 'b_pending',
  }
  assert runner.session.state['a'] == 'a_pending'
  assert runner.session.state['b'] == 'b_pending'


def test_credential_request_from_a_none_returning_long_running_tool():
  auth_config = _oauth_auth_config(client_id='oauth_client_id')
  mock_model = testing_utils.MockModel.create(
      responses=[Part.from_function_call(name='call_external_api', args={})]
  )

  def call_external_api(tool_context: ToolContext) -> None:
    tool_context.request_credential(auth_config)
    return None

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[LongRunningFunctionTool(func=call_external_api)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = runner.run('test')

  auth_request_events = [
      event
      for event in events
      if any(
          function_call.name == functions.REQUEST_EUC_FUNCTION_CALL_NAME
          for function_call in event.get_function_calls()
      )
  ]
  assert len(auth_request_events) == 1
  # A content-less function response event must not leak a falsy role: that
  # would make the auth request look like empty content and get dropped from
  # the next LLM request.
  assert auth_request_events[0].content.role == 'user'


def test_resumable_invocation_still_pauses_on_a_mutating_long_running_tool():
  mock_model = testing_utils.MockModel.create(
      responses=[Part.from_function_call(name='start_job', args={})]
  )

  def start_job(tool_context: ToolContext) -> None:
    tool_context.state['job_status'] = 'pending'
    return None

  app = App(
      name='test_app',
      root_agent=Agent(
          name='root_agent',
          model=mock_model,
          tools=[LongRunningFunctionTool(func=start_job)],
      ),
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = runner.run('go')

  behavioral_events = [
      event
      for event in testing_utils.simplify_resumable_app_events(events)
      if not isinstance(event[1], dict)
  ]
  # The invocation paused right after the function call: no end-of-agent event
  # and no second model turn.
  assert behavioral_events == [
      ('root_agent', Part.from_function_call(name='start_job', args={})),
  ]
  assert len(mock_model.requests) == 1
  assert runner.session.state['job_status'] == 'pending'


class _DeferredResponseTool(BaseTool):
  """A tool that defers its response while mutating its actions."""

  def __init__(self):
    super().__init__(name='deferred_tool', description='Defers its response.')
    self._defers_response = True

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Optional[dict[str, Any]]:
    tool_context.state['deferred'] = 'pending'
    return None


@pytest.mark.asyncio
async def test_defers_response_tool_still_produces_no_event():
  tool = _DeferredResponseTool()
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=[]),
      tools=[tool],
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  function_call_event = Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      content=types.Content(
          parts=[types.Part(function_call=types.FunctionCall(name=tool.name))]
      ),
  )

  assert (
      await functions.handle_function_calls_async(
          invocation_context, function_call_event, {tool.name: tool}
      )
      is None
  )
