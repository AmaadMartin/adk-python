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

"""Unit tests for auth_preprocessor module."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import logging
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from authlib.oauth2.rfc6749.parameters import parse_authorization_code_response
from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.auth import oauth2_credential_util
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_preprocessor import _AuthLlmRequestProcessor
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_tool import AuthToolArguments
from google.adk.events.event import Event
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


class TestAuthLlmRequestProcessor:
  """Tests for _AuthLlmRequestProcessor class."""

  @pytest.fixture
  def processor(self):
    """Create an _AuthLlmRequestProcessor instance."""
    return _AuthLlmRequestProcessor()

  @pytest.fixture
  def mock_llm_agent(self):
    """Create a mock LlmAgent."""
    from google.adk.agents.llm_agent import LlmAgent

    agent = Mock(spec=LlmAgent)
    agent.canonical_tools = AsyncMock(return_value=[])
    return agent

  @pytest.fixture
  def mock_non_llm_agent(self):
    """Create a mock non-LLM agent."""
    agent = Mock()
    agent.__class__.__name__ = 'BaseAgent'
    return agent

  @pytest.fixture
  def mock_session(self):
    """Create a mock session."""
    session = Mock()
    session.state = {}
    session.events = []
    return session

  @pytest.fixture
  def mock_invocation_context(self, mock_llm_agent, mock_session):
    """Create a mock invocation context."""
    context = Mock(spec=InvocationContext)
    context.agent = mock_llm_agent
    context.session = mock_session
    context._get_events.side_effect = lambda **_: context.session.events
    return context

  @pytest.fixture
  def mock_llm_request(self):
    """Create a mock LlmRequest."""
    return Mock(spec=LlmRequest)

  @pytest.fixture
  def mock_auth_config(self):
    """Create a mock AuthConfig."""
    config = Mock(spec=AuthConfig)
    config.credential_key = None
    config.raw_auth_credential = None
    config.exchanged_auth_credential = None
    return config

  @pytest.fixture
  def mock_function_response_with_auth(self, mock_auth_config):
    """Create a mock function response with auth data."""
    function_response = Mock()
    function_response.name = REQUEST_EUC_FUNCTION_CALL_NAME
    function_response.id = 'auth_response_id'
    function_response.response = mock_auth_config
    return function_response

  @pytest.fixture
  def mock_function_response_without_auth(self):
    """Create a mock function response without auth data."""
    function_response = Mock()
    function_response.name = 'some_other_function'
    function_response.id = 'other_response_id'
    return function_response

  @pytest.fixture
  def mock_user_event_with_auth_response(
      self, mock_function_response_with_auth
  ):
    """Create a mock user event with auth response."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_responses.return_value = [
        mock_function_response_with_auth
    ]
    return event

  @pytest.fixture
  def mock_user_event_without_auth_response(
      self, mock_function_response_without_auth
  ):
    """Create a mock user event without auth response."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_responses.return_value = [
        mock_function_response_without_auth
    ]
    return event

  @pytest.fixture
  def mock_user_event_no_responses(self):
    """Create a mock user event with no responses."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = Mock()  # Non-None content
    event.get_function_responses.return_value = []
    return event

  @pytest.fixture
  def mock_agent_event(self):
    """Create a mock agent-authored event."""
    event = Mock(spec=Event)
    event.author = 'test_agent'
    event.content = Mock()  # Non-None content
    return event

  @pytest.fixture
  def mock_event_no_content(self):
    """Create a mock event with no content."""
    event = Mock(spec=Event)
    event.author = 'user'
    event.content = None
    return event

  @pytest.fixture
  def mock_agent_event_with_content(self):
    """Create a mock agent event with content."""
    event = Mock(spec=Event)
    event.author = 'test_agent'
    event.content = Mock()  # Non-None content
    return event

  @pytest.mark.asyncio
  async def test_non_llm_agent_returns_early(
      self, processor, mock_llm_request, mock_session
  ):
    """Test that non-LLM agents return early."""
    mock_context = Mock(spec=InvocationContext)
    # Using spec=[] ensures hasattr(agent, 'canonical_tools') returns False.
    mock_context.agent = Mock(spec=[])
    mock_context.agent.__class__.__name__ = 'BaseAgent'
    mock_context.session = mock_session

    result = []
    async for event in processor.run_async(mock_context, mock_llm_request):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_empty_events_returns_early(
      self, processor, mock_invocation_context, mock_llm_request
  ):
    """Test that empty events list returns early."""
    mock_invocation_context.session.events = []

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_no_events_with_content_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_event_no_content,
  ):
    """Test that no events with content returns early."""
    mock_invocation_context.session.events = [mock_event_no_content]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_with_content_not_user_authored_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_event_no_content,
      mock_agent_event_with_content,
  ):
    """Test that last event with content not user-authored returns early."""
    # Mix of events: user event with no content, then agent event with content
    mock_invocation_context.session.events = [
        mock_event_no_content,
        mock_agent_event_with_content,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_no_responses_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_no_responses,
  ):
    """Test that user event with no responses returns early."""
    mock_invocation_context.session.events = [mock_user_event_no_responses]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  async def test_last_event_no_auth_responses_returns_early(
      self,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_without_auth_response,
  ):
    """Test that user event with non-auth responses returns early."""
    mock_invocation_context.session.events = [
        mock_user_event_without_auth_response
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  async def test_ignores_auth_responses_outside_current_branch(
      self,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
  ):
    """Test auth responses hidden by branch filtering are ignored."""
    mock_invocation_context.session.events = [
        mock_user_event_with_auth_response
    ]
    mock_invocation_context._get_events.side_effect = None
    mock_invocation_context._get_events.return_value = []

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    mock_invocation_context._get_events.assert_called_once_with(
        current_branch=True
    )
    mock_auth_config_validate.assert_not_called()
    mock_auth_handler_class.assert_not_called()
    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  async def test_processes_auth_response_successfully(
      self,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """Test successful processing of auth response in last event."""
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # This resume event carries only function responses.
    mock_user_event_with_auth_response.get_function_calls.return_value = []

    # The frozen adk_request_credential function call the client is answering.
    request_function_call = Mock()
    request_function_call.id = 'auth_response_id'
    request_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    request_function_call.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }
    request_event = Mock(spec=Event)
    request_event.content = Mock()  # Non-None content
    request_event.get_function_calls.return_value = [request_function_call]

    mock_invocation_context.session.events = [
        request_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Verify auth config validation was called
    mock_auth_config_validate.assert_called_once()

    # Verify the auth handler was created with a copy of the frozen request,
    # not with the config the client sent back.
    mock_auth_config.model_copy.assert_called_once_with(deep=True)
    mock_auth_handler_class.assert_called_once_with(
        auth_config=mock_auth_config.model_copy.return_value
    )

    # Verify parse_and_store_auth_response was called
    mock_auth_handler.parse_and_store_auth_response.assert_called_once_with(
        state=mock_invocation_context.session.state
    )

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_preprocessor.handle_function_calls_async')
  async def test_processes_multiple_auth_responses_and_resumes_tools(
      self,
      mock_handle_function_calls,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_auth_config,
  ):
    """Test processing multiple auth responses and resuming tools."""
    # Create multiple auth responses
    auth_response_1 = Mock()
    auth_response_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response_1.id = 'auth_id_1'
    auth_response_1.response = mock_auth_config

    auth_response_2 = Mock()
    auth_response_2.name = REQUEST_EUC_FUNCTION_CALL_NAME
    auth_response_2.id = 'auth_id_2'
    auth_response_2.response = mock_auth_config

    user_event_with_multiple_responses = Mock(spec=Event)
    user_event_with_multiple_responses.author = 'user'
    user_event_with_multiple_responses.content = Mock()  # Non-None content
    user_event_with_multiple_responses.get_function_responses.return_value = [
        auth_response_1,
        auth_response_2,
    ]
    user_event_with_multiple_responses.get_function_calls.return_value = []

    # Create system function call events
    system_function_call_1 = Mock()
    system_function_call_1.id = 'auth_id_1'
    system_function_call_1.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call_1.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_function_call_2 = Mock()
    system_function_call_2.id = 'auth_id_2'
    system_function_call_2.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call_2.args = {
        'function_call_id': 'tool_id_2',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [
        system_function_call_1,
        system_function_call_2,
    ]

    # Create original function call event
    original_function_call_1 = Mock()
    original_function_call_1.id = 'tool_id_1'

    original_function_call_2 = Mock()
    original_function_call_2.id = 'tool_id_2'

    original_event = Mock(spec=Event)
    original_event.content = Mock()  # Non-None content
    original_event.get_function_calls.return_value = [
        original_function_call_1,
        original_function_call_2,
    ]

    # Setup events in order: original -> system -> user_with_responses
    mock_invocation_context.session.events = [
        original_event,
        system_event,
        user_event_with_multiple_responses,
    ]

    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    mock_function_response_event = Mock(spec=Event)
    mock_handle_function_calls.return_value = mock_function_response_event

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Verify auth responses were processed
    assert mock_auth_handler.parse_and_store_auth_response.call_count == 2

    # Verify function calls were resumed
    mock_handle_function_calls.assert_called_once()
    call_args = mock_handle_function_calls.call_args
    assert call_args[0][1] == original_event  # The original event
    assert call_args[0][3] == {'tool_id_1', 'tool_id_2'}  # Tools to resume

    # Verify the function response event was yielded
    assert result == [mock_function_response_event]

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  async def test_no_matching_system_function_calls_returns_early(
      self,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """Test that missing matching system function calls returns early.

    Without a matching ``adk_request_credential`` function call there is no
    frozen request to bind the resume to, so nothing is stored.
    """
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # Create a non-matching system event
    non_matching_function_call = Mock()
    non_matching_function_call.id = (  # Different from 'auth_response_id'
        'different_id'
    )

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [non_matching_function_call]

    mock_invocation_context.session.events = [
        system_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Should drop the unmatched auth response and not resume any tools
    mock_auth_handler.parse_and_store_auth_response.assert_not_called()
    assert result == []

  @pytest.mark.asyncio
  @patch('google.adk.auth.auth_preprocessor.AuthHandler')
  @patch('google.adk.auth.auth_tool.AuthConfig.model_validate')
  @patch('google.adk.auth.auth_tool.AuthToolArguments.model_validate')
  async def test_handles_missing_original_function_calls(
      self,
      mock_auth_tool_args_validate,
      mock_auth_config_validate,
      mock_auth_handler_class,
      processor,
      mock_invocation_context,
      mock_llm_request,
      mock_user_event_with_auth_response,
      mock_auth_config,
  ):
    """Test handling when original function calls are not found."""
    # Setup mocks
    mock_auth_config_validate.return_value = mock_auth_config
    mock_auth_handler = Mock(spec=AuthHandler)
    mock_auth_handler.parse_and_store_auth_response = AsyncMock()
    mock_auth_handler_class.return_value = mock_auth_handler

    # This resume event carries only function responses.
    mock_user_event_with_auth_response.get_function_calls.return_value = []

    # Create matching system function call
    auth_tool_args = Mock(spec=AuthToolArguments)
    auth_tool_args.function_call_id = 'tool_id_1'
    auth_tool_args.auth_config = mock_auth_config
    mock_auth_tool_args_validate.return_value = auth_tool_args

    system_function_call = Mock()
    system_function_call.id = 'auth_response_id'  # Matches the response ID
    system_function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
    system_function_call.args = {
        'function_call_id': 'tool_id_1',
        'auth_config': mock_auth_config,
    }

    system_event = Mock(spec=Event)
    system_event.content = Mock()  # Non-None content
    system_event.get_function_calls.return_value = [system_function_call]

    # Create event with no function calls (original function calls missing)
    empty_event = Mock(spec=Event)
    empty_event.content = Mock()  # Non-None content
    empty_event.get_function_calls.return_value = []

    mock_invocation_context.session.events = [
        empty_event,
        system_event,
        mock_user_event_with_auth_response,
    ]

    result = []
    async for event in processor.run_async(
        mock_invocation_context, mock_llm_request
    ):
      result.append(event)

    # Should process auth response but not find original function calls
    mock_auth_handler.parse_and_store_auth_response.assert_called_once()
    assert result == []

  @pytest.mark.asyncio
  async def test_isinstance_check_for_llm_agent(
      self, processor, mock_llm_request, mock_session
  ):
    """Test that isinstance check works correctly for LlmAgent."""
    # This test ensures the isinstance check work as expected

    # Create a mock that fails isinstance check
    mock_context = Mock(spec=InvocationContext)
    # This will fail isinstance(agent, LlmAgent)
    mock_context.agent = Mock(spec=[])
    mock_context.session = mock_session

    result = []
    async for event in processor.run_async(mock_context, mock_llm_request):
      result.append(event)

    assert result == []


_FROZEN_STATE = 'adk-issued-state'
_FROZEN_REDIRECT_URI = 'https://app.example.com/cb'
_TOOL_TOKEN_URL = 'https://provider.example/token'
_AUTH_FC_ID = 'auth-fc-1'
_TOOL_FC_ID = 'tool-fc-1'
_TOOL_NAME = 'read_calendar'
_EXPIRES_AT = 4102444800


def _oauth2_auth_config(
    redirect_uri: str | None = _FROZEN_REDIRECT_URI,
) -> AuthConfig:
  """Builds the OAuth2 config a tool requests credentials with.

  ``exchanged_auth_credential`` mirrors what ``AuthHandler.generate_auth_uri``
  produces: the tool's client credentials plus the nonce ADK minted.

  Args:
    redirect_uri: The redirect URI the tool pins. Most tools pin none and let
      the client choose, so pass ``None`` for that shape.
  """
  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl='https://provider.example/auth',
                  tokenUrl=_TOOL_TOKEN_URL,
                  scopes={'read': 'Read access'},
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='tool-client',
              client_secret='tool-secret',
              redirect_uri=redirect_uri,
          ),
      ),
      exchanged_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id='tool-client',
              client_secret='tool-secret',
              redirect_uri=redirect_uri,
              state=_FROZEN_STATE,
              auth_uri=f'https://provider.example/auth?state={_FROZEN_STATE}',
              code_verifier='tool-verifier',
          ),
      ),
  )


def _api_key_auth_config(raw: AuthCredential | None) -> AuthConfig:
  """Builds an API-key config, optionally with a frozen raw credential."""
  return AuthConfig(
      auth_scheme=APIKey(**{'name': 'X-API-Key', 'in': APIKeyIn.header}),
      raw_auth_credential=raw,
  )


def _request_event(auth_config: AuthConfig) -> Event:
  """Builds the agent event carrying the frozen credential request.

  This mirrors ``build_auth_request_event`` so the camelCase and
  ``exclude_none`` round trip of the frozen args is under test too.
  """
  function_call = types.FunctionCall(
      name=REQUEST_EUC_FUNCTION_CALL_NAME,
      id=_AUTH_FC_ID,
      args=AuthToolArguments(
          function_call_id=_TOOL_FC_ID,
          auth_config=auth_config,
      ).model_dump(mode='json', exclude_none=True, by_alias=True),
  )
  return Event(
      invocation_id='test_id',
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(function_call=function_call)]
      ),
  )


def _tool_call_event() -> Event:
  """Builds the agent event carrying the original tool function call."""
  function_call = types.FunctionCall(name=_TOOL_NAME, id=_TOOL_FC_ID, args={})
  return Event(
      invocation_id='test_id',
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(function_call=function_call)]
      ),
  )


def _resume_event(payload: dict[str, Any]) -> Event:
  """Builds the user event carrying the client's resume payload."""
  function_response = types.FunctionResponse(
      name=REQUEST_EUC_FUNCTION_CALL_NAME, id=_AUTH_FC_ID, response=payload
  )
  return Event(
      invocation_id='test_id',
      author='user',
      content=types.Content(
          role='user', parts=[types.Part(function_response=function_response)]
      ),
  )


def _resume_payload(
    auth_config: AuthConfig,
    credential: AuthCredential | None,
    auth_scheme: AuthScheme | None = None,
) -> dict[str, Any]:
  """Serializes the AuthConfig a client sends back, as the client would."""
  payload = auth_config.model_copy(deep=True)
  payload.exchanged_auth_credential = credential
  if auth_scheme is not None:
    payload.auth_scheme = auth_scheme
  return payload.model_dump(mode='json', exclude_none=True, by_alias=True)


def _resumed_credential(**oauth2_fields: Any) -> AuthCredential:
  """Builds the OAuth2 credential a client puts in its resume payload."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2, oauth2=OAuth2Auth(**oauth2_fields)
  )


class _FakeOAuth2Session:
  """Stands in for authlib's OAuth2Session at the HTTP boundary."""

  def __init__(
      self,
      client_id: str | None = None,
      client_secret: str | None = None,
      **kwargs: Any,
  ) -> None:
    self.client_id = client_id
    self.client_secret = client_secret
    self.redirect_uri = kwargs.get('redirect_uri')
    self.state = kwargs.get('state')
    self.token_endpoints: list[str] = []
    self.fetch_token_kwargs: list[dict[str, Any]] = []

  def fetch_token(self, token_endpoint: str, **kwargs: Any) -> dict[str, Any]:
    # authlib validates the CSRF state before it contacts the provider, so
    # these tests pin authlib's comparison rather than reimplementing it. An
    # absent redirect URI carries no code, which authlib also rejects.
    parse_authorization_code_response(
        kwargs.get('authorization_response') or '', state=self.state
    )
    self.token_endpoints.append(token_endpoint)
    self.fetch_token_kwargs.append(kwargs)
    return {
        'access_token': 'minted-token',
        'refresh_token': 'minted-refresh',
        'expires_at': _EXPIRES_AT,
    }


class TestAuthResumeBinding:
  """Tests that a resume message cannot redefine the frozen auth request."""

  @pytest.fixture
  def sessions(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> list[_FakeOAuth2Session]:
    """Replaces authlib's session with a fake and records every instance."""
    created: list[_FakeOAuth2Session] = []

    def factory(*args: Any, **kwargs: Any) -> _FakeOAuth2Session:
      session = _FakeOAuth2Session(*args, **kwargs)
      created.append(session)
      return session

    monkeypatch.setattr(oauth2_credential_util, 'OAuth2Session', factory)
    return created

  async def _run(
      self,
      events: Sequence[Event],
      tools: Sequence[Callable[[ToolContext], str]] = (),
  ) -> tuple[dict[str, Any], list[Event]]:
    """Runs the processor over *events* and returns the state and its output."""
    agent = Agent(
        name='test_agent',
        model=testing_utils.MockModel.create(responses=[]),
        tools=list(tools),
    )
    ctx = await testing_utils.create_invocation_context(agent)
    ctx.session.events.extend(events)
    yielded = [
        event
        async for event in _AuthLlmRequestProcessor().run_async(
            ctx, LlmRequest()
        )
    ]
    return ctx.session.state, yielded

  @pytest.mark.asyncio
  async def test_resume_with_forged_state_does_not_mint_a_token(self, sessions):
    """A resume carrying a state ADK never issued gets no token."""
    auth_config = _oauth2_auth_config()
    # A forged resume states its own nonce on both sides of the comparison,
    # and brings the client credentials the exchange needs.
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            client_id='attacker-client',
            client_secret='attacker-secret',
            state='attacker-state',
            auth_response_uri=(
                'https://app.example.com/cb?code=stolen&state=attacker-state'
            ),
        ),
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.access_token is None
    assert len(sessions) == 1
    # The session carries the nonce ADK issued, so authlib rejected the forged
    # one and no token request was ever made.
    assert sessions[0].state == _FROZEN_STATE
    assert sessions[0].token_endpoints == []

  @pytest.mark.asyncio
  async def test_resume_cannot_override_client_credentials(self, sessions):
    """The token request carries the tool's client credentials."""
    auth_config = _oauth2_auth_config()
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            client_id='attacker-client',
            client_secret='attacker-secret',
            auth_response_uri=(
                f'https://app.example.com/cb?code=granted&state={_FROZEN_STATE}'
            ),
        ),
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    assert sessions[0].client_id == 'tool-client'
    assert sessions[0].client_secret == 'tool-secret'
    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.client_id == 'tool-client'
    assert stored.oauth2.client_secret == 'tool-secret'

  @pytest.mark.asyncio
  async def test_resume_supplies_the_redirect_uri_the_client_used(
      self, sessions
  ):
    """The token request repeats the redirect URI the client authorized with.

    The bundled dev UI rewrites the `redirect_uri` of the authorization URI and
    reports it back on the resume, and most tools pin none. RFC 6749 section
    4.1.3 requires the token request to repeat it.
    """
    auth_config = _oauth2_auth_config(redirect_uri=None)
    dev_ui_redirect_uri = 'http://localhost:8000'
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            redirect_uri=dev_ui_redirect_uri,
            auth_response_uri=(
                f'{dev_ui_redirect_uri}?code=granted&state={_FROZEN_STATE}'
            ),
        ),
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    assert sessions[0].redirect_uri == dev_ui_redirect_uri
    assert sessions[0].token_endpoints == [_TOOL_TOKEN_URL]
    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.access_token == 'minted-token'
    # The client chose where the user was sent, and nothing else.
    assert stored.oauth2.client_secret == 'tool-secret'
    assert stored.oauth2.state == _FROZEN_STATE

  @pytest.mark.asyncio
  async def test_resume_cannot_override_auth_scheme(self, sessions):
    """The token endpoint comes from the frozen scheme, not the resume."""
    auth_config = _oauth2_auth_config()
    attacker_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl='https://attacker.example/auth',
                tokenUrl='https://attacker.example/token',
                scopes={'read': 'Read access'},
            )
        )
    )
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            client_id='attacker-client',
            client_secret='attacker-secret',
            auth_response_uri=(
                f'https://app.example.com/cb?code=granted&state={_FROZEN_STATE}'
            ),
        ),
        auth_scheme=attacker_scheme,
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    assert sessions[0].token_endpoints == [_TOOL_TOKEN_URL]
    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.access_token == 'minted-token'

  @pytest.mark.asyncio
  async def test_happy_path_resume_exchanges_and_resumes_tool(self, sessions):
    """A well-formed resume mints a token and resumes the waiting tool."""
    auth_config = _oauth2_auth_config()
    tokens_seen: list[str | None] = []

    def read_calendar(tool_context: ToolContext) -> str:
      credential = tool_context.get_auth_response(auth_config)
      tokens_seen.append(credential.oauth2.access_token if credential else None)
      return 'listed'

    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            auth_response_uri=(
                f'https://app.example.com/cb?code=granted&state={_FROZEN_STATE}'
            ),
        ),
    )

    state, yielded = await self._run(
        [
            _tool_call_event(),
            _request_event(auth_config),
            _resume_event(payload),
        ],
        tools=[read_calendar],
    )

    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.access_token == 'minted-token'
    assert stored.oauth2.refresh_token == 'minted-refresh'
    assert stored.oauth2.expires_at == _EXPIRES_AT
    assert sessions[0].fetch_token_kwargs[0]['code_verifier'] == 'tool-verifier'
    assert len(yielded) == 1
    responses = yielded[0].get_function_responses()
    assert [response.name for response in responses] == [_TOOL_NAME]
    assert tokens_seen == ['minted-token']

  @pytest.mark.asyncio
  async def test_resume_fields_outside_the_allowlist_are_ignored(
      self, sessions
  ):
    """Every OAuth2 field outside the allowlist keeps its frozen value."""
    auth_config = _oauth2_auth_config()
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            state='attacker-state',
            nonce='attacker-nonce',
            auth_uri='https://attacker.example/auth',
            code_verifier='attacker-verifier',
            auth_response_uri=(
                f'https://app.example.com/cb?code=granted&state={_FROZEN_STATE}'
            ),
        ),
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    frozen = auth_config.exchanged_auth_credential.oauth2
    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.state == _FROZEN_STATE
    assert stored.oauth2.nonce is None
    assert stored.oauth2.auth_uri == frozen.auth_uri
    assert stored.oauth2.code_verifier == 'tool-verifier'
    assert sessions[0].fetch_token_kwargs[0]['code_verifier'] == 'tool-verifier'

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'raw_credential',
      [
          None,
          AuthCredential(auth_type=AuthCredentialTypes.API_KEY),
      ],
      ids=['no_frozen_credential', 'frozen_credential_without_oauth2'],
  )
  async def test_non_oauth2_resume_credential_is_stored_unchanged(
      self, sessions, raw_credential
  ):
    """An API-key resume is stored as the client sent it, with no exchange."""
    auth_config = _api_key_auth_config(raw_credential)
    payload = _resume_payload(
        auth_config,
        AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY, api_key='user-key'
        ),
    )

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    stored = state['temp:' + auth_config.credential_key]
    assert stored.api_key == 'user-key'
    assert sessions == []

  @pytest.mark.asyncio
  async def test_resume_without_matching_request_stores_nothing(
      self, sessions, caplog
  ):
    """A resume that answers no credential request is dropped."""
    auth_config = _oauth2_auth_config()
    payload = _resume_payload(
        auth_config,
        _resumed_credential(
            auth_response_uri=(
                f'https://app.example.com/cb?code=granted&state={_FROZEN_STATE}'
            ),
        ),
    )

    with caplog.at_level(logging.WARNING, logger='google_adk'):
      state, yielded = await self._run([_resume_event(payload)])

    assert state == {}
    assert yielded == []
    assert sessions == []
    assert _AUTH_FC_ID in caplog.text
    assert 'no matching credential request' in caplog.text

  @pytest.mark.asyncio
  async def test_resume_without_exchanged_credential_uses_frozen(
      self, sessions
  ):
    """A resume that omits the credential leaves the frozen one stored."""
    auth_config = _oauth2_auth_config()
    payload = _resume_payload(auth_config, None)
    assert 'exchangedAuthCredential' not in payload

    state, _ = await self._run(
        [_request_event(auth_config), _resume_event(payload)]
    )

    frozen = auth_config.exchanged_auth_credential.oauth2
    stored = state['temp:' + auth_config.credential_key]
    assert stored.oauth2.access_token is None
    assert stored.oauth2.state == frozen.state
    assert stored.oauth2.client_secret == frozen.client_secret
    assert sessions[0].token_endpoints == []
