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

from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.openapi_tool.auth.auth_helpers import openid_dict_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.openapi_spec_parser import tool_auth_handler
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolAuthHandler
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import ToolContextCredentialStore
from google.adk.tools.tool_context import ToolContext
import pytest


# Helper function to create a mock ToolContext
def create_mock_tool_context(session: Optional[Session] = None):
  return ToolContext(
      function_call_id='test-fc-id',
      invocation_context=InvocationContext(
          agent=LlmAgent(name='test'),
          session=session or Session(app_name='test', user_id='123', id='123'),
          invocation_id='123',
          session_service=InMemorySessionService(),
      ),
  )


# Test cases for OpenID Connect
class MockOpenIdConnectCredentialExchanger(OAuth2CredentialExchanger):

  def __init__(
      self, expected_scheme, expected_credential, expected_access_token
  ):
    self.expected_scheme = expected_scheme
    self.expected_credential = expected_credential
    self.expected_access_token = expected_access_token

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    if auth_credential.oauth2 and (
        auth_credential.oauth2.auth_response_uri
        or auth_credential.oauth2.auth_code
    ):
      auth_code = (
          auth_credential.oauth2.auth_response_uri
          if auth_credential.oauth2.auth_response_uri
          else auth_credential.oauth2.auth_code
      )
      # Simulate the token exchange
      updated_credential = AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
          http=HttpAuth(
              scheme='bearer',
              credentials=HttpCredentials(
                  token=auth_code + self.expected_access_token
              ),
          ),
      )
      return updated_credential

    # simulate the case of getting auth_uri
    return None


def get_mock_openid_scheme_credential():
  config_dict = {
      'authorization_endpoint': 'test.com',
      'token_endpoint': 'test.com',
  }
  scopes = ['test_scope']
  credential_dict = {
      'client_id': '123',
      'client_secret': '456',
      'redirect_uri': 'test.com',
  }
  return openid_dict_to_scheme_credential(config_dict, scopes, credential_dict)


# Fixture for the OpenID Connect security scheme
@pytest.fixture
def openid_connect_scheme():
  scheme, _ = get_mock_openid_scheme_credential()
  return scheme


# Fixture for a base OpenID Connect credential
@pytest.fixture
def openid_connect_credential():
  _, credential = get_mock_openid_scheme_credential()
  return credential


@pytest.mark.asyncio
async def test_openid_connect_no_auth_response(
    openid_connect_scheme, openid_connect_credential
):
  # Setup Mock exchanger
  mock_exchanger = MockOpenIdConnectCredentialExchanger(
      openid_connect_scheme, openid_connect_credential, None
  )
  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_exchanger=mock_exchanger,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'pending'
  assert result.auth_credential == openid_connect_credential


@pytest.mark.asyncio
async def test_openid_connect_uses_explicit_credential_key(
    openid_connect_scheme, openid_connect_credential
):
  tool_context = create_mock_tool_context()
  handler = ToolAuthHandler.from_tool_context(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_key='my_tool_tokens',
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'pending'
  requested = tool_context.actions.requested_auth_configs['test-fc-id']
  assert requested.credential_key == 'my_tool_tokens'
  # A pending request must not create a cache slot.
  assert not tool_context.state.to_dict()


@pytest.mark.asyncio
async def test_exchanged_credential_is_cached_under_the_configured_key():
  api_key_scheme, api_key_credential = token_to_scheme_credential(
      'apikey', 'header', 'X-API-Key', 'test_api_key'
  )
  tool_context = create_mock_tool_context()
  tool_context.state['temp:my_tool_tokens'] = api_key_credential.model_dump(
      exclude_none=True
  )

  handler = ToolAuthHandler.from_tool_context(
      tool_context,
      api_key_scheme,
      None,
      credential_key='my_tool_tokens',
  )
  result = await handler.prepare_auth_credentials()

  assert result.state == 'done'
  cached = AuthCredential.model_validate(tool_context.state['my_tool_tokens'])
  assert cached.api_key == 'test_api_key'
  # The named slot is the only cache slot: no digest-derived key was written.
  assert set(tool_context.state.to_dict()) == {
      'temp:my_tool_tokens',
      'my_tool_tokens',
  }


@pytest.mark.asyncio
async def test_second_handler_with_same_key_reads_back_across_schemes(
    openid_connect_scheme,
):
  session = Session(app_name='test', user_id='123', id='123')
  api_key_scheme, api_key_credential = token_to_scheme_credential(
      'apikey', 'header', 'X-API-Key', 'shared_api_key'
  )
  first_context = create_mock_tool_context(session)
  first_context.state['temp:shared_tool_tokens'] = (
      api_key_credential.model_dump(exclude_none=True)
  )
  first_handler = ToolAuthHandler.from_tool_context(
      first_context,
      api_key_scheme,
      None,
      credential_key='shared_tool_tokens',
  )
  await first_handler.prepare_auth_credentials()

  second_context = create_mock_tool_context(session)
  second_handler = ToolAuthHandler.from_tool_context(
      second_context,
      openid_connect_scheme,
      None,
      credential_key='shared_tool_tokens',
  )
  result = await second_handler.prepare_auth_credentials()

  assert result.state == 'done'
  assert result.auth_credential.api_key == 'shared_api_key'
  cache_slots = [key for key in session.state if not key.startswith('temp:')]
  assert cache_slots == ['shared_tool_tokens']


@pytest.mark.asyncio
async def test_two_credential_keys_get_two_cache_slots():
  session = Session(app_name='test', user_id='123', id='123')
  api_key_scheme, _ = token_to_scheme_credential(
      'apikey', 'header', 'X-API-Key', 'unused'
  )
  keys_to_api_keys = {
      'tool_a_tokens': 'api_key_a',
      'tool_b_tokens': 'api_key_b',
  }

  for credential_key, api_key in keys_to_api_keys.items():
    tool_context = create_mock_tool_context(session)
    tool_context.state[f'temp:{credential_key}'] = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY, api_key=api_key
    ).model_dump(exclude_none=True)
    handler = ToolAuthHandler.from_tool_context(
        tool_context,
        api_key_scheme,
        None,
        credential_key=credential_key,
    )
    assert (await handler.prepare_auth_credentials()).state == 'done'

  for credential_key, api_key in keys_to_api_keys.items():
    cached = AuthCredential.model_validate(session.state[credential_key])
    assert cached.api_key == api_key


@pytest.mark.asyncio
async def test_credential_key_on_the_credential_selects_the_cache_slot(
    openid_connect_scheme,
):
  _, openid_credential = get_mock_openid_scheme_credential()
  openid_credential = openid_credential.model_copy(
      update={'credential_key': 'extra_named_slot'}
  )
  tool_context = create_mock_tool_context()
  tool_context.state['temp:extra_named_slot'] = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key='extra_api_key'
  ).model_dump(exclude_none=True)

  handler = ToolAuthHandler.from_tool_context(
      tool_context, openid_connect_scheme, openid_credential
  )
  result = await handler.prepare_auth_credentials()

  assert result.state == 'done'
  cached = AuthCredential.model_validate(tool_context.state['extra_named_slot'])
  assert cached.api_key == 'extra_api_key'


@pytest.mark.asyncio
async def test_credential_key_on_the_scheme_selects_the_cache_slot():
  api_key_scheme, _ = token_to_scheme_credential(
      'apikey', 'header', 'X-API-Key', 'unused'
  )
  api_key_scheme = api_key_scheme.model_copy(
      update={'credentialKey': 'scheme_named_slot'}
  )
  tool_context = create_mock_tool_context()
  tool_context.state['temp:scheme_named_slot'] = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key='scheme_api_key'
  ).model_dump(exclude_none=True)

  handler = ToolAuthHandler.from_tool_context(
      tool_context, api_key_scheme, None
  )
  result = await handler.prepare_auth_credentials()

  assert result.state == 'done'
  cached = AuthCredential.model_validate(
      tool_context.state['scheme_named_slot']
  )
  assert cached.api_key == 'scheme_api_key'


def test_empty_credential_key_falls_back_to_the_derived_key(
    openid_connect_scheme, openid_connect_credential
):
  tool_context = create_mock_tool_context()
  derived_key = ToolContextCredentialStore(tool_context).get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )

  empty_key_store = ToolContextCredentialStore(tool_context, credential_key='')

  assert (
      empty_key_store.get_credential_key(
          openid_connect_scheme, openid_connect_credential
      )
      == derived_key
  )


def test_configured_key_does_not_adopt_a_legacy_keyed_credential(
    openid_connect_scheme, openid_connect_credential
):
  _, legacy_credential = token_to_scheme_credential(
      'oauth2Token', 'header', 'bearer', '123123123'
  )
  tool_context = create_mock_tool_context()
  legacy_key = ToolContextCredentialStore(
      tool_context
  )._get_legacy_credential_key(openid_connect_scheme, openid_connect_credential)
  tool_context.state[legacy_key] = legacy_credential.model_dump(
      exclude_none=True
  )

  named_store = ToolContextCredentialStore(
      tool_context, credential_key='named_slot'
  )

  assert (
      named_store.get_credential(
          openid_connect_scheme, openid_connect_credential
      )
      is None
  )
  assert 'named_slot' not in tool_context.state


def test_legacy_keyed_credential_is_migrated_when_no_key_is_configured(
    openid_connect_scheme, openid_connect_credential
):
  _, legacy_credential = token_to_scheme_credential(
      'oauth2Token', 'header', 'bearer', '123123123'
  )
  tool_context = create_mock_tool_context()
  store = ToolContextCredentialStore(tool_context)
  legacy_key = store._get_legacy_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  current_key = store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  assert legacy_key != current_key
  tool_context.state[legacy_key] = legacy_credential.model_dump(
      exclude_none=True
  )

  migrated = store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )

  assert migrated == legacy_credential
  assert (
      AuthCredential.model_validate(tool_context.state[current_key])
      == legacy_credential
  )


@pytest.mark.asyncio
async def test_openid_connect_with_auth_response(
    openid_connect_scheme, openid_connect_credential, monkeypatch
):
  mock_exchanger = MockOpenIdConnectCredentialExchanger(
      openid_connect_scheme,
      openid_connect_credential,
      'test_access_token',
  )
  tool_context = create_mock_tool_context()

  mock_auth_handler = MagicMock()
  returned_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(auth_response_uri='test_auth_response_uri'),
  )
  mock_auth_handler.get_auth_response.return_value = returned_credential
  mock_auth_handler_path = 'google.adk.auth.auth_handler.AuthHandler'
  monkeypatch.setattr(
      mock_auth_handler_path, lambda *args, **kwargs: mock_auth_handler
  )

  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_exchanger=mock_exchanger,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'done'
  assert result.auth_credential.auth_type == AuthCredentialTypes.HTTP
  assert 'test_access_token' in result.auth_credential.http.credentials.token
  # Verify that the credential was stored:
  stored_credential = credential_store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert stored_credential == returned_credential
  mock_auth_handler.get_auth_response.assert_called_once()


@pytest.mark.asyncio
async def test_openid_connect_existing_token(
    openid_connect_scheme, openid_connect_credential
):
  _, existing_credential = token_to_scheme_credential(
      'oauth2Token', 'header', 'bearer', '123123123'
  )
  tool_context = create_mock_tool_context()
  # Store the credential to simulate existing credential
  credential_store = ToolContextCredentialStore(tool_context=tool_context)
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )
  result = await handler.prepare_auth_credentials()
  assert result.state == 'done'
  assert result.auth_credential == existing_credential


@patch.object(tool_auth_handler, 'OAuth2CredentialRefresher')
@pytest.mark.asyncio
async def test_openid_connect_existing_oauth2_token_refresh(
    mock_oauth2_refresher, openid_connect_scheme, openid_connect_credential
):
  """Test that OAuth2 tokens are refreshed when existing credentials are found."""
  # Create existing OAuth2 credential
  existing_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='existing_token',
          refresh_token='refresh_token',
      ),
  )

  # Mock the refreshed credential
  refreshed_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='refreshed_token',
          refresh_token='new_refresh_token',
      ),
  )

  # Setup mock OAuth2CredentialRefresher
  from unittest.mock import AsyncMock

  mock_refresher_instance = MagicMock()
  mock_refresher_instance.is_refresh_needed = AsyncMock(return_value=True)
  mock_refresher_instance.refresh = AsyncMock(return_value=refreshed_credential)
  mock_oauth2_refresher.return_value = mock_refresher_instance

  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)

  # Store the existing credential
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )

  result = await handler.prepare_auth_credentials()

  # Verify OAuth2CredentialRefresher was called for refresh
  mock_oauth2_refresher.assert_called_once()

  mock_refresher_instance.is_refresh_needed.assert_called_once_with(
      existing_credential
  )
  mock_refresher_instance.refresh.assert_called_once_with(
      existing_credential, openid_connect_scheme
  )

  assert result.state == 'done'
  # The result should contain the refreshed credential after exchange
  assert result.auth_credential is not None


@patch.object(tool_auth_handler, 'OAuth2CredentialRefresher')
@pytest.mark.asyncio
async def test_refreshed_credential_is_persisted_to_store(
    mock_oauth2_refresher, openid_connect_scheme, openid_connect_credential
):
  """Test that refreshed OAuth2 credentials are persisted back to the store."""
  # Create existing OAuth2 credential with an "old" refresh token.
  existing_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='old_access_token',
          refresh_token='old_refresh_token',
      ),
  )

  # The refresher will return a credential with rotated tokens.
  refreshed_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id='test_client_id',
          client_secret='test_client_secret',
          access_token='new_access_token',
          refresh_token='new_refresh_token',
      ),
  )

  mock_refresher_instance = MagicMock()
  mock_refresher_instance.is_refresh_needed = AsyncMock(return_value=True)
  mock_refresher_instance.refresh = AsyncMock(return_value=refreshed_credential)
  mock_oauth2_refresher.return_value = mock_refresher_instance

  tool_context = create_mock_tool_context()
  credential_store = ToolContextCredentialStore(tool_context=tool_context)

  # Store the existing (stale) credential.
  key = credential_store.get_credential_key(
      openid_connect_scheme, openid_connect_credential
  )
  credential_store.store_credential(key, existing_credential)

  handler = ToolAuthHandler(
      tool_context,
      openid_connect_scheme,
      openid_connect_credential,
      credential_store=credential_store,
  )

  await handler.prepare_auth_credentials()

  # The critical assertion: the *refreshed* credential must now be in the
  # store so that the next invocation reads the new tokens, not the old ones.
  persisted = credential_store.get_credential(
      openid_connect_scheme, openid_connect_credential
  )
  assert persisted is not None
  assert persisted.oauth2.access_token == 'new_access_token'
  assert persisted.oauth2.refresh_token == 'new_refresh_token'


def test_credential_key_is_stable_across_redirect_uri():
  """get_credential_key should be invariant under redirect_uri changes.

  redirect_uri is deployment configuration (which callback URL the auth
  server should redirect to), not part of the credential identity. Two
  AuthCredential instances that share the same client_id, client_secret,
  and scopes but differ only in redirect_uri should produce the same key.
  """
  scheme, _ = get_mock_openid_scheme_credential()
  credential_local = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='http://localhost:8001/oauth2callback',
      ),
  )
  credential_deployed = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='https://deployed.example.com/oauth2callback',
      ),
  )
  store = ToolContextCredentialStore(tool_context=create_mock_tool_context())

  assert store.get_credential_key(
      scheme, credential_local
  ) == store.get_credential_key(scheme, credential_deployed)


def test_legacy_credential_key_is_stable_across_redirect_uri():
  """_get_legacy_credential_key should be invariant under redirect_uri changes.

  The same redirect_uri-strip behavior must apply to the legacy key path so
  that already-stored credentials remain findable after the fix.
  """
  scheme, _ = get_mock_openid_scheme_credential()
  credential_local = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='http://localhost:8001/oauth2callback',
      ),
  )
  credential_deployed = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='client',
          client_secret='secret',
          redirect_uri='https://deployed.example.com/oauth2callback',
      ),
  )
  store = ToolContextCredentialStore(tool_context=create_mock_tool_context())

  assert store._get_legacy_credential_key(
      scheme, credential_local
  ) == store._get_legacy_credential_key(scheme, credential_deployed)
