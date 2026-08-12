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

from __future__ import annotations

import json
import logging
from unittest import mock

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from google.adk.workflow.utils._workflow_hitl_utils import create_auth_request_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_auth_credential
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.adk.workflow.utils._workflow_hitl_utils import process_auth_resume
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from google.genai import types
import pytest

# --- create_request_input_event ---


class TestCreateRequestInputEvent:

  def test_basic_event(self):
    ri = RequestInput(
        interrupt_id="test-id",
        message="Please approve",
    )
    event = create_request_input_event(ri)

    assert event.long_running_tool_ids == {"test-id"}
    assert event.content is not None
    assert event.content.role == "model"
    fc = event.content.parts[0].function_call
    assert fc.name == "adk_request_input"
    assert fc.id == "test-id"
    assert fc.args["message"] == "Please approve"

  def test_with_payload(self):
    ri = RequestInput(
        interrupt_id="id-1",
        payload={"key": "value"},
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    assert fc.args["payload"] == {"key": "value"}

  def test_with_response_schema(self):
    from pydantic import BaseModel

    class MySchema(BaseModel):
      approved: bool

    ri = RequestInput(
        interrupt_id="id-2",
        response_schema=MySchema,
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    schema = fc.args["response_schema"]
    assert "approved" in schema["properties"]
    assert schema["properties"]["approved"]["type"] == "boolean"


# --- has_request_input_function_call ---


class TestHasRequestInputFunctionCall:

  def test_true_for_request_input_event(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert has_request_input_function_call(event) is True

  def test_false_for_empty_event(self):
    assert has_request_input_function_call(Event()) is False

  def test_false_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="other_tool", args={})
                )
            ]
        )
    )
    assert has_request_input_function_call(event) is False


# --- create_request_input_response ---


class TestCreateRequestInputResponse:

  def test_creates_function_response_part(self):
    part = create_request_input_response("id-1", {"approved": True})
    assert part.function_response.id == "id-1"
    assert part.function_response.name == "adk_request_input"
    assert part.function_response.response == {"approved": True}


# --- get_request_input_interrupt_ids ---


class TestGetRequestInputInterruptIds:

  def test_extracts_ids(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert get_request_input_interrupt_ids(event) == ["id-1"]

  def test_empty_for_no_function_calls(self):
    assert get_request_input_interrupt_ids(Event()) == []

  def test_empty_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="other_tool", args={}, id="id-1"
                    )
                )
            ]
        )
    )
    assert get_request_input_interrupt_ids(event) == []


# --- create_auth_request_event ---


class TestCreateAuthRequestEvent:

  def test_creates_credential_request(self):
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key="test_key",
        ),
        credential_key="test_cred",
    )
    event = create_auth_request_event(auth_config, "auth-id-1")

    assert event.long_running_tool_ids is not None
    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
    assert fc.id == "auth-id-1"
    assert "authConfig" in fc.args

  def test_args_are_json_serializable(self):
    from fastapi.openapi.models import OAuth2
    from fastapi.openapi.models import OAuthFlowAuthorizationCode
    from fastapi.openapi.models import OAuthFlows
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_credential import OAuth2Auth
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=OAuth2(
            flows=OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl=(
                        "https://accounts.google.com/o/oauth2/auth"
                    ),
                    tokenUrl="https://oauth2.googleapis.com/token",
                    scopes={
                        "https://www.googleapis.com/auth/calendar": (
                            "See calendars"
                        )
                    },
                )
            )
        ),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.OAUTH2,
            oauth2=OAuth2Auth(
                client_id="oauth_client_id",
                client_secret="oauth_client_secret",
            ),
        ),
    )
    event = create_auth_request_event(auth_config, "auth-id-1")

    fc = event.content.parts[0].function_call

    # python-mode dump leaves auth_scheme.type a live enum, breaking json.dumps
    json.dumps(fc.args)
    assert fc.args["authConfig"]["authScheme"]["type"] == "oauth2"


# --- process_auth_resume / has_auth_credential ---


def _api_key_auth_config(credential_key: str = "node-cred"):
  """An API-key AuthConfig, the simplest resume shape (no token exchange)."""
  from fastapi.openapi.models import APIKey
  from fastapi.openapi.models import APIKeyIn
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_credential import AuthCredentialTypes
  from google.adk.auth.auth_tool import AuthConfig

  return AuthConfig(
      auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY,
          api_key="placeholder",
      ),
      credential_key=credential_key,
  )


def _empty_state():
  from google.adk.sessions.state import State

  return State(value={}, delta={})


class TestProcessAuthResume:

  @pytest.mark.asyncio
  async def test_plain_value_becomes_api_key_credential(self):
    """A bare string resume response is interpreted per the raw credential type."""
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config()
    state = _empty_state()
    assert has_auth_credential(auth_config, state) is False

    await process_auth_resume("user-supplied-key", auth_config, state)

    stored = state["temp:node-cred"]
    assert stored.auth_type == AuthCredentialTypes.API_KEY
    assert stored.api_key == "user-supplied-key"
    assert has_auth_credential(auth_config, state) is True

  @pytest.mark.asyncio
  async def test_auth_config_response_stores_exchanged_credential(self):
    """A full AuthConfig response is accepted and its exchanged credential kept."""
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="from-web-flow",
    )

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
    )

    assert state["temp:node-cred"].api_key == "from-web-flow"

  @pytest.mark.asyncio
  async def test_response_cannot_redirect_storage_to_another_credential_key(
      self,
  ):
    """The node's own credential_key wins over one supplied in the response.

    Otherwise a resume payload could park the credential under a key the node
    never reads, leaving the node permanently unauthenticated.
    """
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes

    auth_config = _api_key_auth_config(credential_key="node-cred")
    state = _empty_state()
    response = _api_key_auth_config(credential_key="unrelated-cred")
    response.exchanged_auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="k",
    )

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
    )

    assert "temp:node-cred" in state
    assert "temp:unrelated-cred" not in state
    assert has_auth_credential(auth_config, state) is True


class TestHasAuthCredential:

  @pytest.mark.asyncio
  async def test_false_for_a_different_credential_key(self):
    """Credentials are looked up per credential_key, not shared across configs."""

    auth_config = _api_key_auth_config(credential_key="node-cred")
    other_config = _api_key_auth_config(credential_key="other-cred")
    state = _empty_state()

    await process_auth_resume("key", auth_config, state)

    assert has_auth_credential(auth_config, state) is True
    assert has_auth_credential(other_config, state) is False


# --- process_auth_resume: the resume payload cannot pick the auth scheme ---

_FROZEN_TOKEN_URL = "https://provider.example/token"
_ATTACKER_TOKEN_URL = "https://evil.example/token"


def _oauth2_auth_config(credential_key: str = "oauth-cred") -> AuthConfig:
  """An OAuth2 AuthConfig whose token exchange targets a known endpoint."""
  return AuthConfig(
      auth_scheme=OAuth2(
          flows=OAuthFlows(
              authorizationCode=OAuthFlowAuthorizationCode(
                  authorizationUrl="https://provider.example/auth",
                  tokenUrl=_FROZEN_TOKEN_URL,
                  scopes={"read": "Read access"},
              )
          )
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              client_id="oauth_client_id",
              client_secret="oauth_client_secret",
          ),
      ),
      credential_key=credential_key,
  )


def _oidc_auth_config(credential_key: str = "oidc-cred") -> AuthConfig:
  """An OpenID Connect AuthConfig, the non-fastapi member of AuthScheme."""
  return AuthConfig(
      auth_scheme=OpenIdConnectWithConfig(
          authorization_endpoint="https://provider.example/auth",
          token_endpoint=_FROZEN_TOKEN_URL,
          scopes=["read"],
      ),
      raw_auth_credential=AuthCredential(
          auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
          oauth2=OAuth2Auth(
              client_id="oauth_client_id",
              client_secret="oauth_client_secret",
          ),
      ),
      credential_key=credential_key,
  )


def _authorization_code_credential(
    auth_type: AuthCredentialTypes = AuthCredentialTypes.OAUTH2,
) -> AuthCredential:
  """The credential an honest client returns after the consent redirect.

  ADK builds it by copying the node's raw credential, so an honest echo
  always carries the node's own ``auth_type``.
  """
  return AuthCredential(
      auth_type=auth_type,
      oauth2=OAuth2Auth(
          client_id="oauth_client_id",
          client_secret="oauth_client_secret",
          auth_code="the-authorization-code",
          auth_response_uri=(
              "https://app.example/callback?code=the-authorization-code"
          ),
      ),
  )


@pytest.fixture(name="oauth2_session")
def oauth2_session_fixture():
  """Patches the only HTTP boundary the token exchange goes through."""
  with mock.patch(
      "google.adk.auth.oauth2_credential_util.OAuth2Session"
  ) as session:
    session.return_value.fetch_token.return_value = {
        "access_token": "at",
        "expires_in": 3600,
    }
    yield session


def _fetched_token_endpoint(oauth2_session: mock.MagicMock) -> str:
  """The endpoint the exchange actually posted the client secret to."""
  oauth2_session.return_value.fetch_token.assert_called_once()
  return oauth2_session.return_value.fetch_token.call_args.args[0]


class TestProcessAuthResumeSchemeBinding:
  """The node's own auth_config, not the payload, drives storage and exchange.

  The payload contributes the credential and nothing else, and that credential
  is refused when its type is not the one the node asked for.
  """

  @pytest.mark.asyncio
  async def test_refuses_an_oauth2_upgrade_of_an_api_key_node(
      self, oauth2_session: mock.MagicMock
  ):
    """An apiKey node must never be talked into a token exchange."""
    auth_config = _api_key_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = _authorization_code_credential()
    payload = response.model_dump(mode="json", exclude_none=True, by_alias=True)
    payload["authScheme"] = _oauth2_auth_config().auth_scheme.model_dump(
        mode="json", exclude_none=True, by_alias=True
    )
    payload["authScheme"]["flows"]["authorizationCode"][
        "tokenUrl"
    ] = _ATTACKER_TOKEN_URL

    await process_auth_resume(payload, auth_config, state)

    assert "temp:node-cred" not in state
    assert has_auth_credential(auth_config, state) is False
    assert oauth2_session.call_count == 0

  @pytest.mark.asyncio
  async def test_a_redirected_token_endpoint_in_the_payload_is_ignored(
      self, oauth2_session: mock.MagicMock
  ):
    """A client-edited tokenUrl must not receive the client secret."""
    auth_config = _oauth2_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = _authorization_code_credential()
    payload = response.model_dump(mode="json", exclude_none=True, by_alias=True)
    payload["authScheme"]["flows"]["authorizationCode"][
        "tokenUrl"
    ] = _ATTACKER_TOKEN_URL

    await process_auth_resume(payload, auth_config, state)

    assert _fetched_token_endpoint(oauth2_session) == _FROZEN_TOKEN_URL
    assert state["temp:oauth-cred"].oauth2.access_token == "at"

  @pytest.mark.asyncio
  async def test_refuses_an_api_key_credential_for_an_oauth2_node(
      self, oauth2_session: mock.MagicMock, caplog: pytest.LogCaptureFixture
  ):
    """A downgraded credential must not be stored unexchanged."""
    auth_config = _oauth2_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    payload = response.model_dump(mode="json", exclude_none=True, by_alias=True)
    payload["authScheme"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Api-Key",
    }
    payload["exchangedAuthCredential"] = {
        "authType": "apiKey",
        "apiKey": "attacker-key",
    }

    with caplog.at_level(logging.WARNING):
      await process_auth_resume(payload, auth_config, state)

    assert "temp:oauth-cred" not in state
    assert has_auth_credential(auth_config, state) is False
    assert oauth2_session.call_count == 0
    assert "attacker-key" not in str(state.to_dict())
    assert "oauth-cred" in caplog.text
    assert "attacker-key" not in caplog.text

  @pytest.mark.asyncio
  async def test_refuses_a_bare_api_key_credential_for_an_oauth2_node(
      self, oauth2_session: mock.MagicMock
  ):
    """The same downgrade, with the auth_scheme simply left out.

    ``AuthConfig`` validation fails without an ``auth_scheme``, so this
    payload takes the credential branch. Both branches meet the same check.
    """
    auth_config = _oauth2_auth_config()
    state = _empty_state()

    await process_auth_resume(
        {"authType": "apiKey", "apiKey": "attacker-key"}, auth_config, state
    )

    assert "temp:oauth-cred" not in state
    assert has_auth_credential(auth_config, state) is False
    assert oauth2_session.call_count == 0
    assert "attacker-key" not in str(state.to_dict())

  @pytest.mark.asyncio
  async def test_oauth2_echo_exchanges_against_the_requested_endpoint(
      self, oauth2_session: mock.MagicMock
  ):
    """The honest dev UI payload is accepted and hits the frozen endpoint."""
    auth_config = _oauth2_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = _authorization_code_credential()

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
    )

    assert _fetched_token_endpoint(oauth2_session) == _FROZEN_TOKEN_URL
    assert state["temp:oauth-cred"].oauth2.access_token == "at"
    assert has_auth_credential(auth_config, state) is True

  @pytest.mark.asyncio
  async def test_payload_with_explicit_nulls_is_accepted(
      self, oauth2_session: mock.MagicMock
  ):
    """A client that echoes unset scheme fields back as nulls still resumes."""
    auth_config = _oauth2_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = _authorization_code_credential()

    await process_auth_resume(
        response.model_dump(mode="json", by_alias=True),
        auth_config,
        state,
    )

    assert _fetched_token_endpoint(oauth2_session) == _FROZEN_TOKEN_URL
    assert state["temp:oauth-cred"].oauth2.access_token == "at"

  @pytest.mark.asyncio
  async def test_openid_connect_round_trip_is_accepted(
      self, oauth2_session: mock.MagicMock
  ):
    """The OpenIdConnectWithConfig member of the union round-trips too."""
    auth_config = _oidc_auth_config()
    state = _empty_state()
    response = auth_config.model_copy(deep=True)
    response.exchanged_auth_credential = _authorization_code_credential(
        AuthCredentialTypes.OPEN_ID_CONNECT
    )

    await process_auth_resume(
        response.model_dump(mode="json", exclude_none=True, by_alias=True),
        auth_config,
        state,
    )

    assert _fetched_token_endpoint(oauth2_session) == _FROZEN_TOKEN_URL
    assert state["temp:oidc-cred"].oauth2.access_token == "at"

  @pytest.mark.asyncio
  async def test_credential_shaped_payload_still_uses_the_frozen_scheme(
      self, oauth2_session: mock.MagicMock
  ):
    """A bare AuthCredential of the requested type is accepted.

    It carries no auth_scheme, and the exchange runs against the frozen
    config.
    """
    auth_config = _oauth2_auth_config()
    state = _empty_state()

    await process_auth_resume(
        _authorization_code_credential().model_dump(
            mode="json", exclude_none=True, by_alias=True
        ),
        auth_config,
        state,
    )

    assert _fetched_token_endpoint(oauth2_session) == _FROZEN_TOKEN_URL
    assert state["temp:oauth-cred"].oauth2.access_token == "at"
