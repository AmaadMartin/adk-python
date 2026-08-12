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

import logging
from typing import Any
from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..agents.readonly_context import ReadonlyContext
from ..events.event import Event
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.functions import handle_function_calls_async
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..models.llm_request import LlmRequest
from ..sessions.state import State
from .auth_credential import AuthCredential
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments

logger = logging.getLogger("google_adk." + __name__)

# Prefix used by toolset auth credential IDs.
# Auth requests with this prefix are for toolset authentication (before tool
# listing) and don't require resuming a function call.
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = "_adk_toolset_auth_"

# The OAuth2 fields a resume message may contribute. Everything else on
# `oauth2` -- the client credentials, the PKCE verifier and the CSRF state --
# belongs to the tool or to ADK, so it is read back from the frozen request. A
# field added to `OAuth2Auth` later stays frozen-sourced until it is named
# here.
#
# `redirect_uri` is resumable because the client, not the tool, decides where
# the authorization server sends the user. The bundled dev UI rewrites the
# `redirect_uri` of the authorization URI before it opens the consent popup,
# and most tools set none at all. RFC 6749 section 4.1.3 requires the token
# request to repeat that same value, so the client has to report it back. This
# grants the client nothing: ADK still sends the request to the frozen token
# endpoint and reads the token out of the response.
_RESUMABLE_OAUTH2_FIELDS = (
    "auth_response_uri",
    "auth_code",
    "redirect_uri",
    "access_token",
    "refresh_token",
    "id_token",
    "expires_in",
    "expires_at",
)


def _bind_oauth2_to_request(
    requested_auth_config: AuthConfig,
    response_credential: AuthCredential | None,
) -> AuthCredential | None:
  """Rebuilds the credential to store from the frozen request.

  Only the fields a resume message is allowed to contribute are overlaid onto
  the frozen OAuth2 credential, so the state ADK issued is the one compared
  against the state returned by the authorization server.

  Args:
    requested_auth_config: The frozen ``AuthConfig`` recovered from the
      ``adk_request_credential`` function call.
    response_credential: The credential from the client's resume message.

  Returns:
    The credential to store, or *response_credential* unchanged when the frozen
    request carries no OAuth2 credential (API key, HTTP and service-account
    schemes).
  """
  frozen = (
      requested_auth_config.exchanged_auth_credential
      or requested_auth_config.raw_auth_credential
  )
  if frozen is None:
    return response_credential
  bound = frozen.model_copy(deep=True)
  if bound.oauth2 is None:
    return response_credential
  resumed = response_credential.oauth2 if response_credential else None
  if resumed is not None:
    for field in _RESUMABLE_OAUTH2_FIELDS:
      value = getattr(resumed, field)
      if value is not None:
        setattr(bound.oauth2, field, value)
  return bound


async def _store_auth_and_collect_resume_targets(
    events: list[Event],
    auth_fc_ids: set[str],
    auth_responses: dict[str, Any],
    state: State,
) -> set[str]:
  """Store auth credentials and return original function call IDs to resume.

  Scans session events for the ``adk_request_credential`` function calls whose
  IDs are in *auth_fc_ids* and reads back the frozen ``AuthConfig`` from their
  ``AuthToolArguments`` args. Each credential is stored via ``AuthHandler``
  under that frozen config, with only the resumable OAuth2 fields taken from
  the client. A response with no matching request is ignored.

  Args:
    events: Session events to scan. Trusted.
    auth_fc_ids: IDs of ``adk_request_credential`` function calls to match.
    auth_responses: Mapping of FC ID -> auth config response dict from the
      client. Untrusted.
    state: Session state for temporary credential storage.

  Returns:
    Set of original function call IDs to resume, excluding toolset auth.
  """
  # Step 1: Scan events for matching adk_request_credential function calls
  # to extract the frozen AuthConfig from their AuthToolArguments.
  requested_auth_config_by_id: dict[str, AuthConfig] = {}
  for event in events:
    event_function_calls = event.get_function_calls()
    if not event_function_calls:
      continue
    try:
      for function_call in event_function_calls:
        if (
            function_call.id in auth_fc_ids
            and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
        ):
          args = AuthToolArguments.model_validate(function_call.args)
          requested_auth_config_by_id[function_call.id] = args.auth_config
    except TypeError:
      continue

  # Step 2: Store credentials. The frozen request decides how the credential
  # is stored and exchanged; the resume message may only carry what the user
  # just obtained from the identity provider.
  for fc_id in auth_fc_ids:
    if fc_id not in auth_responses:
      continue
    requested_auth_config = requested_auth_config_by_id.get(fc_id)
    if not requested_auth_config:
      logger.warning(
          "Ignoring auth response for %s: no matching credential request.",
          fc_id,
      )
      continue
    response_config = AuthConfig.model_validate(auth_responses[fc_id])
    auth_config = requested_auth_config.model_copy(deep=True)
    auth_config.exchanged_auth_credential = _bind_oauth2_to_request(
        requested_auth_config, response_config.exchanged_auth_credential
    )
    await AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
        state=state
    )

  # Step 3: Collect original function call IDs to resume, skipping
  # toolset auth entries which don't map to a resumable function call.
  tools_to_resume: set[str] = set()
  for fc_id in auth_fc_ids:
    requested_auth_config = requested_auth_config_by_id.get(fc_id)
    if not requested_auth_config:
      continue
    # Re-parse to get function_call_id (AuthConfig doesn't carry it;
    # AuthToolArguments does).
    for event in events:
      event_function_calls = event.get_function_calls()
      if not event_function_calls:
        continue
      for function_call in event_function_calls:
        if (
            function_call.id == fc_id
            and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
        ):
          args = AuthToolArguments.model_validate(function_call.args)
          if args.function_call_id.startswith(
              TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
          ):
            continue
          tools_to_resume.add(args.function_call_id)

  return tools_to_resume


class _AuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles auth information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    if agent is None or not hasattr(agent, "canonical_tools"):
      return
    events = invocation_context._get_events(current_branch=True)
    if not events:
      return

    # Find the last user-authored event with function responses to
    # identify adk_request_credential responses.
    last_event_with_content = None
    for i in range(len(events) - 1, -1, -1):
      event = events[i]
      if event.content is not None:
        last_event_with_content = event
        break

    if not last_event_with_content or last_event_with_content.author != "user":
      return

    responses = last_event_with_content.get_function_responses()
    if not responses:
      return

    # Collect adk_request_credential function response IDs and their
    # response dicts.
    auth_fc_ids: set[str] = set()
    auth_responses: dict[str, Any] = {}
    for function_call_response in responses:
      if function_call_response.name != REQUEST_EUC_FUNCTION_CALL_NAME:
        continue
      auth_fc_ids.add(function_call_response.id)
      auth_responses[function_call_response.id] = (
          function_call_response.response
      )

    if not auth_fc_ids:
      return

    # Store credentials and collect tools to resume.
    tools_to_resume = await _store_auth_and_collect_resume_targets(
        events, auth_fc_ids, auth_responses, invocation_context.session.state
    )

    if not tools_to_resume:
      return

    # Find the original function call event and re-execute the tools
    # that needed auth.
    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      if any([
          function_call.id in tools_to_resume
          for function_call in function_calls
      ]):
        if function_response_event := await handle_function_calls_async(
            invocation_context,
            event,
            {
                tool.name: tool
                for tool in await agent.canonical_tools(
                    ReadonlyContext(invocation_context)
                )
            },
            tools_to_resume,
        ):
          yield function_response_event
        return
    return


request_processor = _AuthLlmRequestProcessor()
