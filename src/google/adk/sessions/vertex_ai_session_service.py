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

import asyncio
from collections.abc import Mapping
import copy
import datetime
import json
import logging
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from google.genai.errors import ClientError
import pydantic
from typing_extensions import override

if TYPE_CHECKING:
  import vertexai

from . import _session_util
from ..events.event import Event
from ..events.event_actions import EventActions
from ..events.event_actions import EventCompaction
from ..utils.vertex_ai_utils import get_express_mode_api_key
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session

logger = logging.getLogger('google_adk.' + __name__)

_COMPACTION_CUSTOM_METADATA_KEY = '_compaction'
_USAGE_METADATA_CUSTOM_METADATA_KEY = '_usage_metadata'

_MILLIS_PER_SECOND = 1000.0
# Compaction field spellings accepted on read, most canonical first. The last
# entry of each timestamp tuple is the legacy adk-js spelling.
_COMPACTION_START_KEYS = ('start_timestamp', 'startTimestamp', 'startTime')
_COMPACTION_END_KEYS = ('end_timestamp', 'endTimestamp', 'endTime')
_COMPACTION_CONTENT_KEYS = ('compacted_content', 'compactedContent')
# A timestamp read under one of these keys is epoch milliseconds, because
# adk-js builds it from `Date.now()`. Every other spelling is epoch seconds.
_LEGACY_MILLIS_TIMESTAMP_KEYS = frozenset({'startTime', 'endTime'})
# adk-js JSON round-trips its whole compacted event into `raw_event`, so these
# compaction fields also arrive at the top level of the raw event.
_LEGACY_RAW_EVENT_COMPACTION_KEYS = (
    'isCompacted',
    'startTime',
    'endTime',
    'compactedContent',
)

_SESSION_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')


def _extract_short_session_id(
    session_id: str, expected_engine_id: str | None = None
) -> str:
  """Extracts the short session ID if a full resource name is provided."""
  if isinstance(session_id, str) and '/' in session_id:
    parts = session_id.split('/')
    if len(parts) >= 2 and parts[-2] == 'sessions':
      if (
          len(parts) >= 4
          and parts[-4] == 'reasoningEngines'
          and expected_engine_id
      ):
        passed_engine_id = parts[-3]
        if passed_engine_id != expected_engine_id:
          raise ValueError(
              'Session resource name mismatch: session belongs to '
              f'reasoningEngine {passed_engine_id!r}, but service is '
              f'configured for {expected_engine_id!r}.'
          )
      return parts[-1]
  return session_id


def _validate_session_id(session_id: str) -> None:
  """Rejects session IDs that could escape the URL path segment."""
  if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
      session_id
  ):
    raise ValueError(
        f'Invalid session_id {session_id!r}: must match'
        f' {_SESSION_ID_PATTERN.pattern}.'
    )


def _quote_filter_literal(value: str) -> str:
  """Quotes filter values so embedded metacharacters stay inside the literal."""
  escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
  return f'"{escaped_value}"'


def _set_internal_custom_metadata(
    metadata_dict: dict[str, Any], *, key: str, value: dict[str, Any]
) -> None:
  """Stores internal metadata alongside user-provided custom metadata."""
  existing_custom_metadata = metadata_dict.get('custom_metadata') or {}
  metadata_dict['custom_metadata'] = {
      **existing_custom_metadata,
      key: value,
  }


def _drop_vertex_unsupported_part_fields(content_dict: dict[str, Any]) -> None:
  """Drops Part fields the Vertex AI Agent Engine Sessions API rejects.

  ``part_metadata`` is a Gemini Developer API-only field (the model path guards
  it in ``genai`` ``_Part_to_vertex``); the Agent Engine Sessions API does not
  accept it and fails ``appendEvent`` with ``400 INVALID_ARGUMENT`` ("Unknown
  name \"part_metadata\" at 'event.content.parts[0]'"). Mutates the serialized
  content dict in place; tolerant of either field-name or alias serialization.
  """
  # TODO: remove once the Agent Engine Sessions API accepts part_metadata.
  for part in content_dict.get('parts') or []:
    if isinstance(part, dict):
      part.pop('part_metadata', None)
      part.pop('partMetadata', None)


class VertexAiSessionService(BaseSessionService):
  """Connects to the Vertex AI Agent Engine Session Service using Agent Engine SDK.

  https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview
  """

  def __init__(
      self,
      project: Optional[str] = None,
      location: Optional[str] = None,
      agent_engine_id: Optional[str] = None,
      *,
      express_mode_api_key: Optional[str] = None,
  ):
    """Initializes the VertexAiSessionService.

    Args:
      project: The project id of the project to use.
      location: The location of the project to use.
      agent_engine_id: The resource ID of the agent engine to use.
      express_mode_api_key: The API key to use for Express Mode. If not
        provided, the API key from the GOOGLE_API_KEY environment variable will
        be used. It will only be used if GOOGLE_GENAI_USE_ENTERPRISE is true. Do
        not use Google AI Studio API key for this field. For more details, visit
        https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
    """
    try:
      import vertexai  # noqa: F401
    except ImportError as e:
      from ..utils._dependency import missing_extra

      raise missing_extra('google-cloud-aiplatform', 'gcp') from e

    self._project = project
    self._location = location
    self._agent_engine_id = agent_engine_id
    self._express_mode_api_key = get_express_mode_api_key(
        project, location, express_mode_api_key
    )

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    """Creates a new session.

    Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      state: The initial state of the session.
      session_id: The ID of the session.
      **kwargs: Additional arguments to pass to the session creation. E.g. set
        ttl='7200s' to set the session time-to-live or
        expire_time='2025-10-01T00:00:00Z' to set the session expiration time.
        See https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.reasoningEngines.sessions
        for more details.

    Returns:
      The created session.
    """
    if kwargs.get('ttl') is not None and kwargs.get('expire_time') is not None:
      raise ValueError(
          "Cannot specify both 'ttl' and 'expire_time' simultaneously."
      )
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)

    config: dict[str, Any] = {'session_state': state} if state else {}
    if session_id:
      session_id = _extract_short_session_id(
          session_id, expected_engine_id=reasoning_engine_id
      )
      _validate_session_id(session_id)
      config['session_id'] = session_id
    config.update(kwargs)
    async with self._get_api_client() as api_client:
      api_response = await api_client.agent_engines.sessions.create(
          name=f'reasoningEngines/{reasoning_engine_id}',
          user_id=user_id,
          config=config,
      )
      logger.debug('Create session response: %s', api_response)
      get_session_response = api_response.response
      session_id = get_session_response.name.split('/')[-1]

    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=getattr(get_session_response, 'session_state', None) or {},
        last_update_time=get_session_response.update_time.timestamp(),
    )
    return session

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    session_id = _extract_short_session_id(
        session_id, expected_engine_id=reasoning_engine_id
    )
    _validate_session_id(session_id)
    session_resource_name = (
        f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}'
    )
    async with self._get_api_client() as api_client:
      # Get session resource and events in parallel.
      list_events_kwargs = {}
      if config and config.after_timestamp:
        # Filter events based on timestamp.
        list_events_kwargs['config'] = {
            'filter': 'timestamp>="{}"'.format(
                datetime.datetime.fromtimestamp(
                    config.after_timestamp, tz=datetime.timezone.utc
                ).isoformat()
            )
        }

      try:
        if config and config.num_recent_events == 0:
          get_session_response = await api_client.agent_engines.sessions.get(
              name=session_resource_name
          )
          events_iterator = None
        else:
          get_session_response, events_iterator = await asyncio.gather(
              api_client.agent_engines.sessions.get(name=session_resource_name),
              api_client.agent_engines.sessions.events.list(
                  name=session_resource_name,
                  **list_events_kwargs,
              ),
          )
      except ClientError as e:
        if e.code == 404:
          logger.debug(
              'Session %s not found in Vertex AI Agent Engine.',
              session_resource_name,
          )
          return None
        raise
      if get_session_response.user_id != user_id:
        raise ValueError(
            f'Session {session_id} does not belong to user {user_id}.'
        )

      update_timestamp = get_session_response.update_time.timestamp()
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=getattr(get_session_response, 'session_state', None) or {},
          last_update_time=update_timestamp,
      )
      # Preserve the entire event stream that Vertex returns rather than trying
      # to discard events written milliseconds after the session resource was
      # updated. Clock skew between those writes can otherwise drop tool_result
      # events and permanently break the replayed conversation.
      if events_iterator is not None:
        async for event in events_iterator:
          session.events.append(_from_api_event(event))

    if config:
      # Filter events based on num_recent_events. Note `0` must return an empty
      # list (and `events[-0:]` would wrongly return everything).
      if config.num_recent_events is not None:
        session.events = (
            session.events[-config.num_recent_events :]
            if config.num_recent_events
            else []
        )

    return session

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)

    async with self._get_api_client() as api_client:
      sessions = []
      config = {}
      if user_id is not None:
        config['filter'] = f'user_id={_quote_filter_literal(user_id)}'
      sessions_iterator = await api_client.agent_engines.sessions.list(
          name=f'reasoningEngines/{reasoning_engine_id}',
          config=config,
      )

      async for api_session in sessions_iterator:
        sessions.append(
            Session(
                app_name=app_name,
                user_id=api_session.user_id,
                id=api_session.name.split('/')[-1],
                state=getattr(api_session, 'session_state', None) or {},
                last_update_time=api_session.update_time.timestamp(),
            )
        )

    sessions.sort(key=lambda s: (s.last_update_time, s.user_id, s.id))
    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    reasoning_engine_id = self._get_reasoning_engine_id(app_name)
    session_id = _extract_short_session_id(
        session_id, expected_engine_id=reasoning_engine_id
    )
    _validate_session_id(session_id)
    session_resource_name = (
        f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}'
    )

    async with self._get_api_client() as api_client:
      # Enforce ownership: delete_session otherwise ignores user_id entirely.
      try:
        existing = await api_client.agent_engines.sessions.get(
            name=session_resource_name
        )
      except ClientError as e:
        if e.code == 404:
          return
        raise
      if existing.user_id != user_id:
        raise ValueError(
            f'Session {session_id} does not belong to user {user_id}.'
        )

      try:
        await api_client.agent_engines.sessions.delete(
            name=session_resource_name,
        )
      except Exception as e:
        logger.error('Error deleting session %s: %s', session_id, e)
        raise

  @override
  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    """Not supported by the Vertex AI Agent Engine backend.

    The Vertex AI Agent Engine API does not expose user state independently of
    a session.  To read user state, enumerate sessions via ``list_sessions``
    and call ``get_session`` on each result to access the merged state.

    Raises:
      NotImplementedError: Always, because the Vertex AI Agent Engine API does
        not provide a way to query user state without a session.
    """
    raise NotImplementedError(
        'VertexAiSessionService does not support get_user_state. '
        'The Vertex AI Agent Engine API does not expose user state '
        'independently of a session. To read user state, enumerate sessions '
        'via list_sessions and call get_session on each result.'
    )

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    await super().append_event(session=session, event=event)

    _validate_session_id(session.id)
    reasoning_engine_id = self._get_reasoning_engine_id(session.app_name)

    # Build config (Monolithic approach)
    config: dict[str, Any] = {}
    if event.content:
      content_dict = event.content.model_dump(exclude_none=True, mode='json')
      _drop_vertex_unsupported_part_fields(content_dict)
      config['content'] = content_dict
    if event.actions:
      config['actions'] = {
          'skip_summarization': event.actions.skip_summarization,
          'state_delta': event.actions.state_delta,
          'artifact_delta': event.actions.artifact_delta,
          'transfer_agent': event.actions.transfer_to_agent,
          'escalate': event.actions.escalate,
          'requested_auth_configs': {
              k: json.loads(v.model_dump_json(exclude_none=True, by_alias=True))
              for k, v in event.actions.requested_auth_configs.items()
          },
      }
    if event.error_code:
      config['error_code'] = event.error_code
    if event.error_message:
      config['error_message'] = event.error_message

    metadata_dict: dict[str, Any] = {
        'partial': event.partial,
        'turn_complete': event.turn_complete,
        'interrupted': event.interrupted,
        'branch': event.branch,
        'custom_metadata': event.custom_metadata,
        'long_running_tool_ids': (
            list(event.long_running_tool_ids)
            if event.long_running_tool_ids
            else None
        ),
    }
    if event.grounding_metadata:
      metadata_dict['grounding_metadata'] = event.grounding_metadata.model_dump(
          exclude_none=True, mode='json'
      )

    # ALWAYS write to custom_metadata
    if event.actions and event.actions.compaction:
      compaction_dict = event.actions.compaction.model_dump(
          exclude_none=True, mode='json'
      )
      _set_internal_custom_metadata(
          metadata_dict,
          key=_COMPACTION_CUSTOM_METADATA_KEY,
          value=compaction_dict,
      )
    if event.usage_metadata:
      usage_dict = event.usage_metadata.model_dump(
          exclude_none=True, mode='json'
      )
      _set_internal_custom_metadata(
          metadata_dict,
          key=_USAGE_METADATA_CUSTOM_METADATA_KEY,
          value=usage_dict,
      )

    config['event_metadata'] = metadata_dict

    # Persist the full event state using raw_event. If the client-side SDK
    # does not support this field, it will raise a ValidationError, and we
    # will fall back to legacy field-based storage.
    config['raw_event'] = event.model_dump(
        exclude_none=True,
        mode='json',
        by_alias=True,
    )
    if isinstance(config['raw_event'].get('content'), dict):
      _drop_vertex_unsupported_part_fields(config['raw_event']['content'])

    # Retry without raw_event if client side validation fails for older SDK
    # versions.
    async with self._get_api_client() as api_client:

      async def _do_append(cfg: dict[str, Any]) -> None:
        await api_client.agent_engines.sessions.events.append(
            name=(
                f'reasoningEngines/{reasoning_engine_id}/sessions/{session.id}'
            ),
            author=event.author,
            invocation_id=event.invocation_id,
            timestamp=datetime.datetime.fromtimestamp(
                event.timestamp, tz=datetime.timezone.utc
            ),
            config=cfg,
        )

      try:
        await _do_append(config)
      except pydantic.ValidationError:
        logger.warning('Vertex SDK does not support raw_event, falling back.')
        if 'raw_event' in config:
          del config['raw_event']
        await _do_append(config)
    return event

  def _get_reasoning_engine_id(self, app_name: str) -> str:
    if self._agent_engine_id:
      return self._agent_engine_id

    if app_name.isdigit():
      return app_name

    pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$'
    match = re.fullmatch(pattern, app_name)

    if not match:
      raise ValueError(
          f'App name {app_name} is not valid. It should either be the full'
          ' ReasoningEngine resource name, or the reasoning engine id.'
      )

    return match.groups()[-1]

  def _api_client_http_options_override(
      self,
  ) -> Optional[Union[types.HttpOptions, types.HttpOptionsDict]]:
    return None

  def _get_api_client(self) -> vertexai.AsyncClient:
    """Instantiates an API client for the given project and location.

    Returns:
      An API client for the given project and location or express mode api key.
    """
    import vertexai

    if self._express_mode_api_key:
      return vertexai.Client(
          http_options=self._api_client_http_options_override(),
          api_key=self._express_mode_api_key,
      ).aio
    return vertexai.Client(
        project=self._project,
        location=self._location,
        http_options=self._api_client_http_options_override(),
    ).aio


def _get_raw_event(api_event_obj: object) -> dict[str, Any] | None:
  """Extracts raw_event dict from SessionEvent object safely."""
  for attribute_name in ('raw_event', 'rawEvent'):
    raw_event: object = getattr(api_event_obj, attribute_name, None)
    if raw_event is None:
      continue
    if not isinstance(raw_event, Mapping):
      return None

    normalized: dict[str, Any] = {}
    for key, value in raw_event.items():
      if not isinstance(key, str):
        return None
      normalized[key] = value
    return normalized
  return None


def _first_present_key(
    payload: Mapping[str, Any], keys: tuple[str, ...]
) -> Optional[str]:
  """Returns the first key of ``keys`` that ``payload`` contains."""
  return next((key for key in keys if key in payload), None)


def _scale_legacy_timestamp(key: Optional[str], value: Any) -> Any:
  """Converts a timestamp read under a legacy adk-js key to epoch seconds."""
  if key in _LEGACY_MILLIS_TIMESTAMP_KEYS:
    return value / _MILLIS_PER_SECOND
  return value


def _normalize_compaction_payload(
    payload: Any, *, event_name: str
) -> Optional[EventCompaction]:
  """Reads a persisted compaction payload in any spelling ADK has written.

  Accepts the canonical ``adk-python`` payload (``start_timestamp`` or
  ``startTimestamp``, with a ``Content`` summary) and the legacy ``adk-js``
  payload (``startTime`` / ``endTime`` / ``compactedContent``, with a flat
  string summary). A timestamp is scaled to seconds only when it was read
  under a legacy key, because those hold epoch milliseconds by construction:
  ``adk-js`` derives them from ``Date.now()``. The magnitude of the value is
  never inspected, so a canonical timestamp is returned unchanged.

  This is a read-compatibility boundary over records another SDK persisted, so
  an unreadable payload degrades to ``None`` instead of raising. Raising would
  fail the whole ``get_session`` call and lose the entire session.

  Args:
    payload: The persisted payload, or ``None`` when the event has none.
    event_name: The API resource name of the event, for the warning log.

  Returns:
    The compaction, or ``None`` if the payload is absent or unreadable.
  """
  if payload is None:
    return None
  if not isinstance(payload, Mapping):
    logger.warning(
        'Ignoring the compaction of event %s: expected a mapping, got %s.',
        event_name,
        type(payload).__name__,
    )
    return None

  start_key = _first_present_key(payload, _COMPACTION_START_KEYS)
  end_key = _first_present_key(payload, _COMPACTION_END_KEYS)
  content_key = _first_present_key(payload, _COMPACTION_CONTENT_KEYS)
  content = payload.get(content_key)
  try:
    # Only the canonical keys are forwarded: `EventCompaction` forbids extras,
    # so foreign keys such as `isCompacted` would defeat the normalization.
    return EventCompaction.model_validate({
        'start_timestamp': _scale_legacy_timestamp(
            start_key, payload.get(start_key)
        ),
        'end_timestamp': _scale_legacy_timestamp(end_key, payload.get(end_key)),
        'compacted_content': (
            {'role': 'model', 'parts': [{'text': content}]}
            if isinstance(content, str)
            else content
        ),
    })
  except (pydantic.ValidationError, TypeError, ValueError) as err:
    # The payload holds user conversation text, so log only its shape and the
    # error type. A pydantic ValidationError message quotes the input value,
    # so it must not reach the log, and neither must a traceback carrying it.
    logger.warning(
        'Ignoring an unreadable compaction on event %s; %s, field types: %s.',
        event_name,
        type(err).__name__,
        {key: type(value).__name__ for key, value in payload.items()},
    )
    return None


def _normalize_raw_event_compaction(
    event_dict: dict[str, Any], *, event_name: str
) -> None:
  """Rewrites the compaction of a raw event dict in place, canonically.

  ``adk-js`` JSON round-trips its whole compacted event into ``raw_event``, so
  the compaction arrives at the top level of the raw event. A JS process that
  reads an event and re-appends it also parks the same legacy payload under
  ``actions.compaction``. Both spellings are normalized here so
  ``Event.model_validate`` neither drops nor rejects them.

  Args:
    event_dict: The raw event dict, mutated in place.
    event_name: The API resource name of the event, for the warning log.
  """
  legacy_payload = {
      key: event_dict.pop(key)
      for key in _LEGACY_RAW_EVENT_COMPACTION_KEYS
      if key in event_dict
  }
  actions = event_dict.get('actions') or {}
  if not isinstance(actions, dict):
    # A non-mapping `actions` is not a compaction problem. Leave it for
    # `Event.model_validate` to reject.
    return
  # The nested payload wins: it is the more specific channel, and a JS
  # re-append writes the same data to both.
  compaction = _normalize_compaction_payload(
      actions.get('compaction', legacy_payload or None), event_name=event_name
  )
  if compaction is None:
    actions.pop('compaction', None)
  else:
    actions['compaction'] = compaction
    event_dict['actions'] = actions


def _from_api_event(api_event_obj: vertexai.types.SessionEvent) -> Event:
  """Converts an API event object to an Event object."""
  # Prioritize reading from raw_event to restore full state. Fall back to
  # top-level fields for older data that lacks raw_event.
  raw_event_dict = _get_raw_event(api_event_obj)
  if raw_event_dict:
    event_dict = copy.deepcopy(raw_event_dict)
    timestamp_obj = getattr(api_event_obj, 'timestamp', None)
    event_dict.update({
        'id': api_event_obj.name.split('/')[-1],
        'invocation_id': getattr(api_event_obj, 'invocation_id', None),
        'author': getattr(api_event_obj, 'author', None),
    })
    if timestamp_obj:
      event_dict['timestamp'] = timestamp_obj.timestamp()
    _normalize_raw_event_compaction(event_dict, event_name=api_event_obj.name)
    return Event.model_validate(event_dict)

  actions = getattr(api_event_obj, 'actions', None)
  event_metadata = getattr(api_event_obj, 'event_metadata', None)
  if event_metadata:
    long_running_tool_ids_list = getattr(
        event_metadata, 'long_running_tool_ids', None
    )
    long_running_tool_ids = (
        set(long_running_tool_ids_list) if long_running_tool_ids_list else None
    )
    partial = getattr(event_metadata, 'partial', None)
    turn_complete = getattr(event_metadata, 'turn_complete', None)
    interrupted = getattr(event_metadata, 'interrupted', None)
    branch = getattr(event_metadata, 'branch', None)
    custom_metadata = getattr(event_metadata, 'custom_metadata', None)
    # Extract compaction data stored in custom_metadata.
    # NOTE: This read path must be kept permanently because sessions
    # written before native compaction support store compaction data
    # in custom_metadata under the compaction metadata key. The payload is
    # normalized below, because adk-js writes it in a legacy spelling.
    compaction_data = None
    usage_metadata_data = None
    if custom_metadata and (
        _COMPACTION_CUSTOM_METADATA_KEY in custom_metadata
        or _USAGE_METADATA_CUSTOM_METADATA_KEY in custom_metadata
    ):
      custom_metadata = dict(custom_metadata)  # avoid mutating the API response
      compaction_data = custom_metadata.pop(
          _COMPACTION_CUSTOM_METADATA_KEY, None
      )
      usage_metadata_data = custom_metadata.pop(
          _USAGE_METADATA_CUSTOM_METADATA_KEY, None
      )
      if not custom_metadata:
        custom_metadata = None
    grounding_metadata = _session_util.decode_model(
        getattr(event_metadata, 'grounding_metadata', None),
        types.GroundingMetadata,
    )
  else:
    long_running_tool_ids = None
    partial = None
    turn_complete = None
    interrupted = None
    branch = None
    custom_metadata = None
    compaction_data = None
    usage_metadata_data = None
    grounding_metadata = None

  if actions:
    actions_dict = actions.model_dump(exclude_none=True, mode='python')
    rename_map = {'transfer_agent': 'transfer_to_agent'}
    renamed_actions_dict = {
        rename_map.get(k, k): v for k, v in actions_dict.items()
    }
  else:
    renamed_actions_dict = {}
  compaction = _normalize_compaction_payload(
      compaction_data
      if compaction_data is not None
      else renamed_actions_dict.get('compaction'),
      event_name=api_event_obj.name,
  )
  if compaction is None:
    renamed_actions_dict.pop('compaction', None)
  else:
    renamed_actions_dict['compaction'] = compaction
  event_actions = EventActions.model_validate(renamed_actions_dict)

  usage_metadata = None
  if usage_metadata_data:
    usage_metadata = types.GenerateContentResponseUsageMetadata.model_validate(
        usage_metadata_data
    )

  timestamp_obj = getattr(api_event_obj, 'timestamp', None)
  timestamp = (
      timestamp_obj.timestamp()
      if timestamp_obj
      else datetime.datetime.now(datetime.timezone.utc).timestamp()
  )

  return Event(
      id=api_event_obj.name.split('/')[-1],
      invocation_id=api_event_obj.invocation_id,
      author=api_event_obj.author,
      actions=event_actions,
      content=_session_util.decode_model(
          getattr(api_event_obj, 'content', None), types.Content
      ),
      timestamp=timestamp,
      error_code=getattr(api_event_obj, 'error_code', None),
      error_message=getattr(api_event_obj, 'error_message', None),
      partial=partial,
      turn_complete=turn_complete,
      interrupted=interrupted,
      branch=branch,
      custom_metadata=custom_metadata,
      grounding_metadata=grounding_metadata,
      long_running_tool_ids=long_running_tool_ids,
      usage_metadata=usage_metadata,
  )
