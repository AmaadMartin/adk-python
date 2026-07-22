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

from typing import Any
from typing import AsyncGenerator
from typing import Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from google.genai import live
from google.genai import types

from ..features import FeatureName
from ..features import is_feature_enabled
from ..models.llm_response import LlmResponse
import logging

logger = logging.getLogger(__name__)

class StreamingResponseAggregator:
  """Aggregates partial streaming responses.

  It aggregates content from partial responses, and generates LlmResponses for
  individual (partial) model responses, as well as for aggregated content.
  """

  def __init__(
      self,
      live_session_id: str | None = None,
      model_version: str | None = None,
      is_gemini_3_x_live: bool = False,
  ) -> None:
    self._text = ''
    self._thought_text = ''
    self._usage_metadata = None
    self._grounding_metadata: Optional[types.GroundingMetadata] = None
    self._citation_metadata: Optional[types.CitationMetadata] = None
    self._response = None

    # For progressive SSE streaming mode: accumulate parts in order
    self._parts_sequence: list[types.Part] = []
    self._current_text_buffer: str = ''
    self._current_text_is_thought: Optional[bool] = None
    self._finish_reason: Optional[types.FinishReason] = None

    # For streaming function call arguments
    self._current_fc_name: Optional[str] = None
    self._current_fc_args: dict[str, Any] = {}
    self._current_fc_id: Optional[str] = None
    self._current_thought_signature: Optional[bytes] = None
    # For Live API state
    self._live_session_id = live_session_id
    self._model_version = model_version
    self._is_gemini_3_x_live = is_gemini_3_x_live
    self._input_transcription_text: str = ''
    self._output_transcription_text: str = ''
    self._last_grounding_metadata: Optional[types.GroundingMetadata] = None
    self._tool_call_metadata: Optional[types.GroundingMetadata] = None
    self._tool_call_parts: list[types.Part] = []
    self._is_thought: bool = False

  def _flush_text_buffer_to_sequence(self) -> None:
    """Flush current text buffer to parts sequence.

    This helper is used in progressive SSE mode to maintain part ordering.
    It only merges consecutive text parts of the same type (thought or regular).
    """
    if self._current_text_buffer:
      if self._current_text_is_thought:
        self._parts_sequence.append(
            types.Part(text=self._current_text_buffer, thought=True)
        )
      else:
        self._parts_sequence.append(
            types.Part.from_text(text=self._current_text_buffer)
        )
      self._current_text_buffer = ''
      self._current_text_is_thought = None

  def _get_value_from_partial_arg(
      self, partial_arg: types.PartialArg, json_path: str
  ) -> tuple[Any, bool]:
    """Extract value from a partial argument.

    Args:
      partial_arg: The partial argument object
      json_path: JSONPath for this argument

    Returns:
      Tuple of (value, has_value) where has_value indicates if a value exists
    """
    value: Any = None
    has_value = False

    if partial_arg.string_value is not None:
      # For streaming strings, append chunks to existing value
      string_chunk = partial_arg.string_value
      has_value = True

      # Get current value for this path (if any)
      path_without_prefix = (
          json_path[2:] if json_path.startswith('$.') else json_path
      )
      path_parts = path_without_prefix.split('.')

      # Try to get existing value
      existing_value: Any = self._current_fc_args
      for part in path_parts:
        if isinstance(existing_value, dict) and part in existing_value:
          existing_value = existing_value[part]
        else:
          break

      # Append to existing string or set new value
      if isinstance(existing_value, str):
        value = existing_value + string_chunk
      else:
        value = string_chunk

    elif partial_arg.number_value is not None:
      value = partial_arg.number_value
      has_value = True
    elif partial_arg.bool_value is not None:
      value = partial_arg.bool_value
      has_value = True
    elif partial_arg.null_value is not None:
      value = None
      has_value = True

    return value, has_value

  def _set_value_by_json_path(self, json_path: str, value: Any) -> None:
    """Set a value in _current_fc_args using JSONPath notation.

    Args:
      json_path: JSONPath string like "$.location" or "$.location.latitude"
      value: The value to set
    """
    # Remove leading "$." from jsonPath
    if json_path.startswith('$.'):
      path = json_path[2:]
    else:
      path = json_path

    # Split path into components
    path_parts = path.split('.')

    # Navigate to the correct location and set the value
    current = self._current_fc_args
    for part in path_parts[:-1]:
      if part not in current:
        current[part] = {}
      current = current[part]

    # Set the final value
    current[path_parts[-1]] = value

  def _flush_function_call_to_sequence(self) -> None:
    """Flush current function call to parts sequence.

    This creates a complete FunctionCall part from accumulated partial args.
    """
    if self._current_fc_name:
      # Create function call part with accumulated args
      fc_part = types.Part.from_function_call(
          name=self._current_fc_name,
          args=self._current_fc_args.copy(),
      )

      # Set the ID if provided (directly on the function_call object)
      if self._current_fc_id and fc_part.function_call:
        fc_part.function_call.id = self._current_fc_id

      # Set thought_signature if provided (on the Part, not FunctionCall)
      if self._current_thought_signature:
        fc_part.thought_signature = self._current_thought_signature

      self._parts_sequence.append(fc_part)

      # Reset FC state
      self._current_fc_name = None
      self._current_fc_args = {}
      self._current_fc_id = None
      self._current_thought_signature = None

  def _process_streaming_function_call(self, fc: types.FunctionCall) -> None:
    """Process a streaming function call with partialArgs.

    Args:
      fc: The function call object with partial_args
    """
    # Save function name if present (first chunk)
    if fc.name:
      self._current_fc_name = fc.name
    if fc.id:
      self._current_fc_id = fc.id

    # Process each partial argument
    for partial_arg in fc.partial_args or []:
      json_path = partial_arg.json_path
      if not json_path:
        continue

      # Extract value from partial arg
      value, has_value = self._get_value_from_partial_arg(
          partial_arg, json_path
      )

      # Set the value using JSONPath (only if a value was provided)
      if has_value:
        self._set_value_by_json_path(json_path, value)

    # Check if function call is complete
    if not fc.will_continue:
      # Function call complete, flush it
      self._flush_text_buffer_to_sequence()
      self._flush_function_call_to_sequence()

  def _process_function_call_part(self, part: types.Part) -> None:
    """Process a function call part (streaming or non-streaming).

    Args:
      part: The part containing a function call
    """
    fc = part.function_call
    if fc is None:
      return

    # Check if this is a streaming FC (has partialArgs or will_continue=True)
    # The first chunk of a streaming function call may have will_continue=True
    # but no partial_args yet, so we need to check both conditions.
    if fc.partial_args or fc.will_continue:
      # Streaming function call arguments

      # Generate ID on first chunk if not provided by LLM
      if not fc.id and not self._current_fc_id:
        # Lazy import to avoid circular dependency
        from ..flows.llm_flows.functions import generate_client_function_call_id

        fc.id = generate_client_function_call_id()

      # Save thought_signature from the part (first chunk should have it)
      if part.thought_signature and not self._current_thought_signature:
        self._current_thought_signature = part.thought_signature
      self._process_streaming_function_call(fc)
    else:
      # Non-streaming function call (standard format with args)
      # Skip empty function calls (used as streaming end markers)
      if fc.name:
        # Generate ID if not provided by LLM
        if not fc.id:
          # Lazy import to avoid circular dependency
          from ..flows.llm_flows.functions import generate_client_function_call_id

          fc.id = generate_client_function_call_id()
        # Flush any buffered text first, then add the FC part
        self._flush_text_buffer_to_sequence()
        self._parts_sequence.append(part)

  def _build_full_text_response(
      self,
      text: str,
      is_thought: bool = False,
      grounding_metadata: types.GroundingMetadata | None = None,
      interrupted: bool = False,
  ) -> LlmResponse:
    """Builds a full text response.

    The text should not be partial and the returned LlmResponse is not
    partial.

    Args:
      text: The text to be included in the response.
      is_thought: Whether the text is a thought.
      grounding_metadata: The grounding metadata to include.
      interrupted: Whether this response was interrupted.

    Returns:
      An LlmResponse containing the full text.
    """
    part = types.Part.from_text(text=text)
    if is_thought:
      part.thought = True

    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[part],
        ),
        grounding_metadata=grounding_metadata,
        interrupted=interrupted,
        partial=False,
        live_session_id=self._live_session_id,
    )

  async def process_live_server_message(
      self, message: "live.LiveServerMessage"
  ) -> AsyncGenerator[LlmResponse, None]:
    logger.debug('Got LLM Live message: %s', message)
    if message.usage_metadata:
      # Remap live token usage to GenerateContentResponse usage metadata.
      yield LlmResponse(
          usage_metadata=self._to_generate_content_usage_metadata(
              message.usage_metadata
          ),
          model_version=self._model_version,
          live_session_id=self._live_session_id,
      )
    if message.server_content:
      content = message.server_content.model_turn
      grounding_metadata = message.server_content.grounding_metadata
      if grounding_metadata:
        self._last_grounding_metadata = self._merge_grounding_metadata(
            self._last_grounding_metadata, grounding_metadata
        )

      # Standalone grounding_metadata event (when content is empty)
      if (
          not (content and content.parts)
          and message.server_content.grounding_metadata
          and not message.server_content.turn_complete
      ):
        yield LlmResponse(
            grounding_metadata=message.server_content.grounding_metadata,
            interrupted=message.server_content.interrupted,
            model_version=self._model_version,
            live_session_id=self._live_session_id,
            turn_complete_reason=getattr(
                message.server_content, 'turn_complete_reason', None
            ),
        )

      if content and content.parts:
        llm_response = LlmResponse(
            content=content,
            interrupted=message.server_content.interrupted,
            model_version=self._model_version,
            live_session_id=self._live_session_id,
            turn_complete_reason=getattr(
                message.server_content, 'turn_complete_reason', None
            ),
        )
        # grounding_metadata is yielded again at turn_complete,
        # so avoid duplicating it here if turn_complete is true.
        if not message.server_content.turn_complete:
          if message.server_content.grounding_metadata is not None:
            llm_response.grounding_metadata = (
                message.server_content.grounding_metadata
            )
        if content.parts[0].text:
          current_is_thought = getattr(content.parts[0], 'thought', False)
          if self._text and current_is_thought != self._is_thought:
            yield self._build_full_text_response(self._text, self._is_thought)
            self._text = ''
            self._is_thought = False

          self._text += content.parts[0].text
          self._is_thought = current_is_thought
          llm_response.partial = True
        # don't yield the merged self._text event when receiving audio data
        elif self._text and not content.parts[0].inline_data:
          yield self._build_full_text_response(
              self._text, self._is_thought, self._last_grounding_metadata
          )
          self._text = ''
          self._is_thought = False
          self._last_grounding_metadata = None
        yield llm_response
      # Note: in some cases, tool_call may arrive before
      # generation_complete, causing transcription to appear after
      # tool_call in the session log.
      if message.server_content.input_transcription:
        # Gemini 3.x Live only sends a single final input
        # transcription
        if self._is_gemini_3_x_live:
          if message.server_content.input_transcription.text:
            yield LlmResponse(
                input_transcription=types.Transcription(
                    text=message.server_content.input_transcription.text,
                    finished=True,
                ),
                partial=False,
                model_version=self._model_version,
                live_session_id=self._live_session_id,
            )
        else:
          if message.server_content.input_transcription.text:
            self._input_transcription_text += (
                message.server_content.input_transcription.text
            )
            yield LlmResponse(
                input_transcription=types.Transcription(
                    text=message.server_content.input_transcription.text,
                    finished=False,
                ),
                partial=True,
                model_version=self._model_version,
                live_session_id=self._live_session_id,
            )
          # finished=True and partial transcription may happen in the same
          # message.
          if message.server_content.input_transcription.finished:
            yield LlmResponse(
                input_transcription=types.Transcription(
                    text=self._input_transcription_text,
                    finished=True,
                ),
                partial=False,
                model_version=self._model_version,
                live_session_id=self._live_session_id,
            )
            self._input_transcription_text = ''
      if message.server_content.output_transcription:
        if message.server_content.output_transcription.text:
          self._output_transcription_text += (
              message.server_content.output_transcription.text
          )
          yield LlmResponse(
              output_transcription=types.Transcription(
                  text=message.server_content.output_transcription.text,
                  finished=False,
              ),
              partial=True,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
        if message.server_content.output_transcription.finished:
          yield LlmResponse(
              output_transcription=types.Transcription(
                  text=self._output_transcription_text,
                  finished=True,
              ),
              partial=False,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
          self._output_transcription_text = ''
      # The Gemini API or Vertex AI might not send a transcription finished signal.
      # Instead, we rely on generation_complete, turn_complete or
      # interrupted signals to flush any pending transcriptions.
      if (
          message.server_content.interrupted
          or message.server_content.turn_complete
          or message.server_content.generation_complete
      ):
        if self._input_transcription_text:
          yield LlmResponse(
              input_transcription=types.Transcription(
                  text=self._input_transcription_text,
                  finished=True,
              ),
              partial=False,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
          self._input_transcription_text = ''
        if self._output_transcription_text:
          yield LlmResponse(
              output_transcription=types.Transcription(
                  text=self._output_transcription_text,
                  finished=True,
              ),
              partial=False,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
          self._output_transcription_text = ''
      if message.server_content.turn_complete:
        # Capture final grounding metadata before self._last_grounding_metadata is cleared in the next block.
        final_grounding_metadata = (
            grounding_metadata
            or self._last_grounding_metadata
            or (
                types.GroundingMetadata()
                if self._is_gemini_3_x_live
                else None
            )
        )
        if (
            final_grounding_metadata
            and final_grounding_metadata.retrieval_queries
            and not final_grounding_metadata.grounding_chunks
        ):
          logger.warning(
              'Incomplete grounding_metadata received: retrieval_queries=%s'
              ' but grounding_chunks is empty. This may indicate a'
              ' transient issue with the Vertex AI Search backend.',
              final_grounding_metadata.retrieval_queries,
          )

        if self._text:
          yield self._build_full_text_response(
              self._text,
              self._is_thought,
              self._last_grounding_metadata,
              message.server_content.interrupted,
          )
          self._text = ''
          self._is_thought = False
          self._last_grounding_metadata = None
        if self._tool_call_parts:
          logger.debug('Returning aggregated self._tool_call_parts')
          yield LlmResponse(
              content=types.Content(role='model', parts=self._tool_call_parts),
              grounding_metadata=self._tool_call_metadata,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
          self._tool_call_parts = []
          if self._tool_call_metadata is not None:
            self._last_grounding_metadata = None
          self._tool_call_metadata = None

        yield LlmResponse(
            turn_complete=True,
            interrupted=message.server_content.interrupted,
            # If self._last_grounding_metadata was cleared in the full self._text yield,
            # avoid duplicating it here.
            grounding_metadata=grounding_metadata
            or self._last_grounding_metadata
            or (
                types.GroundingMetadata()
                if self._is_gemini_3_x_live
                else None
            ),
            model_version=self._model_version,
            live_session_id=self._live_session_id,
            turn_complete_reason=getattr(
                message.server_content, 'turn_complete_reason', None
            ),
        )
        self._last_grounding_metadata = None  # Reset after yielding
      # in case of empty content or parts, we still surface it
      # in case it's an interrupted message, we merge the previous partial
      # self._text. Other we don't merge. because content can be none when model
      # safety threshold is triggered
      if message.server_content.interrupted:
        if self._text:
          yield self._build_full_text_response(
              self._text,
              self._is_thought,
              self._last_grounding_metadata,
              interrupted=True,
          )
          self._text = ''
          self._is_thought = False
          self._last_grounding_metadata = None
        else:
          yield LlmResponse(
              interrupted=message.server_content.interrupted,
              grounding_metadata=self._last_grounding_metadata,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
          self._last_grounding_metadata = None
    if message.tool_call:
      logger.debug('Received tool call: %s', message.tool_call)
      if self._text:
        yield self._build_full_text_response(
            self._text, self._is_thought, self._last_grounding_metadata
        )
        self._text = ''
        self._is_thought = False
        self._last_grounding_metadata = None
      self._tool_call_parts.extend([
          types.Part(function_call=function_call)
          for function_call in message.tool_call.function_calls
      ])
      if not self._is_gemini_3_x_live:
        if self._tool_call_metadata is None:
          self._tool_call_metadata = self._last_grounding_metadata
      # Gemini 3.x Live does not emit turn_complete until it receives the
      # tool response, so yield tool calls immediately to avoid
      # deadlocking the conversation. Other models (e.g. 2.5-pro,
      # native-audio) send turn_complete after tool calls, so buffer
      # and merge them into a single response at turn_complete.
      if self._is_gemini_3_x_live and self._tool_call_parts:
        logger.debug(
            'Yielding self._tool_call_parts immediately for Gemini 3.x live tool'
            ' call'
        )
        yield LlmResponse(
            content=types.Content(role='model', parts=self._tool_call_parts),
            grounding_metadata=self._last_grounding_metadata,
            model_version=self._model_version,
            live_session_id=self._live_session_id,
        )
        self._tool_call_parts = []
        self._last_grounding_metadata = None
    if message.session_resumption_update:
      logger.debug('Received session resumption message: %s', message)
      yield (
          LlmResponse(
              live_session_resumption_update=message.session_resumption_update,
              model_version=self._model_version,
              live_session_id=self._live_session_id,
          )
      )
    if message.voice_activity:
      logger.debug('Received voice activity: %s', message.voice_activity)
      yield LlmResponse(
          voice_activity=message.voice_activity,
          model_version=self._model_version,
          live_session_id=self._live_session_id,
      )
    if message.go_away:
      logger.debug('Received GoAway message: %s', message.go_away)
      yield LlmResponse(
          go_away=message.go_away,
          model_version=self._model_version,
          live_session_id=self._live_session_id,
      )



  async def close_live(self) -> AsyncGenerator[LlmResponse, None]:
    if self._tool_call_parts:
      logger.debug('Exited loop with pending tool_call_parts')
      yield LlmResponse(
          content=types.Content(role='model', parts=self._tool_call_parts),
          model_version=self._model_version,
          live_session_id=self._live_session_id,
      )

  @staticmethod
  def _merge_grounding_metadata(
      existing: types.GroundingMetadata | None,
      new: types.GroundingMetadata | None,
  ) -> types.GroundingMetadata | None:
    """Merges two GroundingMetadata instances, accumulating list fields safely."""
    if existing is None:
      return new
    if new is None:
      return existing
    existing_data = existing.model_dump(exclude_none=True)
    new_data = new.model_dump(exclude_none=True)

    # Get offset from existing grounding chunks for shifting support indices
    chunk_offset = len(existing_data.get('grounding_chunks', []))

    for key, val in new_data.items():
      if isinstance(val, list) and all(isinstance(x, str) for x in val):
        existing_list = existing_data.get(key, [])
        for item in val:
          if item not in existing_list:
            existing_list.append(item)
        existing_data[key] = existing_list
      elif key == 'grounding_chunks':
        existing_chunks = existing_data.get('grounding_chunks', [])
        existing_chunks.extend(val)
        existing_data['grounding_chunks'] = existing_chunks
      elif key == 'grounding_supports':
        existing_supports = existing_data.get('grounding_supports', [])
        for support in val:
          if (
              'grounding_chunk_indices' in support
              and support['grounding_chunk_indices']
          ):
            support['grounding_chunk_indices'] = [
                idx + chunk_offset for idx in support['grounding_chunk_indices']
            ]
          existing_supports.append(support)
        existing_data['grounding_supports'] = existing_supports
      else:
        existing_data[key] = val
    return types.GroundingMetadata(**existing_data)

  def _to_generate_content_usage_metadata(
      self, usage_metadata: types.UsageMetadata
  ) -> types.GenerateContentResponseUsageMetadata:
    """Converts live API usage metadata to GenerateContentResponse usage metadata.

    The live API names output tokens `response_token_count`/
    `response_tokens_details`, whereas `GenerateContentResponseUsageMetadata`
    names them `candidates_token_count`/`candidates_tokens_details`.

    Args:
      usage_metadata: The live API usage metadata.

    Returns:
      The converted usage metadata.
    """
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage_metadata.prompt_token_count,
        cached_content_token_count=usage_metadata.cached_content_token_count,
        candidates_token_count=usage_metadata.response_token_count,
        total_token_count=usage_metadata.total_token_count,
        thoughts_token_count=usage_metadata.thoughts_token_count,
        tool_use_prompt_token_count=usage_metadata.tool_use_prompt_token_count,
        prompt_tokens_details=usage_metadata.prompt_tokens_details,
        cache_tokens_details=usage_metadata.cache_tokens_details,
        candidates_tokens_details=usage_metadata.response_tokens_details,
        tool_use_prompt_tokens_details=usage_metadata.tool_use_prompt_tokens_details,
        traffic_type=usage_metadata.traffic_type,
    )

  async def process_response(
      self, response: types.GenerateContentResponse
  ) -> AsyncGenerator[LlmResponse, None]:
    """Processes a single model response.

    Args:
      response: The response to process.

    Yields:
      The generated LlmResponse(s), for the partial response, and the aggregated
      response if needed.
    """
    # results = []
    self._response = response
    llm_response = LlmResponse.create(response)
    self._usage_metadata = llm_response.usage_metadata
    if llm_response.grounding_metadata:
      self._grounding_metadata = llm_response.grounding_metadata
    if llm_response.citation_metadata:
      self._citation_metadata = llm_response.citation_metadata

    # ========== Progressive SSE Streaming (new feature) ==========
    # Save finish_reason for final aggregation
    if llm_response.finish_reason:
      self._finish_reason = llm_response.finish_reason

    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
      # Accumulate parts while preserving their order
      # Only merge consecutive text parts of the same type (thought or regular)
      if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
          if part.text:
            # Check if we need to flush the current buffer first
            # (when text type changes from thought to regular or vice versa)
            if (
                self._current_text_buffer
                and part.thought != self._current_text_is_thought
            ):
              self._flush_text_buffer_to_sequence()

            # Accumulate text to buffer
            if not self._current_text_buffer:
              self._current_text_is_thought = part.thought
            self._current_text_buffer += part.text
          elif part.function_call:
            # Process function call (handles both streaming Args and
            # non-streaming Args)
            self._process_function_call_part(part)
          else:
            # Other non-text parts (bytes, etc.)
            # Flush any buffered text first, then add the non-text part
            self._flush_text_buffer_to_sequence()
            self._parts_sequence.append(part)

      # Mark ALL intermediate chunks as partial
      llm_response.partial = True
      yield llm_response
      return

    # ========== Non-Progressive SSE Streaming (old behavior) ==========
    if (
        llm_response.content
        and llm_response.content.parts
        and llm_response.content.parts[0].text
    ):
      part0 = llm_response.content.parts[0]
      part_text = part0.text or ''
      if part0.thought:
        self._thought_text += part_text
      else:
        self._text += part_text
      llm_response.partial = True
    elif (self._thought_text or self._text) and (
        not llm_response.content
        or not llm_response.content.parts
        # don't yield the merged text event when receiving audio data
        or not llm_response.content.parts[0].inline_data
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=self._thought_text, thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=self._text))
      yield LlmResponse(
          content=types.ModelContent(parts=parts),
          usage_metadata=llm_response.usage_metadata,
          grounding_metadata=llm_response.grounding_metadata,
          citation_metadata=llm_response.citation_metadata,
          finish_reason=llm_response.finish_reason,
          model_version=llm_response.model_version,
      )
      self._thought_text = ''
      self._text = ''
    yield llm_response

  def close(self) -> Optional[LlmResponse]:
    """Generate an aggregated response at the end, if needed.

    This should be called after all the model responses are processed.

    Returns:
      The aggregated LlmResponse.
    """
    if not self._response:
      return None

    candidate = (
        self._response.candidates[0] if self._response.candidates else None
    )

    finish_reason = self._finish_reason
    if not finish_reason and candidate:
      finish_reason = candidate.finish_reason

    error_code = None
    error_message = None
    if finish_reason and finish_reason != types.FinishReason.STOP:
      error_code = finish_reason
      error_message = candidate.finish_message if candidate else None
    elif not candidate and self._response.prompt_feedback:
      error_code = self._response.prompt_feedback.block_reason
      error_message = self._response.prompt_feedback.block_reason_message

    # ========== Progressive SSE Streaming (new feature) ==========
    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
      self._flush_text_buffer_to_sequence()
      self._flush_function_call_to_sequence()

      final_parts = self._parts_sequence
      content = types.ModelContent(parts=final_parts) if final_parts else None

      return LlmResponse(
          content=content,
          grounding_metadata=self._grounding_metadata,
          citation_metadata=self._citation_metadata,
          error_code=error_code,
          error_message=error_message,
          usage_metadata=self._usage_metadata,
          finish_reason=finish_reason,
          partial=False,
          model_version=self._response.model_version,
      )

    # ========== Non-Progressive SSE Streaming (old behavior) ==========
    parts = []
    if self._thought_text:
      parts.append(types.Part(text=self._thought_text, thought=True))
    if self._text:
      parts.append(types.Part.from_text(text=self._text))
    content = types.ModelContent(parts=parts) if parts else None

    return LlmResponse(
        content=content,
        grounding_metadata=self._grounding_metadata,
        citation_metadata=self._citation_metadata,
        error_code=error_code,
        error_message=error_message,
        usage_metadata=self._usage_metadata,
        finish_reason=finish_reason,
        partial=False,
        model_version=self._response.model_version,
    )
