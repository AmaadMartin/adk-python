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
from typing import AsyncGenerator
from typing import Union

from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.genai import types

from ..utils import model_name_utils
from ..utils.content_utils import filter_audio_parts
from ..utils.context_utils import Aclosing
from ..utils.variant_utils import GoogleLLMVariant
from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

RealtimeInput = Union[types.Blob, types.ActivityStart, types.ActivityEnd]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.genai import live


class GeminiLlmConnection(BaseLlmConnection):
  """The Gemini model connection."""

  def __init__(
      self,
      gemini_session: live.AsyncSession,
      api_backend: GoogleLLMVariant = GoogleLLMVariant.VERTEX_AI,
      model_version: str | None = None,
  ):
    self._gemini_session = gemini_session
    self._input_transcription_text: str = ''
    self._output_transcription_text: str = ''
    self._api_backend = api_backend
    self._model_version = model_version
    self._is_gemini_3_x_live = model_name_utils._is_gemini_3_x_live(
        model_version
    )
    self._is_gemini_3_5_live_translate = (
        model_name_utils.is_gemini_3_5_live_translate(model_version)
    )

  async def send_history(self, history: list[types.Content]) -> None:
    """Sends the conversation history to the gemini model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user; otherwise, it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """

    # TODO: Remove this filter and translate unary contents to streaming
    # contents properly.

    # Filter out audio parts from history because:
    # 1. audio has already been transcribed.
    # 2. sending audio via connection.send or connection.send_live_content is
    # not supported by LIVE API (session will be corrupted).
    # This method is called when:
    # 1. Agent transfer to a new agent
    # 2. Establishing a new live connection with previous ADK session history

    contents = [
        filtered
        for content in history
        if (filtered := filter_audio_parts(content)) is not None
    ]

    if contents:
      logger.debug('Sending history to live connection: %s', contents)
      await self._gemini_session.send_client_content(
          turns=contents,
          turn_complete=contents[-1].role == 'user',
      )
    else:
      logger.info('no content is sent')

  async def send_content(self, content: types.Content) -> None:
    """Sends a user content to the gemini model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """
    await self._send_content(content)

  async def _send_content(
      self, content: types.Content, *, partial: bool = False
  ) -> None:
    """Sends content, optionally as a partial (non-turn-completing) update.

    Args:
      content: The content to send to the model.
      partial: Whether this content is a partial turn update that does not
        complete the model turn.
    """
    assert content.parts
    if content.parts[0].function_response:
      # All parts have to be function responses.
      function_responses = [part.function_response for part in content.parts]
      logger.debug('Sending LLM function response: %s', function_responses)
      await self._gemini_session.send_tool_response(
          function_responses=function_responses
      )
    else:
      logger.debug('Sending LLM new content %s', content)
      if (
          not partial
          and self._is_gemini_3_x_live
          and len(content.parts) == 1
          and content.parts[0].text
      ):
        logger.debug('Using send_realtime_input for Gemini 3.x Live text input')
        await self._gemini_session.send_realtime_input(
            text=content.parts[0].text
        )
      else:
        await self._gemini_session.send(
            input=types.LiveClientContent(
                turns=[content],
                turn_complete=not partial,
            )
        )

  async def send_realtime(self, input: RealtimeInput) -> None:
    """Sends a chunk of audio or a frame of video to the model in realtime.

    Args:
      input: The input to send to the model.
    """
    if isinstance(input, types.Blob):
      # The blob is binary and is very large. So let's not log it.
      logger.debug('Sending LLM Blob.')
      if self._is_gemini_3_x_live or self._is_gemini_3_5_live_translate:
        if input.mime_type and input.mime_type.startswith('audio/'):
          await self._gemini_session.send_realtime_input(audio=input)
        elif input.mime_type and input.mime_type.startswith('image/'):
          await self._gemini_session.send_realtime_input(video=input)
        else:
          logger.warning(
              'Blob not sent. Unknown or empty mime type for'
              ' send_realtime_input: %s',
              input.mime_type,
          )
      else:
        await self._gemini_session.send_realtime_input(media=input)

    elif isinstance(input, types.ActivityStart):
      logger.debug('Sending LLM activity start signal.')
      await self._gemini_session.send_realtime_input(activity_start=input)
    elif isinstance(input, types.ActivityEnd):
      logger.debug('Sending LLM activity end signal.')
      await self._gemini_session.send_realtime_input(activity_end=input)
    else:
      raise ValueError('Unsupported input type: %s' % type(input))

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Yields:
      LlmResponse: The model response.
    """
    async with Aclosing(self._gemini_session.receive()) as agen:
      # Reuse StreamingResponseAggregator to accumulate
      aggregator = StreamingResponseAggregator(
          live_session_id=self._gemini_session.session_id,
          is_gemini_3_x_live=self._is_gemini_3_x_live,
          model_version=self._model_version,
      )

      async for message in agen:
        async for resp in aggregator.process_live_server_message(message):
          yield resp
      async for resp in aggregator.close_live():
        yield resp

  async def close(self) -> None:
    """Closes the llm server connection."""

    await self._gemini_session.close()
