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

"""Handles basic information to build the LLM request."""

from __future__ import annotations

from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...utils import model_name_utils
from ...utils.output_schema_utils import can_use_output_schema_with_tools
from ._base_llm_processor import BaseLlmRequestProcessor


def _merge_run_config_http_options(
    config: types.GenerateContentConfig,
    run_config_http_options: types.HttpOptions,
) -> None:
  """Merges RunConfig http_options into the request config, RunConfig wins.

  base_url and api_version are configuration-time settings, not request-time,
  so they are intentionally not merged here.
  """
  if config.http_options is None:
    config.http_options = run_config_http_options
    return

  if run_config_http_options.headers:
    if config.http_options.headers is None:
      config.http_options.headers = {}
    config.http_options.headers.update(run_config_http_options.headers)

  for field in ('timeout', 'retry_options', 'extra_body'):
    value = getattr(run_config_http_options, field, None)
    if value is not None:
      setattr(config.http_options, field, value)


def _build_basic_request(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Populate basic LlmRequest fields from agent configuration.

  Sets up model, config, output_schema, and live connect configuration
  based on the agent and run configuration.

  Args:
    invocation_context: The invocation context containing agent and run config.
    llm_request: The LlmRequest to populate.
  """
  agent = invocation_context.agent
  model = agent.canonical_model
  llm_request.model = model if isinstance(model, str) else model.model

  # Preserved across the agent-config overwrite below, then merged back.
  run_config_http_options = llm_request.config.http_options

  llm_request.config = (
      agent.generate_content_config.model_copy(deep=True)
      if agent.generate_content_config
      else types.GenerateContentConfig()
  )

  if run_config_http_options:
    _merge_run_config_http_options(llm_request.config, run_config_http_options)
  # Only set output_schema if no tools are specified. as of now, model don't
  # support output_schema and tools together. we have a workaround to support
  # both output_schema and tools at the same time. see
  # _output_schema_processor.py for details
  #
  # task-mode agents skip output_schema configuration in
  # the basic flow. Structured output for tasks is collected via the
  # finish_task tool schema instead.
  if getattr(agent, 'mode', None) != 'task' and agent.output_schema:
    if not agent.tools or can_use_output_schema_with_tools(model):
      llm_request.set_output_schema(agent.output_schema)

  llm_request.live_connect_config.response_modalities = (
      [
          types.Modality(m)
          for m in invocation_context.run_config.response_modalities
      ]
      if invocation_context.run_config.response_modalities is not None
      else None
  )
  llm_request.live_connect_config.speech_config = (
      invocation_context.run_config.speech_config
  )
  llm_request.live_connect_config.output_audio_transcription = (
      invocation_context.run_config.output_audio_transcription
  )
  llm_request.live_connect_config.input_audio_transcription = (
      invocation_context.run_config.input_audio_transcription
  )
  llm_request.live_connect_config.realtime_input_config = (
      invocation_context.run_config.realtime_input_config
  )
  llm_request.live_connect_config.explicit_vad_signal = (
      invocation_context.run_config.explicit_vad_signal
  )
  llm_request.live_connect_config.translation_config = (
      invocation_context.run_config.translation_config
  )
  active_model_name = (
      getattr(getattr(agent, 'canonical_live_model', None), 'model', None)
      or llm_request.model
  )
  is_gemini_3_x = model_name_utils._is_gemini_3_x_live(active_model_name)
  llm_request.live_connect_config.enable_affective_dialog = (
      None
      if is_gemini_3_x
      else invocation_context.run_config.enable_affective_dialog
  )
  llm_request.live_connect_config.proactivity = (
      None if is_gemini_3_x else invocation_context.run_config.proactivity
  )
  llm_request.live_connect_config.session_resumption = (
      invocation_context.run_config.session_resumption
  )
  llm_request.live_connect_config.history_config = (
      invocation_context.run_config.history_config
  )
  llm_request.live_connect_config.context_window_compression = (
      invocation_context.run_config.context_window_compression
  )
  llm_request.live_connect_config.avatar_config = (
      invocation_context.run_config.avatar_config
  )


import inspect


def _mark_live_async_tools_non_blocking(llm_request: LlmRequest) -> None:
  """Marks live streaming and response-scheduling tools as NON_BLOCKING.

  These tools emit asynchronous FunctionResponses, which the Live API only
  accepts for NON_BLOCKING declarations.
  """
  if not llm_request.config.tools:
    return
  for gemini_tool in llm_request.config.tools:
    for declaration in gemini_tool.function_declarations or []:
      tool = llm_request.tools_dict.get(declaration.name)
      if tool is None:
        continue
      is_streaming_tool = inspect.isasyncgenfunction(getattr(tool, 'func', None))
      if tool.response_scheduling is not None or is_streaming_tool:
        declaration.behavior = types.Behavior.NON_BLOCKING


class _BasicLlmRequestProcessor(BaseLlmRequestProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    _build_basic_request(invocation_context, llm_request)

    agent = invocation_context.agent
    if agent is None or not hasattr(agent, 'tools') or not agent.tools:
      return

    from ...utils.context_utils import Aclosing
    from .base_llm_flow import _resolve_toolset_auth

    # Resolve toolset authentication before tool listing.
    # This ensures credentials are ready before get_tools() is called.
    async with Aclosing(
        _resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event

    if invocation_context.end_invocation:
      return

    multiple_tools = len(agent.tools) > 1
    model = agent.canonical_model

    import asyncio
    from ...agents.llm_agent import _convert_tool_union_to_tools
    from ...agents.readonly_context import ReadonlyContext
    from ...tools.base_toolset import BaseToolset
    from ...tools.tool_context import ToolContext

    resolved_tools_per_union = await asyncio.gather(*(
        _convert_tool_union_to_tools(
            tool_union,
            ReadonlyContext(invocation_context),
            model,
            multiple_tools,
        )
        for tool_union in agent.tools
    ))

    for tool_union, tools in zip(agent.tools, resolved_tools_per_union):
      tool_context = ToolContext(invocation_context)

      # If it's a toolset, process it first
      if isinstance(tool_union, BaseToolset):
        await tool_union.process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

      # Then process all tools from this tool union
      for tool in tools:
        llm_request.append_tools([tool])
        await tool.process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

    if invocation_context.live_request_queue is not None:
      _mark_live_async_tools_non_blocking(llm_request)


request_processor = _BasicLlmRequestProcessor()
