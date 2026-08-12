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

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps.app import App
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from .. import testing_utils


def _assert_valid_content_union(value: object) -> None:
  """Fails if the plugin wrote something google-genai cannot serialize.

  `GenerateContentConfig.system_instruction` is typed as `ContentUnion`, so
  building a config around the value validates it against every arm of the
  union.

  Args:
    value: The rewritten system instruction to validate.
  """
  types.GenerateContentConfig(system_instruction=value)


def _make_callback_context() -> Mock:
  """Builds a callback context carrying a real session."""
  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context
  return mock_callback_context


@pytest.mark.asyncio
async def test_global_instruction_plugin_with_string():
  """Test GlobalInstructionPlugin with a string global instruction."""
  plugin = GlobalInstructionPlugin(
      global_instruction=(
          "You are a helpful assistant with a friendly personality."
      )
  )

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should now contain the global instruction
  assert (
      "You are a helpful assistant with a friendly personality."
      in llm_request.config.system_instruction
  )


@pytest.mark.asyncio
async def test_global_instruction_plugin_with_instruction_provider():
  """Test GlobalInstructionPlugin with an InstructionProvider function."""

  async def build_global_instruction(readonly_context: ReadonlyContext) -> str:
    return f"You are assistant for user {readonly_context.session.user_id}."

  plugin = GlobalInstructionPlugin(global_instruction=build_global_instruction)

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="alice", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context
  mock_callback_context.session = mock_session

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should contain the dynamically generated instruction
  assert (
      "You are assistant for user alice."
      in llm_request.config.system_instruction
  )


@pytest.mark.asyncio
async def test_global_instruction_plugin_empty_instruction():
  """Test GlobalInstructionPlugin with empty global instruction."""
  plugin = GlobalInstructionPlugin(global_instruction="")

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction="Original instruction"
      ),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should remain unchanged
  assert llm_request.config.system_instruction == "Original instruction"


@pytest.mark.asyncio
async def test_global_instruction_plugin_leads_existing():
  """Test that GlobalInstructionPlugin prepends global instructions."""
  plugin = GlobalInstructionPlugin(
      global_instruction="You are a helpful assistant."
  )

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction="Existing instructions."
      ),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should contain global instruction before existing ones
  expected = "You are a helpful assistant.\n\nExisting instructions."
  assert llm_request.config.system_instruction == expected


@pytest.mark.asyncio
async def test_global_instruction_plugin_prepends_to_list():
  """Test GlobalInstructionPlugin prepends to a list of instructions."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=["Existing instruction."]
      ),
  )

  await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  expected = ["Global instruction.", "Existing instruction."]
  assert llm_request.config.system_instruction == expected


@pytest.mark.asyncio
async def test_global_instruction_plugin_prepends_part_to_content():
  """Test GlobalInstructionPlugin prepends a Part to a Content instruction."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=types.Content(
              role="system",
              parts=[types.Part(text="Always answer in French.")],
          )
      ),
  )

  await plugin.before_model_callback(
      callback_context=_make_callback_context(), llm_request=llm_request
  )

  result = llm_request.config.system_instruction
  assert isinstance(result, types.Content)
  assert result.parts is not None
  assert all(isinstance(part, types.Part) for part in result.parts)
  assert result == types.Content(
      role="system",
      parts=[
          types.Part(text="Global instruction."),
          types.Part(text="Always answer in French."),
      ],
  )
  _assert_valid_content_union(result)


@pytest.mark.asyncio
async def test_global_instruction_plugin_prepends_to_content_without_parts():
  """Test GlobalInstructionPlugin handles a Content that carries no parts."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=types.Content(role="system", parts=None)
      ),
  )

  await plugin.before_model_callback(
      callback_context=_make_callback_context(), llm_request=llm_request
  )

  result = llm_request.config.system_instruction
  assert isinstance(result, types.Content)
  assert result.role == "system"
  assert result.parts == [types.Part(text="Global instruction.")]
  _assert_valid_content_union(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_instruction",
    [
        types.Part(text="Existing instruction."),
        types.File(name="files/existing-instruction"),
    ],
)
async def test_global_instruction_plugin_prepends_to_single_value(
    existing_instruction: types.Part | types.File,
):
  """Test GlobalInstructionPlugin wraps a bare Part or File into a list."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=existing_instruction
      ),
  )

  await plugin.before_model_callback(
      callback_context=_make_callback_context(), llm_request=llm_request
  )

  result = llm_request.config.system_instruction
  assert len(result) == 2
  assert not any(isinstance(item, tuple) for item in result)
  assert result == ["Global instruction.", existing_instruction]
  assert result[1] is existing_instruction
  _assert_valid_content_union(result)


@pytest.mark.asyncio
async def test_global_instruction_plugin_does_not_mutate_existing_content():
  """Test GlobalInstructionPlugin leaves the agent's own Content untouched."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  existing_instruction = types.Content(
      role="system", parts=[types.Part(text="Always answer in French.")]
  )
  existing_parts = existing_instruction.parts
  llm_request = LlmRequest(
      model="gemini-2.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=existing_instruction
      ),
  )

  await plugin.before_model_callback(
      callback_context=_make_callback_context(), llm_request=llm_request
  )

  assert llm_request.config.system_instruction is not existing_instruction
  assert existing_instruction.parts is existing_parts
  assert existing_instruction.parts == [
      types.Part(text="Always answer in French.")
  ]


class _ContentInstructionPlugin(BasePlugin):
  """Plugin that installs one shared Content as the system instruction.

  `LlmAgent` rejects a `system_instruction` on `generate_content_config`, so an
  earlier plugin is how a `types.Content` reaches `GlobalInstructionPlugin`.
  """

  def __init__(self, instruction: types.Content) -> None:
    super().__init__(name="content_instruction")
    self.instruction = instruction

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    llm_request.config.system_instruction = self.instruction
    return None


@pytest.mark.asyncio
async def test_global_instruction_plugin_content_reaches_the_model_request():
  """Test the Content instruction survives a run through the public Runner."""
  shared_instruction = types.Content(
      role="system", parts=[types.Part(text="Always answer in French.")]
  )
  mock_model = testing_utils.MockModel.create(responses=["Bonjour.", "Salut."])
  runner = testing_utils.InMemoryRunner(
      app=App(
          name="test_app",
          root_agent=Agent(name="root_agent", model=mock_model),
          plugins=[
              _ContentInstructionPlugin(shared_instruction),
              GlobalInstructionPlugin("You are a helpful assistant."),
          ],
      )
  )

  runner.run("Bonjour")
  runner.run("Encore")

  expected = types.Content(
      role="system",
      parts=[
          types.Part(text="You are a helpful assistant."),
          types.Part(text="Always answer in French."),
      ],
  )
  assert len(mock_model.requests) == 2
  for request in mock_model.requests:
    assert request.config.system_instruction == expected
  # The Content is shared across turns, so the plugin must leave it alone
  # rather than accumulate the global instruction on it.
  assert shared_instruction.parts == [
      types.Part(text="Always answer in French.")
  ]
