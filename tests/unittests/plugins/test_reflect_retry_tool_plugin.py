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
from unittest.mock import Mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.reflect_retry_tool_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


class MockTool(BaseTool):
  """Mock tool for testing purposes."""

  def __init__(self, name: str = "mock_tool"):
    self.name = name
    self.description = f"Mock tool named {name}"

  async def run(self, **kwargs) -> Any:
    return "mock result"


class CustomErrorExtractionPlugin(ReflectAndRetryToolPlugin):
  """Custom plugin for testing error extraction from tool responses."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.error_conditions = {}

  def set_error_condition(self, condition_func):
    """Set a custom error condition function for testing."""
    self.error_condition = condition_func

  async def extract_error_from_result(
      self, *, tool, tool_args, tool_context, result
  ):
    """Extract error based on custom conditions set for testing."""
    if hasattr(self, "error_condition"):
      return self.error_condition(result)
    return None


class TestReflectAndRetryToolPlugin:
  """Comprehensive tests for ReflectAndRetryToolPlugin focusing on behavior."""

  def get_plugin(self):
    """Create a default plugin instance for testing."""
    return ReflectAndRetryToolPlugin()

  def get_custom_plugin(self):
    """Create a plugin with custom parameters."""
    return ReflectAndRetryToolPlugin(
        name="custom_plugin",
        max_retries=5,
        throw_exception_if_retry_exceeded=False,
    )

  def get_mock_tool(self):
    """Create a mock tool for testing."""
    return MockTool("test_tool_id")

  def get_mock_tool_context(self):
    """Create a mock tool context."""
    return Mock(spec=ToolContext)

  def get_custom_error_plugin(self):
    """Create a custom error extraction plugin for testing."""
    return CustomErrorExtractionPlugin(max_retries=3)

  def get_sample_tool_args(self):
    """Sample tool arguments for testing."""
    return {"param1": "value1", "param2": 42, "param3": True}

  def test_plugin_initialization_default(self):
    """Test plugin initialization with default parameters."""
    plugin = self.get_plugin()

    assert plugin.name == "reflect_retry_tool_plugin"
    assert plugin.max_retries == 3
    assert plugin.throw_exception_if_retry_exceeded is True

  def test_plugin_initialization_custom(self):
    """Test plugin initialization with custom parameters."""
    plugin = ReflectAndRetryToolPlugin(
        name="custom_name",
        max_retries=10,
        throw_exception_if_retry_exceeded=False,
    )

    assert plugin.name == "custom_name"
    assert plugin.max_retries == 10
    assert plugin.throw_exception_if_retry_exceeded is False

  @pytest.mark.asyncio
  async def test_after_tool_callback_successful_call(self):
    """Test after_tool_callback with successful tool call."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    result = {"success": True, "data": "test_data"}

    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=result,
    )

    # Should return None for successful calls
    assert callback_result is None

  @pytest.mark.asyncio
  async def test_after_tool_callback_ignore_retry_response(self):
    """Test that retry responses are ignored in after_tool_callback."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    retry_result = {"response_type": REFLECT_AND_RETRY_RESPONSE_TYPE}

    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=retry_result,
    )

    # Retry responses should be ignored
    assert callback_result is None

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_max_retries_zero(self):
    """Test error callback when max_retries is 0.

    This should return None so that the exception is rethrown
    """
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=0)
    error = ValueError("Test error")

    with pytest.raises(ValueError, match=r"Test error") as exc_info:
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Should re-raise the original exception when max_retries is 0
    assert exc_info.value is error

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_max_retries_zero_without_exception(
      self,
  ):
    """Test error callback when max_retries is 0 and exception is disabled."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=False
    )
    error = ValueError("Test error")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Should return a retry exceeded message instead of raising
    assert result is not None
    assert result["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert result["error_type"] == "ValueError"
    assert result["retry_count"] == 0
    assert "the retry limit has been exceeded" in result["reflection_guidance"]

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_max_retries_zero_with_dict_error(self):
    """Test error callback when max_retries is 0 and error is a dict."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = CustomErrorExtractionPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=True
    )
    dict_error = {"status": "error", "message": "Custom dict error"}
    plugin.set_error_condition(lambda result: dict_error)

    with pytest.raises(Exception, match=r"Custom dict error") as exc_info:
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          result={"some": "result"},
      )

    # `_ensure_exception` wraps a non-Exception error in a plain `Exception`;
    # before it existed, `raise <dict>` produced a TypeError. Pin the exact
    # type so that regression cannot pass as a subclass match.
    assert type(exc_info.value) is Exception

  @pytest.mark.asyncio
  async def test_on_tool_error_callback_first_failure(self):
    """Test first tool failure creates reflection response."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Test error message")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    assert result is not None
    assert result["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert result["error_type"] == "ValueError"
    assert result["error_details"] == "Test error message"
    assert result["retry_count"] == 1
    assert "test_tool_id" in result["reflection_guidance"]
    assert "Test error message" in result["reflection_guidance"]

  @pytest.mark.asyncio
  async def test_retry_behavior_with_consecutive_failures(self):
    """Test the retry behavior with consecutive failures."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = RuntimeError("Runtime error")

    # First failure
    result1 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert result1["retry_count"] == 1

    # Second failure - should have different retry count based on plugin logic
    result2 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    # The plugin's internal logic determines the exact retry count
    assert result2 is not None
    assert result2["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert result2["retry_count"] == 2

  @pytest.mark.asyncio
  async def test_different_tools_behavior(self):
    """Test behavior when using different tools."""
    plugin = self.get_plugin()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    tool1 = MockTool("tool1")
    tool2 = MockTool("tool2")
    error = ValueError("Test error")

    # First failure on tool1
    result1 = await plugin.on_tool_error_callback(
        tool=tool1,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert result1["retry_count"] == 1

    # Failure on tool2
    result2 = await plugin.on_tool_error_callback(
        tool=tool2,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    # Since tool is different, retry count should start over.
    assert result2 is not None
    assert result2["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert result2["retry_count"] == 1

  @pytest.mark.asyncio
  async def test_max_retries_exceeded_with_exception(self):
    """Test that original exception is raised when max retries exceeded."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=True
    )
    error = ConnectionError("Connection failed")

    # First call should succeed and return a retry response
    await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Second call should exceed max_retries and raise
    with pytest.raises(ConnectionError, match=r"Connection failed") as exc_info:
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Verify exception properties
    assert exc_info.value is error

  @pytest.mark.asyncio
  async def test_max_retries_exceeded_with_dict_error(self):
    """Test that Exception is raised when max retries exceeded with dict error."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = CustomErrorExtractionPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=True
    )
    dict_error = {"status": "error", "message": "Custom dict error"}
    plugin.set_error_condition(lambda result: dict_error)

    # First call should fail and return a retry response
    result1 = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result={"some": "result"},
    )
    assert result1 is not None
    assert result1["retry_count"] == 1

    # Second call should exceed max_retries and raise
    with pytest.raises(Exception, match=r"Custom dict error") as exc_info:
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          result={"some": "result"},
      )

    # `_ensure_exception` wraps a non-Exception error in a plain `Exception`;
    # before it existed, `raise <dict>` produced a TypeError. Pin the exact
    # type so that regression cannot pass as a subclass match.
    assert type(exc_info.value) is Exception

  @pytest.mark.asyncio
  async def test_max_retries_exceeded_without_exception(self):
    """Test max retries exceeded returns failure message when exception is disabled."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=2, throw_exception_if_retry_exceeded=False
    )
    error = TimeoutError("Timeout occurred")

    # Call until we exceed the retry limit
    result = None
    for _ in range(3):
      result = await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Should get a retry exceeded message on the last call
    assert result is not None
    assert result["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert result["error_type"] == "TimeoutError"
    assert "the retry limit has been exceeded" in result["reflection_guidance"]
    assert "Do not attempt to use the" in result["reflection_guidance"]

  @pytest.mark.asyncio
  async def test_successful_call_resets_retry_behavior(self):
    """Test that successful calls reset the retry behavior."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Test error")

    # First failure
    result1 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert result1["retry_count"] == 1

    # Successful call
    await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result={"success": True},
    )

    # Next failure should start fresh
    result2 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert result2["retry_count"] == 1  # Should restart from 1

  @pytest.mark.asyncio
  async def test_none_result_handling(self):
    """Test handling of None results in after_tool_callback."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()

    # None result should be handled gracefully
    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=None,
    )

    assert callback_result is None

  @pytest.mark.asyncio
  async def test_empty_tool_args_handling(self):
    """Test handling of empty tool arguments."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    empty_args = {}
    error = ValueError("Test error")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=empty_args,
        tool_context=mock_tool_context,
        error=error,
    )

    assert result is not None
    # Empty args should be represented in the response
    assert "{}" in result["reflection_guidance"]

  @pytest.mark.asyncio
  async def test_retry_count_progression(self):
    """Test that retry counts progress correctly for the same tool."""
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=5)
    error = ValueError("Test error")
    tool = MockTool("single_tool")

    for i in range(1, 4):
      result = await plugin.on_tool_error_callback(
          tool=tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
      assert result["retry_count"] == i

  @pytest.mark.asyncio
  async def test_max_retries_parameter_behavior(self):
    """Test that max_retries parameter affects behavior correctly."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    # Test with very low max_retries
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )
    error = ValueError("Test error")

    # First call is fine
    await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Second call exceeds limit
    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Should hit max retries quickly with max_retries=1
    assert "the retry limit has been exceeded." in result["reflection_guidance"]

  @pytest.mark.asyncio
  async def test_default_extract_error_returns_none(self):
    """Test that default extract_error_from_result returns None."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    result = {"status": "success", "data": "some data"}

    error = await plugin.extract_error_from_result(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=result,
    )
    assert error is None

  @pytest.mark.asyncio
  async def test_custom_error_detection_and_success_handling(self):
    """Test custom error detection, success handling, and retry progression."""
    custom_error_plugin = self.get_custom_error_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    custom_error_plugin.set_error_condition(
        lambda result: result if result.get("status") == "error" else None
    )

    # Test error detection
    error_result = {"status": "error", "message": "Something went wrong"}
    callback_result = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=error_result,
    )
    assert callback_result is not None
    assert callback_result["response_type"] == REFLECT_AND_RETRY_RESPONSE_TYPE
    assert callback_result["retry_count"] == 1

    # Test success handling
    success_result = {"status": "success", "data": "operation completed"}
    callback_result = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=success_result,
    )
    assert callback_result is None

  @pytest.mark.asyncio
  async def test_retry_state_management(self):
    """Test retry state management with custom errors and mixed error types."""
    custom_error_plugin = self.get_custom_error_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    custom_error_plugin.set_error_condition(
        lambda result: result if result.get("failed") else None
    )

    # Custom error followed by exception
    custom_error = {"failed": True, "reason": "Network timeout"}
    result1 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=custom_error,
    )
    assert result1["retry_count"] == 1

    # Exception should increment retry count
    exception = ValueError("Invalid parameter")
    result2 = await custom_error_plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=exception,
    )
    assert result2["retry_count"] == 2

    # Success should reset
    success = {"result": "success"}
    result3 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=success,
    )
    assert result3 is None

    # Next error should start fresh
    result4 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=custom_error,
    )
    assert result4["retry_count"] == 1

  @pytest.mark.asyncio
  async def test_hallucinating_tool_name(self):
    """Test that hallucinating tool name is handled correctly."""
    wrong_function_call = types.Part.from_function_call(
        name="increase_by_one", args={"x": 1}
    )
    correct_function_call = types.Part.from_function_call(
        name="increase", args={"x": 1}
    )
    responses: list[types.Content] = [
        wrong_function_call,
        correct_function_call,
        "response1",
    ]
    mock_model = testing_utils.MockModel.create(responses=responses)

    function_called = 0

    def increase(x: int) -> int:
      nonlocal function_called
      function_called += 1
      return x + 1

    agent = LlmAgent(name="root_agent", model=mock_model, tools=[increase])
    runner = testing_utils.TestInMemoryRunner(
        agent=agent, plugins=[self.get_plugin()]
    )

    events = await runner.run_async_with_new_session("test")
    # Filter out agent_state events (no content).
    events = [e for e in events if e.content is not None]

    # Assert that the first event is a function call with the wrong name
    assert events[0].content.parts[0].function_call.name == "increase_by_one"

    # Assert that the second event is a function response with the
    # reflection_guidance
    assert (
        events[1].content.parts[0].function_response.response["error_type"]
        == "ValueError"
    )
    assert (
        events[1].content.parts[0].function_response.response["retry_count"]
        == 1
    )
    assert (
        "Wrong Function Name"
        in events[1]
        .content.parts[0]
        .function_response.response["reflection_guidance"]
    )

    # Assert that the third event is a function call with the correct name
    assert events[2].content.parts[0].function_call.name == "increase"
    assert function_called == 1

  def test_negative_max_retries_rejected(self):
    """Test that a negative retry budget is rejected at construction."""
    with pytest.raises(
        ValueError, match=r"max_retries must be a non-negative integer"
    ):
      ReflectAndRetryToolPlugin(max_retries=-1)

  @pytest.mark.asyncio
  async def test_reflection_response_does_not_reset_the_retry_count(self):
    """Test that feeding a reflection response back does not clear failures.

    The plugin's own reflection guidance is delivered to the model as the
    tool result, so it comes back through after_tool_callback. Treating it
    as a success would reset the counter and make the retry budget
    unenforceable.
    """
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=3)
    error = ValueError("Test error")

    reflection = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert reflection["retry_count"] == 1

    passthrough = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=reflection,
    )
    assert passthrough is None

    next_failure = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    assert next_failure["retry_count"] == 2
