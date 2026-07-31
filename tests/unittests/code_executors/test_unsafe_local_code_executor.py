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

import multiprocessing
import sys
import textwrap
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.errors.code_executor_not_available_error import CodeExecutorNotAvailableError
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Provides a mock InvocationContext."""
  mock_agent = MagicMock(spec=BaseAgent)
  mock_session = MagicMock(spec=Session)
  mock_session_service = MagicMock(spec=BaseSessionService)
  return InvocationContext(
      invocation_id="test_invocation",
      agent=mock_agent,
      session=mock_session,
      session_service=mock_session_service,
  )


class TestUnsafeLocalCodeExecutor:

  def test_init_default(self):
    executor = UnsafeLocalCodeExecutor()
    assert not executor.stateful
    assert not executor.optimize_data_file

  def test_init_stateful_raises_error(self):
    with pytest.raises(
        ValueError,
        match="Cannot set `stateful=True` in UnsafeLocalCodeExecutor.",
    ):
      UnsafeLocalCodeExecutor(stateful=True)

  def test_init_optimize_data_file_raises_error(self):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor."
        ),
    ):
      UnsafeLocalCodeExecutor(optimize_data_file=True)

  def test_execute_code_simple_print(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == "hello world\n"
    assert result.stderr == ""
    assert result.output_files == []

  def test_execute_code_with_error(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='raise ValueError("Test error")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == ""
    assert "Test error" in result.stderr
    assert result.output_files == []

  def test_execute_code_variable_assignment(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="x = 10\nprint(x * 2)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "20\n"
    assert result.stderr == ""

  def test_execute_code_empty(self, mock_invocation_context: InvocationContext):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code="")
    result = executor.execute_code(mock_invocation_context, code_input)
    assert result.stdout == ""
    assert result.stderr == ""

  def test_execute_code_nested_function_call(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code=(textwrap.dedent("""
                def helper(name):
                  return f'hi {name}'

                def run():
                  print(helper('ada'))

                run()
                """)))

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stderr == ""
    assert result.stdout == "hi ada\n"

  def test_execute_code_timeout(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor(timeout_seconds=1)
    code_input = CodeExecutionInput(code="import time\ntime.sleep(2)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "Code execution timed out after 1 seconds." in result.stderr

  def test_execute_code_raises_when_worker_queue_cannot_be_created(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A sandbox forbidding the worker's IPC queue yields an actionable error."""
    broken_pipe = BrokenPipeError(32, "Broken pipe")

    class _NoQueueContext:

      def Queue(self):
        raise broken_pipe

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: _NoQueueContext()
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    with pytest.raises(
        RuntimeError, match="could not start a worker process"
    ) as exc_info:
      executor.execute_code(mock_invocation_context, code_input)

    assert "BrokenPipeError" in str(exc_info.value)
    assert exc_info.value.__cause__ is broken_pipe

  def test_execute_code_raises_when_worker_process_cannot_start(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A sandbox that forbids spawning the worker yields an actionable error."""

    class _NoStartContext:

      def Queue(self):
        return MagicMock()

      def Process(self, **kwargs):
        process = MagicMock()
        process.start.side_effect = PermissionError(
            1, "Operation not permitted"
        )
        return process

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: _NoStartContext()
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    with pytest.raises(
        RuntimeError, match="could not start a worker process"
    ) as exc_info:
      executor.execute_code(mock_invocation_context, code_input)

    assert "PermissionError" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, PermissionError)

  def test_execute_code_propagates_non_oserror_setup_failure(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A start-method error that is not an OSError is not rewritten."""
    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        MagicMock(side_effect=ValueError("cannot find context for spawn")),
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    with pytest.raises(ValueError, match="cannot find context"):
      executor.execute_code(mock_invocation_context, code_input)

  def test_execute_code_closes_queue_when_worker_cannot_start(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A worker that cannot start reports the typed error without leaking."""
    result_queue = MagicMock()
    no_memory = OSError("Cannot allocate memory")

    class _NoStartContext:

      def Queue(self):
        return result_queue

      def Process(self, **kwargs):
        process = MagicMock()
        process.start.side_effect = no_memory
        return process

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: _NoStartContext()
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    with pytest.raises(
        CodeExecutorNotAvailableError, match="may not permit multiprocessing"
    ) as exc_info:
      executor.execute_code(mock_invocation_context, code_input)

    # RuntimeError stays the base class so existing handlers keep working.
    assert isinstance(exc_info.value, RuntimeError)
    assert exc_info.value.__cause__ is no_memory
    result_queue.close.assert_called_once_with()

  def test_execute_code_reports_unavailable_without_interpreter_path(
      self, mock_invocation_context: InvocationContext
  ):
    """An interpreter that cannot be re-invoked is reported, not left to fail.

    Drives the real multiprocessing machinery rather than a fake context, so
    the up-front check is what has to produce the error.
    """
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')
    original_executable = sys.executable
    multiprocessing.set_executable("")

    try:
      with pytest.raises(
          CodeExecutorNotAvailableError, match="no interpreter path to"
      ):
        executor.execute_code(mock_invocation_context, code_input)
    finally:
      multiprocessing.set_executable(original_executable)
