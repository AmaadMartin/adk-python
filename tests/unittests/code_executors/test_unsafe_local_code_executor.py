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

import logging
import multiprocessing
import multiprocessing.spawn
import sys
import textwrap
from typing import Any
from typing import Callable
from typing import Optional
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import unsafe_local_code_executor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
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


class _NeverStartedProcess:
  """A worker process whose `start()` fails, recording teardown attempts."""

  def __init__(self, start_error: OSError):
    self._start_error = start_error
    self.terminate_calls = 0
    self.join_calls = 0

  def start(self) -> None:
    raise self._start_error

  def terminate(self) -> None:
    self.terminate_calls += 1

  def join(self, timeout: Optional[float] = None) -> None:
    del timeout  # Unused; the worker never started.
    self.join_calls += 1


class _FailingLaunchContext:
  """A spawn context handing out a worker process that cannot be started."""

  def __init__(self, start_error: OSError, queue_factory: Callable[[], Any]):
    self._start_error = start_error
    self._queue_factory = queue_factory
    self.process: Optional[_NeverStartedProcess] = None

  # `Queue` and `Process` mirror the `multiprocessing` context API the
  # executor calls, hence the capitalised names.
  def Queue(self) -> Any:
    return self._queue_factory()

  def Process(self, **kwargs: Any) -> _NeverStartedProcess:
    del kwargs  # The worker is never started, so its target is irrelevant.
    self.process = _NeverStartedProcess(self._start_error)
    return self.process


class _RecordingQueue:
  """Delegates to a real queue, counting the cleanup calls made on it."""

  def __init__(self, delegate: Any):
    self._delegate = delegate
    self.close_calls = 0
    self.join_thread_calls = 0

  def close(self) -> None:
    self.close_calls += 1
    self._delegate.close()

  def join_thread(self) -> None:
    self.join_thread_calls += 1
    self._delegate.join_thread()


def _patch_failing_launch(
    monkeypatch: pytest.MonkeyPatch,
    start_error: OSError,
    queue_factory: Optional[Callable[[], Any]] = None,
) -> _FailingLaunchContext:
  """Makes the spawn context hand back a worker that cannot be started.

  The real spawn `Queue` is captured before patching so the executor still
  operates on a genuine queue, which keeps the cleanup assertions honest.
  """
  fake_context = _FailingLaunchContext(
      start_error,
      queue_factory or multiprocessing.get_context("spawn").Queue,
  )
  monkeypatch.setattr(
      unsafe_local_code_executor.multiprocessing,
      "get_context",
      lambda method: fake_context,
  )
  return fake_context


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

  def test_execute_code_reports_worker_launch_failure(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A worker that cannot be spawned is reported, not raised."""
    _patch_failing_launch(monkeypatch, BrokenPipeError(32, "Broken pipe"))
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == ""
    assert result.output_files == []
    assert "cannot spawn a child process" in result.stderr
    assert "not an error in the code" in result.stderr
    assert sys.executable in result.stderr
    assert "ContainerCodeExecutor" in result.stderr
    assert "VertexAiCodeExecutor" in result.stderr
    assert "BrokenPipeError" in result.stderr
    assert "Broken pipe" in result.stderr

  def test_execute_code_reports_launch_failure_for_a_forbidden_sandbox(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A sandbox that forbids spawning gets the same diagnostic."""
    _patch_failing_launch(
        monkeypatch, PermissionError(1, "Operation not permitted")
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert result.output_files == []
    assert "cannot spawn a child process" in result.stderr
    assert "not an error in the code" in result.stderr
    assert "ContainerCodeExecutor" in result.stderr
    assert "VertexAiCodeExecutor" in result.stderr
    assert "PermissionError" in result.stderr
    assert "Operation not permitted" in result.stderr

  def test_execute_code_reports_launch_failure_when_the_queue_fails(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """A sandbox forbidding the worker's IPC queue gets the same diagnostic."""
    broken_pipe = BrokenPipeError(32, "Broken pipe")

    def _raise_broken_pipe() -> Any:
      raise broken_pipe

    _patch_failing_launch(
        monkeypatch, broken_pipe, queue_factory=_raise_broken_pipe
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "cannot spawn a child process" in result.stderr
    assert "BrokenPipeError" in result.stderr

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

  def test_execute_code_closes_the_queue_when_the_launch_fails(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """The queue created before the failed launch is released, not leaked."""
    recording_queue = _RecordingQueue(
        multiprocessing.get_context("spawn").Queue()
    )
    _patch_failing_launch(
        monkeypatch,
        OSError("Cannot allocate memory"),
        queue_factory=lambda: recording_queue,
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert "Cannot allocate memory" in result.stderr
    assert recording_queue.close_calls == 1
    assert recording_queue.join_thread_calls == 1

  def test_execute_code_logs_the_launch_failure(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
      caplog: pytest.LogCaptureFixture,
  ):
    """The cause survives for operators even though nothing is raised."""
    _patch_failing_launch(monkeypatch, BrokenPipeError(32, "Broken pipe"))
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')
    logger_name = unsafe_local_code_executor.logger.name

    with caplog.at_level(logging.ERROR, logger=logger_name):
      executor.execute_code(mock_invocation_context, code_input)

    records = [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.ERROR
    ]
    assert len(records) == 1
    assert logger_name.startswith("google_adk.")
    assert records[0].exc_info is not None

  def test_execute_code_does_not_tear_down_a_worker_that_never_started(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
  ):
    """The timeout teardown must not run for a process that never started."""
    fake_context = _patch_failing_launch(
        monkeypatch, PermissionError(1, "Operation not permitted")
    )
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    executor.execute_code(mock_invocation_context, code_input)

    assert fake_context.process is not None
    assert fake_context.process.terminate_calls == 0
    assert fake_context.process.join_calls == 0

  def test_execute_code_reports_launch_failure_without_interpreter_path(
      self, mock_invocation_context: InvocationContext
  ):
    """An interpreter that cannot be re-invoked is reported, not left to fail.

    Drives the real multiprocessing machinery rather than a fake context, so
    the up-front check is what has to produce the diagnostic.
    """
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')
    original_executable = multiprocessing.spawn.get_executable()
    multiprocessing.set_executable("")

    try:
      result = executor.execute_code(mock_invocation_context, code_input)
    finally:
      multiprocessing.set_executable(original_executable)

    assert result.stdout == ""
    assert result.output_files == []
    assert "no interpreter path to re-invoke" in result.stderr
    assert "ContainerCodeExecutor" in result.stderr
    assert "VertexAiCodeExecutor" in result.stderr
