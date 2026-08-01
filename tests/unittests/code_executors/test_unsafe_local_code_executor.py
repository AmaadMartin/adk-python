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

import queue
import signal
import textwrap
import threading
import time
from typing import Optional
from unittest.mock import MagicMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.unsafe_local_code_executor import _collect_result
from google.adk.code_executors.unsafe_local_code_executor import _WORKER_POLL_SECONDS
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


class _FakeProcess:
  """A stand-in for a started worker process, in whatever state a test needs."""

  def __init__(self, *, alive: bool, exitcode: Optional[int] = None):
    self._alive = alive
    self.exitcode = exitcode
    self.terminate_calls = 0
    self.join_calls = 0

  def is_alive(self) -> bool:
    return self._alive

  def terminate(self) -> None:
    self.terminate_calls += 1
    self._alive = False

  def join(self, timeout: Optional[float] = None) -> None:
    del timeout  # The executor always joins an already-finished process.
    self.join_calls += 1


class _PollRecordingQueue(queue.Queue[tuple[str, Optional[str]]]):
  """A queue that records the timeouts it is polled with.

  The first `empty_reads` reads come up empty however full the queue is,
  standing in for the instant in which the worker has been reaped but the
  result it wrote is still travelling through the queue's feeder thread and
  pipe.
  """

  def __init__(self, empty_reads: int = 0):
    super().__init__()
    self.timeouts: list[Optional[float]] = []
    self._empty_reads = empty_reads

  def get(
      self, block: bool = True, timeout: Optional[float] = None
  ) -> tuple[str, Optional[str]]:
    self.timeouts.append(timeout)
    if len(self.timeouts) <= self._empty_reads:
      raise queue.Empty
    return super().get(block=block, timeout=timeout)


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

  def test_execute_code_worker_exit_is_not_reported_as_a_timeout(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor(timeout_seconds=30)
    code_input = CodeExecutionInput(code="import os\nos._exit(3)")

    started = time.monotonic()
    result = executor.execute_code(mock_invocation_context, code_input)
    elapsed = time.monotonic() - started

    assert result.stdout == ""
    assert "exited with code 3" in result.stderr
    assert "timed out" not in result.stderr
    # The wait ended on the worker's death, not on the 30 second deadline.
    assert elapsed < 15

  @pytest.mark.skipif(
      not hasattr(signal, "SIGKILL"), reason="SIGKILL is POSIX-only."
  )
  def test_execute_code_worker_killed_by_signal_reports_the_signal_exit_code(
      self, mock_invocation_context: InvocationContext
  ):
    # Standing in for the OOM killer, which reaps the worker the same way.
    executor = UnsafeLocalCodeExecutor(timeout_seconds=30)
    code_input = CodeExecutionInput(
        code="import os, signal\nos.kill(os.getpid(), signal.SIGKILL)"
    )

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "exited with code -9" in result.stderr
    assert "timed out" not in result.stderr

  def test_execute_code_worker_exit_does_not_hang_without_a_timeout(
      self, mock_invocation_context: InvocationContext
  ):
    executor = UnsafeLocalCodeExecutor()
    assert executor.timeout_seconds is None
    code_input = CodeExecutionInput(code="import os\nos._exit(7)")
    results: list[CodeExecutionResult] = []

    # A daemon thread keeps a regression from wedging the whole test session:
    # pytest-timeout is not available to fail the hang for us.
    thread = threading.Thread(
        target=lambda: results.append(
            executor.execute_code(mock_invocation_context, code_input)
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "execute_code() never returned."
    assert "exited with code 7" in results[0].stderr


class TestCollectResult:
  """Covers the wait's branches that a real worker cannot force reliably."""

  def test_collect_result_returns_a_result_the_dead_worker_already_wrote(self):
    process = _FakeProcess(alive=False, exitcode=0)
    result_queue = _PollRecordingQueue(empty_reads=1)
    result_queue.put(("out\n", None))

    assert _collect_result(process, result_queue, None) == ("out\n", "")
    # The first read came up empty; the result was only found by the second.
    assert len(result_queue.timeouts) == 2

  def test_collect_result_reports_the_exit_code_when_nothing_arrives(self):
    process = _FakeProcess(alive=False, exitcode=3)

    stdout, stderr = _collect_result(process, _PollRecordingQueue(), None)

    assert stdout == ""
    assert stderr == (
        "Code execution process exited with code 3 without returning a result."
    )

  def test_collect_result_kills_a_live_worker_at_the_deadline(self):
    process = _FakeProcess(alive=True)
    result_queue = _PollRecordingQueue()

    stdout, stderr = _collect_result(process, result_queue, 0.05)

    assert stdout == ""
    assert stderr == "Code execution timed out after 0.05 seconds."
    assert process.terminate_calls == 1
    assert process.join_calls == 1
    # No poll outlives the deadline it is counting down to.
    assert result_queue.timeouts
    assert all(
        poll is not None and poll < _WORKER_POLL_SECONDS
        for poll in result_queue.timeouts
    )
