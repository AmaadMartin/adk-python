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
import os
import signal
import textwrap
import time
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


def _written_pid(pid_file) -> int | None:
  """Returns the pid the executed code recorded, or None if it has not yet."""
  try:
    recorded = pid_file.read_text().strip()
  except OSError:
    return None
  return int(recorded) if recorded else None


def _is_alive(pid: int) -> bool:
  """Returns whether `pid` is a live (non-zombie) process."""
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
      state = stat_file.read().rsplit(")", 1)[1].split()[0]
  except OSError:
    return False
  return state != "Z"


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

  def test_execute_code_returns_when_the_process_outlives_its_result(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
      caplog: pytest.LogCaptureFixture,
  ):
    """Code that leaves a thread running still returns its output promptly."""
    monkeypatch.setattr(
        unsafe_local_code_executor, "_RESULT_JOIN_GRACE_SECONDS", 0.1
    )
    caplog.set_level(logging.WARNING)
    executor = UnsafeLocalCodeExecutor(timeout_seconds=30)
    code_input = CodeExecutionInput(code=textwrap.dedent("""
        import threading
        import time

        threading.Thread(target=time.sleep, args=(60,)).start()
        print('done')
        """))

    started = time.monotonic()
    result = executor.execute_code(mock_invocation_context, code_input)
    elapsed = time.monotonic() - started

    assert result.stdout == "done\n"
    assert result.stderr == ""
    # The thread sleeps for 60 seconds; waiting on it would blow this bound.
    assert elapsed < 30
    assert any("did not exit" in r.getMessage() for r in caplog.records)

  @pytest.mark.skipif(
      os.name != "posix", reason="Process teardown is checked on POSIX only."
  )
  def test_execute_code_reaps_a_process_that_overstays_its_grace(
      self,
      mock_invocation_context: InvocationContext,
      monkeypatch: pytest.MonkeyPatch,
      tmp_path,
  ):
    """An execution that will not finish exiting is killed, not leaked."""
    monkeypatch.setattr(
        unsafe_local_code_executor, "_RESULT_JOIN_GRACE_SECONDS", 0.1
    )
    pid_file = tmp_path / "execution.pid"
    executor = UnsafeLocalCodeExecutor(timeout_seconds=30)
    code_input = CodeExecutionInput(code=textwrap.dedent(f"""
        import os
        import threading
        import time

        with open({str(pid_file)!r}, 'w') as f:
          f.write(str(os.getpid()))
        threading.Thread(target=time.sleep, args=(60,)).start()
        print('done')
        """))

    result = executor.execute_code(mock_invocation_context, code_input)

    # The execution ran to completion, so it recorded its pid before hanging
    # around.
    assert result.stdout == "done\n"
    execution_pid = _written_pid(pid_file)
    assert execution_pid is not None
    try:
      # `_kill_execution` ends with a join, so the pid of this direct child is
      # released. A child that was killed but not reaped would still answer
      # signal 0 as a zombie.
      with pytest.raises(ProcessLookupError):
        os.kill(execution_pid, 0)
    finally:
      try:
        os.kill(execution_pid, signal.SIGKILL)
      except OSError:
        pass

  def test_execute_code_does_not_report_a_process_that_exits_promptly(
      self,
      mock_invocation_context: InvocationContext,
      caplog: pytest.LogCaptureFixture,
  ):
    """A well-behaved execution is not reported as overstaying its grace."""
    caplog.set_level(logging.WARNING)
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(code='print("hi")')

    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "hi\n"
    assert result.stderr == ""
    assert not any("did not exit" in r.getMessage() for r in caplog.records)

  def test_kill_execution_signals_group_before_killing_it(self, monkeypatch):
    """The group gets SIGTERM and its grace period before SIGKILL."""
    signalled = []
    monkeypatch.setattr(
        unsafe_local_code_executor.os,
        "killpg",
        lambda group, sig: signalled.append((group, sig)),
    )
    monkeypatch.setattr(
        unsafe_local_code_executor,
        "_execution_group",
        lambda process: 4321,
    )
    process = MagicMock()
    process.is_alive.return_value = False
    process.terminate.side_effect = lambda: signalled.append(("child", "term"))

    unsafe_local_code_executor._kill_execution(process)

    assert signalled == [
        (4321, signal.SIGTERM),
        ("child", "term"),
        (4321, signal.SIGKILL),
    ]
    process.join.assert_any_call(
        unsafe_local_code_executor._TERMINATE_GRACE_SECONDS
    )

  @pytest.mark.skipif(
      not hasattr(os, "killpg")
      or not hasattr(os, "fork")
      or not os.path.isdir("/proc"),
      reason="Process-group teardown is checked on POSIX with /proc only.",
  )
  def test_kill_execution_kills_what_the_code_spawned(self, tmp_path):
    """Killing a live execution takes the processes it spawned with it."""
    pid_file = tmp_path / "spawned.pid"
    # Forked rather than spawned through `sys.executable`, so the descendant
    # exists within milliseconds and the test never waits on interpreter
    # start-up.
    code = textwrap.dedent(f"""
        import os
        import time

        spawned = os.fork()
        if spawned == 0:
          time.sleep(60)
          os._exit(0)
        with open({str(pid_file)!r}, 'w') as f:
          f.write(str(spawned))
        time.sleep(60)
        """)
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=unsafe_local_code_executor._execute_in_process,
        args=(code, {}, result_queue),
        daemon=True,
    )
    process.start()
    spawned_pid = None
    try:
      # Waiting for the pid to be written rather than for a fixed duration:
      # the only thing that has to have happened is the fork. The file exists
      # from the moment it is opened, so its content is what is polled for.
      deadline = time.time() + 30
      while time.time() < deadline and not _written_pid(pid_file):
        time.sleep(0.05)
      spawned_pid = _written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")

      unsafe_local_code_executor._kill_execution(process)

      assert not process.is_alive()
      deadline = time.time() + 10
      while time.time() < deadline and _is_alive(spawned_pid):
        time.sleep(0.05)
      assert not _is_alive(spawned_pid)
    finally:
      if spawned_pid is not None:
        try:
          os.kill(spawned_pid, signal.SIGKILL)
        except OSError:
          pass
      if process.is_alive():
        process.kill()
      process.join()
      result_queue.close()
