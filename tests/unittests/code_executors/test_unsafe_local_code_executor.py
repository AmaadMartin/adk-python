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


def _await_written_pid(pid_file) -> int | None:
  """Waits for the executed code to record the pid it forked."""
  # Waiting for the pid to be written rather than for a fixed duration: the
  # only thing that has to have happened is the fork. The file exists from the
  # moment it is opened, so its content is what is polled for.
  deadline = time.time() + 30
  while time.time() < deadline and not _written_pid(pid_file):
    time.sleep(0.05)
  return _written_pid(pid_file)


def _await_death(pid: int) -> bool:
  """Waits for `pid` to stop being a live process, returning whether it did."""
  deadline = time.time() + 10
  while time.time() < deadline and _is_alive(pid):
    time.sleep(0.05)
  return not _is_alive(pid)


def _kill_if_alive(pid: int | None) -> None:
  """Cleans up a sleeper a failing test may have left behind."""
  if pid is None:
    return
  try:
    os.kill(pid, signal.SIGKILL)
  except OSError:
    pass


def _fork_a_sleeper(pid_file, tail: str) -> str:
  """Code that forks a 60s sleeper, records its pid, then runs `tail`.

  Forked rather than spawned through `sys.executable`, so the descendant
  exists within milliseconds and the tests never wait on interpreter start-up.

  Args:
    pid_file: The path the forked pid is written to.
    tail: A single statement run once the pid has been recorded.
  """
  return textwrap.dedent(f"""
      import os
      import time

      spawned = os.fork()
      if spawned == 0:
        # Detached from the caller's stdio, so a sleeper that outlives the
        # code cannot hold the test harness's pipes open.
        os.close(0)
        os.close(1)
        os.close(2)
        time.sleep(60)
        os._exit(0)
      with open({str(pid_file)!r}, 'w') as f:
        f.write(str(spawned))
      {tail}
      """)


_needs_posix_process_groups = pytest.mark.skipif(
    not hasattr(os, "killpg")
    or not hasattr(os, "fork")
    or not os.path.isdir("/proc"),
    reason="Process-group teardown is checked on POSIX with /proc only.",
)


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

  def test_kill_execution_signals_group_before_killing_it(self, monkeypatch):
    """The group gets SIGTERM and its grace period before SIGKILL."""
    signalled = []
    monkeypatch.setattr(
        unsafe_local_code_executor.os,
        "killpg",
        lambda group, sig: signalled.append((group, sig)),
    )
    process = MagicMock()
    process.is_alive.return_value = False
    process.terminate.side_effect = lambda: signalled.append(("child", "term"))

    unsafe_local_code_executor._kill_execution(process, 4321)

    assert signalled == [
        (4321, signal.SIGTERM),
        ("child", "term"),
        (4321, signal.SIGKILL),
    ]
    process.join.assert_any_call(
        unsafe_local_code_executor._TERMINATE_GRACE_SECONDS
    )

  def test_kill_execution_without_a_group_only_kills_the_process(
      self, monkeypatch
  ):
    """An execution that never reported a group signals no group at all."""
    signalled = []
    monkeypatch.setattr(
        unsafe_local_code_executor.os,
        "killpg",
        lambda group, sig: signalled.append((group, sig)),
    )
    process = MagicMock()
    process.is_alive.return_value = False

    unsafe_local_code_executor._kill_execution(process, 0)

    assert signalled == []
    process.terminate.assert_called_once_with()
    process.join.assert_any_call(
        unsafe_local_code_executor._TERMINATE_GRACE_SECONDS
    )

  @pytest.mark.skipif(
      not hasattr(os, "setsid"), reason="Detaching a group is POSIX-only."
  )
  def test_no_group_is_reported_when_the_process_cannot_detach(
      self, monkeypatch
  ):
    """A failed detach must never report the caller's own group."""

    def _refuse_to_detach():
      raise OSError("cannot detach")

    monkeypatch.setattr(os, "setsid", _refuse_to_detach, raising=False)
    execution_group = multiprocessing.get_context("spawn").Value(
        "i", 0, lock=False
    )
    result_queue = multiprocessing.get_context("spawn").Queue()
    try:
      # Run in-process on purpose: the caller's own group is exactly what must
      # not be reported, so the test process has to be the one that detaches.
      unsafe_local_code_executor._execute_in_process(
          'print("ok")', {}, result_queue, execution_group
      )

      assert execution_group.value == 0
      assert result_queue.get(timeout=30) == ("ok\n", None)
    finally:
      result_queue.close()
      result_queue.join_thread()

  @_needs_posix_process_groups
  def test_execution_process_reports_the_group_its_code_runs_in(self):
    """The captured group is the one the executed code itself runs in."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    execution_group = ctx.Value("i", 0, lock=False)
    process = ctx.Process(
        target=unsafe_local_code_executor._execute_in_process,
        args=(
            "import os\nprint(os.getpgrp())",
            {},
            result_queue,
            execution_group,
        ),
        daemon=True,
    )
    process.start()
    try:
      stdout, error = result_queue.get(timeout=30)
      process.join(30)

      assert error is None
      assert execution_group.value == int(stdout.strip())
      assert execution_group.value != os.getpgid(0)
    finally:
      if process.is_alive():
        process.kill()
      process.join()
      result_queue.close()

  @_needs_posix_process_groups
  def test_kill_execution_kills_what_the_code_spawned(self, tmp_path):
    """Killing a live execution takes the processes it spawned with it."""
    pid_file = tmp_path / "spawned.pid"
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    execution_group = ctx.Value("i", 0, lock=False)
    process = ctx.Process(
        target=unsafe_local_code_executor._execute_in_process,
        args=(
            _fork_a_sleeper(pid_file, "time.sleep(60)"),
            {},
            result_queue,
            execution_group,
        ),
        daemon=True,
    )
    process.start()
    spawned_pid = None
    try:
      spawned_pid = _await_written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")

      unsafe_local_code_executor._kill_execution(process, execution_group.value)

      assert not process.is_alive()
      assert _await_death(spawned_pid)
    finally:
      _kill_if_alive(spawned_pid)
      if process.is_alive():
        process.kill()
      process.join()
      result_queue.close()

  @_needs_posix_process_groups
  def test_kill_execution_reaches_descendants_after_the_process_is_reaped(
      self, tmp_path
  ):
    """The group stays usable once the execution process is gone and reaped."""
    pid_file = tmp_path / "spawned.pid"
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    execution_group = ctx.Value("i", 0, lock=False)
    process = ctx.Process(
        target=unsafe_local_code_executor._execute_in_process,
        # `os._exit(0)` stands in for the ways an execution can die without
        # reporting anything: a crashed interpreter, an OOM kill, or code that
        # exits the process itself.
        args=(
            _fork_a_sleeper(pid_file, "os._exit(0)"),
            {},
            result_queue,
            execution_group,
        ),
        daemon=True,
    )
    process.start()
    spawned_pid = None
    try:
      spawned_pid = _await_written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")
      execution_pid = process.pid
      assert execution_pid is not None
      process.join(30)
      assert not process.is_alive()
      # The lookup the group used to be resolved through is gone with it, so
      # only a group captured before the execution died can still be killed.
      with pytest.raises(ProcessLookupError):
        os.getpgid(execution_pid)

      unsafe_local_code_executor._kill_execution(process, execution_group.value)

      assert _await_death(spawned_pid)
    finally:
      _kill_if_alive(spawned_pid)
      if process.is_alive():
        process.kill()
      process.join()
      result_queue.close()

  @_needs_posix_process_groups
  def test_execute_code_kills_a_process_the_code_left_running(
      self, mock_invocation_context: InvocationContext, tmp_path
  ):
    """Code that succeeds still cannot leave a process behind."""
    pid_file = tmp_path / "spawned.pid"
    executor = UnsafeLocalCodeExecutor()
    code_input = CodeExecutionInput(
        code=_fork_a_sleeper(pid_file, "print('done')")
    )
    spawned_pid = None
    try:
      result = executor.execute_code(mock_invocation_context, code_input)
      spawned_pid = _written_pid(pid_file)

      assert result.stdout == "done\n"
      assert result.stderr == ""
      assert spawned_pid is not None
      assert _await_death(spawned_pid)
    finally:
      _kill_if_alive(spawned_pid)

  @_needs_posix_process_groups
  def test_execute_code_kills_descendants_when_the_code_exits_abruptly(
      self, mock_invocation_context: InvocationContext, tmp_path
  ):
    """An execution that dies without reporting still takes its group with it."""
    pid_file = tmp_path / "spawned.pid"
    # The timeout has to outlast the spawned interpreter's start-up, or the
    # code never reaches the fork this test is about; a worker that dies
    # without reporting is only noticed when the deadline expires, so the call
    # costs the whole timeout.
    executor = UnsafeLocalCodeExecutor(timeout_seconds=10)
    code_input = CodeExecutionInput(
        code=_fork_a_sleeper(pid_file, "os._exit(0)")
    )
    spawned_pid = None
    try:
      result = executor.execute_code(mock_invocation_context, code_input)
      spawned_pid = _written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")

      # Reporting an abrupt exit as a timeout is a separate defect; this test
      # pins the teardown, so it asserts the wording as it stands today.
      assert result.stdout == ""
      assert "Code execution timed out after 10 seconds." in result.stderr
      assert _await_death(spawned_pid)
    finally:
      _kill_if_alive(spawned_pid)
