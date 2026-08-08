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
import sys
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


# Linux keeps a /proc entry for a short while after a process is gone: "Z"
# while it waits to be reaped by its parent, then "X" ("x" on older kernels)
# once it has been reaped and is being torn down. Neither is a running process.
_DEAD_STATES = frozenset("ZXx")


def _shows_a_live_process(stat: str) -> bool:
  """Returns whether a /proc/<pid>/stat line describes a running process."""
  # The state is the field after the final ")": the process name is
  # parenthesised and may itself contain parentheses.
  fields = stat.rpartition(")")[2].split()
  # An entry torn down between opening and reading it comes back empty.
  return bool(fields) and fields[0] not in _DEAD_STATES


def _is_alive(pid: int) -> bool:
  """Returns whether `pid` is a process that has not exited."""
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
      return _shows_a_live_process(stat_file.read())
  except OSError:
    return False


def _has_exited(pid: int, timeout_seconds: float) -> bool:
  """Waits for `pid` to exit, and returns the reading the wait settled on.

  The caller must not read /proc again: that samples a different instant, and
  a process on its way out moves between states the wait has already accepted
  as dead. A settled reading is enough, because a process that has exited does
  not come back.
  """
  deadline = time.monotonic() + timeout_seconds
  exited = not _is_alive(pid)
  while not exited and time.monotonic() < deadline:
    time.sleep(0.05)
    exited = not _is_alive(pid)
  return exited


@pytest.mark.parametrize(
    "stat,expected",
    [
        ("4321 (python3) R 1 4321 4321 0 -1", True),
        ("4321 (python3) S 1 4321 4321 0 -1", True),
        ("4321 (python3) D 1 4321 4321 0 -1", True),
        # A name with parentheses in it must not be mistaken for the state.
        ("4321 (py (3)) S 1 4321 4321 0 -1", True),
        ("4321 (py (3)) Z 1 4321 4321 0 -1", False),
        # Exited, waiting for its parent to reap it.
        ("4321 (python3) Z 1 4321 4321 0 -1", False),
        # Reaped, /proc entry not yet torn down.
        ("4321 (python3) X 0 -1 -1 0 -1", False),
        # The same state, as older kernels spell it.
        ("4321 (python3) x 0 -1 -1 0 -1", False),
        # The entry disappeared between being opened and being read.
        ("", False),
    ],
)
def test_only_a_process_that_has_not_exited_counts_as_alive(
    stat: str, expected: bool
):
  """A zombie, a reaped process and a vanished entry are all not alive."""
  assert _shows_a_live_process(stat) is expected


def test_a_pid_with_no_proc_entry_is_not_alive():
  """A pid the kernel has no entry for at all is reported as not alive."""
  assert not _is_alive(-1)


def test_the_wait_keeps_the_reading_it_settled_on(monkeypatch):
  """A later reading cannot overturn the one the wait settled on."""
  # The third reading is what a re-read after the wait would have consumed.
  readings = iter([True, False, True])
  monkeypatch.setattr(
      sys.modules[__name__], "_is_alive", lambda pid: next(readings)
  )

  assert _has_exited(4321, 10)


def test_the_wait_gives_up_when_the_process_outlives_the_deadline():
  """A process that never exits is reported as still running."""
  assert not _has_exited(os.getpid(), 0)


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
      deadline = time.monotonic() + 30
      while time.monotonic() < deadline and not _written_pid(pid_file):
        time.sleep(0.05)
      spawned_pid = _written_pid(pid_file)
      if spawned_pid is None:
        pytest.skip("this environment could not start the execution process")

      # A fork that had already died would make the assertion below vacuous.
      assert _is_alive(spawned_pid), "the code's fork was not running"

      unsafe_local_code_executor._kill_execution(process)

      assert not process.is_alive()
      assert _has_exited(
          spawned_pid, 10
      ), f"pid {spawned_pid} outlived the process-group teardown"
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
