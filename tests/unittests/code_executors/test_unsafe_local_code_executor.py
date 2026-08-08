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

import json
import logging
import multiprocessing
import multiprocessing.spawn
import os
import signal
import subprocess
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


def _is_alive(pid: int) -> bool:
  """Returns whether `pid` is a live (non-zombie) process."""
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
      state = stat_file.read().rsplit(")", 1)[1].split()[0]
  except OSError:
    return False
  return state != "Z"


# Runs the reported scenario in a child interpreter: the launch failure only
# surfaces while the multiprocessing resource tracker has not been started yet,
# and a dead tracker must not be left behind in the pytest worker.
_UNUSABLE_INTERPRETER_SCRIPT = textwrap.dedent("""
    import json
    import multiprocessing
    import sys
    from unittest.mock import MagicMock

    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.code_executors.code_execution_utils import CodeExecutionInput
    from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
    from google.adk.sessions.base_session_service import BaseSessionService
    from google.adk.sessions.session import Session

    multiprocessing.set_executable(sys.argv[1])
    result = UnsafeLocalCodeExecutor().execute_code(
        InvocationContext(
            invocation_id="unusable_interpreter",
            agent=MagicMock(spec=BaseAgent),
            session=MagicMock(spec=Session),
            session_service=MagicMock(spec=BaseSessionService),
        ),
        CodeExecutionInput(code='print("hello world")'),
    )
    print(json.dumps({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_file_count": len(result.output_files),
        "sys_executable": sys.executable,
    }))
    """)

# Generous: the child only has to import the package and fail to spawn. It
# bounds a regression that would otherwise hang the run.
_CHILD_TIMEOUT_SECONDS = 120


# Held so the fake context can build a real queue while the module-level
# `get_context` is patched out.
_SPAWN_CONTEXT = multiprocessing.get_context("spawn")


class _RecordingQueue:
  """Wraps a real spawn queue, recording the release calls made on it."""

  def __init__(self, calls: list[str]):
    self._queue = _SPAWN_CONTEXT.Queue()
    self._calls = calls

  def close(self) -> None:
    self._calls.append("close")
    self._queue.close()

  def join_thread(self) -> None:
    self._calls.append("join_thread")
    self._queue.join_thread()


class _FakeProcess:
  """Records the lifecycle calls made on a worker that never starts."""

  def __init__(self, start_error: OSError | None):
    self._start_error = start_error
    self.calls: list[str] = []

  def start(self) -> None:
    self.calls.append("start")
    if self._start_error is not None:
      raise self._start_error

  def terminate(self) -> None:
    self.calls.append("terminate")

  def join(self, timeout: float | None = None) -> None:
    self.calls.append("join")

  def kill(self) -> None:
    self.calls.append("kill")


class _FakeSpawnContext:
  """A spawn context that fails the way an unusable interpreter makes it fail."""

  def __init__(
      self,
      *,
      queue_error: OSError | None = None,
      start_error: OSError | None = None,
  ):
    self._queue_error = queue_error
    self.queue_calls: list[str] = []
    self.process = _FakeProcess(start_error)

  # `Queue` and `Process` are capitalised to mirror the multiprocessing
  # context API the executor calls.
  def Queue(self) -> _RecordingQueue:
    if self._queue_error is not None:
      raise self._queue_error
    return _RecordingQueue(self.queue_calls)

  def Process(self, **kwargs: object) -> _FakeProcess:
    return self.process


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

  def test_execute_code_reports_the_unusable_interpreter(self, tmp_path):
    """The reported bug, end to end against the real multiprocessing stack."""
    interpreter = tmp_path / "no-such-python"

    completed = subprocess.run(
        [sys.executable, "-c", _UNUSABLE_INTERPRETER_SCRIPT, str(interpreter)],
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["stdout"] == ""
    assert result["output_file_count"] == 0
    assert str(interpreter) in result["stderr"]
    # The interpreter multiprocessing re-invokes, which stops being
    # `sys.executable` as soon as anything calls `set_executable`.
    assert result["sys_executable"] not in result["stderr"]
    assert "no code was executed" in result["stderr"]
    assert "ContainerCodeExecutor" in result["stderr"]

  def test_execute_code_reports_a_missing_interpreter_path(
      self, mock_invocation_context: InvocationContext
  ):
    original = multiprocessing.spawn.get_executable()
    multiprocessing.set_executable("")
    try:
      result = UnsafeLocalCodeExecutor().execute_code(
          mock_invocation_context, CodeExecutionInput(code='print("hi")')
      )
    finally:
      multiprocessing.set_executable(original)

    assert result.stdout == ""
    assert result.output_files == []
    assert "no interpreter path to re-invoke" in result.stderr
    assert "ContainerCodeExecutor" in result.stderr

  def test_execute_code_reports_a_worker_that_cannot_start(
      self, mock_invocation_context: InvocationContext, monkeypatch
  ):
    context = _FakeSpawnContext(start_error=BrokenPipeError(32, "Broken pipe"))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)

    result = UnsafeLocalCodeExecutor().execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("hi")')
    )

    assert result.stdout == ""
    assert result.output_files == []
    assert "BrokenPipeError" in result.stderr
    assert "Broken pipe" in result.stderr
    assert os.fsdecode(multiprocessing.spawn.get_executable()) in result.stderr
    assert "ContainerCodeExecutor" in result.stderr

  def test_execute_code_reports_a_queue_that_cannot_be_created(
      self, mock_invocation_context: InvocationContext, monkeypatch
  ):
    context = _FakeSpawnContext(
        queue_error=PermissionError(1, "Operation not permitted")
    )
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)

    result = UnsafeLocalCodeExecutor().execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("hi")')
    )

    assert result.stdout == ""
    assert result.output_files == []
    assert "PermissionError" in result.stderr
    assert "Operation not permitted" in result.stderr
    assert os.fsdecode(multiprocessing.spawn.get_executable()) in result.stderr
    assert "ContainerCodeExecutor" in result.stderr

  def test_execute_code_releases_the_queue_when_the_worker_cannot_start(
      self, mock_invocation_context: InvocationContext, monkeypatch
  ):
    context = _FakeSpawnContext(start_error=OSError("Cannot allocate memory"))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)

    UnsafeLocalCodeExecutor().execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("hi")')
    )

    assert context.queue_calls == ["close", "join_thread"]

  def test_execute_code_does_not_tear_down_a_worker_that_never_started(
      self, mock_invocation_context: InvocationContext, monkeypatch
  ):
    context = _FakeSpawnContext(start_error=BrokenPipeError(32, "Broken pipe"))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)

    UnsafeLocalCodeExecutor().execute_code(
        mock_invocation_context, CodeExecutionInput(code='print("hi")')
    )

    assert context.process.calls == ["start"]

  def test_execute_code_propagates_a_non_oserror_setup_failure(
      self, mock_invocation_context: InvocationContext, monkeypatch
  ):
    def no_spawn_context(method: str) -> None:
      raise ValueError("cannot find context for spawn")

    monkeypatch.setattr(multiprocessing, "get_context", no_spawn_context)

    with pytest.raises(ValueError, match="cannot find context"):
      UnsafeLocalCodeExecutor().execute_code(
          mock_invocation_context, CodeExecutionInput(code='print("hi")')
      )

  def test_execute_code_logs_the_launch_failure(
      self, mock_invocation_context: InvocationContext, monkeypatch, caplog
  ):
    context = _FakeSpawnContext(start_error=BrokenPipeError(32, "Broken pipe"))
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: context)
    logger_name = unsafe_local_code_executor.logger.name

    with caplog.at_level(logging.ERROR, logger=logger_name):
      UnsafeLocalCodeExecutor().execute_code(
          mock_invocation_context, CodeExecutionInput(code='print("hi")')
      )

    errors = [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
