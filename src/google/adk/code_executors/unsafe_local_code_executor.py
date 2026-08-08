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

from __future__ import annotations

from contextlib import redirect_stdout
import io
import logging
import multiprocessing
import multiprocessing.spawn
import os
import queue
import re
import signal
import traceback
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

# How long to wait for a timed-out execution to exit after SIGTERM before
# escalating to SIGKILL, so that the timeout itself cannot block forever.
_TERMINATE_GRACE_SECONDS = 5

# Reported when the worker process cannot be started at all, which says nothing
# about the code the caller asked to run.
_WORKER_LAUNCH_FAILURE_MESSAGE = (
    'UnsafeLocalCodeExecutor could not start the worker process it runs code'
    ' in, so no code was executed: this is a limitation of the environment,'
    ' not an error in the code. Running code requires re-invoking the Python'
    " interpreter at '{interpreter}', which failed ({cause}). Configure a"
    ' usable interpreter with multiprocessing.set_executable(), or use a code'
    ' executor that does not spawn a local process, such as'
    ' ContainerCodeExecutor or VertexAiCodeExecutor.'
)


def _execute_in_process(
    code: str,
    globals_: dict[str, Any],
    result_queue: multiprocessing.Queue[tuple[str, str | None]],
) -> None:
  """Executes code in a separate process and puts result in queue."""
  # Detach into a new session/process group before running anything, so that a
  # timed-out execution can be killed together with everything it spawned.
  if hasattr(os, 'setsid'):
    try:
      os.setsid()
    except OSError:
      logger.debug('Could not detach the execution process group.')

  stdout = io.StringIO()
  error = None
  try:
    with redirect_stdout(stdout):
      exec(code, globals_, globals_)
  except BaseException:
    error = traceback.format_exc()
  result_queue.put((stdout.getvalue(), error))


def _execution_group(
    process: multiprocessing.process.BaseProcess,
) -> int | None:
  """Returns the group the execution detached into, or None if it has not."""
  if process.pid is None or not hasattr(os, 'killpg'):
    return None
  try:
    group = os.getpgid(process.pid)
    # Only report the group once the execution has detached into its own;
    # otherwise the group is still ours and signalling it would take down the
    # agent along with the code it is running.
    return group if group != os.getpgid(0) else None
  except OSError:
    return None


def _signal_group(group: int, sig: int) -> None:
  """Signals every process left in a group, tolerating an empty one."""
  try:
    os.killpg(group, sig)
  except OSError:
    logger.debug('Could not signal the execution process group.')


def _kill_execution(process: multiprocessing.process.BaseProcess) -> None:
  """Kills a timed-out execution along with any process it spawned."""
  # Resolved up front: once the execution process has been reaped its group can
  # no longer be looked up through it, and the group is what holds whatever the
  # code spawned.
  group = _execution_group(process)

  # SIGTERM first, so the code and its children get the same grace period the
  # execution process itself gets before anything is killed outright.
  if group is not None:
    _signal_group(group, signal.SIGTERM)
  process.terminate()
  process.join(_TERMINATE_GRACE_SECONDS)

  # Escalate unconditionally: the execution process exiting says nothing about
  # a child of it that is ignoring SIGTERM.
  if group is not None:
    _signal_group(group, signal.SIGKILL)
  if process.is_alive():
    process.kill()
    process.join()


def _worker_launch_failure_result(cause: str) -> CodeExecutionResult:
  """Builds the result reported when the worker process cannot be started."""
  # The interpreter multiprocessing re-invokes, which is not `sys.executable`
  # once anything has called `multiprocessing.set_executable()`.
  interpreter = multiprocessing.spawn.get_executable()
  return CodeExecutionResult(
      stderr=_WORKER_LAUNCH_FAILURE_MESSAGE.format(
          interpreter=os.fsdecode(interpreter) if interpreter else '<unset>',
          cause=cause,
      )
  )


def _prepare_globals(code: str, globals_: dict[str, Any]) -> None:
  """Prepare globals for code execution, injecting __name__ if needed."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    globals_['__name__'] = '__main__'


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context.

  Code runs in a `spawn` worker process, so a runtime that cannot re-invoke the
  Python interpreter to start one gets a report in `stderr` instead of an
  execution.
  """

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data: Any) -> None:
    """Initializes the UnsafeLocalCodeExecutor."""
    if 'stateful' in data and data['stateful']:
      raise ValueError('Cannot set `stateful=True` in UnsafeLocalCodeExecutor.')
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor.'
      )
    super().__init__(**data)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Executes the code in a spawned worker process.

    Args:
      invocation_context: The invocation context of the code execution.
      code_execution_input: The code execution input.

    Returns:
      The code execution result. Errors raised by the executed code, execution
      timeouts, and a worker process that could not be started at all are all
      reported in `stderr`.
    """
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)
    # Execute the code.
    globals_: dict[str, Any] = {}
    _prepare_globals(code_execution_input.code, globals_)

    # An interpreter path that is missing entirely fails inside multiprocessing
    # with a TypeError rather than an OSError, so it is reported here and the
    # handler below stays narrow.
    if not multiprocessing.spawn.get_executable():
      return _worker_launch_failure_result(
          'multiprocessing has no interpreter path to re-invoke; see'
          ' multiprocessing.set_executable'
      )

    result_queue: multiprocessing.Queue[tuple[str, str | None]] | None = None
    try:
      ctx = multiprocessing.get_context('spawn')
      result_queue = ctx.Queue()
      process = ctx.Process(
          target=_execute_in_process,
          args=(code_execution_input.code, globals_, result_queue),
          daemon=True,
      )
      process.start()
    except OSError as exc:
      # Reported rather than raised: `execute_code` is declared to return a
      # result, and the flow processor that calls it does not catch. The
      # traceback is logged so operators still get the cause.
      logger.exception(
          'UnsafeLocalCodeExecutor could not start its worker process.'
      )
      if result_queue is not None:
        # The queue holds a semaphore and a pipe from the moment it exists.
        result_queue.close()
        result_queue.join_thread()
      return _worker_launch_failure_result(f'{type(exc).__name__}: {exc}')

    output = ''
    error = ''
    try:
      output, err = result_queue.get(timeout=self.timeout_seconds)
      process.join()
      if err:
        error = err
    except queue.Empty:
      _kill_execution(process)
      error = f'Code execution timed out after {self.timeout_seconds} seconds.'

    # Collect the final result.
    result_queue.close()
    result_queue.join_thread()
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
