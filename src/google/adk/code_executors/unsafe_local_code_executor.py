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
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import os
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


def _execute_in_process(
    code: str, globals_: dict[str, Any], result_connection: Connection
) -> None:
  """Executes code in a separate process and sends the result back."""
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
  result_connection.send((stdout.getvalue(), error))


def _execution_group(process: BaseProcess) -> int | None:
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


def _kill_execution(process: BaseProcess) -> None:
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


def _prepare_globals(code: str, globals_: dict[str, Any]) -> None:
  """Prepare globals for code execution, injecting __name__ if needed."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    globals_['__name__'] = '__main__'


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context."""

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data):
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
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)
    # Execute the code.
    globals_ = {}
    _prepare_globals(code_execution_input.code, globals_)

    ctx = multiprocessing.get_context('spawn')
    result_connection, worker_connection = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_execute_in_process,
        args=(code_execution_input.code, globals_, worker_connection),
        daemon=True,
    )
    process.start()
    # Leaves the worker holding the only write end, so end of file on the read
    # end means the worker is gone. That is what tells a worker which died
    # without reporting anything apart from one that is still working, and it
    # is why the wait below cannot block forever on a dead worker.
    worker_connection.close()

    output = ''
    error = ''
    if not result_connection.poll(self.timeout_seconds):
      _kill_execution(process)
      error = f'Code execution timed out after {self.timeout_seconds} seconds.'
    else:
      try:
        result = result_connection.recv()
      except (EOFError, OSError):
        # End of file, or a message cut short by the worker dying mid-send.
        result = None
      # Reaped before the exit code is read: it is only set once the worker has
      # been waited for.
      process.join()
      if result is None:
        error = (
            'The code execution worker exited without returning a result'
            f' (exit code {process.exitcode}).'
        )
        logger.warning(error)
      else:
        output, err = result
        if err:
          error = err

    # Collect the final result.
    result_connection.close()
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
