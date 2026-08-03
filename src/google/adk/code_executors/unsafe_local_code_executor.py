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
import ctypes
import io
import logging
import multiprocessing
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


def _execute_in_process(
    code: str,
    globals_: dict[str, Any],
    result_queue: multiprocessing.Queue,
    execution_group: ctypes.c_int,
) -> None:
  """Executes code in a separate process and puts result in queue."""
  # Detach into a new session/process group before running anything, so that
  # the execution can be killed together with everything it spawned.
  if hasattr(os, 'setsid'):
    try:
      os.setsid()
      # Reported from in here, once, rather than looked up later through this
      # process: the group outlives the process, but a lookup through its pid
      # dies with it, and what the code spawns lives in the group.
      execution_group.value = os.getpgrp()
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


def _signal_group(group: int, sig: int) -> None:
  """Signals every process left in a group, tolerating an empty one."""
  try:
    os.killpg(group, sig)
  except OSError:
    logger.debug('Could not signal the execution process group.')


def _kill_execution(
    process: multiprocessing.process.BaseProcess, group: int
) -> None:
  """Kills an execution along with any process it spawned.

  Args:
    process: The execution process, which is joined (and so reaped) here.
    group: The process group the execution reported detaching into, or 0 if it
      never reported one and only `process` itself can be killed.
  """
  # SIGTERM first, so the code and its children get the same grace period the
  # execution process itself gets before anything is killed outright.
  if group:
    _signal_group(group, signal.SIGTERM)
  process.terminate()
  process.join(_TERMINATE_GRACE_SECONDS)

  # Escalate unconditionally: the execution process exiting says nothing about
  # a child of it that is ignoring SIGTERM.
  if group:
    _signal_group(group, signal.SIGKILL)
  if process.is_alive():
    process.kill()
    process.join()


def _prepare_globals(code: str, globals_: dict[str, Any]) -> None:
  """Prepare globals for code execution, injecting __name__ if needed."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    globals_['__name__'] = '__main__'


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context.

  The code runs in a worker process that detaches into its own process group.
  Anything the code leaves running in that group is killed when the call
  returns, so an execution cannot outlive `execute_code`.
  """

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
    result_queue = ctx.Queue()
    # Lock-free on purpose: one writer, one reader, one int. A lock could be
    # left held by an execution that is SIGKILLed, which would hang this read.
    execution_group = ctx.Value('i', 0, lock=False)
    process = ctx.Process(
        target=_execute_in_process,
        args=(
            code_execution_input.code,
            globals_,
            result_queue,
            execution_group,
        ),
        daemon=True,
    )
    process.start()

    output = ''
    error = ''
    try:
      output, err = result_queue.get(timeout=self.timeout_seconds)
      if err:
        error = err
    except queue.Empty:
      error = f'Code execution timed out after {self.timeout_seconds} seconds.'
    finally:
      # Torn down on every path, not just on timeout: code that returns
      # normally can still leave a process of its own running, and the group
      # is signalled before the worker is reaped so that its pid -- and with
      # it the group id -- cannot be recycled first.
      _kill_execution(process, execution_group.value)

    # Collect the final result.
    result_queue.close()
    result_queue.join_thread()
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
