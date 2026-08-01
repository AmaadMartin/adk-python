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
import math
import multiprocessing
from multiprocessing.process import BaseProcess
import queue
import re
import time
import traceback
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

# How often the wait re-checks the worker, so that a worker that dies without
# producing a result ends the wait instead of stalling it.
_WORKER_POLL_SECONDS = 0.1

# How long a result still gets to arrive after the worker is seen to be gone:
# the worker can be reaped in the instant between writing to the queue and
# exiting, and that write is still on its way through the pipe.
_RESULT_FLUSH_SECONDS = 1


def _execute_in_process(
    code: str, globals_: dict[str, Any], result_queue: multiprocessing.Queue
) -> None:
  """Executes code in a separate process and puts result in queue."""
  stdout = io.StringIO()
  error = None
  try:
    with redirect_stdout(stdout):
      exec(code, globals_, globals_)
  except BaseException:
    error = traceback.format_exc()
  result_queue.put((stdout.getvalue(), error))


def _collect_result(
    process: BaseProcess,
    result_queue: multiprocessing.Queue[tuple[str, str | None]],
    timeout_seconds: int | None,
) -> tuple[str, str]:
  """Waits for the execution's output, or reports why none arrived.

  The wait polls rather than blocking on the queue alone, so that a worker that
  dies without putting anything on the queue -- a spawn child that cannot
  start, an OOM kill, code that calls `os._exit` -- is reported as what it is
  instead of being mistaken for a timeout or, with no timeout configured,
  waited on forever.

  Args:
    process: The started execution process.
    result_queue: The queue the execution reports its result on.
    timeout_seconds: How long a running execution is given, or None for as long
      as it keeps running.

  Returns:
    The execution's (stdout, stderr).
  """
  # No timeout is an infinite deadline rather than a special case: it still
  # lets a dead worker end the wait, and `timeout_seconds=0` still deadlines
  # immediately.
  deadline = time.monotonic() + (
      math.inf if timeout_seconds is None else timeout_seconds
  )
  while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      process.terminate()
      process.join()
      return '', f'Code execution timed out after {timeout_seconds} seconds.'
    try:
      stdout, error = result_queue.get(
          timeout=min(_WORKER_POLL_SECONDS, remaining)
      )
    except queue.Empty:
      if process.is_alive():
        continue
      # The worker can be reaped while the bytes it wrote are still in the
      # queue's pipe, so a dead worker does not yet mean no result is coming.
      try:
        stdout, error = result_queue.get(timeout=_RESULT_FLUSH_SECONDS)
      except queue.Empty:
        return '', (
            f'Code execution process exited with code {process.exitcode} '
            'without returning a result.'
        )
    process.join()
    return stdout, error or ''


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
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_execute_in_process,
        args=(code_execution_input.code, globals_, result_queue),
        daemon=True,
    )
    process.start()

    output, error = _collect_result(process, result_queue, self.timeout_seconds)

    # Collect the final result.
    result_queue.close()
    result_queue.join_thread()
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
