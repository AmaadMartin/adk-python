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
import queue
import re
import sys
import traceback
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

_WORKER_LAUNCH_FAILURE_MESSAGE = (
    'Failed to start the worker process that UnsafeLocalCodeExecutor runs'
    ' code in: this Python interpreter cannot spawn a child process in this'
    ' environment ({cause}). No code was executed; this is a limitation of'
    ' the environment, not an error in the code. This usually means'
    " sys.executable ('{executable}') is not a usable Python interpreter, as"
    ' in an embedded or hermetic runtime, or that the sandbox forbids'
    ' creating processes. Use ContainerCodeExecutor or VertexAiCodeExecutor'
    ' to execute code in an environment like this.'
)


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


def _prepare_globals(code: str, globals_: dict[str, Any]) -> None:
  """Prepare globals for code execution, injecting __name__ if needed."""
  if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
    globals_['__name__'] = '__main__'


def _worker_launch_failure_result(cause: str) -> CodeExecutionResult:
  """Builds the result reported when the worker process cannot be started."""
  return CodeExecutionResult(
      stderr=_WORKER_LAUNCH_FAILURE_MESSAGE.format(
          cause=cause, executable=sys.executable
      )
  )


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context.

  Code runs in a `spawn` worker process, so an environment that cannot
  re-invoke the interpreter to start one has to use a remote executor instead:
  `execute_code` reports that in `stderr` there rather than running anything.
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
    """Executes the code in a spawned worker process.

    Errors raised by the executed code, execution timeouts, and a worker
    process that could not be started at all are all reported in `stderr`.
    """
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)
    # Execute the code.
    globals_ = {}
    _prepare_globals(code_execution_input.code, globals_)

    # The worker is started by re-invoking this interpreter. A `None` path --
    # what an interpreter that cannot locate itself leaves here -- makes
    # multiprocessing raise a bare TypeError rather than an OSError, so a
    # missing path is reported here rather than by the handler below.
    if not multiprocessing.spawn.get_executable():
      return _worker_launch_failure_result(
          'multiprocessing has no interpreter path to re-invoke; see'
          ' multiprocessing.set_executable'
      )

    result_queue = None
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
      # Returned rather than raised: `BaseCodeExecutor.execute_code` is
      # declared to return a result, and callers such as `SkillToolset`
      # truncate a raised message to 200 characters, which would cut off the
      # remediation this diagnostic exists to deliver. The traceback is logged
      # so operators still get the cause.
      logger.exception(
          'UnsafeLocalCodeExecutor could not start its worker process.'
      )
      # `Queue()` registers a semaphore with the resource tracker, so it holds
      # OS resources as soon as it exists.
      if result_queue is not None:
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
      process.terminate()
      process.join()
      error = f'Code execution timed out after {self.timeout_seconds} seconds.'

    # Collect the final result.
    result_queue.close()
    result_queue.join_thread()
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )
