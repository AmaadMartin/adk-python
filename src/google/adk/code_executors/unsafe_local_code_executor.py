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
import traceback
from typing import Any

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..errors.code_executor_not_available_error import CodeExecutorNotAvailableError
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult

logger = logging.getLogger('google_adk.' + __name__)


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


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context.

  Code runs in a `spawn` worker process, so this executor requires an
  environment that permits starting worker processes by re-invoking the
  interpreter. Where it does not, `execute_code` raises
  `CodeExecutorNotAvailableError` and a remote executor has to be used instead.
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

    Args:
      invocation_context: The invocation context of the code execution.
      code_execution_input: The code execution input.

    Returns:
      The code execution result. Errors raised by the executed code, and
      execution timeouts, are reported in `stderr`.

    Raises:
      CodeExecutorNotAvailableError: The worker process could not be created,
        e.g. because the environment does not permit multiprocessing.
    """
    logger.debug('Executing code:\n```\n%s\n```', code_execution_input.code)
    # Execute the code.
    globals_ = {}
    _prepare_globals(code_execution_input.code, globals_)

    # The worker is started by re-invoking this interpreter. Without a path to
    # re-invoke, multiprocessing raises a bare TypeError rather than an
    # OSError, so that case is reported here instead.
    if not multiprocessing.spawn.get_executable():
      raise CodeExecutorNotAvailableError(
          'UnsafeLocalCodeExecutor could not start a worker process; this'
          ' environment has no interpreter path to re-invoke (see'
          ' `multiprocessing.set_executable`).'
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
      # `Queue()` registers a semaphore with the multiprocessing resource
      # tracker -- itself a spawned helper -- which is what fails first in an
      # unusable environment. Release the queue if it did get that far.
      if result_queue is not None:
        result_queue.close()
      raise CodeExecutorNotAvailableError(
          'UnsafeLocalCodeExecutor could not start a worker process; this'
          ' environment may not permit multiprocessing.'
          f' Original error: {type(exc).__name__}: {exc}'
      ) from exc

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
