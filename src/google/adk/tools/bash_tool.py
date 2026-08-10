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

"""Tool to execute bash commands."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import logging
import os
import pathlib
import shlex
import signal
from typing import Any
from typing import Optional

from google.genai import types

from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

_resource = importlib.import_module("resource") if os.name == "posix" else None

_READ_CHUNK_BYTES = 65536
_TIMEOUT_DRAIN_SECONDS = 1.0


@dataclasses.dataclass(frozen=True)
class BashToolPolicy:
  """Configuration for allowed bash commands and resource limits.

  Set allowed_command_prefixes to ("*",) to allow all commands (default),
  or explicitly list allowed prefixes.

  Values for max_memory_bytes, max_file_size_bytes, and max_child_processes
  will be enforced upon the spawned subprocess.
  """

  allowed_command_prefixes: tuple[str, ...] = ("*",)
  blocked_operators: tuple[str, ...] = ()
  timeout_seconds: Optional[int] = 30
  max_memory_bytes: Optional[int] = None
  max_file_size_bytes: Optional[int] = None
  max_child_processes: Optional[int] = None


def _validate_command(command: str, policy: BashToolPolicy) -> Optional[str]:
  """Validates a bash command against the permitted prefixes."""
  stripped = command.strip()
  if not stripped:
    return "Command is required."

  for op in policy.blocked_operators:
    if op in command:
      return f"Command contains blocked operator: {op}"

  if "*" in policy.allowed_command_prefixes:
    return None

  for prefix in policy.allowed_command_prefixes:
    if stripped.startswith(prefix):
      return None

  allowed = ", ".join(policy.allowed_command_prefixes)
  return f"Command blocked. Permitted prefixes are: {allowed}"


def _set_resource_limits(policy: BashToolPolicy) -> None:
  """Sets resource limits for the subprocess based on the provided policy."""
  if _resource is None:
    return
  try:
    _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
    if policy.max_memory_bytes:
      _resource.setrlimit(
          _resource.RLIMIT_AS,
          (policy.max_memory_bytes, policy.max_memory_bytes),
      )
    if policy.max_file_size_bytes:
      _resource.setrlimit(
          _resource.RLIMIT_FSIZE,
          (policy.max_file_size_bytes, policy.max_file_size_bytes),
      )
    if policy.max_child_processes:
      _resource.setrlimit(
          _resource.RLIMIT_NPROC,
          (policy.max_child_processes, policy.max_child_processes),
      )
  except (ValueError, OSError) as e:
    logger.warning("Failed to set resource limits: %s", e)


async def _read_stream(
    stream: Optional[asyncio.StreamReader], chunks: list[bytes]
) -> None:
  """Reads `stream` to EOF, appending every chunk to `chunks`.

  `chunks` belongs to the caller, so whatever has been read survives the
  cancellation of this coroutine. That is what lets the timeout path report
  the output a command produced before it was killed.

  Args:
    stream: The pipe to read, or None when the process has no such pipe.
    chunks: The caller's accumulator, appended to in read order.
  """
  if stream is None:
    return
  while True:
    chunk = await stream.read(_READ_CHUNK_BYTES)
    if not chunk:
      return
    chunks.append(chunk)


def _decode(chunks: list[bytes], name: str) -> str:
  """Joins and decodes `chunks`, or reports that `name` produced nothing."""
  return b"".join(chunks).decode(errors="replace") or f"<no {name} captured>"


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
  """Sends SIGKILL to the subprocess's process group if it still exists."""
  try:
    if process.pid:
      os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass


class ExecuteBashTool(BaseTool):
  """Tool to execute a validated bash command within a workspace directory."""

  def __init__(
      self,
      *,
      workspace: pathlib.Path | None = None,
      policy: Optional[BashToolPolicy] = None,
  ):
    if workspace is None:
      workspace = pathlib.Path.cwd()
    policy = policy or BashToolPolicy()
    allowed_hint = (
        "any command"
        if "*" in policy.allowed_command_prefixes
        else (
            "commands matching prefixes:"
            f" {', '.join(policy.allowed_command_prefixes)}"
        )
    )
    super().__init__(
        name="execute_bash",
        description=(
            "Executes a bash command with the working directory set to the"
            f" workspace. Allowed: {allowed_hint}. All commands require user"
            " confirmation."
        ),
    )
    self._workspace = workspace
    self._policy = policy

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    command = args.get("command")
    if not command:
      return {"error": "Command is required."}

    # Static validation.
    error = _validate_command(command, self._policy)
    if error:
      return {"error": error}

    # Always request user confirmation.
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=f"Please approve or reject the bash command: {command}",
      )
      tool_context.actions.skip_summarization = True
      return {
          "error": (
              "This tool call requires confirmation, please approve or reject."
          )
      }
    elif not tool_context.tool_confirmation.confirmed:
      return {"error": "This tool call is rejected."}

    if os.name != "posix":
      return {"error": "ExecuteBashTool is only supported on POSIX systems."}

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    try:
      process = await asyncio.create_subprocess_exec(
          *shlex.split(command),
          cwd=str(self._workspace),
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
          start_new_session=True,
          preexec_fn=lambda: _set_resource_limits(self._policy),
      )

      async def _collect() -> None:
        """Drains both pipes to EOF, then reaps the process."""
        await asyncio.gather(
            _read_stream(process.stdout, stdout_chunks),
            _read_stream(process.stderr, stderr_chunks),
        )
        # Reaping is part of the deadline: a command that closes both pipes
        # and keeps running must still time out rather than block forever.
        await process.wait()

      collector = asyncio.create_task(_collect())
      try:
        # asyncio.wait leaves the collector running when the deadline
        # expires, unlike asyncio.wait_for, which cancels it and discards
        # everything it has read.
        done, _ = await asyncio.wait(
            {collector}, timeout=self._policy.timeout_seconds
        )
        if not done:
          _kill_process_group(process)
          # The kill closes both write ends, so give the collector a bounded
          # moment to observe EOF and take what is still buffered.
          await asyncio.wait({collector}, timeout=_TIMEOUT_DRAIN_SECONDS)
          return {
              "error": (
                  f"Command timed out after {self._policy.timeout_seconds}"
                  " seconds."
              ),
              "stdout": _decode(stdout_chunks, "stdout"),
              "stderr": _decode(stderr_chunks, "stderr"),
              "returncode": process.returncode,
          }
        # Re-raise anything _collect() failed with; the finally below would
        # otherwise swallow it.
        collector.result()
      finally:
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        _kill_process_group(process)

      return {
          "stdout": _decode(stdout_chunks, "stdout"),
          "stderr": _decode(stderr_chunks, "stderr"),
          "returncode": process.returncode,
      }
    except Exception as e:  # pylint: disable=broad-except
      logger.exception("ExecuteBashTool execution failed")
      return {
          "error": f"Execution failed: {str(e)}",
          "stdout": _decode(stdout_chunks, "stdout"),
          "stderr": _decode(stderr_chunks, "stderr"),
      }

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      return "TOOL_ERROR"
    return None
