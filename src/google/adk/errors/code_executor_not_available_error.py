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


class CodeExecutorNotAvailableError(RuntimeError):
  """Raised when a code executor cannot run in the current environment.

  This signals an environment or deployment problem rather than a problem with
  the submitted code: retrying the same code cannot make it succeed. For
  example, `UnsafeLocalCodeExecutor` raises this when the environment cannot
  start multiprocessing worker processes, in which case a remote executor has
  to be used instead.

  Inherits from RuntimeError (for backward compatibility).
  """

  def __init__(
      self, message: str = "The code executor is not available."
  ) -> None:
    """Initializes the CodeExecutorNotAvailableError exception.

    Args:
        message (str): A message describing why the executor is unavailable.
    """
    self.message = message
    super().__init__(self.message)
