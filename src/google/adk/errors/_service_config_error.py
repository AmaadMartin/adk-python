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


class ServiceConfigError(ValueError):
  """Raised when a service or storage backend cannot be configured.

  This is raised when an option such as ``memory_service_uri``,
  ``artifact_service_uri`` or ``eval_storage_uri`` names a backend that ADK
  cannot resolve, or when the named backend rejects the value.

  Inherits from ValueError so that callers embedding ADK can catch it with a
  plain ``except ValueError``.
  """
