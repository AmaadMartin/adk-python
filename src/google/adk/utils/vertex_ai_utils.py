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

"""Utilities for Vertex AI, including Express Mode and resource names.

This module is for ADK internal use only.
Please do not rely on the implementation details.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from .env_utils import is_enterprise_mode_enabled

_AGENT_ENGINE_RESOURCE_NAME_PATTERN = re.compile(
    r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)'
    r'/reasoningEngines/(\d+)$'
)


def parse_agent_engine_resource_name(
    resource_name: str,
) -> Optional[tuple[str, str, str]]:
  """Parses an Agent Engine resource name into its components.

  Args:
    resource_name: A resource name of the form
      ``projects/{project}/locations/{location}/reasoningEngines/{id}``.

  Returns:
    A ``(project, location, reasoning_engine_id)`` tuple, or ``None`` if
    ``resource_name`` does not match the expected format. Callers are
    responsible for raising their own error on ``None``.
  """
  match = _AGENT_ENGINE_RESOURCE_NAME_PATTERN.fullmatch(resource_name)
  if match is None:
    return None
  return match.group(1), match.group(2), match.group(3)


def get_express_mode_api_key(
    project: Optional[str],
    location: Optional[str],
    express_mode_api_key: Optional[str],
) -> Optional[str]:
  """Validates and returns the API key for Express Mode."""
  if (project or location) and express_mode_api_key:
    raise ValueError(
        'Cannot specify project or location and express_mode_api_key. '
        'Either use project and location, or just the express_mode_api_key.'
    )
  if is_enterprise_mode_enabled():
    return express_mode_api_key or os.environ.get('GOOGLE_API_KEY', None)
  else:
    return None
