# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may in obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import _lazy
from .base_plugin import BasePlugin
from .plugin_manager import PluginManager

if TYPE_CHECKING:
  from ._reflect_retry_model_plugin import ReflectAndRetryModelPlugin
  from .debug_logging_plugin import DebugLoggingPlugin
  from .logging_plugin import LoggingPlugin
  from .reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

__all__ = [
    "BasePlugin",
    "DebugLoggingPlugin",
    "LoggingPlugin",
    "PluginManager",
    "ReflectAndRetryModelPlugin",
    "ReflectAndRetryToolPlugin",
]

_LAZY_MEMBERS: dict[str, str] = {
    "DebugLoggingPlugin": ".debug_logging_plugin",
    "LoggingPlugin": ".logging_plugin",
    "ReflectAndRetryModelPlugin": "._reflect_retry_model_plugin",
    "ReflectAndRetryToolPlugin": ".reflect_retry_tool_plugin",
}


__getattr__, __dir__ = _lazy.accessors(globals(), _LAZY_MEMBERS)
