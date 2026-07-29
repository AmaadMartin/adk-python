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

import os

_PROTOBUF_VERSION_CHECK_ENV = "TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"

# google-antigravity ships protobuf gencode newer than the runtime ADK
# resolves (google-cloud-aiplatform caps protobuf<7.0.0), so importing it
# raises google.protobuf.runtime_version.VersionError. Scoping the bypass to
# that import is enough -- the flag is re-read on every check and the SDK
# loads its gencode eagerly -- and leaving it set would disable the check for
# every other proto in this process and in every subprocess it spawns.
# See https://protobuf.dev/support/cross-version-runtime-guarantee.
_previous_protobuf_version_check = os.environ.get(_PROTOBUF_VERSION_CHECK_ENV)
os.environ[_PROTOBUF_VERSION_CHECK_ENV] = "true"
try:
  try:
    import google.antigravity  # noqa: F401
  except ImportError as e:
    raise ImportError(
        "The 'google-antigravity' package is required to use the ADK"
        " Antigravity integration. Install it with: pip install"
        ' "google-adk[antigravity]"'
    ) from e

  # Imported inside the guarded scope because it pulls in further
  # google.antigravity submodules (Agent, AgentConfig, types).
  from ._antigravity_agent import AntigravityAgent
finally:
  if _previous_protobuf_version_check is None:
    os.environ.pop(_PROTOBUF_VERSION_CHECK_ENV, None)
  else:
    os.environ[_PROTOBUF_VERSION_CHECK_ENV] = _previous_protobuf_version_check

__all__ = [
    "AntigravityAgent",
]
