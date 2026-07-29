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

"""Environment-isolation guard for the Antigravity lab import.

Importing the lab must disable protobuf's gencode/runtime compatibility check
only while the ``google-antigravity`` SDK itself is loading. Leaking
``TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK`` into ``os.environ`` would turn
the check off for every other proto in the process and in every subprocess it
spawns.

Each assertion runs in a fresh interpreter because ``sys.modules`` caching makes
an in-process re-import a no-op once any other test has imported the lab.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_ENV_VAR = "TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"


def _run_in_subprocess(
    code: str, *, env_var_value: str | None
) -> subprocess.CompletedProcess[str]:
  """Runs ``code`` in a child interpreter with ``_ENV_VAR`` preset."""
  env = os.environ.copy()
  if env_var_value is None:
    env.pop(_ENV_VAR, None)
  else:
    env[_ENV_VAR] = env_var_value
  return subprocess.run(
      [sys.executable, "-c", code],
      capture_output=True,
      text=True,
      check=False,
      env=env,
  )


def test_import_does_not_leak_protobuf_version_check_env():
  """An import that found the var unset leaves it unset."""
  code = textwrap.dedent(f"""
      import os

      import google.adk.labs.antigravity  # noqa: F401

      assert {_ENV_VAR!r} not in os.environ, 'env var leaked out of the import'
      """)

  result = _run_in_subprocess(code, env_var_value=None)

  assert result.returncode == 0, result.stderr


def test_import_restores_preexisting_protobuf_version_check_env():
  """An import that found the var set restores the user's own value."""
  code = textwrap.dedent(f"""
      import os

      import google.adk.labs.antigravity  # noqa: F401

      assert os.environ[{_ENV_VAR!r}] == 'false', os.environ[{_ENV_VAR!r}]
      """)

  result = _run_in_subprocess(code, env_var_value="false")

  assert result.returncode == 0, result.stderr


def test_failed_import_does_not_leak_protobuf_version_check_env():
  """A failed import restores the environment and keeps its error message."""
  code = textwrap.dedent(f"""
      import os
      import sys

      # Simulate google-antigravity not being installed: a None entry in
      # sys.modules makes `import google.antigravity` raise ImportError.
      sys.modules['google.antigravity'] = None

      reason = ''
      try:
        import google.adk.labs.antigravity  # noqa: F401
      except ImportError as e:
        reason = str(e)

      assert 'google-adk[antigravity]' in reason, reason
      assert {_ENV_VAR!r} not in os.environ, 'env var leaked out of the import'
      """)

  result = _run_in_subprocess(code, env_var_value=None)

  assert result.returncode == 0, result.stderr


def test_protobuf_version_check_is_active_after_import():
  """The gencode/runtime check rejects mismatches again after the import."""
  code = textwrap.dedent("""
      import google.adk.labs.antigravity  # noqa: F401

      from google.protobuf import runtime_version

      rejected = False
      try:
        runtime_version.ValidateProtobufRuntimeVersion(
            runtime_version.DOMAIN,
            runtime_version.MAJOR + 1,
            0,
            0,
            '',
            'test.proto',
        )
      except runtime_version.VersionError:
        rejected = True

      assert rejected, 'protobuf version check is still disabled'
      """)

  result = _run_in_subprocess(code, env_var_value=None)

  assert result.returncode == 0, result.stderr
