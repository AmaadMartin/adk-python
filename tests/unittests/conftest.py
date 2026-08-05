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

import contextlib
import os
from unittest import mock

import pytest

pytest.register_assert_rewrite('google.adk.cli.agent_test_runner')

from pytest import fixture
from pytest import FixtureRequest
from pytest import hookimpl
from pytest import Metafunc

_ENV_VARS = {
    'GOOGLE_API_KEY': 'fake_google_api_key',
    'GOOGLE_CLOUD_PROJECT': 'fake_google_cloud_project',
    'GOOGLE_CLOUD_LOCATION': 'fake_google_cloud_location',
    'ADK_ALLOW_WIP_FEATURES': 'true',
    'ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS': 'true',
}

ENV_SETUPS = {
    'GOOGLE_AI': {
        'GOOGLE_GENAI_USE_ENTERPRISE': '0',
        **_ENV_VARS,
    },
    'VERTEX': {
        'GOOGLE_GENAI_USE_ENTERPRISE': '1',
        **_ENV_VARS,
    },
}


@fixture
def env_variables(request: FixtureRequest):
  # Set up the environment
  env_name: str = request.param
  envs = ENV_SETUPS[env_name]
  original_env = {key: os.environ.get(key) for key in envs}
  os.environ.update(envs)

  yield  # Run the test

  # Restore the environment
  for key in envs:
    if (original_val := original_env.get(key)) is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = original_val


# Store original environment variables to restore later
_original_env = {}


@hookimpl(tryfirst=True)
def pytest_sessionstart(session):
  """Set up environment variables at the beginning of the test session."""
  if not ENV_SETUPS:
    return
  # Use the first env setup to initialize environment for module-level imports
  env_name = next(iter(ENV_SETUPS.keys()))
  envs = ENV_SETUPS[env_name]
  global _original_env
  _original_env = {key: os.environ.get(key) for key in envs}
  os.environ.update(envs)


@hookimpl(trylast=True)
def pytest_sessionfinish(session):
  """Restore original environment variables at the end of the test session."""
  global _original_env
  for key, original_val in _original_env.items():
    if original_val is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = original_val
  _original_env = {}


@hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: Metafunc):
  """Generate test cases for each environment setup."""
  if env_variables.__name__ in metafunc.fixturenames:
    if not _is_explicitly_marked(env_variables.__name__, metafunc):
      metafunc.parametrize(
          env_variables.__name__, ENV_SETUPS.keys(), indirect=True
      )


def _is_explicitly_marked(mark_name: str, metafunc: Metafunc) -> bool:
  if hasattr(metafunc.function, 'pytestmark'):
    for mark in metafunc.function.pytestmark:
      if mark.name == 'parametrize' and mark.args[0] == mark_name:
        return True
  return False


# unittest.mock records every patch started with patch(...).start() here and
# only drops it on stop(); this is the list patch.stopall() drains. It is a
# private attribute, so degrade to a no-op if a future Python removes it.
_ACTIVE_PATCHES = getattr(
    getattr(mock, '_patch', None), '_active_patches', None
)


def _describe_patch(patcher: 'mock._patch | mock._patch_dict') -> str:
  """Renders a started patcher for the failure message."""
  if not isinstance(patcher, mock._patch):
    # patch.dict lands in the same list, patches a mapping in place, and so
    # carries no target attribute to name.
    return f'patch.dict({type(patcher.in_dict).__name__})'
  target = patcher.target
  target_name = getattr(target, '__name__', None) or repr(target)
  return f'{target_name}.{patcher.attribute}'


@fixture(autouse=True)
def _no_leaked_mock_patches(request: FixtureRequest):
  """Fails any test that leaves a mock patch it started still installed.

  A patch(...).start() without a matching stop() replaces the target attribute
  for the rest of the worker process, so later tests fail in ways that depend
  on ordering and look unrelated. This fixture is function scoped and autouse,
  and because conftest autouse fixtures are finalized last it observes the
  state after setup_method/teardown_method, setUp/tearDown, addCleanup and the
  mocker fixture have all run.
  """
  if _ACTIVE_PATCHES is None:
    yield
    return
  before = list(_ACTIVE_PATCHES)
  yield
  # _patch has no __eq__, so `in` compares identity; keeping `before` alive
  # also stops id() reuse from hiding a leak.
  leaked = [p for p in _ACTIVE_PATCHES if p not in before]
  if not leaked:
    return
  # Describe before stopping: _patch.__exit__ deletes .target.
  described = ', '.join(_describe_patch(p) for p in leaked)
  # Reverse order, so nested patches of one attribute unwind to the original.
  for patcher in reversed(leaked):
    # A patcher that cannot be stopped must not skip the remaining repairs or
    # hide the report below.
    with contextlib.suppress(Exception):
      patcher.stop()
  pytest.fail(
      f'{request.node.nodeid} finished with {len(leaked)} mock patch(es) still'
      f' installed: {described}. Every patch(...).start() needs a matching'
      ' stop() -- prefer addCleanup(patcher.stop), a teardown hook, the mocker'
      ' fixture, or patch(...) as a context manager/decorator. The leaked'
      ' patches were undone so later tests are unaffected.',
      pytrace=False,
  )
