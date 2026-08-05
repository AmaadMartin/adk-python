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

"""Tests for the mock patch leak guard in tests/unittests/conftest.py.

Each case runs a synthetic pytest suite under the real conftest with
``pytester``. The nested runs must stay out of process: an in-process run
shares ``unittest.mock._patch._active_patches`` with this process, so a nested
suite that leaks on purpose would trip the guard on the test running it.
"""

import pathlib
import re
from unittest import mock

from pytest import Pytester

pytest_plugins = ['pytester']

_CONFTEST_SOURCE = (pathlib.Path(__file__).parent / 'conftest.py').read_text()

_VICTIM_MODULE = """
TARGET = 'original'
MODULE_TARGET = 'original'
DICT_TARGET = {'key': 'original'}


class _Holder:
  value = 'original'


HOLDER = _Holder()
"""

_LEAKY_SUITE = """
from unittest import mock

import victim_leaky


class TestLeakyClass:

  def setup_method(self):
    self.mocked = mock.patch.object(victim_leaky, 'TARGET').start()

  def test_first(self):
    assert victim_leaky.TARGET is self.mocked

  def test_second(self):
    assert victim_leaky.TARGET is self.mocked
"""

_INSTANCE_LEAK_SUITE = """
from unittest import mock

import victim_instance


def test_leaks_an_instance_patch():
  mocked = mock.patch.object(victim_instance.HOLDER, 'value').start()
  assert victim_instance.HOLDER.value is mocked
"""

_NESTED_LEAK_SUITE = """
from unittest import mock

import victim_nested


def test_leaks_two_nested_patches():
  mock.patch.object(victim_nested, 'TARGET', 'first').start()
  mock.patch.object(victim_nested, 'TARGET', 'second').start()
  assert victim_nested.TARGET == 'second'


def test_leaks_one_patch():
  mock.patch.object(victim_nested, 'TARGET', 'third').start()
  assert victim_nested.TARGET == 'third'
"""

_RESTORED_CHECK_SUITE = """
import victim_nested


def test_target_is_restored():
  assert victim_nested.TARGET == 'original'
"""

_MIXED_LEAK_SUITE = """
from unittest import mock

import victim_mixed


def test_leaks_a_patch_dict_and_a_patch():
  mock.patch.dict(victim_mixed.DICT_TARGET, {'key': 'leaked'}).start()
  mock.patch.object(victim_mixed, 'TARGET', 'leaked').start()
  assert victim_mixed.DICT_TARGET == {'key': 'leaked'}
  assert victim_mixed.TARGET == 'leaked'
"""

_MIXED_RESTORED_SUITE = """
import victim_mixed


def test_both_leaks_are_undone():
  assert victim_mixed.DICT_TARGET == {'key': 'original'}
  assert victim_mixed.TARGET == 'original'
"""

# The created attribute is gone by the time the guard stops the patcher, so
# _patch.__exit__ raises AttributeError from its delattr.
_UNSTOPPABLE_LEAK_SUITE = """
from unittest import mock

import victim_unstoppable


def test_leaks_a_patcher_that_cannot_be_stopped():
  mock.patch.object(victim_unstoppable, 'TARGET', 'leaked').start()
  mock.patch.object(
      victim_unstoppable, 'CREATED', 'leaked', create=True
  ).start()
  del victim_unstoppable.CREATED
"""

_UNSTOPPABLE_RESTORED_SUITE = """
import victim_unstoppable


def test_the_stoppable_patch_is_still_undone():
  assert victim_unstoppable.TARGET == 'original'
"""

_SWAPPED_LEAK_SUITE = """
from unittest import mock

import pytest

import victim_swapped


@pytest.fixture(scope='module')
def module_patcher():
  patcher = mock.patch.object(victim_swapped, 'MODULE_TARGET', 'patched')
  patcher.start()
  yield patcher
  patcher.stop()


def test_stops_a_preexisting_patch_and_leaks_its_own(module_patcher):
  module_patcher.stop()
  mock.patch.object(victim_swapped, 'TARGET', 'leaked').start()
  assert victim_swapped.TARGET == 'leaked'
"""

_CLEAN_SUITE = """
import unittest
from unittest import mock

import victim_clean


class TestTearDownStopall(unittest.TestCase):

  def setUp(self):
    self.mocked = mock.patch.object(victim_clean, 'TARGET').start()

  def tearDown(self):
    mock.patch.stopall()

  def test_patch_is_active(self):
    assert victim_clean.TARGET is self.mocked


class TestAddCleanup(unittest.TestCase):

  def setUp(self):
    patcher = mock.patch.object(victim_clean, 'TARGET')
    self.mocked = patcher.start()
    self.addCleanup(patcher.stop)

  def test_patch_is_active(self):
    assert victim_clean.TARGET is self.mocked


def test_mocker_fixture(mocker):
  mocked = mocker.patch.object(victim_clean, 'TARGET')
  assert victim_clean.TARGET is mocked


def test_context_manager():
  with mock.patch.object(victim_clean, 'TARGET') as mocked:
    assert victim_clean.TARGET is mocked


@mock.patch.object(victim_clean, 'TARGET')
def test_decorator(mocked):
  assert victim_clean.TARGET is mocked
"""

_MODULE_SCOPED_SUITE = """
from unittest import mock

import pytest

import victim_clean


@pytest.fixture(scope='module')
def module_patch():
  patcher = mock.patch.object(victim_clean, 'MODULE_TARGET', 'patched')
  patcher.start()
  yield
  patcher.stop()


def test_without_the_module_patch():
  assert victim_clean.MODULE_TARGET == 'original'


def test_module_patch_is_not_flagged(module_patch):
  assert victim_clean.MODULE_TARGET == 'patched'


def test_module_patch_outlives_the_previous_test(module_patch):
  assert victim_clean.MODULE_TARGET == 'patched'


def test_stopping_a_preexisting_patch_is_not_flagged(module_patch):
  mock.patch.stopall()
  assert victim_clean.MODULE_TARGET == 'original'
"""

# test_clean.py holds 5 tests and test_module.py holds 4.
_CLEAN_SUITE_TEST_COUNT = 9


def _write_suite(pytester: Pytester, conftest: str, **modules: str) -> None:
  """Writes the conftest and the modules of one synthetic nested suite."""
  pytester.makeconftest(conftest)
  pytester.makepyfile(**modules)


def test_guard_fails_the_test_that_leaks_a_patch(pytester: Pytester):
  """A leaked patch errors the leaking test and names it and its target."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_leaky=_VICTIM_MODULE,
      test_leaky=_LEAKY_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=2, errors=2)
  output = result.stdout.str()
  assert 'test_leaky.py::TestLeakyClass::test_first' in output
  assert 'test_leaky.py::TestLeakyClass::test_second' in output
  assert 'victim_leaky.TARGET' in output


def test_guard_names_an_instance_target_by_repr(pytester: Pytester):
  """A patched instance has no __name__, so the guard falls back to repr."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_instance=_VICTIM_MODULE,
      test_instance=_INSTANCE_LEAK_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=1, errors=1)
  assert re.search(
      r'_Holder object at 0x[0-9a-f]+>\.value', result.stdout.str()
  )


def test_guard_undoes_the_leak_so_later_tests_are_clean(pytester: Pytester):
  """The guard unwinds leaked patches in reverse, restoring the original."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_nested=_VICTIM_MODULE,
      test_a_leak=_NESTED_LEAK_SUITE,
      test_b_restored=_RESTORED_CHECK_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=3, errors=2)
  output = result.stdout.str()
  assert 'test_a_leak.py::test_leaks_two_nested_patches' in output
  assert '2 mock patch(es) still installed' in output
  assert 'addCleanup(patcher.stop)' in output


def test_guard_reports_and_undoes_a_leaked_patch_dict(pytester: Pytester):
  """A leaked patch.dict is named and undone though it carries no target."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_mixed=_VICTIM_MODULE,
      test_a_mixed=_MIXED_LEAK_SUITE,
      test_b_restored=_MIXED_RESTORED_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=2, errors=1)
  output = result.stdout.str()
  assert 'patch.dict(dict)' in output
  assert 'victim_mixed.TARGET' in output


def test_guard_repairs_the_rest_when_one_patcher_cannot_stop(
    pytester: Pytester,
):
  """A patcher that raises on stop must not block the other repairs."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_unstoppable=_VICTIM_MODULE,
      test_a_unstoppable=_UNSTOPPABLE_LEAK_SUITE,
      test_b_restored=_UNSTOPPABLE_RESTORED_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=2, errors=1)
  assert '2 mock patch(es) still installed' in result.stdout.str()


def test_guard_detects_a_leak_that_replaces_a_stopped_patch(pytester: Pytester):
  """The active patch count is unchanged here, so detection needs identity."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_swapped=_VICTIM_MODULE,
      test_swapped=_SWAPPED_LEAK_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=1, errors=1)
  assert 'victim_swapped.TARGET' in result.stdout.str()


def test_guard_ignores_correctly_cleaned_up_patches(pytester: Pytester):
  """Every supported cleanup idiom passes the guard without an error."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_clean=_VICTIM_MODULE,
      test_clean=_CLEAN_SUITE,
      test_module=_MODULE_SCOPED_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=_CLEAN_SUITE_TEST_COUNT)


def test_guard_is_a_noop_when_private_mock_api_is_absent(pytester: Pytester):
  """Without mock._patch._active_patches the guard must not fail anything."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE + '\n_ACTIVE_PATCHES = None\n',
      victim_leaky=_VICTIM_MODULE,
      test_leaky=_LEAKY_SUITE,
  )

  result = pytester.runpytest_subprocess()

  result.assert_outcomes(passed=2)


def test_guard_is_xdist_compatible(pytester: Pytester):
  """Under xdist each worker snapshots its own process-local patch list."""
  _write_suite(
      pytester,
      _CONFTEST_SOURCE,
      victim_leaky=_VICTIM_MODULE,
      victim_clean=_VICTIM_MODULE,
      test_leaky=_LEAKY_SUITE,
      test_clean=_CLEAN_SUITE,
      test_module=_MODULE_SCOPED_SUITE,
  )

  result = pytester.runpytest_subprocess('-n', '2', '--dist', 'loadfile')

  result.assert_outcomes(passed=2 + _CLEAN_SUITE_TEST_COUNT, errors=2)


def test_private_mock_patch_registry_still_exists():
  """The guard degrades to a no-op without this; fail loudly if it vanishes."""
  assert hasattr(mock._patch, '_active_patches')
