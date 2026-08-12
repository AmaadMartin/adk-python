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

"""Tests for the env_variables parametrization hook in conftest.py."""

import os

import pytest


@pytest.mark.parametrize('env_variables', ['VERTEX'], indirect=True)
class TestClassLevelParametrizeMark:
  """A class-level mark must be honoured, and applied exactly once."""

  def test_runs_once_under_vertex(self, env_variables, request):
    assert request.node.callspec.params == {'env_variables': 'VERTEX'}
    assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '1'


@pytest.mark.parametrize(argnames='unrelated', argvalues=[1])
class TestKeywordOnlyParametrizeMark:
  """A class-level mark with no positional args must not break the hook."""

  def test_env_variables_is_still_auto_parametrized(
      self, env_variables, unrelated, request
  ):
    assert request.node.callspec.params['env_variables'] in (
        'GOOGLE_AI',
        'VERTEX',
    )


@pytest.mark.parametrize(argnames='unrelated', argvalues=[1])
def test_keyword_form_mark_on_function_does_not_crash_collection(
    env_variables, unrelated, request
):
  assert request.node.callspec.params['env_variables'] in (
      'GOOGLE_AI',
      'VERTEX',
  )


@pytest.mark.parametrize('unrelated', [1])
def test_mark_for_another_argument_does_not_suppress_auto_parametrization(
    env_variables, unrelated, request
):
  assert request.node.callspec.params['env_variables'] in (
      'GOOGLE_AI',
      'VERTEX',
  )


@pytest.mark.parametrize('unrelated_a,unrelated_b', [(1, 2)])
def test_comma_joined_mark_for_other_arguments_still_auto_parametrizes(
    env_variables, unrelated_a, unrelated_b, request
):
  assert request.node.callspec.params['env_variables'] in (
      'GOOGLE_AI',
      'VERTEX',
  )


@pytest.mark.parametrize('env_variables', ['GOOGLE_AI'], indirect=True)
def test_function_level_mark_still_wins(env_variables, request):
  assert request.node.callspec.params == {'env_variables': 'GOOGLE_AI'}
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '0'


@pytest.mark.parametrize(
    argnames='env_variables', argvalues=['VERTEX'], indirect=True
)
def test_keyword_form_mark_is_honoured(env_variables, request):
  assert request.node.callspec.params == {'env_variables': 'VERTEX'}
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '1'


@pytest.mark.parametrize(
    'other, env_variables', [(1, 'VERTEX')], indirect=['env_variables']
)
def test_comma_joined_mark_is_honoured(other, env_variables, request):
  assert request.node.callspec.params == {'other': 1, 'env_variables': 'VERTEX'}
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '1'


@pytest.mark.parametrize('env_variables,', [('VERTEX',)], indirect=True)
def test_trailing_comma_mark_is_honoured(env_variables, request):
  assert request.node.callspec.params == {'env_variables': 'VERTEX'}
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '1'


@pytest.mark.parametrize(['env_variables'], [('VERTEX',)], indirect=True)
def test_sequence_form_mark_is_honoured(env_variables, request):
  assert request.node.callspec.params == {'env_variables': 'VERTEX'}
  assert os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] == '1'


def test_unmarked_test_is_auto_parametrized(env_variables, request):
  assert request.node.callspec.params['env_variables'] in (
      'GOOGLE_AI',
      'VERTEX',
  )
