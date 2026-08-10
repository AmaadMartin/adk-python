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

"""Tests for utilities in eval."""

import dataclasses
import os
import subprocess
import sys
import textwrap
from unittest import mock

from google.adk.cli.utils import evals
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pydantic
import pytest


@mock.patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
@mock.patch(
    'google.adk.evaluation.gcs_eval_set_results_manager.GcsEvalSetResultsManager',
    autospec=True,
)
@mock.patch(
    'google.adk.evaluation.gcs_eval_sets_manager.GcsEvalSetsManager',
    autospec=True,
)
def test_create_gcs_eval_managers_from_uri_success(
    mock_gcs_eval_sets_manager, mock_gcs_eval_set_results_manager
):
  mock_gcs_eval_sets_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetsManager
  )
  mock_gcs_eval_set_results_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetResultsManager
  )

  managers = evals.create_gcs_eval_managers_from_uri('gs://test-bucket')

  assert managers is not None
  mock_gcs_eval_sets_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  mock_gcs_eval_set_results_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  assert managers.eval_sets_manager == mock_gcs_eval_sets_manager.return_value
  assert (
      managers.eval_set_results_manager
      == mock_gcs_eval_set_results_manager.return_value
  )


def test_create_gcs_eval_managers_from_uri_failure():
  with pytest.raises(ValueError):
    evals.create_gcs_eval_managers_from_uri('unsupported-uri')


def _run_in_fresh_interpreter(script: str) -> subprocess.CompletedProcess[str]:
  """Runs a script in a new interpreter, so the import state is pristine.

  The defect under test only appears on a cold import, and calling the factory
  warms the container up for the rest of the process. An in-process test would
  therefore pass or fail depending on which sibling test ran first.

  Args:
      script: The Python source to run. It is dedented before it runs.

  Returns:
      The completed process, with stdout and stderr captured as text.
  """
  return subprocess.run(
      [sys.executable, '-c', textwrap.dedent(script)],
      capture_output=True,
      text=True,
      check=False,
  )


def test_gcs_eval_managers_constructible_on_a_fresh_import():
  """The container must build in a frame that has no manager class local."""
  result = _run_in_fresh_interpreter("""
      from google.adk.cli.utils import evals

      eval_sets_manager = object()
      eval_set_results_manager = object()

      managers = evals.GcsEvalManagers(
          eval_sets_manager=eval_sets_manager,
          eval_set_results_manager=eval_set_results_manager,
      )

      assert managers.eval_sets_manager is eval_sets_manager
      assert managers.eval_set_results_manager is eval_set_results_manager
  """)

  assert result.returncode == 0, result.stderr


def test_gcs_eval_managers_is_not_a_pydantic_model():
  """A pydantic model here cannot resolve its TYPE_CHECKING-only fields."""
  assert not issubclass(evals.GcsEvalManagers, pydantic.BaseModel)
  assert dataclasses.is_dataclass(evals.GcsEvalManagers)


def test_evals_module_does_not_import_google_cloud_storage():
  """Importing the module must not pull in the optional GCS dependency."""
  result = _run_in_fresh_interpreter("""
      import sys

      # Poisoning sys.modules makes `import google.cloud.storage` raise.
      sys.modules['google.cloud.storage'] = None

      import google.adk.cli.utils.evals

      assert 'google.adk.evaluation.gcs_eval_sets_manager' not in sys.modules
      assert (
          'google.adk.evaluation.gcs_eval_set_results_manager'
          not in sys.modules
      )
  """)

  assert result.returncode == 0, result.stderr


def _event(author: str, text: str, invocation_id: str) -> Event:
  return Event(
      author=author,
      invocation_id=invocation_id,
      content=types.Content(
          role='user' if author == 'user' else 'model',
          parts=[types.Part(text=text)],
      ),
  )


def _session(events: list[Event]) -> Session:
  return Session(id='s1', app_name='app', user_id='u1', events=events)


def test_convert_session_to_eval_invocations_groups_events_by_invocation():
  session = _session([
      _event('user', 'first question', 'inv-1'),
      _event('agent', 'first answer', 'inv-1'),
      _event('user', 'second question', 'inv-2'),
      _event('agent', 'second answer', 'inv-2'),
  ])

  invocations = evals.convert_session_to_eval_invocations(session)

  assert [i.invocation_id for i in invocations] == ['inv-1', 'inv-2']
  assert [i.user_content.parts[0].text for i in invocations] == [
      'first question',
      'second question',
  ]
  assert [i.final_response.parts[0].text for i in invocations] == [
      'first answer',
      'second answer',
  ]


def test_convert_session_to_eval_invocations_handles_missing_history():
  """The CLI calls this before a session has any turns, and on no session."""
  assert evals.convert_session_to_eval_invocations(_session([])) == []
  assert evals.convert_session_to_eval_invocations(None) == []
