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

"""Behaviour of the committed AgentConfig.json under a JSON Schema validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.adk import agents
from jsonschema import Draft202012Validator
from jsonschema import exceptions
import pytest
import yaml

SCHEMA_PATH = (
    Path(agents.__file__).parent / "config_schemas" / "AgentConfig.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)

# The agent_class each typed branch of the union stands for. BaseAgentConfig
# is the permissive fallback branch and is excluded on purpose.
TYPED_BRANCHES = {
    "LlmAgentConfig": "LlmAgent",
    "LoopAgentConfig": "LoopAgent",
    "ParallelAgentConfig": "ParallelAgent",
    "SequentialAgentConfig": "SequentialAgent",
}
# Validate a document against one branch alone, with $defs injected so the
# $refs inside the branch still resolve.
TYPED_VALIDATORS = {
    name: Draft202012Validator(
        {**SCHEMA["$defs"][name], "$defs": SCHEMA["$defs"]}
    )
    for name in TYPED_BRANCHES
}

# The samples dir lives at the repo root on GitHub but under
# "open_source_workspace/" in the source tree, and neither ships in the sdist.
_ADK_ROOT = Path(__file__).resolve().parents[3]
_CANDIDATE_SAMPLES_DIRS = (
    _ADK_ROOT / "contributing" / "samples",
    _ADK_ROOT / "open_source_workspace" / "contributing" / "samples",
)
SAMPLES_DIR = next(
    (candidate for candidate in _CANDIDATE_SAMPLES_DIRS if candidate.is_dir()),
    _CANDIDATE_SAMPLES_DIRS[0],
)
if not SAMPLES_DIR.is_dir():
  pytest.skip(
      "contributing/samples is not available in this layout",
      allow_module_level=True,
  )


def collect_tagged_samples() -> list[Any]:
  """Returns pytest params for the samples annotated with the ADK schema."""
  params = []
  for path in sorted(SAMPLES_DIR.rglob("*.yaml")):
    if "AgentConfig.json" not in path.read_text(encoding="utf-8"):
      continue
    params.append(
        pytest.param(path, id=str(path.relative_to(SAMPLES_DIR.parent)))
    )
  return params


TAGGED_SAMPLES = collect_tagged_samples()


def matching_typed_branches(document: Any) -> set[str]:
  """Returns the typed branches that accept the document."""
  return {
      name
      for name, validator in TYPED_VALIDATORS.items()
      if validator.is_valid(document)
  }


def test_agent_config_schema_uses_any_of_at_the_top_level() -> None:
  """The branches overlap by design, so exactly-one matching is unachievable."""
  assert "oneOf" not in SCHEMA
  assert [branch["$ref"] for branch in SCHEMA["anyOf"]] == [
      "#/$defs/LlmAgentConfig",
      "#/$defs/LoopAgentConfig",
      "#/$defs/ParallelAgentConfig",
      "#/$defs/SequentialAgentConfig",
      "#/$defs/BaseAgentConfig",
  ]


def test_samples_are_tagged_with_the_agent_config_schema() -> None:
  """The parametrized sample tests below are not vacuous."""
  assert TAGGED_SAMPLES


@pytest.mark.parametrize("sample_path", TAGGED_SAMPLES)
def test_sample_agent_config_validates_against_committed_schema(
    sample_path: Path,
) -> None:
  """Every sample that points at the schema validates against it."""
  document = yaml.safe_load(sample_path.read_text(encoding="utf-8"))

  error = exceptions.best_match(VALIDATOR.iter_errors(document))

  assert error is None, f"{sample_path}: {error}"


@pytest.mark.parametrize("sample_path", TAGGED_SAMPLES)
def test_sample_agent_config_never_matches_another_agent_class_branch(
    sample_path: Path,
) -> None:
  """A sample matches no typed branch other than the one it declares."""
  document = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
  agent_class = document.get("agent_class", "LlmAgent")

  own_branch = {
      name for name, value in TYPED_BRANCHES.items() if value == agent_class
  }
  assert matching_typed_branches(document) <= own_branch


@pytest.mark.parametrize(
    "document, expected_branch",
    [
        ({"name": "a", "instruction": "i"}, "LlmAgentConfig"),
        (
            {"agent_class": "LlmAgent", "name": "a", "instruction": "i"},
            "LlmAgentConfig",
        ),
        ({"agent_class": "LoopAgent", "name": "a"}, "LoopAgentConfig"),
        ({"agent_class": "ParallelAgent", "name": "a"}, "ParallelAgentConfig"),
        (
            {"agent_class": "SequentialAgent", "name": "a"},
            "SequentialAgentConfig",
        ),
    ],
)
def test_minimal_config_matches_only_its_own_typed_branch(
    document: dict[str, Any], expected_branch: str
) -> None:
  """A minimal config for an agent class selects that class's branch alone."""
  assert VALIDATOR.is_valid(document)
  assert matching_typed_branches(document) == {expected_branch}


def test_custom_agent_class_matches_no_typed_branch() -> None:
  """A custom agent class still validates, through BaseAgentConfig."""
  document = {
      "agent_class": "my_package.MyAgent",
      "name": "a",
      "my_custom_field": "value",
  }

  assert VALIDATOR.is_valid(document)
  assert not matching_typed_branches(document)
