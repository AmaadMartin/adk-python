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

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from google.adk import agents
import pytest

# generate_agent_config_schema is a standalone script (not a package), so load
# it by path. The scripts dir lives at the repo root on GitHub but under
# "open_source_workspace/" in the source tree, so try both layouts.
_ADK_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = next(
    (
        candidate
        for candidate in (
            _ADK_ROOT / "scripts",
            _ADK_ROOT / "open_source_workspace" / "scripts",
        )
        if candidate.exists()
    ),
    _ADK_ROOT / "scripts",
)

# The agent_class each typed branch of the union stands for.
TYPED_BRANCHES = {
    "LlmAgentConfig": "LlmAgent",
    "LoopAgentConfig": "LoopAgent",
    "ParallelAgentConfig": "ParallelAgent",
    "SequentialAgentConfig": "SequentialAgent",
}

COMMITTED_SCHEMA_PATH = (
    Path(agents.__file__).parent / "config_schemas" / "AgentConfig.json"
)


def import_script(name: str) -> ModuleType:
  file_path = SCRIPTS_DIR / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script {name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


generate_agent_config_schema = import_script("generate_agent_config_schema")
apply_agent_class_discriminator = (
    generate_agent_config_schema.apply_agent_class_discriminator
)


def make_schema() -> dict[str, Any]:
  """Returns a small schema shaped like the generated AgentConfig union."""
  defs: dict[str, Any] = {
      "BaseAgentConfig": {
          "additionalProperties": True,
          "properties": {
              "agent_class": {"default": "BaseAgent", "type": "string"},
              "name": {"type": "string"},
          },
          "required": ["name"],
          "title": "BaseAgentConfig",
      }
  }
  for def_name, agent_class in TYPED_BRANCHES.items():
    defs[def_name] = {
        "additionalProperties": False,
        "properties": {
            "agent_class": {"default": agent_class, "type": "string"},
            "name": {"type": "string"},
        },
        "required": ["name"],
        "title": def_name,
    }
  return {
      "$defs": defs,
      "description": "The config for the YAML schema to create an agent.",
      "oneOf": [
          {"$ref": f"#/$defs/{name}"}
          for name in (*TYPED_BRANCHES, "BaseAgentConfig")
      ],
      "title": "AgentConfig",
  }


def test_top_level_one_of_becomes_any_of() -> None:
  """The union is renamed in place, so the key order does not change."""
  schema = make_schema()

  result = apply_agent_class_discriminator(schema)

  assert "oneOf" not in result
  assert result["anyOf"] == schema["oneOf"]
  assert list(result) == ["$defs", "description", "anyOf", "title"]


@pytest.mark.parametrize(
    "def_name, agent_class",
    [
        ("LlmAgentConfig", "LlmAgent"),
        ("LoopAgentConfig", "LoopAgent"),
        ("ParallelAgentConfig", "ParallelAgent"),
        ("SequentialAgentConfig", "SequentialAgent"),
    ],
)
def test_each_typed_branch_pins_its_agent_class(
    def_name: str, agent_class: str
) -> None:
  """Each typed branch constrains agent_class to the class it represents."""
  result = apply_agent_class_discriminator(make_schema())

  agent_class_schema = result["$defs"][def_name]["properties"]["agent_class"]
  assert agent_class_schema["const"] == agent_class


def test_base_agent_branch_is_left_permissive() -> None:
  """BaseAgentConfig keeps matching an unrecognised agent_class."""
  schema = make_schema()

  result = apply_agent_class_discriminator(schema)

  assert (
      result["$defs"]["BaseAgentConfig"] == schema["$defs"]["BaseAgentConfig"]
  )


def test_the_input_schema_is_not_mutated() -> None:
  """The transform reads its argument and returns a new document."""
  schema = make_schema()
  before = copy.deepcopy(schema)

  apply_agent_class_discriminator(schema)

  assert schema == before


def test_applying_the_transform_twice_changes_nothing() -> None:
  """The transform is idempotent, so a regenerated schema stays stable."""
  once = apply_agent_class_discriminator(make_schema())

  assert apply_agent_class_discriminator(once) == once


def test_missing_agent_branch_raises() -> None:
  """A renamed typed branch aborts the generator instead of being skipped."""
  schema = make_schema()
  del schema["$defs"]["LoopAgentConfig"]

  with pytest.raises(KeyError, match="LoopAgentConfig' is missing"):
    apply_agent_class_discriminator(schema)


def test_schema_without_a_union_raises() -> None:
  """A document that holds no union aborts the generator."""
  schema = make_schema()
  del schema["oneOf"]

  with pytest.raises(ValueError, match="oneOf"):
    apply_agent_class_discriminator(schema)


def test_committed_schema_is_already_post_processed() -> None:
  """The committed AgentConfig.json carries the transform already."""
  committed = json.loads(COMMITTED_SCHEMA_PATH.read_text(encoding="utf-8"))

  assert apply_agent_class_discriminator(committed) == committed
