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

"""Script to generate AgentConfig.json from AgentConfig."""

from __future__ import annotations

import copy
import json
import os
from typing import Any
from typing import TYPE_CHECKING

from google.adk.agents.agent_config import AgentConfig
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import override

if TYPE_CHECKING:
  from pydantic._internal._core_utils import CoreSchemaOrField

# The branch of the AgentConfig union that accepts every other agent_class
# value. It stays unpinned so an unrecognised class still has somewhere to go.
_FALLBACK_DEF = "BaseAgentConfig"


class CustomGenerateJsonSchema(GenerateJsonSchema):
  """Custom schema generator that handles invalid types by falling back."""

  @override
  def handle_invalid_for_json_schema(
      self, schema: CoreSchemaOrField, error_info: str
  ) -> JsonSchemaValue:
    try:
      return super().handle_invalid_for_json_schema(schema, error_info)
    except PydanticInvalidForJsonSchema:
      # Return a fallback schema instead of failing
      return {
          "type": "object",
          "description": f"Fallback for invalid schema: {error_info}",
      }


def apply_agent_class_discriminator(schema: dict[str, Any]) -> dict[str, Any]:
  """Restores the agent_class routing rule that pydantic cannot render.

  `AgentConfig` is a root model over a union tagged with a callable pydantic
  `Discriminator`. Pydantic cannot express a callable discriminator in JSON
  Schema, so the union comes out as a bare `oneOf` and the routing rule is
  lost. That `oneOf` is unsatisfiable: `BaseAgentConfig` allows extra
  properties by design, so it matches almost every agent document, and
  `agent_class` is an unconstrained string in every typed branch. An ordinary
  `LlmAgent` document therefore matches two branches, and `oneOf` means
  exactly one.

  This transform makes the union an `anyOf` and pins `agent_class` on every
  branch but `BaseAgentConfig`, using the default that branch already
  declares. `BaseAgentConfig` stays permissive, which matches the runtime rule
  that an unrecognised `agent_class` routes to `BaseAgent`.

  Args:
    schema: A JSON Schema document produced by
      `AgentConfig.model_json_schema()`.

  Returns:
    A new document with the transform applied. The argument is not mutated,
    and applying the transform to its own output changes nothing.

  Raises:
    ValueError: If the document holds no top-level `oneOf` or `anyOf` union.
    KeyError: If a branch of the union names a definition that is missing, or
      that declares no default `agent_class`.
  """
  if "oneOf" not in schema and "anyOf" not in schema:
    raise ValueError(
        "Expected a top-level 'oneOf' or 'anyOf' union in the AgentConfig"
        " schema."
    )

  # Rebuild the mapping so the renamed key keeps its original position.
  result = {
      ("anyOf" if key == "oneOf" else key): value
      for key, value in copy.deepcopy(schema).items()
  }

  defs = result.get("$defs", {})
  for branch in result["anyOf"]:
    def_name = branch["$ref"].rsplit("/", 1)[-1]
    if def_name == _FALLBACK_DEF:
      continue
    properties = defs[def_name]["properties"]
    agent_class = properties["agent_class"]
    # Sort the keys so the const lands where pydantic would have emitted it.
    properties["agent_class"] = dict(
        sorted({**agent_class, "const": agent_class["default"]}.items())
    )
  return result


def main() -> None:
  """Generates the AgentConfig.json schema."""
  # Use the custom generator to avoid failing on httpx.Client
  schema = apply_agent_class_discriminator(
      AgentConfig.model_json_schema(schema_generator=CustomGenerateJsonSchema)
  )

  # Find the repo root relative to this file.
  script_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.dirname(script_dir)

  output_path = os.path.join(
      repo_root, "src/google/adk/agents/config_schemas/AgentConfig.json"
  )

  # Ensure directory exists
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
    f.write("\n")

  print(f"Successfully generated {output_path}")


if __name__ == "__main__":
  main()
