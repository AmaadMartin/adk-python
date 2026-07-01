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

import keyword
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_serializer

from ..._gemini_schema_util import _to_snake_case


def rename_python_keywords(s: str, prefix: str = 'param_') -> str:
  """Renames Python keywords by adding a prefix.

  Example:
  ```
  rename_python_keywords('if') -> 'param_if'
  rename_python_keywords('for') -> 'param_for'
  ```

  Args:
      s: The input string.
      prefix: The prefix to add to the keyword.

  Returns:
      The renamed string.
  """
  if keyword.iskeyword(s):
    return prefix + s
  return s


class ApiParameter(BaseModel):
  """Data class representing a function parameter."""

  original_name: str
  param_location: str
  param_schema: Union[str, Schema]
  description: Optional[str] = ''
  py_name: Optional[str] = ''
  type_value: type[Any] = Field(default=None, init_var=False)
  type_hint: str = Field(default=None, init_var=False)
  required: bool = False

  def model_post_init(self, _: Any):
    if not self.py_name:
      inferred_name = rename_python_keywords(_to_snake_case(self.original_name))
      self.py_name = inferred_name or self._default_py_name()
    if isinstance(self.param_schema, str):
      self.param_schema = Schema.model_validate_json(self.param_schema)

    self.description = self.description or self.param_schema.description or ''
    self.type_value = TypeHintHelper.get_type_value(self.param_schema)
    self.type_hint = TypeHintHelper.get_type_hint(self.param_schema)
    return self

  def _default_py_name(self) -> str:
    location_defaults = {
        'body': 'body',
        'query': 'query_param',
        'path': 'path_param',
        'header': 'header_param',
        'cookie': 'cookie_param',
    }
    return location_defaults.get(self.param_location or '', 'value')

  @model_serializer
  def _serialize(self):
    return {
        'original_name': self.original_name,
        'param_location': self.param_location,
        'param_schema': self.param_schema,
        'description': self.description,
        'py_name': self.py_name,
    }

  def __str__(self):
    return f'{self.py_name}: {self.type_hint}'

  def to_arg_string(self):
    """Converts the parameter to an argument string for function call."""
    return f'{self.py_name}={self.py_name}'

  def to_dict_property(self):
    """Converts the parameter to a key:value string for dict property."""
    return f'"{self.py_name}": {self.py_name}'

  def to_pydoc_string(self):
    """Converts the parameter to a PyDoc parameter docstr."""
    return PydocHelper.generate_param_doc(self)


class TypeHintHelper:
  """Helper class for generating type hints."""

  @staticmethod
  def get_type_value(schema: Schema) -> Any:
    """Generates the Python type value for a given parameter."""
    param_type = schema.type if schema.type else Any

    if param_type == 'integer':
      return int
    elif param_type == 'number':
      return float
    elif param_type == 'boolean':
      return bool
    elif param_type == 'string':
      return str
    elif param_type == 'array':
      items_type = Any
      if schema.items and schema.items.type:
        items_type = schema.items.type

      if items_type == 'object':
        return List[Dict[str, Any]]
      else:
        type_map = {
            'integer': int,
            'number': float,
            'boolean': bool,
            'string': str,
            'object': Dict[str, Any],
            'array': List[Any],
        }
        return List[type_map.get(items_type, 'Any')]
    elif param_type == 'object':
      return Dict[str, Any]
    else:
      return Any

  @staticmethod
  def get_type_hint(schema: Schema) -> str:
    """Generates the Python type in string for a given parameter."""
    param_type = schema.type if schema.type else 'Any'

    if param_type == 'integer':
      return 'int'
    elif param_type == 'number':
      return 'float'
    elif param_type == 'boolean':
      return 'bool'
    elif param_type == 'string':
      return 'str'
    elif param_type == 'array':
      items_type = 'Any'
      if schema.items and schema.items.type:
        items_type = schema.items.type

      if items_type == 'object':
        return 'List[Dict[str, Any]]'
      else:
        type_map = {
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'string': 'str',
        }
        return f"List[{type_map.get(items_type, 'Any')}]"
    elif param_type == 'object':
      return 'Dict[str, Any]'
    else:
      return 'Any'


class PydocHelper:
  """Helper class for generating PyDoc strings."""

  @staticmethod
  def generate_param_doc(
      param: ApiParameter,
  ) -> str:
    """Generates a parameter documentation string.

    Args:
      param: ApiParameter - The parameter to generate the documentation for.

    Returns:
      str: The generated parameter Python documentation string.
    """
    description = param.description.strip() if param.description else ''
    param_doc = f'{param.py_name} ({param.type_hint}): {description}'

    if param.param_schema.type == 'object':
      properties = param.param_schema.properties
      if properties:
        param_doc += ' Object properties:\n'
        for prop_name, prop_details in properties.items():
          prop_desc = prop_details.description or ''
          prop_type = TypeHintHelper.get_type_hint(prop_details)
          param_doc += f'       {prop_name} ({prop_type}): {prop_desc}\n'

    return param_doc

  @staticmethod
  def generate_return_doc(responses: Dict[str, Response]) -> str:
    """Generates a return value documentation string.

    Args:
      responses: Dict[str, TypedDict[Response]] - Response in an OpenAPI
        Operation

    Returns:
      str: The generated return value Python documentation string.
    """
    return_doc = ''

    # Only consider 2xx responses for return type hinting.
    # Returns the 2xx response with the smallest status code number and with
    # content defined.
    sorted_responses = sorted(responses.items(), key=lambda item: int(item[0]))
    qualified_response = next(
        filter(
            lambda r: r[0].startswith('2') and r[1].content,
            sorted_responses,
        ),
        None,
    )
    if not qualified_response:
      return ''
    response_details = qualified_response[1]

    description = (response_details.description or '').strip()
    content = response_details.content or {}

    # Group content types by schema to handle multiple content types.
    unique_schemas: List[tuple[Schema, List[str]]] = []
    for content_type, schema_details in content.items():
      schema = schema_details.schema_ or Schema()
      for existing_schema, content_types in unique_schemas:
        if existing_schema == schema:
          content_types.append(content_type)
          break
      else:
        unique_schemas.append((schema, [content_type]))

    # Generate combined type hint
    unique_type_hints = list(
        dict.fromkeys(
            TypeHintHelper.get_type_hint(schema)
            for schema, _ in unique_schemas
        )
    )
    combined_type_hint = (
        unique_type_hints[0]
        if len(unique_type_hints) == 1
        else f'Union[{", ".join(unique_type_hints)}]'
    )

    return_doc = f'Returns ({combined_type_hint}): {description}'

    properties_docs = []
    for schema, content_types in unique_schemas:
      if schema.type != 'object' or not schema.properties:
        continue
      header = (
          f'        Object properties ({", ".join(sorted(content_types))}):\n'
          if len(unique_schemas) > 1
          else ' Object properties:\n'
      )
      prop_lines = [
          f'        {k} ({TypeHintHelper.get_type_hint(v)}): {v.description or ""}'
          for k, v in schema.properties.items()
      ]
      properties_docs.append(header + '\n'.join(prop_lines) + '\n')

    if properties_docs:
      nl = '\n' if len(unique_schemas) > 1 else ''
      return_doc += f'{nl}{"".join(properties_docs)}'

    return return_doc
