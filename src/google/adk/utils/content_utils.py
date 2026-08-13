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

from collections.abc import Iterable
import json
from typing import Any

from google.genai import types
from pydantic import BaseModel

SKIP_THOUGHT_SIGNATURE_VALIDATOR: bytes = b'skip_thought_signature_validator'
"""Placeholder ``Part.thought_signature`` that bypasses backend validation.

Set it on a part you synthesize yourself (a model turn or tool call/response
the model never produced) so the Gemini backend accepts the fabricated part
instead of rejecting it for a missing signature.
"""


def is_audio_part(part: types.Part) -> bool:
  return (
      part.inline_data is not None
      and part.inline_data.mime_type is not None
      and part.inline_data.mime_type.startswith('audio/')
  ) or (
      part.file_data is not None
      and part.file_data.mime_type is not None
      and part.file_data.mime_type.startswith('audio/')
  )


def filter_audio_parts(content: types.Content) -> types.Content | None:
  if not content.parts:
    return None
  filtered_parts = [part for part in content.parts if not is_audio_part(part)]
  if not filtered_parts:
    return None
  return types.Content(role=content.role, parts=filtered_parts)


def extract_text_from_content(content: types.Content | None) -> str:
  """Extracts text from a Content object, filtering out thoughts."""
  if not content or not content.parts:
    return ''
  return ''.join(p.text for p in content.parts if p.text and not p.thought)


def _join_texts(texts: Iterable[str | None]) -> str | None:
  """Joins the non-empty fragments with a newline, or returns None."""
  return '\n'.join(text for text in texts if text) or None


def content_union_to_text(value: types.ContentUnion | None) -> str | None:
  """Reads the text carried by any arm of ``types.ContentUnion``.

  ``types.GenerateContentConfig.system_instruction`` is a ``ContentUnion``, so a
  reader must handle a ``str``, a ``types.Part``, a ``types.Content`` and a list
  of those. Several text fragments join with ``'\\n'``. Arms that carry no text
  at all (a ``types.File``, a PIL image, inline binary data) are dropped, so a
  value that carries only binary data reads as ``None``.

  Args:
    value: The value to read.

  Returns:
    The recovered text, or ``None`` when the value carries no text. Never the
    empty string. The argument is never mutated.
  """
  if not value:
    return None
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    return _join_texts(content_union_to_text(item) for item in value)
  if isinstance(value, types.Content):
    return _join_texts(
        content_union_to_text(part) for part in value.parts or []
    )
  if isinstance(value, types.Part):
    return value.text or None
  return None


def to_user_content(value: Any) -> types.Content:
  """Coerces an arbitrary value into a user-role Content.

  - types.Content -> re-wrapped with role='user' (parts list shared, not
    deep-copied)
  - str -> single text part
  - BaseModel -> model_dump_json() text part
  - dict/list -> json.dumps() text part (non-ASCII preserved, not escaped)
  - anything else -> str() text part
  """
  if isinstance(value, types.Content):
    return types.Content(role='user', parts=value.parts)
  if isinstance(value, str):
    text = value
  elif isinstance(value, BaseModel):
    text = value.model_dump_json()
  elif isinstance(value, (dict, list)):
    text = json.dumps(value, ensure_ascii=False)
  else:
    text = str(value)
  return types.Content(role='user', parts=[types.Part(text=text)])
