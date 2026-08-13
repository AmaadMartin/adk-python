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
"""Utility functions for handling artifact URIs."""

from __future__ import annotations

from collections.abc import Collection
import re
from typing import NamedTuple

from google.genai import types

from ..errors import input_validation_error

USER_NAMESPACE_PREFIX = "user:"


class ParsedArtifactUri(NamedTuple):
  """The result of parsing an artifact URI."""

  app_name: str
  user_id: str
  session_id: str | None
  filename: str
  version: int


_SESSION_SCOPED_ARTIFACT_URI_RE = re.compile(
    r"artifact://apps/([^/]+)/users/([^/]+)/sessions/([^/]+)/artifacts/(.+)/versions/(\d+)"
)
_USER_SCOPED_ARTIFACT_URI_RE = re.compile(
    r"artifact://apps/([^/]+)/users/([^/]+)/artifacts/(.+)/versions/(\d+)"
)


def parse_artifact_uri(uri: str) -> ParsedArtifactUri | None:
  """Parses an artifact URI.

  Args:
      uri: The artifact URI to parse.

  Returns:
      A ParsedArtifactUri if parsing is successful, None otherwise.
  """
  if not uri or not uri.startswith("artifact://"):
    return None

  match = _SESSION_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=match.group(3),
        filename=match.group(4),
        version=int(match.group(5)),
    )

  match = _USER_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=None,
        filename=match.group(3),
        version=int(match.group(4)),
    )

  return None


def get_artifact_uri(
    app_name: str,
    user_id: str,
    filename: str,
    version: int,
    session_id: str | None = None,
) -> str:
  """Constructs an artifact URI.

  Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      filename: The name of the artifact file.
      version: The version of the artifact.
      session_id: The ID of the session.

  Returns:
      The constructed artifact URI.
  """
  if session_id:
    return f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
  else:
    return f"artifact://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"


def is_artifact_ref(artifact: types.Part) -> bool:
  """Checks if an artifact part is an artifact reference.

  Args:
      artifact: The artifact part to check.

  Returns:
      True if the artifact part is an artifact reference, False otherwise.
  """
  return bool(
      artifact.file_data
      and artifact.file_data.file_uri
      and artifact.file_data.file_uri.startswith("artifact://")
  )


def validate_artifact_reference_scope(
    *,
    app_name: str,
    user_id: str,
    session_id: str | None,
    parsed_uri: ParsedArtifactUri,
) -> None:
  """Ensures artifact references cannot escape the caller's scope."""
  if parsed_uri.app_name != app_name or parsed_uri.user_id != user_id:
    raise input_validation_error.InputValidationError(
        "Artifact references must stay within the same app and user scope."
    )
  if parsed_uri.session_id is not None and parsed_uri.session_id != session_id:
    raise input_validation_error.InputValidationError(
        "Session-scoped artifact references must stay within the same"
        " session scope."
    )


def is_whitespace_padded_filename(filename: str) -> bool:
  """Checks whether an artifact filename has leading or trailing whitespace.

  A filename is a storage key, so two filenames that differ only by padding
  must address two different artifacts. `FileArtifactService` maps a filename
  onto a directory name, and Windows path normalization drops trailing spaces
  and periods from a path component
  (https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats#trim-characters).
  A padded filename is therefore not storable distinctly on every supported
  platform, so every backend rejects it. Do not turn this check back into a
  `strip()`: silently trimming makes `' a.txt'` overwrite `'a.txt'`.

  The `user:` prefix marks a namespace instead of forming part of the key, so
  it is removed before the check and `'user: a.txt'` counts as padded.

  Args:
    filename: The caller-supplied artifact filename.

  Returns:
    True if the filename is whitespace-padded.
  """
  scoped = filename.removeprefix(USER_NAMESPACE_PREFIX)
  return scoped != scoped.strip()


def validate_artifact_filename(filename: str) -> None:
  """Rejects an artifact filename with leading or trailing whitespace.

  Args:
    filename: The caller-supplied artifact filename.

  Raises:
    InputValidationError: If the filename is whitespace-padded. See
      `is_whitespace_padded_filename` for why padding cannot be stored.
  """
  if is_whitespace_padded_filename(filename):
    raise input_validation_error.InputValidationError(
        f"Artifact filename {filename!r} must not have leading or trailing"
        " whitespace."
    )


def validate_no_case_collision(
    existing_filenames: Collection[str], filename: str
) -> None:
  """Rejects a filename that differs only in case from a stored one.

  A filename is a storage key, so two filenames that differ only in case must
  address two different artifacts. `FileArtifactService` maps a filename onto a
  directory name, and the case-insensitive filesystems ADK supports (APFS,
  NTFS) resolve `Report.txt` and `report.txt` to one directory, so the second
  save appends a version to the first artifact and the two artifacts alias each
  other. That pair is not storable distinctly on every supported platform, so
  every backend rejects the second name.

  An exact match is always allowed: re-saving a stored key is how a new version
  is created.

  The comparison uses `str.lower()` rather than `str.casefold()`, which folds
  `ß` to `ss` and would reject `straße.txt` against `STRASSE.txt` — a pair APFS
  and NTFS keep apart. Unicode normalization is a distinct equivalence and is
  not handled here.

  Args:
    existing_filenames: The keys already stored in the same scope as
      `filename`, in the same representation (a `user:` prefix is kept on both
      sides).
    filename: The caller-supplied artifact filename.

  Raises:
    InputValidationError: If `filename` differs only in case from one of
      `existing_filenames`.
  """
  if filename in existing_filenames:
    return
  lowered = filename.lower()
  for existing in existing_filenames:
    if existing.lower() == lowered:
      raise input_validation_error.InputValidationError(
          f"Artifact filename {filename!r} differs only in case from existing"
          f" artifact {existing!r}."
      )


def validate_path_segment(value: str, field_name: str) -> None:
  """Rejects values that could alter the constructed path.

  Args:
    value: The caller-supplied identifier (e.g. user_id or session_id).
    field_name: Human-readable name used in the error message.

  Raises:
    InputValidationError: If the value contains traversal segments, null bytes,
      or is an absolute path / starts with a slash.
  """
  if not value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not be empty."
    )
  if "\x00" in value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not contain null bytes."
    )
  if isinstance(value, str) and (
      value.startswith("/") or value.startswith("\\")
  ):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not be an absolute path or start with a"
        " slash."
    )
  if (
      value in (".", "..")
      or ".." in value.split("/")
      or ".." in value.split("\\")
  ):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not contain traversal segments."
    )
