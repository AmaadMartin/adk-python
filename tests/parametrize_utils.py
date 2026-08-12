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

"""Shared parametrize-mark parsing for the conftest backend hooks."""

from pytest import Metafunc


def is_explicitly_marked(mark_name: str, metafunc: Metafunc) -> bool:
  """Reports whether the test already parametrizes `mark_name` itself.

  `iter_markers` walks the function, its class and its module, so a class-level
  mark counts. `argnames` may be positional or keyword, and either a
  comma-joined string or a sequence.

  Args:
    mark_name: The name of the argument to look for.
    metafunc: The test whose `parametrize` marks to read.

  Returns:
    True when one of the test's own `parametrize` marks declares `mark_name`.
  """
  for mark in metafunc.definition.iter_markers('parametrize'):
    argnames = mark.args[0] if mark.args else mark.kwargs.get('argnames', ())
    if isinstance(argnames, str):
      argnames = argnames.split(',')
    if mark_name in (name.strip() for name in argnames):
      return True
  return False
