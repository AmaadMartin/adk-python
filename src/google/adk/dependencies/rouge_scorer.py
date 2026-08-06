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

# nltk 3.10.1 is the only release that shipped nltk/inisec.py, a meta-path
# finder that refuses to import any module whose file resolves under the
# current working directory while nltk is on the call stack. site-packages
# resolves under the current working directory whenever the virtual
# environment lives inside the project (the common '.venv/' layout), so nltk
# blocks its own dependency and rouge_score cannot import. Reverted upstream
# in nltk/nltk#3732 (nltk 3.10.2).
_NLTK_CWD_GUARD_MARKER = 'from current working directory for security reasons'

_NLTK_CWD_GUARD_HELP = (
    'Failed to import rouge_score because nltk 3.10.1 blocked one of its'
    ' imports. nltk 3.10.1 refuses to import any module whose file is located'
    ' under the current working directory, and site-packages is under it'
    " whenever the virtual environment lives inside the project (a '.venv/'"
    ' directory in the project root). Upgrade nltk to a release without that'
    ' import hook: pip install --upgrade "nltk!=3.10.1". The -P and'
    ' PYTHONSAFEPATH remedies named in the nltk error do not help, because'
    ' nltk checks where the file lives, not what is on sys.path.'
)

try:
  from rouge_score import rouge_scorer as rouge_scorer
  from rouge_score import tokenizers as tokenizers
except ImportError as e:
  if _NLTK_CWD_GUARD_MARKER not in str(e):
    raise
  raise ImportError(_NLTK_CWD_GUARD_HELP) from e
