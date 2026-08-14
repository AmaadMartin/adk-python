#!/bin/bash
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

# pre-commit wrapper for the addlicense tool.
#
# addlicense is an optional Go binary
# (go install github.com/google/addlicense@latest). When it is not installed
# the hook warns and passes, so a missing Go toolchain never blocks a commit.

if ! command -v addlicense >/dev/null 2>&1; then
  echo "Warning: addlicense not installed, skipping"
  exit 0
fi

exec addlicense -c "Google LLC" -l apache "$@"
