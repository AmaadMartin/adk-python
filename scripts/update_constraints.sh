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

# Manage constraints.txt: check if up-to-date or automatically update it.
#
# Usage:
#   ./scripts/update_constraints.sh          # Updates constraints.txt in-place if out of date
#   ./scripts/update_constraints.sh --check  # Check only, exits with 1 if out of date (for CI)

set -e

# Parse arguments
CHECK_ONLY=false
for arg in "$@"; do
  case $arg in
    --check)
      CHECK_ONLY=true
      shift
      ;;
  esac
done

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

cleanup() {
  rm -f constraints-*.tmp
}
trap cleanup EXIT

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")
EXIT_CODE=0

# Calculate 4 days ago date
if [ "$CHECK_ONLY" = false ]; then
  if date -v-4d +%Y-%m-%d >/dev/null 2>&1; then
    EXCLUDE_NEWER_DATE=$(date -v-4d +%Y-%m-%d)
  else
    EXCLUDE_NEWER_DATE=$(date -d "4 days ago" +%Y-%m-%d)
  fi
fi

for ver in "${PYTHON_VERSIONS[@]}"; do
  TARGET_FILE="constraints-${ver}.txt"
  echo "Processing $TARGET_FILE..."

  if [ ! -f "$TARGET_FILE" ]; then
    if [ "$CHECK_ONLY" = true ]; then
      echo "❌ $TARGET_FILE is missing!"
      EXIT_CODE=1
      continue
    fi
  fi

  # Default date to what we calculated (for update mode)
  date_to_use="$EXCLUDE_NEWER_DATE"

  if [ -f "$TARGET_FILE" ]; then
    if [ "$CHECK_ONLY" = true ]; then
      # In check mode, extract the date used when it was generated
      date_to_use=$(grep -h "#    uv pip compile" "$TARGET_FILE" | grep -oE -- '--exclude-newer [0-9]{4}-[0-9]{2}-[0-9]{2}' | cut -d' ' -f2 || true)
      # Exactly one date, or there is nothing to verify against. Resolving
      # without --exclude-newer would silently check the pins against the live
      # index, and a header recording several dates would expand into stray
      # argv. Leave the pattern unquoted; quoting makes bash match it
      # literally.
      if [[ ! $date_to_use =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "❌ $TARGET_FILE has no single '--exclude-newer YYYY-MM-DD' snapshot date in its header,"
        echo "   so there is nothing to verify its pins against. Regenerate it locally and commit the changes:"
        echo "   $ ./scripts/update_constraints.sh"
        EXIT_CODE=1
        continue
      fi
    fi
  fi

  # Construct the command from scratch. The date is always known here: check
  # mode has just validated it, and update mode owns it.
  #
  # --no-emit-package google-adk keeps the resolved package out of the output
  # without dropping its requirements. The community and toolbox extras depend
  # back on google-adk from PyPI, so uv resolves the published release as a
  # node in the graph and would otherwise pin it. These files are applied to
  # the very install they constrain, so such a pin holds the user at whatever
  # release was current when the files were last regenerated.
  GENERATION_CMD="uv pip compile pyproject.toml --all-extras --no-emit-package google-adk --python-version $ver"
  GENERATION_CMD="$GENERATION_CMD --exclude-newer $date_to_use"
  GENERATION_CMD="$GENERATION_CMD --index-url https://pypi.org/simple -o $TARGET_FILE"

  echo "Found generation command: $GENERATION_CMD"

  NEW_FILE="constraints-${ver}.txt.new.tmp"

  # Seed the resolution with the committed pins: uv reads its own output file
  # and prefers the versions already recorded there. Passing that file with
  # --constraint instead makes uv record it as a resolution source and stamp
  # "-c constraints-<ver>.txt.stable.tmp" into the published annotations, which
  # names a scratch file the reader never has.
  rm -f "$NEW_FILE"
  if [ -s "$TARGET_FILE" ]; then
    cp "$TARGET_FILE" "$NEW_FILE"
  fi

  # Modify the GENERATION_CMD to output to NEW_FILE.
  RUN_CMD=$(echo "$GENERATION_CMD" | sed -E "s/-o [^ ]+/-o $NEW_FILE/")
  RUN_CMD=$(echo "$RUN_CMD" | sed -E "s/--output-file [^ ]+/--output-file $NEW_FILE/")
  RUN_CMD=$(echo "$RUN_CMD" | sed -E "s/--output-file=[^ ]+/--output-file=$NEW_FILE/")

  # Seeded pins are preferences, not hard constraints: uv keeps one while it
  # stays valid and picks a new version when it does not. A stale pin can no
  # longer abort the resolution, so a single attempt is enough.
  echo "Running: $RUN_CMD"
  if ! eval "$RUN_CMD"; then
    echo "❌ Resolution failed for $TARGET_FILE."
    rm -f "$NEW_FILE"
    EXIT_CODE=1
    continue
  fi

  # Reconstruct NEW_FILE to have the clean GENERATION_CMD in its header
  CLEAN_FILE="constraints-${ver}.txt.clean.tmp"
  {
    echo "# This file was autogenerated by uv via the following command:"
    echo "#    $GENERATION_CMD"
    tail -n +3 "$NEW_FILE"
  } > "$CLEAN_FILE"
  mv "$CLEAN_FILE" "$NEW_FILE"

  # Compare
  if diff -u "$TARGET_FILE" "$NEW_FILE"; then
    echo "✅ $TARGET_FILE is up-to-date."
    rm -f "$NEW_FILE"
  else
    if [ "$CHECK_ONLY" = true ]; then
      echo "❌ $TARGET_FILE is OUT OF DATE!"
      echo "   Please run the update script locally to update it and commit the changes:"
      echo "   $ ./scripts/update_constraints.sh"
      rm -f "$NEW_FILE"
      EXIT_CODE=1
    else
      echo "🔄 $TARGET_FILE was OUT OF DATE. Updating it automatically..."
      cp "$NEW_FILE" "$TARGET_FILE"
      echo "✅ $TARGET_FILE has been updated locally."
      rm -f "$NEW_FILE"
      EXIT_CODE=1
    fi
  fi
done

exit $EXIT_CODE
