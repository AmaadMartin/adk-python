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

# Manage the constraints-<ver>.txt pin files: check them, update them, or
# refresh the snapshot they resolve against.
#
# Usage:
#   ./scripts/update_constraints.sh            # Rewrite the files that no longer match
#                                              # pyproject.toml, reusing the snapshot date
#                                              # each file already records
#   ./scripts/update_constraints.sh --check    # Report drift only; never writes (for CI)
#   ./scripts/update_constraints.sh --refresh  # Advance every snapshot date to 4 days ago,
#                                              # then rewrite
#
# Exit codes: 0 success, 1 drift or resolution failure, 2 usage error.

set -e

usage() {
  echo "Usage: ./scripts/update_constraints.sh [--check | --refresh]" >&2
  echo "  --check     Report drift only; never writes constraints-<ver>.txt." >&2
  echo "  --refresh   Advance every snapshot date to 4 days ago before resolving." >&2
}

# Parse arguments. An unrecognised flag is a usage error rather than a silent
# fall-through to update mode: a typo such as --chek would otherwise turn the
# CI check job into a job that rewrites the files and verifies nothing.
CHECK_ONLY=false
REFRESH=false
for arg in "$@"; do
  case $arg in
    --check)
      CHECK_ONLY=true
      ;;
    --refresh)
      REFRESH=true
      ;;
    *)
      echo "❌ Unknown option: $arg" >&2
      usage
      exit 2
      ;;
  esac
done

if [ "$CHECK_ONLY" = true ] && [ "$REFRESH" = true ]; then
  echo "❌ --check and --refresh cannot be combined: --check never writes." >&2
  usage
  exit 2
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

cleanup() {
  rm -f constraints-*.tmp
}
trap cleanup EXIT

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13" "3.14")
EXIT_CODE=0

# Calculate 4 days ago date. This is the snapshot a file gets when it is
# created, or when --refresh advances it.
if date -v-4d +%Y-%m-%d >/dev/null 2>&1; then
  TODAY_MINUS_4=$(date -v-4d +%Y-%m-%d)
else
  TODAY_MINUS_4=$(date -d "4 days ago" +%Y-%m-%d)
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

  # Read back the snapshot date the file was generated with.
  recorded_date=""
  if [ -f "$TARGET_FILE" ]; then
    recorded_date=$(grep -h "#    uv pip compile" "$TARGET_FILE" | grep -oE -- '--exclude-newer [0-9]{4}-[0-9]{2}-[0-9]{2}' | cut -d' ' -f2 || true)
  fi

  if [ "$CHECK_ONLY" = true ] && [ -z "$recorded_date" ]; then
    echo "❌ $TARGET_FILE records no snapshot date!"
    echo "   Without an '--exclude-newer YYYY-MM-DD' header there is nothing to verify its pins against."
    echo "   Please regenerate it locally and commit the changes:"
    echo "   $ ./scripts/update_constraints.sh --refresh"
    EXIT_CODE=1
    continue
  fi

  # Reuse the recorded date in every mode. Recomputing it in update mode
  # rewrites all five files on any day the script runs, which buries a real
  # dependency change in unrelated churn; --refresh is the one way to advance
  # the snapshot.
  if [ "$REFRESH" = true ] || [ -z "$recorded_date" ]; then
    date_to_use="$TODAY_MINUS_4"
  else
    date_to_use="$recorded_date"
  fi

  # Construct the command from scratch
  GENERATION_CMD="uv pip compile pyproject.toml --all-extras --python-version $ver"
  GENERATION_CMD="$GENERATION_CMD --exclude-newer $date_to_use"
  GENERATION_CMD="$GENERATION_CMD --index-url https://pypi.org/simple -o $TARGET_FILE"

  echo "Found generation command: $GENERATION_CMD"

  STABLE_FILE="constraints-${ver}.txt.stable.tmp"
  NEW_FILE="constraints-${ver}.txt.new.tmp"

  # Copy the existing constraints to STABLE_FILE if it exists and is not empty
  if [ -s "$TARGET_FILE" ]; then
    cp "$TARGET_FILE" "$STABLE_FILE"
  else
    touch "$STABLE_FILE"
  fi

  # Modify the GENERATION_CMD to output to NEW_FILE.
  RUN_CMD=$(echo "$GENERATION_CMD" | sed -E "s/-o [^ ]+/-o $NEW_FILE/")
  RUN_CMD=$(echo "$RUN_CMD" | sed -E "s/--output-file [^ ]+/--output-file $NEW_FILE/")
  RUN_CMD=$(echo "$RUN_CMD" | sed -E "s/--output-file=[^ ]+/--output-file=$NEW_FILE/")

  # Execute the command, also adding STABLE_FILE as a constraint to stabilize resolution.
  echo "Running: $RUN_CMD --constraint $STABLE_FILE"
  if ! eval "$RUN_CMD --constraint $STABLE_FILE"; then
    if [ "$CHECK_ONLY" = true ]; then
      echo "❌ Resolution failed with stable constraints for $TARGET_FILE."
      echo "   This usually happens when a new dependency requirement in pyproject.toml conflicts with existing pinned versions."
      echo "   To fix this, run the update script locally to resolve conflicts and update constraints:"
      echo "   $ ./scripts/update_constraints.sh"
      rm -f "$STABLE_FILE" "$NEW_FILE"
      EXIT_CODE=1
      continue
    else
      echo "⚠️ Resolution failed with stable constraints. Retrying without constraints to allow upgrades..."
      echo "Running: $RUN_CMD"
      if ! eval "$RUN_CMD"; then
        echo "❌ Resolution failed even without constraints."
        rm -f "$STABLE_FILE" "$NEW_FILE"
        EXIT_CODE=1
        continue
      fi
    fi
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
    rm -f "$STABLE_FILE" "$NEW_FILE"
  else
    if [ "$CHECK_ONLY" = true ]; then
      echo "❌ $TARGET_FILE is OUT OF DATE!"
      echo "   Please run the update script locally to update it and commit the changes:"
      echo "   $ ./scripts/update_constraints.sh"
      rm -f "$STABLE_FILE" "$NEW_FILE"
      EXIT_CODE=1
    else
      echo "🔄 $TARGET_FILE was OUT OF DATE. Updating it automatically..."
      cp "$NEW_FILE" "$TARGET_FILE"
      echo "✅ $TARGET_FILE has been updated locally."
      rm -f "$STABLE_FILE" "$NEW_FILE"
    fi
  fi
done

exit $EXIT_CODE
