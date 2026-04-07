#!/bin/bash
# Pre-commit quality gate for Claude Code.
# Runs ruff lint/format check + pytest before allowing git commit.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only intercept git commit/merge commands
if ! [[ "$COMMAND" =~ git[[:space:]]+(commit|merge) ]]; then
  exit 0
fi

CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
cd "$CWD" || exit 0

ERRORS=""

# 1. Ruff lint
LINT_OUT=$("$CWD/.venv/bin/ruff" check voice_pipeline/ 2>&1)
if [ $? -ne 0 ]; then
  ERRORS="${ERRORS}[ruff check] ${LINT_OUT}\n"
fi

# 2. Ruff format
FMT_OUT=$("$CWD/.venv/bin/ruff" format --check voice_pipeline/ 2>&1)
if [ $? -ne 0 ]; then
  ERRORS="${ERRORS}[ruff format] ${FMT_OUT}\n"
fi

# 3. Pytest (unit tests, stop on first failure)
TEST_OUT=$("$CWD/.venv/bin/pytest" -x -q 2>&1 | tail -5)
if [ $? -ne 0 ]; then
  ERRORS="${ERRORS}[pytest] ${TEST_OUT}\n"
fi

if [ -n "$ERRORS" ]; then
  jq -n --arg reason "$ERRORS" '{
    "hookSpecificOutput": {
      "hookEventName": "PreToolUse",
      "permissionDecision": "deny",
      "permissionDecisionReason": $reason
    }
  }'
fi

exit 0
