#!/usr/bin/env bash
set -euo pipefail
# Uses the activated environment. Arguments can override these ABC defaults.
args=(--policy abcvla-bimanual-yam --backend tensorrt --port 8202)
if [[ -n "${ABCVLA_BUNDLE:-}" ]]; then
  args+=(--engine-dir "$ABCVLA_BUNDLE")
fi
exec "${VLA_PYTHON:-python}" -m vla_edge.serving.server "${args[@]}" "$@"
