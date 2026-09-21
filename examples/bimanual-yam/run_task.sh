#!/usr/bin/env bash
# Shared task entry point; the connected server selects the policy controller.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${YAM_PYTHON:-python3}" "$HERE/run_task.py" "$@"
