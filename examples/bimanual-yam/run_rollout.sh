#!/usr/bin/env bash
# Run eval rollout(s), then always home, open the grippers, and de-energize.
# This applies to clean exit, crash, and Ctrl-C. If the launcher remains alive
# after its explicit completion line, the wrapper terminates it so the final
# hardware shutdown can proceed. Usage: run_rollout.sh [-n N]
#
#   YAM_PYTHON  python interpreter to use (default: python3)
set -uo pipefail
cd "$(dirname "$0")"
PY="${YAM_PYTHON:-python3}"
# Optional test knob: pad each backend's inference pause to this budget for
# cadence-matched comparisons. It is disabled by default because padding adds
# a second pause between chunks and makes observations stale during execution.
# Enable explicitly, for example: YAM_INFER_BUDGET_S=0.75 run_rollout.sh ...
export YAM_INFER_BUDGET_S="${YAM_INFER_BUDGET_S:-0}"
LOG=$(mktemp /tmp/yam_rollout.XXXXXX.log)
GRACE=20
TERM_GRACE=25

home_arms() { "$PY" -u home_arms.py --yes; }
trap home_arms EXIT

# run_task.sh points YAM_LEFT_CONFIG at its gitignored runtime copy; direct
# invocations use the tracked config as-is.
yes '' | "$PY" -u launch_yaml_eval_molmoact.py \
    --left_config_path "${YAM_LEFT_CONFIG:-configs/yam_left.yaml}" \
    --right_config_path configs/yam_right.yaml "$@" \
    > >(tee "$LOG") 2>&1 &
LPID=$!   # process-substitution keeps $! = the launcher itself, not tee

while kill -0 "$LPID" 2>/dev/null; do
    if grep -q '^\[session\] complete$' "$LOG" 2>/dev/null; then
        sleep $GRACE
        if kill -0 "$LPID" 2>/dev/null; then
            echo "[wrapper] launcher hung ${GRACE}s past session summary; terminating" >&2
            kill -TERM "$LPID"
            for ((tick = 0; tick < TERM_GRACE * 4; tick++)); do
                kill -0 "$LPID" 2>/dev/null || break
                sleep 0.25
            done
            if kill -0 "$LPID" 2>/dev/null; then
                echo "[wrapper] launcher did not park within ${TERM_GRACE}s; sending SIGKILL" >&2
                kill -KILL "$LPID"
            fi
        fi
        break
    fi
    sleep 3
done
wait "$LPID" 2>/dev/null
