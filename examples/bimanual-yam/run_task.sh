#!/usr/bin/env bash
# Run one eval rollout with an arbitrary instruction, any time.
#
#   examples/bimanual-yam/run_task.sh "fold the blue shirt neatly on the table"
#   examples/bimanual-yam/run_task.sh "pick up the rubik cube with one arm and put it in the black basket"
#
# What it does:
#   1. copies configs/yam_left.yaml to configs/yam_left.run.yaml (gitignored)
#      and writes the instruction + server into the copy, so the tracked
#      config never gets dirtied; the copy keeps the last run's values as a
#      record of what ran
#   2. preflight: inference server, camera server, and both CAN links must be up
#   3. launches run_rollout.sh with the stage file armed and the velocity cap set
#
# While it runs:
#   - `touch /tmp/yam_done` ends the default one-rollout session, labels it
#     success, parks both arms, and lets the wrapper home and de-energize them
#   - Ctrl-C aborts; run_rollout.sh always homes + de-energizes the arms after
#
# Env overrides (all optional):
#   YAM_PYTHON          python interpreter to use   (default python3)
#   YAM_CAN_LEFT        left-arm CAN interface name  (default: left config)
#   YAM_CAN_RIGHT       right-arm CAN interface name (default: right config)
#   YAM_SERVER          inference server host:port    (default 127.0.0.1:8202,
#                       the vla-edge MolmoAct2 server; override applies to ONE
#                       invocation)
#                       A server without RTC guidance cannot be used with
#                       YAM_RTC=1
#   YAM_MAX_JOINT_VEL   arm speed cap, rad/s          (default 2.2; start lower
#                       while validating a new rig)
#   YAM_STAGE_FILE      success-marker path           (default /tmp/yam_done)
#   YAM_ACTION_HORIZON  actions consumed per chunk    (default 30, full chunk)
#   YAM_ACTION_EMA      arm smoothing alpha           (default 1.0 = raw,
#                       training-rig semantics; <1.0 enables smoothing)
#   YAM_LIMITER         clamp (default; one +/-tick*V_MAX command per tick
#                       vs the measured arm pose, never blocks or rewinds)
#                       | ramp (legacy blocking catch-up; produces a
#                       command sawtooth the servos render as grinding)
#   YAM_GRIPPER_RATE    gripper travel per tick       (default 0.15)
#   YAM_ASYNC_PLAN      background planning+merge     (default 1; 0 = sync)
#   YAM_REPLAN_THRESHOLD async: queue fraction remaining that triggers a
#                       replan (default 0.2; tune for backend latency)
#   YAM_RTC             RTC queue replacement         (default 0; validate on
#                       the target backend before enabling)
#   YAM_RTC_HORIZON     raw queued prefix length      (default 10)
#   YAM_RTC_TRIGGER     auto/continuous/queue rows    (default auto = delay+horizon)
#   YAM_RTC_SCHEDULE    zeros/ones/linear/exp         (default linear)
#   YAM_RTC_MAX_GUIDANCE maximum guidance weight      (default 10.0)
# Extra args after the instruction are passed through to the launcher.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "usage: $0 \"<instruction>\" [extra launcher args]" >&2
    exit 2
fi
INSTR=$1; shift

CFG=configs/yam_left.yaml
RIGHT_CFG=configs/yam_right.yaml
RUN_CFG=configs/yam_left.run.yaml
PY="${YAM_PYTHON:-python3}"
export YAM_STAGE_FILE="${YAM_STAGE_FILE:-/tmp/yam_done}"
export YAM_MAX_JOINT_VEL="${YAM_MAX_JOINT_VEL:-2.2}"

# Use the rig setup written into the configs. Explicit env values still win.
read -r CFG_CAN_LEFT CFG_CAN_RIGHT < <("$PY" - "$CFG" "$RIGHT_CFG" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    left = yaml.safe_load(stream)["robot"]["channel"]
with open(sys.argv[2], encoding="utf-8") as stream:
    right = yaml.safe_load(stream)["robot"]["channel"]
print(left, right)
PY
)
export YAM_CAN_LEFT="${YAM_CAN_LEFT:-$CFG_CAN_LEFT}"
export YAM_CAN_RIGHT="${YAM_CAN_RIGHT:-$CFG_CAN_RIGHT}"

# A null limit makes I2RT run its own hard-stop calibration while constructing
# the rollout robot. Calibration must be an explicit, supervised setup step.
"$PY" - "$CFG" "$RIGHT_CFG" <<'PY'
import sys

import yaml

from calibrate_grippers import validate_limits

for side, path in zip(("left", "right"), sys.argv[1:], strict=True):
    with open(path, encoding="utf-8") as stream:
        limits = yaml.safe_load(stream)["robot"].get("gripper_limits")
    try:
        validate_limits(limits)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"[preflight] {side} gripper is not calibrated: {exc}. "
            "Run `python calibrate_grippers.py` before a rollout."
        ) from None
PY

# Work on a gitignored copy so the tracked config stays clean; the copy is
# the on-file record of what actually ran.
cp -f "$CFG" "$RUN_CFG"

# --- server selection -------------------------------------------------------
# Default is the local vla-edge server on port 8202. YAM_SERVER overrides one
# invocation. The value is written into the runtime config for the rollout.
SRV_TARGET="${YAM_SERVER:-127.0.0.1:8202}"
SRV="$SRV_TARGET" "$PY" - "$RUN_CFG" <<'EOF'
import json, os, re, sys
path = sys.argv[1]
text = open(path).read()
new, n = re.subn(r'(?m)^(\s*molmoact_server:\s*).*$',
                 lambda m: m.group(1) + json.dumps(os.environ["SRV"]),
                 text)
assert n == 1, f"expected exactly one molmoact_server line in {path}, found {n}"
open(path, "w").write(new)
EOF

# --- preflight -------------------------------------------------------------
SERVER=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open('$RUN_CFG'))['eval']['molmoact_server'])")
PORT=${SERVER##*:}
fail=0
ss -tln 2>/dev/null | grep -q ":${PORT}\b" || { echo "[preflight] no inference server listening on :$PORT ($SERVER)" >&2; fail=1; }
# Refuse RTC when the server does not advertise MolmoAct2 guidance support.
RTC_EFF="${YAM_RTC:-0}"
if [[ "$RTC_EFF" == "1" ]]; then
    curl -sf --max-time 5 "http://${SERVER}/act" | grep -q "MolmoAct2" || {
        echo "[preflight] YAM_RTC=1 but :$PORT does not identify as a MolmoAct2 server. Unset YAM_RTC for this backend." >&2
        fail=1
    }
    if curl -sf --max-time 5 "http://${SERVER}/act" | grep -q '"rtc_available":false'; then
        echo "[preflight] YAM_RTC=1 but the server reports RTC blocked. Restart it without the blocking flag or run with YAM_RTC=0." >&2
        fail=1
    fi
fi
ss -tln 2>/dev/null | grep -q ':5555\b'    || { echo "[preflight] camera server not listening on :5555; run start_camera_server.sh" >&2; fail=1; }
for ifc in "${YAM_CAN_LEFT:-can_left}" "${YAM_CAN_RIGHT:-can_right}"; do
    ip -br link show "$ifc" 2>/dev/null | grep -q ' UP ' || { echo "[preflight] $ifc is not UP" >&2; fail=1; }
done
[[ $fail -eq 0 ]] || exit 1

# --- write the instruction (json.dumps output is a valid YAML scalar) ------
INSTR="$INSTR" "$PY" - "$RUN_CFG" <<'EOF'
import json, os, re, sys
path = sys.argv[1]
text = open(path).read()
new, n = re.subn(r'(?m)^(\s*language_instruction:\s*).*$',
                 lambda m: m.group(1) + json.dumps(os.environ["INSTR"]),
                 text)
assert n == 1, f"expected exactly one language_instruction line in {path}, found {n}"
open(path, "w").write(new)
EOF

# --- record the starting scene (all three cameras) so placement is on file --
SNAP_DIR=yam_eval_runs/scene_snaps/$(date +%Y%m%d_%H%M%S)
mkdir -p "$SNAP_DIR"
SNAP_DIR="$SNAP_DIR" "$PY" - <<'EOF' || echo "[run_task] WARNING: scene snapshot failed" >&2
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath("run_task.sh")))
from camera_client import CameraClient
c = CameraClient(endpoint="tcp://127.0.0.1:5555", request_timeout_ms=2000, max_frame_age_sec=5.0)
for k, v in c.get_obs().items():
    if getattr(v, "ndim", 0) == 3:
        cv2.imwrite(os.path.join(os.environ["SNAP_DIR"], f"{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
EOF
echo "[run_task] scene snaps : $SNAP_DIR"

rm -f "$YAM_STAGE_FILE"
echo "[run_task] instruction : $INSTR"
echo "[run_task] server      : $SERVER   vel cap: $YAM_MAX_JOINT_VEL rad/s   horizon: ${YAM_ACTION_HORIZON:-30}   ema: ${YAM_ACTION_EMA:-1.0}   limiter: ${YAM_LIMITER:-clamp}   async: ${YAM_ASYNC_PLAN:-1}   rtc: ${YAM_RTC:-0}   rtc_trigger: ${YAM_RTC_TRIGGER:-auto}"
echo "[run_task] end cleanly : touch $YAM_STAGE_FILE   (labels the rollout success)"
export YAM_LEFT_CONFIG="$RUN_CFG"
exec bash run_rollout.sh "$@"
