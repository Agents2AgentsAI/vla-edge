#!/usr/bin/env bash
# Start the long-lived ZMQ camera server for the bimanual-YAM stack.
# Owns all 3 RealSense cameras so the eval client can pull obs on demand.
# Leave it running across eval sessions.
#
#   YAM_PYTHON  python interpreter to use (default: python3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/configs/yam_left.yaml}"
PY="${YAM_PYTHON:-python3}"
exec "$PY" "$HERE/camera_server.py" --config "$CONFIG"
