#!/usr/bin/env bash
# Install vla-edge and the Bimanual YAM robot client on Jetson AGX Thor.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
I2RT_COMMIT="${YAM_I2RT_COMMIT:-47fee5e7dec4e30ca054f798bda1c8894b465ed2}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: activate a virtual environment before running this script" >&2
    echo "  python3 -m venv .venv && source .venv/bin/activate" >&2
    exit 1
fi

PYTHON="$VIRTUAL_ENV/bin/python"
if ! "$PYTHON" -c 'import sys; assert sys.prefix != sys.base_prefix' 2>/dev/null; then
    echo "error: $VIRTUAL_ENV does not contain a working Python environment" >&2
    exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "error: this installer is for Jetson AGX Thor (aarch64)" >&2
    exit 1
fi

echo "[1/6] Updating pip"
"$PYTHON" -m ensurepip --upgrade
"$PYTHON" -m pip install --upgrade pip

echo "[2/6] Installing the Jetson PyTorch build"
"$PYTHON" -m pip install torch==2.10.0 torchvision \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple

echo "[3/6] Installing the Jetson linear algebra runtime"
"$PYTHON" -m pip install nvpl-blas nvpl-lapack nvidia-cudss-cu13
"$PYTHON" "$REPO_ROOT/scripts/jetson_thor_postinstall.py"

echo "[4/6] Installing vla-edge and the robot client"
"$PYTHON" -m pip install -e "${REPO_ROOT}[torch,camera,eval]"
"$PYTHON" -m pip install -r "$HERE/requirements.txt"

echo "[5/6] Installing the I2RT YAM driver"
# ruckig 0.15.3 builds from source. The build-only constraint keeps its old
# CMake configuration compatible without changing the runtime environment.
"$PYTHON" -m pip install \
    --build-constraint "$HERE/build-constraints.txt" \
    "git+https://github.com/i2rt-robotics/i2rt@${I2RT_COMMIT}"

echo "[6/6] Verifying the installation"
"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import torch
import vla_edge
import i2rt
import einops
import pyrealsense2

assert torch.version.cuda, "CPU-only torch: reinstall from the Jetson SBSA index"
assert "sm_110" in torch.cuda.get_arch_list(), "torch does not include Thor sm_110"
print(f"installation OK: torch {torch.__version__}, CUDA {torch.version.cuda}")
PY
