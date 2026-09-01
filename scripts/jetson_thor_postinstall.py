#!/usr/bin/env python3
"""Wire Jetson PyTorch libraries and system TensorRT into a virtual environment.

Why this exists: the jetson-ai-lab sbsa/cu130 torch wheels link libnvpl_* (ARM
BLAS/LAPACK) and libcudss. pip installs those into site-packages subdirectories
that the dynamic loader does not search, so `import torch` fails with
    ImportError: libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file
even though the packages are installed.

A .pth-loaded module dlopens them with RTLD_GLOBAL at interpreter startup, which
works for ANY invocation of the venv's python -- unlike LD_LIBRARY_PATH, which
is lost when a script runs `.venv/bin/python` directly.

Run after creating or rebuilding the venv:
    python scripts/jetson_postinstall.py [--python /path/to/.venv/bin/python]
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

SHIM = '''\
"""Preload NVPL (ARM BLAS/LAPACK) and cuDSS for the Jetson/SBSA torch build."""
import ctypes
import glob
import os
import sys


def _preload():
    for sp in [p for p in sys.path if p.endswith("site-packages")]:
        for rel, pats in (
            (("nvpl", "lib"), ("libnvpl_blas_lp64_gomp.so*", "libnvpl_lapack_lp64_gomp.so*")),
            (("nvidia", "cu13", "lib"), ("libcudss.so*",)),
        ):
            d = os.path.join(sp, *rel)
            if not os.path.isdir(d):
                continue
            for pat in pats:                      # order matters: blas before lapack
                for f in sorted(glob.glob(os.path.join(d, pat))):
                    try:
                        ctypes.CDLL(f, ctypes.RTLD_GLOBAL)
                        break
                    except OSError:
                        pass


_preload()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter of the venv to patch (default: current)",
    )
    args = ap.parse_args()

    if platform.machine() != "aarch64":
        print(f"not aarch64 ({platform.machine()}) -- shim not needed, nothing done")
        return 0

    sp = subprocess.run(
        [
            args.python,
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    site = Path(sp)
    (site / "_nvpl_preload.py").write_text(SHIM)
    (site / "zz_nvpl_preload.pth").write_text("import _nvpl_preload\n")
    print(f"wrote {site}/_nvpl_preload.py + zz_nvpl_preload.pth")
    wire_system_tensorrt(args.python)

    chk = subprocess.run(
        [
            args.python,
            "-c",
            (
                "import torch;print('torch',torch.__version__,'cuda',"
                "torch.version.cuda,'available',torch.cuda.is_available(),"
                "torch.cuda.get_arch_list())"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print((chk.stdout or chk.stderr).strip())
    if "cuda None" in chk.stdout:
        print("WARNING: this is a CPU-ONLY torch. Reinstall from the sbsa/cu130 index.")
        return 1
    return chk.returncode


def wire_system_tensorrt(python: str = sys.executable) -> None:
    """Expose JetPack's system TensorRT python bindings to this venv.

    On Jetson, `import tensorrt` comes from the OS package in
    /usr/lib/pythonX.Y/dist-packages (there is no usable pip wheel for the
    Thor JetPack stack), which a clean venv cannot see. A .pth file makes
    exactly that one directory visible without --system-site-packages.
    """
    version = subprocess.run(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    system_dir = Path(f"/usr/lib/python{version}/dist-packages")
    if not (system_dir / "tensorrt").is_dir():
        print(
            "system TensorRT not found under",
            system_dir,
            "- install JetPack's python3-libnvinfer packages",
        )
        return
    site_dir = Path(
        subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    pth = site_dir / "zz_system_tensorrt.pth"
    pth.write_text(str(system_dir) + "\n")
    print(f"wrote {pth} -> {system_dir} (system TensorRT visible)")


if __name__ == "__main__":
    raise SystemExit(main())
