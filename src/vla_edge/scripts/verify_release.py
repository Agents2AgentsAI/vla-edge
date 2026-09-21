"""Verify every file in a released bundle without importing GPU libraries."""
from __future__ import annotations

import argparse
from pathlib import Path

from vla_edge.backends.abcvla.bundle import read_json, verify_file


def verify(directory: Path) -> int:
    root = directory.resolve(strict=True)
    manifest = read_json(root / "MANIFEST.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported release manifest")
    for name, declaration in manifest["files"].items():
        if declaration.get("path") != name:
            raise ValueError(f"manifest path mismatch: {name}")
        verify_file(root, declaration, name)
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(f"Verified {verify(args.bundle)} bundle files (size and SHA-256).")


if __name__ == "__main__":
    main()
