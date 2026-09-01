"""Regenerate MANIFEST.json for a prebuilt-artifact bundle.

Usage: python scripts/make_manifest.py <bundle_dir>

Rehashes every file in the bundle (except the manifest itself and top-level
docs) and rewrites the `files` map in place, preserving all other manifest
fields. Run on the machine the artifacts were built on.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SKIP = {"MANIFEST.json", "README.md", "LICENSE", "NOTICE"}
SKIP_DIRS = {"__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    bundle = Path(sys.argv[1]).resolve()
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(bundle).parts):
            continue
        rel = path.relative_to(bundle).as_posix()
        if rel in SKIP:
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"{rel}: {files[rel]['bytes']} bytes")
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {manifest_path} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
