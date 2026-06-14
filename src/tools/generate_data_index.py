"""
generate_data_index.py — Regenerate output/data/index.json
===========================================================
Walks output/data/, builds a nested directory tree with file sizes, and
writes output/data/index.json so external repositories can discover what
data files are available without cloning the (ignored) binary content.

The index is tracked by git via the !output/data/index.json .gitignore
exception. Re-run this script whenever files are added or removed.

Run
---
  python3 src/tools/generate_data_index.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "output" / "data"
INDEX_PATH = DATA_ROOT / "index.json"


def _build_tree(directory: Path, root: Path) -> dict:
    tree = {}
    for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if entry.name == "index.json":
            continue
        if entry.is_dir():
            subtree = _build_tree(entry, root)
            if subtree is not None:
                tree[entry.name] = subtree
        elif entry.is_file():
            tree[entry.name] = {
                "size_bytes": entry.stat().st_size,
                "type": entry.suffix.lstrip(".") or "file",
            }
    return tree


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} does not exist", file=sys.stderr)
        sys.exit(1)

    tree = _build_tree(DATA_ROOT, ROOT)
    file_count = sum(
        1
        for p in DATA_ROOT.rglob("*")
        if p.is_file() and p.name != "index.json"
    )

    index = {
        "_generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_description": (
            "Nested directory tree of output/data/. "
            "Each leaf is {size_bytes, type}. "
            "Regenerate with: python3 src/tools/generate_data_index.py"
        ),
        "_root": "output/data",
        "_registry": "config/analysis/pkl_paths.json",
        "file_count": file_count,
        "tree": tree,
    }

    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Written {file_count} entries → {INDEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
