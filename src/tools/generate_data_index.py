"""
generate_data_index.py — Regenerate output/data/index.json
===========================================================
Walks output/data/, builds a nested directory tree with file sizes, and
writes output/data/index.json so external repositories can discover what
data files are available without cloning the (ignored) binary content.

The index is tracked by git via the !output/data/index.json .gitignore
exception. Re-run this script whenever files are added or removed.

Each file leaf is annotated with:
  - size_bytes, type       — always present
  - themes                 — list of pipeline theme tags (may be empty)
  - publication_export     — true if this file is intended for
                             LOWE_RECONSTRUCTION_PUBLICATION

Theme rules are loaded from config/analysis/pkl_paths.json (_path_themes).
Longest matching prefix wins; prefixes are matched against the file's path
relative to output/data/ using forward-slash separators.

Run
---
  python3 src/tools/generate_data_index.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "output" / "data"
INDEX_PATH = DATA_ROOT / "index.json"
REGISTRY_PATH = ROOT / "config" / "analysis" / "pkl_paths.json"

# Directories under output/data/ whose files are unconditionally
# publication exports regardless of theme rules.
_PUBLICATION_EXPORT_DIRS = {
    "analysis/day-night",
    "analysis/hep",
    "analysis/sensitivity",
    "event",
    "common",
}


def _load_theme_rules(registry: dict) -> list[tuple[str, list[str]]]:
    """Return (prefix, themes) pairs sorted longest-prefix-first."""
    raw = registry.get("_path_themes", {})
    return sorted(raw.items(), key=lambda kv: len(kv[0]), reverse=True)


def _themes_for(rel_path: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    """Return themes for a file at rel_path (relative to output/data/, / separators)."""
    for prefix, themes in rules:
        if rel_path == prefix or rel_path.startswith(prefix + "/"):
            return themes
    return []


def _is_publication_export(rel_path: str) -> bool:
    for d in _PUBLICATION_EXPORT_DIRS:
        if rel_path == d or rel_path.startswith(d + "/"):
            return True
    return False


def _build_tree(
    directory: Path,
    root: Path,
    data_root: Path,
    rules: list[tuple[str, list[str]]],
) -> dict:
    tree = {}
    for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if entry.name == "index.json":
            continue
        if entry.is_dir():
            subtree = _build_tree(entry, root, data_root, rules)
            if subtree is not None:
                tree[entry.name] = subtree
        elif entry.is_file():
            rel = entry.relative_to(data_root).as_posix()
            # rel includes the filename; strip it to get the directory portion
            rel_dir = str(Path(rel).parent).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = ""
            tree[entry.name] = {
                "size_bytes":        entry.stat().st_size,
                "type":              entry.suffix.lstrip(".") or "file",
                "themes":            _themes_for(rel_dir, rules),
                "publication_export": _is_publication_export(rel_dir),
            }
    return tree


def _collect_publication_exports(tree: dict, prefix: str = "") -> list[str]:
    """Recursively collect relative paths of publication_export=True files."""
    paths = []
    for name, value in tree.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            if "size_bytes" in value:
                if value.get("publication_export"):
                    paths.append(path)
            else:
                paths.extend(_collect_publication_exports(value, path))
    return sorted(paths)


def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} does not exist", file=sys.stderr)
        sys.exit(1)

    registry = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        print(f"WARNING: registry not found at {REGISTRY_PATH}", file=sys.stderr)

    rules = _load_theme_rules(registry)
    tree  = _build_tree(DATA_ROOT, ROOT, DATA_ROOT, rules)

    file_count = sum(
        1
        for p in DATA_ROOT.rglob("*")
        if p.is_file() and p.name != "index.json"
    )

    pub_exports = _collect_publication_exports(tree)

    index = {
        "_generated":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_description": (
            "Nested directory tree of output/data/. "
            "Each leaf is {size_bytes, type, themes, publication_export}. "
            "Regenerate with: python3 src/tools/generate_data_index.py"
        ),
        "_root":     "output/data",
        "_registry": "config/analysis/pkl_paths.json",
        "_themes":   registry.get("_themes", {}),
        "_publication_exports": pub_exports,
        "file_count": file_count,
        "tree": tree,
    }

    INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n")
    pub_count = len(pub_exports)
    print(
        f"Written {file_count} entries → {INDEX_PATH.relative_to(ROOT)}"
        f"  ({pub_count} publication exports)"
    )


if __name__ == "__main__":
    main()
