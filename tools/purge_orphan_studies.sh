#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# purge_orphan_studies.sh — quarantine study artifacts that are no longer in
# lib/study.py STUDY_VARIANTS, plus wrong-energy and pre-refactor leftovers.
#
# Default action is a QUARANTINE MOVE, not a delete: files are relocated under
# a timestamped root that mirrors their original tree, so the whole operation
# is undone with a single mv. Deletion is a separate, explicit flag.
#
# Usage:
#   tools/purge_orphan_studies.sh                      # dry run (default)
#   tools/purge_orphan_studies.sh --execute            # perform the quarantine move
#   tools/purge_orphan_studies.sh --execute --delete   # permanently delete instead
#   tools/purge_orphan_studies.sh --manifest <file>    # use a specific manifest
#   tools/purge_orphan_studies.sh --undo <quarantine>  # move everything back
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST=""
EXECUTE=0
DELETE=0
UNDO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)  EXECUTE=1; shift ;;
    --delete)   DELETE=1;  shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --undo)     UNDO="$2";  shift 2 ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- undo mode -------------------------------------------------------------
if [[ -n "$UNDO" ]]; then
  [[ -d "$UNDO" ]] || { echo "no such quarantine dir: $UNDO" >&2; exit 1; }
  echo "Restoring everything under $UNDO to its original location..."
  find "$UNDO" -type f | while read -r f; do
    dest="/${f#"$UNDO"/}"
    mkdir -p "$(dirname "$dest")"
    mv -n "$f" "$dest"
  done
  echo "Restore complete. Remove the empty tree with: rm -rf '$UNDO'"
  exit 0
fi

# ---- pick the newest manifest if none given --------------------------------
if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(ls -t "$REPO"/output/logs/orphan_purge_manifest_*.txt 2>/dev/null | head -1 || true)"
fi
[[ -n "$MANIFEST" && -f "$MANIFEST" ]] || {
  echo "No manifest found. Expected output/logs/orphan_purge_manifest_*.txt" >&2
  echo "Regenerate it, or pass --manifest <file>." >&2; exit 1; }

QROOT="/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/_quarantine_$(date +%Y%m%d_%H%M%S)"

echo "manifest : $MANIFEST"
if [[ $DELETE -eq 1 ]]; then
  echo "mode     : PERMANENT DELETE"
else
  echo "mode     : quarantine move -> $QROOT"
fi
[[ $EXECUTE -eq 1 ]] || echo "run      : DRY RUN (pass --execute to act)"
echo

# manifest lines look like:  "MM-DD HH:MM   <size>  /abs/path"
mapfile -t PATHS < <(awk '$0 !~ /^#/ && $NF ~ /^\// {print $NF}' "$MANIFEST")

total=0; missing=0; acted=0; bytes=0
for p in "${PATHS[@]}"; do
  total=$((total+1))
  if [[ ! -e "$p" ]]; then missing=$((missing+1)); continue; fi
  if [[ -d "$p" ]]; then sz=$(du -sb "$p" 2>/dev/null | cut -f1); else sz=$(stat -c%s "$p" 2>/dev/null || echo 0); fi
  sz=${sz:-0}; bytes=$((bytes+sz))
  if [[ $EXECUTE -eq 1 ]]; then
    if [[ $DELETE -eq 1 ]]; then
      rm -rf -- "$p"
    else
      dest="$QROOT/${p#/}"
      mkdir -p "$(dirname "$dest")"
      mv -- "$p" "$dest"
    fi
    acted=$((acted+1))
  else
    echo "  would handle: $p"
  fi
done

echo
echo "listed in manifest : $total"
echo "already gone       : $missing"
echo "present            : $((total-missing))  ($(awk -v b=$bytes 'BEGIN{printf "%.2f", b/1e9}') GB)"
if [[ $EXECUTE -eq 1 ]]; then
  echo "acted on           : $acted"
  if [[ $DELETE -eq 0 ]]; then
    echo
    echo "Quarantine root: $QROOT"
    echo "Undo with      : tools/purge_orphan_studies.sh --undo '$QROOT'"
    echo "Make permanent : rm -rf '$QROOT'"
  fi
else
  echo "acted on           : 0 (dry run)"
fi
