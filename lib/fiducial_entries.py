"""Per-(config, analysis, energy) fiducial records next to the merged BestFiducials*.json."""
import json
import os
from typing import Any, Dict


# ─── Per-entry fiducial records (2026-10-14) ───────────────────────────────────────────────────────────────────────
# BestFiducials*.json is shared by every config, analysis and energy; any fiducial scan rewrites it, so its file time
# said nothing about one template's inputs and the staleness guard rejected valid templates of other configs. Every
# (config, analysis, energy) entry is therefore also written to its own file, rewritten only when its content changes;
# the guard of lib/template_guards.py depends on that file. The merged BestFiducials*.json stays the file every reader uses.
FIDUCIAL_ENTRY_ANALYSES = ("DAYNIGHT", "HEP", "SENSITIVITY")


def best_fiducials_stem(truth_fiducial: bool = False) -> str:
    """Stem of the BestFiducials*.json a step reads and writes.

    BestFiducials_fiduc_truth for the truth-position studies. Otherwise BestFiducials, unless the environment variable
    SOLAR_FIDUCIAL_STEM is set: run_sensitivity --no-update_best_fiducials (2026-10-16) sets it to BestFiducials_<study
    label>, seeded from the folder's BestFiducials.json, so a run that re-optimises the volumes keeps them to itself and
    the shared BestFiducials.json (the final analysis volumes) is left as it was.
    """
    if truth_fiducial:
        return "BestFiducials_fiduc_truth"
    return os.environ.get("SOLAR_FIDUCIAL_STEM") or "BestFiducials"


def isolated_suffix() -> str:
    """"_<label>" in a run with --no-update_best_fiducials (SOLAR_FIDUCIAL_STEM=BestFiducials_<label>), else "".

    Such a run also keeps its per-sample Fiducial_Scan pkls and event-level exports (Ref arrays, FiducializationMask,
    AnalysisMask) under the label, so the shared default files stay the final analysis ones."""
    stem = os.environ.get("SOLAR_FIDUCIAL_STEM") or ""
    return stem[len("BestFiducials"):] if stem.startswith("BestFiducials_") else ""


def fiducial_entry_path(root: str, folder: str, stem: str, config: str, analysis_key: str, energy: str) -> str:
    return os.path.join(str(root), "config", "analysis", "fiducial", str(folder).lower(), "entries", stem, config,
                        f"{analysis_key}_{energy}.json")


def write_fiducial_entries(root: str, folder: str, stem: str, fiducials: Dict[str, Any], configs=None) -> int:
    """Write the per-entry files of `fiducials` (merged BestFiducials content) whose content changed; returns the count."""
    written = 0
    for config in (configs if configs is not None else list(fiducials)):
        for analysis_key in FIDUCIAL_ENTRY_ANALYSES:
            for energy, entry in (fiducials.get(config, {}).get(analysis_key, {}) or {}).items():
                if not isinstance(entry, dict):
                    continue
                path = fiducial_entry_path(root, folder, stem, config, analysis_key, energy)
                text = json.dumps(entry, indent=2, sort_keys=True)
                if os.path.isfile(path):
                    with open(path) as handle:
                        if handle.read() == text:
                            continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.tmp{os.getpid()}"
                with open(tmp, "w") as handle:
                    handle.write(text)
                os.replace(tmp, path)
                written += 1
    return written
