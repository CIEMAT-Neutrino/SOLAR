"""
cleanup_stale_outputs.py — find (and optionally delete) stale study outputs
==========================================================================
The study registry (lib/study.py STUDY_VARIANTS) is the single source of truth for which
study labels exist and which analyses each one runs. Files on disk that carry a label the
registry no longer knows, or a label under an analysis that variant does not run, are
debris from earlier registry versions: they are never regenerated, but plot scripts,
best-cut readers and the thesis tables can still pick them up.

Categories (all reported; deleted only with --apply):
  dead_label        labelled file/dir whose label is not in the registry, or not valid for
                    the analysis / location it sits in (e.g. oscpoint_* under SENSITIVITY,
                    unc_sig8, charge_Q200, metric_*, energy_* Sensitivity templates)
  unused_energy     Sensitivity template directory for an energy estimator no Sensitivity
                    variant or default uses (e.g. SignalParticleK: the energy group is
                    DayNight-only)
  per_point_grid    per-oscillation-point signal templates (`*_dm2_*` other than the two
                    04_best_cuts.py scan points). --flyweight is the default; these are
                    only read by --no-flyweight runs. Skipped with --keep-grid.
  stale_results     results/<profile>/<unc>/ chi2 grids for cuts that no best-cut map of
                    that folder selects any more
  stale_held_map    highest_SENSITIVITY_<label>.pkl of a held-cut study whose cuts differ
                    from the nominal Truncated map (run_sensitivity.py re-seeds it)
  malformed         doubled-prefix files and directories named after the config

Usage (inside the container: apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=<repo>/ --pwd <repo>/ <repo>/containers/solar_v1.0.sif python3 ...)
-----
  python3 src/tools/cleanup_stale_outputs.py                      # dry run, all configs
  python3 src/tools/cleanup_stale_outputs.py --config hd_1x2x6_centralAPA
  python3 src/tools/cleanup_stale_outputs.py --apply               # delete what the dry run listed
A manifest is always written to output/logs/cleanup_stale_outputs_<timestamp>.txt.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys
from glob import glob
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle

from lib import load_analysis_info, root
from lib.study import STUDY_VARIANTS, study_context
from rich import print as rprint

ANALYSES = ("DAYNIGHT", "HEP", "SENSITIVITY")
ANALYSIS_DIRS = {"DAYNIGHT": "day-night", "HEP": "hep", "SENSITIVITY": "sensitivity"}
ANALYSIS_JSON = {"DAYNIGHT": "daynight-json", "HEP": "hep-json", "SENSITIVITY": "sensitivity-json"}
ENERGIES = ("SignalParticleK", "MainK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy")
FOLDERS = ("truncated", "nominal", "reduced")
LABEL_FAMILY = (
    r"(?:unc_(?:sig|bkg)\d+(?:_nobkgfit)?|charge_Q\d+|metric_(?:raw|smoothed)"
    r"|bkg_gamma(?:_cluster|_total)?|energy_(?:spk|maink)|oscpoint_(?:solar|reactor)"
    r"|nuisance_(?:nominal|sin13|escale)|fiduc_truth|membrane_veto_off"
    r"|bkgmodel_(?:nominal|reduced)|legacy_fit|\d+yr)"
)
LABEL_RE = re.compile(rf"(?:^|_)({LABEL_FAMILY})(?=_|\.|$)")


def _flag(extra: List[str], flag: str, default: bool) -> bool:
    value = default
    for token in extra:
        if token == f"--{flag}":
            value = True
        elif token == f"--no-{flag}":
            value = False
    return value


def _option(extra: List[str], option: str, default=None):
    value = default
    for idx, token in enumerate(extra[:-1]):
        if token == f"--{option}":
            value = extra[idx + 1]
    return value


def registry() -> dict:
    """Valid labels per analysis, plus the labels that own their own Rebin / template files."""
    valid: Dict[str, Set[str]] = {an: set() for an in ANALYSES}
    rebin: Dict[str, Set[str]] = {an: set() for an in ANALYSES}
    energies: Dict[str, Set[str]] = {an: {"SolarEnergy"} for an in ANALYSES}
    held: Set[str] = set()
    fiducial: Set[str] = set()
    for variants in STUDY_VARIANTS.values():
        for v in variants:
            label = v.get("label")
            if not label:
                continue
            extra = list(v.get("extra", []))
            allowed = [a.upper() for a in (v.get("analysis_override") or ["DayNight", "HEP", "Sensitivity"])]
            if v.get("skip_best_cuts"):
                held.add(label)
            if _flag(extra, "truth_fiducial", False):
                fiducial.add(label)
            for an in allowed:
                valid[an].add(label)
                if v.get("energy_override"):
                    energies[an].add(v["energy_override"])
                ctx = study_context(argparse.Namespace(
                    study_label=label,
                    charge_threshold=float(_option(extra, "charge_threshold", 0) or 0),
                    dm2=_option(extra, "dm2"),
                    exposure=None,
                    truth_fiducial=_flag(extra, "truth_fiducial", False),
                    membrane_veto=_flag(extra, "membrane_veto", True),
                    folder=v.get("folder") or "Truncated",
                ), analysis=an.capitalize() if an != "DAYNIGHT" else "DayNight")
                if ctx.template_suffix:
                    rebin[an].add(label)
    return {"valid": valid, "rebin": rebin, "energies": energies, "held": held, "fiducial": fiducial}


def label_of(name: str) -> Optional[str]:
    m = LABEL_RE.search(name)
    return m.group(1) if m else None


class Report:
    def __init__(self):
        self.items: List[Tuple[str, str, str]] = []   # (category, path, reason)

    def add(self, category: str, path: str, reason: str):
        self.items.append((category, path, reason))

    def size(self, path: str) -> int:
        if os.path.isdir(path):
            total = 0
            for dp, _, fns in os.walk(path):
                for fn in fns:
                    try:
                        total += os.path.getsize(os.path.join(dp, fn))
                    except OSError:
                        pass
            return total
        try:
            return os.path.getsize(path)
        except OSError:
            return 0


def scan_config(cfg: str, signal: str, reg: dict, info: dict, rep: Report, keep_grid: bool) -> None:
    path = info["PATH"]
    valid, rebin_labels, energies = reg["valid"], reg["rebin"], reg["energies"]
    solar_dm2, react_dm2 = float(info["SOLAR_DM2"]), float(info["REACT_DM2"])
    scan_points = {f"dm2_{d:.3e}_sin13_{info['SIN13']:.3e}_sin12_{info['SIN12']:.3e}" for d in (solar_dm2, react_dm2)}

    def check_labelled(entry: str, an: str, pool: Set[str], where: str):
        label = label_of(os.path.basename(entry))
        if label is None:
            return False
        if label not in pool:
            why = "label not in registry" if not any(label in s for s in valid.values()) else f"label does not run {an}"
            rep.add("dead_label", entry, f"{where}: {why}")
            return True
        return False

    # ── DAYNIGHT / HEP: maps, results, significance bins, jsons ────────────────────────
    for an in ("DAYNIGHT", "HEP"):
        for folder in FOLDERS:
            for entry in sorted(glob(f"{path}/{an}/{folder}/{cfg}/{signal}/*")):
                check_labelled(entry, an, valid[an], f"{an}/{folder}")

    # ── SENSITIVITY: maps, label dirs, template dirs ───────────────────────────────────
    nominal_map = f"{path}/SENSITIVITY/{cfg}/{signal}/truncated/{cfg}_{signal}_highest_SENSITIVITY.pkl"
    nominal_cuts = None
    if os.path.exists(nominal_map):
        try:
            nm = pickle.load(open(nominal_map, "rb"))
            ent = [v for k, v in nm.items() if k[0] == cfg and k[1] == signal]
            if len(ent) == 1:
                nominal_cuts = tuple(int(ent[0][k]) for k in ("NHits", "AdjCl", "OpHits"))
        except Exception:
            pass
    for folder in FOLDERS:
        base = f"{path}/SENSITIVITY/{cfg}/{signal}/{folder}"
        if not os.path.isdir(base):
            continue
        maps_by_label: Dict[Optional[str], Tuple[int, int, int]] = {}
        for entry in sorted(os.listdir(base)):
            full = f"{base}/{entry}"
            if entry == cfg or entry.startswith(f"{cfg}_{signal}_{cfg}_{signal}"):
                rep.add("malformed", full, "doubled prefix / directory named after the config")
                continue
            label = label_of(entry)
            if os.path.isfile(full):
                if check_labelled(full, "SENSITIVITY", valid["SENSITIVITY"], f"SENSITIVITY/{folder}"):
                    continue
                if "highest_SENSITIVITY" in entry:
                    try:
                        m = pickle.load(open(full, "rb"))
                        ent = [v for k, v in m.items() if k[0] == cfg and k[1] == signal]
                        if len(ent) == 1:
                            cuts = tuple(int(ent[0][k]) for k in ("NHits", "AdjCl", "OpHits"))
                            maps_by_label[label] = cuts
                            if label in reg["held"] and nominal_cuts and cuts != nominal_cuts:
                                rep.add("stale_held_map", full,
                                        f"holds {cuts}, nominal Truncated map is {nominal_cuts}")
                    except Exception:
                        pass
                continue
            # directory: label output dir, or template dir {energy}[_{label}]
            energy = next((e for e in ENERGIES if entry == e or entry.startswith(f"{e}_")), None)
            if energy is None:
                check_labelled(full, "SENSITIVITY", valid["SENSITIVITY"], f"SENSITIVITY/{folder} output dir")
                continue
            if label is not None:
                if label not in rebin_labels["SENSITIVITY"]:
                    why = ("label not in registry" if not any(label in s for s in valid.values())
                           else "variant shares the unlabeled templates (or does not run Sensitivity)")
                    rep.add("dead_label", full, f"SENSITIVITY template dir: {why}")
                    continue
            elif energy not in energies["SENSITIVITY"]:
                rep.add("unused_energy", full, "no Sensitivity variant or default uses this energy")
                continue
        # template-dir contents
        for entry in sorted(os.listdir(base)):
            full = f"{base}/{entry}"
            if not os.path.isdir(full):
                continue
            energy = next((e for e in ENERGIES if entry == e or entry.startswith(f"{e}_")), None)
            if energy is None or any(p == full for _, p, _ in rep.items):
                continue
            label = label_of(entry)
            for fn in os.listdir(full):
                fp = f"{full}/{fn}"
                if fn.startswith(f"{cfg}_{signal}_{cfg}_{signal}"):
                    rep.add("malformed", fp, "doubled prefix")
                elif "_dm2_" in fn and fn.endswith(".pkl") and not keep_grid:
                    if not any(sp in fn for sp in scan_points):
                        rep.add("per_point_grid", fp, "per-point oscillation template (flyweight is the default)")
            # chi2 grids for cuts no map selects
            if label is None:
                keep_cuts = {c for lab, c in maps_by_label.items() if lab is None or lab not in rebin_labels["SENSITIVITY"]}
            else:
                keep_cuts = {c for lab, c in maps_by_label.items() if lab == label}
            if not keep_cuts:
                continue
            for fp in glob(f"{full}/results/*/*/*NHits*_AdjCl*_OpHits*"):
                m = re.search(r"NHits(\d+)_AdjCl(\d+)_OpHits(\d+)", os.path.basename(fp))
                if m and tuple(int(x) for x in m.groups()) not in keep_cuts:
                    rep.add("stale_results", fp, f"cut not selected by any map of {folder} (kept: {sorted(keep_cuts)})")
        # background template dirs
        bbase = f"{path}/SENSITIVITY/{cfg}/background/{folder}"
        if os.path.isdir(bbase):
            for entry in sorted(os.listdir(bbase)):
                full = f"{bbase}/{entry}"
                energy = next((e for e in ENERGIES if entry == e or entry.startswith(f"{e}_")), None)
                label = label_of(entry)
                if energy is None:
                    continue
                if label is not None and label not in rebin_labels["SENSITIVITY"]:
                    rep.add("dead_label", full, "SENSITIVITY background template dir: label has no own templates")
                elif label is None and energy not in energies["SENSITIVITY"]:
                    rep.add("unused_energy", full, "no Sensitivity variant or default uses this energy")

    # ── Rebin pkls (signal + background) and fiducial scans ───────────────────────────
    for kind in ("signal", "background"):
        for folder in FOLDERS:
            for an in ANALYSES:
                for fp in glob(f"{path}/{kind}/{folder}/{an}/{cfg}/*/*_Rebin_*.pkl"):
                    check_labelled(fp, an, rebin_labels[an], f"{kind}/{folder}/{an} Rebin")
    for folder in FOLDERS:
        for fp in glob(f"{path}/FIDUCIAL/{folder}/{cfg}/*/*_Fiducial_Scan_*.pkl"):
            check_labelled(fp, "FIDUCIAL", reg["fiducial"], f"FIDUCIAL/{folder}")

    # ── repo-local records: config JSONs and output mirrors ───────────────────────────
    for an in ANALYSES:
        for fp in glob(f"{root}/config/{cfg}/{ANALYSIS_JSON[an]}/*/*.json") + glob(f"{root}/config/{cfg}/{ANALYSIS_JSON[an]}/*/*/*.json"):
            check_labelled(fp, an, valid[an], f"config/{ANALYSIS_JSON[an]}")
        for fp in glob(f"{root}/config/{cfg}/best-sigma-json/{ANALYSIS_DIRS[an].replace('day-night', 'daynight')}/*/*.json"):
            check_labelled(fp, an, valid[an], "config/best-sigma-json")
        for d in glob(f"{root}/output/data/analysis/{ANALYSIS_DIRS[an]}/{cfg}/{signal}/*/*/") + \
                 glob(f"{root}/output/images/analysis/{ANALYSIS_DIRS[an]}/{cfg}/{signal}/*/*/"):
            entry = os.path.basename(d.rstrip("/"))
            if entry == "default":
                continue
            check_labelled(d.rstrip("/"), an, valid[an], f"output {ANALYSIS_DIRS[an]} dir")
        for fp in glob(f"{root}/output/data/solar/*/{cfg}/{signal}/*/{ANALYSIS_DIRS[an].replace('day-night', 'daynight')}/*_*.pkl"):
            check_labelled(fp, an, valid[an], "output/data/solar")
    for d in glob(f"{root}/output/images/analysis/sensitivity/templates/*/*/"):
        check_labelled(d.rstrip("/"), "SENSITIVITY", valid["SENSITIVITY"], "output/images templates")
    for fp in glob(f"{root}/config/analysis/fiducial/*/BestFiducials_*.json"):
        check_labelled(fp, "FIDUCIAL", reg["fiducial"], "config/analysis/fiducial")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", default=None, help="Detector config(s); default: every config under {PATH}/SENSITIVITY")
    parser.add_argument("--signal", default="marley")
    parser.add_argument("--apply", action="store_true", help="Delete the listed files/dirs (default: dry run)")
    parser.add_argument("--keep-grid", action="store_true", help="Do not list per-point oscillation templates")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to these categories (dead_label unused_energy per_point_grid stale_results stale_held_map malformed)")
    args = parser.parse_args()

    info = load_analysis_info(str(root))
    configs = args.config or sorted(os.path.basename(p) for p in glob(f"{info['PATH']}/SENSITIVITY/*") if os.path.isdir(p))
    reg = registry()
    rprint(f"[cyan][INFO][/cyan] Registry labels per analysis: " + ", ".join(f"{an}={len(v)}" for an, v in reg["valid"].items()))
    rep = Report()
    for cfg in configs:
        scan_config(cfg, args.signal, reg, info, rep, keep_grid=args.keep_grid)
    items = [it for it in rep.items if not args.only or it[0] in args.only]
    # de-duplicate nested paths: drop entries whose parent dir is also listed
    listed_dirs = {p for _, p, _ in items if os.path.isdir(p)}
    items = [it for it in items if not any(it[1] != d and it[1].startswith(d + "/") for d in listed_dirs)]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{root}/output/logs", exist_ok=True)
    manifest = f"{root}/output/logs/cleanup_stale_outputs_{stamp}.txt"
    by_cat: Dict[str, List[Tuple[str, str]]] = {}
    total = 0
    with open(manifest, "w") as fh:
        fh.write(f"# cleanup_stale_outputs {stamp} apply={args.apply} configs={configs}\n")
        for cat, p, why in items:
            sz = rep.size(p)
            total += sz
            by_cat.setdefault(cat, []).append((p, why))
            fh.write(f"{cat}\t{sz}\t{p}\t{why}\n")
    for cat, rows in sorted(by_cat.items()):
        rprint(f"\n[bold]{cat}[/bold]: {len(rows)} item(s)")
        for p, why in rows[:12]:
            rprint(f"  {os.path.relpath(p, str(root)) if p.startswith(str(root)) else p.replace(info['PATH'], '$PATH')}  [dim]{why}[/dim]")
        if len(rows) > 12:
            rprint(f"  ... {len(rows) - 12} more (see manifest)")
    rprint(f"\n[cyan][INFO][/cyan] {len(items)} item(s), {total / 1e6:.1f} MB. Manifest: {manifest}")
    if not args.apply:
        rprint("[cyan][INFO][/cyan] Dry run. Re-run with --apply to delete.")
        return
    removed = 0
    for _, p, _ in items:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
        except OSError as exc:
            rprint(f"[yellow][WARNING][/yellow] Could not remove {p}: {exc}")
    rprint(f"[green][DONE][/green] Removed {removed}/{len(items)} item(s).")


if __name__ == "__main__":
    main()
