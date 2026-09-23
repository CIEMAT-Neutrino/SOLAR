"""
study_status.py — one table for every config x folder x analysis x study, plus a rerun queue
============================================================================================
Reads the records the pipeline leaves on disk (highest_* maps, Exposure pkls, Sensitivity
Contours pkls) for the nominal run and every STUDY_VARIANTS label, prints a status table and
writes a submission queue for the rows that are stale.

Trend check
  Every study row is compared with a reference row (the default run, or the neighbouring
  variant of a scan) under the relation physics expects, see EXPECTED_TRENDS. The `trend`
  column reads e.g. ">= default 6.69 vs 5.79 OK" and is coloured
    green   expected relation holds
    orange  unexpected, but the study or its reference is stale (known bug / rerun pending):
            re-check after the queue
    red     unexpected on two fresh results: a bug to evaluate (listed at the end)
    grey    informational (estimator swap) or reference missing

Staleness rules (each row lists the reasons that apply)
  missing          no record on disk for a (config, folder, analysis, label) the registry expects
  epoch            record older than the last statistic change for that analysis (--epoch), or older than the
                   last code change that touches that particular study (STUDY_EPOCHS)
  values_bug       DayNight/HEP map 'Values' does not match the exposure curve at 30 yr
                   (05_best_sigmas read the 0.1-yr row for held cuts before 2026-09-17)
  cut!=ref         Sensitivity study that is meant to HOLD the nominal Truncated cut (skip_best_cuts in
                   lib.study) but was evaluated at another one. Studies that scan their own cut by design
                   (energy_*, fiduc_truth*) are never flagged.
  fit!=pull        Sensitivity contours not produced by the pull fit

Usage (inside the container: apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=<repo>/ --pwd <repo>/ <repo>/containers/solar_v1.0.sif python3 ...)
  python3 src/tools/study_status.py                       # all configs, table + queue
  python3 src/tools/study_status.py --per-config          # + one queue per config and a parallel launcher
  python3 src/tools/study_status.py --config hd_1x2x6_centralAPA
  python3 src/tools/study_status.py --epoch Sensitivity=2026-09-17T17:00 --epoch DayNight=2026-09-07T22:13
Outputs: output/logs/study_status_<stamp>.md / .csv and output/logs/study_queue_stale_<stamp>.sh
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib import load_analysis_info, root
from lib.study import OPTIONAL_GROUPS, STUDY_VARIANTS

OPTIONAL_STUDIES = {v["label"] for g in OPTIONAL_GROUPS for v in STUDY_VARIANTS[g]}
from rich import print as rprint

ANALYSES = ("DayNight", "HEP", "Sensitivity")
LOCAL_DIR = {"DayNight": "day-night", "HEP": "hep", "Sensitivity": "sensitivity"}
DEFAULT_EPOCHS = {
    # DayNight/HEP/Sensitivity: adaptive MC-support gate on the cut scan now also covers
    # non-essential backgrounds (radiological), not just essential ones -- see
    # lib.background.adaptive_mc_thresholds and [[sensitivity-one-event-floor]]. A cut chosen
    # before this can be resting on a background estimate backed by 0-1 simulated events.
    "DayNight": "2026-09-18T13:00",
    "Sensitivity": "2026-09-18T13:00",
    "HEP": "2026-09-18T13:00",
}
# Code changes that only touch one study: a record for that study is stale when older than its
# epoch, on top of the analysis-wide epoch above. `ok` therefore means "produced after every change
# that can alter this row", not just "after the last statistic change".
STUDY_EPOCHS = {
    # 01_daynight/01_hep forward --no-membrane_veto and 03_analysis no longer hard-codes plane == 0
    "membrane_veto_off": "2026-09-20T17:38",
    # truth containment / truth-position consistency cut (lib/fiducial.py)
    "fiduc_truth": "2026-09-20T13:01",
    "fiduc_truth_refvol": "2026-09-20T13:01",
    # energy studies rerun with the fiducial volume held at the SolarEnergy reference (lib/study.py, 2026-09-21)
    "energy_spk": "2026-09-21T11:13",
    "energy_maink": "2026-09-21T11:13",
}
# Studies that do not apply to a geometry (label -> config-name prefix): the veto only acts on VD planes;
# bkgmodel is not run on vd_1x8x14_3view_30deg_shielded (2026-09-22, by request; also lib.study excluded_configs).
NOT_APPLICABLE = {
    "membrane_veto_off": ("hd_",),
    "bkgmodel_nominal": ("vd_1x8x14_3view_30deg_shielded",),
    "bkgmodel_reduced": ("vd_1x8x14_3view_30deg_shielded",),
}


# Expected relation of a study to its reference: {label: {analysis or "*": (ref_label, relation)}}.
# relation: ">=" / "<=" (the study should be at least/at most the reference, REL_TOL slack),
# "==" (same within EQ_TOL), None (informational, no expectation). ref_label "default" is the
# nominal run of the same folder; "default:Truncated" pins the Truncated nominal run.
EXPECTED_TRENDS: Dict[str, Dict[str, tuple]] = {
    # signal-normalisation prior: tighter prior -> larger Delta chi2 / significance
    "unc_sig0":  {"*": ("unc_sig2", ">=")},
    "unc_sig2":  {"*": ("default", ">=")},
    "unc_sig6":  {"*": ("default", "<=")},
    "unc_sig20": {"*": ("default", ">=")},
    "unc_sig40": {"*": ("default", "<=")},
    # background prior: non-binding for Sensitivity and the DayNight Asimov metric
    "unc_bkg0":  {"*": ("default", "==")},
    "unc_bkg4":  {"*": ("default", "==")},
    "unc_bkg6":  {"*": ("default", "==")},
    # fewer nuisances -> larger Delta chi2 (full profile = default)
    "nuisance_nominal": {"*": ("nuisance_sin13", ">=")},
    "nuisance_sin13":   {"*": ("default", ">=")},
    "nuisance_escale":  {"*": ("default", ">=")},
    # a different fitter is expected to give different chi2 (that difference is what the study shows, and
    # lib/study.py notes the validation-gate warnings), so there is no equality to enforce: informational
    "legacy_fit":       {"*": ("default", None)},
    # charge threshold scan: informational only. A higher threshold does NOT necessarily lower the
    # significance (removing low-charge background can cost less signal than it removes background),
    # so no trend is imposed; each row is shown against its neighbour without a verdict.
    "charge_Q0":   {"*": ("default", None)},
    "charge_Q50":  {"*": ("charge_Q0", None)},
    "charge_Q100": {"*": ("charge_Q50", None)},
    "charge_Q500": {"*": ("charge_Q100", None)},
    # background normalisation folders vs the Truncated baseline
    "bkgmodel_nominal": {"*": ("default:Truncated", "<=")},
    "bkgmodel_reduced": {"*": ("default:Truncated", ">=")},
    # oscillation point: solar = default inputs; reactor dm2 gives a smaller day-night effect
    "oscpoint_solar":   {"*": ("default", "==")},
    "oscpoint_reactor": {"DayNight": ("default", "<="), "HEP": ("default", "==")},
    # ideal fiducialisation should not lose sensitivity
    "fiduc_truth": {"*": ("default", ">=")},
    # optional diagnostic (lib.study.OPTIONAL_GROUPS): informational, read against fiduc_truth
    "fiduc_truth_refvol": {"*": ("fiduc_truth", None)},
    # different energy estimators: informational only
    "energy_spk": {"*": ("default", None)}, "energy_maink": {"*": ("default", None)},
    "bkg_gamma_cluster": {"*": ("default", None)}, "bkg_gamma_total": {"*": ("default", None)},
    # membrane planes add VD signal; a no-op on HD
    # DayNight/HEP: extra membrane matches should not lose sensitivity. Sensitivity is informational: the
    # cuts are held at the veto-on optimum and the extra events carry poorly reconstructed drift (X), so
    # it can drop (VD shielded 0.34 vs 0.40) without a bug; the study measures the effect at held cuts.
    "membrane_veto_off": {"DayNight": ("default", ">="), "HEP": ("default", ">="), "Sensitivity": ("default", None)},
}
FOLDER_TRENDS = {"Nominal": "<=", "Reduced": ">="}   # folder defaults vs the Truncated default
REL_TOL, EQ_TOL, ABS_TOL = 0.02, 0.05, 0.05


def judge(value: float, ref: float, relation: Optional[str]) -> Optional[bool]:
    """True/False for an expected relation, None when there is no expectation."""
    if relation is None or not (np.isfinite(value) and np.isfinite(ref)):
        return None
    # relative slack with an absolute floor: a Delta chi2 of 0.1 vs 0.2 is noise, not a trend
    scale = max(abs(ref), 1e-9)
    slack_rel, slack_eq = max(REL_TOL * scale, ABS_TOL), max(EQ_TOL * scale, ABS_TOL)
    if relation == ">=":
        return value >= ref - slack_rel
    if relation == "<=":
        return value <= ref + slack_rel
    return abs(value - ref) <= slack_eq


def _mtime(path: str) -> Optional[dt.datetime]:
    return dt.datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def _fmt(t: Optional[dt.datetime]) -> str:
    return t.strftime("%m-%d %H:%M") if t else "-"


def variants() -> List[dict]:
    """Registry rows: label, group, folder, analyses, energy."""
    out = []
    for group, vs in STUDY_VARIANTS.items():
        for v in vs:
            if not v.get("label"):
                continue
            out.append({
                "label": v["label"], "group": group, "folder": (v.get("folder") or "Truncated"),
                "analyses": list(v.get("analysis_override") or ANALYSES),
                "energy": v.get("energy_override") or "SolarEnergy",
                "held_cut": bool(v.get("skip_best_cuts", False)),
            })
    return out


_FIDUCIAL_CACHE: Dict[str, Optional[dict]] = {}


def read_fiducial(root: str, folder: str, cfg: str, an: str, energy: str, label: Optional[str]) -> str:
    """FiducialX/Y/Z (cm) used for (folder, config, analysis, energy) from BestFiducials*.json.

    fiduc_truth is the only variant with its own labeled file (BestFiducials_fiduc_truth.json,
    written with --truth_fiducial); every other variant reuses the folder's plain
    BestFiducials.json, keyed by its own energy (energy_override), see STUDY_VARIANTS.
    """
    stem = "BestFiducials_fiduc_truth" if label == "fiduc_truth" else "BestFiducials"
    path = f"{root}/config/analysis/fiducial/{folder.lower()}/{stem}.json"
    if path not in _FIDUCIAL_CACHE:
        payload = None
        if os.path.exists(path):
            try:
                import json
                payload = json.load(open(path))
            except Exception:
                payload = None
        _FIDUCIAL_CACHE[path] = payload
    payload = _FIDUCIAL_CACHE[path]
    if not payload:
        return "-"
    node = payload.get(cfg, {}).get(an.upper(), {}).get(energy)
    if node is None:
        return "-"
    try:
        return f"{int(node['FiducialX'])}/{int(node['FiducialY'])}/{int(node['FiducialZ'])}"
    except Exception:
        return "-"


def read_map(path: str, an: str):
    """(cut tuple NHits/OpHits/AdjCl, Values, energy) from a DayNight/HEP highest DataFrame."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_pickle(path)
        if df.empty:
            return None
        col = df.columns[0]
        return (
            (int(df.loc["NHits", col]), int(df.loc["OpHits", col]), int(df.loc["AdjCl", col])),
            float(df.loc["Values", col]), col[2],
        )
    except Exception:
        return None


# The reported statistic: the Exposure pkl also carries the other estimators (and the +/-Error bands)
# for the same cut, and taking the maximum over all of them picked the Gaussian row for HEP reactor.
REPORTED_VARIABLE = {"DayNight": "Asimov", "HEP": "ProfileLikelihood"}


def read_exposure(path: str, an: str, cut) -> Dict[str, float]:
    """Smoothed significance at 20 and 30 yr for `cut`, from the Exposure pkl."""
    out = {"S20": np.nan, "S30": np.nan}
    if not os.path.exists(path):
        return out
    try:
        df = pd.read_pickle(path)
    except Exception:
        return out
    sel = df[df["SpectrumType"] == "Smoothed"]
    if "Mode" in sel.columns and an == "HEP":
        sel = sel[sel["Mode"] == "NoRebin"]
    if cut is not None:
        sel = sel[(sel["NHits"] == cut[0]) & (sel["OpHits"] == cut[1]) & (sel["AdjCl"] == cut[2])]
    if "Variable" in sel.columns and (sel["Variable"] == REPORTED_VARIABLE.get(an)).any():
        sel = sel[sel["Variable"] == REPORTED_VARIABLE[an]]
    best = None
    for _, r in sel.iterrows():
        e, s = np.atleast_1d(np.asarray(r["Exposure"], float)), np.atleast_1d(np.asarray(r["Significance"], float))
        if e.size != s.size or e.size < 2:
            continue
        s30 = float(np.interp(30.0, e, s))
        if best is None or s30 > best[1]:
            best = (float(np.interp(20.0, e, s)), s30)
    if best:
        out["S20"], out["S30"] = best
    return out


def read_contours(path: str, info: dict):
    """Delta chi2 at the alternate oscillation point (solar row, reactor row), cut, fit, profile."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_pickle(path)
    except Exception:
        return None
    res = {"solar": np.nan, "react": np.nan, "cut": None, "fit": None, "profile": None}
    for _, r in df.iterrows():
        if r.get("Variable") != "sin12":
            continue
        sig, dm2, vals = np.asarray(r["Significance"], float), np.asarray(r["Dm2"], float), np.asarray(r["Values"], float)
        other = info["REACT_DM2"] if r["Label"] == "solar" else info["SOLAR_DM2"]
        try:
            res[r["Label"]] = float(sig[int(np.argmin(np.abs(dm2 - other))), int(np.argmin(np.abs(vals - info["SIN12"])))])
        except Exception:
            pass
        res["cut"] = (int(r["NHits"]), int(r["OpHits"]), int(r["AdjCl"]))
        res["fit"] = r.get("FitMethod")
        res["profile"] = r.get("NuisanceProfile")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=None)
    ap.add_argument("--signal", default="marley")
    ap.add_argument("--epoch", nargs="*", default=[], help="Analysis=ISO-datetime; records older than this are stale")
    ap.add_argument("--tag", default=None, help="Stamp for the output files (default: now)")
    ap.add_argument("--per-config", action="store_true",
                    help="Write one queue script per config plus a launcher that runs them in parallel")
    args = ap.parse_args()

    info = load_analysis_info(str(root))
    P = info["PATH"]
    configs = args.config or sorted(os.path.basename(p) for p in glob(f"{P}/SENSITIVITY/*") if os.path.isdir(p))
    epochs = {k: dt.datetime.fromisoformat(v) for k, v in DEFAULT_EPOCHS.items()}
    for item in args.epoch:
        k, v = item.split("=", 1)
        epochs[k] = dt.datetime.fromisoformat(v)
    sig = args.signal
    regs = variants()

    rows = []
    for cfg in configs:
        ref_cut = None
        ref_map = f"{P}/SENSITIVITY/{cfg}/{sig}/truncated/{cfg}_{sig}_highest_SENSITIVITY.pkl"
        if os.path.exists(ref_map):
            try:
                m = pickle.load(open(ref_map, "rb"))
                ent = [v for k, v in m.items() if k[0] == cfg and k[1] == sig]
                if len(ent) == 1:
                    ref_cut = (int(ent[0]["NHits"]), int(ent[0]["OpHits"]), int(ent[0]["AdjCl"]))
            except Exception:
                pass
        # default rows for every folder that has any record, then the registry variants
        entries = []
        for folder in ("Truncated", "Nominal", "Reduced"):
            has = any(os.path.exists(f"{P}/{an.upper()}/{folder.lower()}/{cfg}/{sig}") or
                      os.path.isdir(f"{root}/output/data/analysis/{LOCAL_DIR[an]}/{cfg}/{sig}/{folder.lower()}/default")
                      for an in ANALYSES) or folder == "Truncated"
            if has:
                entries.append({"label": None, "group": "default", "folder": folder, "analyses": list(ANALYSES), "energy": "SolarEnergy"})
        entries += [e for e in regs if not any(cfg.startswith(pre) for pre in NOT_APPLICABLE.get(e["label"], ()))]
        for e in entries:
            label, folder = e["label"], e["folder"]
            sfx = f"_{label}" if label else ""
            sub = label or "default"
            for an in e["analyses"]:
                row = {"config": cfg, "folder": folder, "analysis": an, "study": sub, "group": e["group"],
                       "cut": "-", "fiducial": read_fiducial(str(root), folder, cfg, an, e["energy"], label),
                       "value": "-", "at_eval": "-", "record": "-", "reasons": [], "metric": np.nan}
                if an in ("DayNight", "HEP"):
                    mp = f"{P}/{an.upper()}/{folder.lower()}/{cfg}/{sig}/{cfg}_{sig}_highest_{an}{sfx}.pkl"
                    ex = f"{root}/output/data/analysis/{LOCAL_DIR[an]}/{cfg}/{sig}/{folder.lower()}/{sub}/{cfg}_{sig}_{an}_Exposure.pkl"
                    rec = read_map(mp, an)
                    t = _mtime(mp)
                    if rec is None:
                        row["reasons"].append("missing")
                    else:
                        cut, values, energy = rec
                        curve = read_exposure(ex, an, cut)
                        row["cut"] = f"{cut[0]}/{cut[1]}/{cut[2]}"
                        row["metric"] = values
                        row["value"] = f"{values:.2f}σ@30y"
                        row["at_eval"] = f"{curve['S20']:.2f}σ@20y" if np.isfinite(curve["S20"]) else "-"
                        # The 05_best_sigmas bug wrote the 0.1-yr value (factors of 10 off); the
                        # legitimate Smoothed rows differ from each other at the 1e-3 level.
                        if np.isfinite(curve["S30"]) and abs(curve["S30"] - values) > max(0.1, 0.05 * abs(curve["S30"])):
                            row["reasons"].append("values_bug")
                        elif not np.isfinite(curve["S30"]):
                            row["reasons"].append("no_exposure_pkl")
                else:
                    cp = f"{root}/output/data/analysis/sensitivity/{cfg}/{sig}/{folder.lower()}/{sub}/{cfg}_{sig}_Sensitivity_Contours.pkl"
                    c10 = read_contours(cp.replace("_Contours", "_10Y_Contours"), info)
                    rec = read_contours(cp, info)
                    t = _mtime(cp)
                    if rec is None:
                        row["reasons"].append("missing")
                    else:
                        cut = rec["cut"]
                        row["cut"] = f"{cut[0]}/{cut[1]}/{cut[2]}" if cut else "-"
                        row["metric"] = rec["solar"]
                        row["value"] = f"Δχ² {rec['solar']:.2f}/{rec['react']:.2f}@30y"
                        # "value" is the quoted 30-yr number (EVALUATION_EXPOSURE_YEARS for Sensitivity).
                        # at_eval reports the 10-yr SECONDARY pass (10Y_Contours), same convention as the
                        # DayNight/HEP branch above reporting its own secondary read (S20 there, S10 here).
                        # 2026-09-21 briefly duplicated "value" into at_eval on the theory that Sensitivity
                        # is "quoted at 30 yr so at_eval should say 30 yr too" -- reverted 2026-09-23: it
                        # silently dropped the 10-yr column every downstream consumer of this table (and at
                        # least one external checker validating against *_10Y_Contours.pkl) expected there.
                        row["at_eval"] = f"Δχ² {c10['solar']:.2f}/{c10['react']:.2f}@10y" if c10 else "-"
                        if rec["fit"] != "pull" and label != "legacy_fit":   # legacy_fit is the non-pull fit by design
                            row["reasons"].append("fit!=pull")
                        if label and e["held_cut"] and ref_cut and cut != ref_cut:
                            row["reasons"].append(f"cut!=ref{ref_cut}")
                        if label is None and folder == "Truncated":
                            ref_cut = ref_cut or cut
                row["record"] = _fmt(t)
                ep = epochs.get(an, dt.datetime.min)
                if label in STUDY_EPOCHS:
                    ep = max(ep, dt.datetime.fromisoformat(STUDY_EPOCHS[label]))
                if t is not None and t < ep:
                    row["reasons"].append("epoch")
                rows.append(row)

    table = pd.DataFrame(rows)
    table["status"] = table["reasons"].apply(lambda r: "ok" if not r else ",".join(r))

    # ── trend check against the expected reference ──────────────────────────────────
    def _lookup(cfg, folder, an, label):
        if label.startswith("default:"):
            folder = label.split(":", 1)[1]
            label = "default"
        hit = table[(table.config == cfg) & (table.folder == folder) & (table.analysis == an) & (table.study == label)]
        return (float(hit.iloc[0]["metric"]), hit.iloc[0]["status"]) if len(hit) else (np.nan, "missing")

    trend, verdict = [], []
    for _, r in table.iterrows():
        if r.study == "default":
            rel = FOLDER_TRENDS.get(r.folder)
            ref_label = "default:Truncated" if rel else None
        else:
            spec = EXPECTED_TRENDS.get(r.study, {})
            ref_label, rel = spec.get(r.analysis, spec.get("*", (None, None)))
        if ref_label is None:
            trend.append("-"); verdict.append("n/a"); continue
        ref_val, ref_status = _lookup(r.config, r.folder, r.analysis, ref_label)
        ok = judge(r.metric, ref_val, rel)
        ref_name = ref_label.replace("default:Truncated", "Truncated default")
        if not np.isfinite(r.metric) or not np.isfinite(ref_val):
            trend.append(f"{rel or 'vs'} {ref_name}: n/a"); verdict.append("n/a"); continue
        stale_pair = (r.status != "ok" or ref_status != "ok")
        if ok is None:
            trend.append(f"vs {ref_name}: {r.metric:.2f} vs {ref_val:.2f}"); verdict.append("info")
        elif ok:
            trend.append(f"{rel} {ref_name}: {r.metric:.2f} vs {ref_val:.2f} OK"); verdict.append("expected")
        elif stale_pair:
            # unexpected, but one side carries a known bug or still awaits its rerun
            trend.append(f"{rel} {ref_name}: {r.metric:.2f} vs {ref_val:.2f} UNEXPECTED (rerun pending)")
            verdict.append("unexpected_stale")
        else:
            # unexpected on two fresh results: a bug to evaluate
            trend.append(f"{rel} {ref_name}: {r.metric:.2f} vs {ref_val:.2f} UNEXPECTED (fresh)")
            verdict.append("unexpected")
    table["trend"], table["verdict"] = trend, verdict
    stamp = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{root}/output/logs", exist_ok=True)
    cols = ["config", "folder", "analysis", "study", "cut", "fiducial", "value", "at_eval", "record", "status", "trend"]
    badge = {"expected": "🟢", "unexpected": "🔴", "unexpected_stale": "🟠", "info": "⚪", "n/a": "⚪"}
    md = f"{root}/output/logs/study_status_{stamp}.md"
    with open(md, "w") as fh:
        fh.write(f"# Study status {stamp}\n\nEpochs: " + ", ".join(f"{k}={v.isoformat(timespec='minutes')}" for k, v in epochs.items()) + "\n\n")
        fh.write("cut = NHits/OpHits/AdjCl. fiducial = FiducialX/Y/Z (cm) from BestFiducials*.json for that "
                 "config/folder/analysis/energy (- if no record). DayNight/HEP: smoothed Asimov / profile-likelihood significance; "
                 "Sensitivity: Δχ² solar-fit-at-reactor / reactor-fit-at-solar (full profile unless a nuisance study).\n\n"
                 "trend: 🟢 study follows the expected relation to its reference; 🟠 it does not, but one side has a known bug "
                 "or still awaits its rerun (re-check after the queue); 🔴 it does not and both sides are fresh -> a bug to "
                 "evaluate; ⚪ informational or no "
                 f"reference. Relations use {REL_TOL:.0%} slack for >=/<= and {EQ_TOL:.0%} for ==; the reference for a scan is the "
                 "neighbouring variant (unc_sig0 vs unc_sig2, charge_Q100 vs charge_Q50, ...). See EXPECTED_TRENDS in "
                 "src/tools/study_status.py.\n\n")
        n_bad = int((table.verdict == "unexpected").sum()); n_good = int((table.verdict == "expected").sum())
        n_wait = int((table.verdict == "unexpected_stale").sum())
        fh.write(f"Trend summary: 🟢 {n_good} expected, 🟠 {n_wait} unexpected but rerun pending, 🔴 {n_bad} unexpected on fresh results.\n\n")
        fresh_bad = table[table.verdict == "unexpected"]
        if len(fresh_bad):
            fh.write("Rows to evaluate (unexpected, both sides fresh):\n\n")
            for _, r in fresh_bad.iterrows():
                fh.write(f"- {r.config} / {r.folder} / {r.analysis} / {r.study}: {r.trend}\n")
            fh.write("\n")
        for cfg in configs:
            fh.write(f"\n## {cfg}\n\n| " + " | ".join(cols[1:]) + " |\n|" + "---|" * (len(cols) - 1) + "\n")
            for _, r in table[table.config == cfg].iterrows():
                cells = [str(r[c]) for c in cols[1:]]
                cells[-1] = f"{badge[r.verdict]} {r.trend}" if r.trend != "-" else "-"
                if r.verdict in ("unexpected", "unexpected_stale"):
                    cells[-1] = f"**{cells[-1]}**"
                fh.write("| " + " | ".join(cells) + " |\n")
    table[cols].to_csv(md.replace(".md", ".csv"), index=False)

    # ── queue ──────────────────────────────────────────────────────────────────────────
    label_group = {v["label"]: v["group"] for v in regs}
    stale = table[table.status != "ok"]
    # The exact invocation the repo runs under (absolute paths: the launcher may be started from anywhere).
    apptainer = (f"apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home={root}/ --pwd {root}/ "
                 f"{root}/containers/solar_v1.0.sif")

    def _header(script: str, log: str) -> str:
        return ("#!/bin/bash\n# Rerun queue for stale study results — generated by src/tools/study_status.py\n"
                f"# Run inside the analysis container, e.g.:\n#   nohup {apptainer} bash {os.path.relpath(script, str(root))}"
                f" > {os.path.relpath(script, str(root))}.stdout 2>&1 &\n"
                "# Order matters: the nominal Truncated run of each config first (held-cut studies are seeded from its map).\n"
                f"set -u\ncd {root}\nLOG={log}\n"
                # apptainer writes remote.yaml under ~/.apptainer; on AFS homes that chmod fails
                # ("unable to correct the permission on .../.apptainer/remote.yaml"), so point
                # its config dir at local disk before any nested apptainer/python call.
                "export APPTAINER_CONFIGDIR=${APPTAINER_CONFIGDIR:-/pc/choozdsk01/users/manthey/.apptainer}\n"
                "run() { echo \"### $(date '+%F %T') $*\" | tee -a $LOG; \"$@\" 2>&1 | tee -a $LOG; }\n\n")

    q = f"{root}/output/logs/study_queue_stale_{stamp}.sh"
    handles = {}
    if args.per_config:
        launcher = f"{root}/output/logs/study_queue_stale_{stamp}_launch_all.sh"
        with open(launcher, "w") as lh:
            lh.write("#!/bin/bash\n# Launch one queue per config in parallel, each inside the analysis container\n"
                     "# (apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=<repo>/ --pwd <repo>/ <repo>/containers/solar_v1.0.sif).\n"
                     "# Configs write to disjoint PNFS/output trees; the shared JSON records are lock-protected\n"
                     "# (lib.io.merge_and_write_json). Presentations at the end of each run_sensitivity call\n"
                     "# are per folder, not per config: re-run src/tools/run_presentations.py once at the end.\n"
                     f"cd {root}\n"
                     "export APPTAINER_CONFIGDIR=${APPTAINER_CONFIGDIR:-/pc/choozdsk01/users/manthey/.apptainer}\n"
                     "mkdir -p \"$APPTAINER_CONFIGDIR\"\n")
            for cfg in configs:
                if stale[stale.config == cfg].empty:
                    continue
                qc = f"{root}/output/logs/study_queue_stale_{stamp}_{cfg}.sh"
                handles[cfg] = open(qc, "w")
                handles[cfg].write(_header(qc, f"output/logs/study_queue_stale_{stamp}_{cfg}.log"))
                lh.write(f"nohup {apptainer} bash {os.path.relpath(qc, str(root))} > {os.path.relpath(qc, str(root))}.stdout 2>&1 &\n")
            lh.write("wait\necho \"[ALL QUEUES DONE] $(date)\"\n")
        os.chmod(launcher, 0o755)
    with open(q, "w") as fh:
        fh.write(_header(q, f"output/logs/study_queue_stale_{stamp}.log"))
        for cfg in configs:
            sub = stale[stale.config == cfg]
            if sub.empty:
                continue
            fh.write(f"# ───────────── {cfg} ─────────────\n")
            if cfg in handles:
                # per-config script gets the same lines: tee every write for this config
                _fh_all = fh

                class _Tee:
                    def write(self_, text):
                        _fh_all.write(text); handles[cfg].write(text)
                fh = _Tee()
            # defaults, Truncated first
            for folder in ("Truncated", "Nominal", "Reduced"):
                d = sub[(sub.study == "default") & (sub.folder == folder)]
                if d.empty:
                    continue
                ans = list(d.analysis)
                light = set(ans) == {"Sensitivity"}
                extra = " --no-fiducialization --no-rebin" + (" --skip-templates --skip_best_cuts" if light else "")
                fh.write(f"run python3 src/pipelines/run_sensitivity.py --config {cfg} --folder {folder} --analysis {' '.join(ans)}{extra}  "
                         f"# stale: {'; '.join(sorted(set(d.status)))}\n")
            # studies: one run_studies call per (group, variant, analyses)
            seen = set()
            for _, r in sub[(sub.study != "default") & ~sub.study.isin(OPTIONAL_STUDIES)].iterrows():
                key = (r.study,)
                if key in seen:
                    continue
                seen.add(key)
                ans = sorted(set(sub[sub.study == r.study].analysis), key=ANALYSES.index)
                fh.write(f"run python3 src/pipelines/run_studies.py --config {cfg} --study {label_group[r.study]} --variant {r.study} "
                         f"--analysis {' '.join(ans)}  # stale: {'; '.join(sorted(set(sub[sub.study == r.study].status)))}\n")
            fh.write("\n")
            if cfg in handles:
                handles[cfg].write("echo \"[QUEUE DONE] $(date)\" | tee -a $LOG\n")
                handles[cfg].close()
                os.chmod(f"{root}/output/logs/study_queue_stale_{stamp}_{cfg}.sh", 0o755)
                fh = _fh_all
        fh.write("echo \"[QUEUE DONE] $(date)\" | tee -a $LOG\n")
    os.chmod(q, 0o755)
    if args.per_config:
        rprint(f"[cyan][INFO][/cyan] Per-config queues + launcher: {root}/output/logs/study_queue_stale_{stamp}_launch_all.sh")

    from rich.console import Console
    from rich.table import Table as RichTable
    console = Console(width=220)
    colour = {"expected": "green", "unexpected": "bold red", "unexpected_stale": "bold dark_orange", "info": "dim", "n/a": "dim"}
    for cfg in configs:
        rt = RichTable(title=cfg, show_lines=False, pad_edge=False)
        for c in cols[1:]:
            rt.add_column(c, overflow="fold")
        for _, r in table[table.config == cfg].iterrows():
            st = f"[green]{r.status}[/green]" if r.status == "ok" else f"[yellow]{r.status}[/yellow]"
            tr = f"[{colour[r.verdict]}]{r.trend}[/{colour[r.verdict]}]"
            rt.add_row(*[str(r[c]) for c in cols[1:-2]], st, tr)
        console.print(rt)
    n_bad = int((table.verdict == "unexpected").sum()); n_good = int((table.verdict == "expected").sum())
    n_wait = int((table.verdict == "unexpected_stale").sum())
    rprint(f"[cyan][INFO][/cyan] trend: [green]{n_good} expected[/green], [dark_orange]{n_wait} unexpected but rerun pending[/dark_orange], "
           f"[bold red]{n_bad} unexpected on fresh results (evaluate)[/bold red]")
    for _, r in table[table.verdict == "unexpected"].iterrows():
        rprint(f"  [bold red]EVALUATE[/bold red] {r.config} / {r.folder} / {r.analysis} / {r.study}: {r.trend}")
    rprint(f"\n[cyan][INFO][/cyan] {len(table)} rows, {len(stale)} stale. Table: {md} (+ .csv). Queue: {q}")


if __name__ == "__main__":
    main()
