"""
study_status.py — one table for every config x folder x analysis x study, plus a rerun queue
============================================================================================
Reads the records the pipeline leaves on disk (highest_* maps, Exposure pkls, Sensitivity
Contours pkls) for the nominal run and every STUDY_VARIANTS label, prints a status table and
writes a submission queue for the rows that are stale.

Staleness rules (each row lists the reasons that apply)
  missing          no record on disk for a (config, folder, analysis, label) the registry expects
  epoch            record older than the last statistic change for that analysis (--epoch)
  behind_default   study record older than the default record it is compared against
  values_bug       DayNight/HEP map 'Values' does not match the exposure curve at 30 yr
                   (05_best_sigmas read the 0.1-yr row for held cuts before 2026-09-17)
  cut!=ref         Sensitivity study evaluated at a cut other than the nominal Truncated cut
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
from lib.study import STUDY_VARIANTS
from rich import print as rprint

ANALYSES = ("DayNight", "HEP", "Sensitivity")
LOCAL_DIR = {"DayNight": "day-night", "HEP": "hep", "Sensitivity": "sensitivity"}
DEFAULT_EPOCHS = {
    # DayNight: asymmetry penalty terms removed (daynight_nopenalty_rerun.log)
    "DayNight": "2026-09-07T22:13",
    # Sensitivity: one-expected-event floor removed from the pull fit (see solar_analyses.md 5.3)
    "Sensitivity": "2026-09-17T17:00",
    "HEP": "2026-09-01T00:00",
}


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
            })
    return out


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
    ap.add_argument("--no-behind-default", action="store_true", help="Do not treat 'older than the default record' as stale")
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
        entries += regs
        default_time: Dict[tuple, Optional[dt.datetime]] = {}
        for e in entries:
            label, folder = e["label"], e["folder"]
            sfx = f"_{label}" if label else ""
            sub = label or "default"
            for an in e["analyses"]:
                row = {"config": cfg, "folder": folder, "analysis": an, "study": sub, "group": e["group"],
                       "cut": "-", "value": "-", "at_eval": "-", "record": "-", "reasons": []}
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
                        row["value"] = f"Δχ² {rec['solar']:.2f}/{rec['react']:.2f}@30y"
                        row["at_eval"] = f"Δχ² {c10['solar']:.2f}/{c10['react']:.2f}@10y" if c10 else "-"
                        if rec["fit"] != "pull":
                            row["reasons"].append("fit!=pull")
                        if label and ref_cut and cut != ref_cut:
                            row["reasons"].append(f"cut!=ref{ref_cut}")
                        if label is None and folder == "Truncated":
                            ref_cut = ref_cut or cut
                row["record"] = _fmt(t)
                if t is not None and t < epochs.get(an, dt.datetime.min):
                    row["reasons"].append("epoch")
                if label is None:
                    default_time[(folder, an)] = t
                elif t is not None and not args.no_behind_default:
                    dref = default_time.get(("Truncated", an))
                    if dref and t < dref:
                        row["reasons"].append("behind_default")
                rows.append(row)

    table = pd.DataFrame(rows)
    table["status"] = table["reasons"].apply(lambda r: "ok" if not r else ",".join(r))
    stamp = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{root}/output/logs", exist_ok=True)
    cols = ["config", "folder", "analysis", "study", "cut", "value", "at_eval", "record", "status"]
    md = f"{root}/output/logs/study_status_{stamp}.md"
    with open(md, "w") as fh:
        fh.write(f"# Study status {stamp}\n\nEpochs: " + ", ".join(f"{k}={v.isoformat(timespec='minutes')}" for k, v in epochs.items()) + "\n\n")
        fh.write("cut = NHits/OpHits/AdjCl. DayNight/HEP: smoothed Asimov / profile-likelihood significance; "
                 "Sensitivity: Δχ² solar-fit-at-reactor / reactor-fit-at-solar (full profile unless a nuisance study).\n\n")
        for cfg in configs:
            fh.write(f"\n## {cfg}\n\n| " + " | ".join(cols[1:]) + " |\n|" + "---|" * (len(cols) - 1) + "\n")
            for _, r in table[table.config == cfg].iterrows():
                fh.write("| " + " | ".join(str(r[c]) for c in cols[1:]) + " |\n")
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
            for _, r in sub[sub.study != "default"].iterrows():
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

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000)
    print(table[cols].to_string(index=False))
    rprint(f"\n[cyan][INFO][/cyan] {len(table)} rows, {len(stale)} stale. Table: {md} (+ .csv). Queue: {q}")


if __name__ == "__main__":
    main()
