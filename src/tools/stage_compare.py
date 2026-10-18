"""Stage SOLAR-sb outputs for the plotting repository under the thesis sample names.

Copies the families the standard plots read (analysis results, solar cutflow/nhits/weighted, TPC
smearing chain, workflow/discrimination, best-sigma-json, the study status report) from a SOLAR
worktree into DST with the *_final sample names replaced by the thesis names (marley, gamma,
neutron, radiological) in paths and inside pickles/json/markdown. Only the default study is
staged, so a plot that needs anything else fails instead of mixing old and new data."""
import os, re, sys, json, pickle, shutil
import numpy as np, pandas as pd

SRC, DST = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "sb_rescan3"
PAT = re.compile(r"(marley|gamma|neutron|radiological)_final(?=$|[^A-Za-z0-9])")
CONFIGS = ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"]


def ren(o):
    if isinstance(o, str):
        return PAT.sub(r"\1", o)
    if isinstance(o, pd.DataFrame):
        o = o.copy()
        o.columns = o.columns.map(ren) if not isinstance(o.columns, pd.MultiIndex) else pd.MultiIndex.from_tuples([ren(t) for t in o.columns], names=o.columns.names)
        o.index = o.index.map(ren) if not isinstance(o.index, pd.MultiIndex) else pd.MultiIndex.from_tuples([ren(t) for t in o.index], names=o.index.names)
        for c in o.columns:
            if o[c].dtype == object:
                o[c] = o[c].map(ren)
        return o
    if isinstance(o, pd.Series):
        o = o.copy(); o.index = o.index.map(ren)
        return o.map(ren) if o.dtype == object else o
    if isinstance(o, dict):
        return {ren(k): ren(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(ren(x) for x in o)
    if isinstance(o, np.ndarray) and o.dtype.kind in "OUS":
        return np.vectorize(ren, otypes=[object])(o).astype(o.dtype) if o.size else o
    return o


n = {"pkl": 0, "json": 0, "other": 0}
pkl_jobs = []


def stage_pkl(job):
    sp, dp = job
    with open(sp, "rb") as fh:
        obj = pickle.load(fh)
    with open(dp, "wb") as fh:
        pickle.dump(ren(obj), fh)
def stage(src_root, rel_dir, keep=lambda p: True):
    base = os.path.join(src_root, rel_dir)
    for dirpath, _, files in os.walk(base):
        rel = os.path.relpath(dirpath, src_root)
        parts = rel.split(os.sep)
        # default study only: skip study-label directories under analysis/<ana>/<cfg>/<name>/<folder>/
        if parts[0] == "analysis" and len(parts) >= 6 and parts[5] != "default":
            continue
        for f in files:
            sp = os.path.join(dirpath, f)
            if not keep(sp) or f.endswith(".lock"):
                continue
            dp = os.path.join(DST, PAT.sub(r"\1", rel), PAT.sub(r"\1", f))
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            if f.endswith(".pkl"):
                pkl_jobs.append((sp, dp))
                n["pkl"] += 1
            elif f.endswith(".json"):
                json.dump(ren(json.load(open(sp))), open(dp, "w"), indent=2); n["json"] += 1
            elif f.endswith((".md", ".csv", ".txt")):
                open(dp, "w").write(ren(open(sp).read())); n["other"] += 1
            else:
                shutil.copy2(sp, dp); n["other"] += 1


final_only = lambda p: bool(PAT.search(p)) or "/samples/" in p
os.makedirs(DST, exist_ok=True)
for fam in ["output/data/analysis", "output/data/solar/cutflow", "output/data/solar/nhits", "output/data/solar/weighted",
            "output/data/TPC/smearing", "output/data/workflow/discrimination"]:
    if os.path.isdir(os.path.join(SRC, fam)):
        stage(SRC, fam, keep=final_only)
for cfg in CONFIGS:
    if os.path.isdir(os.path.join(SRC, "config", cfg, "best-sigma-json")):
        stage(SRC, os.path.join("config", cfg, "best-sigma-json"))
logs = sorted(f for f in os.listdir(os.path.join(SRC, "output/logs")) if re.match(rf"study_status_{TAG}_\d+\.md$", f))
if logs:
    os.makedirs(os.path.join(DST, "output/logs"), exist_ok=True)
    # the sync takes the newest study_status_<timestamp>.md
    open(os.path.join(DST, "output/logs", "study_status_20261005_v99.md"), "w").write(ren(open(os.path.join(SRC, "output/logs", logs[-1])).read()))
# The pickles are renamed in parallel (most of the staging time).
from multiprocessing import Pool
with Pool(16) as pool:
    pool.map(stage_pkl, pkl_jobs, chunksize=4)
print("staged", n, "->", DST)
