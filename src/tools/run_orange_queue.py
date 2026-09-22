"""
run_orange_queue.py -- rerun every stale (orange) study with a resource-aware job pool
======================================================================================
For each config, waits for its wave-4 stage to finish (the Truncated default the held-cut
studies read must be fresh), asks study_status.py which rows are ORANGE (unexpected trend, stale own record), and reruns
them concurrently. Nothing is rerun blindly: fiduc_truth and the energy group optimise their
own cuts, so they are skipped (only "behind_default" is not a reason to rerun them).

Resource policy (so a busy shared host is never pushed over):
  * at most --max-jobs jobs at once overall and --per-config per config;
  * a job starts only when MemAvailable > --min-mem-gb and 1-min load < --max-load;
  * jobs start >= --stagger seconds apart (the heavy data loading happens at start-up);
  * run under a systemd user service with MemoryMax/MemoryHigh, so the pool cannot take
    the machine even if a job misbehaves.
Failed jobs are retried once. Summary: output/logs/orange_queue_<stamp>.log
"""
import argparse, csv, datetime as dt, os, re, subprocess, sys, time
from collections import defaultdict

ROOT = "/pc/choozdsk01/users/manthey/SOLAR"
APP = ["apptainer", "exec", "-B", "/pnfs,/afs,/pc,/cvmfs", f"--home={ROOT}/", f"--pwd={ROOT}/",
       f"{ROOT}/containers/solar_v1.0.sif"]
GROUPS = [(r"^unc_", "unc"), (r"^nuisance_", "nuisance"), (r"^charge_", "charge"),
          (r"^membrane_veto", "membrane_veto"), (r"^legacy_fit$", "legacy_fit"),
          (r"^oscpoint_", "oscpoint"), (r"^bkgmodel_", "bkgmodel"), (r"^bkg_gamma_", "bkg_gamma")]
SELF_OPTIMISED = re.compile(r"^(fiduc_truth|energy_.*)$")
ORDER = ["bkgmodel", "charge", "membrane_veto", "unc", "nuisance", "oscpoint", "bkg_gamma", "legacy_fit"]
ANALYSES = ["DayNight", "HEP", "Sensitivity"]

ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA",
                "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"])
ap.add_argument("--max-jobs", type=int, default=8)
ap.add_argument("--per-config", type=int, default=3)
ap.add_argument("--min-mem-gb", type=float, default=25)
ap.add_argument("--max-load", type=float, default=36)
ap.add_argument("--stagger", type=int, default=60)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG = open(f"{ROOT}/output/logs/orange_queue_{stamp}.log", "a", buffering=1)
def log(m): LOG.write(f"[{dt.datetime.now():%F %T}] {m}\n"); print(m, flush=True)
def mem_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1e6
def group_of(label):
    for pat, g in GROUPS:
        if re.match(pat, label): return g

def jobs_for(cfg):
    tag = f"orange_{cfg}"
    subprocess.run(APP + ["python3", "src/tools/study_status.py", "--config", cfg, "--tag", tag],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(f"{ROOT}/output/logs/study_status_{tag}.csv")))
    def eff(r):   # legacy_fit is fit!=pull by design; that tag alone never means "stale"
        return ",".join(t for t in r["status"].split(",") if not t.startswith("fit!=pull")) or "ok"
    stale = defaultdict(set)
    by_key = defaultdict(list)
    for r in rows: by_key[(r["folder"], r["study"])].append(r)
    for r in rows:
        # orange = unexpected trend AND a stale own record (green/grey epoch-stale rows are not rerun)
        if "rerun pending" not in r["trend"]: continue
        own_stale = eff(r) not in ("ok", "behind_default")
        skip_self = ((r["study"] != "default" and SELF_OPTIMISED.match(r["study"]))
                     or (r["study"] == "default" and r["folder"] == "Truncated")   # wave 4 refreshed it
                     or r["study"] == "legacy_fit")                                # fit!=pull by design
        if own_stale and not skip_self:
            stale[(r["folder"], r["study"])].add(r["analysis"])
        # (a fresh orange row is orange because its REFERENCE is stale: handled just below)
        # an orange row is judged against a reference study: if that one is stale too, rerun it as well,
        # otherwise the row stays orange no matter how often it is rerun
        m = re.match(r"^[<>=]+ ([^:]+):", r["trend"].replace("**", "").split(" ", 1)[-1] if r["trend"][:1] in "🟠🔴🟢⚪" else r["trend"])
        ref = m.group(1) if m else None
        if ref and ref not in ("default", "Truncated default") and not SELF_OPTIMISED.match(ref):
            for rr in by_key.get((r["folder"], ref), []):
                # a reference that is merely "behind_default" ran before the default's cut moved: rerun it too
                if eff(rr) != "ok": stale[(r["folder"], ref)].add(rr["analysis"])
    jobs = []
    for (folder, study), an in stale.items():
        an = [a for a in ANALYSES if a in an]
        if study == "default":
            extra = ["--skip-templates", "--skip_best_cuts"] if an == ["Sensitivity"] else []
            cmd = ["python3", "src/pipelines/run_sensitivity.py", "--config", cfg, "--folder", folder,
                   "--analysis", *an, "--no-fiducialization", "--no-rebin", *extra]
            prio = 0
        else:
            g = group_of(study)
            if g is None: log(f"{cfg}: no group for {study}, skipped"); continue
            cmd = ["python3", "src/pipelines/run_studies.py", "--config", cfg, "--study", g,
                   "--variant", study, "--analysis", *an]
            prio = 1 + ORDER.index(g)
        jobs.append((prio, cfg, f"{folder}/{study}", cmd))
    return sorted(jobs)

pending, running, waiting_cfgs = [], [], list(args.configs)
last_start, done, failed = 0, [], []
log(f"pool: max {args.max_jobs}, per-config {args.per_config}, mem>{args.min_mem_gb}GB, load<{args.max_load}")
while waiting_cfgs or pending or running:
    for cfg in list(waiting_cfgs):
        w = f"{ROOT}/output/logs/study_queue_wave4_20260918_{cfg}.log"
        if os.path.exists(w) and "WAVE4 DONE" in open(w).read():
            js = jobs_for(cfg); waiting_cfgs.remove(cfg)
            log(f"{cfg}: wave4 done -> {len(js)} stale job(s): " + ", ".join(j[2] for j in js))
            pending += [(p, c, n, cmd, 0) for p, c, n, cmd in js]
            pending.sort(key=lambda j: j[0])
    for j in list(running):
        rc = j["proc"].poll()
        if rc is None: continue
        running.remove(j)
        if rc == 0: done.append(j["name"]); log(f"DONE   {j['cfg']} {j['name']} ({(time.time()-j['t0'])/60:.0f} min)")
        elif j["tries"] < 1:
            log(f"FAILED {j['cfg']} {j['name']} rc={rc}; retrying once")
            pending.insert(0, (0, j["cfg"], j["name"], j["cmd"], j["tries"] + 1))
        else: failed.append(f"{j['cfg']} {j['name']}"); log(f"FAILED {j['cfg']} {j['name']} rc={rc}; giving up (see {j['log']})")
    if pending and len(running) < args.max_jobs and time.time() - last_start >= args.stagger:
        load = os.getloadavg()[0]
        for item in pending:
            p, cfg, name, cmd, tries = item
            if sum(r["cfg"] == cfg for r in running) >= args.per_config: continue
            if mem_gb() < args.min_mem_gb or load > args.max_load: break
            pending.remove(item)
            lf = f"{ROOT}/output/logs/orange_{cfg}_{name.replace('/', '_')}_{stamp}.log"
            log(f"START  {cfg} {name}: {' '.join(cmd[1:])}")
            if args.dry_run: done.append(name); break
            proc = subprocess.Popen(APP + cmd, cwd=ROOT, stdout=open(lf, "a"), stderr=subprocess.STDOUT,
                                    start_new_session=True)
            running.append(dict(proc=proc, cfg=cfg, name=name, cmd=cmd, tries=tries, t0=time.time(), log=lf))
            last_start = time.time(); break
    time.sleep(20)
log(f"finished: {len(done)} done, {len(failed)} failed: {failed}")
