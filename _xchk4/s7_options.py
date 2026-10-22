"""Step 5: evaluate fiducial-FoM options offline on the full margin grid (read-only, nothing written outside _xchk4/out).

The background scan is rebuilt from the per-cluster tables (s4_events.py, validated against the scan pkls) so that
per-bin sum w^2 and the heaviest cluster are available; the signal comes from the marley scan pkl. The significance is
lib.fiducial_significance.build_significance_payload per volume, exactly as 02_best_fiducial.py (the base option is
validated against the exported profile pkl and its MC-gate flags).

usage: s7_options.py config [mc_threshold]
"""
import sys, time
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *
from scipy.ndimage import gaussian_filter
from lib.fiducial_significance import aggregate_significance

config = sys.argv[1]
rec = json.load(open(f"{B}/config/analysis/fiducial/truncated/BestFiducials.json"))
thr = float(sys.argv[2]) if len(sys.argv) > 2 else float(rec[config]["DAYNIGHT"]["SolarEnergy"]["MCThreshold"])
info = config_info(config)
dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]; dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
dz = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]
gx, gy, gz = np.arange(0.0, dx / 4, 20), np.arange(0.0, dy / 4, 20), np.arange(0.0, dz / 4, 20)
shape3 = (len(gx), len(gy), len(gz)); nb = len(reco_energy_edges) - 1
E = np.asarray(reco_energy_centers, dtype=float)


def fill_gaps(h):  # 01_fiducialize.py fill_gaps along the energy axis
    h = np.array(h, dtype=float)
    right = h[..., 1:]; left = h[..., :-1]
    transfer = right * 0.5 * ((right > 0) & (left == 0))
    h[..., :-1] += transfer; h[..., 1:] -= transfer
    return h


def limits(d):
    x, y, z = d.RecoX.values, d.RecoY.values, d.RecoZ.values
    if config == "hd_1x2x6_lateralAPA":
        lx = np.abs(x)
    elif config == "hd_1x2x6_centralAPA":
        lx = dx / 2 - np.abs(x)
    else:
        lx = dx / 2 - x
    ly = dy / 2 - np.abs(y)
    lz = np.minimum(z + info["DETECTOR_GAP_Z"], info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - z)
    return lx, ly, lz


def grid_hists(d):
    """W, W2, N, MAXW per (volume, bin): event in volume (i,j,k) iff margin_i < its limit on every axis."""
    lx, ly, lz = limits(d)
    ix = np.searchsorted(gx, lx, side="left"); iy = np.searchsorted(gy, ly, side="left"); iz = np.searchsorted(gz, lz, side="left")
    b = d.Bin.values; w = d.Weight.values
    ok = (b >= 0) & (b < nb)
    ins = ok & (ix > 0) & (iy > 0) & (iz > 0)
    flat = np.ravel_multi_index((ix[ins] - 1, iy[ins] - 1, iz[ins] - 1, b[ins]), shape3 + (nb,))
    size = int(np.prod(shape3)) * nb
    out = {}
    for key, vals in (("W", w[ins]), ("W2", w[ins] ** 2), ("N", np.ones(ins.sum()))):
        a = np.bincount(flat, weights=vals, minlength=size).reshape(shape3 + (nb,))
        a = a[::-1, ::-1, ::-1].cumsum(0).cumsum(1).cumsum(2)[::-1, ::-1, ::-1]
        out[key] = np.ascontiguousarray(a)
    mx = np.zeros(size); np.maximum.at(mx, flat, w[ins]); mx = mx.reshape(shape3 + (nb,))
    for ax in range(3):
        mx = np.flip(np.maximum.accumulate(np.flip(mx, ax), axis=ax), ax)
    out["MAXW"] = np.ascontiguousarray(mx)
    # origin: all quality clusters (no geometric condition), as in 01_fiducialize.py
    out["W"][0, 0, 0] = np.bincount(b[ok], weights=w[ok], minlength=nb)
    out["W2"][0, 0, 0] = np.bincount(b[ok], weights=w[ok] ** 2, minlength=nb)
    out["N"][0, 0, 0] = np.bincount(b[ok], minlength=nb)
    m0 = np.zeros(nb); np.maximum.at(m0, b[ok], w[ok]); out["MAXW"][0, 0, 0] = m0
    return out


# ---- background components from events
t0 = time.time()
raw = {}
for s in BKG:
    raw[COMP[s]] = grid_hists(load_events(config, s))
print(f"grid {shape3} x {nb} bins built in {time.time()-t0:.0f}s", flush=True)

# ---- Counts / Error+ / MCCounts of every component straight from the scan pkls (as 02_best_fiducial.py loads them)
pdf = load_plot_df(config)
SCAN = {}
for comp, g in pdf.groupby("Component"):
    W = np.zeros(shape3 + (nb,)); Er = np.zeros_like(W); M = np.zeros_like(W)
    ii = (g.FiducializedX.values // 20).astype(int); jj = (g.FiducializedY.values // 20).astype(int)
    kk = (g.FiducializedZ.values // 20).astype(int); bb = np.searchsorted(E, g.Energy.values.astype(float) - 1e-9)
    W[ii, jj, kk, bb] = g.Counts.values; Er[ii, jj, kk, bb] = g["Error+"].values; M[ii, jj, kk, bb] = g.MCCounts.values
    SCAN[comp] = {"W": W, "E": Er, "M": M}
SIG = {c: v for c, v in SCAN.items() if c not in raw}
for c in raw:  # the event rebuild must agree with the scan (else the per-cluster quantities below are wrong)
    dW = np.max(np.abs(fill_gaps(raw[c]["W"]) - SCAN[c]["W"]) / np.maximum(SCAN[c]["W"], 1e-300) * (SCAN[c]["W"] > 0))
    dM = np.max(np.abs(fill_gaps(raw[c]["N"]) - SCAN[c]["M"]))
    print(f"[rebuild] {c}: max rel diff counts {dW:.2e}, max abs diff MC {dM:.2e}", flush=True)


def base_bkg(c, h, neff=False):
    """Scan values; with neff the relative error uses (sum w)^2/sum w^2 instead of the MC count (same gap-filling
    as 01_fiducialize.py, whose fill_gaps acts in place on h_raw: a lone cluster split 0.5/0.5 has sqrt(2) rel. error)."""
    out = {k: v.copy() for k, v in SCAN[c].items()}
    if neff:
        Wf, W2f = SCAN[c]["W"], fill_gaps(h["W2"])
        rel = np.where(SCAN[c]["M"] > 0, np.sqrt(np.divide(W2f, Wf ** 2, out=np.zeros_like(Wf), where=Wf > 0) + 0.02 ** 2), 0.0)
        out["E"] = Wf * rel
    out["MAXW"] = fill_gaps(h["MAXW"])
    return out


def build(option):
    o = option
    comps = {c: dict(v) for c, v in SIG.items()}
    for c, h in raw.items():
        comps[c] = base_bkg(c, h, neff="neff" in o)
    if "floorR" in o or "floorAll" in o:
        n = float(o.get("floorR", o.get("floorAll")))
        for c in (["radiological"] if "floorR" in o else list(raw)):
            low = comps[c]["M"] < n
            comps[c]["W"] = np.where(low, 0.0, comps[c]["W"]); comps[c]["E"] = np.where(low, 0.0, comps[c]["E"])
    if "msmooth" in o:
        s = float(o["msmooth"]); sig3 = (s, s, s, 0)
        for c in comps:
            comps[c]["W"] = gaussian_filter(comps[c]["W"], sig3, mode="nearest")
            comps[c]["E"] = np.sqrt(gaussian_filter(comps[c]["E"] ** 2, sig3, mode="nearest"))
            comps[c]["M"] = gaussian_filter(comps[c]["M"], sig3, mode="nearest")
    return comps


def evaluate(comps, an, option):
    acfg, scfg = analysis_setup(an)
    lo, hi = acfg["energy_min"], acfg["energy_max"]
    sc, bc = acfg["signal_components"], acfg["background_components"]
    S = sum(comps[c]["W"] for c in sc); SE = np.sqrt(sum(comps[c]["E"] ** 2 for c in sc)); SM = sum(comps[c]["M"] for c in sc)
    Bk = sum(comps[c]["W"] for c in bc); BE = np.sqrt(sum(comps[c]["E"] ** 2 for c in bc)); BM = sum(comps[c]["M"] for c in bc)
    win = (E >= lo) & (E <= hi)
    drop = np.zeros(shape3 + (nb,), bool)
    if "capdrop" in option:
        mx = np.max(np.stack([comps[c]["MAXW"] for c in bc if "MAXW" in comps[c]]), axis=0)
        drop = np.divide(mx, Bk, out=np.zeros_like(Bk), where=Bk > 0) > float(option["capdrop"])
    ess = sorted(get_essential_background_components(B, list(bc)))
    gate = np.all([comps[c]["M"][..., win].sum(-1) >= thr for c in ess], axis=0) if ess else np.ones(shape3, bool)
    gate[0, 0, 0] = True
    Z = np.zeros(shape3); Zraw = np.zeros(shape3)
    for i, j, k in np.ndindex(shape3):
        m = pd.DataFrame({"Energy": E, "CountsSignal": S[i, j, k], "CountsBackground": Bk[i, j, k],
                          "Error+Signal": SE[i, j, k], "Error+Background": BE[i, j, k],
                          "MCCountsSignal": SM[i, j, k], "MCCountsBackground": BM[i, j, k]})
        p = build_significance_payload(m, acfg["significance_type"], acfg.get("combine_mode", "quadrature"), scfg, 100.0, lo, hi)
        if drop[i, j, k].any():
            Z[i, j, k] = aggregate_significance(np.where(drop[i, j, k], 0.0, p["SmoothedSignificance"]), acfg.get("combine_mode", "quadrature"))
            Zraw[i, j, k] = aggregate_significance(np.where(drop[i, j, k], 0.0, p["RawSignificance"]), acfg.get("combine_mode", "quadrature"))
        else:
            Z[i, j, k] = p["SmoothedTotal"]; Zraw[i, j, k] = p["RawTotal"]
    return Z, Zraw, gate


def smoothness(Z, ref):
    """max step / Z(ref) and number of steps > 10 % of Z(ref) on the three profiles through ref."""
    zr = Z[ref] if Z[ref] > 0 else np.max(Z)
    res = {}
    for ax, name in enumerate("XYZ"):
        sl = list(ref); sl[ax] = slice(None)
        d = np.abs(np.diff(Z[tuple(sl)]))
        res[name] = (float(d.max() / zr) if len(d) else 0.0, int((d > 0.1 * zr).sum()))
    return res


OPTIONS = {"base": {}, "neff": {"neff": 1}, "floorR2": {"floorR": 2}, "floorAll2": {"floorAll": 2},
           "capdrop0.5": {"capdrop": 0.5}, "capdrop0.2": {"capdrop": 0.2}, "msmooth1": {"msmooth": 1.0},
           "floorR2+capdrop0.5": {"floorR": 2, "capdrop": 0.5}, "floorR2+neff": {"floorR": 2, "neff": 1},
           "floorR2+neff+msmooth1": {"floorR": 2, "neff": 1, "msmooth": 1.0}}
if len(sys.argv) > 3:
    OPTIONS = {k: v for k, v in OPTIONS.items() if k in sys.argv[3].split(",") or k == "base"}
prof = pd.read_pickle(profile_path(config))
rows, store = [], {}
for oname, o in OPTIONS.items():
    comps = build(o)
    for an in ["DayNight", "HEP", "Sensitivity"]:
        t0 = time.time()
        Z, Zraw, gate = evaluate(comps, an, o)
        store[(oname, an)] = (Z, Zraw, gate)
        zg = np.where(gate, Z, -np.inf)
        best = np.unravel_index(np.argmax(zg), shape3)
        bvol = tuple(int(v) for v in (gx[best[0]], gy[best[1]], gz[best[2]]))
        if oname == "base":
            p = prof[(prof.Analysis == an) & (prof.Band == "Full window")]
            worst = 0.0; gmis = 0
            for _, r in p.iterrows():
                ref = [r.ReferenceX, r.ReferenceY, r.ReferenceZ]; ref["XYZ".index(r.Variable)] = r.Margin
                idx = tuple(int(round(v / 20)) for v in ref)
                worst = max(worst, abs(Z[idx] - r.SmoothedSignificance) / max(r.SmoothedSignificance, 1e-12),
                            abs(Zraw[idx] - r.RawSignificance) / max(r.RawSignificance, 1e-12))
                if thr == r.MCThreshold:
                    gmis += int(bool(gate[idx]) != bool(r.PassesMCGate))
            print(f"[validate] {config} {an}: {len(p)} profile points, worst rel diff {worst:.2e}, gate mismatches {gmis} (profile thr {p.MCThreshold.iloc[0]}, here {thr:g})", flush=True)
        e = rec[config][an.upper()]["SolarEnergy"]
        rvol = (e["FiducialX"], e["FiducialY"], e["FiducialZ"]); ridx = tuple(int(round(v / 20)) for v in rvol)
        bb = store[("base", an)]
        bbest = np.unravel_index(np.argmax(np.where(bb[2], bb[0], -np.inf)), shape3)
        sm_own = smoothness(Z, best); sm_base = smoothness(Z, bbest)
        rows.append({"Config": CONFIGS[config], "Analysis": an, "Option": oname, "Chosen": "/".join(map(str, bvol)),
                     "Z": float(Z[best]), "ZatBaseVol": float(Z[bbest]), "ZatRecordVol": float(Z[ridx]),
                     "Record": "/".join(map(str, rvol)), "PassingVols": int(gate.sum()),
                     **{f"step{k}": v[0] for k, v in sm_own.items()}, **{f"n10{k}": v[1] for k, v in sm_own.items()},
                     **{f"baseVol_step{k}": v[0] for k, v in sm_base.items()}})
        print(f"{oname:>22s} {an:>11s}: chosen {bvol} Z {Z[best]:.3f} | max step X/Y/Z through own optimum "
              + " ".join(f"{v[0]:.2f}({v[1]})" for v in sm_own.values()) + f" | through base optimum "
              + " ".join(f"{v[0]:.2f}" for v in sm_base.values()) + f"  [{time.time()-t0:.0f}s]", flush=True)
os.makedirs(f"{B}/_xchk4/out", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{B}/_xchk4/out/options_{config}.csv", index=False)
pd.to_pickle({"axes": (gx, gy, gz), "store": store, "threshold": thr}, f"{B}/_xchk4/out/options_grid_{config}.pkl")
