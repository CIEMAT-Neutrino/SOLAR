"""HEP-only energy tuning harness (read-only study, 2026-10-08; tools/hep_working_point_check.sh).

For one config, rebuild a candidate HEP energy per cluster and evaluate the 01_hep.py metric:
raw spectra in 1 MeV bins (hep_rebin), profile-likelihood discovery significance with the 2% global background
normalisation, the 01_hep.py bin mask (summed background MCCounts, 8B included, >= 1), the HEP band fiducial
volumes of BestFiducials.json, at the HEP best cut (and optionally a coarse cut rescan). Absolute scale matched
to the HEP_Counts pkl of a reference run (raw hep in 14-20 MeV at 20 years), then x1.5 to 30 years.

Energy model (all configs use a linear SolarEnergy calibration f(x) = (x - I) / A):
  upper branch f(Energy - UPPER.OFFSET), lower branch f(Energy - LOWER.OFFSET), no offset f(Energy);
  offset branch: offset-RF score >= branch threshold -> upper; S/B: score < t (above the gate t_high) -> no offset.
Candidates: base (as run), classifier off, S/B gate/completeness, HEP-trained S/B RF, offset-RF branch threshold,
hep-weighted offset RF, combinations. Tuning on fold 0 ((Flag + Event) % 2), check on fold 1; each fold x2.
"""
import os, sys, pickle, argparse, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib import *
from lib.sigma import evaluate_profile_likelihood_discovery as pl
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--folder", default="Truncated")
ap.add_argument("--backgrounds", nargs="+", default=["gamma_final", "neutron_final", "radiological_final"])
ap.add_argument("--ref_counts", required=True, help="HEP_Counts pkl of the reference run (absolute scale)")
ap.add_argument("--cut", nargs=3, type=int, default=None, help="NHits OpHits AdjCl (default: best-sigma json)")
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--dump", action="store_true")
ap.add_argument("--mc_filter", type=int, default=2, help="03_analysis.py --mc_filter_threshold")
a = ap.parse_args()

config, folder = a.config, a.folder
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
params = json.load(open(f"{root}/config/{config}/{config}_params.json"))
dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]; dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
fids = json.load(open(f"{root}/config/analysis/fiducial/{folder.lower()}/BestFiducials.json"))
bands = get_best_fiducial_bands(fids, config, "SolarEnergy", "HEP")
cal = f"{root}/config/{config}/marley_final/{config}_calib"
disc = json.load(open(f"{cal}/{config}_discriminant_calibration.json"))
sbj = json.load(open(f"{cal}/{config}_signal_background.json"))
solar = json.load(open(f"{cal}/{config}_marley_final_energy_calibration.json"))["SOLAR"]
assert params["SAMPLE_FIT"]["SolarEnergy"] == "linear"
f_cal = lambda x: (x - solar["INTERSECTION"]) / solar["ENERGY_AMP"]
UP, LO = disc["UPPER"]["OFFSET"], disc["LOWER"]["OFFSET"]
calib_root = load_analysis_info(str(root)).get("DATA_PATHS", {}).get("CALIBRATION")
branch_model = pickle.load(open(f"{calib_root}/{config}/marley_final/models/{config}_marley_final_random_forest_discriminant.pkl", "rb"))
branch_feats = list(branch_model.feature_names_in_)
sb_feats = sbj["FEATURES"]
if a.cut is None:
    c = json.load(open(f"{root}/config/{config}/best-sigma-json/hep/{folder.lower()}/{config}_highest_HEP.json"))[config]["SolarEnergy"]
    a.cut = [int(c["NHits"]), int(c["OpHits"]), int(c["AdjCl"])]
edges = np.asarray(hep_rebin, float); nb = len(edges) - 1
window = (edges[:-1] >= 14) & (edges[1:] <= 30)


def X(df, feats):
    return np.clip(np.nan_to_num(np.column_stack([np.asarray(df[f], float) for f in feats]), nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def load(name):
    configs = {config: [name]}
    run, _ = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]})
    sig = name.startswith("marley")
    p = ({"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"], "PARTICLE_TYPE": "signal",
          "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"} if sig else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"})
    run = compute_reco_workflow(run, configs, params=p, workflow="SIGNIFICANCE", rm_branches=False)
    r = run["Reco"]; key = name.split("_")[0]
    q = (surface_quality_mask(str(root), key, folder, r["SignalParticleSurface"], r, info)
         & accepted_flash_planes(r["MatchedOpFlashPlane"], str(root), True) & (np.asarray(r["MatchedOpFlashPE"]) > 0)
         & (np.asarray(r["Energy"], float) >= 5.0))
    if sig and "Generator" in r:
        q &= np.asarray(r["Generator"]) == 1
    # the band volume follows the candidate energy; keep both band masks to re-assign per candidate
    bandmask = [np.asarray(build_fiducial_spatial_mask(run, config, dx, dy, info, folder, band), dtype=bool)[q] for band in bands]
    cols = set(sb_feats) | set(branch_feats) | {"Energy", "NHits", "AdjClNum", "MatchedOpFlashNHits", "SolarEnergy", "SolarEnergyOffset",
                                                "SolarEnergyNoOffset", "SignalBackgroundScore", "SignalParticleK", "ElectronK", "RecoX", "RecoY", "RecoZ"}
    d = {k: np.asarray(r[k])[q] for k in cols if k in r}
    d["fold"] = ((np.asarray(r["Flag"]).astype(np.int64) + np.asarray(r["Event"]).astype(np.int64)) % 2)[q]
    d["bandmask"] = bandmask
    if sig:
        d["hep"] = np.asarray(r["SignalParticleWeighthepOscMean"], float)[q]; d["8B"] = np.asarray(r["SignalParticleWeightb8OscMean"], float)[q]
    else:
        d["w"] = (np.asarray(r["SignalParticleWeight"], float) / background_weight_divisor(str(root), folder, name, config, r["SignalParticleK"]))[q]
    X_ = np.column_stack([d.get(k, np.zeros(len(d["Energy"]))) for k in ("RecoX", "RecoY", "RecoZ")]).astype(float)
    d["DWall"] = np.minimum.reduce([dx / 2 - np.abs(X_[:, 0]), dy / 2 - np.abs(X_[:, 1]), X_[:, 2] + info["DETECTOR_GAP_Z"], info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - X_[:, 2]])
    d["branch_score"] = branch_model.predict_proba(X(d, branch_feats))[:, 1]
    d["E_up"], d["E_lo"], d["E_no"] = f_cal(d["Energy"] - UP), f_cal(d["Energy"] - LO), f_cal(d["Energy"])
    print(f"{name}: {len(d['Energy'])} clusters (Energy >= 5, quality)", flush=True)
    return key, d


S = dict(load(n) for n in ["marley_final"] + a.backgrounds)
m = S["marley"]
# sanity: rebuilt offset energy vs the as-run SolarEnergyOffset
rebuilt = np.where(m["branch_score"] >= disc["ML_THRESHOLD"], m["E_up"], m["E_lo"])
if "SolarEnergyOffset" in m:
    print(f"sanity: rebuilt offset energy matches SolarEnergyOffset for {np.mean(np.abs(rebuilt - m['SolarEnergyOffset']) < 1e-3):.4f} of marley clusters")

# reference scale from the HEP_Counts pkl
ref = pd.read_pickle(a.ref_counts); ref = ref[(ref.SpectrumType == "Raw") & (ref.Component == "hep")].iloc[0]
ref_hep = float(np.sum(np.asarray(ref.Counts, float)[(np.asarray(ref.Energy, float) >= 14) & (np.asarray(ref.Energy, float) < 20)]))


def cut_mask(d, cut):
    nh, oph, adj = cut
    return (d["NHits"] >= nh) & (d["MatchedOpFlashNHits"] >= oph) & (d["AdjClNum"] < adj)


def spectra(E, cut, fold=None):
    s = np.zeros(nb); b = np.zeros(nb); mc = np.zeros(nb)
    for k, d in S.items():
        e = E[k]; sel = cut_mask(d, cut)
        inb = np.zeros(len(e), dtype=bool)
        for band, bm in zip(bands, d["bandmask"]):
            inb |= bm & (e >= band["energy_min"]) & (e < band["energy_max"])
        sel &= inb & (e >= edges[0]) & (e < edges[-1])
        if fold is not None:
            sel &= d["fold"] == fold
        idx = np.digitize(e[sel], edges) - 1
        sc = 2.0 if fold is not None else 1.0
        n_mc = np.bincount(idx, minlength=nb)
        keep = n_mc >= a.mc_filter  # 03_analysis.py --mc_filter_threshold: per sample, per 1 MeV bin
        if k == "marley":
            s += keep * sc * np.bincount(idx, weights=d["hep"][sel], minlength=nb); b += keep * sc * np.bincount(idx, weights=d["8B"][sel], minlength=nb)
        else:
            b += keep * sc * np.bincount(idx, weights=d["w"][sel], minlength=nb)
        mc += keep * n_mc
    return s, b, mc


def Z(E, cut, fold, scale):
    s, b, mc = spectra(E, cut, fold)
    mask = window & (mc >= 1)
    return float(pl(np.where(mask, scale * s, 0), np.where(mask, scale * b, 0), 0.02)), float(s[edges[:-1] >= 20].sum() / max(s[window].sum(), 1e-30))


def energy(branch_t, sb_score_key, t_low, gate, t_high, mode="offset"):
    out = {}
    for k, d in S.items():
        e_off = np.where(d["branch_score"] >= branch_t, d["E_up"], d["E_lo"])
        if sb_score_key is None:
            out[k] = e_off; continue
        sc = d[sb_score_key]
        thr = np.where(e_off >= gate, t_high, t_low)
        bkg = (sc < thr) & (d["Energy"] >= sbj.get("MIN_ENERGY", 0))
        out[k] = np.where(bkg, d["E_no"] if mode == "offset" else -99.0, e_off)
    return out


base_E = {k: np.where(np.isfinite(d["SolarEnergy"]), d["SolarEnergy"], -99.0) for k, d in S.items()}
s0, _, _ = spectra(base_E, a.cut)
scale = ref_hep / s0[(edges[:-1] >= 14) & (edges[:-1] < 20)].sum() * 1.5
print(f"## {config}: HEP cut {a.cut}, bands {[(b['FiducialX'], b['FiducialY'], b['FiducialZ']) for b in bands]}, scale {scale:.4g}")

if a.dump:
    s_, b_, mc_ = spectra(base_E, a.cut)
    rr = pd.read_pickle(a.ref_counts); rr = rr[rr.SpectrumType == "Raw"]
    E0 = np.asarray(rr.iloc[0].Energy, float)
    comp = {c: np.asarray(rr[rr.Component == c].iloc[0].Counts, float) for c in rr.Component.unique()}
    bref = sum(v for c, v in comp.items() if c != "hep")
    print("bin  hep_h  hep_pkl  bkg_h  bkg_pkl  mc_h")
    for i, e in enumerate(edges[:-1] + 0.5):
        j = np.argmin(np.abs(E0 - e))
        if e >= 12:
            print(f"{e:5.1f} {scale/1.5*s_[i]:8.3g} {comp['hep'][j]:8.3g} {scale/1.5*b_[i]:8.3g} {bref[j]:8.3g} {mc_[i]:5.0f}")
    sys.exit(0)

rows = []


def add(label, E, cut=None):
    cut = cut or a.cut
    z0, _ = Z(E, cut, 0, scale); z1, _ = Z(E, cut, 1, scale); za, f20 = Z(E, cut, None, scale)
    rows.append({"label": label, "cut": tuple(cut), "Z0": z0, "Z1": z1, "Zall": za, "hep>=20": f20})


add("base (as run)", base_E)
add("rebuilt default", energy(disc["ML_THRESHOLD"], "SignalBackgroundScore", sbj["SB_THRESHOLD"], sbj.get("SB_GATE_ENERGY", 99), sbj.get("SB_THRESHOLD_HIGH", sbj["SB_THRESHOLD"])))
add("S/B off", energy(disc["ML_THRESHOLD"], None, 0, 99, 0))

# 1. S/B working point above a gate (thresholds: quantiles of the fold-0 CC scores above the gate)
mf0 = m["fold"] == 0
for gate in (10, 12, 14):
    e_off = np.where(m["branch_score"] >= disc["ML_THRESHOLD"], m["E_up"], m["E_lo"])
    ref_sc = m["SignalBackgroundScore"][mf0 & (e_off >= gate)]
    for comp in (0.80, 0.85, 0.90, 0.95, 0.99):
        t = max(sbj["SB_THRESHOLD"], float(np.quantile(ref_sc, 1 - comp)))
        for mode in ("offset", "cut"):
            add(f"S/B gate{gate} C{comp} {mode}", energy(disc["ML_THRESHOLD"], "SignalBackgroundScore", sbj["SB_THRESHOLD"], gate, t, mode))

# 2. HEP-trained S/B classifier (Energy >= 8, hep-weighted signal, external backgrounds), current inputs (+DWall)
for tag, feats in (("hepSB", sb_feats), ("hepSB+DWall", sb_feats + ["DWall"])):
    parts = [(k, np.where(S[k]["Energy"] >= 8)[0]) for k in S if k != "radiological"]
    Xt = np.vstack([X({f: S[k][f][i] for f in feats}, feats) for k, i in parts])
    lab = np.concatenate([np.full(len(i), int(k == "marley")) for k, i in parts])
    fold = np.concatenate([S[k]["fold"][i] for k, i in parts])
    w = np.concatenate([(S[k]["hep"][i] / S[k]["hep"][i].sum()) if k == "marley" else np.full(len(i), 0.5 / max(len(i), 1)) for k, i in parts])
    w = w / w.mean(); sc = np.full(len(Xt), np.nan)
    for k in (0, 1):
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=20, random_state=a.seed + k, n_jobs=8).fit(Xt[fold == k], lab[fold == k], sample_weight=w[fold == k])
        sc[fold == 1 - k] = rf.predict_proba(Xt[fold == 1 - k])[:, 1]
    off = 0
    for k in S:
        S[k][tag] = np.ones(len(S[k]["Energy"]))  # below 8 MeV / radiological: signal-like (current S/B still applies below the gate)
    for k, i in parts:
        S[k][tag][i] = sc[off:off + len(i)]; off += len(i)
    for k in S:  # radiological: score it with the fold-0 model is not available; keep it signal-like above the gate (conservative: no gain)
        pass
    e_off = np.where(m["branch_score"] >= disc["ML_THRESHOLD"], m["E_up"], m["E_lo"])
    ref_sc = m[tag][mf0 & (e_off >= 12) & (m["Energy"] >= 8)]
    for comp in (0.80, 0.85, 0.90, 0.95):
        t = float(np.quantile(ref_sc, 1 - comp))
        E = {}
        for k, d in S.items():
            eo = np.where(d["branch_score"] >= disc["ML_THRESHOLD"], d["E_up"], d["E_lo"])
            low_bkg = (eo < 12) & (d["SignalBackgroundScore"] < sbj["SB_THRESHOLD"]) & (d["Energy"] >= sbj.get("MIN_ENERGY", 0))
            high_bkg = (eo >= 12) & (d[tag] < t)
            E[k] = np.where(low_bkg | high_bkg, d["E_no"], eo)
        add(f"{tag} gate12 C{comp}", E)

# 3. offset-RF branch threshold, and a hep-weighted offset RF (two folds), with the default S/B
for bt in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    add(f"branch t{bt}", energy(bt, "SignalBackgroundScore", sbj["SB_THRESHOLD"], sbj.get("SB_GATE_ENERGY", 99), sbj.get("SB_THRESHOLD_HIGH", sbj["SB_THRESHOLD"])))
if "ElectronK" not in m:
    print("ElectronK not loaded: hep-weighted offset RF skipped")
upper = None if "ElectronK" not in m else np.asarray(m["ElectronK"] > m["SignalParticleK"] + disc["DISCRIMINANT_THRESHOLD"], dtype=int)
for _ in ([0] if upper is not None else []):
  Xb = X(m, branch_feats); hw = m["hep"] / m["hep"].mean()
  hs = np.full(len(Xb), np.nan)
  for k in (0, 1):
      rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=a.seed + k, n_jobs=8).fit(Xb[m["fold"] == k], upper[m["fold"] == k], sample_weight=hw[m["fold"] == k])
      hs[m["fold"] == 1 - k] = rf.predict_proba(Xb[m["fold"] == 1 - k])[:, 1]
      for kk, d in S.items():
          if kk != "marley":
              d.setdefault("hep_branch", np.zeros(len(d["Energy"])))
              d["hep_branch"][d["fold"] == 1 - k] = rf.predict_proba(X(d, branch_feats)[d["fold"] == 1 - k])[:, 1]
  m["hep_branch"] = hs
  for bt in (0.3, 0.4, 0.5, 0.6):
      E = {}
      for k, d in S.items():
          eo = np.where(d["hep_branch"] >= bt, d["E_up"], d["E_lo"])
          thr = np.where(eo >= sbj.get("SB_GATE_ENERGY", 99), sbj.get("SB_THRESHOLD_HIGH", sbj["SB_THRESHOLD"]), sbj["SB_THRESHOLD"])
          bkg = (d["SignalBackgroundScore"] < thr) & (d["Energy"] >= sbj.get("MIN_ENERGY", 0))
          E[k] = np.where(bkg, d["E_no"], eo)
      add(f"hep-weighted branch RF t{bt}", E)

R = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(R.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# fold-0 choice per family, checked on fold 1; then a coarse cut rescan for the best candidates
R["family"] = R.label.str.replace(r" (C|t)[0-9.]+.*$", "", regex=True)
base = R.iloc[0]
print("\nPICKS (fold-0 best per family; fold-1 check vs base)")
picks = []
for fam, g in R.iloc[3:].groupby("family", sort=False):
    p = g.loc[g.Z0.idxmax()]; picks.append(p.label)
    print(f"  {fam:28s} {p.label:34s} Z0 {p.Z0:.3f} Z1 {p.Z1:.3f} (base {base.Z1:.3f}) all {p.Zall:.3f} (base {base.Zall:.3f}) hep>=20 {p['hep>=20']:.3f}")
print("HEPTUNE DONE")
