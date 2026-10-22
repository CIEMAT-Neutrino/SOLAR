"""Step 2b/3: per-cluster table of a background sample exactly as 01_fiducialize.py sees it (Truncated, reco positions).

usage: s4_events.py config sample   -> _xchk4/events/<config>_<sample>.pkl
Validation: the weighted histogram rebuilt from the table reproduces the scan pkl Counts/MCCounts at (0,0,0) and
at a few other volumes.
"""
import os, sys, json
B = "/pc/choozdsk01/users/manthey/SOLAR-sb"
sys.path.insert(0, B); os.chdir(B)
from lib import *
from lib.background import surface_quality_mask, background_weight_divisor
from lib.fiducial import accepted_flash_planes

config, name = sys.argv[1], sys.argv[2]
folder = "Truncated"
sample_key = name.split("_")[0].lower()
configs = {config: [name]}
info = json.load(open(f"config/{config}/{config}_config.json"))
run, _ = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=False)
run = compute_reco_workflow(run, configs, params={"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"},
                            rm_branches=False, workflow="SIGNIFICANCE", debug=False)
r = run["Reco"]
div = background_weight_divisor(str(root), folder, name, config, r["SignalParticleK"])
w = np.asarray(r["SignalParticleWeight"], dtype=float) / div
q = (np.asarray(surface_quality_mask(str(root), sample_key, folder, r["SignalParticleSurface"], r, info), bool)
     & accepted_flash_planes(r["MatchedOpFlashPlane"], str(root), True) & (r["MatchedOpFlashPE"] > 0))
keys = ["Event", "SolarEnergy", "RecoX", "RecoY", "RecoZ", "SignalParticleK", "SignalParticleSurface", "MainPDG", "MainK",
        "MainParentPDG", "Generator", "NHits", "Charge", "MatchedOpFlashPE", "TPC"]
out = {"Weight": w[q], "Divisor": np.broadcast_to(div, w.shape)[q], "Index": np.where(q)[0]}
for k in keys:
    if k in r:
        try:
            out[k] = np.asarray(r[k])[q]
        except Exception:
            pass
for k in ["MainVertex", "EndVertex"]:
    if k in r:
        v = np.asarray(r[k])[q]
        if v.ndim == 2 and v.shape[1] >= 3:
            out[f"{k}X"], out[f"{k}Y"], out[f"{k}Z"] = v[:, 0], v[:, 1], v[:, 2]
df = pd.DataFrame(out)
os.makedirs(f"{B}/_xchk4/events", exist_ok=True)
df.to_pickle(f"{B}/_xchk4/events/{config}_{name}.pkl")


# validation against the scan pkl
def in_volume(d, fx, fy, fz):
    dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]; dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    if fx == 0 and fy == 0 and fz == 0:
        return np.ones(len(d), bool)
    x, y, z = d.RecoX.values, d.RecoY.values, d.RecoZ.values
    if config == "hd_1x2x6_lateralAPA":
        mx = np.abs(x) > fx
    elif config == "hd_1x2x6_centralAPA":
        mx = np.abs(x) < dx / 2 - fx
    else:
        mx = x < dx / 2 - fx
    return (mx & (np.abs(y) < dy / 2 - fy) & (z > fz - info["DETECTOR_GAP_Z"])
            & (z < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fz))


def fill_gaps(hist):
    hist = np.asarray(hist, dtype=float).copy()
    right = hist[1:]; left = hist[:-1]
    transfer = right * 0.5 * ((right > 0) & (left == 0))
    hist[:-1] += transfer; hist[1:] -= transfer
    return hist


scan = pd.read_pickle(f"{analysis_info['PATH'] if 'analysis_info' in dir() else load_analysis_info(str(root))['PATH']}/FIDUCIAL/truncated/{config}/{name}/{config}_{name}_SolarEnergy_Fiducial_Scan.pkl")
nb = len(reco_energy_edges) - 1
idx = np.digitize(df.SolarEnergy.values, reco_energy_edges) - 1
ok = (idx >= 0) & (idx < nb)
worst = 0.0
rng = np.random.default_rng(0)
rows = [scan.index[0]] + list(rng.choice(scan.index, 5, replace=False))
for i in rows:
    s = scan.loc[i]
    m = in_volume(df, s.FiducializedX, s.FiducializedY, s.FiducializedZ) & ok
    h = fill_gaps(np.bincount(idx[m], weights=df.Weight.values[m], minlength=nb))
    mc = fill_gaps(np.bincount(idx[m], minlength=nb).astype(float))
    c = np.asarray(s.Counts); mcs = np.asarray(s.MCCounts)
    rel = np.max(np.abs(h - c) / np.maximum(np.abs(c), 1e-30) * (np.abs(c) > 0)) if np.any(c) else 0
    worst = max(worst, rel, float(np.max(np.abs(mc - mcs))))
    print(f"vol {s.FiducializedX}/{s.FiducializedY}/{s.FiducializedZ}: sum counts table {h.sum():.6g} scan {c.sum():.6g}  MC {mc.sum():.0f} vs {mcs.sum():.0f}")
print(f"{config} {name}: {len(df)} quality clusters, worst rel. diff vs scan {worst:.2e}")
