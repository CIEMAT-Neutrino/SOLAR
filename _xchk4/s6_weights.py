"""Step 3: per-cluster weight distributions inside each analysis window at the recorded default volume.

Writes _xchk4/out/weights_summary.csv and prints per (config, analysis, sample): number of clusters, weight decades,
window background, N_eff = (sum w)^2/sum w^2 (window and per bin), share of the 1/5/10 heaviest clusters, and per bin the
largest single-cluster share. Weights per year (Truncated). Window by bin centres as in 02_best_fiducial.py.
"""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *
rec = json.load(open(f"{B}/config/analysis/fiducial/truncated/BestFiducials.json"))
os.makedirs(f"{B}/_xchk4/out", exist_ok=True)
rows, binrows = [], []
for config in CONFIGS:
    ev = {s: load_events(config, s) for s in BKG}
    for an in ["DayNight", "HEP", "Sensitivity"]:
        acfg, _ = analysis_setup(an)
        lo, hi = acfg["energy_min"], acfg["energy_max"]
        e = rec[config][an.upper()]["SolarEnergy"]
        vol = (e["FiducialX"], e["FiducialY"], e["FiducialZ"])
        tot = {}
        for s, d in ev.items():
            m = in_volume(d, config, vol) & (d.BinCenter >= lo) & (d.BinCenter <= hi)
            q = d[m]; w = np.sort(q.Weight.values)[::-1]
            sw, sw2 = w.sum(), (w ** 2).sum()
            dec = np.histogram(np.log10(np.clip(w, 1e-12, None)), bins=np.arange(-4, 8))[0] if len(w) else np.zeros(11, int)
            rows.append({"Config": CONFIGS[config], "Analysis": an, "Volume": "/".join(map(str, vol)), "Sample": COMP[s],
                         "N": len(w), "SumW": sw, "Neff": sw ** 2 / sw2 if sw2 > 0 else 0.0,
                         "wmin": w.min() if len(w) else np.nan, "wmed": np.median(w) if len(w) else np.nan, "wmax": w.max() if len(w) else np.nan,
                         "Top1": w[:1].sum() / sw if sw else np.nan, "Top5": w[:5].sum() / sw if sw else np.nan,
                         "Top10": w[:10].sum() / sw if sw else np.nan,
                         "Decades[-4..7)": " ".join(str(int(v)) for v in dec)})
            for b, g in q.groupby("Bin"):
                gw = g.Weight.values
                binrows.append({"Config": CONFIGS[config], "Analysis": an, "Sample": COMP[s], "E": float(g.BinCenter.iloc[0]),
                                "N": len(gw), "SumW": gw.sum(), "Neff": gw.sum() ** 2 / (gw ** 2).sum(), "MaxShare": gw.max() / gw.sum()})
out = pd.DataFrame(rows); bins = pd.DataFrame(binrows)
# per bin: share of the total background (all samples) carried by the heaviest single cluster
bt = bins.groupby(["Config", "Analysis", "E"]).SumW.sum().rename("BinTotal")
bins = bins.join(bt, on=["Config", "Analysis", "E"])
bins["SampleShareOfBin"] = bins.SumW / bins.BinTotal
out.to_csv(f"{B}/_xchk4/out/weights_summary.csv", index=False); bins.to_csv(f"{B}/_xchk4/out/weights_per_bin.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print()
piv = bins.pivot_table(index=["Config", "Analysis", "E"], columns="Sample", values=["Neff", "SampleShareOfBin"])
print(piv.to_string(float_format=lambda v: f"{v:.3g}"))
