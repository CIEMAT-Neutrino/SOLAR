"""Step 1: reproduce SmoothedSignificance of the exported profiles for single volumes."""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *

BANDS = {"Full window": None}
for config in CONFIGS:
    prof = pd.read_pickle(profile_path(config))
    pdf = load_plot_df(config)
    worst = 0.0; n = 0
    for (an, band), part in prof.groupby(["Analysis", "Band"]):
        acfg, scfg = analysis_setup(an)
        if band == "Full window":
            bc = acfg
        else:
            lo, hi = [float(v) for v in band.replace(" MeV", "").split("-")]
            bc = band_config(acfg, (lo, hi))
        sample = part.sample(min(6, len(part)), random_state=1)
        for _, r in sample.iterrows():
            ref = [r["ReferenceX"], r["ReferenceY"], r["ReferenceZ"]]
            ref["XYZ".index(r["Variable"])] = r["Margin"]
            p = payload(pdf, tuple(int(v) for v in ref), bc, scfg)
            rel = abs(p["SmoothedTotal"] - r["SmoothedSignificance"]) / max(abs(r["SmoothedSignificance"]), 1e-12)
            rraw = abs(p["RawTotal"] - r["RawSignificance"]) / max(abs(r["RawSignificance"]), 1e-12)
            worst = max(worst, rel, rraw); n += 1
    print(f"{config}: {n} points checked, worst relative difference {worst:.2e}; mcthr in profile {prof['MCThreshold'].unique()}")
