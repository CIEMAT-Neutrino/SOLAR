"""Compare a study profile pkl written by 02_best_fiducial.py (suffix) with the default profile or the offline grid.
usage: s8_cmpprof.py config suffix [offline_option]"""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *
config, suf = sys.argv[1], sys.argv[2]
p = pd.read_pickle(profile_path(config, suffix=suf))
if len(sys.argv) > 3:
    g = pd.read_pickle(f"{B}/_xchk4/out/options_grid_{config}.pkl")
    worst = 0.0
    for _, r in p[p.Band == "Full window"].iterrows():
        ref = [r.ReferenceX, r.ReferenceY, r.ReferenceZ]; ref["XYZ".index(r.Variable)] = r.Margin
        Z = g["store"][(sys.argv[3], r.Analysis)][0][tuple(int(v // 20) for v in ref)]
        worst = max(worst, abs(Z - r.SmoothedSignificance) / max(r.SmoothedSignificance, 1e-12))
    print(f"{config} {suf} vs offline {sys.argv[3]}: worst rel diff {worst:.2e} over {len(p[p.Band=='Full window'])} points")
else:
    d = pd.read_pickle(profile_path(config))
    k = ["Analysis", "Band", "Reference", "Variable", "Margin"]
    m = p.merge(d, on=k, suffixes=("", "_def"))
    print(f"{config} {suf} vs default: {len(m)} rows, max |dZ| {np.max(np.abs(m.SmoothedSignificance - m.SmoothedSignificance_def)):.2e}, "
          f"gate diffs {(m.PassesMCGate != m.PassesMCGate_def).sum()}")
for an in ["DayNight", "HEP", "Sensitivity"]:
    q = p[(p.Analysis == an) & (p.Band == "Full window") & (p.Reference == "Optimum")]
    r0 = q.iloc[0]
    print(f" {an} optimum {r0.ReferenceX}/{r0.ReferenceY}/{r0.ReferenceZ}")
    for v in "XYZ":
        s = q[q.Variable == v].sort_values("Margin")
        print(f"   {v}: " + " ".join(f"{int(a)}:{b:.2f}" for a, b in zip(s.Margin, s.SmoothedSignificance)))
