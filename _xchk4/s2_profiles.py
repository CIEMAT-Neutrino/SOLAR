"""Print one-coordinate profiles (DayNight etc.) through the optimum, with raw/smoothed significance and MC."""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *
an = sys.argv[1] if len(sys.argv) > 1 else "DayNight"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for config in CONFIGS:
    prof = pd.read_pickle(profile_path(config))
    p = prof[(prof.Analysis == an) & (prof.Band == "Full window") & (prof.Reference == "Optimum")]
    r0 = p.iloc[0]
    print(f"== {CONFIGS[config]} {an} optimum {r0.ReferenceX}/{r0.ReferenceY}/{r0.ReferenceZ}")
    for v in "XYZ":
        q = p[p.Variable == v].sort_values("Margin")
        print(f" {v}: " + " ".join(f"{int(m)}:{s:.2f}({rs:.2f}){'' if g else '*'}" for m, s, rs, g in zip(q.Margin, q.SmoothedSignificance, q.RawSignificance, q.PassesMCGate)))
