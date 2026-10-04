"""
07_waveform_profiles.py — scintillation profiles from the waveform templates
============================================================================
The peak-aligned waveform templates of 06_waveforms.py are the photon arrival-time profile
convolved with the single-photoelectron (SPE) response of the digitiser, and that response is
not the same in the two detectors:

  VD  WaveformDigitizerSim (standard_daphne): analytic SPE, exp rise 13 ns to the peak at
      28 ns, exp fall 386 ns, 9 ADC/PE, no undershoot.
  HD  OpDetDigitizerDUNE with the DAPHNE2 FBK 2022 testbench SPE template
      (duneopdet config_data/SPE_DAPHNE2_FBK_2022.dat, 512 samples at 16 ns): peak at
      sample 9, ~1 us fall and an undershoot down to -30 % at ~2 us.

So the templates themselves cannot be compared between HD and VD. This step unfolds each
clean, amplitude-normalised template with its detector's SPE response (non-negative least
squares with a small Tikhonov term), giving the photon arrival-time profile, and compares
those. The reconvolution residual is stored as a check that the response is the right one
(the lowest-PE HD templates should be close to the SPE template itself).

Profile features (anchored at the profile maximum, the prompt spike):
  PromptFraction  light within [-2, +4] ticks of the maximum (~112 ns)   (singlet-like)
  LateFraction    light more than 1 us after the maximum
  TauSlow         exponential + constant fit of the profile from 96 ns after the maximum
  Residual        RMS of (reconvolved profile - template), in units of the template peak

Inputs:  output/data/PDS/waveform/<config>/<name>/<config>_<name>_Waveform_Templates.pkl
Outputs: same folder, <config>_<name>_Waveform_Profiles.pkl (profile per category) and
         <config>_<name>_Waveform_Profile_Summary.pkl (features per category).

Run
---
  python3 src/physics/detector/pds/07_waveform_profiles.py --config hd_1x2x6_centralAPA --name marley_waveform
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from scipy.optimize import curve_fit, nnls

data_path = f"{root}/output/data/PDS/waveform"
save_path = f"{root}/output/images/PDS/waveform"

parser = argparse.ArgumentParser(description="Unfold waveform templates into scintillation profiles")
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, default="marley_waveform")
parser.add_argument("--min_n", type=int, default=50, help="Minimum waveforms in a template")
parser.add_argument("--regularisation", type=float, default=0.02, help="Tikhonov weight (template peak = 1)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
config, name = args.config, args.name

TICK = 0.016  # us
SPE_FILE = "/cvmfs/dune.opensciencegrid.org/products/dune/duneopdet/v10_08_01d00/config_data/SPE_DAPHNE2_FBK_2022.dat"


def spe_response(geometry, n):
    """SPE response sampled at 16 ns, peak normalised to 1, n samples."""
    if geometry == "vd":
        t = np.arange(n) * TICK
        peak, front, back = 0.028, 0.013, 0.386  # standard_daphne (duneopdet WaveformDigitizerSim.fcl)
        r = np.where(t < peak, np.exp((t - peak) / front), np.exp(-(t - peak) / back))
    else:
        r = np.loadtxt(SPE_FILE)[:n]
        r = np.concatenate([r, np.zeros(max(0, n - len(r)))])
    return r / r.max()


def unfold(template, response, regularisation):
    n = len(template)
    ok = np.isfinite(template)
    m = np.zeros((n, n))
    for j in range(n):
        m[j:, j] = response[: n - j]
    a = np.vstack([m[ok], regularisation * np.eye(n)])
    b = np.concatenate([template[ok], np.zeros(n)])
    profile, _ = nnls(a, b, maxiter=50 * n)
    reco = m @ profile
    residual = np.sqrt(np.mean((reco[ok] - template[ok]) ** 2))
    return profile, reco, residual


def exp_const(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def profile_features(profile, time):
    """Features anchored at the profile maximum (the prompt, singlet-like spike): the onset of
    an unfolded profile is ill-defined because the regularisation leaves small early bins."""
    total = profile.sum()
    if total <= 0:
        return {}
    imax = int(np.argmax(profile))
    t = time - time[imax]
    feats = {
        "ProfilePeak": time[imax],
        "PromptFraction": profile[max(imax - 2, 0):imax + 5].sum() / total,  # ~112 ns around the maximum
        "LateFraction": profile[t > 1.0].sum() / total,
    }
    sel = t >= 6 * TICK
    try:
        popt, pcov = curve_fit(exp_const, t[sel], profile[sel] / profile.max(), p0=[0.1, 1.0, 0.0],
                               bounds=([0, 0.05, -0.1], [2, 20, 0.5]), maxfev=20000)
        feats.update({"TauSlow": popt[1], "TauSlowError": float(np.sqrt(pcov[1, 1])), "SlowOffset": popt[2]})
    except (RuntimeError, ValueError):
        pass
    return feats


filename = f"{data_path}/{config}/{name}/{config}_{name}_Waveform_Templates.pkl"
if not os.path.isfile(filename):
    sys.exit(f"ERROR: {filename} not found (run 06_waveforms.py first)")
templates = pd.read_pickle(filename)
templates = templates[(templates["Norm"] == "Amplitude") & (templates["Selection"] == "Clean") & (templates["N"] >= args.min_n)]
label_cols = [c for c in ["Source", "Signal", "Generator", "PEBin", "Plane"] if c in templates]

rows, summary = [], []
for _, row in templates.iterrows():
    # Unfold [-0.64, +4.8] us around the peak: beyond that the templates are NaN for VD single
    # windows and dominated by uncorrelated late pulses.
    keep = (np.asarray(row["Time"]) >= -0.64) & (np.asarray(row["Time"]) <= 4.8)
    time = np.asarray(row["Time"])[keep]
    mean = np.asarray(row["Mean"], dtype=float)[keep]
    if np.isfinite(mean).sum() < 50:
        continue
    response = spe_response(row["Geometry"], len(mean))
    profile, reco, residual = unfold(mean, response, args.regularisation)
    labels = {k: row[k] for k in ["Geometry", "Config", "Name", "NInputs"] + label_cols}
    norm = profile.sum() if profile.sum() > 0 else 1
    rows.append({**labels, "N": row["N"], "Time": time, "Profile": profile / norm, "Reconvolved": reco,
                 "Template": mean, "Residual": residual})
    summary.append({**labels, "N": row["N"], "Residual": residual, **profile_features(profile, time)})

profiles, summary = pd.DataFrame(rows), pd.DataFrame(summary)
save_df(profiles, data_path, config, name, filename="Waveform_Profiles", rm=args.rewrite, debug=args.debug)
save_df(summary, data_path, config, name, filename="Waveform_Profile_Summary", rm=args.rewrite, debug=args.debug)

main = summary[(summary["Plane"] == "Total") & (summary.get("Generator", pd.Series("All", index=summary.index)).fillna("All") == "All")]
rprint(main[[c for c in ["Source", "Signal", "PEBin", "N", "PromptFraction", "LateFraction", "TauSlow", "Residual"] if c in main]]
       .round(3).to_string(index=False))

fig = make_subplots(rows=1, cols=1)
sel = profiles[(profiles["Plane"] == "Total") & (profiles["PEBin"] == "All")]
if "Generator" in sel:
    sel = sel[sel["Generator"].fillna("All") == "All"]
for idx, (_, row) in enumerate(sel.iterrows()):
    fig.add_trace(go.Scatter(x=row["Time"], y=row["Profile"], mode="lines", line_shape="hvh",
                             name=f"{row['Source']} {row['Signal']} ({row['N']})",
                             line=dict(color=default[idx % len(default)], width=2)))
fig = format_coustom_plotly(fig, title=f"Unfolded photon profiles - {config} {name}", log=(False, True), legend_title="Category")
fig.update_xaxes(title_text="Time from template peak (us)")
fig.update_yaxes(title_text="Fraction of photons per 16 ns")
save_figure(fig, save_path, config, name, filename="Waveform_Profiles", rm=args.rewrite, debug=args.debug)
