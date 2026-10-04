"""
export_waveform_repo.py — bundle the PDS waveform characterisation for the plot/thesis repos
============================================================================================
Collects the per-sample pickles of src/physics/detector/pds/06_waveforms.py and
07_waveform_profiles.py (output/data/PDS/waveform/<config>/<name>/) into a few small files,
one row per (detector sample, category), plus the SPE responses of both digitisers, a README
and meta.json. Nothing is recomputed.

  output/data/PDS/waveform/export/
    waveform_templates.pkl         average peak-aligned waveforms (Clean; Amplitude and Integral norm)
    waveform_features.pkl          per-waveform feature histograms (PeakTime, T50/T90/T99, FWHM, ...)
    waveform_summary.pkl           medians / 16-84 % quantiles / fractions per category
    waveform_template_summary.pkl  the same pulse-shape parameters measured on the average waveforms
    waveform_profiles.pkl          templates unfolded with the SPE response (photon arrival profiles)
    waveform_profile_summary.pkl   prompt / late fractions and slow time constant of the profiles
    waveform_examples.pkl          a few raw peak-aligned waveforms per category
    spe_responses.pkl              the SPE response used by each digitiser (16 ns samples)
    README.md, meta.json

Only the plane-integrated ("Total") and all-generator categories are exported; the
per-plane / per-generator detail stays in the per-sample pickles.

Usage:  python3 src/tools/export_waveform_repo.py
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/data/PDS/waveform"
DST = BASE / "export"
TICK = 0.016
SPE_FILE = "/cvmfs/dune.opensciencegrid.org/products/dune/duneopdet/v10_08_01d00/config_data/SPE_DAPHNE2_FBK_2022.dat"

# Sample label, detector description and the source recommended for the thesis figures.
SAMPLES = {
    ("hd_1x2x6_centralAPA", "marley_waveform"): "HD central APA (LAr), marley, PIC 2026-09-28",
    ("vd_1x8x14_3view_30deg_nominal", "marley_waveform"): "VD nominal (10 ppm Xe), marley, FNAL batch 006",
    ("vd_1x8x14_3view_30deg_nominal", "marley_flash"): "VD nominal (10 ppm Xe), marley, FNAL 9,967 jobs",
    ("vd_1x8x14_3view_30deg_nominal", "radiological_flash"): "VD nominal (10 ppm Xe), radiological, FNAL (1M clusters)",
    ("vd_1x8x14_3view_30deg_shielded", "marley_waveform"): "VD shielded (10 ppm Xe), marley, PIC 10 jobs",
    ("vd_1x8x14_3view_30deg_shielded", "marley_flash"): "VD shielded (10 ppm Xe), marley, PIC 280 jobs",
    ("vd_1x8x14_3view_30deg_shielded", "radiological_waveform"): "VD shielded (10 ppm Xe), radiological, PIC 7 jobs",
    ("vd_1x8x14_3view_30deg_shielded", "radiological_flash"): "VD shielded (10 ppm Xe), radiological, PIC (1M clusters)",
}
KINDS = ["Templates", "Features", "Summary", "Template_Summary", "Profiles", "Profile_Summary", "Examples"]


def spe_responses():
    t = np.arange(0, 5.2, TICK)
    peak, front, back = 0.028, 0.013, 0.386
    vd = 151.5 * 0.0594 * np.where(t < peak, np.exp((t - peak) / front), np.exp(-(t - peak) / back))
    hd = np.loadtxt(SPE_FILE)
    return pd.DataFrame([
        {"Geometry": "vd", "Digitizer": "WaveformDigitizerSim standard_daphne (analytic: 13 ns rise, 386 ns fall)",
         "Time": t, "ADC": vd, "Normalised": vd / vd.max()},
        {"Geometry": "hd", "Digitizer": "OpDetDigitizerDUNE, testbench SPE_DAPHNE2_FBK_2022.dat",
         "Time": np.arange(len(hd)) * TICK, "ADC": hd, "Normalised": hd / hd.max()},
    ])


def main():
    DST.mkdir(parents=True, exist_ok=True)
    out, counts = {}, {}
    for kind in KINDS:
        frames = []
        for (config, name), label in SAMPLES.items():
            f = BASE / config / name / f"{config}_{name}_Waveform_{kind}.pkl"
            if not f.is_file():
                continue
            df = pd.read_pickle(f)
            if "Plane" in df:
                df = df[df["Plane"] == "Total"]
            if "Generator" in df:
                df = df[df["Generator"].fillna("All") == "All"].drop(columns="Generator")
            if kind == "Templates":
                df = df[df["Selection"] == "Clean"]
            df.insert(3, "Sample", label)
            df.insert(4, "Detector", "HD" if config.startswith("hd") else "VD")
            frames.append(df)
        if frames:
            out[kind] = pd.concat(frames, ignore_index=True)
            out[kind].to_pickle(DST / f"waveform_{kind.lower()}.pkl")
            counts[kind] = len(out[kind])
    spe_responses().to_pickle(DST / "spe_responses.pkl")

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "src/physics/detector/pds/06_waveforms.py, 07_waveform_profiles.py; bundled by src/tools/export_waveform_repo.py",
        "samples": {f"{c}/{n}": l for (c, n), l in SAMPLES.items()},
        "rows": counts,
        "tick_us": TICK,
        "keys": ["Geometry", "Config", "Name", "Sample", "Detector", "Source", "Signal", "PEBin"],
    }
    (DST / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (DST / "README.md").write_text(README)
    print(f"Wrote {DST}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


README = """# PDS waveform characterisation (HD vs VD)

Bundle written by `src/tools/export_waveform_repo.py` (SOLAR repo) from the per-sample pickles of
`src/physics/detector/pds/06_waveforms.py` and `07_waveform_profiles.py`. All times in microseconds,
16 ns ticks (62.5 MHz for both detectors).

## What a waveform is
SolarNuAna stores, for every reconstructed OpFlash, the raw ADC waveform of its main (highest-PE) OpHit
channel. VD: `opdigi10ppm` (WaveformDigitizerSim, analytic SPE: 13 ns rise, 386 ns fall, ~9 ADC/PE,
pedestal 100, 320-tick windows extended on retrigger, 10 ppm Xe doping). HD: OpDetDigitizerDUNE with the
DAPHNE2 FBK 2022 testbench SPE template (peak at 144 ns, ~1 us fall, undershoot to -30 % at ~2 us),
pedestal 1500, 1000-tick windows. **The SPE responses differ** (see `spe_responses.pkl`), so raw shapes
characterise "detector + electronics" as simulated, not the scintillation alone; `waveform_profiles.pkl`
has the templates unfolded with each SPE response (photon arrival-time profiles).

## Categories (key columns)
- `Detector` HD / VD, `Sample` human-readable sample label, `Config`, `Name`.
- `Source`: `Matched` = the flash matched to each TPC cluster (reliable for both detectors: the flash time
  locates the pulse in the waveform); `Truth` = every flash of the event (no waveform timestamp: VD uses only
  single 320-tick windows; for HD the window-opening pulse is often not the flash's main hit, treat HD Truth
  with care).
- `Signal`: flash with light from the MARLEY neutrino (`Pur > 0`) or not.
- `PEBin`: MaxPE of the main OpHit: 1.5-5, 5-20, 20-100, 100-1000, >1000, All.

## Pulse-shape parameters (per waveform and on the average waveforms)
- `PeakTime`: peak - onset, onset = interpolated 10 % crossing of the leading edge.
- `RiseTime`: 10-90 % rise.
- `T50/T90/T99`: time from the onset containing 50/90/99 % of the positive-lobe charge, integrated over
  4.8 us after the onset (common window both detectors record; limits baseline and late-pulse bias).
  Only waveforms covering the whole window enter (`ChargeContainedFraction`).
- `FWHM`, `Undershoot` (minimum after the peak / amplitude; per waveform it is mostly noise at low PE,
  use the template value), `UndershootTime`, `LobeCharge` (positive-lobe charge / amplitude, us).
- `FPrompt` (charge in [-3, +6] ticks of the peak / integral), `FLate` (> 1 us), `ADCperPE`, `SNR`,
  `Baseline`, `BaselineRMS`, `SecondaryPulseFraction` (later pulses on the same channel: with Xe doping a
  low-light flash is a train of delayed single-PE pulses). HD integral fractions are distorted by the
  undershoot; prefer T50/T90/T99 and the unfolded profiles.

## Files
| file | one row per | arrays |
|---|---|---|
| waveform_templates.pkl | sample x Source x Signal x PEBin x Norm | Time, Mean, STD, SEM (N waveforms) |
| waveform_features.pkl | ... x Variable | Values, Edges, Counts, Density |
| waveform_summary.pkl | sample x Source x Signal x PEBin | `<feat>Median/Mean/STD/P16/P84`, fractions |
| waveform_template_summary.pkl | same | parameters of the average waveform |
| waveform_profiles.pkl | same | Time, Profile (sums to 1), Template, Reconvolved, Residual |
| waveform_profile_summary.pkl | same | PromptFraction (100 ns), LateFraction (>1 us), TauSlow |
| waveform_examples.pkl | same | Waveforms (up to 8 raw peak-aligned, baseline-subtracted) |
| spe_responses.pkl | detector | Time, ADC, Normalised |

## Recommended selections
- Main HD vs VD comparison: `Source == "Matched"`, `Signal == "Signal"` (MARLEY clusters),
  HD `hd_1x2x6_centralAPA/marley_waveform` vs VD `vd_1x8x14_3view_30deg_nominal/marley_flash`
  (largest VD MARLEY sample); `PEBin` "All" plus the 20-100 and 100-1000 bins.
- Nominal vs shielded VD agree bin by bin (same Xe doping); use them as a cross-check.
- HD statistics are small (24 jobs, ~1,500 matched clusters).
"""

if __name__ == "__main__":
    main()
