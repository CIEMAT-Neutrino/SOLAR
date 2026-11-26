#!/usr/bin/env python3
"""Build the solar and reactor fit hypotheses of the Sensitivity analysis as a plotting pickle.

Inputs are the per-year templates exported by SOLAR (``src/physics/sensitivity/template_plot.py``):
the ``SolarEnergy_Sensitivity_Templates`` DataFrame (signal at the solar best fit, background,
axes and selected cuts) and the raw reactor-point signal template from
``{PATH}/SENSITIVITY/{config}/{name}/{folder}/{energy}/``. Both references use the same cuts.

Rows written (one 40 x 30 map each, nadir x energy):
  Hypothesis Solar   / Reactor   x  Component Signal / Background / Total   -> Counts
  Hypothesis Solar - Reactor        Component Residual                      -> Residual

``Total`` is the expectation mu^0 = p + T^bkg of eq. sens_response scaled to ``--exposure``.
``Residual`` is the signed Poisson-deviance residual of the solar Asimov data against the
reactor expectation, sign(o - mu) sqrt(2 dl), on the bins kept by the fit mask
(mu > 0 and T^bkg > 0; masked bins are NaN). Its squares sum to the un-profiled chi2_0.

Example (the templates of ``template_plot.py`` are in ``--input_dir``, where the output is written too):
    python3 src/physics/sensitivity/generate_sensitivity_hypotheses.py --config hd_1x2x6_centralAPA \\
        --input_dir output/data/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated \\
        --reactor <PATH>/SENSITIVITY/hd_1x2x6_centralAPA/marley/truncated/SolarEnergy/hd_1x2x6_centralAPA_marley_NHits1_AdjCl4_OpHits9_dm2_7.540e-05_sin13_2.200e-02_sin12_3.040e-01.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def deviance_terms(o, mu):
    """Per-bin Baker-Cousins terms dl = mu - o + o ln(o/mu), evaluated without cancellation."""
    x = (o - mu) / mu
    small = np.abs(x) < 1e-3
    series = x**2 / 2 - x**3 / 6 + x**4 / 12
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = (1 + x) * np.log1p(x) - x
    dl = mu * np.where(small, series, exact)
    return np.where(o > 0, dl, mu)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="hd_1x2x6_centralAPA")
    parser.add_argument("--name", default="marley")
    parser.add_argument("--energy", default="SolarEnergy")
    parser.add_argument("--input_dir", required=True,
                        help="Directory holding {config}_{name}_{energy}_Sensitivity_Templates.pkl; the output is written here too.")
    parser.add_argument("--output_dir", default=None, help="Directory of the output pkl (default: --input_dir).")
    parser.add_argument("--reactor", required=True, help="Raw per-year reactor-point signal template (.pkl, nadir x energy).")
    parser.add_argument("--exposure", type=float, default=30.0, help="Exposure in years (default: 30, the Sensitivity reference).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    templates = pd.read_pickle(input_dir / f"{args.config}_{args.name}_{args.energy}_Sensitivity_Templates.pkl")
    templates = templates.set_index("Component")
    signal_solar = np.asarray(templates.loc["Signal", "Counts"], dtype=float)
    background = np.asarray(templates.loc["Background", "Counts"], dtype=float)
    signal_reactor = np.asarray(pd.read_pickle(args.reactor), dtype=float)
    # With the Sensitivity score categories (SOLAR SENSITIVITY_SCORE_CATEGORIES) the raw template stacks one block of nadir
    # rows per category; the exported solar/background maps are summed over the categories, so the reactor one is too.
    n_rows = signal_solar.shape[0]
    if signal_reactor.shape != signal_solar.shape and signal_reactor.shape[1:] == signal_solar.shape[1:] \
            and signal_reactor.shape[0] % n_rows == 0:
        n_cat = signal_reactor.shape[0] // n_rows
        print(f"Reactor template: summing {n_cat} score categories of {n_rows} nadir rows")
        signal_reactor = signal_reactor.reshape(n_cat, n_rows, -1).sum(axis=0)
    if signal_reactor.shape != signal_solar.shape:
        raise SystemExit(f"Reactor template shape {signal_reactor.shape} != solar {signal_solar.shape}")

    scale = args.exposure
    exposure_label = f"{args.exposure:g} yr"
    maps = {
        "Solar": {"Signal": scale * signal_solar, "Background": scale * background},
        "Reactor": {"Signal": scale * signal_reactor, "Background": scale * background},
    }
    for hypothesis in maps.values():
        hypothesis["Total"] = hypothesis["Signal"] + hypothesis["Background"]

    o = maps["Solar"]["Total"]
    mu = maps["Reactor"]["Total"]
    mask = (mu > 0) & (maps["Reactor"]["Background"] > 0)
    residual = np.full_like(o, np.nan)
    residual[mask] = np.sign(o[mask] - mu[mask]) * np.sqrt(2 * deviance_terms(o[mask], mu[mask]))

    base = templates.iloc[0]
    common = {
        "Config": args.config,
        "Name": args.name,
        "EnergyLabel": args.energy,
        "Folder": base["Folder"],
        "NHits": int(base["NHits"]),
        "OpHits": int(base["OpHits"]),
        "AdjCl": int(base["AdjCl"]),
        "Exposure": args.exposure,
        "ExposureUnit": "year",
        "EnergyBins": list(base["EnergyBins"]),
        "NadirBins": list(base["NadirBins"]),
    }
    rows = []
    for hypothesis, components in maps.items():
        for component, values in components.items():
            rows.append({**common, "Hypothesis": hypothesis, "Component": component,
                         "Counts": values.tolist(), "CountsUnit": f"events / MeV / {exposure_label}",
                         "Residual": None, "ResidualUnit": None})
    rows.append({**common, "Hypothesis": "Solar - Reactor", "Component": "Residual",
                 "Counts": None, "CountsUnit": None,
                 "Residual": residual.tolist(), "ResidualUnit": r"\sigma"})

    Path(args.output_dir or input_dir).mkdir(parents=True, exist_ok=True)
    output = Path(args.output_dir or input_dir) / f"{args.config}_{args.name}_{args.energy}_Sensitivity_Hypotheses.pkl"
    pd.DataFrame(rows).to_pickle(output)

    chi2_0 = float(np.nansum(residual**2))
    print(f"Saved {output}")
    print(f"  S_solar = {maps['Solar']['Signal'].sum():.4g}, S_reactor = {maps['Reactor']['Signal'].sum():.4g}, "
          f"B = {maps['Solar']['Background'].sum():.4g} ({exposure_label})")
    print(f"  fit mask keeps {int(mask.sum())} of {mask.size} bins; unprofiled chi2_0(solar data | reactor) = {chi2_0:.4f}")
    print(f"  max |residual| = {np.nanmax(np.abs(residual)):.3f} sigma")


if __name__ == "__main__":
    main()
