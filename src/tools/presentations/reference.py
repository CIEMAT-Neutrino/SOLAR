import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import textwrap
from pathlib import Path

from common import (
    ROOT,
    STANDARD_CONFIGS,
    config_alias,
    default_pdf_export_enabled,
    export_marp_pdf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SOLAR Reference MARP presentation"
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export PDF alongside Markdown (default: on)",
    )
    return parser.parse_args()


def output_markdown_path():
    return ROOT / "output" / "presentations" / "SOLARReference.md"


def _read_json_safe(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _fmt_pct(v):
    try:
        return f"{float(v) * 100:.0f}%"
    except Exception:
        return str(v)


def _fmt_sci(v):
    try:
        f = float(v)
        return f"{f:.2e}"
    except Exception:
        return str(v)


def gather_physics_params():
    payload = _read_json_safe(ROOT / "analysis" / "physics.json")
    return {
        "solar_dm2": payload.get("SOLAR_DM2", 6.0e-5),
        "react_dm2": payload.get("REACT_DM2", 7.54e-5),
        "atm_dm2": payload.get("ATM_DM2", 2.515e-3),
        "sin12": payload.get("SIN12", 0.304),
        "sin13": payload.get("SIN13", 0.022),
        "latitude_deg": payload.get("DUNE_LATITUDE_DEG", 44.35),
        "osc_backend": payload.get("OSCILLATION_BACKEND", "nufast"),
    }


def gather_analysis_config():
    payload = _read_json_safe(ROOT / "analysis" / "config.json")
    nuisance_profiles = payload.get("NUISANCE_PROFILES", {})
    default_profile = payload.get("DEFAULT_NUISANCE_PROFILE", "full")
    thresholds = payload.get("ANALYSIS_THRESHOLDS", {})
    uncertainties = payload.get("ANALYSIS_UNCERTAINTIES", {})
    best_sigma_refs = payload.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {})
    return {
        "nuisance_profiles": nuisance_profiles,
        "default_profile": default_profile,
        "thresholds": thresholds,
        "uncertainties": uncertainties,
        "best_sigma_refs": best_sigma_refs,
    }


def gather_background_config():
    payload = _read_json_safe(ROOT / "analysis" / "backgrounds.json")
    essential = payload.get("BACKGROUND_SAMPLES", {}).get("ESSENTIAL", {})
    analyses = payload.get("BACKGROUND_SAMPLES", {}).get("ANALYSES", {})
    pdf_backend = payload.get("TRUTH_PIPELINE", {}).get("PDF_BACKEND", "truth")
    return {
        "essential": essential,
        "analyses": analyses,
        "pdf_backend": pdf_backend,
    }


def render_config_table():
    lines = [
        "### Detector Configurations",
        "",
        "| Alias | Key | Module |",
        "|---|---|---|",
    ]
    module_map = {
        "hd_1x2x6_centralAPA": "HD FD module — central APA (two drift regions)",
        "hd_1x2x6_lateralAPA": "HD FD module — lateral APA (one drift region)",
        "vd_1x8x14_3view_30deg_nominal": "VD FD module — top LAr volume",
        "vd_1x8x14_3view_30deg_shielded": "VD FD module — bottom volume with passive shielding",
    }
    for config_key, display_name in STANDARD_CONFIGS:
        lines.append(f"| **{display_name}** | `{config_key}` | {module_map.get(config_key, '')} |")
    return "\n".join(lines)


def render_physics_table(params):
    rows = [
        ("Δm²₂₁ (solar best fit)", _fmt_sci(params["solar_dm2"]), "eV²", "NuFiT 6.1 solar"),
        ("Δm²₂₁ (reactor best fit)", _fmt_sci(params["react_dm2"]), "eV²", "NuFiT 6.1 reactor"),
        ("Δm²₃₁ (atmospheric)", _fmt_sci(params["atm_dm2"]), "eV²", "NuFiT 6.1"),
        ("sin²θ₁₂", f"{params['sin12']:.3f}", "—", "NuFiT 6.1"),
        ("sin²θ₁₃", f"{params['sin13']:.3f}", "—", "NuFiT 6.1"),
        ("DUNE latitude", f"{params['latitude_deg']:.2f}°N", "—", "SURF, Lead SD"),
    ]
    lines = [
        "### Oscillation Parameters (NuFiT 6.1)",
        "",
        "| Parameter | Value | Units | Source |",
        "|---|---:|---|---|",
    ]
    for name, val, units, src in rows:
        lines.append(f"| {name} | {val} | {units} | {src} |")
    return "\n".join(lines)


def render_uncertainty_table(uncertainties):
    lines = [
        "### Analysis Uncertainties",
        "",
        "| Analysis | Signal σ | Background σ |",
        "|---|---:|---:|",
    ]
    for analysis in ["DAYNIGHT", "HEP", "SENSITIVITY"]:
        cfg = uncertainties.get(analysis, {})
        sig = _fmt_pct(cfg.get("signal_uncertainty", 0.0))
        bkg = _fmt_pct(cfg.get("background_uncertainty", 0.02))
        lines.append(f"| {analysis} | {sig} | {bkg} |")
    return "\n".join(lines)


def render_threshold_table(thresholds):
    lines = [
        "### Analysis Thresholds",
        "",
        "| Analysis | MC threshold | Significance threshold |",
        "|---|---:|---:|",
    ]
    for analysis in ["DAYNIGHT", "HEP", "SENSITIVITY", "FIDUCIALIZATION"]:
        cfg = thresholds.get(analysis, {})
        mc = cfg.get("MC", "—")
        sig = cfg.get("SIGNIFICANCE", "—")
        lines.append(f"| {analysis} | {mc} | {sig} |")
    return "\n".join(lines)


def render_nuisance_table(nuisance_profiles, default_profile):
    lines = [
        "### Sensitivity Nuisance Profiles",
        "",
        f"Default profile: **{default_profile}**",
        "",
        "| Profile | Marginalize sin²θ₁₃ | Energy scale uncertainty |",
        "|---|:---:|:---:|",
    ]
    for pname, cfg in nuisance_profiles.items():
        marker = " *(default)*" if pname == default_profile else ""
        m13 = "✓" if cfg.get("MARGINALIZE_SIN13") else "✗"
        esc = "✓" if cfg.get("ENERGY_SCALE_UNCERTAINTY") else "✗"
        lines.append(f"| **{pname}**{marker} | {m13} | {esc} |")
    return "\n".join(lines)


def render_background_table(bkg_config):
    essential = bkg_config["essential"]
    analyses = bkg_config["analyses"]
    all_components = sorted({c for v in analyses.values() for c in v} | set(essential.keys()))
    header_analyses = ["DAYNIGHT", "HEP", "SENSITIVITY"]
    lines = [
        "### Background Component Selection",
        "",
        "| Component | Essential | " + " | ".join(header_analyses) + " |",
        "|---|:---:|" + "|:---:".join([""] * len(header_analyses)) + "|",
    ]
    for comp in all_components:
        ess = "✓" if essential.get(comp) else "✗"
        cols = ["✓" if comp in analyses.get(a, []) else "✗" for a in header_analyses]
        lines.append(f"| {comp} | {ess} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


def render_stat_methods_table(best_sigma_refs):
    rows = [
        ("DayNight", "Asimov two-sample Poisson LLR", best_sigma_refs.get("DAYNIGHT", "Asimov"),
         "[src/physics/daynight/01_daynight.py](../../src/physics/daynight/01_daynight.py)"),
        ("HEP", "Profile-likelihood (global β nuisance)", best_sigma_refs.get("HEP", "ProfileLikelihood"),
         "[src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py)"),
        ("Sensitivity", "Baker-Cousins Poisson deviance (2D fit)", best_sigma_refs.get("SENSITIVITY", "Asimov"),
         "[src/physics/sensitivity/04_best_cuts.py](../../src/physics/sensitivity/04_best_cuts.py)"),
    ]
    lines = [
        "### Statistical Methods per Analysis",
        "",
        "| Analysis | Method | Best-cut reference | Implementation |",
        "|---|---|---|---|",
    ]
    for analysis, method, ref, impl in rows:
        lines.append(f"| **{analysis}** | {method} | {ref} | {impl} |")
    return "\n".join(lines)


def build_markdown(params, analysis_cfg, bkg_cfg):
    alias_bullets = "\n".join(f"- **{alias}** → `{config}`" for config, alias in STANDARD_CONFIGS)
    config_table = render_config_table()
    physics_table = render_physics_table(params)
    uncertainty_table = render_uncertainty_table(analysis_cfg["uncertainties"])
    threshold_table = render_threshold_table(analysis_cfg["thresholds"])
    nuisance_table = render_nuisance_table(
        analysis_cfg["nuisance_profiles"], analysis_cfg["default_profile"]
    )
    background_table = render_background_table(bkg_cfg)
    stat_table = render_stat_methods_table(analysis_cfg["best_sigma_refs"])

    text = textwrap.dedent(f"""
    ---
    marp: true
    math: katex
    description: SOLAR analysis reference — physics parameters, statistical methods, and shared mathematics
    paginate: true
    theme: dune
    ---

    <!-- AUTO-GENERATED: src/tools/presentations/reference.py -->

    <!-- _class: titlepage -->

    # SOLAR Analysis Reference

    ---

    ## Overview

    This reference deck documents the shared physics conventions, statistical methods, and mathematical derivations used across the DUNE SOLAR analyses.

    **Analyses covered:** DayNight asymmetry, HEP discovery, Sensitivity (sin²θ₁₂/sin²θ₁₃ contours)

    **Config aliases:**
    {alias_bullets}

    ---

    {config_table}

    ---

    ## Physics Parameters

    ---

    {physics_table}

    ---

    ### Oscillation Backends

    Three backends compute P(νₑ→νₑ) probability matrices. Selected via `--oscillation_backend`.

    | Backend | Description | Use case |
    |---|---|---|
    | **nufast** *(default)* | NuFast C++ on-the-fly MSW+vacuum | Production; no pkl files required |
    | **prob3** | Prob3++ on-the-fly | Cross-check against NuFast |
    | **file** | Pre-computed oscillogram pkl | Legacy; requires prior run |

    The oscillogram pkl is a 2D array $(E, \\cos\\eta)$ evaluated at the best-fit point. For **nufast** and **prob3**, it is regenerated each run.

    Earth density profile: **PREM** (Preliminary Reference Earth Model). Day-night asymmetry integrates over SURF latitude {params['latitude_deg']:.2f}°N nadir angle distribution.

    ---

    ### MSW Matter Effect

    The solar neutrino survival probability in matter obeys the MSW (Mikheyev–Smirnov–Wolfenstein) equation. Inside the Earth, the effective mixing angle is modified by:

    $$
    \\tan 2\\theta_{{\\mathrm{{eff}}}} = \\frac{{\\sin 2\\theta_{{12}}}}{{\\cos 2\\theta_{{12}} - A/\\Delta m^2_{{21}}}}
    $$

    where $A = 2\\sqrt{{2}}\\,G_F N_e E$ is the matter potential.

    The day-night asymmetry arises because night-time neutrinos traverse Earth matter before detection. The regeneration factor $f_{{\\mathrm{{reg}}}} \\propto \\sin^2\\theta_{{12}}\\,\\Delta m^2_{{21}}$ controls asymmetry magnitude. See [Lim & Marciano 1988](https://doi.org/10.1103/PhysRevD.37.1368).

    ---

    ## Detector and Signal

    ---

    ### Signal: Solar 8B and HEP Neutrinos

    DUNE detects solar neutrinos via **charged-current (CC) interaction** on argon:
    $$
    \\nu_e + {{}}^{{40}}\\mathrm{{Ar}} \\to e^- + {{}}^{{40}}\\mathrm{{K}}^*
    $$

    Events are simulated with **MARLEY** ([Clark et al. 2020](https://arxiv.org/abs/2011.03279)), which models the LAr nuclear response.

    The detected energy spectrum is the convolution of:
    - Solar 8B/hep flux $\\Phi(E_\\nu)$ ([Bahcall et al. 2005](https://arxiv.org/abs/astro-ph/0412440))
    - CC cross-section $\\sigma(E_\\nu)$
    - Oscillation-weighted survival probability $P_{{ee}}(E_\\nu, \\cos\\eta)$
    - Detector energy response $H(E_\\nu \\to E_{{\\mathrm{{reco}}}})$

    ---

    ### Cut Parameters

    Three analysis cut variables applied to reconstructed events:

    | Variable | Description | Typical range |
    |---|---|---|
    | **NHits** | Number of TPC wire hits | 15–50 |
    | **OpHits** | Number of optical hits | 0–50 |
    | **AdjCl** | Adjacent cluster count | 0–5 |

    Best cuts are selected per analysis by the significance at a chosen exposure crossing (Asimov or ProfileLikelihood curve). The cut scan iterates over the full (NHits × OpHits × AdjCl) grid defined in [config/analysis/config.json](../../config/analysis/config.json).

    ---

    ### Fiducialization

    Each production simulates **one half** of the physical module (one cathode/drift side). The fiducial volume is defined by symmetric margins (FidX, FidY, FidZ in cm) from detector boundaries.

    - **lateralAPA**: workspace = one drift; `drift_factor = 2` (two APA sides)
    - **centralAPA**: full two-drift workspace; `drift_factor = 1`; X margin applied from both sides
    - **VD**: one-sided top-boundary fiducial; `drift_factor = 1`

    Fiducial mass (kt):
    $$
    M_{{\\mathrm{{fid}}}} = L_X^{{\\mathrm{{fid}}}} \\times L_Y^{{\\mathrm{{fid}}}} \\times L_Z^{{\\mathrm{{fid}}}} \\times \\rho_{{\\mathrm{{LAr}}}} \\times d_{{\\mathrm{{drift}}}} \\times F_{{\\mathrm{{det}}}} \\times 0.5 \\times 10^{{-9}}
    $$

    with $\\rho_{{\\mathrm{{LAr}}}} = 1.396$ g/cm³.

    ---

    ## Background

    ---

    {background_table}

    ---

    ### Background Model

    Backgrounds are sampled from GEANT4 simulations and reweighted using KDE-based position PDFs built per surface component (APA, cathode, membrane, endcap, …).

    The weight for a simulated background event at position $(x, y, z)$ from surface $s$:
    $$
    w_s(x,y,z) = \\frac{{\\hat{{f}}_s(x,y,z)}}{{f_{{\\mathrm{{sim}},s}}(x,y,z)}}
    $$

    where $\\hat{{f}}_s$ is the KDE estimate of the true surface activity density and $f_{{\\mathrm{{sim}},s}}$ is the simulation density.

    Shielded VD configs apply an additional per-component rate reduction factor from measured cavern spectra (stored in `SPECTRA.SHIELDING` in [config/analysis/backgrounds.json](../../config/analysis/backgrounds.json)).

    ---

    ## Smoothing

    ---

    ### Histogram Smoothing Model

    Linear smoothing kernel $K$ applied to histogram $h$:
    $$
    \\tilde{{h}}_i = \\sum_j K_{{ij}}\\, h_j
    $$

    Integral-preserving renormalization after smoothing:
    $$
    \\tilde{{h}}_i \\leftarrow \\tilde{{h}}_i \\cdot \\frac{{\\sum_j h_j}}{{\\sum_j \\tilde{{h}}_j}}
    $$

    Variance propagation through same linear operator:
    $$
    v^{{\\mathrm{{out}}}} = (K \\odot K)\\,v, \\qquad \\sigma^{{\\mathrm{{out}}}}_i = \\sqrt{{v^{{\\mathrm{{out}}}}_i}}
    $$

    Smoothing config lives in [config/analysis/smoothing.json](../../config/analysis/smoothing.json) per analysis and stage.

    ---

    ### Threshold-Slice Smoothing

    Used for the DayNight and HEP threshold region — bins below threshold keep raw values, bins above receive smoothed values:
    $$
    h^{{\\mathrm{{out}}}}_i =
    \\begin{{cases}}
    h_i, & i < i_{{\\mathrm{{thr}}}} \\\\
    \\tilde{{h}}_i, & i \\ge i_{{\\mathrm{{thr}}}}
    \\end{{cases}}
    $$

    ---

    ## Statistical Methods

    ---

    {stat_table}

    ---

    {uncertainty_table}

    ---

    {threshold_table}

    ---

    ### Barlow-Beeston Lite MC Mask

    Bins where the expected background has fewer than `min_mc_per_bin` raw MC events are excluded from significance computations. This prevents log-likelihood divergence from bins with Poisson-zero MC support. Implementation follows the [Barlow-Beeston lite criterion](https://www.sciencedirect.com/science/article/pii/009350659390005W) as in [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html).

    Practical effect: bins with $N_{{\\mathrm{{MC}}}} < \\texttt{{min\\_mc\\_per\\_bin}}$ are masked (set to zero weight) before the LLR or PL evaluation. Affects primarily the low-statistics high-energy tail.

    ---

    {nuisance_table}

    ---

    ## Analysis Cross-References

    | Deck | Script | Content |
    |---|---|---|
    | [HEP Workflow](TruncatedHEPSignificanceWorkflow.md) | [src/tools/presentations/hep.py](../../src/tools/presentations/hep.py) | PL significance, exposure, spike debug |
    | [DayNight Workflow](TruncatedDayNightSignificanceWorkflow.md) | [src/tools/presentations/daynight.py](../../src/tools/presentations/daynight.py) | Asimov/Gaussian day-night statistics |
    | [Sensitivity Workflow](TruncatedSensitivitySignificanceWorkflow.md) | [src/tools/presentations/sensitivity.py](../../src/tools/presentations/sensitivity.py) | sin²θ₁₂/sin²θ₁₃ contour grids |
    | [Truth Pipeline](TruthPipeline.md) | [src/tools/presentations/truth.py](../../src/tools/presentations/truth.py) | Spectra, oscillograms, background PDFs |
    | **Reference** *(this deck)* | [src/tools/presentations/reference.py](../../src/tools/presentations/reference.py) | Physics params, math, shared conventions |
    """)

    text = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in text.splitlines()
    ).strip() + "\n"
    return text


def main():
    args = parse_args()
    export_pdf = args.pdf if args.pdf is not None else default_pdf_export_enabled()
    out_md = output_markdown_path()

    params = gather_physics_params()
    analysis_cfg = gather_analysis_config()
    bkg_cfg = gather_background_config()

    markdown = build_markdown(params, analysis_cfg, bkg_cfg)
    out_md.write_text(markdown)
    print(f"Wrote {out_md}")
    if export_pdf:
        try:
            out_pdf, pdf_error = export_marp_pdf(out_md)
            if out_pdf is not None:
                print(f"Wrote {out_pdf}")
            else:
                print(f"Warning: could not export PDF for {out_md}: {pdf_error}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: PDF export failed for {out_md}: {exc}")
    else:
        print(f"Skipped PDF export for {out_md} (auto-disabled or --no-pdf)")


if __name__ == "__main__":
    main()
