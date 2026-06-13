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
    default_pdf_export_enabled,
    export_marp_pdf,
    find_latest,
    pick_most_recent,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Truth Pipeline MARP presentation"
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export PDF alongside Markdown (default: on)",
    )
    return parser.parse_args()


def output_markdown_path():
    return ROOT / "output" / "presentations" / "TruthPipeline.md"


def _read_json_safe(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _rel(path_obj):
    try:
        return path_obj.relative_to(ROOT).as_posix()
    except Exception:
        return None


def _img(rel_path, caption=None):
    if rel_path is None:
        return "*(plot not found)*"
    parts = [f'<img src="../../{rel_path}">']
    if caption:
        parts.append(f"<p><em>{caption}</em></p>")
    return "\n".join(parts)


# ── Solar spectrum / asymmetry plots (config-independent) ────────────────────

def gather_truth_solar_plots():
    base = ROOT / "images" / "solar" / "truth"
    specs = {}
    for key, patterns in [
        ("solar_spectrum", ["solar_neutrino_spectrum.png", "Solar_Neutrino_Spectrum.png"]),
        ("cc_spectrum",    ["dune_CC_solar_neutrino_spectrum.png"]),
        ("nadir_conv",     ["dune_nadir_convolution.png"]),
        ("daynight_1d",    ["1D_Convolved_DayNight.png"]),
        ("daynight_asym",  ["1D_Convolved_DayNight_Asymmetry.png", "Ideal_DayNight_Asymmetry.png"]),
    ]:
        candidates = [base / p for p in patterns]
        # also check subfolder/newfigure.png variants
        for p in patterns:
            stem = Path(p).stem
            candidates.append(base / stem / "newfigure.png")
        match = pick_most_recent([c for c in candidates if c.exists()])
        specs[key] = _rel(match) if match else None

    # global solar spectrum from images/solar/
    if specs["solar_spectrum"] is None:
        for p in ["solar_neutrino_spectrum.png", "Solar_Neutrino_Spectrum.png"]:
            candidate = ROOT / "images" / "solar" / p
            if candidate.exists():
                specs["solar_spectrum"] = _rel(candidate)
                break
    return specs


def gather_nadir_pdf_plot():
    candidate = ROOT / "images" / "common" / "line_plots" / "NadirPDF.png"
    return _rel(candidate) if candidate.exists() else None


# ── Background energy distributions ──────────────────────────────────────────

def gather_background_dist_plots():
    specs = {}
    for config_key, display_name in STANDARD_CONFIGS:
        base = ROOT / "images" / "background" / config_key / "truth"
        if not base.exists():
            specs[config_key] = None
            continue
        pngs = sorted(base.glob("*.png"))
        # prefer the multi-component combined plot (index 0 or "all")
        combined = [p for p in pngs if "_0_" in p.name and "logx" not in p.name.lower()]
        match = combined[0] if combined else (pngs[0] if pngs else None)
        specs[config_key] = _rel(match) if match else None
    return specs


def gather_background_logx_plots():
    specs = {}
    for config_key, display_name in STANDARD_CONFIGS:
        base = ROOT / "images" / "background" / config_key / "truth"
        if not base.exists():
            specs[config_key] = None
            continue
        candidates = sorted(base.glob("*_0__logx.png"))
        specs[config_key] = _rel(candidates[0]) if candidates else None
    return specs


def gather_combined_background_plot():
    p = ROOT / "images" / "background" / "all_productions_combined_by_type.png"
    return _rel(p) if p.exists() else None


def gather_shielding_plots():
    base = ROOT / "images" / "background"
    plots = {}
    for key in ["cavernwall_gamma", "cavern_neutron", "foam_gamma", "cryostat_gamma"]:
        candidates = sorted(base.glob(f"vd_shielding_{key}_*.png"))
        plots[key] = _rel(candidates[0]) if candidates else None
    comparison = base / "shielding_comparison_vd_1x8x14_3view_30deg_shielded_cavernwall_gamma.png"
    plots["comparison"] = _rel(comparison) if comparison.exists() else None
    return plots


# ── PDF validation plots ──────────────────────────────────────────────────────

def gather_pdf_validation_plots():
    specs = {}
    for config_key, _ in STANDARD_CONFIGS:
        base = ROOT / "images" / "background" / config_key
        gamma_dir = base / "gamma"
        neutron_dir = base / "neutron"
        gamma_checks = sorted(gamma_dir.glob("*_pdf_check.png")) if gamma_dir.exists() else []
        neutron_checks = sorted(neutron_dir.glob("*_pdf_check.png")) if neutron_dir.exists() else []
        specs[config_key] = {
            "gamma": [_rel(p) for p in gamma_checks[:3]],
            "neutron": [_rel(p) for p in neutron_checks[:3]],
        }
    return specs


# ── Oscillogram plots ─────────────────────────────────────────────────────────
# Truth pipeline reads from pre-computed file backend oscillograms.
# These live in images/analysis/<analysis>/oscillogram/<config>/marley/<folder>/
# We prefer HEP oscillograms since they are often most complete.

def gather_oscillogram_plots():
    specs = {}
    for config_key, display_name in STANDARD_CONFIGS:
        for analysis in ["hep", "sensitivity", "daynight"]:
            for folder in ["truncated", "nominal", "reduced"]:
                osc_dir = (
                    ROOT / "images" / "analysis" / analysis
                    / "oscillogram" / config_key / "marley" / folder
                )
                osc = find_latest(
                    osc_dir,
                    [f"{config_key}_marley_Oscillogram_SolarEnergy.png"]
                )
                nadir = find_latest(
                    osc_dir,
                    [f"{config_key}_marley_Oscillogram_NadirProjection_SolarEnergy.png"]
                )
                if osc or nadir:
                    specs[config_key] = {
                        "oscillogram": _rel(osc),
                        "nadir": _rel(nadir),
                        "display_name": display_name,
                        "source": f"{analysis}/{folder}",
                    }
                    break
            if config_key in specs:
                break
        if config_key not in specs:
            specs[config_key] = {
                "oscillogram": None,
                "nadir": None,
                "display_name": display_name,
                "source": None,
            }
    return specs


# ── Renderers ─────────────────────────────────────────────────────────────────

def render_solar_spectrum_slides(solar_plots):
    slides = []
    pairs = [
        ("solar_spectrum", "Solar neutrino flux spectrum (Bahcall SSM)"),
        ("cc_spectrum", "DUNE CC-detected solar neutrino spectrum (MARLEY)"),
    ]
    available = [(key, cap) for key, cap in pairs if solar_plots.get(key)]
    if available:
        slides.append("\n".join([
            "### Solar Neutrino Spectra",
            "",
            '<div class="two-col">',
        ] + [
            f"  <div><p><strong>{cap}</strong></p>{_img(solar_plots[key])}</div>"
            for key, cap in available
        ] + ["</div>"]))
    else:
        slides.append("### Solar Neutrino Spectra\n\nNo spectrum plots found.")

    nadir_conv = solar_plots.get("nadir_conv")
    if nadir_conv:
        slides.append("\n".join([
            "### Nadir-Averaged Spectrum",
            "",
            '<div class="center">',
            f"  {_img(nadir_conv, 'DUNE CC spectrum convolved with SURF nadir distribution')}",
            "</div>",
        ]))

    return "\n\n---\n\n".join(slides)


def render_daynight_asymmetry_slides(solar_plots):
    dn_1d = solar_plots.get("daynight_1d")
    dn_asym = solar_plots.get("daynight_asym")
    if not (dn_1d or dn_asym):
        return "### Day-Night Reference\n\nNo day-night asymmetry plots found."

    parts = [
        "### Day-Night Asymmetry Reference",
        "",
        '<div class="two-col">',
        "  <div>",
        '    <p><strong>1D Day/Night convolved spectrum</strong></p>',
        f"    {_img(dn_1d) if dn_1d else '<p>Not found.</p>'}",
        "  </div>",
        "  <div>",
        '    <p><strong>Ideal day-night asymmetry ΔP/P</strong></p>',
        f"    {_img(dn_asym) if dn_asym else '<p>Not found.</p>'}",
        "  </div>",
        "</div>",
    ]
    return "\n".join(parts)


def render_oscillogram_slides(osc_specs):
    slides = []
    for config_key, spec in osc_specs.items():
        osc = spec.get("oscillogram")
        nadir = spec.get("nadir")
        name = spec.get("display_name", config_key)
        src = spec.get("source", "")
        if osc or nadir:
            osc_block = f'    <img src="../../{osc}">' if osc else "    <p>Not available.</p>"
            nadir_block = f'    <img src="../../{nadir}">' if nadir else "    <p>Not available.</p>"
            slides.append("\n".join([
                f"### {name}",
                "",
                f"*Source: {src}*" if src else "",
                "",
                '<div class="two-col">',
                "  <div>",
                '    <p><strong>P(νe→νe) heatmap</strong></p>',
                osc_block,
                "  </div>",
                "  <div>",
                '    <p><strong>Nadir projection</strong></p>',
                nadir_block,
                "  </div>",
                "</div>",
            ]))
        else:
            slides.append(f"### {name}\n\nNo oscillogram found.")
    if not slides:
        return "### Oscillograms\n\nNo oscillogram PNGs found. Run analysis pipeline first."
    return "\n\n---\n\n".join(slides)


def render_background_dist_slides(dist_plots, logx_plots):
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        lin = dist_plots.get(config_key)
        log = logx_plots.get(config_key)
        if lin or log:
            lin_block = f'    <img src="../../{lin}">' if lin else "    <p>Not available.</p>"
            log_block = f'    <img src="../../{log}">' if log else "    <p>Not available.</p>"
            slides.append("\n".join([
                f"### {display_name} — Background Distributions",
                "",
                '<div class="two-col">',
                "  <div>",
                '    <p><strong>Linear scale</strong></p>',
                lin_block,
                "  </div>",
                "  <div>",
                '    <p><strong>Log x-scale</strong></p>',
                log_block,
                "  </div>",
                "</div>",
            ]))
        else:
            slides.append(f"### {display_name}\n\nNo background distribution plots found.")
    if not slides:
        return "### Background Distributions\n\nNo background distribution plots found."
    return "\n\n---\n\n".join(slides)


def render_shielding_slides(shielding_plots):
    rows = [
        ("cavernwall_gamma", "Cavern wall γ shielding"),
        ("cavern_neutron", "Cavern neutron shielding"),
        ("foam_gamma", "Foam γ shielding"),
        ("cryostat_gamma", "Cryostat γ shielding"),
    ]
    available = [(key, label) for key, label in rows if shielding_plots.get(key)]
    comparison = shielding_plots.get("comparison")
    if not available and not comparison:
        return "### VD Shielding\n\nNo shielding plots found."
    slides = []
    if available:
        col_blocks = [
            f"  <div><p><strong>{label}</strong></p><img src='../../{shielding_plots[key]}'></div>"
            for key, label in available
        ]
        slides.append("\n".join([
            "### VD Bottom Shielded — Reduction Factors",
            "",
            '<div class="two-col">',
            "\n".join(col_blocks[:2]),
            "</div>",
        ]))
        if len(col_blocks) > 2:
            slides.append("\n".join([
                "### VD Bottom Shielded — Additional Components",
                "",
                '<div class="two-col">',
                "\n".join(col_blocks[2:]),
                "</div>",
            ]))
    if comparison:
        slides.append("\n".join([
            "### VD Shielded vs Nominal — Comparison",
            "",
            '<div class="center">',
            f"  {_img(comparison)}",
            "</div>",
        ]))
    return "\n\n---\n\n".join(slides)


def render_pdf_validation_slides(pdf_specs):
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        spec = pdf_specs.get(config_key, {})
        gamma_plots = spec.get("gamma", [])
        if gamma_plots:
            imgs = "\n".join(
                f"  <div><img src='../../{p}'></div>" for p in gamma_plots if p
            )
            slides.append("\n".join([
                f"### {display_name} — Gamma PDF Validation",
                "",
                '<div class="two-col">',
                imgs,
                "</div>",
            ]))
    if not slides:
        return "### PDF Validation\n\nNo PDF validation plots found."
    return "\n\n---\n\n".join(slides)


def _gather_configs_from_json():
    payload = _read_json_safe(ROOT / "analysis" / "backgrounds.json")
    return payload.get("DEFAULT_CONFIGS", [c for c, _ in STANDARD_CONFIGS])


def build_markdown(
    solar_plots,
    nadir_pdf,
    dist_plots,
    logx_plots,
    combined_plot,
    shielding_plots,
    pdf_specs,
    osc_specs,
):
    configs = _gather_configs_from_json()
    payload = _read_json_safe(ROOT / "analysis" / "backgrounds.json")
    truth_pipeline = payload.get("TRUTH_PIPELINE", {})
    bg_names = truth_pipeline.get("BACKGROUND_NAMES", ["gamma", "neutron"])
    sig_names = truth_pipeline.get("SIGNAL_NAMES", ["marley"])
    pdf_backend = truth_pipeline.get("PDF_BACKEND", "truth")

    combined_block = (
        f'<div class="center"><img src="../../{combined_plot}"></div>'
        if combined_plot
        else "*(combined background plot not found)*"
    )
    nadir_block = (
        f'<div class="center"><img src="../../{nadir_pdf}"></div>'
        if nadir_pdf
        else "*(NadirPDF.png not found)*"
    )

    text = textwrap.dedent(f"""
    ---
    marp: true
    math: katex
    description: Truth-level background and oscillation pipeline outputs
    paginate: true
    theme: dune
    ---

    <!-- AUTO-GENERATED: src/tools/presentations/truth.py -->

    <!-- _class: titlepage -->

    # Truth Pipeline

    ---

    ## Introduction

    This presentation documents the outputs of the **truth-level pre-processing pipeline** ([src/pipelines/run_truth.py](../../src/pipelines/run_truth.py)).

    The truth pipeline produces all inputs required by the analysis pipeline before any fiducialization or cut scan:
    - Solar neutrino flux spectra and oscillation probability matrices
    - Background energy distributions and KDE momentum PDFs
    - Solar background DataFrames aggregated from GEANT4 truth

    **Configs processed:** {', '.join(configs)}
    **Background names:** {', '.join(bg_names)}
    **Signal names:** {', '.join(sig_names)}
    **PDF backend:** {pdf_backend}

    ---

    ### Pipeline Stages

    | Stage | Script | Flag | Description |
    |---|---|---|---|
    | 0X | `01_process_oscillation.py` | `--oscillations` | Generate/rebin oscillation pkl |
    | 0Y | `02_background_spectra.py` | `--no-spectra` | Load truth-level flux spectra |
    | 0Y | `03_background_pdf.py` | `--no-pdf` | Build momentum PDFs per surface component |
    | 0Y | `common/line_plot.py` | `--no-plot` | Config-independent line plots (NadirPDF, etc.) |
    | 0Y | `05_signal_nadir_kde.py` | `--signal-kde` | Per-nadir signal KDE (legacy; off by default) |
    | 0Z | `solar_background.py` | `--no-background` | Aggregate background energy distributions |
    | 0Z | `solar_background_plot.py` | `--no-plot` | Plot background distributions |
    | 0Z | `common/oscillogram_plot.py` | `--no-plot` | P(νe→νe) heatmaps from pre-computed pkl |

    Oscillation pkl: generated by NuFast (`nufast`) or Prob3++ (`prob3`) on-the-fly; `file` backend reads pre-existing pkl.

    ---

    ## Solar Neutrino Physics

    ---

    {render_solar_spectrum_slides(solar_plots)}

    ---

    {render_daynight_asymmetry_slides(solar_plots)}

    ---

    ### Nadir Distribution at SURF

    {nadir_block}

    The nadir angle distribution at SURF (latitude {_read_physics_lat():.2f}°N) weights day-time and night-time neutrinos. Night-time neutrinos traverse Earth matter, enabling day-night asymmetry measurement.

    ---

    ## Oscillation Templates

    ---

    {render_oscillogram_slides(osc_specs)}

    ---

    ## Background Distributions

    ---

    ### All Configs — Combined Background

    {combined_block}

    ---

    {render_background_dist_slides(dist_plots, logx_plots)}

    ---

    {render_shielding_slides(shielding_plots)}

    ---

    ## Background PDFs

    ---

    {render_pdf_validation_slides(pdf_specs)}

    ---

    ## Coverage and Notes

    - Oscillogram plots are sourced from the first available analysis (HEP → Sensitivity → DayNight), first available folder.
    - Background distribution plots come from `images/background/{{config}}/truth/`.
    - Shielding reduction factors for VD Bottom Shielded are stored in `analysis/backgrounds.json` under `SPECTRA.SHIELDING`.
    - PDF backend: **{pdf_backend}** (set in `TRUTH_PIPELINE.PDF_BACKEND` in [analysis/backgrounds.json](../../analysis/backgrounds.json)).
    - Re-run to refresh after truth pipeline:
      - `/usr/bin/python3 src/tools/presentations/truth.py`
    """)

    text = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in text.splitlines()
    ).strip() + "\n"
    return text


def _read_physics_lat():
    payload = _read_json_safe(ROOT / "analysis" / "physics.json")
    return float(payload.get("DUNE_LATITUDE_DEG", 44.35))


def main():
    args = parse_args()
    export_pdf = args.pdf if args.pdf is not None else default_pdf_export_enabled()
    out_md = output_markdown_path()

    solar_plots = gather_truth_solar_plots()
    nadir_pdf = gather_nadir_pdf_plot()
    dist_plots = gather_background_dist_plots()
    logx_plots = gather_background_logx_plots()
    combined_plot = gather_combined_background_plot()
    shielding_plots = gather_shielding_plots()
    pdf_specs = gather_pdf_validation_plots()
    osc_specs = gather_oscillogram_plots()

    markdown = build_markdown(
        solar_plots, nadir_pdf, dist_plots, logx_plots,
        combined_plot, shielding_plots, pdf_specs, osc_specs,
    )
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
