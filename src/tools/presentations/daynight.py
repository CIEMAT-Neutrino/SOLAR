import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import textwrap
from glob import glob
from pathlib import Path
from common import (
    DEFAULT_ENERGY,
    ROOT,
    STANDARD_CONFIGS,
    analysis_json_globs,
    compute_fiducial_mass_kt,
    config_alias,
    default_pdf_export_enabled,
    energy_candidates,
    export_marp_pdf,
    find_latest,
    gather_oscillogram_specs,
    output_energy_label,
    pick_most_recent,
    render_oscillogram_slides,
)


def fmt_float(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_int(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "-"


def read_json(path):
    with open(path, "r") as f_in:
        return json.load(f_in)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DayNight MARP presentation from workflow outputs"
    )
    parser.add_argument(
        "--energy",
        type=str,
        default=DEFAULT_ENERGY,
        choices=[
            "SignalParticleK",
            "ClusterEnergy",
            "TotalEnergy",
            "SelectedEnergy",
            "SolarEnergy",
        ],
        help="Energy variable used when selecting plots and summaries",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="truncated",
        choices=["truncated", "nominal", "reduced"],
        help="Folder scope used when selecting plots and summaries",
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export PDF alongside Markdown (default: on)",
    )
    return parser.parse_args()


def output_markdown_path(energy, folder):
    folder_label = folder.title()
    if energy == DEFAULT_ENERGY:
        return ROOT / "output" / "presentations" / f"{folder_label}DayNightSignificanceWorkflow.md"
    return ROOT / "output" / "presentations" / f"{energy}{folder_label}DayNightSignificanceWorkflow.md"



def gather_best_sigma_rows(energy):
    rows = {"nominal": [], "reduced": [], "truncated": []}
    row_map = {}
    patterns = analysis_json_globs("DAYNIGHT", "*_highest_DayNight.json")

    for pattern in patterns:
        for json_path in sorted(glob(pattern)):
            p = Path(json_path)
            folder = p.parts[-4] if "/pnfs/" in json_path else p.parts[-3]
            if folder not in rows:
                continue

            payload = read_json(p)
            for cfg, samples in payload.items():
                marley = samples.get("marley", {})
                if not marley:
                    continue

                energy_key = None
                for candidate in energy_candidates(energy):
                    if candidate in marley:
                        energy_key = candidate
                        break
                if energy_key is None:
                    continue
                vals = marley[energy_key]
                row_key = (folder, cfg, energy_key)
                if row_key in row_map:
                    continue

                row_map[row_key] = {
                    "Config": cfg,
                    "NHits": vals.get("NHits"),
                    "OpHits": vals.get("OpHits"),
                    "AdjCl": vals.get("AdjCl"),
                    "Value": vals.get("Values"),
                }
                rows[folder].append(row_map[row_key])

    for folder in rows:
        rows[folder] = sorted(rows[folder], key=lambda row: row["Config"])
    return rows


def gather_fiducial_rows(energy):
    fid_rows = []
    for folder in ["nominal", "reduced", "truncated"]:
        path = ROOT / "data" / "solar" / "fiducial" / folder / "BestFiducials.json"
        if not path.exists():
            continue

        payload = read_json(path)
        for cfg, analyses in payload.items():
            daynight = analyses.get("DAYNIGHT", {})
            if not daynight:
                continue
            selected_key = None
            for candidate in energy_candidates(energy):
                if candidate in daynight:
                    selected_key = candidate
                    break
            if selected_key is None:
                continue

            vals = daynight[selected_key]
            fid_rows.append(
                {
                    "Folder": folder,
                    "Config": cfg,
                    "FidX": vals.get("FiducialX"),
                    "FidY": vals.get("FiducialY"),
                    "FidZ": vals.get("FiducialZ"),
                    "BeforeFid": vals.get("NoFiducialSignificance", vals.get("RawSignificance")),
                    "AfterFid": vals.get("BestFiducialSignificance", vals.get("SmoothedSignificance")),
                    "Exposure": compute_fiducial_mass_kt(
                        cfg,
                        vals.get("FiducialX"),
                        vals.get("FiducialY"),
                        vals.get("FiducialZ"),
                    ),
                }
            )

    return sorted(fid_rows, key=lambda row: (row["Folder"], row["Config"]))


def gather_daynight_plot_specs(folder, energy):
    standard_configs = STANDARD_CONFIGS
    plot_dir = ROOT / "output" / "images" / "analysis" / "day-night"
    energy_label = output_energy_label(energy)

    slides = []
    for config_key, display_name in standard_configs:
        config_dir = plot_dir / config_key / "marley" / folder
        expected_significance = find_latest(
            config_dir,
            [
                f"{config_key}_marley_{energy_label}_DayNight_Significance_Exposure_*_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_DayNight_Significance_Exposure_*.png",
            ],
        )
        expected_exposure = find_latest(
            config_dir,
            [
                f"{config_key}_marley_{energy_label}_DayNight_Exposure_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_DayNight_Exposure_*.png",
            ],
        )

        slides.append(
            {
                "name": display_name,
                "config": config_key,
                "folder": folder,
                "exposure": expected_exposure.relative_to(ROOT).as_posix()
                if expected_exposure is not None
                else None,
                "significance": expected_significance.relative_to(ROOT).as_posix()
                if expected_significance is not None
                else None,
            }
        )

    return slides


def _find_fiducial_plot(folder, config_key, label, energy):
    root_dir = ROOT / "output" / "images" / "solar" / "fiducial"
    energy_label = output_energy_label(energy)
    candidate_dirs = [
        root_dir / config_key / "marley" / folder,
        root_dir / folder / config_key / "marley",
        root_dir / config_key / folder,
        root_dir / folder / config_key,
    ]
    patterns = [
        f"{config_key}_marley_{energy_label}_DAYNIGHT_{label}Fiducial_Significance*.png",
        f"{config_key}_{energy_label}_DAYNIGHT_{label}Fiducial_Significance*.png",
        f"{config_key}_marley_{energy_label}_{label}Fiducial_Significance*.png",
        f"{config_key}_{energy_label}_{label}Fiducial_Significance*.png",
    ]
    candidates = []
    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))
    match = pick_most_recent(candidates)
    if match is not None:
        return match.relative_to(ROOT).as_posix()
    return None


def gather_fiducial_plot_specs(folder, energy):
    specs = []
    for config_key, display_name in STANDARD_CONFIGS:
        best_plot = _find_fiducial_plot(folder, config_key, "Best", energy)
        no_plot = _find_fiducial_plot(folder, config_key, "No", energy)
        specs.append(
            {
                "name": display_name,
                "config": config_key,
                "folder": folder,
                "best": best_plot,
                "no": no_plot,
            }
        )
    return specs


def render_sigma_table(folder, rows):
    title = "Best DayNight Cuts by Config" if folder == "truncated" else f"Best DayNight Cuts by Config ({folder.title()})"
    lines = [
        f"### {title}",
        "",
        "| Config | NHits | OpHits | AdjCl | Significance |",
        "|---|---:|---:|---:|---:|",
    ]

    if not rows:
        lines.append("| *(no workflow JSON found)* | - | - | - | - |")
        return "\n".join(lines)

    for row in rows:
        lines.append(
            "| "
            + f"{config_alias(row['Config'])} | {fmt_int(row['NHits'])} | {fmt_int(row['OpHits'])} | {fmt_int(row['AdjCl'])} | {fmt_float(row['Value'])} |"
        )
    return "\n".join(lines)


def render_fid_table(folder, rows):
    filtered_rows = [row for row in rows if str(row.get("Folder", "")).lower() == folder]
    title = "Fiducial Optimization Summary" if folder == "truncated" else f"Fiducial Optimization Summary ({folder.title()})"
    lines = [
        f"### {title}",
        "",
        "| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    if not filtered_rows:
        lines.append("| *(no DAYNIGHT entries found)* | - | - | - | - | - | - |")
        return "\n".join(lines)

    for row in filtered_rows:
        lines.append(
            "| "
            + f"{config_alias(row['Config'])} | {fmt_int(row['FidX'])} | {fmt_int(row['FidY'])} | {fmt_int(row['FidZ'])} | {fmt_float(row['BeforeFid'])} | {fmt_float(row['AfterFid'])} | {fmt_float(row.get('Exposure'), digits=2)} |"
        )

    return "\n".join(lines)


def render_daynight_plot_slides(plot_specs):
    slides = []
    for spec in plot_specs:
        if spec["exposure"] and spec["significance"]:
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        "    <p><strong>Significance</strong></p>",
                        f"    <img src=\"../../{spec['significance']}\">",
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Exposure</strong></p>",
                        f"    <img src=\"../../{spec['exposure']}\">",
                        "  </div>",
                        "</div>",
                    ]
                )
            )
        else:
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        f"No matching exposure/significance pair found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        slides.append("### Plot outputs\n\nNo DayNight plot PNGs were found in output/images/analysis/day-night.")

    return "\n\n---\n\n".join(slides)


def render_fiducial_plot_slides(folder, specs):
    slides = []
    for spec in specs:
        if spec["best"] or spec["no"]:
            best_img = (
                f"    <img src=\"../../{spec['best']}\">"
                if spec["best"]
                else "    <p>Best fiducial plot not available.</p>"
            )
            no_img = (
                f"    <img src=\"../../{spec['no']}\">"
                if spec["no"]
                else "    <p>No fiducial plot not available.</p>"
            )
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        "    <p><strong>No Fiducial</strong></p>",
                        no_img,
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Best Fiducial</strong></p>",
                        best_img,
                        "  </div>",
                        "</div>",
                    ]
                )
            )

    if not slides:
        return "### Fiducial plots\n\nNo fiducial optimization plots were found for this folder."

    return "\n\n---\n\n".join(slides)


def render_folder_sections(folder, fid_specs, daynight_specs, best_sigma_rows, fid_rows, osc_specs=None):
    is_main = folder == "truncated"
    fid_title = "Fiducialization" if is_main else f"Fiducialization ({folder.title()})"
    daynight_title = "DayNight Results" if is_main else f"DayNight Results ({folder.title()})"
    osc_title = "Oscillograms" if is_main else f"Oscillograms ({folder.title()})"
    osc_section = ""
    if osc_specs:
        osc_section = f"""## {osc_title}

---

{render_oscillogram_slides(osc_specs)}

---

"""
    return f"""## {fid_title}

---

{render_fiducial_plot_slides(folder, fid_specs)}

---

{render_fid_table(folder, fid_rows)}

---

## {daynight_title}

---

{render_daynight_plot_slides(daynight_specs)}

---

{render_sigma_table(folder, best_sigma_rows.get(folder, []))}

---

{osc_section}"""


def build_markdown(
    energy,
    folder,
    best_sigma_rows,
    fid_rows,
    daynight_specs,
    fid_specs,
    osc_specs=None,
):
    coverage = {folder: len(rows) for folder, rows in best_sigma_rows.items()}
    energy_label = output_energy_label(energy)
    alias_bullets = "\n".join(
        [f"- {config}: {alias}" for config, alias in STANDARD_CONFIGS]
    )
    selected_sections = render_folder_sections(
        folder,
        fid_specs,
        daynight_specs,
        best_sigma_rows,
        fid_rows,
        osc_specs=osc_specs,
    )
    selected_title = folder.title()
    text = textwrap.dedent(f"""
    ---
    marp: true
    description: Inputs, workflow outputs, and per-config DayNight results
    paginate: true
    theme: dune
    ---
    
    <!-- AUTO-GENERATED: scripts/generate_daynight_presentation.py -->

    <!-- _class: titlepage -->
    
    # DayNight Significance Workflow
    
    ---
    
    ## Introduction
    
    This presentation summarizes the workflow and outputs of the DayNight significance analysis for the SOLAR project.
    - This deck is auto-generated from workflow outputs.
    - This deck is scoped to the **{selected_title}** folder for the **{energy_label}** reconstruction algorithm.
    
    Config aliases:
    {alias_bullets}

    ---
    
    ### Workflow

    - config: list of detector configs
    - folder: **{selected_title}**
    - analysis: DayNight
    - exposure: default **30 years**
    - threshold in daynight/01_daynight.py: default 8.0 MeV
    - optional cuts override: nhits, ophits, adjcls
    - MC threshold (`--mc_threshold`): minimum MC counts required in each essential background (gamma, neutron) per cut; prevents selecting cuts that eliminate backgrounds statistically
    - best-curve reference in sensitivity/05_best_sigmas.py: **Asimov** (two-sample Poisson LLR)
    - day-fraction (`--day_fraction`): fraction of exposure in daytime; default 0.5
    - oscillation band (`--oscillation_band`): residual uncertainty on θ₁₂, Δm²₂₁; combined in quadrature with earth-density band
    
    ---
    
    ### Workflow Outputs
    
    - Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../../data/solar/fiducial/truncated/BestFiducials.json)
    - Best cut summaries (JSON): [data/analysis/best-sigma-json/daynight/truncated](../../data/analysis/best-sigma-json/daynight/truncated)
    - Backward-compatible local fallback: [data/analysis/daynight-json/truncated](../../data/analysis/daynight-json/truncated)
    - Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated)
    - Figures: [output/images/analysis/day-night/truncated](../../output/images/analysis/day-night/truncated)
    
    ---
    
    ### Day-Night Discovery Statistic

    Two statistics are computed in parallel by [src/physics/daynight/01_daynight.py](../../src/physics/daynight/01_daynight.py):

    **Gaussian (legacy):** per-bin $Z_i = \Delta S_i / \sqrt{{B_i^{{eff}}}}$ combined as $Z = \sqrt{{\sum_i Z_i^2}}$, where $B_i^{{eff}}$ accounts for unequal day/night fractions:
    $$
    B_i^{{eff}} = \frac{{n_i^{{night}}}}{{g^2}} + \frac{{n_i^{{day}}}}{{f^2}}, \quad n_i^{{night}} = g(B_i + S_i^{{night}}),\; n_i^{{day}} = f(B_i + S_i^{{day}})
    $$

    **Asimov LLR (default):** two-sample Poisson log-likelihood ratio — see next slide.

    ---

    ### Day-Night Discovery Statistic — Asimov LLR

    Under $H_0$ (common day/night rate), the MLE is the pooled rate, giving expected counts $h_i^{{night}} = g(n_i^{{night}} + n_i^{{day}})$, $h_i^{{day}} = f(n_i^{{night}} + n_i^{{day}})$.  The test statistic sums linearly over bins:
    $$
    q_0 = 2\sum_i \left[ n_i^{{night}} \ln\\frac{{n_i^{{night}}}}{{h_i^{{night}}}} + n_i^{{day}} \ln\\frac{{n_i^{{day}}}}{{h_i^{{day}}}} \\right], \quad Z = \sqrt{{q_0}}
    $$

    Asymmetry uncertainty is bracketed by scaling the night signal: $S_i^{{night,k}} = S_i^{{day}} + k(S_i^{{night}} - S_i^{{day}})$ with $k \in \{{1 \pm \sigma_{{tot}}\}}$, $\sigma_{{tot}} = \sqrt{{\sigma_{{earth}}^2 + \sigma_{{osc}}^2}}$.

    ---

    ### Day-Night Discovery Statistic Details

    - Both Gaussian and Asimov curves are stored per cut; **Asimov is the default** for best-cut selection in [src/physics/sensitivity/05_best_sigmas.py](../../src/physics/sensitivity/05_best_sigmas.py) and exposure plots.
    - σ2/σ3 crossing exposures are tracked independently for both statistics: `Sigma2`/`Sigma3` (Gaussian) and `AsimovSigma2`/`AsimovSigma3`; fastest-discovery selection uses the Asimov crossing columns.
    - MC threshold gate: cuts where any essential background (gamma, neutron) has fewer than `--mc_threshold` MC events are skipped; prevents selecting cuts that deplete backgrounds statistically.
    - Smoothing is applied per component above threshold; the threshold slice keeps unsmoothed bins below threshold and replaces bins above with smoothed values.

    ---

    ### Context: Super-Kamiokande Day-Night Analysis

    Super-K measures the solar day-night effect with an energy-spectral chi-squared [[Abe et al., PRD 94, 052010 (2016)](https://doi.org/10.1103/PhysRevD.94.052010); [Renshaw et al., PRL 112, 091805 (2014)](https://doi.org/10.1103/PhysRevLett.112.091805)]:
    $$
    \\chi^2_{{SK}} = \\sum_{{k\\in\\{{D,N\\}}}} \\sum_j \\frac{{(N_{{kj}}-\\mu_{{kj}})^2}}{{\\sigma_{{kj}}^2}} + \\text{{(systematic penalties)}}
    $$
    In the statistical-only limit, DUNE's $Z_{{global}}^2$ is equivalent:
    $$
    Z_{{global}}^2 = \\sum_i \\frac{{(\\Delta S_i)^2}}{{B_i}} \\equiv \\chi^2_{{DN}}\\bigg|_{{\\sigma_i=\\sqrt{{B_i}}}}
    $$

    ---

    ### Similarities and Differences vs. Super-K

    **Shared structure:**
    - Energy-binned counting; day signal enters null hypothesis as background
    - MSW Earth matter effect is the physical driver of the night excess

    **DUNE vs. Super-K differences:**
    - No systematic nuisance penalty terms in DUNE baseline; second curve folds in background uncertainty
    - DUNE projects future discovery exposure; Super-K measures $A_{{DN}} = 2(\\Phi_N - \\Phi_D)/(\\Phi_N + \\Phi_D)$ from existing data
    - Energy binning only; Super-K also sub-bins by solar zenith angle for additional sensitivity

    ---

    ### Histogram Smoothing Math I

    - Linear smoothing model used per histogram bin:
    $$
    \\tilde{{h}}_i = \\sum_j K_{{ij}} h_j
    $$

    - Integral-preserving normalization applied after smoothing:
    $$
    \\tilde{{h}}_i \\leftarrow \\tilde{{h}}_i \\cdot \\frac{{\\sum_j h_j}}{{\\sum_j \\tilde{{h}}_j}}
    $$

    ---

    ### Histogram Smoothing Math II

    - Threshold-slice smoothing used for DayNight threshold region:
    $$
    h^{{\\mathrm{{out}}}}_i =
    \\begin{{cases}}
    h_i, & i < i_{{\\mathrm{{thr}}}} \\\\
    \\tilde{{h}}_i, & i \\ge i_{{\\mathrm{{thr}}}}
    \\end{{cases}}
    $$

    - Variance propagation through the same linear operator:
    $$
    v^{{\\mathrm{{out}}}} = (K \\odot K)\\,v, \\qquad \\sigma^{{\\mathrm{{out}}}}_i = \\sqrt{{v^{{\\mathrm{{out}}}}_i}}
    $$
    
    ---

{selected_sections}

---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: {coverage.get('nominal', 0)}
  - reduced: {coverage.get('reduced', 0)}
  - truncated: {coverage.get('truncated', 0)}
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
    - /usr/bin/python3 scripts/generate_daynight_presentation.py --folder {folder}
""")
    text = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in text.splitlines()
    ).strip() + "\n"
    return text


def main():
    args = parse_args()
    export_pdf = args.pdf if args.pdf is not None else default_pdf_export_enabled()
    out_md = output_markdown_path(args.energy, args.folder)
    best_sigma_rows = gather_best_sigma_rows(args.energy)
    fid_rows = gather_fiducial_rows(args.energy)
    selected_daynight_specs = gather_daynight_plot_specs(args.folder, args.energy)
    selected_fid_specs = gather_fiducial_plot_specs(args.folder, args.energy)
    selected_osc_specs = gather_oscillogram_specs(args.folder, args.energy, "DayNight")

    markdown = build_markdown(
        args.energy,
        args.folder,
        best_sigma_rows,
        fid_rows,
        selected_daynight_specs,
        selected_fid_specs,
        osc_specs=selected_osc_specs,
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
