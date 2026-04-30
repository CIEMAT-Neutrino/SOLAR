import argparse
import re
import textwrap
from presentation_common import (
    DEFAULT_ENERGY,
    ROOT,
    STANDARD_CONFIGS,
    config_alias,
    default_pdf_export_enabled,
    export_marp_pdf,
    output_energy_label,
    pick_most_recent,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Sensitivity MARP presentation from workflow outputs"
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
        help="Energy variable used when selecting plots",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="truncated",
        choices=["truncated", "nominal", "reduced"],
        help="Folder scope used when selecting plots",
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
        return ROOT / "presentations" / f"{folder_label}SensitivitySignificanceWorkflow.md"
    return ROOT / "presentations" / f"{energy}{folder_label}SensitivitySignificanceWorkflow.md"



def _relative(path_obj):
    return path_obj.relative_to(ROOT).as_posix()



def _extract_cut_info(filename):
    match = re.search(
        r"NHits(?P<nhits>\d+)_AdjCl(?P<adjcl>\d+)_OpHits(?P<ophits>\d+)(?:_Signal(?P<signal>\d+))?(?:_Bkg(?P<bkg>\d+))?",
        filename,
    )
    if not match:
        return None
    values = match.groupdict()
    return {
        "NHits": int(values["nhits"]) if values.get("nhits") else None,
        "AdjCl": int(values["adjcl"]) if values.get("adjcl") else None,
        "OpHits": int(values["ophits"]) if values.get("ophits") else None,
        "Signal": int(values["signal"]) if values.get("signal") else None,
        "Bkg": int(values["bkg"]) if values.get("bkg") else None,
    }


def _find_latest(base_dirs, patterns):
    candidates = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))
    return pick_most_recent(candidates)


def _matches_cut(path, cut_info):
    if cut_info is None:
        return False
    this_cut = _extract_cut_info(path.name)
    if this_cut is None:
        return False
    for key in ["NHits", "AdjCl", "OpHits", "Signal", "Bkg"]:
        if cut_info.get(key) is None:
            continue
        if this_cut.get(key) != cut_info.get(key):
            return False
    return True


def _find_latest_with_cut(base_dirs, patterns, cut_info):
    candidates = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))

    if cut_info:
        filtered = [path for path in candidates if _matches_cut(path, cut_info)]
        if filtered:
            return pick_most_recent(filtered)

    return pick_most_recent(candidates)


def _gather_component_plot(folder, config_key, label, energy, cut_info):
    folder_title = folder.title()
    energy_label = output_energy_label(energy)
    base_dirs = [
        ROOT / "images" / "analysis" / "sensitivity" / folder / config_key,
        ROOT / "images" / "sensitivity" / folder / config_key,
    ]
    selected = _find_latest_with_cut(
        base_dirs,
        [
            f"{config_key}_{label}_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
            f"{config_key}_{label}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
        ],
        cut_info,
    )
    if selected is None:
        return None
    return _relative(selected)


def gather_result_specs(folder, energy):
    specs = []
    energy_label = output_energy_label(energy)
    folder_title = folder.title()
    for config_key, display_name in STANDARD_CONFIGS:
        contour_dir = ROOT / "images" / "analysis" / "sensitivity" / config_key / folder

        solar_sin12_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_solar_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_solar_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        react_sin12_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_react_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_react_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        solar_sin13_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_solar_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_solar_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        react_sin13_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_react_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_react_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )

        selected_primary = solar_sin12_path or react_sin12_path or solar_sin13_path or react_sin13_path
        cut_source = selected_primary or solar_sin12_path or react_sin12_path
        cut_info = _extract_cut_info(cut_source.name) if cut_source else None

        solar_sin12 = _relative(solar_sin12_path) if solar_sin12_path is not None else None
        solar_sin13 = _relative(solar_sin13_path) if solar_sin13_path is not None else None
        react_sin12 = _relative(react_sin12_path) if react_sin12_path is not None else None
        react_sin13 = _relative(react_sin13_path) if react_sin13_path is not None else None

        specs.append(
            {
                "name": display_name,
                "config": config_key,
                "folder": folder,
                "primary": _relative(selected_primary) if selected_primary is not None else None,
                "solar_sin12": solar_sin12,
                "solar_sin13": solar_sin13,
                "react_sin12": react_sin12,
                "react_sin13": react_sin13,
                "cut_info": cut_info,
            }
        )
    return specs


def gather_template_specs(folder, energy, result_specs):
    base = ROOT / "images" / "analysis" / "sensitivity" / "templates" / folder
    energy_label = output_energy_label(energy)
    selected_signal = {}
    selected_background = {}
    cut_by_config = {
        spec["config"]: spec.get("cut_info")
        for spec in result_specs
    }

    for config_key, _ in STANDARD_CONFIGS:
        selected_background_path = _find_latest_with_cut(
            [base / config_key / "marley", base],
            [
                f"{config_key}_marley_Sensitivity_Templates_{energy_label}_NHits*_AdjCl*_OpHits*.png",
                f"*_{folder.title()}_Background_{energy_label}_NHits*_AdjCl*_OpHits*.png",
                f"{folder.title()}_Background_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
            cut_by_config.get(config_key),
        )

        selected = _find_latest_with_cut(
            [base / config_key],
            [
                f"{config_key}_Selected_Signal_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
            cut_by_config.get(config_key),
        )
        selected_background[config_key] = (
            _relative(selected_background_path) if selected_background_path is not None else None
        )
        selected_signal[config_key] = _relative(selected) if selected is not None else None

    return {
        "background": selected_background,
        "signal": selected_signal,
    }


def render_results_slides(specs):
    slides = []
    for spec in specs:
        if spec["primary"]:
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"center\">",
                        f"  <img src=\"../{spec['primary']}\">",
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
                        f"No combined sensitivity contour found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        return "### Sensitivity results\n\nNo result plots were found."
    return "\n\n---\n\n".join(slides)


def render_component_pair_slides(specs, left_key, right_key, left_label, right_label):
    slides = []
    for spec in specs:
        left = spec.get(left_key)
        right = spec.get(right_key)
        if left or right:
            left_content = f"    <img src=\"../{left}\">" if left else "    <p>Plot not available.</p>"
            right_content = f"    <img src=\"../{right}\">" if right else "    <p>Plot not available.</p>"
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        f"    <p><strong>{left_label}</strong></p>",
                        left_content,
                        "  </div>",
                        "  <div>",
                        f"    <p><strong>{right_label}</strong></p>",
                        right_content,
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
                        f"No {left_label}/{right_label} contour pair found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        return "### Contour grids\n\nNo component contour grids were found."
    return "\n\n---\n\n".join(slides)


def render_template_slides(template_specs):
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        background_plot = template_specs.get("background", {}).get(config_key)
        signal_plot = template_specs.get("signal", {}).get(config_key)
        if background_plot:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Background Template",
                        "",
                        "<div class=\"center\">",
                        f"  <img src=\"../{background_plot}\">",
                        "</div>",
                    ]
                )
            )
        else:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Background Template",
                        "",
                        f"No background template found for {display_name}",
                    ]
                )
            )

        if signal_plot:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Signal Template",
                        "",
                        "<div class=\"center\">",
                        f"  <img src=\"../{signal_plot}\">",
                        "</div>",
                    ]
                )
            )
        else:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Signal Template",
                        "",
                        f"No selected-signal template found for {display_name}",
                    ]
                )
            )

    return "\n\n---\n\n".join(slides)


def render_cut_table(folder, specs):
    title = "Selected Sensitivity Cuts by Config" if folder == "truncated" else f"Selected Sensitivity Cuts by Config ({folder.title()})"
    lines = [
        f"### {title}",
        "",
        "| Config | NHits | OpHits | AdjCl | Signal Unc. (%) | Bkg Unc. (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    has_any = False
    for spec in specs:
        values = spec.get("cut_info") or {}
        nhits = values.get("NHits", "-")
        ophits = values.get("OpHits", "-")
        adjcl = values.get("AdjCl", "-")
        signal = values.get("Signal", "-")
        bkg = values.get("Bkg", "-")
        if values:
            has_any = True
        lines.append(
            f"| {config_alias(spec['config'])} | {nhits} | {ophits} | {adjcl} | {signal} | {bkg} |"
        )

    if not has_any:
        lines.append("| *(no parsable cut metadata found in selected filenames)* | - | - | - | - | - |")
    return "\n".join(lines)


def render_folder_sections(folder, result_specs, template_specs):
    is_main = folder == "truncated"
    sin12_title = "Main Result: Contour Grids (sin12)" if is_main else f"Contour Grids (sin12, {folder.title()})"
    sin13_title = "Contour Grids (sin13)" if is_main else f"Contour Grids (sin13, {folder.title()})"
    templates_title = "Template Building" if is_main else f"Template Building ({folder.title()})"
    return f"""## {sin12_title}

---

{render_component_pair_slides(result_specs, 'solar_sin12', 'react_sin12', 'Solar Contour (sin12)', 'Reactor Contour (sin12)')}

---

## {sin13_title}

---

{render_component_pair_slides(result_specs, 'solar_sin13', 'react_sin13', 'Solar Contour (sin13)', 'Reactor Contour (sin13)')}

---

## {templates_title}

---

{render_template_slides(template_specs)}

---

{render_cut_table(folder, result_specs)}
"""


def build_markdown(energy, folder, folder_specs, folder_templates):
    alias_bullets = "\n".join([f"- {config}: {alias}" for config, alias in STANDARD_CONFIGS])
    energy_label = output_energy_label(energy)
    coverage = sum(1 for item in folder_specs if item.get("primary"))
    selected_title = folder.title()
    selected_sections = render_folder_sections(folder, folder_specs, folder_templates)

    text = textwrap.dedent(
        f"""
    ---
    marp: true
    description: Inputs, workflow outputs, and per-config Sensitivity results
    paginate: true
    theme: dune
    ---

    <!-- AUTO-GENERATED: scripts/generate_sensitivity_presentation.py -->

    <!-- _class: titlepage -->

    # Sensitivity Workflow

    ---

    ## Introduction

    This presentation summarizes the workflow and outputs of the Sensitivity analysis for the SOLAR project.
    - This deck is auto-generated from workflow outputs.
    - This deck is scoped to the **{selected_title}** folder for the **{energy_label}** reconstruction algorithm.

    Config aliases:
    {alias_bullets}

    ---

    ### Workflow

    - Orchestrator: [src/analysis/10SensitivityAnalysis.py](../src/analysis/10SensitivityAnalysis.py)
    - Step 1 (Background template): [src/analysis/14SensitivityBackgroundTemplate.py](../src/analysis/14SensitivityBackgroundTemplate.py)
    - Step 2 (Signal template): [src/analysis/14SensitivitySignalTemplate.py](../src/analysis/14SensitivitySignalTemplate.py)
    - Step 3 (Grid fit scan and best-cut storage): [src/analysis/14Sensitivity.py](../src/analysis/14Sensitivity.py)
    - Step 4 (Contour rendering): [src/analysis/14SensitivityContourPlot.py](../src/analysis/14SensitivityContourPlot.py)

    ---

    ### Workflow Outputs

    - Main contour grids (sin12/sin13): [images/analysis/sensitivity](../images/analysis/sensitivity)
    - Signal/background templates (figures): [images/analysis/sensitivity/templates](../images/analysis/sensitivity/templates)
    - Grid-scan data products (PKL): [data/analysis/sensitivity](../data/analysis/sensitivity)
    - Remote workflow outputs (PNFS): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY)

    ---

    ### Sensitivity Fit Summary

    - [src/analysis/14Sensitivity.py](../src/analysis/14Sensitivity.py) builds fake observed maps as signal template at each oscillation point plus the corresponding background template, then fits that map against the solar and reactor reference templates with free signal and background normalizations.
    - The fit in [lib/lib_root.py](../lib/lib_root.py) minimizes a Poisson deviance-like objective, not a generic least-squares chi-square.
    - For observed count $o_i$ and expected count $e_i$, the per-bin contribution is $2(e_i - o_i + o_i \\log(o_i / e_i))$ for $o_i > 0$ and $2e_i$ for $o_i = 0$, plus quadratic penalty terms on the fitted signal and background normalization shifts.
    - The saved grid values are the minimized fit objective returned by `Sensitivity_Fitter.Fit`; contour labels may display $\\sqrt{{\\chi^2}}$ as a visualization proxy, but the workflow fundamentally stores the minimized deviance / chi-square-like statistic itself.

    ---

{selected_sections}

---

## Coverage and Notes

- Configs with selected sin12 solar contour plots:
    - {folder}: {coverage}
- Cut table values are parsed from selected result filenames when available.
- Re-run script to refresh this folder after each workflow run:
    - /usr/bin/python3 scripts/generate_sensitivity_presentation.py --folder {folder}
"""
    )

    text = "\n".join(
        line[4:] if line.startswith("    ") else line for line in text.splitlines()
    ).strip() + "\n"
    return text


def main():
    args = parse_args()
    export_pdf = args.pdf if args.pdf is not None else default_pdf_export_enabled()
    out_md = output_markdown_path(args.energy, args.folder)

    folder_specs = gather_result_specs(args.folder, args.energy)
    folder_templates = gather_template_specs(args.folder, args.energy, folder_specs)

    markdown = build_markdown(
        args.energy,
        args.folder,
        folder_specs,
        folder_templates,
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