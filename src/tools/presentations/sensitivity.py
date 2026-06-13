import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import re
import textwrap

try:
    import numpy as np
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
from common import (
    DEFAULT_ENERGY,
    ROOT,
    STANDARD_CONFIGS,
    compute_fiducial_mass_kt,
    config_alias,
    default_pdf_export_enabled,
    energy_candidates,
    export_marp_pdf,
    gather_oscillogram_specs,
    output_energy_label,
    pick_most_recent,
    render_oscillogram_slides,
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
    parser.add_argument(
        "--all_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Gather and render all slide sections: sin²θ₁₃ contours, "
            "template plots, significance spectra, and nuisance-profile comparison. "
            "Off by default — only sin²θ₁₂ contours and fiducial sections rendered."
        ),
    )
    return parser.parse_args()


def output_markdown_path(energy, folder):
    folder_label = folder.title()
    if energy == DEFAULT_ENERGY:
        return ROOT / "output" / "presentations" / f"{folder_label}SensitivitySignificanceWorkflow.md"
    return ROOT / "output" / "presentations" / f"{energy}{folder_label}SensitivitySignificanceWorkflow.md"



def _relative(path_obj):
    return path_obj.relative_to(ROOT).as_posix()


def fmt_float(value, digits=3):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value):
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


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


def gather_significance_specs(folder, energy):
    plot_dir = ROOT / "images" / "analysis" / "sensitivity"
    energy_label = output_energy_label(energy)
    specs = []
    for config_key, display_name in STANDARD_CONFIGS:
        sig_dir = plot_dir / config_key / "marley" / folder
        path = _find_latest(
            [sig_dir],
            [f"{config_key}_marley_{energy_label}_Sensitivity_Significance_Exposure_*.png"],
        )
        specs.append({
            "name": display_name,
            "config": config_key,
            "path": _relative(path) if path is not None else None,
        })
    return specs


def render_significance_slides(specs):
    slides = []
    for spec in specs:
        if spec["path"]:
            slides.append("\n".join([
                f"### {spec['name']}",
                "",
                "<div class=\"center\">",
                f"  <img src=\"../../{spec['path']}\">",
                "</div>",
            ]))
        else:
            slides.append(f"### {spec['name']}\n\nNo significance plot found for {spec['name']}.")
    if not slides:
        return "### Significance plots\n\nNo Sensitivity significance plots were found."
    return "\n\n---\n\n".join(slides)


def gather_result_specs(folder, energy, profile=None):
    specs = []
    energy_label = output_energy_label(energy)
    folder_title = folder.title()
    for config_key, display_name in STANDARD_CONFIGS:
        contour_dir = (
            ROOT / "images" / "analysis" / "sensitivity" / config_key / "marley" / folder / profile
            if profile else
            ROOT / "images" / "analysis" / "sensitivity" / config_key / "marley" / folder
        )

        solar_sin12_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_marley_solar_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_marley_solar_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        react_sin12_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_marley_react_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_marley_react_sin12_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        solar_sin13_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_marley_solar_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_marley_solar_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
            ],
        )
        react_sin13_path = _find_latest(
            [contour_dir],
            [
                f"{config_key}_marley_react_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*_Signal*_Bkg*.png",
                f"{config_key}_marley_react_sin13_df_{folder_title}_{energy_label}_NHits*_AdjCl*_OpHits*.png",
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
                        f"  <img src=\"../../{spec['primary']}\">",
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
            left_content = f"    <img src=\"../../{left}\">" if left else "    <p>Plot not available.</p>"
            right_content = f"    <img src=\"../../{right}\">" if right else "    <p>Plot not available.</p>"
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
        # signal_plot = template_specs.get("signal", {}).get(config_key)
        signal_plot = False
        if background_plot:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Templates",
                        "",
                        "<div class=\"center\">",
                        f"  <img src=\"../../{background_plot}\">",
                        "</div>",
                    ]
                )
            )
        else:
            slides.append(
                "\n".join(
                    [
                        f"### {display_name} Templates",
                        "",
                        f"No template found for {display_name}",
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
                        f"  <img src=\"../../{signal_plot}\">",
                        "</div>",
                    ]
                )
            )
        # else:
        #     slides.append(
        #         "\n".join(
        #             [
        #                 f"### {display_name} Signal Template",
        #                 "",
        #                 f"No selected-signal template found for {display_name}",
        #             ]
        #         )
        #     )

    return "\n\n---\n\n".join(slides)


def gather_fiducial_rows(energy):
    fid_rows = []
    for folder in ["nominal", "reduced", "truncated"]:
        path = ROOT / "data" / "solar" / "fiducial" / folder / "BestFiducials.json"
        if not path.exists():
            continue
        payload = read_json(path)
        for cfg, analyses in payload.items():
            sens = analyses.get("SENSITIVITY", {})
            if not sens:
                continue
            selected_key = None
            for candidate in energy_candidates(energy):
                if candidate in sens:
                    selected_key = candidate
                    break
            if selected_key is None:
                continue
            vals = sens[selected_key]
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


def _find_fiducial_plot(folder, config_key, label, energy):
    root_dir = ROOT / "images" / "solar" / "fiducial"
    energy_label = output_energy_label(energy)
    candidate_dirs = [
        root_dir / folder / config_key / "marley",
        root_dir / config_key / "marley" / folder,
        root_dir / config_key / folder,
        root_dir / folder / config_key,
    ]
    patterns = [
        f"{config_key}_marley_{energy_label}_SENSITIVITY_{label}Fiducial_Significance*.png",
        f"{config_key}_{energy_label}_SENSITIVITY_{label}Fiducial_Significance*.png",
    ]
    candidates = []
    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))
    expected = pick_most_recent(candidates)
    if expected is not None:
        return _relative(expected)
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


def render_fiducial_plot_slides(specs):
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
        lines.append("| *(no SENSITIVITY entries found)* | - | - | - | - | - | - |")
        return "\n".join(lines)
    for row in filtered_rows:
        lines.append(
            "| "
            + f"{config_alias(row['Config'])} | {fmt_int(row['FidX'])} | {fmt_int(row['FidY'])} | {fmt_int(row['FidZ'])} | {fmt_float(row['BeforeFid'])} | {fmt_float(row['AfterFid'])} | {fmt_float(row.get('Exposure'), digits=2)} |"
        )
    return "\n".join(lines)


def gather_sensitivity_significance_rows(folder, energy):
    """Read Sensitivity_Significance.pkl for each config and return {config: Z_1D}.

    Z_1D = sqrt(sum(per-bin AsimovTS)) — 1D projected Asimov significance for the
    best-sensitivity-score cut (NHits/OpHits/AdjCl) stored in the pkl.
    """
    if not _HAS_PANDAS:
        return {}
    result = {}
    energy_label = output_energy_label(energy)
    for config_key, _ in STANDARD_CONFIGS:
        pkl_path = (
            ROOT / "data" / "analysis" / "sensitivity"
            / config_key / "marley" / folder
            / f"{config_key}_marley_Sensitivity_Significance.pkl"
        )
        if not pkl_path.exists():
            continue
        try:
            df = pd.read_pickle(pkl_path)
            if df.empty:
                continue
            row = df.iloc[0]
            sig = np.asarray(row["Significance"], dtype=float)
            z = float(np.sqrt(np.nansum(sig)))
            result[config_key] = z
        except Exception:
            pass
    return result


def render_cut_table(folder, specs, sig_rows=None):
    sig_rows = sig_rows or {}
    title = "Selected Sensitivity Cuts by Config" if folder == "truncated" else f"Selected Sensitivity Cuts by Config ({folder.title()})"
    lines = [
        f"### {title}",
        "",
        "| Config | NHits | OpHits | AdjCl | Signal Unc. (%) | Bkg Unc. (%) | 1D Asimov Z (σ) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    has_any = False
    for spec in specs:
        values = spec.get("cut_info") or {}
        nhits = values.get("NHits", "-")
        ophits = values.get("OpHits", "-")
        adjcl = values.get("AdjCl", "-")
        signal = values.get("Signal", "-")
        bkg = values.get("Bkg", "-")
        z = sig_rows.get(spec["config"])
        z_str = fmt_float(z, digits=2) if z is not None else "-"
        if values:
            has_any = True
        lines.append(
            f"| {config_alias(spec['config'])} | {nhits} | {ophits} | {adjcl} | {signal} | {bkg} | {z_str} |"
        )

    if not has_any:
        lines.append("| *(no parsable cut metadata found in selected filenames)* | - | - | - | - | - | - |")
    return "\n".join(lines)


def render_nuisance_comparison_section(default_profile, default_specs, other_profiles_specs, nuisance_profiles=None):
    """Two-column comparison slides for each non-default profile vs the default."""
    if not other_profiles_specs:
        return ""
    nuisance_profiles = nuisance_profiles or {}
    table_rows = []
    for pname in [default_profile] + list(other_profiles_specs.keys()):
        cfg = nuisance_profiles.get(pname, {})
        items = ", ".join(f"{k}={v}" for k, v in cfg.items()) if cfg else "defaults"
        table_rows.append(f"| **{pname}** | {items} |")
    profile_table = (
        "| Profile | Settings |\n|---|---|\n" + "\n".join(table_rows)
    )

    slides = [
        "## Nuisance Parameter Comparison",
        "",
        "---",
        "",
        "### Profile Settings",
        "",
        profile_table,
    ]
    for other_name, other_specs in other_profiles_specs.items():
        slides += ["", "---", "", f"### {default_profile} vs {other_name}"]
        for default_spec, other_spec in zip(default_specs, other_specs):
            config_name = default_spec["name"]
            for key, label in [
                ("solar_sin12", "Solar sin²θ₁₂"),
                ("react_sin12", "Reactor sin²θ₁₂"),
            ]:
                left = default_spec.get(key)
                right = other_spec.get(key)
                if not (left or right):
                    continue
                left_img = f"    <img src=\"../../{left}\">" if left else "    <p>Not available.</p>"
                right_img = f"    <img src=\"../../{right}\">" if right else "    <p>Not available.</p>"
                slides += [
                    "",
                    "---",
                    "",
                    f"### {config_name}: {label}",
                    "",
                    "<div class=\"two-col\">",
                    "  <div>",
                    f"    <p><strong>{default_profile}</strong></p>",
                    left_img,
                    "  </div>",
                    "  <div>",
                    f"    <p><strong>{other_name}</strong></p>",
                    right_img,
                    "  </div>",
                    "</div>",
                ]
    return "\n".join(slides)


def render_folder_sections(folder, result_specs, template_specs, fid_rows=None, fid_specs=None, significance_specs=None, sig_rows=None, show_sin13=False, show_templates=False, show_significance_spectra=False, osc_specs=None):
    is_main = folder == "truncated"
    fid_title   = "Fiducialization" if is_main else f"Fiducialization ({folder.title()})"
    sin12_title = "Main Result: Contour Grids (sin12)" if is_main else f"Contour Grids (sin12, {folder.title()})"
    sin13_title = "Contour Grids (sin13)" if is_main else f"Contour Grids (sin13, {folder.title()})"
    templates_title = "Template Building" if is_main else f"Template Building ({folder.title()})"
    sig_title   = "Significance Spectra" if is_main else f"Significance Spectra ({folder.title()})"
    fid_rows = fid_rows or []
    fid_specs = fid_specs or []
    significance_specs = significance_specs or []

    optional_sin13 = f"""
## {sin13_title}

---

{render_component_pair_slides(result_specs, 'solar_sin13', 'react_sin13', 'Solar Contour (sin13)', 'Reactor Contour (sin13)')}

---
""" if show_sin13 else ""

    optional_significance = f"""
## {sig_title}

---

{render_significance_slides(significance_specs)}

---
""" if show_significance_spectra else ""

    optional_templates = f"""
## {templates_title}

---
""" if show_templates else ""

    osc_section = ""
    if osc_specs:
        osc_title = "Oscillograms" if folder == "truncated" else f"Oscillograms ({folder.title()})"
        osc_section = f"\n## {osc_title}\n\n---\n\n{render_oscillogram_slides(osc_specs)}\n\n---\n\n"

    return f"""## {fid_title}

---

{render_fiducial_plot_slides(fid_specs)}

---

{render_fid_table(folder, fid_rows)}

---

## {sin12_title}

---

{render_component_pair_slides(result_specs, 'solar_sin12', 'react_sin12', 'Solar Contour (sin12)', 'Reactor Contour (sin12)')}

---
{optional_sin13}{optional_significance}{optional_templates}

{render_template_slides(template_specs)}

---

{render_cut_table(folder, result_specs, sig_rows=sig_rows)}
{osc_section}"""


def _sensitivity_math_slides():
    return """\
### 2D Template Construction

Signal and background are represented as 2D histograms with axes **(reconstructed neutrino energy \xd7 nadir cos(η))**.

For each oscillation point $(\\Delta m^2,\\, \\sin^2\\theta_{13},\\, \\sin^2\\theta_{12})$, the signal template is built by convolving the detector energy-response matrix $H$ with the oscillation-probability matrix $P$:
$$
T^{\\mathrm{sig}}_{ij}(\\vec{\\theta}) = T \\cdot M_{\\mathrm{det}} \\cdot \\left[ P(\\vec{\\theta})\\, H \\right]_{ij}
$$
where $i$ indexes nadir bins and $j$ indexes energy bins ([src/physics/sensitivity/02\_signal\_template.py](../../src/physics/sensitivity/02_signal_template.py)).

The background template $T^{\\mathrm{bkg}}_{ij}$ is independent of oscillation parameters ([src/physics/sensitivity/01\_background\_template.py](../../src/physics/sensitivity/01_background_template.py)).

---

### Asimov Grid Construction

For each oscillation point $\\vec{\\theta}_k$ in the scan grid, the **Asimov (fake) observed dataset** is:
$$
o_{ij}(\\vec{\\theta}_k) = T^{\\mathrm{sig}}_{ij}(\\vec{\\theta}_k) + T^{\\mathrm{bkg}}_{ij}
$$
Same Asimov construction as the HEP profile-likelihood (*Background Normalization Model* slide): no statistical fluctuations, expected sensitivity in the median experiment.

Two **reference templates** are fixed at the solar and reactor best-fit oscillation points:
$$
p^{\\mathrm{solar}}_{ij} = T^{\\mathrm{sig}}_{ij}(\\vec{\\theta}_{\\mathrm{solar}}), \\qquad p^{\\mathrm{react}}_{ij} = T^{\\mathrm{sig}}_{ij}(\\vec{\\theta}_{\\mathrm{react}})
$$

---

### Objective Function (Poisson Deviance)

The fit minimizes a **Baker-Cousins Poisson deviance** ([Baker & Cousins 1984](https://doi.org/10.1016/0029-554X(84)90016-4)) with two free normalization nuisances $A_{\\mathrm{pred}},\\, A_{\\mathrm{bkg}}$:
$$
\\chi^2(A_{\\mathrm{pred}}, A_{\\mathrm{bkg}}) = 2\\sum_{i,j} \\Delta\\ell_{ij} + \\left(\\frac{A_{\\mathrm{pred}}}{\\sigma_{\\mathrm{pred}}}\\right)^{\\!2} + \\left(\\frac{A_{\\mathrm{bkg}}}{\\sigma_{\\mathrm{bkg}}}\\right)^{\\!2}
$$
Expected model: $e_{ij} = (1+A_{\\mathrm{bkg}})\\,T^{\\mathrm{bkg}}_{ij} + (1+A_{\\mathrm{pred}})\\,p_{ij}$.
Per-bin deviance: $\\Delta\\ell_{ij} = e_{ij} - o_{ij} + o_{ij}\\ln(o_{ij}/e_{ij})$ for $o_{ij}>0$, else $\\Delta\\ell_{ij} = e_{ij}$.

Implemented in [`lib/root.py: Sensitivity_Fitter`](../../lib/root.py). Default minimizer: **scipy L-BFGS-B** (joint 2D); ROOT TH2F input uses [iminuit (Minuit)](https://iminuit.readthedocs.io/en/stable/).

---

### Nuisance Parameters: Comparison with HEP

Both analyses use **Gaussian-constrained normalization nuisances** added to the Poisson deviance:

| Feature | HEP Profile-Likelihood | Sensitivity |
|---|---|---|
| Histogram | 1D (energy) | 2D (energy \xd7 nadir) |
| Goal | Discovery significance | $\\chi^2$ map over $(\\Delta m^2, \\sin^2\\theta)$ |
| Nuisances | 1 global $\\beta$ (background) | $A_{\\mathrm{pred}} + A_{\\mathrm{bkg}}$ |
| Solution | Closed-form quadratic ([Cowan 2010](https://arxiv.org/abs/1007.1727)) | scipy L-BFGS-B (joint 2D) |
| Deviance | $2\\sum_i [n_i \\ln(n_i/\\hat{\\beta}b_i) - (n_i - \\hat{\\beta}b_i)]$ | $2\\sum_{ij} [e_{ij} - o_{ij} + o_{ij}\\ln(o_{ij}/e_{ij})]$ |
| Penalty | $[(\\hat{\\beta}-1)/\\sigma_{\\mathrm{rel}}]^2$ | Conditional (see next slide) |
| MC mask | Barlow-Beeston (static) | BB mask: bins with bkg template = 0 excluded |
| Nuisance disable | — | set $\\sigma \\le 0$ to drop that penalty term |

---

### Nuisance Parameter Model

The penalty term is applied **conditionally** based on each nuisance being active ($\\sigma > 0$):

$$
\\mathcal{P}(A_{\\mathrm{pred}}, A_{\\mathrm{bkg}}) = \\begin{cases}
\\left(\\dfrac{A_{\\mathrm{pred}}}{\\sigma_{\\mathrm{pred}}}\\right)^{\\!2} + \\left(\\dfrac{A_{\\mathrm{bkg}}}{\\sigma_{\\mathrm{bkg}}}\\right)^{\\!2} & \\sigma_{\\mathrm{pred}} > 0 \\text{ and } \\sigma_{\\mathrm{bkg}} > 0 \\\\[6pt]
\\left(\\dfrac{A_{\\mathrm{bkg}}}{\\sigma_{\\mathrm{bkg}}}\\right)^{\\!2} & \\sigma_{\\mathrm{pred}} \\le 0 \\text{ and } \\sigma_{\\mathrm{bkg}} > 0 \\\\[6pt]
\\left(\\dfrac{A_{\\mathrm{pred}}}{\\sigma_{\\mathrm{pred}}}\\right)^{\\!2} & \\sigma_{\\mathrm{pred}} > 0 \\text{ and } \\sigma_{\\mathrm{bkg}} \\le 0 \\\\[6pt]
0 & \\text{otherwise}
\\end{cases}
$$

Setting $\\sigma \\le 0$ **disables** that nuisance entirely — the corresponding $A$ is still a free parameter in the minimization but receives no Gaussian pull. Default: $\\sigma_{\\mathrm{pred}} = 4\\%$, $\\sigma_{\\mathrm{bkg}} = 2\\%$.

**Three minimization backends** (selected automatically by input type):

| Backend | Input | Method |
|---|---|---|
| scipy L-BFGS-B | `np.ndarray` | Joint 2D over $(A_{\\mathrm{pred}}, A_{\\mathrm{bkg}})$ — **default** |
| Minuit 1D + profiled bkg | `np.ndarray` + `profile_bkg=True` | 1D Minuit over $A_{\\mathrm{pred}}$; `minimize_scalar` at each step for $A_{\\mathrm{bkg}}$ |
| Minuit 2D | `ROOT.TH2F` | Joint 2D Minuit |

Implemented in [`lib/root.py: Sensitivity_Fitter.NumpyOperator`](../../lib/root.py).

---

### Oscillation Grid Scan and Best-Cut Score

For each analysis cut $(N_{\\mathrm{hits}}, N_{\\mathrm{ophits}}, N_{\\mathrm{adjcl}})$ and each oscillation point $\\vec{\\theta}_k$:
1. Build Asimov dataset $o_{ij}(\\vec{\\theta}_k)$.
2. Fit against **solar** reference template → $\\chi^2_{\\mathrm{solar}}(\\vec{\\theta}_k)$.
3. Fit against **reactor** reference template → $\\chi^2_{\\mathrm{react}}(\\vec{\\theta}_k)$.

The **cut quality score** is the average $\\chi^2$ when fitting with the *wrong* hypothesis:
$$
\\mathrm{Score} = \\tfrac{1}{2}\\left[\\chi^2_{\\mathrm{solar}}(\\vec{\\theta}_{\\mathrm{react}}) + \\chi^2_{\\mathrm{react}}(\\vec{\\theta}_{\\mathrm{solar}})\\right]
$$
Higher score = better discrimination between solar and reactor hypotheses. The best cut maximizes this.

---

### Implemented Improvements

Improvements 2–5 implemented in [lib/root.py](../../lib/root.py) and [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py):

1. **Replace heuristic score with profile-LR** *(proposed, not yet implemented)*: use $\\Delta\\chi^2 = \\chi^2_{\\mathrm{null}} - \\chi^2_{\\mathrm{best}}$ and report $Z = \\sqrt{\\Delta\\chi^2}$ (Wilks theorem) instead of average cross-hypothesis $\\chi^2$.
2. ✅ **Barlow-Beeston mask** (`bb_mask = bkg > 0`): bins where the background template is zero are excluded from the fit, preventing spurious large deviance contributions from zero-MC-support bins.
3. ✅ **Removed `abs()`** from `ROOTOperator` and `NumpyOperator`: the Baker-Cousins deviance is always $\\ge 0$ at the minimum; `abs()` distorts gradients and can impair Minuit convergence.
4. ✅ **Tightened parameter limits**: $\\pm 100\\sigma \\to \\pm 10\\sigma$, reducing search space and avoiding minimization in flat tails.
5. ✅ **Replaced Minuit with scipy L-BFGS-B** (joint 2D): `scipy.optimize.minimize(..., method="L-BFGS-B")` minimizes over $(A_{\\mathrm{pred}}, A_{\\mathrm{bkg}})$ jointly with ±10$\\sigma$ bounds. Uses gradient info → fewer function evaluations than Minuit for smooth convex objectives. `_profile_a_bkg` retained as `profile_bkg=True` option for comparison. No analytic closed form exists for either nuisance (unlike HEP's $\\hat{\\beta}$) due to per-bin coupling.

---

### Sensitivity Fit Summary

- [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py) builds Asimov maps from signal + background templates, then fits each oscillation-grid point against solar and reactor reference templates with free normalizations.
- The fit minimizes the **Baker-Cousins Poisson deviance** ([Baker & Cousins 1984](https://doi.org/10.1016/0029-554X(84)90016-4)) — identical in form to the per-bin LLR in the HEP profile-likelihood, extended to 2D (energy × nadir).
- Penalty terms are conditional on $\\sigma > 0$; both active by default ($\\sigma_{\\mathrm{pred}}=4\\%$, $\\sigma_{\\mathrm{bkg}}=2\\%$). Set $\\sigma \\le 0$ to disable.
- Improvements 2–5 implemented in [lib/root.py](../../lib/root.py): BB mask, no `abs()`, ±10σ limits, scipy L-BFGS-B joint 2D minimization.
- Full mathematical derivations: [docs/hep\\_likelihood\\_derivation.tex](../../docs/hep_likelihood_derivation.tex)."""


def build_markdown(energy, folder, folder_specs, folder_templates, fid_rows=None, fid_specs=None, significance_specs=None, sig_rows=None, comparison_profiles_specs=None, default_profile=None, nuisance_profiles=None, show_sin13=False, show_templates=False, show_significance_spectra=False, osc_specs=None):
    alias_bullets = "\n".join([f"- {config}: {alias}" for config, alias in STANDARD_CONFIGS])
    energy_label = output_energy_label(energy)
    coverage = sum(1 for item in folder_specs if item.get("primary"))
    selected_title = folder.title()
    selected_sections = render_folder_sections(
        folder, folder_specs, folder_templates,
        fid_rows=fid_rows, fid_specs=fid_specs, significance_specs=significance_specs, sig_rows=sig_rows,
        show_sin13=show_sin13, show_templates=show_templates, show_significance_spectra=show_significance_spectra,
        osc_specs=osc_specs,
    )
    nuisance_section = render_nuisance_comparison_section(
        default_profile or "full",
        folder_specs,
        comparison_profiles_specs or {},
        nuisance_profiles=nuisance_profiles or {},
    )

    text = textwrap.dedent(
        f"""
    ---
    marp: true
    description: Inputs, workflow outputs, and per-config Sensitivity results
    paginate: true
    theme: dune
    ---

    <!-- AUTO-GENERATED: src/tools/presentations/sensitivity.py -->

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

    - Orchestrator: [src/pipelines/run\_sensitivity.py](../../src/pipelines/run_sensitivity.py)
    - Step 1 (Background template): [src/physics/sensitivity/01\_background\_template.py](../../src/physics/sensitivity/01_background_template.py)
    - Step 2 (Signal template): [src/physics/sensitivity/02\_signal\_template.py](../../src/physics/sensitivity/02_signal_template.py)
    - Step 3 (Grid fit scan and best-cut storage): [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py)
    - Step 4 (Contour rendering): [src/physics/sensitivity/contour\_plot.py](../../src/physics/sensitivity/contour_plot.py)

    ---

    ### Workflow Outputs

    - Main contour grids (sin12/sin13): [images/analysis/sensitivity](../../images/analysis/sensitivity)
    - Signal/background templates (figures): [images/analysis/sensitivity/templates](../../images/analysis/sensitivity/templates)
    - Grid-scan data products (PKL): [data/analysis/sensitivity](../../data/analysis/sensitivity)
    - Remote workflow outputs (PNFS): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY)

    ---

{_sensitivity_math_slides()}

---

{selected_sections}

---

{nuisance_section}

{"---" if nuisance_section else ""}

## Coverage and Notes

- Configs with selected sin12 solar contour plots:
    - {folder}: {coverage}
- Cut table values are parsed from selected result filenames when available.
- Re-run script to refresh this folder after each workflow run:
    - /usr/bin/python3 scripts/generate_sensitivity_presentation.py --folder {folder}
- Full mathematical derivations: [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex)
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

    # Load nuisance profile config from analysis/config.json
    analysis_json = ROOT / "analysis" / "config.json"
    _analysis_info = json.loads(analysis_json.read_text()) if analysis_json.exists() else {}
    nuisance_profiles = _analysis_info.get("NUISANCE_PROFILES", {})
    default_profile = _analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")

    # Load metrics config (controls which sections to gather/render)
    _workflow_section = _analysis_info.get("WORKFLOW", {}).get("SENSITIVITY", {})
    _metrics = _workflow_section.get("METRICS", {})
    _do_sin13             = args.all_metrics or _metrics.get("sin13", False)
    _do_templates         = args.all_metrics or _metrics.get("templates", False)
    _do_significance_spec = args.all_metrics or _metrics.get("significance_spectra", False)
    _do_nuisance_cmp      = args.all_metrics or _metrics.get("nuisance_comparison", False)

    # Gather main specs: prefer profile subdir, fall back to flat folder
    folder_specs = gather_result_specs(args.folder, args.energy, profile=default_profile)
    if not any(s.get("primary") for s in folder_specs):
        folder_specs = gather_result_specs(args.folder, args.energy)
        default_profile = None

    folder_templates = gather_template_specs(args.folder, args.energy, folder_specs) if _do_templates else {"background": {}, "signal": {}}
    fid_rows  = gather_fiducial_rows(args.energy)
    fid_specs = gather_fiducial_plot_specs(args.folder, args.energy)
    significance_specs = gather_significance_specs(args.folder, args.energy) if _do_significance_spec else []
    sig_rows = gather_sensitivity_significance_rows(args.folder, args.energy)

    # Comparison: only gather when enabled (each profile requires filesystem globs)
    comparison_profiles_specs = {}
    if _do_nuisance_cmp and default_profile and nuisance_profiles:
        for pname in nuisance_profiles:
            if pname == default_profile:
                continue
            pspecs = gather_result_specs(args.folder, args.energy, profile=pname)
            if any(s.get("primary") for s in pspecs):
                comparison_profiles_specs[pname] = pspecs

    osc_specs = gather_oscillogram_specs(args.folder, args.energy, "Sensitivity")

    markdown = build_markdown(
        args.energy,
        args.folder,
        folder_specs,
        folder_templates,
        fid_rows=fid_rows,
        fid_specs=fid_specs,
        significance_specs=significance_specs,
        sig_rows=sig_rows,
        comparison_profiles_specs=comparison_profiles_specs,
        default_profile=default_profile,
        nuisance_profiles=nuisance_profiles,
        show_sin13=_do_sin13,
        show_templates=_do_templates,
        show_significance_spectra=_do_significance_spec,
        osc_specs=osc_specs,
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