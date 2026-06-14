import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import re
import textwrap
from glob import glob
from pathlib import Path

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

PNFS_HEP = Path("/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP")
LOCAL_JSON = ROOT / "config"
DEFAULT_REFERENCE = "ProfileLikelihood"


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


def _fmt_components(mode, components):
    mode_key = str(mode or "all").strip().lower()
    values = [str(item) for item in (components or []) if str(item).strip()]
    if mode_key in {"all", "any", "*"}:
        return "All components"
    if mode_key in {"only", "include", "includes", "selected", "whitelist"}:
        return ", ".join(values) if values else "(none selected)"
    if mode_key in {"exclude", "except", "blacklist"}:
        return f"All except: {', '.join(values)}" if values else "All components"
    return ", ".join(values) if values else "All components"


def gather_hep_threshold_rows():
    path = ROOT / "analysis" / "config.json"
    if not path.exists():
        return []
    payload = read_json(path)
    hep = payload.get("HEP", {})
    thresholds = hep.get("THRESHOLDS", {})
    rows = []
    for key in sorted(thresholds.keys()):
        threshold = thresholds.get(key, {})
        rows.append(
            {
                "Config": key,
                "Energy": threshold.get("energy"),
                "Threshold": threshold.get("threshold"),
            }
        )
    return rows

def gather_hep_smoothing_stage_rows():
    analysis_path = ROOT / "analysis" / "smoothing.json"
    if not analysis_path.exists():
        return []

    payload = read_json(analysis_path)
    smoothing = payload.get("SMOOTHING", {})
    hep = smoothing.get("ANALYSES", {}).get("HEP", {})
    stages = hep.get("STAGES", {})
    default_enabled = bool(hep.get("enabled", smoothing.get("enabled", False)))
    default_method = str(hep.get("method", smoothing.get("method", "none")))

    rows = []
    for stage_key in sorted(stages.keys()):
        stage = stages.get(stage_key, {})
        stage_enabled = bool(stage.get("enabled", default_enabled))
        stage_method = str(stage.get("method", default_method))
        dims = stage.get("dimensions", {})
        sigma = "-"
        if isinstance(dims, dict):
            if "1d" in dims and isinstance(dims.get("1d"), dict):
                sigma_val = dims.get("1d", {}).get("sigma")
                sigma = f"{float(sigma_val):.2f}" if sigma_val is not None else "-"
            elif "2d" in dims and isinstance(dims.get("2d"), dict):
                sigma_x = dims.get("2d", {}).get("sigma_x")
                sigma_y = dims.get("2d", {}).get("sigma_y")
                if sigma_x is not None or sigma_y is not None:
                    sx = f"{float(sigma_x):.2f}" if sigma_x is not None else "-"
                    sy = f"{float(sigma_y):.2f}" if sigma_y is not None else "-"
                    sigma = f"x={sx}, y={sy}"

        rows.append(
            {
                "Stage": stage_key.title(),
                "Enabled": "Yes" if stage_enabled else "No",
                "Method": stage_method,
                "Mode": str(stage.get("component_mode", "all")),
                "Components": _fmt_components(
                    stage.get("component_mode", "all"),
                    stage.get("components", []),
                ),
                "Sigma": sigma,
            }
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HEP MARP presentation from workflow outputs"
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
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
        choices=["Asimov", "Gaussian", "ProfileLikelihood"],
        help="Significance reference used to pick HEP exposure/significance plots",
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
        return ROOT / "output" / "presentations" / f"{folder_label}HEPSignificanceWorkflow.md"
    return ROOT / "output" / "presentations" / f"{energy}{folder_label}HEPSignificanceWorkflow.md"


def _find_latest(base_dir, patterns, exclude=None):
    if not base_dir.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(base_dir.glob(pattern))
    if exclude:
        candidates = [c for c in candidates if not any(ex in c.name for ex in exclude)]
    return pick_most_recent(candidates)


def gather_best_sigma_rows(energy):
    rows = {"nominal": [], "reduced": [], "truncated": []}
    row_map = {}
    patterns = [
        str(PNFS_HEP / "*" / "*" / "marley" / "*_highest_HEP.json"),
        str(LOCAL_JSON / "*" / "hep-json" / "*" / "*" / "*_highest_HEP.json"),
    ]

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

                selected_key = None
                for candidate in energy_candidates(energy):
                    if candidate in marley:
                        selected_key = candidate
                        break
                if selected_key is None:
                    continue

                vals = marley[selected_key]
                row_key = (folder, cfg, selected_key)
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
            hep = analyses.get("HEP", {})
            if not hep:
                continue

            selected_key = None
            for candidate in energy_candidates(energy):
                if candidate in hep:
                    selected_key = candidate
                    break
            if selected_key is None:
                continue

            vals = hep[selected_key]
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


def gather_hep_plot_specs(folder, energy, reference):
    plot_dir = ROOT / "output" / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    significance_refs = [reference, "ProfileLikelihood", "Asimov", "Gaussian"]
    significance_refs = list(dict.fromkeys(significance_refs))
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        canonical_dir = plot_dir / config_key / "marley" / folder
        expected_significance = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Significance_{ref}_Exposure_*.png"
                for ref in significance_refs
            ],
        )
        expected_exposure = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Exposure_{ref}_Threshold_*.png"
                for ref in significance_refs
            ] + [
                f"{config_key}_marley_{energy_label}_HEP_Exposure_{ref}_*.png"
                for ref in significance_refs
            ],
            exclude=["highest_spiked"],
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


def gather_reference_comparison_specs(folder, energy):
    plot_dir = ROOT / "output" / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        canonical_dir = plot_dir / config_key / "marley" / folder
        expected_significance = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Significance_Comparison_Exposure_*_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_HEP_Significance_Comparison_Exposure_*.png",
            ],
        )
        expected_exposure = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Exposure_Comparison_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_HEP_Exposure_Comparison_*.png",
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


def gather_adaptive_rebin_specs(folder, energy):
    plot_dir = ROOT / "output" / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        canonical_dir = plot_dir / config_key / "marley" / folder
        expected_asimov = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison*.png",
            ],
        )
        expected_gaussian = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison*.png",
            ],
        )
        expected_profile = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_*.png",
                f"{config_key}_marley_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison*.png",
            ],
        )
        slides.append(
            {
                "name": display_name,
                "config": config_key,
                "folder": folder,
                "asimov": expected_asimov.relative_to(ROOT).as_posix()
                if expected_asimov is not None
                else None,
                "gaussian": expected_gaussian.relative_to(ROOT).as_posix()
                if expected_gaussian is not None
                else None,
                "profile": expected_profile.relative_to(ROOT).as_posix()
                if expected_profile is not None
                else None,
            }
        )

    return slides


def gather_spiked_debug_specs(folder, energy, reference):
    """Gather exposure and significance plots generated with --pkl_label highest_spiked for debug slides."""
    plot_dir = ROOT / "output" / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    refs = list(dict.fromkeys([reference, "ProfileLikelihood", "Asimov", "Gaussian"]))
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        canonical_dir = plot_dir / config_key / "marley" / folder
        expected_exposure = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Exposure_{ref}_*highest_spiked*.png"
                for ref in refs
            ],
        )
        expected_significance = _find_latest(
            canonical_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_Significance_{ref}_*highest_spiked*.png"
                for ref in refs
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
        f"{config_key}_marley_{energy_label}_HEP_{label}Fiducial_Significance*.png",
        f"{config_key}_{energy_label}_HEP_{label}Fiducial_Significance*.png",
    ]
    candidates = []
    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(base_dir.glob(pattern))
    expected = pick_most_recent(candidates)
    if expected is not None:
        return expected.relative_to(ROOT).as_posix()
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
    title = "Best HEP Cuts by Config" if folder == "truncated" else f"Best HEP Cuts by Config ({folder.title()})"
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
        lines.append("| *(no HEP entries found)* | - | - | - | - | - | - |")
        return "\n".join(lines)

    for row in filtered_rows:
        lines.append(
            "| "
            + f"{config_alias(row['Config'])} | {fmt_int(row['FidX'])} | {fmt_int(row['FidY'])} | {fmt_int(row['FidZ'])} | {fmt_float(row['BeforeFid'])} | {fmt_float(row['AfterFid'])} | {fmt_float(row.get('Exposure'), digits=2)} |"
        )

    return "\n".join(lines)


def render_smoothing_stage_table(rows):
    lines = [
        "### Smoothing by Stage (HEP)",
        "",
        "| Stage | Enabled | Method | Component Mode | Smoothed Components | Sigma |",
        "|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| *(no HEP smoothing stage config found)* | - | - | - | - | - |")
        return "\n".join(lines)

    for row in rows:
        lines.append(
            "| "
            + f"{row['Stage']} | {row['Enabled']} | {row['Method']} | {row['Mode']} | {row['Components']} | {row['Sigma']} |"
        )
    return "\n".join(lines)


def render_hep_plot_slides(plot_specs):
    def _hep_lower_panel_note(significance_path):
        if not significance_path:
            return "Lower panel note: not available (significance plot missing)."
        if "BottomRigorous" in significance_path:
            return (
                "local discovery significance per adaptive energy bin, "
                "computed as z_local = sqrt(q0), with q0 = -2 ln(L(mu=0)/L(mu_hat)). "
                # "Dotted black = raw z_local; blue bars = smoothed z_local after adaptive rebinning; "
                "horizontal error bars show adaptive-bin widths in reconstructed energy."
            )
        return (
            "local discovery density, estimated as z_local / DeltaE (sigma per MeV), "
            "where z_local comes from the per-bin discovery test statistic. "
            # "Dotted black = raw; solid color = smoothed; translucent blue bars = adaptive-bin density estimate. "
            "Compare where discovery is concentrated."
            # "and how smoothing/rebinning changes threshold-region behavior."
        )

    slides = []
    for spec in plot_specs:
        if spec["exposure"] or spec["significance"]:
            significance_block = (
                f"    <img src=\"../../{spec['significance']}\">"
                if spec["significance"]
                else "    <p>Significance plot not available.</p>"
            )
            exposure_block = (
                f"    <img src=\"../../{spec['exposure']}\">"
                if spec["exposure"]
                else "    <p>Exposure plot not available.</p>"
            )
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        "    <p><strong>Significance</strong></p>",
                        significance_block,
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Exposure</strong></p>",
                        exposure_block,
                        "  </div>",
                        "</div>",
                        "",
                        "<div class=\"comparison-note\">",
                        f"  <strong>Lower subplot guide:</strong> {_hep_lower_panel_note(spec.get('significance'))}",
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
                        f"No matching HEP significance or exposure plot found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        slides.append("### Plot outputs\n\nNo HEP plot PNGs were found in output/images/analysis/hep.")

    return "\n\n---\n\n".join(slides)


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


def render_reference_comparison_slides(specs):
    slides = []
    for spec in specs:
        if spec["exposure"] and spec["significance"]:
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        "    <p><strong>Significance Reference Comparison</strong></p>",
                        f"    <img src=\"../../{spec['significance']}\">",
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Exposure Reference Comparison</strong></p>",
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
                        f"No matching reference-comparison pair found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        slides.append("### Reference comparisons\n\nNo HEP reference-comparison PNGs were found.")

    return "\n\n---\n\n".join(slides)


def render_adaptive_rebin_slides(specs):
    def _extract_threshold_text(spec):
        for key in ["asimov", "gaussian", "profile"]:
            path = spec.get(key)
            if not path:
                continue
            match = re.search(r"_Threshold_([0-9]+(?:\.[0-9]+)?)", path)
            if match:
                return match.group(1)
        return "configured"

    slides = []
    for spec in specs:
        available_blocks = []
        if spec.get("gaussian"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>Gaussian</strong></p>",
                        f"  <img src=\"../../{spec['gaussian']}\">",
                        "</div>",
                    ]
                )
            )
        if spec.get("asimov"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>Asimov</strong></p>",
                        f"  <img src=\"../../{spec['asimov']}\">",
                        "</div>",
                    ]
                )
            )
        if spec.get("profile"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>ProfileLikelihood</strong></p>",
                        f"  <img src=\"../../{spec['profile']}\">",
                        "</div>",
                    ]
                )
            )

        if available_blocks:
            threshold_text = _extract_threshold_text(spec)
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"three-col\">",
                        "\n".join([f"  {block}" for block in available_blocks]),
                        "</div>",
                        "",
                        "<div class=\"comparison-note\">",
                        "  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.",
                        f"  Compare the exposure turn-on point near threshold {threshold_text} MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.",
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
                        f"No matching adaptive-rebin comparison set found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        slides.append("### Adaptive rebin comparisons\n\nNo HEP adaptive-rebin comparison PNGs were found.")

    return "\n\n---\n\n".join(slides)


def render_spiked_debug_slides(specs):
    """Render debug slides for the best spiked curves excluded from the main selection."""
    any_available = any(spec.get("exposure") or spec.get("significance") for spec in specs)
    if not any_available:
        return "### Spike Debug\n\nNo spiked-cut plots found. Run workflow with `--pkl_label highest_spiked` to generate them."

    slides = []
    for spec in specs:
        if spec.get("exposure") or spec.get("significance"):
            exposure_block = (
                f"    <img src=\"../../{spec['exposure']}\">"
                if spec.get("exposure")
                else "    <p>Exposure plot not available.</p>"
            )
            significance_block = (
                f"    <img src=\"../../{spec['significance']}\">"
                if spec.get("significance")
                else "    <p>Significance plot not available.</p>"
            )
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']} — best excluded (spiked)",
                        "",
                        "<div class=\"comparison-note\">",
                        "  <strong>Debug:</strong> Highest-significance cut <em>excluded</em> from main selection due to a spike in the pre-isotonic PL curve. Compare against the main result to assess the impact of the filter.",
                        "</div>",
                        "",
                        "<div class=\"two-col\">",
                        "  <div>",
                        "    <p><strong>Significance</strong></p>",
                        significance_block,
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Exposure</strong></p>",
                        exposure_block,
                        "  </div>",
                        "</div>",
                    ]
                )
            )
        else:
            slides.append(
                f"### {spec['name']} — best excluded (spiked)\n\nNo spiked plots found."
            )

    return "\n\n---\n\n".join(slides)


def render_folder_sections(
    folder,
    fid_specs,
    hep_specs,
    reference_specs,
    adaptive_specs,
    best_sigma_rows,
    fid_rows,
    spiked_specs=None,
    osc_specs=None,
):
    def _bottom_variant(path, target_variant):
        if not path:
            return None
        rel = Path(path)
        abs_path = ROOT / rel
        if not abs_path.exists():
            return path

        # Prefer direct suffix replacement when a Bottom* style is already present.
        replaced_name = re.sub(r"_Bottom[^_.]+", f"_Bottom{target_variant}", rel.name)
        if replaced_name != rel.name:
            candidate = rel.with_name(replaced_name)
            if (ROOT / candidate).exists():
                return candidate.as_posix()

        # Fallback: pick any matching Bottom* variant with same stem prefix.
        stem_prefix = rel.stem.split("_Bottom")[0]
        candidates = sorted((ROOT / rel.parent).glob(f"{stem_prefix}_Bottom{target_variant}.png"))
        if candidates:
            return candidates[-1].relative_to(ROOT).as_posix()

        return path

    hep_by_config = {spec.get("config"): spec for spec in (hep_specs or [])}
    hep_specs_with_intuitive_significance = []
    for spec in hep_specs:
        merged = dict(spec)
        merged["significance"] = _bottom_variant(spec.get("significance"), "Intuitive")
        hep_specs_with_intuitive_significance.append(merged)

    reference_specs_with_rigorous_significance = []
    for spec in reference_specs:
        merged = dict(spec)
        hep_spec = hep_by_config.get(spec.get("config"))
        if hep_spec and hep_spec.get("significance"):
            merged["significance"] = _bottom_variant(hep_spec.get("significance"), "Rigorous")
        else:
            merged["significance"] = _bottom_variant(spec.get("significance"), "Rigorous")
        reference_specs_with_rigorous_significance.append(merged)

    is_main = folder == "truncated"
    fid_title = "Fiducialization" if is_main else f"Fiducialization ({folder.title()})"
    hep_title = "HEP Results" if is_main else f"HEP Results ({folder.title()})"
    reference_title = "Reference Comparison" if is_main else f"Reference Comparison ({folder.title()})"
    adaptive_title = "Adaptive Rebin Comparison" if is_main else f"Adaptive Rebin Comparison ({folder.title()})"
    osc_title = "Oscillograms" if is_main else f"Oscillograms ({folder.title()})"
    osc_section = ""
    if osc_specs:
        osc_section = f"## {osc_title}\n\n---\n\n{render_oscillogram_slides(osc_specs)}\n\n---\n\n"
    return f"""## {fid_title}

---

{render_fiducial_plot_slides(fid_specs)}

---

{render_fid_table(folder, fid_rows)}

---

## {hep_title}

---

{render_hep_plot_slides(hep_specs_with_intuitive_significance)}

---

## {reference_title}

---

{render_reference_comparison_slides(reference_specs_with_rigorous_significance)}

---

## {adaptive_title}

---

{render_adaptive_rebin_slides(adaptive_specs)}

---

{render_sigma_table(folder, best_sigma_rows.get(folder, []))}

{_render_spike_debug_section(spiked_specs)}
{osc_section}"""


def _render_spike_debug_section(spiked_specs):
    if not spiked_specs:
        return ""
    any_available = any(spec.get("exposure") for spec in spiked_specs)
    title = "Spike Debug (excluded from selection)"
    return f"---\n\n## {title}\n\n---\n\n{render_spiked_debug_slides(spiked_specs)}\n"


def build_markdown(
    folder,
    threshold_rows,
    best_sigma_rows,
    fid_rows,
    hep_specs,
    reference_specs,
    adaptive_specs,
    fid_specs,
    reference,
    smoothing_stage_rows,
    spiked_specs=None,
    osc_specs=None,
):
    coverage = {folder: len(rows) for folder, rows in best_sigma_rows.items()}
    alias_bullets = "\n".join([f"- {config}: {alias}" for config, alias in STANDARD_CONFIGS])
    selected_sections = render_folder_sections(
        folder,
        fid_specs,
        hep_specs,
        reference_specs,
        adaptive_specs,
        best_sigma_rows,
        fid_rows,
        spiked_specs=spiked_specs,
        osc_specs=osc_specs,
    )
    selected_title = folder.title()

    text = textwrap.dedent(
        f"""
    ---
    marp: true
    math: katex
    description: Inputs, workflow outputs, and per-config HEP results
    paginate: true
    theme: dune
    ---

    <!-- AUTO-GENERATED: scripts/generate_hep_presentation.py -->

    <!-- _class: titlepage -->

    # HEP Significance Workflow

    ---

    ## Introduction

    This presentation summarizes the workflow and outputs of the HEP significance analysis for the SOLAR project.
    - This deck is auto-generated from workflow outputs.
    - This deck is scoped to the **{selected_title}** folder for the **SolarEnergy** reconstruction algorithm.

    Config aliases:
    {alias_bullets}

    ---

    ### Workflow

    - config: list of detector configs
    - folder: **{selected_title}**
    - analysis: HEP
    - exposure: default **30 years**
    - threshold in hep/01_hep.py: from [config/analysis/config.json](../../config/analysis/config.json) HEP -> THRESHOLDS -> {threshold_rows[folder] if folder in threshold_rows else "(no threshold config found)"}
    - optional cuts override: nhits, ophits, adjcls
    - significance reference in plots: {reference}
    - best-cut selection in sensitivity/05_best_sigmas.py: **ProfileLikelihood** (smoothed, 3σ crossing)

    ---

    ### Workflow Skip Flags

    Used for [src/pipelines/run_sensitivity.py](../../src/pipelines/run_sensitivity.py):
    - `--no-computation`: skip all analysis, run plot macros only
    - `--no-significance`: skip 01_hep.py/01_daynight.py/06_significance.py only
    - `--no-fiducialization`: skip signal/01_fiducialize.py only
    - `--no-rebin`: skip signal/03_analysis.py rebinning step only

    ---

    ### Workflow Outputs

    - Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../../data/solar/fiducial/truncated/BestFiducials.json)
    - Best cut summaries (JSON): [data/analysis/hep-json/truncated](../../data/analysis/hep-json/truncated)
    - Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated)
    - Figures: [output/images/analysis/hep/truncated](../../output/images/analysis/hep/truncated)

    ---

    ### Histogram and Significance Flow I: Building, Smoothing, and Evaluation

    - Step 1: Build HEP rates and threshold region in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) per component.
    - Step 2: Apply component-aware smoothing via [lib/smoothing.py](../../lib/smoothing.py) using HEP smoothing config.
    - Step 3: Evaluate Gaussian, Asimov, and ProfileLikelihood significance curves in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) for **all** analysis cuts. ProfileLikelihood uses a single global background normalization nuisance profiled jointly across all bins (see *Background Normalization Model* slide). Background bins with fewer than `min_mc_per_bin` raw MC events are masked using the [Barlow-Beeston lite criterion](https://www.sciencedirect.com/science/article/pii/009350659390005W) (as implemented in [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html)) to suppress LLR divergence from empty bins. Smoothed histogram rates are clipped to ≥ 0 before the PL step to prevent negative-rate blowup at high exposures.
    
    ---

    ### Histogram and Significance Flow II: Post-Processing and Plotting

    - Step 4: Select the best cut by ProfileLikelihood in [src/physics/sensitivity/05_best_sigmas.py](../../src/physics/sensitivity/05_best_sigmas.py). Cuts whose PL curve contains a single-step jump exceeding `max_pl_jump` σ in either the raw or smoothed pre-isotonic column are flagged as spiked, excluded from the main `highest` selection, and saved separately as `highest_spiked` for inspection.
    - Step 5: Render exposure/significance and comparison plots in [src/physics/hep/exposure_plot.py](../../src/physics/hep/exposure_plot.py), [src/physics/hep/significance_plot.py](../../src/physics/hep/significance_plot.py), [src/physics/hep/significance_comparison.py](../../src/physics/hep/significance_comparison.py), and [src/physics/hep/exposure_comparison.py](../../src/physics/hep/exposure_comparison.py).

    ---

    ### Background Normalization Model

    The background systematic is a **single global scale factor** β — not independent per-bin nuisances.
    Significance is computed from $q_0 = -2\ln\lambda(\mu=0)$ following [Cowan et al. 2010 (arXiv:1007.1727)](https://arxiv.org/abs/1007.1727) — the standard ATLAS/CMS formulation for discovery tests.

    β satisfies the closed-form quadratic (total counts $N = \sum n_i$, $B = \sum b_i$, $\sigma^2 = \sigma_\mathrm{{rel}}^2$):
    $$
    \hat{{\\beta}}^2 + (B\sigma^2 - 1)\hat{{\\beta}} - N\sigma^2 = 0
    $$

    **Why not per-bin nuisances?** With $k$ independent per-bin β's each with a 2% Gaussian constraint, the null hypothesis has $k$ dials to absorb signal bin-by-bin. This creates an artificial flat region in significance vs exposure that ends in a sharp kink when the absorbing capacity is exhausted (at $f \\sim 1/(b_r \\cdot \\sigma^2)$ per bin — typically around 20 years for shielded configs with tight OpHits cuts). A global β has its absorbing regime end at $f \\sim 1/(B_\\mathrm{{total}} \\cdot \\sigma^2)$ (typically sub-year), giving a smooth, physically correct PL curve throughout the analysis window.

    ---

    ### ProfileLikelihood Smoothing

    PL curves are post-processed with **Gaussian kernel smoothing followed by isotonic regression** to produce a continuous, monotone exposure curve:
      1. [`scipy.ndimage.gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) convolves the raw PL significance array with a Gaussian kernel (σ = 6 exposure-grid index units, tunable via `_PL_SMOOTH_SIGMA` in [`src/physics/hep/01_hep.py`](../../src/physics/hep/01_hep.py)). This mirrors the approach used by [ROOT `TH1::Smooth`](https://root.cern.ch/doc/master/classTH1.html#a16) for smoothing discrete numerical histograms.
      2. [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) is then applied to enforce strict monotonicity. It finds the non-decreasing sequence that minimises the L2 distance from the smoothed values, ensuring more data cannot reduce sensitivity.

    These steps remove residual numerical oscillations from the profile-likelihood solver at low signal-to-background ratios.

    ---

    ### ProfileLikelihood Error Bands

    The ±1σ bands on PL exposure curves use **signal normalization variation**: signal events are scaled by $(1 \pm \sigma_s)$ where $\sigma_s$ is the signal reconstruction efficiency systematic (`--signal_uncertainty`, typically 10%):
    $$
    s_i^{{\pm}} = s_i \cdot (1 \pm \sigma_s)
    $$

    The background is **never shifted**, so the profiled nuisance $\hat{{\\beta}}$ is unaffected. Both bands collapse symmetrically when signal is negligible ($\pm\sigma_s \cdot 0 = 0$). This avoids the asymmetric-collapse artifact of background-shifting approaches, where $\hat{{\\beta}}_{{null}}$ pull contributions drive the upper band non-zero independent of signal strength.

    - **Upper band** ($\delta = +1$): more signal → higher $q_0$ → easier discovery.
    - **Lower band** ($\delta = -1$): less signal → lower $q_0$. Both bands collapse symmetrically for configurations where signal is negligible.

    ---

    {render_smoothing_stage_table(smoothing_stage_rows)}

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
    - /usr/bin/python3 scripts/generate_hep_presentation.py --folder {folder}
- Full mathematical derivations (signal model, PL formulation, adaptive rebinning, BB mask, spike detection): [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex)

---

### Adaptive Rebinning: Strategy

- Rebinning is applied in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) through [lib/smoothing.py](../../lib/smoothing.py) using `apply_adaptive_tail_rebin`.
- It is controlled by [config/analysis/config.json](../../config/analysis/config.json) under `ADAPTIVE_REBIN -> ANALYSES -> HEP`.
- At each exposure, bins are merged from the high-energy tail until the expected detectable signal per rebinned group reaches the configured threshold.
- This stabilizes low-statistics significance estimates while preserving discovery sensitivity.

Rebin threshold criterion used for each grouped bin:
$$
S_{{\mathrm{{group}}}} = \sum_{{i \in \mathrm{{group}}}} S_i^{{\mathrm{{det}}}} \ge T
$$
$$
T = \max\!\left(\\texttt{{min\_expected\_events}},\,-\ln(1-\\texttt{{min\_count\_probability}})\\right)
$$

---

### Adaptive Rebinning: Discovery

Discovery significance is then evaluated on rebinned inputs:
$$
Z = Z\!\left(S_{{\mathrm{{group}}}},\,B_{{\mathrm{{group}}}},\,\sigma_{{B,\mathrm{{group}}}}\\right)
$$

ProfileLikelihood implementation in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py):
- PL is computed for **every** analysis cut combination.
- Original fine binning used throughout — no adaptive rebin. PL is optimal at the finest resolution; the likelihood ratio naturally suppresses bins with negligible signal without merging.
- A **single global background normalization nuisance** (β ~ Gaussian(1, σ_rel)) is profiled jointly across all bins. See the *Background Normalization Model* slide.

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
    threshold_rows = gather_hep_threshold_rows()
    best_sigma_rows = gather_best_sigma_rows(args.energy)
    fid_rows = gather_fiducial_rows(args.energy)
    selected_hep_specs = gather_hep_plot_specs(args.folder, args.energy, args.reference)
    selected_reference_specs = gather_reference_comparison_specs(args.folder, args.energy)
    selected_adaptive_specs = gather_adaptive_rebin_specs(args.folder, args.energy)
    selected_fid_specs = gather_fiducial_plot_specs(args.folder, args.energy)
    selected_spiked_specs = gather_spiked_debug_specs(args.folder, args.energy, args.reference)
    smoothing_stage_rows = gather_hep_smoothing_stage_rows()
    selected_osc_specs = gather_oscillogram_specs(args.folder, args.energy, "HEP")

    markdown = build_markdown(
        args.folder,
        threshold_rows,
        best_sigma_rows,
        fid_rows,
        selected_hep_specs,
        selected_reference_specs,
        selected_adaptive_specs,
        selected_fid_specs,
        args.reference,
        smoothing_stage_rows,
        spiked_specs=selected_spiked_specs,
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
