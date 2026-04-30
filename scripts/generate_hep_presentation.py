import argparse
import json
import re
import textwrap
from glob import glob
from pathlib import Path

from presentation_common import default_pdf_export_enabled, export_marp_pdf

ROOT = Path("/pc/choozdsk01/users/manthey/SOLAR")
PNFS_HEP = Path("/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP")
# 0ZBestSigmas currently writes JSON summaries under daynight-json for all analyses.
LOCAL_JSON = ROOT / "data" / "analysis" / "daynight-json"
DEFAULT_ENERGY = "SolarEnergy"
DEFAULT_REFERENCE = "ProfileLikelihood"

STANDARD_CONFIGS = [
    ("hd_1x2x6_centralAPA", "HD Central"),
    ("hd_1x2x6_lateralAPA", "HD Lateral"),
    ("vd_1x8x14_3view_30deg_nominal", "VD Top"),
    ("vd_1x8x14_3view_30deg_shielded", "VD Bottom Shielded"),
]
CONFIG_ALIAS_MAP = {config: alias for config, alias in STANDARD_CONFIGS}


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


def gather_hep_smoothing_stage_rows():
    analysis_path = ROOT / "import" / "analysis.json"
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


def config_alias(config_name):
    return CONFIG_ALIAS_MAP.get(config_name, config_name)


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
        return ROOT / "presentations" / f"{folder_label}HEPSignificanceWorkflow.md"
    return ROOT / "presentations" / f"{energy}{folder_label}HEPSignificanceWorkflow.md"


def energy_candidates(energy):
    if energy == DEFAULT_ENERGY:
        return ["SolarEnergy", "Solar"]
    return [energy]


def output_energy_label(energy):
    return "SolarEnergy" if energy == DEFAULT_ENERGY else energy


def _pick_most_recent(paths):
    existing = [path for path in paths if path is not None and path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _find_latest(base_dir, patterns):
    if not base_dir.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(base_dir.glob(pattern))
    return _pick_most_recent(candidates)


def gather_best_sigma_rows(energy):
    rows = {"nominal": [], "reduced": [], "truncated": []}
    row_map = {}
    patterns = [
        str(PNFS_HEP / "*" / "*" / "marley" / "*_highest_HEP.json"),
        str(LOCAL_JSON / "*" / "*" / "marley" / "*_highest_HEP.json"),
    ]

    for pattern in patterns:
        for json_path in sorted(glob(pattern)):
            p = Path(json_path)
            folder = p.parts[-4]
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
                }
            )

    return sorted(fid_rows, key=lambda row: (row["Folder"], row["Config"]))


def gather_hep_plot_specs(folder, energy, reference):
    plot_dir = ROOT / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    significance_refs = [reference, "ProfileLikelihood", "Asimov", "Gaussian"]
    significance_refs = list(dict.fromkeys(significance_refs))
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        search_dirs = [
            plot_dir / folder / config_key,
            plot_dir / config_key / folder,
            plot_dir / config_key,
        ]

        significance_candidates = []
        for this_ref in significance_refs:
            for base_dir in search_dirs:
                significance_candidates.extend(
                    [
                        _find_latest(
                            base_dir,
                            [
                                f"{config_key}_{energy_label}_HEP_Significance_{this_ref}_Exposure_*.png"
                            ],
                        ),
                        _find_latest(
                            base_dir / "marley",
                            [
                                f"{config_key}_marley_{energy_label}_HEP_Significance_{this_ref}_Exposure_*.png"
                            ],
                        ),
                    ]
                )
        expected_significance = _pick_most_recent(significance_candidates)

        exposure_candidates = []
        for this_ref in significance_refs:
            for base_dir in search_dirs:
                exposure_candidates.extend(
                    [
                        _find_latest(
                            base_dir,
                            [
                                f"{config_key}_{energy_label}_HEP_Exposure_{this_ref}_Threshold_*.png",
                                f"{config_key}_{energy_label}_HEP_Exposure_{this_ref}_*.png",
                            ],
                        ),
                        _find_latest(
                            base_dir / "marley",
                            [
                                f"{config_key}_marley_{energy_label}_HEP_Exposure_{this_ref}_Threshold_*.png",
                                f"{config_key}_marley_{energy_label}_HEP_Exposure_{this_ref}_*.png",
                            ],
                        ),
                    ]
                )
        expected_exposure = _pick_most_recent(exposure_candidates)
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
    plot_dir = ROOT / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        search_dirs = [
            plot_dir / config_key / folder,
            plot_dir / config_key / "marley" / folder,
            plot_dir / folder / config_key,
            plot_dir / folder / config_key / "marley",
            plot_dir / config_key,
            plot_dir / config_key / "marley",
        ]

        significance_patterns = [
            f"{config_key}_{energy_label}_HEP_Significance_Comparison_Exposure_*_Threshold_*.png",
            f"{config_key}_{energy_label}_HEP_Significance_Comparison_Exposure_*.png",
            f"{config_key}_marley_{energy_label}_HEP_Significance_Comparison_Exposure_*_Threshold_*.png",
            f"{config_key}_marley_{energy_label}_HEP_Significance_Comparison_Exposure_*.png",
        ]
        expected_significance = _pick_most_recent(
            [_find_latest(base_dir, significance_patterns) for base_dir in search_dirs]
        )

        exposure_patterns = [
            f"{config_key}_{energy_label}_HEP_Exposure_Comparison_Threshold_*.png",
            f"{config_key}_{energy_label}_HEP_Exposure_Comparison_*.png",
            f"{config_key}_marley_{energy_label}_HEP_Exposure_Comparison_Threshold_*.png",
            f"{config_key}_marley_{energy_label}_HEP_Exposure_Comparison_*.png",
        ]
        expected_exposure = _pick_most_recent(
            [_find_latest(base_dir, exposure_patterns) for base_dir in search_dirs]
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
    plot_dir = ROOT / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        search_dirs = [
            plot_dir / config_key / folder,
            plot_dir / config_key / "marley" / folder,
            plot_dir / folder / config_key,
            plot_dir / folder / config_key / "marley",
            plot_dir / config_key,
            plot_dir / config_key / "marley",
        ]
        expected_asimov = _pick_most_recent(
            [
                _find_latest(
                    base_dir,
                    [
                        f"{config_key}_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison*.png",
                        f"{config_key}_marley_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_marley_{energy_label}_HEP_Asimov_AdaptiveRebin_Comparison*.png",
                    ],
                )
                for base_dir in search_dirs
            ]
        )
        expected_gaussian = _pick_most_recent(
            [
                _find_latest(
                    base_dir,
                    [
                        f"{config_key}_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison*.png",
                        f"{config_key}_marley_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_marley_{energy_label}_HEP_Gaussian_AdaptiveRebin_Comparison*.png",
                    ],
                )
                for base_dir in search_dirs
            ]
        )
        expected_profile = _pick_most_recent(
            [
                _find_latest(
                    base_dir,
                    [
                        f"{config_key}_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison*.png",
                        f"{config_key}_marley_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_*.png",
                        f"{config_key}_marley_{energy_label}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison*.png",
                    ],
                )
                for base_dir in search_dirs
            ]
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


def gather_rebinned_significance_specs(folder, energy, reference):
    plot_dir = ROOT / "images" / "analysis" / "hep"
    energy_label = output_energy_label(energy)
    slides = []
    for config_key, display_name in STANDARD_CONFIGS:
        config_dir = plot_dir / folder / config_key / "marley"
        expected = _find_latest(
            config_dir,
            [
                f"{config_key}_marley_{energy_label}_HEP_RebinnedSignificance_{reference}_Smoothed_Threshold_*_Exposure_*.png",
                f"{config_key}_marley_{energy_label}_HEP_RebinnedSignificance_{reference}_Smoothed*.png",
            ],
        )

        slides.append(
            {
                "name": display_name,
                "config": config_key,
                "folder": folder,
                "plot": expected.relative_to(ROOT).as_posix() if expected is not None else None,
            }
        )

    return slides


def _find_fiducial_plot(folder, config_key, label, energy):
    root_dir = ROOT / "images" / "solar" / "fiducial"
    energy_label = output_energy_label(energy)

    # Strict pattern: use only HEP-tagged fiducial significance outputs.
    # Prefer config-first layout (images/solar/fiducial/<config>/<folder>/...),
    # then fall back to older folder-first layout when missing.
    search_dirs = [
        root_dir / config_key / folder,
        root_dir / config_key / folder / "marley",
        root_dir / folder / config_key,
        root_dir / folder / config_key / "marley",
    ]
    patterns = [
        f"{config_key}_{energy_label}_HEP_{label}Fiducial_Significance*.png",
        f"{config_key}_marley_{energy_label}_HEP_{label}Fiducial_Significance*.png",
    ]
    for base_dir in search_dirs:
        expected = _find_latest(base_dir, patterns)
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
        "| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    if not filtered_rows:
        lines.append("| *(no HEP entries found)* | - | - | - | - | - |")
        return "\n".join(lines)

    for row in filtered_rows:
        lines.append(
            "| "
            + f"{config_alias(row['Config'])} | {fmt_int(row['FidX'])} | {fmt_int(row['FidY'])} | {fmt_int(row['FidZ'])} | {fmt_float(row['BeforeFid'])} | {fmt_float(row['AfterFid'])} |"
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
            "Compare where discovery is concentrated "
            # "and how smoothing/rebinning changes threshold-region behavior."
        )

    slides = []
    for spec in plot_specs:
        if spec["exposure"] or spec["significance"]:
            significance_block = (
                f"    <img src=\"../{spec['significance']}\">"
                if spec["significance"]
                else "    <p>Significance plot not available.</p>"
            )
            exposure_block = (
                f"    <img src=\"../{spec['exposure']}\">"
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
        slides.append("### Plot outputs\n\nNo HEP plot PNGs were found in images/analysis/hep.")

    return "\n\n---\n\n".join(slides)


def render_fiducial_plot_slides(specs):
    slides = []
    for spec in specs:
        if spec["best"] or spec["no"]:
            best_img = (
                f"    <img src=\"../{spec['best']}\">"
                if spec["best"]
                else "    <p>Best fiducial plot not available.</p>"
            )
            no_img = (
                f"    <img src=\"../{spec['no']}\">"
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
                        f"    <img src=\"../{spec['significance']}\">",
                        "  </div>",
                        "  <div>",
                        "    <p><strong>Exposure Reference Comparison</strong></p>",
                        f"    <img src=\"../{spec['exposure']}\">",
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
        if spec.get("asimov"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>Adaptive Rebin Comparison (Asimov)</strong></p>",
                        f"  <img src=\"../{spec['asimov']}\">",
                        "</div>",
                    ]
                )
            )
        if spec.get("gaussian"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>Adaptive Rebin Comparison (Gaussian)</strong></p>",
                        f"  <img src=\"../{spec['gaussian']}\">",
                        "</div>",
                    ]
                )
            )
        if spec.get("profile"):
            available_blocks.append(
                "\n".join(
                    [
                        "<div>",
                        "  <p><strong>Adaptive Rebin Comparison (ProfileLikelihood)</strong></p>",
                        f"  <img src=\"../{spec['profile']}\">",
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


def render_rebinned_significance_slides(specs):
    slides = []
    for spec in specs:
        if spec["plot"]:
            slides.append(
                "\n".join(
                    [
                        f"### {spec['name']}",
                        "",
                        "<div class=\"center\">",
                        f"  <img src=\"../{spec['plot']}\">",
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
                        f"No rebinned-significance output found for {spec['name']}",
                    ]
                )
            )

    if not slides:
        slides.append("### Rebinned significance\n\nNo HEP rebinned-significance PNGs were found.")

    return "\n\n---\n\n".join(slides)


def render_folder_sections(
    folder,
    fid_specs,
    hep_specs,
    reference_specs,
    adaptive_specs,
    best_sigma_rows,
    fid_rows,
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
"""


def build_markdown(
    folder,
    best_sigma_rows,
    fid_rows,
    hep_specs,
    reference_specs,
    adaptive_specs,
    fid_specs,
    reference,
    smoothing_stage_rows,
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
    )
    selected_title = folder.title()

    text = textwrap.dedent(
        f"""
    ---
    marp: true
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
    - threshold in 13HEP.py: default 10.0 MeV
    - optional cuts override: nhits, ophits, adjcls
    - significance reference in plots: {reference}
    - best-curve reference in 0ZBestSigmas.py: Smoothed or Raw

    ---

    ### Workflow Outputs

    - Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../data/solar/fiducial/truncated/BestFiducials.json)
    - Best cut summaries (JSON): [data/analysis/daynight-json/truncated](../data/analysis/daynight-json/truncated)
    - Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated)
    - Figures: [images/analysis/hep/truncated](../images/analysis/hep/truncated)

    ---

    ### Histogram and Significance Flow

    - Step 1: Build HEP rates and threshold region in [src/analysis/13HEP.py](../src/analysis/13HEP.py) per component.
    - Step 2: Apply component-aware smoothing via [lib/lib_smooth.py](../lib/lib_smooth.py) using HEP smoothing config.
    - Step 3: Evaluate Gaussian and Asimov significance curves in [src/analysis/13HEP.py](../src/analysis/13HEP.py).
    - Step 4: Pick best cuts in [src/analysis/0ZBestSigmas.py](../src/analysis/0ZBestSigmas.py), then evaluate ProfileLikelihood exposure in [src/analysis/13HEPProfileLikelihood.py](../src/analysis/13HEPProfileLikelihood.py).
    - Step 5: Render exposure/significance and comparison plots in [src/analysis/13HEPExposurePlot.py](../src/analysis/13HEPExposurePlot.py), [src/analysis/13HEPSignificancePlot.py](../src/analysis/13HEPSignificancePlot.py), [src/analysis/13HEPSignificanceComparisonPlot.py](../src/analysis/13HEPSignificanceComparisonPlot.py), and [src/analysis/13HEPExposureComparisonPlot.py](../src/analysis/13HEPExposureComparisonPlot.py).

    ---

    ### Adaptative Rebinning: Strategy

    - Rebinning is applied in [src/analysis/13HEP.py](../src/analysis/13HEP.py) through [lib/lib_smooth.py](../lib/lib_smooth.py) using `apply_adaptive_tail_rebin`.
    - It is controlled by [import/analysis.json](../import/analysis.json) under `ADAPTIVE_REBIN -> ANALYSES -> HEP`.
    - At each exposure, bins are merged from the high-energy tail until the expected detectable signal per rebinned group reaches the configured threshold.
    - This stabilizes low-statistics significance estimates while preserving discovery sensitivity in sparse tails.

    Rebin threshold criterion used for each grouped bin:
    $$
    S_{{\mathrm{{group}}}} = \sum_{{i \in \mathrm{{group}}}} S_i^{{\mathrm{{det}}}} \ge T
    $$
    $$
    T = \max\!\left(\\texttt{{min\_expected\_events}},\,-\ln(1-\\texttt{{min\_count\_probability}})\\right)
    $$

    ---

    ### Adaptative Rebinning: Discovery

    Discovery significance is then evaluated on rebinned inputs:
    $$
    Z = Z\!\left(S_{{\mathrm{{group}}}},\,B_{{\mathrm{{group}}}},\,\sigma_{{B,\mathrm{{group}}}}\\right)
    $$

    Latest ProfileLikelihood updates used in this deck:
    - Adaptive rebinning is enabled by default in [src/analysis/13HEPProfileLikelihood.py](../src/analysis/13HEPProfileLikelihood.py).
    - Detection-mask zeroing is disabled for the ProfileLikelihood path to avoid artificial early-exposure suppression.
    - Adaptive rebin starts are frozen (computed once at reference exposure and reused across the scan) to reduce discontinuities and keep curves monotonic.

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

    best_sigma_rows = gather_best_sigma_rows(args.energy)
    fid_rows = gather_fiducial_rows(args.energy)
    selected_hep_specs = gather_hep_plot_specs(args.folder, args.energy, args.reference)
    selected_reference_specs = gather_reference_comparison_specs(args.folder, args.energy)
    selected_adaptive_specs = gather_adaptive_rebin_specs(args.folder, args.energy)
    selected_fid_specs = gather_fiducial_plot_specs(args.folder, args.energy)
    smoothing_stage_rows = gather_hep_smoothing_stage_rows()

    markdown = build_markdown(
        args.folder,
        best_sigma_rows,
        fid_rows,
        selected_hep_specs,
        selected_reference_specs,
        selected_adaptive_specs,
        selected_fid_specs,
        args.reference,
        smoothing_stage_rows,
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
