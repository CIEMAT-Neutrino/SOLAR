import os
import sys
import json
from typing import Dict, List, Optional, Set, Tuple

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.fiducial_entries import best_fiducials_stem, isolated_suffix, write_fiducial_entries
from lib.fiducial_significance import (
    GROUP_COLUMNS,
    FIDUCIAL_COLUMNS,
    combine_components,
    build_significance_payload,
    get_essential_background_components,
)

_analysis_info = load_analysis_info(str(root))

save_path = f"{root}/output/images/solar/fiducial"
data_path = f"{root}/config/analysis/fiducial"

ANALYSIS_CHOICES = ["DayNight", "HEP", "Sensitivity"]


def _deep_merge_dict(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_and_write_json(path: str, payload: dict) -> None:
    # BestFiducials.json is shared by every detector config: lib.io.merge_and_write_json
    # serialises the read-merge-write with a file lock so parallel per-config pipelines
    # cannot drop each other's entries.
    merge_and_write_json(path, payload, indent=4)


def select_best_fiducial(plot_df: pd.DataFrame, analysis_config: Dict, smoothing_config: Dict, exposure: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = combine_components(
        plot_df,
        analysis_config.get("signal_components", []),
    )
    background_df = combine_components(
        plot_df,
        analysis_config.get("background_components", []),
    )
    merged_df = pd.merge(
        signal_df,
        background_df,
        on=GROUP_COLUMNS,
        how="outer",
        suffixes=("Signal", "Background"),
    )
    for column in [
        "MCCountsSignal",
        "CountsSignal",
        "Error+Signal",
        "Error-Signal",
        "MCCountsBackground",
        "CountsBackground",
        "Error+Background",
        "Error-Background",
    ]:
        if column not in merged_df.columns:
            merged_df[column] = 0.0
    merged_df = merged_df.fillna(0.0)

    significance_rows = []
    for fiducial_values, group in merged_df.groupby(FIDUCIAL_COLUMNS):
        payload = build_significance_payload(
            group,
            analysis_config.get("significance_type", "gaussian"),
            analysis_config.get("combine_mode", "quadrature"),
            smoothing_config,
            exposure,
            analysis_config.get("energy_min"),
            analysis_config.get("energy_max"),
        )
        if payload is None:
            continue
        significance_rows.append({
            "FiducializedX": int(fiducial_values[0]),
            "FiducializedY": int(fiducial_values[1]),
            "FiducializedZ": int(fiducial_values[2]),
            "RawSignificance": payload["RawTotal"],
            "SmoothedSignificance": payload["SmoothedTotal"],
        })

    return merged_df, pd.DataFrame(significance_rows)


def fiducial_mc_diagnostics(
    plot_df: pd.DataFrame,
    gated_plot_df: pd.DataFrame,
    analysis_config: Dict,
    mc_threshold: float,
    root_path: str,
    chosen: pd.Series,
    smoothing_config: Dict,
    exposure: float,
) -> Dict:
    """Record whether MC statistics decided the fiducial volume. The selection itself is unchanged.

    MCLimited is set when (a) the MC gate changed the choice: without it the optimiser would pick a
    volume where an essential background has fewer than mc_threshold MC events, or (b) an essential
    background has fewer than 2 x mc_threshold MC events in the chosen volume (energy window).
    Flagged volumes are candidates for re-optimisation with more background MC.
    """
    essential = sorted(get_essential_background_components(root_path, list(analysis_config.get("background_components", []))))
    window = plot_df
    if analysis_config.get("energy_min") is not None:
        window = window.loc[window["Energy"] >= float(analysis_config["energy_min"])]
    if analysis_config.get("energy_max") is not None:
        window = window.loc[window["Energy"] <= float(analysis_config["energy_max"])]
    volume = (
        (window["FiducializedX"] == int(chosen["FiducializedX"]))
        & (window["FiducializedY"] == int(chosen["FiducializedY"]))
        & (window["FiducializedZ"] == int(chosen["FiducializedZ"]))
    )
    chosen_mc = {c: float(window.loc[volume & (window["Component"] == c), "MCCounts"].sum()) for c in essential}
    _, ungated_df = select_best_fiducial(plot_df, analysis_config, smoothing_config, exposure)
    ungated = {}
    gate_changed = False
    if not ungated_df.empty:
        best = ungated_df.loc[ungated_df["SmoothedSignificance"].idxmax()]
        ungated = {
            "FiducialX": int(best["FiducializedX"]), "FiducialY": int(best["FiducializedY"]),
            "FiducialZ": int(best["FiducializedZ"]), "SmoothedSignificance": float(best["SmoothedSignificance"]),
        }
        gate_changed = any(int(best[f"Fiducialized{a}"]) != int(chosen[f"Fiducialized{a}"]) for a in "XYZ")
    near = bool(mc_threshold > 0 and any(v < 2 * float(mc_threshold) for v in chosen_mc.values()))
    reasons = (["MC gate changed the volume"] if gate_changed else []) + (
        [f"essential background MC below 2x{mc_threshold:g} in the chosen volume"] if near else [])
    return {
        "MCLimited": bool(gate_changed or near),
        "MCLimitedReason": "; ".join(reasons),
        "ChosenEssentialMC": chosen_mc,
        "UngatedBest": ungated,
        "VolumesPassingGate": int(gated_plot_df[FIDUCIAL_COLUMNS].drop_duplicates().shape[0]),
        "VolumesScanned": int(plot_df[FIDUCIAL_COLUMNS].drop_duplicates().shape[0]),
    }


# Components to drop from the truth-fiducial scan's significance sum via
# --exclude_radiological_from_scan. Historically justified as "radiological has no
# reconstructable truth position" (its configured key was MainX/Y/Z, the decay's
# bookkeeping site) -- but truth_key_check_{daynight,sensitivity}.csv
# (src/physics/signal/truth_position_study.py) showed EndX/Y/Z tracks the reconstructed
# cluster much better (e.g. VD nominal, N=1293: 69% in-box vs Main's 19%), so
# config/analysis/backgrounds.json now configures radiological's truth_position_keys as
# EndX/Y/Z (2026-09-22) and this exclusion is a separate, weaker empirical knob, not a
# "can't be fiducialized" claim. Do NOT reuse the ESSENTIAL map here
# (get_essential_background_components, below) -- it drives the unrelated MC-support gate
# and has no entry for every real background component (e.g. HEP's "8B", a genuine
# solar-8B physics background with a legitimate truth position), so treating "not
# essential" as "not fiducializable" silently drops components it was never meant to.
NON_FIDUCIALIZABLE_TRUTH_COMPONENTS = {"radiological"}


def filter_fiducializable_components(components: List[str]) -> List[str]:
    return [c for c in components if str(c).lower().split("_")[0] not in NON_FIDUCIALIZABLE_TRUTH_COMPONENTS]


def fiducial_profile_rows(plot_df: pd.DataFrame, gated_plot_df: pd.DataFrame, analysis_config: Dict,
                          smoothing_config: Dict, exposure: float, root_path: str, chosen: Tuple[int, int, int],
                          band_label: str, meta: Dict) -> List[Dict]:
    """One-coordinate profiles of the smoothed fiducial significance through the origin and through `chosen`."""
    _, all_sig = select_best_fiducial(plot_df, analysis_config, smoothing_config, exposure)
    if all_sig.empty:
        return []
    _, gated_sig = select_best_fiducial(gated_plot_df, analysis_config, smoothing_config, exposure)
    passing = set(map(tuple, gated_sig[["FiducializedX", "FiducializedY", "FiducializedZ"]].astype(int).values)) if not gated_sig.empty else set()
    window = plot_df
    if analysis_config.get("energy_min") is not None:
        window = window.loc[window["Energy"] >= float(analysis_config["energy_min"])]
    if analysis_config.get("energy_max") is not None:
        window = window.loc[window["Energy"] <= float(analysis_config["energy_max"])]
    essential = sorted(get_essential_background_components(root_path, list(analysis_config.get("background_components", []))))
    mc = {c: window.loc[window["Component"] == c].groupby(["FiducializedX", "FiducializedY", "FiducializedZ"])["MCCounts"].sum()
          for c in essential}
    axes = {"X": "FiducializedX", "Y": "FiducializedY", "Z": "FiducializedZ"}
    rows = []
    for ref_name, ref in (("Origin", (0, 0, 0)), ("Optimum", tuple(int(v) for v in chosen))):
        for coord, col in axes.items():
            others = [(k, ref[i]) for i, k in enumerate(axes.values()) if k != col]
            sel = all_sig
            for k, v in others:
                sel = sel.loc[sel[k].astype(int) == int(v)]
            for _, r in sel.sort_values(col).iterrows():
                key = (int(r["FiducializedX"]), int(r["FiducializedY"]), int(r["FiducializedZ"]))
                row = {**meta, "Band": band_label, "Reference": ref_name, "Variable": coord, "Margin": int(r[col]),
                       "SmoothedSignificance": float(r["SmoothedSignificance"]), "RawSignificance": float(r["RawSignificance"]),
                       "PassesMCGate": key in passing, "AtReference": key == ref,
                       "ReferenceX": ref[0], "ReferenceY": ref[1], "ReferenceZ": ref[2],
                       "SelectedX": int(chosen[0]), "SelectedY": int(chosen[1]), "SelectedZ": int(chosen[2])}
                for c in essential:
                    row[f"MC_{c}"] = float(mc[c].get(key, 0.0)) if len(mc[c]) else 0.0
                row["MC_essential"] = float(sum(row[f"MC_{c}"] for c in essential))
                rows.append(row)
    # Z relative to its value at the reference point (the curve passes through 1 there): readable across configs.
    for ref_name in ("Origin", "Optimum"):
        at_ref = [r for r in rows if r["Reference"] == ref_name and r["AtReference"]]
        z_ref = at_ref[0]["SmoothedSignificance"] if at_ref else float("nan")
        for r in rows:
            if r["Reference"] == ref_name:
                r["RelativeSignificance"] = r["SmoothedSignificance"] / z_ref if z_ref else float("nan")
    return rows


def apply_fiducial_mc_threshold(
    plot_df: pd.DataFrame,
    analysis_config: Dict,
    mc_threshold: int,
    root_path: str,
) -> pd.DataFrame:
    background_components = list(analysis_config.get("background_components", []))
    essential_components = get_essential_background_components(root_path, background_components)
    if not essential_components:
        return plot_df

    # Always include the no-fiducial baseline (0,0,0) regardless of MC threshold.
    # It is the reference point: any tighter fiducial must improve over it.
    # Configs with suppressed backgrounds have fewer MC events by physics, not by
    # poor statistics — excluding (0,0,0) forces the optimizer into spuriously tight cuts.
    # When no volume passes the gate the baseline is returned alone (the early returns
    # below used to drop it, and the caller then kept a stale volume from a previous run).
    no_fid = plot_df.loc[
        (plot_df["FiducializedX"] == 0)
        & (plot_df["FiducializedY"] == 0)
        & (plot_df["FiducializedZ"] == 0)
    ]

    filtered = plot_df.copy()
    energy_min = analysis_config.get("energy_min")
    energy_max = analysis_config.get("energy_max")
    if energy_min is not None:
        filtered = filtered.loc[filtered["Energy"] >= float(energy_min)]
    if energy_max is not None:
        filtered = filtered.loc[filtered["Energy"] <= float(energy_max)]

    component_mc = (
        filtered.loc[filtered["Component"].isin(essential_components)]
        .groupby(FIDUCIAL_COLUMNS + ["Component"])["MCCounts"]
        .sum()
        .reset_index()
    )
    if component_mc.empty:
        return no_fid.copy()

    component_mc["Pass"] = component_mc["MCCounts"] >= float(mc_threshold)
    pass_counts = (
        component_mc.loc[component_mc["Pass"]]
        .groupby(FIDUCIAL_COLUMNS)["Component"]
        .nunique()
        .reset_index(name="PassComponents")
    )
    pass_counts = pass_counts.loc[
        pass_counts["PassComponents"] >= len(essential_components),
        FIDUCIAL_COLUMNS,
    ]
    if pass_counts.empty:
        return no_fid.copy()

    result = plot_df.merge(pass_counts, on=FIDUCIAL_COLUMNS, how="inner")
    if not no_fid.empty:
        result = pd.concat([result, no_fid]).drop_duplicates(
            subset=GROUP_COLUMNS + ["Component"], keep="first"
        )
    return result


def _parse_bin_mc_floor(spec: str) -> Dict[str, float]:
    """'radiological:2,gamma:1' -> {'radiological': 2.0, 'gamma': 1.0}; empty -> {} (option off)."""
    floors = {}
    for item in [s.strip() for s in str(spec).split(",") if s.strip()]:
        component, _, value = item.partition(":")
        floors[component.strip()] = float(value) if value else 2.0
    return floors


def apply_bin_mc_floor(plot_df: pd.DataFrame, floors: Dict[str, float]) -> pd.DataFrame:
    """Zero Counts and errors of component bins with MCCounts below the component's floor (MCCounts kept)."""
    out = plot_df.copy()
    for component, floor in floors.items():
        low = (out["Component"] == component) & (out["MCCounts"] < float(floor))
        out.loc[low, ["Counts", "Error+", "Error-"]] = 0.0
        rprint(f"[cyan][INFO][/cyan] bin MC floor {component} < {floor:g}: {int(low.sum())} (volume, bin) entries zeroed")
    return out


def smooth_scan_in_margins(plot_df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Gaussian smoothing (mode 'nearest') of Counts, Error^2 and MCCounts across the fiducial grid, per component and
    energy bin. The grid is the regular F_x/F_y/F_z scan of 01_fiducialize.py (20 cm steps)."""
    from scipy.ndimage import gaussian_filter

    out = plot_df.copy()
    axes = [np.sort(out[c].unique()) for c in FIDUCIAL_COLUMNS]
    energies = np.sort(out["Energy"].unique())
    idx = [np.searchsorted(a, out[c].to_numpy()) for a, c in zip(axes, FIDUCIAL_COLUMNS)]
    e_idx = np.searchsorted(energies, out["Energy"].to_numpy())
    shape = tuple(len(a) for a in axes) + (len(energies),)
    for component in out["Component"].unique():
        rows = (out["Component"] == component).to_numpy()
        sel = tuple(i[rows] for i in idx) + (e_idx[rows],)
        for column, power in (("Counts", 1), ("MCCounts", 1), ("Error+", 2), ("Error-", 2)):
            grid = np.zeros(shape)
            grid[sel] = out.loc[rows, column].to_numpy(dtype=float) ** power
            grid = gaussian_filter(grid, sigma=(sigma, sigma, sigma, 0), mode="nearest")
            out.loc[rows, column] = grid[sel] ** (1.0 / power)
    return out


parser = argparse.ArgumentParser(description="Plot the energy distribution of the particles")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--signal", type=str, help="The name of the configuration", default="marley")
parser.add_argument("--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal", "Pessimistic"], default="Nominal")
parser.add_argument("--analysis", nargs="+", type=str, help="The analyses to optimize fiducials for", choices=ANALYSIS_CHOICES, default=ANALYSIS_CHOICES)
parser.add_argument("--energy", nargs="+", type=str, help="The energy for the analysis", choices=["SignalParticleK", "MainK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy", "SolarEnergyOffset", "SolarEnergyNoOffset", "ElectronEnergy"], default=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"])
parser.add_argument("--exposure", type=float, help="The exposure in years.", default=100)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--mc_threshold",
    type=float,
    default=get_analysis_threshold(str(root), "FIDUCIALIZATION", stage="MC", fallback=0.0),
    help="Minimum summed MCCounts required for every essential background component in fiducial selection",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--truth_fiducial",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Load '_fiduc_truth'-labeled Fiducial_Scan pkls (produced by 01_fiducialize.py "
        "--truth_fiducial) and write best fiducials to BestFiducials_fiduc_truth.json."
    ),
)
parser.add_argument(
    "--ignore_energy_window",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Ignore the analysis-specific energy_min/energy_max window during fiducialization. "
        "Use when the energy variable (e.g. SignalParticleK, MainK) has signal below the "
        "standard analysis window (e.g. HEP 14-30 MeV) and you still want a fiducial entry."
    ),
)

parser.add_argument(
    "--exclude_radiological_from_scan",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Drop radiological from background_components before the significance scan (both the "
        "MC-support gate and the significance-payload sum) -- every other configured background "
        "(gamma, neutron, and HEP's 8B) is kept. Radiological's truth position (MainX/Y/Z) is the "
        "decay's bookkeeping location, not a reconstructable cluster position, so at --truth_fiducial "
        "it dominates the scan with a background that cannot actually be fiducialized and drives the "
        "optimizer toward volumes that do not help the topological/energy cuts that follow. Default: "
        "on when --truth_fiducial is set, off otherwise; pass explicitly to override either way."
    ),
)
parser.add_argument(
    "--export_profiles",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Also export one-coordinate profiles of the fiducial significance: FiducialX, FiducialY and FiducialZ scanned "
        "one at a time over the full grid, with the other two held at the origin (0/0/0) or at the chosen volume, for "
        "the full window and for each energy band (ungated, with the MC-gate flag and the essential MC per point). "
        "Written to FIDUCIAL/<folder>/<config>/<signal>/<config>_<signal>_<energy>_Fiducial_Profiles<label>.pkl."
    ),
)
parser.add_argument(
    "--write_records",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Write BestFiducials*.json and the per-entry records (--no-write_records: profiles only, records untouched).",
)
parser.add_argument(
    "--reference_energy",
    type=str,
    default=None,
    help=(
        "Hold the fiducial volume of this energy variable (e.g. SolarEnergy) from BestFiducials.json instead of "
        "scanning. For the energy-estimator study: the volume is spatial, and an un-cut scan on a truth-energy "
        "variable (SignalParticleK/MainK) is background-dominated (~0.02 sigma everywhere), so its argmax is noise."
    ),
)

args = parser.parse_args()
# Study knob SOLAR_FIDUCIAL_EXPOSURE=<years> (fid_exp30): exposure of the fiducial figure of merit (default --exposure 100).
if os.environ.get("SOLAR_FIDUCIAL_EXPOSURE"):
    args.exposure = float(os.environ["SOLAR_FIDUCIAL_EXPOSURE"])
    rprint(f"[cyan][INFO][/cyan] fiducial figure of merit at {args.exposure:g} yr (SOLAR_FIDUCIAL_EXPOSURE)")
if args.exclude_radiological_from_scan is None:
    args.exclude_radiological_from_scan = bool(args.truth_fiducial)
config = args.config
name = args.signal
configs = {config: [name]}

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}", exist_ok=True)  # parallel samples race

_best_fiducials_stem = best_fiducials_stem(args.truth_fiducial)
filename = f"{data_path}/{args.folder.lower()}/{_best_fiducials_stem}.json"
# Only this run's entries are collected here; _merge_and_write_json merges them into the shared file
# under a lock. Seeding this dict from the file at start-up would write every other config's entries
# back as they were when this process started, undoing whatever a concurrent run stored meanwhile.
best_fiducials = {}

_scan_suffix = "_fiduc_truth" if args.truth_fiducial else isolated_suffix()
_profile_rows = []
# The fidsmooth knobs change the scan itself: never let them write the default records or the default profiles.
if os.environ.get("SOLAR_FIDUCIAL_BIN_MC_FLOOR") or float(os.environ.get("SOLAR_FIDUCIAL_MARGIN_SMOOTH", "0") or 0) > 0:
    if args.write_records and not _scan_suffix:
        raise SystemExit("[ERROR] SOLAR_FIDUCIAL_BIN_MC_FLOOR / SOLAR_FIDUCIAL_MARGIN_SMOOTH need --no-write_records or an isolated study suffix")
    if args.export_profiles and not os.environ.get("SOLAR_FIDUCIAL_PROFILE_SUFFIX"):
        raise SystemExit("[ERROR] SOLAR_FIDUCIAL_BIN_MC_FLOOR / SOLAR_FIDUCIAL_MARGIN_SMOOTH with --export_profiles need SOLAR_FIDUCIAL_PROFILE_SUFFIX")

for config in configs:
    for name, energy_label in product(configs[config], args.energy):
        df_list = []
        signal_df = pd.read_pickle(
            f"{_analysis_info['PATH']}/FIDUCIAL/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy_label}_Fiducial_Scan{_scan_suffix}.pkl"
        )
        df_list.append(signal_df)
        for bkg_label in get_background_samples(str(root)):
            # For --truth_fiducial, load background scans using truth positions (MainX/Y/Z for
            # radiological, SignalParticleX/Y/Z for particle-gun backgrounds). This maintains
            # consistency with the signal fiducialization when optimizing fiducial cuts.
            filepath = f"{_analysis_info['PATH']}/FIDUCIAL/{args.folder.lower()}/{config}/{bkg_label}/{config}_{bkg_label}_{energy_label}_Fiducial_Scan{_scan_suffix}.pkl"
            if not os.path.exists(filepath):
                continue
            bkg_df = pd.read_pickle(filepath)
            df_list.append(bkg_df)

        raw_df = pd.concat(df_list, ignore_index=True)
        plot_df = explode(raw_df, ["Counts", "Error+", "Error-", "Energy", "MCCounts"], debug=args.debug).copy()
        plot_df["Counts"] = pd.to_numeric(plot_df["Counts"], errors="coerce").fillna(0.0)
        plot_df["Error+"] = pd.to_numeric(plot_df["Error+"], errors="coerce").fillna(0.0)
        plot_df["Error-"] = pd.to_numeric(plot_df["Error-"], errors="coerce").fillna(0.0)
        plot_df["Energy"] = pd.to_numeric(plot_df["Energy"], errors="coerce")
        plot_df["MCCounts"] = pd.to_numeric(plot_df["MCCounts"], errors="coerce").fillna(0.0)
        # Study knob SOLAR_FIDUCIAL_Z_LIMIT_Y=1 (fid_zlim): the beam-direction margin may not exceed the largest vertical
        # margin of the grid, so F_z cannot use its longer reach (quarter of the module length) to trade statistics.
        if os.environ.get("SOLAR_FIDUCIAL_Z_LIMIT_Y") == "1":
            _y_max = float(plot_df["FiducializedY"].max())
            plot_df = plot_df.loc[plot_df["FiducializedZ"] <= _y_max].copy()
            rprint(f"[cyan][INFO][/cyan] {config} {energy_label}: FiducialZ limited to the FiducialY reach ({_y_max:g} cm)")
        # Study knob SOLAR_FIDUCIAL_BIN_MC_FLOOR="radiological:2[,gamma:2,...]" (fidsmooth study, default off): a component's
        # energy bin backed by fewer than N simulated clusters (scan MCCounts, i.e. after the 01_fiducialize gap filling)
        # is zeroed, as 03_analysis.py does with RADIOLOGICAL_MIN_MC_PER_BIN. A lone heavy cluster then cannot switch a
        # bin on and off as the margin moves.
        # Default (2026-10-18, user): radiological bins need RADIOLOGICAL_MIN_MC_PER_BIN simulated clusters, the floor that
        # 03_analysis.py applies; the env knob replaces it ("none" switches every floor off).
        _floor_env = os.environ.get("SOLAR_FIDUCIAL_BIN_MC_FLOOR")
        if _floor_env is None:
            _rad_floor = _analysis_info.get("RADIOLOGICAL_MIN_MC_PER_BIN")
            _floor_env = f"radiological:{_rad_floor}" if _rad_floor else ""
        _mc_floor = {} if _floor_env.strip().lower() == "none" else _parse_bin_mc_floor(_floor_env)
        if _mc_floor:
            plot_df = apply_bin_mc_floor(plot_df, _mc_floor)
        # Study knob SOLAR_FIDUCIAL_MARGIN_SMOOTH=<sigma in grid steps> (fidsmooth study, default off): Gaussian smoothing of
        # every component's Counts, Error^2 and MCCounts across the (F_x, F_y, F_z) grid, bin by bin in energy.
        _margin_sigma = float(os.environ.get("SOLAR_FIDUCIAL_MARGIN_SMOOTH", "0") or 0)
        if _margin_sigma > 0:
            plot_df = smooth_scan_in_margins(plot_df, _margin_sigma)
            rprint(f"[cyan][INFO][/cyan] {config} {energy_label}: scan smoothed in the margins (sigma {_margin_sigma:g} steps)")

        best_fiducials.setdefault(config, {})

        for analysis_name in args.analysis:
            analysis_key = analysis_name.upper()
            analysis_config = get_fiducialization_config(str(root), analysis_key)
            # Study knob SOLAR_FIDUCIAL_BANDS="HEP=14-22,22-30;DAYNIGHT=6-12,12-18" (fid_hepbands22): replace the energy bands of
            # the listed analyses (the bands are written into the records, so the analysis mask follows them).
            _bands_env = os.environ.get("SOLAR_FIDUCIAL_BANDS")
            if _bands_env:
                for _spec in _bands_env.split(";"):
                    _ana, _, _ranges = _spec.partition("=")
                    if _ana.strip().upper() == analysis_key:
                        analysis_config = dict(analysis_config)
                        analysis_config["energy_bands"] = [
                            {"label": f"b{i}", "energy_min": float(r.split("-")[0]), "energy_max": float(r.split("-")[1])}
                            for i, r in enumerate(_ranges.split(","))]
                        rprint(f"[cyan][INFO][/cyan] {analysis_key}: energy bands {_ranges} (SOLAR_FIDUCIAL_BANDS)")
            _sig_ref = str(
                _analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get(analysis_key, "")
            ).lower()
            if _sig_ref in {"asimov", "gaussian"} and _sig_ref != analysis_config.get("significance_type", "gaussian").lower():
                rprint(
                    f"[yellow][WARNING][/yellow] Overriding {analysis_key} fiducial significance_type "
                    f"({analysis_config.get('significance_type', 'gaussian')!r}) → ({_sig_ref!r}) "
                    f"to match BEST_SIGMA_SIGNIFICANCE_REFERENCE."
                )
                analysis_config = dict(analysis_config)
                analysis_config["significance_type"] = _sig_ref
            if args.ignore_energy_window:
                analysis_config = dict(analysis_config)
                analysis_config.pop("energy_min", None)
                analysis_config.pop("energy_max", None)
            if args.exclude_radiological_from_scan:
                # Radiological's truth position is bookkeeping (the decay site, not a
                # reconstructable cluster position), so it cannot actually be fiducialized;
                # left in, it dominates the truth scan's background sum and the optimizer
                # picks a volume that chases an uncontrollable background instead of the
                # ones (gamma, neutron, 8B, ...) the fiducial cut can genuinely suppress.
                # Only radiological is dropped -- NOT every "non-essential" component, which
                # would also catch real backgrounds like HEP's 8B that simply have no entry
                # in the ESSENTIAL map (get_essential_background_components is for the
                # unrelated MC-support gate and must not be reused for this).
                fiducializable_components = filter_fiducializable_components(
                    list(analysis_config.get("background_components", []))
                )
                analysis_config = dict(analysis_config)
                analysis_config["background_components"] = fiducializable_components
            smoothing_config = get_smoothing_config(
                str(root), analysis_name=analysis_key, dimensions="1d", stage="fiducial"
            )

            if args.reference_energy:
                ref_file = f"{data_path}/{args.folder.lower()}/{best_fiducials_stem()}.json"
                ref_entry = None
                if os.path.exists(ref_file):
                    ref_entry = json.load(open(ref_file)).get(config, {}).get(analysis_key, {}).get(args.reference_energy)
                if ref_entry is None:
                    raise SystemExit(f"[ERROR] --reference_energy {args.reference_energy}: no {analysis_key} entry for {config} in {ref_file}")
                best_fiducials[config].setdefault(analysis_key, {})[energy_label] = {**ref_entry, "Fallback": f"reference_energy:{args.reference_energy}"}
                rprint(
                    f"{analysis_name} {energy_label}: holding {args.reference_energy} volume "
                    f"X={ref_entry['FiducialX']} Y={ref_entry['FiducialY']} Z={ref_entry['FiducialZ']}"
                )
                continue

            gated_plot_df = apply_fiducial_mc_threshold(plot_df, analysis_config, args.mc_threshold, str(root))
            merged_df, significance_df = select_best_fiducial(
                gated_plot_df, analysis_config, smoothing_config, args.exposure
            )
            if significance_df.empty:
                rprint(
                    f"[yellow]No fiducial significance points found for {analysis_name} {energy_label} "
                    f"after essential-component MC threshold {args.mc_threshold}[/yellow]"
                )
                # A truth scan thinned by the purity cut can leave no volume with enough MC (VD HEP). Keeping
                # the previous run's entry was silent staleness and a lower threshold picked a volume that did
                # not survive the analysis cuts, so the truth study inherits the reference volume instead.
                if args.truth_fiducial:
                    ref_file = f"{data_path}/{args.folder.lower()}/{best_fiducials_stem()}.json"
                    ref_entry = None
                    if os.path.exists(ref_file):
                        ref_entry = json.load(open(ref_file)).get(config, {}).get(analysis_key, {}).get(energy_label)
                    if ref_entry is not None:
                        best_fiducials[config].setdefault(analysis_key, {})[energy_label] = {**ref_entry, "Fallback": "reference"}
                        rprint(f"[yellow]{analysis_name} {energy_label}: using the reference volume from BestFiducials.json[/yellow]")
                continue

            max_row = significance_df.loc[significance_df["SmoothedSignificance"].idxmax()]
            no_fid_row = significance_df.loc[
                (significance_df["FiducializedX"] == 0)
                & (significance_df["FiducializedY"] == 0)
                & (significance_df["FiducializedZ"] == 0)
            ]
            no_fid_smoothed = (
                float(no_fid_row.iloc[0]["SmoothedSignificance"])
                if not no_fid_row.empty
                else None
            )
            rprint(
                f"{analysis_name} {energy_label}: best smoothed significance {max_row['SmoothedSignificance']:.2f} at X={int(max_row['FiducializedX'])} Y={int(max_row['FiducializedY'])} Z={int(max_row['FiducializedZ'])}"
            )

            best_fiducials[config].setdefault(analysis_key, {})[energy_label] = {
                "FiducialX": int(max_row["FiducializedX"]),
                "FiducialY": int(max_row["FiducializedY"]),
                "FiducialZ": int(max_row["FiducializedZ"]),
                "RawSignificance": float(max_row["RawSignificance"]),
                "SmoothedSignificance": float(max_row["SmoothedSignificance"]),
                "NoFiducialSignificance": no_fid_smoothed,
                "BestFiducialSignificance": float(max_row["SmoothedSignificance"]),
                "SignalComponents": list(analysis_config.get("signal_components", [])),
                "BackgroundComponents": list(analysis_config.get("background_components", [])),
                "MCThreshold": int(args.mc_threshold),
                "EnergyMin": analysis_config.get("energy_min"),
                "EnergyMax": analysis_config.get("energy_max"),
                "SignificanceType": analysis_config.get("significance_type", "gaussian"),
                **smoothing_metadata(smoothing_config),
            }
            # Mark MC-limited choices (user decision 2026-10-05: keep the strategy, flag for re-optimisation).
            _mc_diag = fiducial_mc_diagnostics(
                plot_df, gated_plot_df, analysis_config, args.mc_threshold, str(root),
                max_row, smoothing_config, args.exposure,
            )
            best_fiducials[config][analysis_key][energy_label].update(_mc_diag)
            if _mc_diag["MCLimited"]:
                rprint(
                    f"[yellow][MC-LIMITED][/yellow] {config} {analysis_name} {energy_label}: {_mc_diag['MCLimitedReason']} "
                    f"(essential MC {_mc_diag['ChosenEssentialMC']}, ungated best {_mc_diag['UngatedBest']})"
                )

            _profile_meta = {"Geometry": "", "Config": config, "Name": name, "Analysis": analysis_name, "Energy": energy_label,
                             "Folder": args.folder, "MCThreshold": int(args.mc_threshold)}
            if args.export_profiles:
                _profile_rows.extend(fiducial_profile_rows(
                    plot_df, gated_plot_df, analysis_config, smoothing_config, args.exposure, str(root),
                    (max_row["FiducializedX"], max_row["FiducializedY"], max_row["FiducializedZ"]), "Full window", _profile_meta))
            # Per-energy-band optimization: find the best fiducial independently for
            # each band. Downstream masking applies the band-specific cut per event.
            band_results = []
            # Study knob SOLAR_FIDUCIAL_SINGLE_BAND=1 (fid_oneband): one volume for the full window, EnergyBands written
            # empty so the seeded per-band entries of the default do not survive the merge.
            _single_band = os.environ.get("SOLAR_FIDUCIAL_SINGLE_BAND") == "1"
            if _single_band:
                best_fiducials[config][analysis_key][energy_label]["EnergyBands"] = []
            for band in ([] if _single_band else analysis_config.get("energy_bands", [])):
                band_config = dict(analysis_config)
                band_config["energy_min"] = band["energy_min"]
                band_config["energy_max"] = band["energy_max"]
                _, band_sig_df = select_best_fiducial(
                    gated_plot_df, band_config, smoothing_config, args.exposure
                )
                if band_sig_df.empty:
                    continue
                band_best = band_sig_df.loc[band_sig_df["SmoothedSignificance"].idxmax()]
                band_label = band.get("label", f"{band['energy_min']}-{band['energy_max']}")
                # Study knob SOLAR_FIDUCIAL_BAND_MIN_Z="abs:<Z>" or "rel:<f>" (fid_bandmin*): a band whose best fiducial-stage
                # significance is below Z (or below f x the full-window significance) carries no information to choose a
                # volume, so it takes the full-window volume instead of its own optimum.
                _bmin = os.environ.get("SOLAR_FIDUCIAL_BAND_MIN_Z")
                if _bmin:
                    _mode, _val = _bmin.split(":")
                    _limit = float(_val) * (float(max_row["SmoothedSignificance"]) if _mode == "rel" else 1.0)
                    if float(band_best["SmoothedSignificance"]) < _limit:
                        rprint(f"  [{band_label}] significance {band_best['SmoothedSignificance']:.3f} < {_limit:.3f} ({_bmin}): "
                               f"full-window volume X={int(max_row['FiducializedX'])} Y={int(max_row['FiducializedY'])} "
                               f"Z={int(max_row['FiducializedZ'])} instead of its own optimum")
                        _full = (band_sig_df["FiducializedX"] == max_row["FiducializedX"]) & \
                                (band_sig_df["FiducializedY"] == max_row["FiducializedY"]) & \
                                (band_sig_df["FiducializedZ"] == max_row["FiducializedZ"])
                        if _full.any():
                            band_best = band_sig_df.loc[_full].iloc[0]
                        else:
                            band_best = max_row
                if args.export_profiles:
                    _profile_rows.extend(fiducial_profile_rows(
                        plot_df, gated_plot_df, band_config, smoothing_config, args.exposure, str(root),
                        (band_best["FiducializedX"], band_best["FiducializedY"], band_best["FiducializedZ"]),
                        f"{band['energy_min']:g}-{band['energy_max']:g} MeV", _profile_meta))
                rprint(
                    f"  [{band_label}] {band['energy_min']}-{band['energy_max']} MeV: "
                    f"significance {band_best['SmoothedSignificance']:.2f} at "
                    f"X={int(band_best['FiducializedX'])} "
                    f"Y={int(band_best['FiducializedY'])} "
                    f"Z={int(band_best['FiducializedZ'])}"
                )
                band_results.append({
                    "label": band_label,
                    "energy_min": band["energy_min"],
                    "energy_max": band["energy_max"],
                    "FiducialX": int(band_best["FiducializedX"]),
                    "FiducialY": int(band_best["FiducializedY"]),
                    "FiducialZ": int(band_best["FiducializedZ"]),
                    "SmoothedSignificance": float(band_best["SmoothedSignificance"]),
                })
            # Written also when empty, so per-band volumes of an earlier record do not survive the merge.
            best_fiducials[config][analysis_key][energy_label]["EnergyBands"] = band_results
            # Study knob SOLAR_FIDUCIAL_FORCE_BAND=<low|high|best|index> (fid_highband, fid_bestband): apply the volume chosen
            # for that energy band (best: the band with the highest fiducial-stage significance) to every event (EnergyBands
            # written empty), instead of per-band volumes plus the full-window volume elsewhere.
            _force = os.environ.get("SOLAR_FIDUCIAL_FORCE_BAND")
            if _force and _force.startswith("vol:"):
                # vol:X,Y,Z (fid_v*): a fixed volume for every event of the analysis, to test single volumes on the final score
                _fx, _fy, _fz = (int(v) for v in _force[4:].split(","))
                best_fiducials[config][analysis_key][energy_label].update(
                    {"FiducialX": _fx, "FiducialY": _fy, "FiducialZ": _fz, "EnergyBands": [], "ForcedBand": _force})
                rprint(f"[cyan][INFO][/cyan] {config} {analysis_name} {energy_label}: fixed volume X={_fx} Y={_fy} Z={_fz} "
                       f"applied to every event")
            elif _force == "max" and band_results:
                # max (fid_bandmax): one volume with each margin the largest of the band volumes (their intersection), so
                # every band's cut is applied to every event
                _fx, _fy, _fz = (max(r[k] for r in band_results) for k in ("FiducialX", "FiducialY", "FiducialZ"))
                best_fiducials[config][analysis_key][energy_label].update(
                    {"FiducialX": _fx, "FiducialY": _fy, "FiducialZ": _fz, "EnergyBands": [], "ForcedBand": "max"})
                rprint(f"[cyan][INFO][/cyan] {config} {analysis_name} {energy_label}: intersection of the band volumes "
                       f"X={_fx} Y={_fy} Z={_fz} applied to every event")
            elif _force and band_results:
                _best = max(range(len(band_results)), key=lambda i: band_results[i]["SmoothedSignificance"])
                _idx = {"low": 0, "high": len(band_results) - 1, "best": _best}.get(_force, None)
                _idx = int(_force) if _idx is None else _idx
                _fb = band_results[_idx]
                _entry = best_fiducials[config][analysis_key][energy_label]
                _entry.update({"FiducialX": _fb["FiducialX"], "FiducialY": _fb["FiducialY"], "FiducialZ": _fb["FiducialZ"],
                               "EnergyBands": [], "ForcedBand": _fb["label"]})
                rprint(f"[cyan][INFO][/cyan] {config} {analysis_name} {energy_label}: volume of band {_fb['label']} "
                       f"({_fb['energy_min']}-{_fb['energy_max']} MeV) applied to every event: "
                       f"X={_fb['FiducialX']} Y={_fb['FiducialY']} Z={_fb['FiducialZ']}")

if args.write_records:
    _merge_and_write_json(filename, best_fiducials)
    # Per-entry records for the template staleness guard (lib/fiducial_entries.py)
    write_fiducial_entries(str(root), args.folder, _best_fiducials_stem, json.load(open(filename)), configs=list(best_fiducials))
else:
    rprint("[cyan][INFO][/cyan] --no-write_records: BestFiducials records left untouched")
if args.export_profiles and _profile_rows:
    _profiles = pd.DataFrame(_profile_rows)
    for (_cfg, _name, _energy), _part in _profiles.groupby(["Config", "Name", "Energy"]):
        _out = f"{_analysis_info['PATH']}/FIDUCIAL/{args.folder.lower()}/{_cfg}/{_name}"
        os.makedirs(_out, exist_ok=True)
        # SOLAR_FIDUCIAL_PROFILE_SUFFIX (default empty): extra label on the profile file name, so study profiles do not
        # replace the default ones.
        _path = f"{_out}/{_cfg}_{_name}_{_energy}_Fiducial_Profiles{_scan_suffix}{os.environ.get('SOLAR_FIDUCIAL_PROFILE_SUFFIX', '')}.pkl"
        if os.path.exists(_path):
            os.remove(_path)  # pnfs: no in-place overwrite
        _part.reset_index(drop=True).to_pickle(_path)
        rprint(f"[cyan][INFO][/cyan] fiducial profiles: {len(_part)} rows -> {_path}")
