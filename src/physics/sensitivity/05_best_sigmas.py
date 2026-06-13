import os
import sys
import json

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


def _is_spiked(arr, threshold: float) -> bool:
    """Return True if any consecutive step in arr exceeds threshold.

    Used on RawPreIsotonicProfileLikelihood before PAVA: spikes appear as large
    positive jumps that PAVA converts into high plateaus in the final curve.
    """
    a = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if a.size < 2:
        return False
    return float(np.max(np.diff(a))) > threshold


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return _to_builtin(value.iloc[0])
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _pick_first_row(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    return df.iloc[0]



def save_sigma_summary_json(
    sigma_results,
    sigma_results_label,
    analysis,
    folder,
    config,
    name,
    debug=True,
):
    analysis_dir = str(analysis).lower()
    pnfs_out_dir = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}"
    local_out_dirs = [
        f"{root}/config/{config}/best-sigma-json/{analysis_dir}/{folder.lower()}/{name}",
        f"{root}/config/{config}/{analysis_dir}-json/{folder.lower()}/{name}",
    ]

    payload = {}
    for (cfg, sample_name, energy_label), values in sigma_results.items():
        payload.setdefault(cfg, {}).setdefault(sample_name, {})[energy_label] = {
            key: _to_builtin(val) for key, val in values.items()
        }

    filename = f"{config}_{name}_{sigma_results_label}_{analysis}.json"
    pnfs_out_path = f"{pnfs_out_dir}/{filename}"

    try:
        if not os.path.exists(pnfs_out_dir):
            os.makedirs(pnfs_out_dir)
        merge_and_write_json(pnfs_out_path, payload, debug=debug)
    except OSError as exc:
        rprint(
            f"[yellow][WARNING][/yellow] Could not write JSON summary to PNFS path {pnfs_out_path}: {exc}"
        )

    for local_out_dir in local_out_dirs:
        local_out_path = f"{local_out_dir}/{filename}"
        try:
            if not os.path.exists(local_out_dir):
                os.makedirs(local_out_dir)
            merge_and_write_json(local_out_path, payload, debug=debug)
        except OSError as exc:
            rprint(
                f"[yellow][WARNING][/yellow] Could not write JSON summary to local path {local_out_path}: {exc}"
            )

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--analysis",
    type=str,
    help="The name of the analysis",
    choices=["DayNight", "Sensitivity", "HEP"],
    required=True,
)
parser.add_argument(
    "--reference",
    type=str,
    help="The histogram reference to use when selecting the best significance curves",
    choices=["Smoothed", "Raw"],
    default=None,
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configuration to load",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["marley"],
)
parser.add_argument(
    "--folder", type=str, help="The name of the results folder", default="Nominal"
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--nhits",
    type=int,
    help="The min niht cut for the analysis",
    default=None,
)
parser.add_argument(
    "--ophits",
    type=int,
    help="The min ophit cut for the analysis",
    default=None,
)
parser.add_argument(
    "--adjcls",
    type=int,
    help="The max adjcl cut for the analysis",
    default=None,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--max_pl_jump",
    type=float,
    default=1.0,
    help=(
        "Maximum allowed single-step jump (σ) in RawPreIsotonicProfileLikelihood "
        "or PreIsotonicProfileLikelihood when classifying a curve as spiked. "
        "Both pre-PAVA columns are checked: spikes can appear in the smoothed-histogram "
        "path (PreIsotonic) even when the raw-histogram path (RawPreIsotonic) is clean. "
        "Spiked curves are excluded from the main highest selection and saved separately "
        "as highest_spiked. Set to 0 to disable filtering (backward-compatible)."
    ),
)

args = parser.parse_args()

analysis_info = load_analysis_info(str(root))
histogram_reference = args.reference or analysis_info.get(
    "BEST_SIGMA_HISTOGRAM_REFERENCE", "Smoothed"
)
significance_reference = analysis_info.get(
    "BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}
).get(args.analysis.upper(), "Asimov")

reference_column = (
    significance_reference
    if histogram_reference == "Smoothed"
    else f"Raw{significance_reference}"
)

# Map significance_reference to the crossing-column prefix:
#   ProfileLikelihood → "PL"   (PLSigma2/PLSigma3, from hep/01_hep.py)
#   Asimov            → "Asimov" (AsimovSigma2/AsimovSigma3, from daynight/01_daynight.py)
#   Gaussian / other  → ""      (Sigma2/Sigma3, tracked via Gaussian path)
_sigma_crossing_prefix = (
    "PL" if significance_reference == "ProfileLikelihood"
    else "Asimov" if significance_reference == "Asimov"
    else ""
)

rprint(
    f"Using {histogram_reference.lower()} histogram reference column {reference_column} for {args.analysis}"
)

fastest_sigma2, fastest_sigma3, highest_sigma, highest_spiked_sigma = {}, {}, {}, {}
fastest_sigmas = [fastest_sigma2, fastest_sigma3]

plot_data = {}
processed_any = False
for config, name, energy_label in product(args.config, args.name, args.energy):
    input_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{energy_label}_{args.analysis}_Results.pkl"
    )
    try:
        sigmas_df = pd.read_pickle(input_path)
    except FileNotFoundError:
        rprint(
            f"[yellow][WARNING][/yellow] Missing results file for {config} {name} {energy_label}: {input_path}"
        )
        continue

    required_columns = [
        "Sigma2",
        "Sigma3",
        "Exposure",
        reference_column,
        "Config",
        "Name",
        "NHits",
        "OpHits",
        "AdjCl",
    ]
    missing_columns = [col for col in required_columns if col not in sigmas_df.columns]
    if sigmas_df.empty or missing_columns:
        rprint(
            f"[yellow][WARNING][/yellow] Skipping {config} {name} {energy_label}: "
            f"results dataframe is empty or missing required columns {missing_columns}."
        )
        continue

    pl_crossing_cols = [
        col for col in ["PLSigma2", "PLSigma3", "AsimovSigma2", "AsimovSigma3"] if col in sigmas_df.columns
    ]

    # Compute spike flag per cut (before exploding, while reference column is still a list).
    # Column selection follows two rules applied in order:
    #   1. Isotonic rule: prefer pre-isotonic columns (only present when pl_isotonic=True,
    #      where spikes are most visible before Gaussian+PAVA smoothing flattens them into
    #      plateaus).  Fall back to the post-isotonic ProfileLikelihood columns when
    #      pl_isotonic=False — the raw PL values are stored there directly.
    #   2. Smoothing rule: if histogram smoothing was active (SmoothingEnabled=True and
    #      SmoothingSigma > 0), use the smoothed-histogram column; otherwise use the
    #      Raw* column.  Both rules apply independently to the pre- and post-isotonic paths
    #      so the filter always operates on the same curve the analysis actually uses.
    _smoothing_active = (
        bool(sigmas_df["SmoothingEnabled"].iloc[0])
        and float(sigmas_df["SmoothingSigma"].iloc[0]) > 0.0
    ) if "SmoothingEnabled" in sigmas_df.columns and "SmoothingSigma" in sigmas_df.columns else False

    _pre_iso_col  = "PreIsotonicProfileLikelihood"    if _smoothing_active else "RawPreIsotonicProfileLikelihood"
    _post_iso_col = "ProfileLikelihood"               if _smoothing_active else "RawProfileLikelihood"

    _pre_isotonic_candidates  = [_pre_iso_col]  if _pre_iso_col  in sigmas_df.columns else []
    _post_isotonic_candidates = [_post_iso_col] if _post_iso_col in sigmas_df.columns else []
    spike_cols = _pre_isotonic_candidates or _post_isotonic_candidates

    if args.max_pl_jump > 0 and spike_cols:
        sigmas_df["_is_spiked"] = False
        for spike_col in spike_cols:
            sigmas_df["_is_spiked"] = sigmas_df["_is_spiked"] | sigmas_df[spike_col].apply(
                lambda arr: _is_spiked(arr, args.max_pl_jump)
            )
        _spike_source = (
            f"pre-isotonic/{'smoothed' if _smoothing_active else 'raw'}"
            if _pre_isotonic_candidates else
            f"post-isotonic fallback/{'smoothed' if _smoothing_active else 'raw'} (pl_isotonic=False)"
        )
        n_spiked = int(sigmas_df["_is_spiked"].sum())
        if n_spiked > 0:
            rprint(
                f"[cyan][INFO][/cyan] {n_spiked}/{len(sigmas_df)} cut rows flagged as spiked "
                f"(max single-step jump > {args.max_pl_jump:.2f} σ, {_spike_source}: {spike_cols}) "
                f"for {config} {name} {energy_label}."
            )
    else:
        sigmas_df["_is_spiked"] = False

    sigmas_df = explode(sigmas_df, ["Sigma2", "Sigma3", "Exposure", reference_column] + pl_crossing_cols)
    for _col in ["Sigma2", "Sigma3", "Exposure", reference_column] + pl_crossing_cols:
        if _col in sigmas_df.columns:
            sigmas_df[_col] = pd.to_numeric(sigmas_df[_col], errors="coerce").fillna(0.0)
    if sigmas_df.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Skipping {config} {name} {energy_label}: exploded dataframe is empty."
        )
        continue

    processed_any = True

    rprint(f"Evaluating {energy_label}")
    for idx, sigma_label in enumerate(["Sigma2", "Sigma3"]):
        # Find the entry with the highest significance (max this_sigma_df["sigma_label"])
        this_sigma_df = sigmas_df[
            (sigmas_df["Config"] == config)
            * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"].isin(nhits))
            * (sigmas_df["OpHits"].isin(nhits[3:]))
            * (sigmas_df["AdjCl"].isin(nhits[::-1]))
        ]

        # Split into spike-free and spiked subsets.
        # Spiked rows are excluded from the main selection; if all rows are spiked,
        # fall back to using all rows so the workflow never silently produces empty output.
        clean_df  = this_sigma_df[~this_sigma_df["_is_spiked"]]
        spiked_df = this_sigma_df[ this_sigma_df["_is_spiked"]]
        if clean_df.empty and not spiked_df.empty:
            rprint(
                f"[yellow][WARNING][/yellow] All cut rows spiked for {config} {name} {energy_label}. "
                "Falling back to full set for highest selection."
            )
            clean_df = this_sigma_df

        if idx == 0:
            # Highest from clean rows
            this_sigma_df_best = clean_df.loc[
                clean_df[reference_column] == clean_df[reference_column].max()
            ].copy()
            this_sigma = _pick_first_row(this_sigma_df_best)
            if this_sigma is None:
                rprint(
                    f"[yellow][WARNING][/yellow] No highest {sigma_label} row found for {config} {name} {energy_label}"
                )
            else:
                # Prefer PL crossing time when Gaussian/Asimov crossing is zero
                # (happens when Asimov metric is disabled and PL is the main metric).
                _sigma_val = float(this_sigma[sigma_label])
                if _sigma_val == 0.0:
                    _pl_cross_col = f"PL{sigma_label}"
                    if _pl_cross_col in this_sigma.index:
                        _sigma_val = float(this_sigma.get(_pl_cross_col, 0.0))
                highest_sigma[(config, name, energy_label)] = {
                    sigma_label: _sigma_val,
                    "Values": this_sigma[reference_column],
                    "NHits": this_sigma["NHits"],
                    "OpHits": this_sigma["OpHits"],
                    "AdjCl": this_sigma["AdjCl"],
                }
                rprint(
                    f'\t*Adding highest sigma with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
                )

            # Highest from spiked rows (debug artifact)
            if not spiked_df.empty:
                this_sigma_df_spiked = spiked_df.loc[
                    spiked_df[reference_column] == spiked_df[reference_column].max()
                ].copy()
                this_sigma_spiked = _pick_first_row(this_sigma_df_spiked)
                if this_sigma_spiked is not None:
                    _spiked_sigma_val = float(this_sigma_spiked[sigma_label])
                    if _spiked_sigma_val == 0.0:
                        _pl_cross_col = f"PL{sigma_label}"
                        if _pl_cross_col in this_sigma_spiked.index:
                            _spiked_sigma_val = float(this_sigma_spiked.get(_pl_cross_col, 0.0))
                    highest_spiked_sigma[(config, name, energy_label)] = {
                        sigma_label: _spiked_sigma_val,
                        "Values": this_sigma_spiked[reference_column],
                        "NHits": this_sigma_spiked["NHits"],
                        "OpHits": this_sigma_spiked["OpHits"],
                        "AdjCl": this_sigma_spiked["AdjCl"],
                    }
                    rprint(
                        f'\t*Adding highest_spiked sigma with nhits {this_sigma_spiked["NHits"]} '
                        f'ophits {this_sigma_spiked["OpHits"]} adjcls {this_sigma_spiked["AdjCl"]}'
                    )

        # Find the entry with the fastest sigma (min crossing time).
        # Use PL-specific crossing column when reference is ProfileLikelihood so cut
        # selection reflects PL discovery time, not Asimov.
        # NOTE: sigma crossings in the results pkl are tracked via the Gaussian significance
        # path in daynight/01_daynight.py regardless of the current significance_reference. When
        # reference_column is Asimov, the cut with max Asimov may have Sigma3=0 (its
        # Gaussian curve never crossed 3σ even though its Asimov did), so requiring
        # reference_column == max would silently produce no result. Instead: among all
        # clean rows that have a crossing (crossing_label > 0), pick the one with the
        # minimum crossing time; break ties by highest reference_column value.
        crossing_label = f"{_sigma_crossing_prefix}{sigma_label}"
        if crossing_label not in clean_df.columns:
            crossing_label = sigma_label
        _rows_with_crossing = clean_df.loc[clean_df[crossing_label] > 0].copy()
        if _rows_with_crossing.empty:
            this_sigma = None
        else:
            _min_crossing = _rows_with_crossing[crossing_label].min()
            this_sigma_df_fast = _rows_with_crossing.loc[
                _rows_with_crossing[crossing_label] == _min_crossing
            ].nlargest(1, reference_column)
            this_sigma = _pick_first_row(this_sigma_df_fast)
        if this_sigma is None:
            rprint(
                f"[yellow][WARNING][/yellow] No fastest {sigma_label} row found for {config} {name} {energy_label}"
            )
            continue

        fastest_sigmas[idx][(config, name, energy_label)] = {
            sigma_label: this_sigma[crossing_label],
            "Values": this_sigma[reference_column],
            "NHits": this_sigma["NHits"],
            "OpHits": this_sigma["OpHits"],
            "AdjCl": this_sigma["AdjCl"],
        }
        rprint(
            f'\t*Adding fastest sigma{idx+2} with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
        )

    for sigma_results, sigma_results_label in zip(
        [highest_sigma, highest_spiked_sigma, fastest_sigma2, fastest_sigma3],
        ["highest", "highest_spiked", "fastest_sigma2", "fastest_sigma3"],
    ):
        # If file already exists, load it and update it with the new results, otherwise create a new file

        save_df(
            pd.DataFrame(sigma_results),
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}",
            config,
            name,
            filename=f"{sigma_results_label}_{args.analysis}",
            rm=args.rewrite,
            debug=args.debug,
        )
        save_sigma_summary_json(
            sigma_results,
            sigma_results_label,
            args.analysis,
            args.folder,
            config,
            name,
            debug=args.debug,
        )

if not processed_any:
    rprint(
        f"[yellow][WARNING][/yellow] No valid {args.analysis} results were processed for the requested inputs."
    )
