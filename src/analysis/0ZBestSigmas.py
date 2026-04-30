import os
import sys
import json

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


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


def _deep_merge_dict(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base and return merged dictionary."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_and_write_json(path: str, updates: dict, debug: bool = True) -> None:
    """Load existing JSON (if any), merge updates, remove old file, and write new content."""
    existing_payload = {}

    if os.path.exists(path):
        try:
            with open(path, "r") as f_in:
                loaded = json.load(f_in)
                if isinstance(loaded, dict):
                    existing_payload = loaded
        except (OSError, json.JSONDecodeError) as exc:
            rprint(
                f"[yellow][WARNING][/yellow] Could not read existing JSON {path}: {exc}. Replacing file content."
            )

    merged_payload = _deep_merge_dict(existing_payload, updates)

    if os.path.exists(path):
        os.remove(path)

    with open(path, "w") as f_out:
        json.dump(merged_payload, f_out, indent=2)

    if debug:
        rprint(f"Saved JSON summary to {path}")


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
        f"{root}/data/analysis/best-sigma-json/{analysis_dir}/{folder.lower()}/{config}/{name}",
        f"{root}/data/analysis/{analysis_dir}-json/{folder.lower()}/{config}/{name}",
    ]
    if analysis_dir != "daynight":
        local_out_dirs.append(f"{root}/data/analysis/daynight-json/{folder.lower()}/{config}/{name}")

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
        _merge_and_write_json(pnfs_out_path, payload, debug=debug)
    except OSError as exc:
        rprint(
            f"[yellow][WARNING][/yellow] Could not write JSON summary to PNFS path {pnfs_out_path}: {exc}"
        )

    for local_out_dir in local_out_dirs:
        local_out_path = f"{local_out_dir}/{filename}"
        try:
            if not os.path.exists(local_out_dir):
                os.makedirs(local_out_dir)
            _merge_and_write_json(local_out_path, payload, debug=debug)
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

args = parser.parse_args()

analysis_info = load_analysis_info(str(root))
histogram_reference = args.reference or analysis_info.get(
    "BEST_SIGMA_HISTOGRAM_REFERENCE", "Smoothed"
)
significance_reference = analysis_info.get(
    "BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}
).get(args.analysis.upper(), "Asimov")

# 0ZBestSigmas runs before profile-likelihood products are generated in the HEP workflow.
# Force Asimov here so best-cut selection always has an available reference column.
if args.analysis.upper() == "HEP" and significance_reference == "ProfileLikelihood":
    rprint(
        "[cyan][INFO][/cyan] Overriding HEP best-sigma reference to Asimov in 0ZBestSigmas (ProfileLikelihood is produced in a later step)."
    )
    significance_reference = "Asimov"

reference_column = (
    significance_reference
    if histogram_reference == "Smoothed"
    else f"Raw{significance_reference}"
)

rprint(
    f"Using {histogram_reference.lower()} histogram reference column {reference_column} for {args.analysis}"
)

fastest_sigma2, fastest_sigma3, highest_sigma = {}, {}, {}
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

    sigmas_df = explode(sigmas_df, ["Sigma2", "Sigma3", "Exposure", reference_column])
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
            * (sigmas_df["NHits"].isin(nhits[:10]))
            * (sigmas_df["OpHits"].isin(nhits[3:10]))
            * (sigmas_df["AdjCl"].isin(nhits[::-1][10:]))
        ]
        # print(this_sigma_df)
        if idx == 0:
            this_sigma_df_best = this_sigma_df.loc[
                this_sigma_df[reference_column] == this_sigma_df[reference_column].max()
            ].copy()
            this_sigma = _pick_first_row(this_sigma_df_best)
            if this_sigma is None:
                rprint(
                    f"[yellow][WARNING][/yellow] No highest {sigma_label} row found for {config} {name} {energy_label}"
                )
            else:
                highest_sigma[(config, name, energy_label)] = {
                    sigma_label: this_sigma[sigma_label],
                    "Values": this_sigma[reference_column],
                    "NHits": this_sigma["NHits"],
                    "OpHits": this_sigma["OpHits"],
                    "AdjCl": this_sigma["AdjCl"],
                }
                rprint(
                    f'\t*Adding highest sigma with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
                )

        # Find the entry with the fastest sigma (min this_sigma_df["sigma_label"])
        this_sigma_df_fast = this_sigma_df.loc[
            (this_sigma_df[sigma_label] > 0)
            * (this_sigma_df[reference_column] == this_sigma_df[reference_column].max())
        ].copy()
        this_sigma = _pick_first_row(this_sigma_df_fast)
        if this_sigma is None:
            rprint(
                f"[yellow][WARNING][/yellow] No fastest {sigma_label} row found for {config} {name} {energy_label}"
            )
            continue

        fastest_sigmas[idx][(config, name, energy_label)] = {
            sigma_label: this_sigma[sigma_label],
            "Values": this_sigma[reference_column],
            "NHits": this_sigma["NHits"],
            "OpHits": this_sigma["OpHits"],
            "AdjCl": this_sigma["AdjCl"],
        }
        rprint(
            f'\t*Adding fastest sigma{idx+2} with nihts {this_sigma["NHits"]} ophits {this_sigma["OpHits"]} adjcls {this_sigma["AdjCl"]}'
        )

    for sigma_results, sigma_results_label in zip(
        [highest_sigma, fastest_sigma2, fastest_sigma3],
        ["highest", "fastest_sigma2", "fastest_sigma3"],
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
