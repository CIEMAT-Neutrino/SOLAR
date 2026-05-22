import os
import sys
import re
from glob import glob as glob_files

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

from lib.lib_root import Sensitivity_Fitter
from lib.lib_osc import get_oscillation_datafiles

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--reference_analysis",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "HEP", "SENSITIVITY"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--dm2",
    type=float,
    help="The dm2 value for the analysis",
    default=None,
)
parser.add_argument(
    "--sin13",
    type=float,
    help="The sin^2(13) value for the analysis",
    default=None,
)
parser.add_argument(
    "--sin12",
    type=float,
    help="The sin^2(12) value for the analysis",
    default=None,
)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    help="Oscillation backend: 'file' (pre-computed pkl scan), 'prob3', or 'nufast' (config grid)",
    default=None,
    choices=["file", "prob3", "nufast"],
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    default="Nominal",
    choices=["Reduced", "Truncated", "Nominal"],
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument(
    "--threshold",
    type=float,
    help="The threshold for the analysis",
    default=get_analysis_threshold(str(root), "SENSITIVITY", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument(
    "--optimize_cuts",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Scan all available (NHits, AdjCl, OpHits) template triplets and retain the best SENSITIVITY combination",
)
parser.add_argument(
    "--max_cut_candidates",
    type=int,
    default=0,
    help="Limit number of cut candidates during optimization (0 keeps all)",
)
parser.add_argument(
    "--exposure", type=float, help="The exposure for the analysis in kton-years", default=30.0
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)

args = parser.parse_args()
if args.debug:
    rprint(args)

config = args.config
name = args.name
configs = {config: [name]}

threshold = args.threshold
thld = np.where(sensitivity_rebin_centers >= threshold)[0][0]
smoothing_config = get_smoothing_config(
    str(root), analysis_name="SENSITIVITY", dimensions="2d", stage="significance"
)
smoothing_config = dict(smoothing_config)
smoothing_config["params"] = dict(smoothing_config.get("params", {}))
smoothing_config["params"]["sigma_y"] = 0.0
smoothing_info = smoothing_metadata(smoothing_config)


def _background_template_candidates(background_path: str, config: str):
    pattern = f"{background_path}/{config}_background_NHits*_AdjCl*_OpHits*.pkl"
    return sorted(glob_files(pattern))


def _parse_background_template(filepath: str):
    name = os.path.basename(filepath).replace('.pkl', '')
    parts = name.split('_')
    cuts = {}
    for part in parts:
        if part.startswith('NHits'):
            cuts['NHits'] = int(part.replace('NHits', ''))
        elif part.startswith('AdjCl'):
            cuts['AdjCl'] = int(part.replace('AdjCl', ''))
        elif part.startswith('OpHits'):
            cuts['OpHits'] = int(part.replace('OpHits', ''))
    return cuts


def resolve_background_template(background_path: str, config: str, nhits: int, adjcl: int, ophits: int) -> str:
    target = f"{background_path}/{config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl"
    if os.path.exists(target):
        return target

    candidates = _background_template_candidates(background_path, config)
    if not candidates:
        raise FileNotFoundError(
            f"No sensitivity background templates found in {background_path} for {config}"
        )

    def score(candidate_path: str):
        cuts = _parse_background_template(candidate_path)
        return (
            abs(cuts.get('NHits', 10**9) - nhits),
            abs(cuts.get('AdjCl', 10**9) - adjcl),
            abs(cuts.get('OpHits', 10**9) - ophits),
        )

    best = min(candidates, key=score)
    best_cuts = _parse_background_template(best)
    rprint(
        f"[yellow][WARNING][/yellow] Missing exact background template NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}; "
        f"using NHits{best_cuts.get('NHits')} AdjCl{best_cuts.get('AdjCl')} OpHits{best_cuts.get('OpHits')} instead"
    )
    return best



def _load_best_cut_map(info: dict, args, config: str, name: str):
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference_analysis.upper()]))
    tried = []
    for analysis in candidates:
        filepath = (
            f"{info['PATH']}/{analysis}/{args.folder.lower()}/{config}/{name}/"
            f"{config}_{name}_highest_{analysis}.pkl"
        )
        tried.append(filepath)
        if os.path.exists(filepath):
            if args.debug:
                rprint(f"[cyan][INFO][/cyan] Using best-cut map from {analysis}")
            return pickle.load(open(filepath, "rb"))

    rprint(
        "[yellow][WARNING][/yellow] Unable to load any best-cut map. Checked:\n"
        + "\n".join(tried)
    )
    return None


def _parse_signal_template(filepath: str):
    base = os.path.basename(filepath)
    match = re.search(
        r"_NHits(?P<nhits>\d+)_AdjCl(?P<adjcl>\d+)_OpHits(?P<ophits>\d+)_"
        r"dm2_(?P<dm2>[-+0-9.eE]+)_sin13_(?P<sin13>[-+0-9.eE]+)_sin12_(?P<sin12>[-+0-9.eE]+)\.pkl$",
        base,
    )
    if match is None:
        return None

    values = match.groupdict()
    return {
        "NHits": int(values["nhits"]),
        "AdjCl": int(values["adjcl"]),
        "OpHits": int(values["ophits"]),
        "dm2": float(values["dm2"]),
        "sin13": float(values["sin13"]),
        "sin12": float(values["sin12"]),
    }


def _same_oscillation(point: tuple, reference: tuple) -> bool:
    return (
        np.isclose(point[0], reference[0])
        and np.isclose(point[1], reference[1])
        and np.isclose(point[2], reference[2])
    )


def _discover_cut_candidates(paths: dict, config: str, name: str, solar_tuple: tuple, react_tuple: tuple):
    signal_pattern = f"{paths['signal_path']}/{config}_{name}_NHits*_AdjCl*_OpHits*_dm2_*_sin13_*_sin12_*.pkl"
    signal_files = sorted(glob_files(signal_pattern))
    if not signal_files:
        return []

    background_candidates = {
        (
            cuts.get("NHits", -1),
            cuts.get("AdjCl", -1),
            cuts.get("OpHits", -1),
        )
        for cuts in map(_parse_background_template, _background_template_candidates(paths["background_path"], config))
    }

    cut_flags = {}
    for filepath in signal_files:
        parsed = _parse_signal_template(filepath)
        if parsed is None:
            continue

        cuts = (parsed["NHits"], parsed["AdjCl"], parsed["OpHits"])
        if cuts not in background_candidates:
            continue

        point = (parsed["dm2"], parsed["sin13"], parsed["sin12"])
        state = cut_flags.setdefault(cuts, {"solar": False, "react": False})
        if _same_oscillation(point, solar_tuple):
            state["solar"] = True
        if _same_oscillation(point, react_tuple):
            state["react"] = True

    valid = [cuts for cuts, state in cut_flags.items() if state["solar"] and state["react"]]
    return sorted(valid, key=lambda item: (item[0], item[2], item[1]))


def _resolve_cut_entries(paths: dict, info: dict, args, analysis_info: dict, config: str, name: str):
    manual_triplet = (
        args.nhits is not None
        and args.adjcls is not None
        and args.ophits is not None
    )
    if manual_triplet:
        return [
            {
                "NHits": int(args.nhits),
                "AdjCl": int(args.adjcls),
                "OpHits": int(args.ophits),
                "source": "manual",
            }
        ]

    if args.optimize_cuts:
        solar_tuple = (
            analysis_info["SOLAR_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
        react_tuple = (
            analysis_info["REACT_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
        candidates = _discover_cut_candidates(paths, config, name, solar_tuple, react_tuple)
        if args.max_cut_candidates > 0:
            candidates = candidates[: args.max_cut_candidates]
        if candidates:
            if args.debug:
                rprint(
                    f"[cyan][INFO][/cyan] Optimizing over {len(candidates)} available cut triplets"
                )
            return [
                {
                    "NHits": int(cut[0]),
                    "AdjCl": int(cut[1]),
                    "OpHits": int(cut[2]),
                    "source": "scan",
                }
                for cut in candidates
            ]

    best_map = _load_best_cut_map(info, args, config, name)
    if best_map is not None:
        key = (config, name, args.energy)
        if key in best_map:
            selected = best_map[key]
        else:
            selected = next(iter(best_map.values()))

        return [
            {
                "NHits": int(args.nhits) if args.nhits is not None else int(selected["NHits"]),
                "AdjCl": int(args.adjcls) if args.adjcls is not None else int(selected["AdjCl"]),
                "OpHits": int(args.ophits) if args.ophits is not None else int(selected["OpHits"]),
                "source": "best-map",
            }
        ]

    fallback = {
        "NHits": int(args.nhits) if args.nhits is not None else 4,
        "AdjCl": int(args.adjcls) if args.adjcls is not None else 10,
        "OpHits": int(args.ophits) if args.ophits is not None else 4,
    }
    rprint(
        "[yellow][WARNING][/yellow] Falling back to default cuts "
        f"NHits{fallback['NHits']} AdjCl{fallback['AdjCl']} OpHits{fallback['OpHits']}"
    )
    return [
        {
            "NHits": fallback["NHits"],
            "AdjCl": fallback["AdjCl"],
            "OpHits": fallback["OpHits"],
            "source": "fallback",
        }
    ]


for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = load_analysis_info(str(root))

    for name in configs[config]:
        energy = args.energy

        paths = {
            "signal_path": f"{info['PATH']}/SENSITIVITY/{config}/{name}/{args.folder.lower()}/{energy}",
            "background_path": f"{info['PATH']}/SENSITIVITY/{config}/background/{args.folder.lower()}/{energy}",
        }

        cut_entries = _resolve_cut_entries(paths, info, args, analysis_info, config, name)

        osc_backend = args.oscillation_backend or analysis_info.get("OSCILLATION_BACKEND", "file")
        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            dm2=args.dm2,
            sin13=args.sin13,
            sin12=args.sin12,
            path=f"{paths['signal_path']}/",
            ext="pkl",
            auto=args.dm2 is None and args.sin13 is None and args.sin12 is None,
            debug=args.debug,
            backend=osc_backend,
        )
        cut_quality = []

        solar_tuple = (
            analysis_info["SOLAR_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
        react_tuple = (
            analysis_info["REACT_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )

        for cut in cut_entries:
            nhits = int(cut["NHits"])
            adjcl = int(cut["AdjCl"])
            ophits = int(cut["OpHits"])
            if args.debug:
                rprint(
                    f"[cyan][INFO][/cyan] Evaluating NHits{nhits} AdjCl{adjcl} OpHits{ophits} (source={cut['source']})"
                )

            react_sin13_df = pd.DataFrame(
                columns=np.unique(sin13_list), index=np.unique(dm2_list)
            )
            react_sin12_df = pd.DataFrame(
                columns=np.unique(sin12_list), index=np.unique(dm2_list)
            )
            solar_sin13_df = pd.DataFrame(
                columns=np.unique(sin13_list), index=np.unique(dm2_list)
            )
            solar_sin12_df = pd.DataFrame(
                columns=np.unique(sin12_list), index=np.unique(dm2_list)
            )

            react_df = pd.DataFrame(columns=["dm2", "sin13", "sin12", "chi2"])
            solar_df = pd.DataFrame(columns=["dm2", "sin13", "sin12", "chi2"])

            pred1_df = np.nan_to_num(
                pd.read_pickle(
                    f'{paths["signal_path"]}/{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["SOLAR_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
                ),
                nan=0.0,
            )
            pred2_df = np.nan_to_num(
                pd.read_pickle(
                    f'{paths["signal_path"]}/{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["REACT_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
                ),
                nan=0.0,
            )

            if args.background:
                background_template = resolve_background_template(
                    paths["background_path"],
                    config,
                    nhits,
                    adjcl,
                    ophits,
                )
                bkg_df = np.nan_to_num(pd.read_pickle(background_template), nan=0.0)
            else:
                bkg_df = 0 * pred1_df

            fake_df_dict = {}
            for dm2, sin13, sin12 in track(
                zip(dm2_list, sin13_list, sin12_list),
                description="Loading data...",
                total=len(dm2_list),
            ):
                this_df = np.nan_to_num(
                    pd.read_pickle(
                        f'{paths["signal_path"]}/{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl'
                    ),
                    nan=0.0,
                )
                fake_df_dict[(dm2, sin13, sin12)] = this_df + bkg_df

            initial_A_pred = 0.0
            initial_A_bkg = 0.0
            solar_fit_at_react = None
            reactor_fit_at_solar = None

            for i in track(
                range(len(fake_df_dict.keys())), description="Computing data..."
            ):
                params = list(fake_df_dict.keys())[i]
                obs_df = list(fake_df_dict.values())[i]

                if args.debug:
                    rprint(
                        "\n--------------------------------",
                        "\n# Parameter Combination " + str(i) + ": ",
                        params,
                    )

                fitter = Sensitivity_Fitter(
                    obs_df[:, thld:],
                    pred1_df[:, thld:],
                    bkg_df[:, thld:],
                    SigmaPred=args.signal_uncertainty,
                    SigmaBkg=args.background_uncertainty,
                    bb_mask=(bkg_df[:, thld:] > 0),
                )

                chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
                if chi2 is None:
                    continue
                chi2_value = float(chi2)
                if args.debug:
                    rprint(f"Solar Chi2: {chi2_value:.2f} (smoothing={smoothing_info['SmoothingMethod']})")

                if params[2] == analysis_info["SIN12"]:
                    solar_sin13_df.loc[params[0], params[1]] = chi2_value

                if params[1] == analysis_info["SIN13"]:
                    solar_sin12_df.loc[params[0], params[2]] = chi2_value

                solar_df.loc[i] = [params[0], params[1], params[2], chi2_value]
                if _same_oscillation(params, react_tuple):
                    solar_fit_at_react = chi2_value

                fitter = Sensitivity_Fitter(
                    obs_df[:, thld:],
                    pred2_df[:, thld:],
                    bkg_df[:, thld:],
                    SigmaPred=args.signal_uncertainty,
                    SigmaBkg=args.background_uncertainty,
                    bb_mask=(bkg_df[:, thld:] > 0),
                )

                chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
                if chi2 is None:
                    continue
                chi2_value = float(chi2)
                if args.debug:
                    rprint(f"Reactor Chi2: {chi2_value:.2f}")
                if params[2] == analysis_info["SIN12"]:
                    react_sin13_df.loc[params[0], params[1]] = chi2_value
                if params[1] == analysis_info["SIN13"]:
                    react_sin12_df.loc[params[0], params[2]] = chi2_value

                react_df.loc[i] = [params[0], params[1], params[2], chi2_value]
                if _same_oscillation(params, solar_tuple):
                    reactor_fit_at_solar = chi2_value

            if args.background:
                path = f'{paths["signal_path"]}/results/signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%'
            else:
                path = f'{paths["signal_path"]}/results/signal_{100*args.signal_uncertainty:.0f}%_only'

            rprint(f"Saving data to {'/'.join(path.split('/')[:-1])}")

            if not os.path.exists(path):
                os.makedirs(path)

            file_names = [
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl",
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl",
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl",
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl",
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_df.pkl",
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_df.pkl",
            ]

            for file_name in file_names:
                if os.path.exists(file_name):
                    os.remove(file_name)

            solar_sin12_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl"
            )
            solar_sin13_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl"
            )
            react_sin12_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl"
            )
            react_sin13_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl"
            )
            solar_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_df.pkl"
            )
            react_df.to_pickle(
                f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_df.pkl"
            )

            if solar_fit_at_react is not None and reactor_fit_at_solar is not None:
                cut_quality.append(
                    {
                        "NHits": nhits,
                        "AdjCl": adjcl,
                        "OpHits": ophits,
                        "SolarFitAtReact": solar_fit_at_react,
                        "ReactorFitAtSolar": reactor_fit_at_solar,
                        "Score": 0.5 * (solar_fit_at_react + reactor_fit_at_solar),
                    }
                )

        if cut_quality:
            best = max(cut_quality, key=lambda item: item["Score"])
            best_payload = {
                (config, name, energy): {
                    "NHits": int(best["NHits"]),
                    "AdjCl": int(best["AdjCl"]),
                    "OpHits": int(best["OpHits"]),
                    "Score": float(best["Score"]),
                    "SolarFitAtReact": float(best["SolarFitAtReact"]),
                    "ReactorFitAtSolar": float(best["ReactorFitAtSolar"]),
                }
            }
            save_pkl(
                best_payload,
                f"{info['PATH']}/SENSITIVITY/{args.folder.lower()}",
                config=config,
                name=name,
                filename="highest_SENSITIVITY",
                rm=args.rewrite,
                debug=args.debug,
            )
            rprint(
                f"[cyan][INFO][/cyan] SENSITIVITY Cuts: NHits={best['NHits']} AdjCl={best['AdjCl']} OpHits={best['OpHits']} (score={best['Score']:.2f})"
            )

            json_payload: dict = {}
            for (cfg, nm, en), values in best_payload.items():
                json_payload.setdefault(cfg, {}).setdefault(nm, {})[en] = values
            for local_dir in [
                f"{root}/data/analysis/sensitivity-json/{args.folder.lower()}/{config}/{name}",
                f"{root}/data/analysis/best-sigma-json/sensitivity/{args.folder.lower()}/{config}/{name}",
            ]:
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                merge_and_write_json(
                    f"{local_dir}/{config}_{name}_highest_Sensitivity.json",
                    json_payload,
                    debug=args.debug,
                )
