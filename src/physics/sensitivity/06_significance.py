import os
import sys
import re
from glob import glob as glob_files
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

from lib.root import Sensitivity_Fitter
from lib.oscillation import get_oscillation_datafiles
from lib.fitting import (
    _sensitivity_apply_energy_scale as _apply_energy_scale,
    _sensitivity_fit_with_escale    as _fit_with_escale,
    sensitivity_chi2_worker         as _chi2_worker,
)

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
    "--signal", type=str, help="The name of the configuration", default="marley"
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
    "--nuisance_profile",
    type=str,
    help="Nuisance parameter profile name (key in NUISANCE_PROFILES in config/analysis/config.json). Defaults to DEFAULT_NUISANCE_PROFILE.",
    default=None,
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
        "MainK",
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
    "--exposure", type=float, help="Exposure in years. Templates are stored PER YEAR (detector_mass_kT x rate) and are scaled by this value at load time, so changing it needs no template regeneration.", default=30.0
)
parser.add_argument(
    "--secondary_exposure", "--secondary-exposure",
    type=float, default=10.0,
    help=(
        "Also evaluate at this exposure in the same pass and write a tagged copy of every "
        "output (e.g. '_10Y'). Templates are per-year, so this is a rescale of the already "
        "loaded templates -- no extra I/O and no pipeline re-run. 0/negative disables."
    ),
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--study_label",
    type=str,
    default=None,
    help="Tag appended to output pkl filenames to isolate study variants from the main analysis.",
)
parser.add_argument(
    "--truth_fiducial",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Truth-position fiducialisation variant. Must match the flag passed to 03_analysis.py so study_context selects the labeled Rebin pkl.",
)
parser.add_argument(
    "--charge_threshold",
    type=float,
    default=0,
    help="Charge threshold Q (ADC). When >0, reads signal/background templates from labeled subfolders.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help=(
        "Number of parallel workers for the chi² grid scan and pkl loading. "
        "Defaults to os.cpu_count(). Use --workers 1 for serial execution (debug/testing)."
    ),
)
parser.add_argument(
    "--fit_background",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Fit background normalization as a free parameter in the chi² fit. "
        "Set to --no-fit_background to fix background normalization and only fit signal amplitude. "
        "Default is True (legacy behavior). For physically meaningful sensitivity, use --no-fit_background."
    ),
)

args = parser.parse_args()
if args.debug:
    rprint(args)

_ctx = study_context(args)
_study_suffix    = _ctx.study_suffix
_template_suffix = _ctx.template_suffix

config = args.config
name = args.signal
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



TEMPLATE_NORMALIZATION_FILE = "TEMPLATE_NORMALIZATION.json"
_REQUIRED_TEMPLATE_VERSION = 2   # v2 = per-year; v1 templates were pre-scaled by exposure


def require_per_year_templates(template_dir: str) -> float:
    """Refuse pre-scaled (v1) templates: rescaling them would be wrong by the old exposure."""
    marker = os.path.join(template_dir, TEMPLATE_NORMALIZATION_FILE)
    if not os.path.exists(marker):
        raise FileNotFoundError(
            f"Missing {TEMPLATE_NORMALIZATION_FILE} in {template_dir}.\n"
            "These templates predate per-year normalization and are already scaled by their "
            "creation exposure; scaling them again would be wrong by that factor. Regenerate:\n"
            "  python3 src/physics/sensitivity/01_background_template.py --rewrite ...\n"
            "  python3 src/physics/sensitivity/02_signal_template.py --rewrite ..."
        )
    meta = json.load(open(marker))
    if int(meta.get("version", 0)) != _REQUIRED_TEMPLATE_VERSION:
        raise ValueError(
            f"Template normalization v{meta.get('version')} in {template_dir}, "
            f"expected v{_REQUIRED_TEMPLATE_VERSION} (per-year). Regenerate the templates."
        )
    return float(meta.get("detector_mass_kT", float("nan")))


def scale_to_exposure(arr, exposure_yr: float):
    """Per-year template -> absolute counts, then drop bins below one expected event.

    The '<1 expected event' cut is exposure dependent, which is exactly why it lives here
    and not in the template writers.
    """
    scaled = np.asarray(arr, dtype=float) * float(exposure_yr)
    scaled[scaled < 1.0] = 0.0
    return scaled


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
    raise FileNotFoundError(
        f"Missing exact sensitivity background template {target}; refusing to use another cut"
    )



def _load_best_cut_map(info: dict, args, config: str, name: str):
    _suffix = f"_{args.study_label}" if getattr(args, 'study_label', None) else ""
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference_analysis.upper()]))
    tried = []
    for analysis in candidates if not _suffix else ["SENSITIVITY"]:
        for suffix in ([_suffix] if _suffix else [""]):
            filepath = (
                f"{info['PATH']}/{analysis}/{args.folder.lower()}/{config}/{name}/"
                f"{config}_{name}_highest_{analysis}{suffix}.pkl"
            )
            tried.append(filepath)
            if os.path.exists(filepath):
                if args.debug:
                    rprint(f"[cyan][INFO][/cyan] Using best-cut map from {analysis}{suffix}")
                return pickle.load(open(filepath, "rb"))

    rprint(
        "[yellow][WARNING][/yellow] Unable to load any best-cut map. Checked:\n"
        + "\n".join(tried)
    )
    return None


def _same_oscillation(point: tuple, reference: tuple) -> bool:
    return (
        np.isclose(point[0], reference[0])
        and np.isclose(point[1], reference[1])
        and np.isclose(point[2], reference[2])
    )


# _apply_energy_scale and _fit_with_escale imported from lib.fitting as aliases above


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

    best_map = _load_best_cut_map(info, args, config, name)
    if best_map is not None:
        key = (config, name, args.energy)
        if key not in best_map:
            raise KeyError(f"No exact best-cut entry for {key} in the selected study map")
        selected = best_map[key]

        return [
            {
                "NHits": int(args.nhits) if args.nhits is not None else int(selected["NHits"]),
                "AdjCl": int(args.adjcls) if args.adjcls is not None else int(selected["AdjCl"]),
                "OpHits": int(args.ophits) if args.ophits is not None else int(selected["OpHits"]),
                "source": "best-map",
            }
        ]

    if args.study_label:
        raise FileNotFoundError(
            f"Missing best-cut map for study '{args.study_label}' and {config}/{name}/{args.energy}; "
            "refusing to use nominal or default cuts"
        )

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


def _compact_and_validate_grid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate a union-of-planes grid without changing its configured limits."""
    validated = df.astype(float).copy()
    values = validated.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if validated.empty or finite.size == 0:
        raise ValueError(
            f"Incomplete {label} sensitivity grid: shape={validated.shape}, no finite values"
        )
    if not np.isfinite(finite).all():
        raise ValueError(f"Invalid non-finite {label} sensitivity grid values")
    validated[validated < 0] = 0.0
    return validated


def _mark_invalid(output_dir: str, reason: str, tag: str = "") -> None:
    marker = f"{output_dir}/{name}_{energy}_Sensitivity_INVALID{tag}.json"
    os.makedirs(output_dir, exist_ok=True)
    # PNFS/dCache is write-once — remove before rewriting.
    if os.path.exists(marker):
        os.remove(marker)
    with open(marker, "w") as handle:
        json.dump({"valid": False, "reason": reason}, handle, indent=2)


for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = load_analysis_info(str(root))

    nuisance_profiles = analysis_info.get("NUISANCE_PROFILES", {})
    default_profile   = analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
    profile_name      = args.nuisance_profile or default_profile or "full"
    profile_overrides = nuisance_profiles.get(profile_name, {})
    for k, v in profile_overrides.items():
        analysis_info[k] = v
    if args.debug:
        rprint(f"[cyan][INFO][/cyan] Nuisance profile: '{profile_name}' overrides={profile_overrides}")

    for name in configs[config]:
        energy = args.energy

        paths = {
            "signal_path": f"{info['PATH']}/SENSITIVITY/{config}/{name}/{args.folder.lower()}/{energy}{_template_suffix}",
            "background_path": f"{info['PATH']}/SENSITIVITY/{config}/background/{args.folder.lower()}/{energy}{_template_suffix}",
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

        for cut in track(cut_entries, description=f"Processing cuts for {config} {name} {energy}..."):
            nhits = int(cut["NHits"])
            adjcl = int(cut["AdjCl"])
            ophits = int(cut["OpHits"])
            if args.debug:
                rprint(
                    f"[cyan][INFO][/cyan] Evaluating NHits{nhits} AdjCl{adjcl} OpHits{ophits} (source={cut['source']})"
                )

            # Load solar + reactor reference templates at best-fit oscillation point
            # These are FIXED predictions used to test against all grid points
            require_per_year_templates(paths["signal_path"])
            pred1_raw = (np.nan_to_num(
                pd.read_pickle(
                    f'{paths["signal_path"]}/{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["SOLAR_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
                ),
                nan=0.0,
            ))
            pred2_raw = (np.nan_to_num(
                pd.read_pickle(
                    f'{paths["signal_path"]}/{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{analysis_info["REACT_DM2"]:.3e}_sin13_{analysis_info["SIN13"]:.3e}_sin12_{analysis_info["SIN12"]:.3e}.pkl'
                ),
                nan=0.0,
            ))

            if args.background:
                background_template = resolve_background_template(
                    paths["background_path"],
                    config,
                    nhits,
                    adjcl,
                    ophits,
                )
                require_per_year_templates(paths["background_path"])
                bkg_raw = np.nan_to_num(pd.read_pickle(background_template), nan=0.0)
            else:
                bkg_raw = 0 * pred1_raw

            _expected_ecols = len(sensitivity_rebin_centers)
            for _label, _arr in [("pred1", pred1_raw), ("pred2", pred2_raw), ("bkg", bkg_raw)]:
                if np.ndim(_arr) >= 2 and _arr.shape[1] != _expected_ecols:
                    raise ValueError(
                        f"Template shape mismatch ({_label}): got {_arr.shape[1]} energy columns, "
                        f"expected {_expected_ecols} (sensitivity_rebin_centers). "
                        f"Regenerate with: python3 sensitivity/02_signal_template.py --rewrite / "
                        f"sensitivity/01_background_template.py --rewrite"
                    )

            # ── Parallel template loading (I/O-bound → ThreadPoolExecutor) ─────────
            n_workers = args.workers if args.workers is not None else (os.cpu_count() or 4)
            n_io = min(n_workers, len(dm2_list), 32)

            _signal_path  = paths["signal_path"]
            # NOTE: loaded RAW (per-year, no background) so the same arrays can be
            # rescaled for every requested exposure without re-reading from disk.

            def _load_one(point):
                dm2_p, sin13_p, sin12_p = point
                pkl = (
                    f"{_signal_path}/{config}_{name}"
                    f"_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"
                    f"_dm2_{dm2_p:.3e}_sin13_{sin13_p:.3e}_sin12_{sin12_p:.3e}.pkl"
                )
                return point, np.nan_to_num(pd.read_pickle(pkl), nan=0.0)

            points = list(zip(dm2_list, sin13_list, sin12_list))
            fake_raw_dict = {}
            rprint(f"[cyan][INFO][/cyan] Loading {len(points)} signal templates "
                   f"({n_io} I/O thread(s))...")
            with ThreadPoolExecutor(max_workers=n_io) as _io_pool:
                for _pt, _df in track(
                    _io_pool.map(_load_one, points),
                    total=len(points),
                    description="Loading templates..."
                ):
                    fake_raw_dict[_pt] = _df

            # ── Per-exposure evaluation ─────────────────────────────────────
            # Templates are stored per-year, so a second exposure is just a rescale:
            # load once above, then run the chi2 scan per requested exposure.
            def _run_exposure(exposure_yr, tag):
                _bkg = scale_to_exposure(bkg_raw, exposure_yr)
                pred1_df = scale_to_exposure(pred1_raw, exposure_yr)
                pred2_df = scale_to_exposure(pred2_raw, exposure_yr)
                fake_df_dict = {
                    pt: scale_to_exposure(arr, exposure_yr) + _bkg
                    for pt, arr in fake_raw_dict.items()
                }
                bkg_df = _bkg

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


                # ── Parallel chi² scan (CPU-bound → ProcessPoolExecutor) ─────────────
                solar_fit_at_react   = None
                reactor_fit_at_solar = None

                marginalize_e_scale = analysis_info.get("ENERGY_SCALE_UNCERTAINTY", False)
                sigma_e_scale       = float(analysis_info.get("ENERGY_SCALE_SIGMA", 0.02))
                e_centers_thld      = sensitivity_rebin_centers[thld:]

                # Limit BLAS threads per worker to 1 so Python-level parallelism is used
                os.environ.setdefault("OMP_NUM_THREADS",     "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
                os.environ.setdefault("MKL_NUM_THREADS",     "1")

                _pred1_slice = pred1_df[:, thld:]
                _pred2_slice = pred2_df[:, thld:]
                _bkg_slice   = bkg_df[:, thld:]

                tasks = [
                    {
                        "params":              params,
                        "obs":                 fake_df_dict[params][:, thld:],
                        "pred1":               _pred1_slice,
                        "pred2":               _pred2_slice,
                        "bkg":                 _bkg_slice,
                        "sigma_pred":          args.signal_uncertainty,
                        "sigma_bkg":           args.background_uncertainty,
                        "marginalize_e_scale": marginalize_e_scale,
                        "sigma_e_scale":       sigma_e_scale,
                        "e_centers_thld":      e_centers_thld,
                        "fit_background":      args.fit_background,
                    }
                    for params in fake_df_dict
                ]

                rprint(f"[cyan][INFO][/cyan] Chi² scan: {len(tasks)} grid point(s), "
                       f"{n_workers} worker(s), escale={marginalize_e_scale}")

                if n_workers == 1:
                    results = [_chi2_worker(t) for t in track(tasks, description="Computing data...")]
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as _cpu_pool:
                        results = list(_cpu_pool.map(_chi2_worker, tasks))

                # ── Collect results (serial post-processing, order preserved) ─────────
                for i, (params, solar_chi2, react_chi2) in track(
                    enumerate(results),
                    total=len(results),
                    description="Collecting results..."
                ):
                    if args.debug:
                        rprint(f"\n--- Combination {i}: {params} ---")

                    if solar_chi2 is None:
                        continue
                    chi2_value = float(solar_chi2)
                    if args.debug:
                        rprint(f"Solar Chi2: {chi2_value:.2f} (smoothing={smoothing_info['SmoothingMethod']})")

                    if params[2] == analysis_info["SIN12"]:
                        solar_sin13_df.loc[params[0], params[1]] = chi2_value
                    if params[1] == analysis_info["SIN13"]:
                        solar_sin12_df.loc[params[0], params[2]] = chi2_value
                    solar_df.loc[i] = [params[0], params[1], params[2], chi2_value]
                    if _same_oscillation(params, react_tuple):
                        solar_fit_at_react = chi2_value

                    if react_chi2 is None:
                        continue
                    chi2_value = float(react_chi2)
                    if args.debug:
                        rprint(f"Reactor Chi2: {chi2_value:.2f}")
                    if params[2] == analysis_info["SIN12"]:
                        react_sin13_df.loc[params[0], params[1]] = chi2_value
                    if params[1] == analysis_info["SIN13"]:
                        react_sin12_df.loc[params[0], params[2]] = chi2_value
                    react_df.loc[i] = [params[0], params[1], params[2], chi2_value]
                    if _same_oscillation(params, solar_tuple):
                        reactor_fit_at_solar = chi2_value

                if analysis_info.get("MARGINALIZE_SIN13", False) and len(solar_df) > 0:
                    sin13_bf    = float(analysis_info["SIN13"])
                    sin13_sigma = float(analysis_info.get("SIN13_SIGMA", 0.00056))
                    if args.debug:
                        rprint(
                            f"[cyan][INFO][/cyan] Profiling sin²θ₁₃ over grid "
                            f"({len(solar_df['sin13'].unique())} values, "
                            f"bf={sin13_bf:.5f}, σ={sin13_sigma:.5f})"
                        )
                    for (dm2_v, sin12_v), grp in solar_df.groupby(["dm2", "sin12"]):
                        pull = ((grp["sin13"].astype(float) - sin13_bf) / sin13_sigma) ** 2
                        solar_sin12_df.loc[dm2_v, sin12_v] = float(
                            (grp["chi2"].astype(float) + pull).min()
                        )
                    for (dm2_v, sin12_v), grp in react_df.groupby(["dm2", "sin12"]):
                        pull = ((grp["sin13"].astype(float) - sin13_bf) / sin13_sigma) ** 2
                        react_sin12_df.loc[dm2_v, sin12_v] = float(
                            (grp["chi2"].astype(float) + pull).min()
                        )

                if args.background:
                    path = f'{paths["signal_path"]}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%'
                else:
                    path = f'{paths["signal_path"]}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_only'

                rprint(f"Saving data to {'/'.join(path.split('/')[:-1])}")

                if not os.path.exists(path):
                    os.makedirs(path)

                file_names = [
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df{tag}.pkl",
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df{tag}.pkl",
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df{tag}.pkl",
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df{tag}.pkl",
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_df{tag}.pkl",
                    f"{path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_df{tag}.pkl",
                ]

                try:
                    solar_sin12_df = _compact_and_validate_grid(solar_sin12_df, "solar sin12")
                    solar_sin13_df = _compact_and_validate_grid(solar_sin13_df, "solar sin13")
                    react_sin12_df = _compact_and_validate_grid(react_sin12_df, "reactor sin12")
                    react_sin13_df = _compact_and_validate_grid(react_sin13_df, "reactor sin13")
                except ValueError as exc:
                    _mark_invalid(path, str(exc), tag)
                    raise

                invalid_marker = f"{path}/{name}_{energy}_Sensitivity_INVALID{tag}.json"
                if os.path.exists(invalid_marker):
                    os.remove(invalid_marker)

                _save_dfs = [
                    (solar_sin12_df, file_names[0]),
                    (solar_sin13_df, file_names[1]),
                    (react_sin12_df, file_names[2]),
                    (react_sin13_df, file_names[3]),
                    (solar_df, file_names[4]),
                    (react_df, file_names[5]),
                ]
                for df, file_path in _save_dfs:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    df.to_pickle(file_path)

                return solar_fit_at_react, reactor_fit_at_solar

            _exposures = [(args.exposure, "")]
            if args.secondary_exposure and args.secondary_exposure != args.exposure:
                _exposures.append((args.secondary_exposure,
                                   f"_{args.secondary_exposure:g}Y"))
            solar_fit_at_react = reactor_fit_at_solar = None
            for _exp, _tag in _exposures:
                rprint(f"[cyan][INFO][/cyan] Evaluating exposure {_exp:g} yr"
                       + (f" (secondary -> tag '{_tag}')" if _tag else " (primary)"))
                _sfar, _rfas = _run_exposure(_exp, _tag)
                if not _tag:
                    solar_fit_at_react, reactor_fit_at_solar = _sfar, _rfas

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
                filename=f"highest_SENSITIVITY{_study_suffix}",
                rm=args.rewrite,
                debug=args.debug,
            )
            rprint(
                f"[cyan][INFO][/cyan] SENSITIVITY Cuts: NHits={best['NHits']} AdjCl={best['AdjCl']} OpHits={best['OpHits']} (score={best['Score']:.2f})"
            )

            json_payload: dict = {}
            for (cfg, nm, en), values in best_payload.items():
                # Flatten to config/energy level (all samples combined)
                json_payload.setdefault(cfg, {}).setdefault(en, {}).update(values)
            _json_stem = f"highest_Sensitivity{_study_suffix}"
            for local_dir in [
                f"{root}/config/{config}/sensitivity-json/{args.folder.lower()}",
                f"{root}/config/{config}/best-sigma-json/sensitivity/{args.folder.lower()}",
            ]:
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                merge_and_write_json(
                    f"{local_dir}/{config}_{_json_stem}.json",
                    json_payload,
                    debug=args.debug,
                )
