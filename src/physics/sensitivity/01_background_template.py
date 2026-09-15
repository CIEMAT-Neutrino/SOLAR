import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


TEMPLATE_NORMALIZATION_FILE = "TEMPLATE_NORMALIZATION.json"
TEMPLATE_NORMALIZATION_VERSION = 2   # v2 = per-year (mass x rate); v1 = pre-scaled by exposure


def write_template_normalization(template_dir: str, detector_mass_kT: float) -> None:
    """Stamp a template directory as per-year so consumers can refuse stale v1 templates."""
    os.makedirs(template_dir, exist_ok=True)
    _marker = os.path.join(template_dir, TEMPLATE_NORMALIZATION_FILE)
    # PNFS/dCache is write-once: opening an existing file for writing raises
    # "Operation not permitted". Remove first, exactly as prepare_file_save() does.
    if os.path.exists(_marker):
        os.remove(_marker)
    with open(_marker, "w") as fh:
        json.dump({
            "version": TEMPLATE_NORMALIZATION_VERSION,
            "per_year": True,
            "units": "detector_mass_kT * rate (counts per year)",
            "detector_mass_kT": float(detector_mass_kT),
            "note": "Multiply by exposure_yr at load time (06_significance.py).",
        }, fh, indent=2)

from lib.root import Sensitivity_Fitter
from lib.oscillation import get_oscillation_datafiles
from lib.oscillation_backends import get_nadir_pdf_nufast
from lib.template_guards import clean_and_validate_template

save_path = f"{root}/output/images/analysis/sensitivity/templates"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and args.signal with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
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
    "--folder",
    type=str,
    help="The name of the results folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
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
    "--exposure",
    type=float,
    help="Exposure in years. Default from ANALYSIS_EXPOSURES['SENSITIVITY']['PRIMARY'] in config/analysis/config.json.",
    default=get_analysis_exposure(str(root), "Sensitivity"),
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK", "MainK",
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
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help="Oscillation backend. 'file' uses pre-computed pkl files for the nadir axis; 'prob3'/'nufast' derive it from config/analysis/physics.json.",
)
parser.add_argument("--study_label", type=str, default=None,
    help="Tag appended to template subfolder and used to locate labeled background Rebins for charge study variants.")
parser.add_argument("--charge_threshold", type=float, default=0,
    help="Charge threshold Q (ADC) forwarded by run_sensitivity.py. Background Rebins are not charge-labeled; this arg is accepted to avoid argparse errors.")

# NEW: Template type argument to distinguish background vs signal
parser.add_argument(
    "--template",
    type=str,
    choices=["background", "signal"],
    default="background",
    help="Template type: 'background' (must use all cuts) or 'signal' (can use best cut). Default: background.",
)

# NEW: Force all cuts flag for safety
parser.add_argument(
    "--force-all-cuts",
    action="store_true",
    default=False,
    help="Force generation for ALL cuts, ignoring any existing best-cut file. "
         "RECOMMENDED for background templates to prevent mismatches with signal templates.",
)
parser.add_argument(
    "--flyweight",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Flyweight mode flag (accepted but ignored for background templates).",
)
parser.add_argument(
    "--truth_fiducial",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Truth-fiducial study: build the background templates from the labeled truth-fiducial "
        "Rebin pkls written by 03_analysis.py --truth_fiducial (truth positions and "
        "BestFiducials_fiduc_truth.json applied to every background sample)."
    ),
)
parser.add_argument(
    "--membrane_veto",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Accepted for parity with 02_signal_template.py: 03_template_compute.py forwards "
        "--no-membrane_veto to both template scripts. The plane selection itself is already "
        "baked into the Rebin pkls this script reads, but study_context needs the flag so "
        "membrane_veto_off background templates get their labeled subfolder instead of "
        "overwriting the nominal ones."
    ),
)

args = parser.parse_args()
_ctx = study_context(args, analysis="Sensitivity")
_study_suffix = _ctx.study_suffix
_template_suffix = _ctx.template_suffix
if args.debug:
    rprint(args)
# smoothing_config = get_smoothing_config(
#     str(root), analysis_name="SENSITIVITY", dimensions="2d", stage="significance"
# )
smoothing_config_1d = get_smoothing_config(
    str(root), analysis_name="SENSITIVITY", dimensions="1d", stage="significance"
)
# smoothing_config = dict(smoothing_config)
# smoothing_config["params"] = dict(smoothing_config.get("params", {}))
# smoothing_config["params"]["sigma_y"] = 0.0
smoothing_info = smoothing_metadata(smoothing_config_1d)


def _load_best_cut_map(info: dict, args):
    _suffix = f"_{args.study_label}" if getattr(args, 'study_label', None) else ""
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    tried = []
    
    # Standard location: {PATH}/SENSITIVITY/{config}/{name}/{folder}/...
    for analysis in candidates if not _suffix else ["SENSITIVITY"]:
        for suffix in ([_suffix] if _suffix else [""]):
            filepath = (
                f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/"
                f"{args.config}_{args.signal}_highest_{analysis}{suffix}.pkl"
            )
            tried.append(filepath)
            rprint(f"[cyan][INFO][/cyan] Checking best-cut map at: {filepath}")
            if os.path.exists(filepath):
                if args.debug:
                    rprint(f"[cyan][INFO][/cyan] Using best-cut map from {analysis}{suffix}")
                rprint(f"[cyan][INFO][/cyan] Loading best-cut map from: {filepath}")
                return pickle.load(open(filepath, "rb"))

    rprint(
        "[yellow][WARNING][/yellow] Unable to load any best-cut map. Checked:\n"
        + "\n".join(tried)
    )
    return None


def _load_best_cut_map_safe(info: dict, args, phase: str = "Template Generation"):
    """
    Load best-cut map with safety checks to prevent template mismatches.
    
    CRITICAL: For BACKGROUND templates, we must ensure we don't prematurely
    limit to a stale best cut. The pipeline order is:
      1. Generate background templates (ALL cuts)
      2. Run 04_best_cuts.py (selects best cut)
      3. Generate signal templates (best cut only)
    
    If background templates use a stale best cut, and 04_best_cuts.py then selects
    a NEW best cut, signal templates will be for the NEW cut while background
    templates are for the OLD cut -> MISMATCH in 06_significance.py.
    """
    
    # For BACKGROUND templates: ALWAYS warn about using best-cut file
    if args.template == "background":
        fastest_sigma = _load_best_cut_map(info, args)
        
        if fastest_sigma is not None and not args.force_all_cuts:
            rprint(
                f"[red][CRITICAL WARNING][/red] {phase}: BEST-CUT FILE EXISTS but we are generating "
                f"BACKGROUND templates. This can lead to a MISMATCH if 04_best_cuts.py selects "
                f"a different cut later in the pipeline. "
                f"Background templates should be generated for ALL cuts BEFORE best-cut selection. "
                f"Recommendation: Use --force-all-cuts to generate for all cuts (safe)."
            )
            
            # Additional check: if study_label is set, we must have the correct best-cut file
            if args.study_label:
                rprint(
                    f"[red][CRITICAL][/red] study_label '{args.study_label}' is set. "
                    f"If the best-cut file is stale, results will be inconsistent. "
                    f"Ensure 04_best_cuts.py has already run for this study label."
                )
        
        # If --force-all-cuts is set, or no best-cut file exists, use all cuts
        if args.force_all_cuts or fastest_sigma is None:
            return None
        else:
            return fastest_sigma
    
    # For SIGNAL templates: Using best-cut file is safe (and expected)
    else:
        fastest_sigma = _load_best_cut_map(info, args)
        if fastest_sigma is None and not args.force_all_cuts:
            rprint(
                "[yellow][WARNING][/yellow] No best-cut map found for signal templates. "
                "This may indicate that 04_best_cuts.py has not run yet. "
                "Signal templates require a best cut to be selected first."
            )
        return fastest_sigma


def _validate_cut_consistency(cut_entries, phase: str, args):
    """Warn about potential pipeline ordering issues."""
    if args.template == "background" and len(cut_entries) == 1:
        rprint(
            f"[red][CRITICAL][/red] {phase}: Generating background templates for ONLY ONE cut. "
            f"This is DANGEROUS if 04_best_cuts.py has not yet run. "
            f"Background templates must cover ALL cuts for best-cut selection to work correctly. "
            f"To force all cuts, use --force-all-cuts. "
            f"Current cut: {cut_entries[0]}"
        )
    elif args.template == "background" and len(cut_entries) > 1:
        rprint(
            f"[green][INFO][/green] {phase}: Generating background templates for {len(cut_entries)} cuts. "
            f"This is safe for best-cut selection."
        )


def _project_1d_to_2d(hist_1d, oscillation_df, nadir_pdf=None):
    """Project a 1D energy spectrum into a nadir-weighted 2D template.

    nadir_pdf: pre-computed normalised weights over oscillation_df.index.
    If None, loads from nadir.root via get_nadir_angle() (file backend).
    """
    hist_1d = np.asarray(hist_1d, dtype=float)
    hist2d = np.tile(hist_1d / len(oscillation_df), (len(oscillation_df), 1))

    if nadir_pdf is not None:
        rebin_nadir = np.asarray(nadir_pdf, dtype=float)
    else:
        nadir = get_nadir_angle()
        interp_nadir = interp1d(*nadir)
        rebin_nadir = interp_nadir(oscillation_df.index)

    hist2d = rebin_nadir * hist2d.T
    norm = np.sum(hist2d)
    if norm > 0:
        hist2d = np.sum(hist_1d) * hist2d.T / norm
    else:
        hist2d = hist2d.T
    return hist2d

for path in [save_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

plot_df = pd.DataFrame()
oscillation_df = pd.DataFrame()
background_samples = []
dm2_list, sin13_list, sin12_list = [], [], []

analysis_info = load_analysis_info(str(root))
info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
_fiducials_stem = "BestFiducials_fiduc_truth" if args.truth_fiducial else "BestFiducials"
fiducials = json.loads(open(f"{root}/config/analysis/fiducial/{args.folder.lower()}/{_fiducials_stem}.json").read())

detector_mass = get_full_detector_mass(args.config, info)

df_list = []
background_samples = []
# Read the labeled background Rebins whenever the variant changed the background event
# selection in 03_analysis.py: a charge threshold, or truth-position fiducialization (which
# re-cuts every background sample on its truth position and the truth best fiducials).
_bkg_selection_changed = getattr(args, "charge_threshold", 0) > 0 or args.truth_fiducial
_bkg_study_label = args.study_label if _bkg_selection_changed else None
for bkg, filepath in load_available_background_dataframes(str(root), "SENSITIVITY", args.folder, args.config, args.energy, study_label=_bkg_study_label):
    bkg_df = pd.read_pickle(filepath)
    df_list.append(bkg_df)
    background_samples.append(bkg)

plot_df = pd.concat(df_list, ignore_index=True)

plot_df = explode(
    plot_df, ["Counts", "Counts/Energy", "Error", "Energy"], debug=args.debug
)
plot_df["Counts"] = plot_df["Counts"].replace(0, np.nan)
plot_df["Counts/Energy"] = plot_df["Counts/Energy"].replace(0, np.nan)

if args.oscillation_backend == "file":
    (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
        dm2=None,
        sin13=None,
        sin12=None,
        path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
        ext="pkl",
    )
    for dm2, sin13, sin12 in product(dm2_list, sin13_list, sin12_list):
        oscillation_df = pd.read_pickle(
            f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
        )
else:
    nadir_edges = np.linspace(-1.0, 1.0, analysis_info["NADIR_BINS"] + 1)
    nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])
    oscillation_df = pd.DataFrame(index=nadir_centers)

# Pre-compute nadir PDF for _project_1d_to_2d — use NuFast-Earth Solar_Weight when
# backend is "nufast" so the run stays fully file-free; otherwise interpolate nadir.root.
_latitude_deg = analysis_info.get("DUNE_LATITUDE_DEG", 44.35)
_nadir_centers_arr = np.asarray(oscillation_df.index, dtype=float)
if args.oscillation_backend == "nufast":
    _nadir_pdf_weights = get_nadir_pdf_nufast(_nadir_centers_arr, _latitude_deg)
    _nadir_plot_x = _nadir_centers_arr
    _nadir_plot_y = _nadir_pdf_weights
else:
    _nadir_raw = get_nadir_angle()
    _nadir_plot_x = _nadir_raw[0]
    _nadir_plot_y = _nadir_raw[1]
    _nadir_pdf_weights = None  # _project_1d_to_2d will call get_nadir_angle() itself

# Panel 2: standalone nadir time-fraction distribution p(cos θ_z) at DUNE latitude
_nadir_fig = make_subplots(rows=1, cols=1)
_nadir_fig.add_trace(go.Scatter(
    x=list(_nadir_plot_x), y=list(_nadir_plot_y),
    mode="lines", fill="tozeroy",
    line=dict(color="steelblue", width=2),
    name="Time-Fraction",
), row=1, col=1)
_nadir_fig = format_coustom_plotly(
    _nadir_fig, title=f"Nadir Time-Fraction at DUNE (lat. {_latitude_deg}°)",
)
_nadir_fig.update_xaxes(title="cos(η) Zenith Angle")
_nadir_fig.update_yaxes(title="Time-Fraction per Bin")
save_figure(
    _nadir_fig, save_path, config=args.config, name=args.signal, subfolder=args.folder.lower(),
    filename="NadirDistribution", rm=args.rewrite, debug=args.plot,
)

cut_entries = []

# CRITICAL: Check for pipeline ordering issues
if args.template == "background":
    # Check if signal templates already exist (indicates 04_best_cuts.py already ran)
    signal_template_dir = (
        f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/"
        f"{args.energy}{_template_suffix}"
    )
    if os.path.exists(signal_template_dir):
        signal_template_files = [f for f in os.listdir(signal_template_dir) if f.startswith("signal_")]
        if signal_template_files:
            rprint(
                f"[red][CRITICAL][/red] Signal templates already exist in {signal_template_dir}. "
                f"This suggests 04_best_cuts.py has already run, but you are now generating "
                f"BACKGROUND templates. This is the WRONG ORDER and will cause a MISMATCH. "
                f"Background templates must be generated BEFORE 04_best_cuts.py."
            )

if args.nhits is not None and args.adjcls is not None and args.ophits is not None:
    cut_entries = [
        {
            "NHits": int(args.nhits),
            "AdjCl": int(args.adjcls),
            "OpHits": int(args.ophits),
        }
    ]
else:
    # SAFE LOAD: Use new function with warnings
    fastest_sigma = _load_best_cut_map_safe(info, args, phase="Background Template Generation")
    
    if fastest_sigma is not None and not args.force_all_cuts:
        cut_entries = [
            {
                "NHits": int(value["NHits"]),
                "AdjCl": int(value["AdjCl"]),
                "OpHits": int(value["OpHits"]),
            }
            for value in fastest_sigma.values()
        ]
        # CRITICAL: Validate this is safe
        _validate_cut_consistency(cut_entries, "Background Template Generation", args)
    elif args.force_all_cuts or fastest_sigma is None:
        # Fall back to all cuts
        available = sorted(
            {
                (int(row["NHits"]), int(row["AdjCl"]), int(row["OpHits"]))
                for _, row in plot_df[["NHits", "AdjCl", "OpHits"]].drop_duplicates().iterrows()
            },
            key=lambda x: (x[0], x[2], x[1]),
        )
        cut_entries = [
            {"NHits": nh, "AdjCl": ad, "OpHits": op}
            for nh, ad, op in available
        ]
        rprint(
            f"[cyan][INFO][/cyan] Using all {len(cut_entries)} cut triplets "
            f"(best-cut file missing or --force-all-cuts set)"
        )
    else:
        # Should not reach here
        available = sorted(
            {
                (int(row["NHits"]), int(row["AdjCl"]), int(row["OpHits"]))
                for _, row in plot_df[["NHits", "AdjCl", "OpHits"]].drop_duplicates().iterrows()
            },
            key=lambda x: (x[0], x[2], x[1]),
        )
        cut_entries = [
            {"NHits": nh, "AdjCl": ad, "OpHits": op}
            for nh, ad, op in available
        ]
        rprint(
            f"[yellow][WARNING][/yellow] Falling back to {len(cut_entries)} cut triplets discovered from background data"
        )

    # A study that optimises its own cuts has no map yet at this stage: Phase 1 is exactly
    # where its all-cut background templates are produced, and run_sensitivity.py asks for
    # that with --force-all-cuts. Only a study holding nominal cuts must never get here.
    if args.study_label and fastest_sigma is None and not args.force_all_cuts:
        raise FileNotFoundError(
            f"Missing best-cut map for study '{args.study_label}'; refusing to discover cuts from nominal background data"
        )

for idx, cut in enumerate(cut_entries):
    if args.energy is not None:
        energy = args.energy
    else:
        energy = args.energy

    total = np.zeros(len(sensitivity_rebin) - 1)
    total_error = np.zeros(len(sensitivity_rebin) - 1)

    nhits = int(cut["NHits"])
    adjcl = int(cut["AdjCl"])
    ophits = int(cut["OpHits"])

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=(
            [
                "Background Components",
                "Raw 2D Background",
                "Smoothed 2D Background",
                "Residual (Smoothed - Raw)",
            ]
        ),
    )

    component_energy = np.asarray(sensitivity_rebin_centers)

    for bkg in background_samples:
        this_df = plot_df[
            (plot_df["Component"] == bkg)
            * (plot_df["NHits"] == nhits)
            * (plot_df["OpHits"] == ophits)
            * (plot_df["AdjCl"] == adjcl)
        ]

        if this_df.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Empty dataframe for {bkg} with NHits{nhits} OpHits{ophits} AdjCl{adjcl}"
            )
            continue

        x = np.asarray(list(this_df["Energy"].values))
        component_energy = x
        y = np.asarray(list(this_df["Counts"].values))
        y_error = np.asarray(list(this_df["Error"].values))
        y = np.nan_to_num(y)
        y_error = np.nan_to_num(y_error)

        component_smoothing_config_1d = get_component_smoothing_config(
            smoothing_config_1d, bkg
        )
        y_smoothed = smooth_histogram_with_config(y, component_smoothing_config_1d)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=bkg,
                line_shape="hvh",
                line=dict(color=this_df["Color"].values[0], dash="dot", width=2),
                opacity=0.45,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_smoothed,
                mode="lines",
                name=bkg,
                line_shape="hvh",
                line=dict(color=this_df["Color"].values[0], dash="solid", width=3),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        if len(y) != len(total):
            rprint(f"[yellow][WARNING][/yellow] Bin count mismatch for {bkg}: expected {len(total)}, got {len(y)}. Stale pkl — regenerate with current sensitivity_rebin. Skipping.")
            continue
        total = total + np.array(y)
        total_error = total_error + y_error**2

    total_smoothed = smooth_histogram_with_config(total, smoothing_config_1d)

    if idx == 0:
        # Panel 5: standalone 1D background spectrum b(E_reco) for the best-selected cut
        _bkg_1d_fig = make_subplots(rows=1, cols=1)
        for _bkg in background_samples:
            _bkg_slice = plot_df[
                (plot_df["Component"] == _bkg)
                & (plot_df["NHits"] == nhits)
                & (plot_df["OpHits"] == ophits)
                & (plot_df["AdjCl"] == adjcl)
            ]
            if _bkg_slice.empty:
                continue
            _bkg_y = np.nan_to_num(np.asarray(list(_bkg_slice["Counts"].values)))
            _bkg_1d_fig.add_trace(go.Scatter(
                x=np.asarray(list(_bkg_slice["Energy"].values), dtype=float),
                y=_bkg_y,
                mode="lines", name=_bkg, line_shape="hvh",
                line=dict(color=_bkg_slice["Color"].values[0], width=2),
            ), row=1, col=1)
        _bkg_1d_fig.add_trace(go.Scatter(
            x=component_energy, y=total_smoothed,
            mode="lines", name="Total (smoothed)", line_shape="hvh",
            line=dict(color="black", dash="solid", width=3),
        ), row=1, col=1)
        _bkg_1d_fig = format_coustom_plotly(
            _bkg_1d_fig,
            title=f"1D Background Spectrum {args.config} {energy}",
            legend_title="Component",
        )
        _bkg_1d_fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)")
        _bkg_1d_fig.update_yaxes(title="Counts per Energy (kt·yr·MeV)⁻¹", type="log", range=[-1, 7])
        save_figure(
            _bkg_1d_fig, save_path, config=args.config, name=args.signal, subfolder=args.folder.lower(),
            filename=f"Background1D_{energy}", rm=args.rewrite, debug=args.plot,
        )

    bkg_hist = _project_1d_to_2d(total, oscillation_df, nadir_pdf=_nadir_pdf_weights)
    smoothed_bkg_hist = _project_1d_to_2d(total_smoothed, oscillation_df, nadir_pdf=_nadir_pdf_weights)

    # The '<1 expected event' mask is exposure dependent and is therefore applied
    # by 06_significance.py after scaling, not baked into the per-year template.
    residual_bkg_hist = smoothed_bkg_hist - bkg_hist
    
    if args.debug:
        print(f"Check Counts: {np.sum(total)} - {np.sum(bkg_hist)}")
        print(f"Smoothed counts: {np.sum(smoothed_bkg_hist)} using {smoothing_info['SmoothingMethod']}")
        rprint(
            f"[cyan][INFO][/cyan] Saving sensitivity background template for {args.config} {args.folder} {energy} with NHits{nhits} AdjCl{adjcl} OpHits{ophits}"
        )
    
    # Apply stability guards before saving: clean NaN/Inf, clip negative and extreme values
    _template_to_save = detector_mass * smoothed_bkg_hist
    _template_to_save = clean_and_validate_template(
        _template_to_save,
        label=f"Background template {args.config} {energy} NHits{nhits} AdjCl{adjcl} OpHits{ophits}",
        debug=args.debug,
    )
        
    save_pkl(
        _template_to_save,
        f"{info['PATH']}/SENSITIVITY",
        config=args.config,
        name=f"background",
        subfolder=f"{args.folder.lower()}/{energy}{_template_suffix}",
        filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.debug,
    )
    write_template_normalization(
        f"{info['PATH']}/SENSITIVITY/{args.config}/background/"
        f"{args.folder.lower()}/{energy}{_template_suffix}",
        detector_mass,
    )

    fig.add_trace(
        go.Heatmap(
            z=np.log10(np.where(bkg_hist > 0, bkg_hist, np.nan)),
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="Turbo",
            colorbar=dict(title="log(Counts)"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=np.log10(np.where(smoothed_bkg_hist > 0, smoothed_bkg_hist, np.nan)),
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="Turbo",
            showscale=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Heatmap(
            z=residual_bkg_hist,
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="RdBu",
            zmid=0.0,
            showscale=False,
        ),
        row=1,
        col=4,
    )

    fig.add_trace(
        go.Scatter(
            x=component_energy,
            y=total,
            error_y=dict(type="data", array=np.sqrt(total_error), visible=True),
            mode="lines",
            name="Total Raw",
            line_shape="hvh",
            line=dict(color="black", dash="dot", width=2),
            opacity=0.45,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=component_energy,
            y=total_smoothed,
            mode="lines",
            name="Total Smoothed",
            line_shape="hvh",
            line=dict(color="black", dash="solid", width=3),
        ),
        row=1,
        col=1,
    )

    add_histogram_style_legend_traces(
        fig,
        row=1,
        col=1,
        legend="legend2",
    )

    fig = format_coustom_plotly(
        fig,
        title=f"{energy} Background {args.config}",
        log=(False, False),
        matches=("x", None),
        tickformat=(".1f", ".0e"),
        legend_title="Component",
        debug=args.debug,
    )

    fig.update_layout(
        legend2=dict(x=0.12, y=0.94, bgcolor="rgba(255,255,255,0.7)"),
    )

    fig.update_xaxes(
        title=f"Reconstructed Energy (MeV)",
    )

    fig.update_yaxes(
        title=f"Counts per Energy (kT·year·MeV)⁻¹",
        type="log",
        range=[-1, 7],
        row=1,
        col=1,
    )

    save_figure(
        fig,
        f"{save_path}",
        config=args.config,
        name=args.signal,
        subfolder=args.folder.lower(),
        filename=f"Background_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.plot,
    )

    if args.energy is not None and isinstance(args.energy, str):
        break
