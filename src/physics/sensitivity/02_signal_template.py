import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

from lib.oscillation import get_oscillation_datafiles

save_path = f"{root}/output/images/analysis/sensitivity/templates"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
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
    "--name", type=str, help="The name of the configuration", default="marley"
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
    "--exposure",
    type=float,
    help="The exposure for the analysis",
    default=30,
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
parser.add_argument("--test", action=argparse.BooleanOptionalAction)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help="Oscillation backend for signal template convolution. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)
parser.add_argument(
    "--scan_mode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Restrict oscillation grid to solar+reactor reference points only. "
         "Fast path for cut optimisation; use without --scan_mode for the full analysis grid.",
)
parser.add_argument(
    "--cuts",
    type=str,
    default=None,
    help='JSON list of cut dicts [{"NHits":N,"AdjCl":A,"OpHits":O}, ...]. '
         "When provided, processes all listed cuts in a single invocation.",
)

args = parser.parse_args()

smoothing_config = get_smoothing_config(
    str(root), analysis_name="SENSITIVITY", dimensions="2d", stage="significance"
)
smoothing_config = dict(smoothing_config)
smoothing_config["params"] = dict(smoothing_config.get("params", {}))
smoothing_config["params"]["sigma_y"] = 0.0
smoothing_info = smoothing_metadata(smoothing_config)


def _resolve_signal_smoothing_config(config: dict):
    """Return effective smoothing config for sensitivity signal templates."""
    signal_labels = ["signal", "solar", "8B", "hep", "marley"]
    component_configs = [get_component_smoothing_config(config, label) for label in signal_labels]
    active_configs = [item for item in component_configs if str(item.get("method", "none")).lower() != "none"]

    if active_configs:
        return active_configs[0], signal_labels

    fallback = get_component_smoothing_config(config, "signal")
    fallback["enabled"] = False
    fallback["method"] = "none"
    return fallback, signal_labels


signal_smoothing_config, signal_smoothing_labels = _resolve_signal_smoothing_config(smoothing_config)
signal_smoothing_active = str(signal_smoothing_config.get("method", "none")).lower() != "none"

if signal_smoothing_active:
    rprint(
        "[yellow][WARNING][/yellow] Signal smoothing is ACTIVE in sensitivity/02_signal_template.py. "
        "This can wash out solar-neutrino wiggles relevant for sensitivity significance. "
        f"method={signal_smoothing_config.get('method')} params={signal_smoothing_config.get('params', {})} "
        f"mode={smoothing_info.get('SmoothingComponentMode', 'all')} "
        f"components={smoothing_info.get('SmoothingComponents', [])} "
        f"checked_labels={signal_smoothing_labels}"
    )
elif args.debug:
    rprint(
        "[cyan][INFO][/cyan] Signal smoothing is disabled for sensitivity templates; "
        "using raw oscillated signal to preserve fine-structure wiggles."
    )


def _load_best_cut_map(info: dict, args):
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    tried = []
    for analysis in candidates:
        filepath = (
            f"{info['PATH']}/{analysis}/{args.folder.lower()}/{args.config}/{args.name}/"
            f"{args.config}_{args.name}_highest_{analysis}.pkl"
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

folder = args.folder
configs = {args.config: [args.name]}

for path in [save_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

run, output = load_multi(
    configs,
    preset="SIGNIFICANCE",
    branches={"Config": ["Geometry"]},
    debug=args.debug,
)
if args.debug:
    rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
        "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
        "PARTICLE_TYPE": "signal",
        "PARTICLE_WEIGHTING": "volume",
        "OSCILLATION_BACKEND": args.oscillation_backend,
    } if "marley" in args.name else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"},
    workflow="SIGNIFICANCE",
    rm_branches=False,
    debug=args.debug)

for config in configs:
    info = json.loads(
        open(f"{root}/config/{config}/{config}_config.json").read()
    )
    fiducials = json.loads(open(f"{root}/output/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())
    selected_fiducial = get_best_fiducial(fiducials, config, args.energy, "SENSITIVITY")
    selected_fiducial_bands = get_best_fiducial_bands(fiducials, config, args.energy, "SENSITIVITY")
    analysis_info = load_analysis_info(str(root))
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    _solar_dm2 = analysis_info["SOLAR_DM2"]
    _react_dm2 = analysis_info["REACT_DM2"]
    _sin13     = analysis_info["SIN13"]
    _sin12     = analysis_info["SIN12"]

    if args.scan_mode:
        _ref_dm2s  = list(dict.fromkeys([_solar_dm2, _react_dm2]))
        dm2_list   = _ref_dm2s
        sin13_list = [_sin13] * len(_ref_dm2s)
        sin12_list = [_sin12] * len(_ref_dm2s)
    elif args.oscillation_backend == "file":
        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            dm2=None,
            sin13=None,
            sin12=None,
            path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
            ext="pkl",
            auto=args.test == False,
            debug=args.debug,
        )
    else:
        # nufast/prob3 full mode: build grid from OSCILLATION_GRID config (no file dependency)
        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            backend=args.oscillation_backend,
            debug=args.debug,
        )

    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    energy = args.energy

    # Build cuts_to_process
    if args.cuts is not None:
        cuts_to_process = json.loads(args.cuts)
    elif args.nhits is not None and args.adjcls is not None and args.ophits is not None:
        cuts_to_process = [{"NHits": args.nhits, "AdjCl": args.adjcls, "OpHits": args.ophits}]
    else:
        loaded = _load_best_cut_map(info, args)
        if loaded is not None:
            cuts_to_process = [
                {"NHits": int(v["NHits"]), "AdjCl": int(v["AdjCl"]), "OpHits": int(v["OpHits"])}
                for v in loaded.values() if v is not None
            ]
        else:
            cuts_to_process = [{"NHits": 4, "AdjCl": 10, "OpHits": 4}]
            rprint("[yellow][WARNING][/yellow] Falling back to default cuts NHits4 AdjCl10 OpHits4")

    # ── Phase 1: pre-compute per-cut histograms ──────────────────────────────────
    cut_data = []  # (nhits, adjcl, ophits, h, fig, title)
    for cut in cuts_to_process:
        nhits  = int(cut["NHits"])
        adjcl  = int(cut["AdjCl"])
        ophits = int(cut["OpHits"])

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Unweighted Smearing",
                "Solar Weighted Smearing",
                f"Oscillated ({'Raw' if not signal_smoothing_active else 'Smoothed'})",
            ),
            shared_xaxes=True,
            shared_yaxes=True,
        )

        quality_mask = (
            (
                (run["Reco"]["SignalParticleSurface"] >= 0)
                & (run["Reco"]["SignalParticleSurface"] < 3)
                if args.name.split("_")[0] in ["gamma", "neutron"]
                else np.ones(len(run["Reco"]["NHits"]), dtype=bool)
            )
            & ((run["Reco"]["SignalParticleSurface"] < 3) if (args.folder in ["Reduced", "Truncated"] and args.name.split("_")[0] in ["gamma", "neutron"]) else np.ones(len(run["Reco"]["NHits"]), dtype=bool))
            & (run["Reco"]["NHits"] > nhits - 1)
            & (run["Reco"]["AdjClNum"] < adjcl)
            & (run["Reco"]["MatchedOpFlashPE"] > 0)
            & (run["Reco"]["MatchedOpFlashNHits"] > ophits - 1)
        )
        spatial_mask = build_energy_band_spatial_mask(
            run, config, detector_x, detector_y, info, args.folder,
            selected_fiducial, selected_fiducial_bands, energy,
        )
        this_filter = np.where(quality_mask & spatial_mask)

        if args.debug:
            print(f"Selected #Events: {len(this_filter[0])} ({len(this_filter[0])/len(run['Reco']['Event'])*100:.2f}%)")

        title = f"{energy} Signal (min #NHits {nhits} / max #AdjClusters {adjcl} / min #OpHits {ophits})"
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{energy}"][this_filter],
            run["Reco"]["SignalParticleK"][this_filter],
            bins=(energy_edges, energy_edges),
        )
        if args.debug:
            print(f"# of events (counts): {np.sum(h)}")
        fig.add_trace(
            go.Heatmap(z=np.log10(h), x=energy_centers, y=energy_centers, colorscale="Turbo", coloraxis="coloraxis"),
            row=1, col=1,
        )
        for weight in ["B8", "hep", ""]:
            h, xedges, yedges = np.histogram2d(
                run["Reco"][f"{energy}"][this_filter],
                run["Reco"]["SignalParticleK"][this_filter],
                bins=(energy_edges, energy_edges),
                weights=run["Reco"][f"SignalParticleWeight{weight.lower()}"][this_filter],
            )
            if args.debug:
                print(f"# of weighted events (counts) {(weight if weight != '' else 'solar')}: {np.sum(h):.2f}")

        h[args.exposure * detector_mass * h < 1] = np.nan
        fig.add_trace(
            go.Heatmap(z=np.log10(h), x=energy_centers, y=energy_centers, colorscale="Turbo", coloraxis="coloraxis"),
            row=1, col=2,
        )
        h = np.nan_to_num(h, nan=0.0)

        cut_data.append((nhits, adjcl, ophits, h, fig, title))

    # ── Phase 2: oscillation loop (outer) × cuts loop (inner) ───────────────────
    for dm2, sin13, sin12 in track(
        zip(dm2_list, sin13_list, sin12_list),
        total=len(dm2_list),
        description="Convolving oscillation files...",
    ):
        if args.debug:
            rprint(f"dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}")

        if args.oscillation_backend == "file":
            oscillation_df = pd.read_pickle(
                f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
            )
        else:
            from lib.oscillation import get_oscillation_map
            osc_map = get_oscillation_map(
                backend=args.oscillation_backend,
                dm2=[float(dm2)],
                sin13=[float(sin13)],
                sin12=[float(sin12)],
                output="df",
                debug=args.debug,
            )
            oscillation_df = next(iter(osc_map.values()))

        is_solar_bf = (
            dm2 == analysis_info["SOLAR_DM2"]
            and sin13 == analysis_info["SIN13"]
            and sin12 == analysis_info["SIN12"]
        )

        for nhits, adjcl, ophits, h, fig, title in cut_data:
            convolved = np.dot(oscillation_df.values, h.T)
            rebin_x, rebin_y, rebin_z, rebin_z_per_x = rebin_hist2d(
                energy_centers,
                np.asarray(list(oscillation_df.index)),
                convolved,
                sensitivity_rebin,  # type: ignore[arg-type]
            )

            if args.debug:
                rprint(
                    f"[cyan][INFO][/cyan] Saving signal template NHits{nhits} AdjCl{adjcl} OpHits{ophits} "
                    f"dm2={dm2:.3e} sin13={sin13:.3e} sin12={sin12:.3e}"
                )
            save_pkl(
                args.exposure * detector_mass * rebin_z,
                f"{info['PATH']}/SENSITIVITY",
                config=args.config,
                name=args.name,
                subfolder=f"{folder.lower()}/{energy}",
                filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                rm=args.rewrite,
                debug=args.debug,
            )

            if is_solar_bf:
                if args.debug:
                    rprint(f"# of events (kT·year): {np.sum(convolved):.2f}")

                _osc_fig = make_subplots(rows=1, cols=1)
                _osc_fig.add_trace(go.Heatmap(
                    z=oscillation_df.values,
                    x=[float(c) for c in oscillation_df.columns],
                    y=list(oscillation_df.index),
                    colorscale="Viridis",
                    colorbar=dict(title="P(νe→νe)"),
                ), row=1, col=1)
                _osc_fig = format_coustom_plotly(
                    _osc_fig,
                    title=f"Solar Oscillogram {config} (Δm²={dm2:.2e} eV², sin²θ₁₂={sin12:.3f})",
                )
                _osc_fig.update_xaxes(title="True Neutrino Energy (MeV)")
                _osc_fig.update_yaxes(title="cos(η) Zenith Angle")
                save_figure(
                    _osc_fig, save_path, config=args.config, name=args.name, subfolder=args.folder.lower(),
                    filename=f"Oscillogram_{energy}", rm=args.rewrite, debug=args.plot,
                )

                _signal_1d = args.exposure * detector_mass * np.sum(rebin_z, axis=0)
                _sig_fig = make_subplots(rows=1, cols=1)
                _sig_fig.add_trace(go.Scatter(
                    x=sensitivity_rebin_centers,
                    y=_signal_1d,
                    mode="lines",
                    fill="tozeroy",
                    line_shape="hvh",
                    name="Solar ν Signal",
                ), row=1, col=1)
                _sig_fig = format_coustom_plotly(
                    _sig_fig, title=f"1D Signal Spectrum {config} {energy} ({args.exposure} kt·yr)",
                )
                _sig_fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)")
                _sig_fig.update_yaxes(title="Events (kt·yr)⁻¹ MeV⁻¹")
                save_figure(
                    _sig_fig, save_path, config=args.config, name=args.name, subfolder=args.folder.lower(),
                    filename=f"Signal1D_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}", rm=args.rewrite, debug=args.plot,
                )

                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(np.where(rebin_z > 0, rebin_z, np.nan)) if not signal_smoothing_active else np.log10(smooth_histogram_with_config(rebin_z, signal_smoothing_config)),
                        x=sensitivity_rebin,
                        y=rebin_y,
                        colorscale="Turbo",
                        coloraxis="coloraxis",
                    ),
                    row=1, col=3,
                )

    # ── Phase 3: format and save per-cut figures ─────────────────────────────────
    for nhits, adjcl, ophits, h, fig, title in cut_data:
        fig = format_coustom_plotly(
            fig,
            title=title,
            legend=dict(x=0.5, y=0.99),
            tickformat=(".0f", ".0f"),
            matches=("x", None),
        )
        fig.update_yaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=1)
        fig.update_yaxes(title="", row=1, col=2)
        fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=3)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=1)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=2)
        fig.update_yaxes(title="Azimuth con(" + unicode("eta") + ")", row=1, col=3)
        fig.update_layout(coloraxis=dict(colorbar=dict(title="log(Counts)")))
        save_figure(
            fig, f"{save_path}",
            config=args.config, name=args.name, subfolder=f"{args.folder.lower()}",
            filename=f"Selected_Signal_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite, debug=args.plot,
        )
