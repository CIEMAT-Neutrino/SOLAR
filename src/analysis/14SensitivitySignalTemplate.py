import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

from lib.lib_osc import get_oscillation_datafiles

save_path = f"{root}/images/analysis/sensitivity/templates"

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
        "[yellow][WARNING][/yellow] Signal smoothing is ACTIVE in 14SensitivitySignalTemplate.py. "
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
        "DEFAULT_SIGNAL_AZIMUTH": ["mean", "day", "night"],
        "PARTICLE_TYPE": "signal",
        "PARTICLE_WEIGHTING": "volume",
    } if "marley" in args.name else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"},
    workflow="SIGNIFICANCE",
    rm_branches=False,
    debug=args.debug)

for config in configs:
    info = json.loads(
        open(f"{root}/config/{config}/{config}_config.json").read()
    )
    fiducials = json.loads(open(f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())
    selected_fiducial = get_best_fiducial(fiducials, config, args.energy, "SENSITIVITY")
    selected_fiducial_bands = get_best_fiducial_bands(fiducials, config, args.energy, "SENSITIVITY")
    analysis_info = load_analysis_info(str(root))
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
        dm2=None,
        sin13=None,
        sin12=None,
        path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
        ext="pkl",
        auto=args.test == False,
        debug=args.debug,
    )

    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    fastest_sigma = {(args.config, args.name, args.energy): None}
    if args.nhits is None or args.adjcls is None or args.ophits is None:
        loaded = _load_best_cut_map(info, args)
        if loaded is not None:
            fastest_sigma = loaded
        else:
            fastest_sigma = {
                (args.config, args.name, args.energy): {
                    "NHits": 4,
                    "AdjCl": 10,
                    "OpHits": 4,
                }
            }
            rprint(
                "[yellow][WARNING][/yellow] Falling back to default cuts NHits4 AdjCl10 OpHits4"
            )

    cut_keys = (
        list(fastest_sigma.keys())
        if args.nhits is None or args.adjcls is None or args.ophits is None
        else [(args.config, args.name, args.energy)]
    )

    for idx, key in enumerate(cut_keys):
        fig = make_subplots(
            rows=1,
            cols=5,
            subplot_titles=(
                "Unweighted Smearing",
                "Solar Weighted Smearing",
                "Oscillated Raw",
                "Oscillated Smoothed",
                "Residual (Smoothed - Raw)",
            ),
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.1,
        )
        if args.energy is not None:
            energy = args.energy
        else:
            energy = key[2]

        if args.nhits is not None:
            nhits = args.nhits
        else:
            selected = fastest_sigma.get(key) or {}
            if args.debug:
                rprint(f"Using optimized nhits {selected['NHits']}")
            nhits = int(selected["NHits"])

        if args.adjcls is not None:
            adjcl = args.adjcls
        else:
            selected = fastest_sigma.get(key) or {}
            if args.debug:
                rprint(f"Using optimized adjcl {selected['AdjCl']}")
            adjcl = int(selected["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            selected = fastest_sigma.get(key) or {}
            if args.debug:
                rprint(f"Using optimized ophits {selected['OpHits']}")
            ophits = int(selected["OpHits"])

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
            print(
                f"Selected #Events: {len(this_filter[0])} ({len(this_filter[0])/len(run['Reco']['Event'])*100:.2f}%)"
            )

        title = f"{energy} Signal (min #NHits {nhits} / max #AdjClusters {adjcl} / min #OpHits {ophits})"
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{energy}"][this_filter],
            run["Reco"]["SignalParticleK"][this_filter],
            bins=(energy_edges, energy_edges),
        )
        if args.debug:
            print(f"# of events (counts): {np.sum(h)}")
        fig.add_trace(
            go.Heatmap(
                z=np.log10(h),
                x=energy_centers,
                y=energy_centers,
                colorscale="Turbo",
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
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
            go.Heatmap(
                z=np.log10(h),
                x=energy_centers,
                y=energy_centers,
                colorscale="Turbo",
                coloraxis="coloraxis",
            ),
            row=1,
            col=2,
        )
        # Substitute the nan values with 0
        h = np.nan_to_num(h, nan=0.0)

        for dm2, sin13, sin12 in track(
            zip(dm2_list, sin13_list, sin12_list),
            total=len(dm2_list),
            description="Convolving oscillation files...",
        ):
            if args.debug:
                rprint(f"dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}")

            oscillation_df = pd.read_pickle(
                f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
            )

            convolved = np.dot(oscillation_df.values, h.T)

            rebin_x, rebin_y, rebin_z, rebin_z_per_x = rebin_hist2d(
                energy_centers,
                np.asarray(list(oscillation_df.index)),
                convolved,
                sensitivity_rebin,  # type: ignore[arg-type]
            )

            # Normalize raw and smoothed templates consistently to avoid fake residuals.
            # raw_rebin_z_per_x = np.divide(
            #     rebin_z,
            #     np.sum(rebin_z, axis=0, keepdims=True),
            #     out=np.zeros_like(rebin_z),
            #     where=np.sum(rebin_z, axis=0, keepdims=True) != 0,
            # )

            if args.debug:
                rprint(
                    f"[cyan][INFO][/cyan] Saving sensitivity signal template for {config} {folder} {energy} with NHits{nhits} AdjCl{adjcl} OpHits{ophits} dm2={dm2:.3e} sin13={sin13:.3e} sin12={sin12:.3e}"
                )
            save_pkl(
                args.exposure * detector_mass * rebin_z,
                f"{info['PATH']}/SENSITIVITY",
                config=args.config,
                name=f"marley",
                subfolder=f"{folder.lower()}/{energy}",
                filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                rm=args.rewrite,
                debug=args.debug,
            )

            if (
                dm2 == analysis_info["SOLAR_DM2"]
                and sin13 == analysis_info["SIN13"]
                and sin12 == analysis_info["SIN12"]
            ):
                if args.debug:
                    rprint(f"# of events (kT·year): {np.sum(convolved):.2f}")

                if signal_smoothing_active:
                    smoothed_rebin_z = smooth_histogram_with_config(rebin_z, signal_smoothing_config)
                else:
                    smoothed_rebin_z = rebin_z.copy()
                
                # smoothed_rebin_z_per_x = np.divide(
                #     smoothed_rebin_z,
                #     np.sum(smoothed_rebin_z, axis=0, keepdims=True),
                #     out=np.zeros_like(smoothed_rebin_z),
                #     where=np.sum(smoothed_rebin_z, axis=0, keepdims=True) != 0,
                # )
                residual_rebin_z = smoothed_rebin_z - rebin_z

                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(np.where(rebin_z > 0, rebin_z, np.nan)),
                        x=sensitivity_rebin,
                        y=rebin_y,
                        colorscale="Turbo",
                        coloraxis="coloraxis",
                    ),
                    row=1,
                    col=3,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(np.where(smoothed_rebin_z > 0, smoothed_rebin_z, np.nan)),
                        x=sensitivity_rebin,
                        y=rebin_y,
                        colorscale="Turbo",
                        coloraxis="coloraxis",
                        showscale=False,
                    ),
                    row=1,
                    col=4,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=residual_rebin_z,
                        x=sensitivity_rebin,
                        y=rebin_y,
                        colorscale="RdBu",
                        zmid=0.0,
                        showscale=False,
                    ),
                    row=1,
                    col=5,
                )

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
        fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=4)
        fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=5)

        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=1)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=2)
        fig.update_yaxes(title="Azimuth con(" + unicode("eta") + ")", row=1, col=3)
        fig.update_yaxes(title="", row=1, col=4)
        fig.update_yaxes(title="", row=1, col=5)
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title="log(Counts)",
                ),
            )
        )

        save_figure(
            fig,
            f"{save_path}",
            config=args.config,
            name=args.name,
            subfolder=f"{args.folder.lower()}",
            filename=f"Selected_Signal_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.plot,
        )

        if args.energy is not None and isinstance(args.energy, str):
            break
