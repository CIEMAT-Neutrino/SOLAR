import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

from lib.lib_osc import get_oscillation_datafiles

save_path = f"{root}/images/solar/fit"

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
    default="HEP",
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
    "--exposure",
    type=float,
    help="The exposure for the analysis",
    default=400,
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
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

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
    analysis_info = json.load(open(f"{root}/import/analysis.json", "r"))

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

    if args.nhits is None or args.adjcls is None or args.ophits is None:
        fastest_sigma = pickle.load(
            open(
                f"{info['PATH']}/{args.reference.upper()}/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_highest_{args.reference}.pkl",
                "rb",
            )
        )

    for idx, key in enumerate(fastest_sigma if args.nhits is None or args.adjcls is None or args.ophits is None else [{(args.config, args.name, args.energy):None}]):
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Unweighted Semearing",
                "Solar Weighted Smearing",
                "Solar Weighted Oscillated",
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
            rprint(f"Using optimized nhits {fastest_sigma[key]['NHits']}")
            nhits = int(fastest_sigma[key]["NHits"])

        if args.adjcls is not None:
            adjcl = args.adjcls
        else:
            rprint(f"Using optimized adjcl {fastest_sigma[key]['AdjCl']}")
            adjcl = int(fastest_sigma[key]["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            rprint(f"Using optimized ophits {fastest_sigma[key]['OpHits']}")
            ophits = int(fastest_sigma[key]["OpHits"])

        this_filter = np.where(
            (run["Reco"]["SignalParticleSurface"] >= 0 if args.name.split("_")[0] in ["gamma", "neutron"] else 1)
            & ((run["Reco"]["SignalParticleSurface"] < 3) if (args.folder in ["Reduced", "Truncated"] and args.name.split("_")[0] in ["gamma", "neutron"]) else 1)
            & (run["Reco"]["NHits"] > nhits - 1)
            & (run["Reco"]["AdjClNum"] < adjcl)
            & (run["Reco"]["MatchedOpFlashPE"] > 0)
            & (run["Reco"]["MatchedOpFlashNHits"] > ophits - 1)
            & (
                np.absolute(run["Reco"]["RecoX"])
                > fiducials[config][energy]["FiducialX"]
                if config == "hd_1x2x6_lateralAPA"
                else np.absolute(run["Reco"]["RecoX"])
                < detector_x / 2
                - fiducials[config][energy]["FiducialX"] 
                if config == "hd_1x2x6_centralAPA"
                else run["Reco"]["RecoX"]
                < detector_x / 2
                - fiducials[config][energy]["FiducialX"]
            )
            & (
                np.absolute(run["Reco"]["RecoY"])
                < detector_y / 2 - fiducials[config][energy]["FiducialY"]
            )
            & (
                (run["Reco"]["RecoZ"]
                > fiducials[config][energy]["FiducialZ"] - info["DETECTOR_GAP_Z"]) if args.folder == "Nominal" else 1
            )
            & (
                (run["Reco"]["RecoZ"]
                < info["DETECTOR_SIZE_Z"]
                + info["DETECTOR_GAP_Z"]
                - fiducials[config][energy]["FiducialZ"]) if args.folder == "Nominal" else 1
            )
        )

        print(
            f"Selected #Events: {len(this_filter[0])} ({len(this_filter[0])/len(run['Reco']['Event'])*100:.2f}%)"
        )

        title = f"{energy} Signal (min #NHits {nhits} / max #AdjClusters {adjcl} / min #OpHits {ophits})"
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{energy}"][this_filter],
            run["Reco"]["SignalParticleK"][this_filter],
            bins=(energy_edges, energy_edges),
        )
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
            print(f"# of weighted events (counts) {(weight if weight != '' else 'solar')}: {np.sum(h):.2f}")

        h[args.exposure * h < 1] = np.nan
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
            rprint(f"dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}")

            oscillation_df = pd.read_pickle(
                f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
            )

            convolved = np.dot(oscillation_df.values, h.T)

            rebin_x, rebin_y, rebin_z, rebin_z_per_x = rebin_hist2d(
                energy_centers,
                np.asarray(list(oscillation_df.index)),
                convolved,
                sensitivity_rebin,
            )

            save_pkl(
                args.exposure * rebin_z,
                f"{info['PATH']}/SENSITIVITY",
                config=args.config,
                name=f"marley",
                subfolder=f"{folder.lower()}/{energy}",
                filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                rm=args.rewrite,
                debug=args.test == False,
            )

            if (
                dm2 == analysis_info["SOLAR_DM2"]
                and sin13 == analysis_info["SIN13"]
                and sin12 == analysis_info["SIN12"]
            ):
                rprint(f"# of events (kT·year): {np.sum(convolved):.2f}")

                fig.add_trace(
                    go.Heatmap(
                        z=np.log10(rebin_z_per_x),
                        x=sensitivity_rebin,
                        y=rebin_y,
                        colorscale="Turbo",
                        coloraxis="coloraxis",
                    ),
                    row=1,
                    col=3,
                )

        fig = format_coustom_plotly(
            fig,
            title=title,
            legend=dict(x=0.5, y=0.99),
            tickformat=(".0f", ".0f"),
            matches=("x", None),
        )
        fig.update_yaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=1)
        fig.update_yaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=2)
        fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=3)

        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=1)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=2)
        fig.update_yaxes(title="Azimuth con(" + unicode("eta") + ")", row=1, col=3)
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title="log(Counts)",
                ),
            )
        )

        save_figure(
            fig,
            f"{save_path}/{args.folder.lower()}",
            config=args.config,
            name=None,
            subfolder=None,
            filename=f"Selected_Signal_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.debug,
        )

        if args.energy is not None and isinstance(args.energy, str):
            break
