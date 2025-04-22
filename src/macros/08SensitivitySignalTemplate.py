import sys, json

sys.path.insert(0, "../../")
from lib import *

from lib.osc_functions import get_oscillation_datafiles

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
    choices=["DayNight", "HEP"],
    default="HEP",
    required=True,
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
    default="Reduced",
    choices=["Reduced", "Nominal"],
)
parser.add_argument(
    "--signal_uncertanty",
    type=float,
    help="The signal uncertanty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertanty",
    type=float,
    help="The background uncertanty for the analysis",
    default=0.02,
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
    default=["Cluster", "Total", "Selected", "Solar"],
)
parser.add_argument(
    "--fiducial", type=int, help="The fiducial cut for the analysis", default=None
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcl", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

folder = args.folder

signal_uncertanty = args.signal_uncertanty
background_uncertanty = args.background_uncertanty

configs = {args.config: [args.name]}

run, output = load_multi(
    configs,
    preset="ANALYSIS",
    branches={"Config": ["Geometry"]},
    debug=args.debug,
)
rprint(output)
run = compute_reco_workflow(run, configs, workflow="ANALYSIS", debug=args.debug)
run, output, this_new_branches = compute_particle_weights(
    run,
    configs,
    {"DEFAULT_SIGNAL_WEIGHT": ["truth"]},
    rm_branches=True,
    output=output,
    debug=args.debug,
)

for args.config in configs:
    info = json.loads(
        open(f"{root}/config/{args.config}/{args.config}_config.json").read()
    )
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))

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

    fastest_sigma = pickle.load(
        open(
            f"{info['PATH']}/{args.reference.upper()}/{folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_highest_{args.reference}.pkl",
            "rb",
        )
    )

    for idx, key in enumerate(fastest_sigma):
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
        if args.fiducial is not None:
            fiducial = args.fiducial
        else:
            rprint(f"Using optimized fiducial cut {fastest_sigma[key]['Fiducialized']}")
            fiducial = int(fastest_sigma[key]["Fiducialized"])

        if args.nhits is not None:
            nhits = args.nhits
        else:
            rprint(f"Using optimized nhits {fastest_sigma[key]['NHits']}")
            nhits = int(fastest_sigma[key]["NHits"])

        if args.adjcl is not None:
            adjcl = args.adjcl
        else:
            rprint(f"Using optimized adjcl {fastest_sigma[key]['AdjCl']}")
            adjcl = int(fastest_sigma[key]["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            rprint(f"Using optimized ophits {fastest_sigma[key]['OpHits']}")
            ophits = int(fastest_sigma[key]["OpHits"])

        this_filter = np.where(
            (run["Reco"]["NHits"] > fastest_sigma[key]["NHits"] - 1)
            & (run["Reco"]["AdjClNum"] < fastest_sigma[key]["AdjCl"])
            & (run["Reco"]["MatchedOpFlashNHits"] > fastest_sigma[key]["OpHits"] - 1)
            & (
                (
                    (run["Reco"]["RecoX"] > -detector_x / 2)
                    & (run["Reco"]["RecoX"] < -0.1 * fiducial)
                )
                + (
                    (run["Reco"]["RecoX"] > 0.1 * fiducial)
                    & (run["Reco"]["RecoX"] < detector_x / 2)
                )
            )
            & (
                (
                    (run["Reco"]["RecoY"] > -detector_y / 2)
                    & (run["Reco"]["RecoY"] < -fiducial)
                )
                + (
                    (run["Reco"]["RecoY"] > fiducial)
                    & (run["Reco"]["RecoY"] < detector_y / 2)
                )
            )
            & (run["Reco"]["RecoZ"] > fiducial - info["DETECTOR_GAP_Z"])
            & (
                run["Reco"]["RecoZ"]
                < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fiducial
            )
        )

        print(
            f"Selected #Events: {len(this_filter[0])} ({len(this_filter[0])/len(run['Reco']['Event'])*100:.2f}%)"
        )

        title = f"{key[2]}Energy Signal (Fiducial {fiducial} cm / min #NHits {nhits} / max #AdjClusters {adjcl} / min #OpHits {ophits})"
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{key[2]}Energy"][this_filter],
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
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{key[2]}Energy"][this_filter],
            run["Reco"]["SignalParticleK"][this_filter],
            bins=(energy_edges, energy_edges),
            weights=run["Reco"]["SignalParticleWeight"][this_filter],
        )
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
                config = args.config,
                name = f"marley",
                subfolder=f"{folder.lower()}/{key[2]}Energy",
                filename=f"Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
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
            save_path,
            config = args.config,
            name = None,
            subfolder = None,
            filename = f"Selected_Signal_{key[2]}Energy_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.debug,
        )
