import os
import sys

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
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
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
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

config = args.config
name = args.name

rewrite = args.rewrite
debug = args.debug

configs = {config: [name]}

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    fiducials = json.loads(open(f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())
    analysis_info = json.load(open(f"{root}/import/analysis.json", "r"))

    for name in configs[config]:
        df_list = []
        # signal_df = pd.read_pickle(
        #     f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/{args.reference}/{config}/{name}/{config}_{name}_rebin.pkl"
        # )
        # df_list.append(signal_df)
        for bkg, bkg_label in [
            ("neutron", "neutron"),
            ("gamma", "gamma"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder.lower()}/SENSITIVITY/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
            )
            df_list.append(bkg_df)

        plot_df = pd.concat(df_list, ignore_index=True)

        plot_df = explode(
            plot_df, ["Counts", "Counts/Energy", "Error", "Energy"], debug=args.debug
        )
        plot_df["Counts"] = plot_df["Counts"].replace(0, np.nan)
        plot_df["Counts/Energy"] = plot_df["Counts/Energy"].replace(0, np.nan)

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

    if args.nhits is None or args.adjcls is None or args.ophits is None:
        fastest_sigma = pickle.load(
            open(
                f"{info['PATH']}/{args.reference.upper()}/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_highest_{args.reference}.pkl",
                "rb",
            )
        )

    for idx, key in enumerate(fastest_sigma if args.nhits is None or args.adjcls is None or args.ophits is None else [{(args.config, args.name, args.energy):None}]):
        if args.energy is not None:
            energy = args.energy
        else:
            energy = key[2]

        total = np.zeros(len(sensitivity_rebin) - 1)
        total_error = np.zeros(len(sensitivity_rebin) - 1)

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

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=(
                [
                    "Background Components",
                    f"min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}",
                ]
            ),
        )

        for bkg in ["gamma", "neutron"]:
            # print(plot_df)
            this_df = plot_df[
                (plot_df["Component"] == bkg)
                * (plot_df["EnergyLabel"] == energy.split("Energy")[0])
                * (plot_df["NHits"] == nhits)
                * (plot_df["OpHits"] == ophits)
                * (plot_df["AdjCl"] == adjcl)
            ]
            
            print(this_df.explode("Counts").groupby(["Component", "Oscillation", "Mean"])["Counts"].sum())
            
            # Check if this_df is empty and find the variable that is causing it
            if this_df.empty:
                rprint(
                    f"Empty dataframe for {bkg} with nhits {nhits}, ophits {ophits}, adjcl {adjcl}"
                )
                for column, var in zip(
                    [
                        "EnergyLabel",
                        "NHits",
                        "OpHits",
                        "AdjCl",
                    ],
                    [energy.split("Energy")[0], nhits, ophits, adjcl],
                ):
                    if this_df[column].values[0] != var:
                        rprint(f"{column} is not {var}")
                continue

            x = np.asarray(list(this_df["Energy"].values))
            y = np.asarray(list(this_df["Counts"].values))
            y_error = np.asarray(list(this_df["Error"].values))
            y = np.nan_to_num(y)
            y_error = np.nan_to_num(y_error)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.asarray(list(this_df["Counts"].values)),
                    mode="lines",
                    name=bkg,
                    line_shape="hvh",
                    line=dict(color=this_df["Color"].values[0]),
                ),
                row=1,
                col=1,
            )
            # print(total)
            # print(y)
            total = total + np.array(y)
            total_error = total_error + y_error**2

        # Total is an array of size len(rebin)-1. Create a new 2d hist of size len(rebin)-1 x len(oscillation_df) by stacking total
        bkg_hist = np.tile(total / len(oscillation_df), (len(oscillation_df), 1))
        # print(bkg_hist)
        nadir = get_nadir_angle()
        interp_nadir = interp1d(*nadir)
        rebin_nadir = interp_nadir(oscillation_df.index)

        bkg_hist = rebin_nadir * bkg_hist.T
        bkg_hist = np.sum(total) * bkg_hist.T / np.sum(bkg_hist)

        bkg_hist[args.exposure * bkg_hist < 1] == 0.0
        print(f"Check Counts: {np.sum(total)} - {np.sum(bkg_hist)}")

        save_pkl(
            args.exposure * bkg_hist,
            f"{info['PATH']}/SENSITIVITY",
            config=config,
            name=f"background",
            subfolder=f"{args.folder.lower()}/{energy}",
            filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.debug,
        )

        fig.add_trace(
            go.Heatmap(
                z=np.log10(bkg_hist),
                x=sensitivity_rebin_centers,
                y=oscillation_df.index,
                colorscale="Turbo",
                colorbar=dict(title="log(Counts)"),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=this_df["Energy"],
                y=total,
                error_y=dict(type="data", array=np.sqrt(total_error), visible=True),
                mode="lines",
                name="Total",
                line_shape="hvh",
                line=dict(color="black", dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig = format_coustom_plotly(
            fig,
            title=f"{energy} Background {config}",
            log=(False, False),
            matches=("x", None),
            tickformat=(".1f", ".0e"),
            legend_title="Component",
            legend=dict(x=0.35, y=0.99),
            debug=args.debug,
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
            f"{root}/images/solar/fit/{args.folder.lower()}",
            config=None,
            name=None,
            subfolder=None,
            filename=f"{args.folder}_Background_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.debug,
        )

        if args.energy is not None and isinstance(args.energy, str):
            break
