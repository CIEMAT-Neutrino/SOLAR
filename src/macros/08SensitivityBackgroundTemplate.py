import sys, json

sys.path.insert(0, "../../")
from lib import *

from lib.root_functions import Sensitivity_Fitter
from lib.osc_functions import get_oscillation_datafiles

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "HEP"],
    default=None,
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
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
rprint(args)

config = args.config
name = args.name
folder = args.folder

signal_uncertanty = args.signal_uncertanty
background_uncertanty = args.background_uncertanty

rewrite = args.rewrite
debug = args.debug

configs = {config: [name]}

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))

    for name in configs[config]:
        df_list = []
        signal_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/nominal/SENSITIVITY/{config}/{name}/{config}_{name}_rebin.pkl"
        )
        df_list.append(signal_df)
        for bkg, bkg_label in [
            ("alpha", "alpha"),
            ("neutron", "neutron"),
            ("gamma", "gamma"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/SENSITIVITY/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
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

    fastest_sigma = pickle.load(
        open(
            f"{info['PATH']}/{args.reference.upper()}/{folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_highest_{args.reference}.pkl",
            "rb",
        )
    )

    for idx, key in enumerate(fastest_sigma):
        if key[2] not in args.energy:
            rprint(f"Skipping {key[2]}Energy")
            continue

        total = np.zeros(len(sensitivity_rebin) - 1)
        total_error = np.zeros(len(sensitivity_rebin) - 1)

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

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=(
                [
                    "Background Components",
                    f"Fiducial {fiducial:.0f} (cm), min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}",
                ]
            ),
        )

        for bkg in ["alpha", "gamma", "neutron"]:
            this_df = plot_df[
                (plot_df["Component"] == bkg)
                * (plot_df["EnergyLabel"] == key[2])
                * (plot_df["NHits"] == nhits)
                * (plot_df["OpHits"] == ophits)
                * (plot_df["AdjCl"] == adjcl)
                * (plot_df["Fiducialized"] == fiducial)
            ]
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
            total = total + y
            total_error = total_error + y_error**2

        # Total is an array of size len(rebin)-1. Create a new 2d hist of size len(rebin)-1 x len(oscillation_df) by stacking total
        bkg_hist = np.tile(total / len(oscillation_df), (len(oscillation_df), 1))

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
            subfolder=f"{folder.lower()}/{key[2]}Energy",
            filename=f"Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
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
            title=f"{key[2]}Energy Background {config}",
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
            f"{root}/images/solar/fit/{folder.lower()}",
            config=None,
            name=None,
            subfolder=None,
            filename=f"{folder}_Background_{key[2]}Energy_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite,
            debug=args.debug,
        )
