import sys, json

sys.path.insert(0, "../../")
from lib import *

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
    default="ClusterEnergy",
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
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
configs = {args.config: [args.name]}
rprint(args)

save_path = f"{root}/images/sensitivity"
for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))

    fastest_sigma = pickle.load(
        open(
            f"{info['PATH']}/{args.reference.upper()}/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_fastest_sigma3_{args.reference}.pkl",
            "rb",
        )
    )
    for name, key in product(configs[config], fastest_sigma):
        if args.energy is not None:
            energy = args.energy
        else:
            energy = key[2]

        if args.background:
            data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/hd_1x2x6_centralAPA/marley/reduced/{energy}/results/signal_{100*args.signal_uncertanty:.0f}%_and_background_{100*args.background_uncertanty:.0f}%"
        else:
            data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/hd_1x2x6_centralAPA/marley/reduced/{energy}/results/signal_{100*args.signal_uncertanty:.0f}%_only"

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

        solar_sin13_df = pd.read_pickle(
            f"{data_path}/marley_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl"
        )
        solar_sin12_df = pd.read_pickle(
            f"{data_path}/marley_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl"
        )
        react_sin13_df = pd.read_pickle(
            f"{data_path}/marley_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl"
        )
        react_sin12_df = pd.read_pickle(
            f"{data_path}/marley_{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl"
        )

        # Substitute 0 values for nan in all dfs
        for df in [solar_sin13_df, solar_sin12_df, react_sin13_df, react_sin12_df]:
            df.replace(0, np.nan, inplace=True)

        contours = np.arange(0, 4, 1)

        # Modify ylgnbu_r coloraxis to have last color with white
        colorscale = [[0, "navy"], [0.5, "teal"], [1, "white"]]
        fig = make_subplots(
            1,
            1,
            subplot_titles=(
                [
                    f"Fiducial {fiducial:.0f} (cm), min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}"
                ]
            ),
        )
        fig.add_trace(
            go.Contour(
                x=solar_sin12_df.columns.astype(float),
                y=solar_sin12_df.index,
                z=np.sqrt(solar_sin12_df.values.astype(float)),
                connectgaps=True,
                coloraxis="coloraxis",
                contours=dict(start=0, end=contours[-1], size=1),
                name="DUNE",
                showlegend=True,
            )
        )

        fig = format_coustom_plotly(
            fig,
            # title=f"DUNE Contours for Solar Best Fit ({unicode('Delta')}m²{subscript(21)} {6e-5:.0e} eV²)",
            title=f"{config} {name} {energy}",
            tickformat=(".2f", ".0e"),
            add_watermark=True,
        )

        fig.update_coloraxes(
            colorbar_title=f"{unicode('sigma')}",
        )
        fig.update_layout(
            coloraxis=dict(colorscale=colorscale),
            xaxis=dict(range=[0.15, 0.45]),
            yaxis=dict(range=[3e-5, 1e-4]),
        )
        # Add an ellipse at position y=7.4 and x=0.303
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=0.304,
            y0=7.455e-5,
            x1=0.312,
            y1=7.595e-5,
            # opacity=0.2,
            fillcolor="black",
            line_color="black",
            # showlegend=True,
            # name=f"(JUNO) 3{unicode('sigma')}"
        )
        fig.add_trace(
            go.Scatter(
                x=[1.5],
                y=[0.75],
                name=f"JUNO (3{unicode('sigma')})",
                text=f"JUNO (3{unicode('sigma')})",
                mode="markers",
                marker=dict(size=12, color="black"),
            )
        )
        sno_paths = ["contour1_tan.csv", "contour2_tan.csv", "contour3_tan.csv"]
        solar_paths = ["contour1.csv", "contour2.csv", "contour3.csv"]
        kamland_paths = ["contour1.csv", "contour2.csv", "contour3.csv"]

        for file_paths, label, color in zip(
            [solar_paths, kamland_paths],
            ["Solar", "KamLAND"],
            ["blue", "grey"],
        ):
            dash_list = ["solid", "dot", "dash"]
            for idx, file_path in enumerate(file_paths):
                compute_sin = False
                deltam_factor = 1e-5
                folder_path = f"{root}/lib/import/contours/{label}"
                # Check if the file exists
                if not os.path.exists(f"{folder_path}/{file_path}"):
                    print(f"File {folder_path}/{file_path} does not exist.")
                    sys.exit(1)

                file_name = file_path.split(".")[0]

                if file_path.split(".")[-1] != "csv":
                    print(f"File {file_path} is not a CSV file.")
                    sys.exit(1)

                if file_name.split("_")[-1] == "tan":
                    compute_sin = True
                    deltam_factor = 1e-4

                # Load the CSV file
                data = load_contour_csv(
                    f"{folder_path}/{file_path}",
                    compute_sin=compute_sin,
                    deltam_factor=deltam_factor,
                )

                # Draw the contour
                fig = draw_contour(fig, idx, label, data, color, dash_list[idx])

        # Show legend inside the plot
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.01,
                # xanchor="left",
                # yanchor="bottom",
                title="Contours",
                orientation="v",
                font=dict(size=18),
            )
        )

        if args.background:
            figure_name = f"{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertanty:.0f}_Bkg{100*args.background_uncertanty:.0f}"
        else:
            figure_name = f"{energy}_Fiducial{fiducial}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertanty:.0f}"

        fig.update_xaxes(title=f"sin²{unicode('theta')}{subscript(12)}")
        fig.update_yaxes(title=f"{unicode('Delta')}m²{subscript(21)} (eV²)")
        save_figure(
            fig,
            save_path,
            config,
            name=None,
            filename=figure_name,
            rm=args.rewrite,
            debug=args.debug,
        )
