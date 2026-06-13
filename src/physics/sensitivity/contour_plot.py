import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

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
    "--exposure",
    type=float,
    help="The exposure for the analysis in kton-years",
    default=30.0,
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--nuisance_profile",
    type=str,
    help="Nuisance parameter profile name (key in NUISANCE_PROFILES in analysis/config.json). Defaults to DEFAULT_NUISANCE_PROFILE.",
    default=None,
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
configs = {args.config: [args.name]}
if args.debug:
    rprint(args)


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

sensitivity = []
save_path = f"{root}/images/analysis/sensitivity"
for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    analysis_info = load_analysis_info(str(root))
    _nuisance_profiles = analysis_info.get("NUISANCE_PROFILES", {})
    _default_profile   = analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
    profile_name       = args.nuisance_profile or _default_profile or "full"
    energy = args.energy
    nhits, adjcl, ophits = -1, -1, -1

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

    for name, key in product(configs[config], cut_keys):
        if args.energy is not None:
            energy = args.energy
        else:
            energy = key[2]

        if args.background:
            data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/{config}/{args.name}/{args.folder.lower()}/{energy}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%"
        else:
            data_path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/{config}/{args.name}/{args.folder.lower()}/{energy}/results/{profile_name}/signal_{100*args.signal_uncertainty:.0f}%_only"

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

        solar_sin13_df = pd.read_pickle(
            f"{data_path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin13_df.pkl"
        )
        solar_sin12_df = pd.read_pickle(
            f"{data_path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_solar_sin12_df.pkl"
        )
        react_sin13_df = pd.read_pickle(
            f"{data_path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin13_df.pkl"
        )
        react_sin12_df = pd.read_pickle(
            f"{data_path}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_react_sin12_df.pkl"
        )

        # Substitute 0 values for nan in all dfs
        contours = np.arange(0, 4, 1)
        for df, df_name in zip([solar_sin13_df, solar_sin12_df, react_sin13_df, react_sin12_df], ["solar_sin13_df", "solar_sin12_df", "react_sin13_df", "react_sin12_df"]):
            df.replace(0, np.nan, inplace=True)

            sensitivity.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Label": df_name.split("_")[0],
                    "Variable": df_name.split("_")[1],
                    "Dm2": df.index.astype(float).values,
                    "Values": df.columns.astype(float).values,
                    "Significance": np.sqrt(df.values.astype(float)).tolist(),
                }
            )

            # Modify ylgnbu_r coloraxis to have last color with white
            colorscale = [[0, "navy"], [0.5, "teal"], [1, "white"]]
            fig = make_subplots(
                1,
                1,
                subplot_titles=(
                    [
                        f"{energy}, min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}"
                    ]
                ),
            )
            fig.add_trace(
                go.Contour(
                    x=df.columns.astype(float),
                    y=df.index,
                    z=np.sqrt(df.values.astype(float)),
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
            if "sin12" in df_name:
                fig.update_layout(
                    coloraxis=dict(colorscale=colorscale),
                    xaxis=dict(range=[0.15, 0.45]),
                    yaxis=dict(range=[3e-5, 1e-4]),
                )
            else:
                fig.update_layout(
                    coloraxis=dict(colorscale=colorscale),
                )

            # Add an ellipse at position y=7.4 and x=0.303
            if "sin12" in df_name:
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
                        folder_path = f"{root}/data/contours/{label}"
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
                    title="Contours",
                    orientation="v",
                    font=dict(size=18),
                    bgcolor="rgba(255,255,255,0.7)",
                )
            )

            if args.background:
                figure_name = f"{df_name}_{args.folder}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertainty:.0f}_Bkg{100*args.background_uncertainty:.0f}"
            else:
                figure_name = f"{df_name}_{args.folder}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_Signal{100*args.signal_uncertainty:.0f}"

            fig.update_xaxes(title=f"sin²{unicode('theta')}{subscript(12)}" if "sin12" in df_name else f"sin²{unicode('theta')}{subscript(13)}")
            fig.update_yaxes(title=f"{unicode('Delta')}m²{subscript(21)} (eV²)")
            save_figure(
                fig,
                save_path,
                config=config,
                name=name,
                subfolder=f"{args.folder.lower()}/{profile_name}",
                filename=figure_name,
                rm=args.rewrite,
                debug=args.plot,
            )
    
    save_pkl(
        pd.DataFrame(sensitivity),
        f"{analysis_info['PATH']}/SENSITIVITY",
        config=config,
        name=name,
        subfolder=args.folder.lower(),
        filename=f"Sensitivity_{energy}" if args.nhits is None and args.adjcls is None and args.ophits is None else f"Sensitivity_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.debug,
    )