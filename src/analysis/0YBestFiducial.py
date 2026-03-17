import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/fiducial"
data_path = f"{root}/data/solar/fiducial"

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
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the background folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure in kT·year",
    default=100,
)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=8
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

plot_dict = {}
for config in configs:
    for name, energy_label in product(configs[config], args.energy):
        df_list = []
        signal_df = pd.read_pickle(
            f"/pc/choozdsk01/users/manthey/SOLAR/data/solar/fiducial/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy_label}_Fiducial_Scan.pkl"
        )
        df_list.append(signal_df)
        for bkg, bkg_label in [
            ("gamma", "gamma"),
            ("neutron", "neutron"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pc/choozdsk01/users/manthey/SOLAR/data/solar/fiducial/{args.folder.lower()}/{config}/{bkg_label}/{config}_{bkg_label}_{energy_label}_Fiducial_Scan.pkl"
            )
            df_list.append(bkg_df)

        raw_df = pd.concat(df_list, ignore_index=True)

        plot_df = explode(
            raw_df,
            ["Counts", "Error+", "Error-", "Energy", "MCCounts"],
            debug=args.debug,
        )
        plot_df["Counts"] = plot_df["Counts"].replace(0, np.nan)

        # Filter signal and background dataframes
        signal_df = plot_df[
            (plot_df["Component"].isin(["Solar", "8B", "hep"]))
            * (plot_df["Energy"] > args.threshold)
        ]
        background_df = plot_df[
            (plot_df["Component"].isin(["neutron", "gamma"]))
            * (plot_df["Energy"] > args.threshold)
        ]

        # Compute background counts by grouping over each FiducializedX/Y/Z values
        background_df = (
            background_df.groupby(
                ["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"]
            )
            .agg(
                {
                    "MCCounts": lambda x: sum(x),  # Sum the MCCounts
                    "Counts": lambda x: sum(x),  # Sum the Counts
                    "Error+": lambda x: sum(x**2)
                    ** 0.5,  # BEGIN: Compute sqrt of summed squares of the Error
                    "Error-": lambda x: sum(x**2) ** 0.5,
                }  # END:
            )
            .reset_index()
        )

        # Apply the compute significance function to each row of the signal and background dataframes that match in Energy and FiducializedX/Y/Z by merging them first
        merged_df = pd.merge(
            signal_df,
            background_df,
            on=["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"],
            suffixes=("Signal", "Background"),
        )
        merged_df["Significance"] = evaluate_significance(
            merged_df["CountsSignal"].to_numpy(),
            merged_df["CountsBackground"].to_numpy(),
            merged_df["Error+Signal"].to_numpy(),
            merged_df["Error+Background"].to_numpy(),
        )

        max_significance = merged_df["Significance"].max()
        print(f"{energy_label} Max significance: {max_significance:.2f}")

        significance_df = (
            merged_df.groupby(
                ["Component", "FiducializedX", "FiducializedY", "FiducializedZ"]
            )
            .agg(
                {
                    "Significance": lambda x: sum(x**2)
                    ** 0.5,  # Get the sqrt of summed squares of the Significance
                }
            )
            .reset_index()
        )

        # Find the row with the maximum significance
        max_row = significance_df.loc[significance_df["Significance"].idxmax()]
        print(max_row)

        # Save the best significance fiducial position
        best_fiducials = {
            config: {
                energy_label: {
                    "FiducialX": int(max_row["FiducializedX"]),
                    "FiducialY": int(max_row["FiducializedY"]),
                    "FiducialZ": int(max_row["FiducializedZ"]),
                }
            }
        }

        # Save as a json file. If already exists, load and update
        filename = f"{data_path}/{args.folder.lower()}/BestFiducials.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                existing_data = json.load(f)
                print(f"Loaded existing best fiducials from {filename}")
            existing_data.setdefault(config, {}).update(
                {energy_label: best_fiducials[config][energy_label]}
            )
            best_fiducials = existing_data

        with open(filename, "w") as f:
            json.dump(best_fiducials, f, indent=4)

        fiducialx = int(max_row["FiducializedX"])
        fiducialy = int(max_row["FiducializedY"])
        fiducialz = int(max_row["FiducializedZ"])

        # max_counts = 0
        max_significance = 0
        for [fiducialx, fiducialy, fiducialz], fiducial_label in zip(
            [
                [
                    max_row["FiducializedX"],
                    max_row["FiducializedY"],
                    max_row["FiducializedZ"],
                ],
                [0, 0, 0],
            ],
            ["Best", "No"],
        ):
            this_plot = plot_df[
                (plot_df["Component"] != "Solar")
                * (plot_df["FiducializedX"] == fiducialx)
                * (plot_df["FiducializedY"] == fiducialy)
                * (plot_df["FiducializedZ"] == fiducialz)
            ]
            print(this_plot.groupby("Component")["Counts"].sum())
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0,
                row_heights=[0.7, 0.3],
            )

            component_color = {
                "gamma": "black",
                "neutron": "rgb(15,133,84)",
                "8B": "rgb(225,124,5)",
                "hep": "rgb(204,80,62)",
            }

            for component in this_plot["Component"].unique():
                component_data = this_plot[this_plot["Component"] == component]
                if args.stacked:
                    fig.add_trace(
                        go.Bar(
                            x=component_data["Energy"].to_numpy(),
                            y=args.exposure * component_data["Counts"].to_numpy(),
                            name=component,
                            marker_color=component_color.get(component, "grey"),
                            legendgroup=1,
                            legendgrouptitle=dict(text="Component", font=dict(size=16)),
                        ),
                        row=1,
                        col=1,
                    )

                else:
                    x = component_data["Energy"].to_numpy().astype(float)
                    y = component_data["Counts"].to_numpy().astype(float)
                    y_error_plus = component_data["Error+"].to_numpy().astype(float)
                    y_error_minus = component_data["Error-"].to_numpy().astype(float)

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=args.exposure * y,
                            error_y=dict(
                                type="data",
                                symmetric=False,
                                array=args.exposure * y_error_plus,
                                arrayminus=args.exposure * y_error_minus,
                                visible=True,
                            ),
                            mode="lines+markers",
                            line_shape="hvh",
                            legendgroup=1,
                            legendgrouptitle=dict(text="Component", font=dict(size=16)),
                            name=f"{component} {args.exposure * component_data['Counts'].sum():.1e}",
                            marker=dict(
                                size=6,
                                color=component_color.get(component, "grey"),
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                # if max(max_counts, (args.exposure*component_data["Counts"]).max()) > max_counts:
                #     max_counts = max(max_counts, (args.exposure*component_data["Counts"]).max())

            # fig.add_vline(10, line=dict(color="grey", dash="dash"))

            fig.update_layout(
                title=f"{energy_label} Significance<br>Fiducial: X={fiducialx}cm, Y={fiducialy}cm, Z={fiducialz}cm",
                showlegend=True,
            )
            # Compute the significance for the hep and b8 components
            this_hep = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "hep")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Counts"].to_list()
            )
            this_hep_error = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "hep")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Error+"].to_list()
            )
            this_8b = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "8B")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Counts"].to_list()
            )
            this_8b_error = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "8B")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Error+"].to_list()
            )
            this_gamma = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "gamma")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Counts"].to_list()
            )
            this_gamma_error = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "gamma")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Error+"].to_list()
            )
            this_neutron = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "neutron")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Counts"].to_list()
            )
            this_neutron_error = args.exposure * np.array(
                this_plot[
                    (this_plot["Component"] == "neutron")
                    * (this_plot["Energy"] >= args.threshold)
                ]["Error+"].to_list()
            )
            this_bkg = np.add(this_gamma, this_neutron)

            hep_significance = evaluate_significance(
                this_hep,
                this_bkg,
            )
            b8_significance = evaluate_significance(
                this_8b,
                this_bkg,
            )

            # Add the significance to the plot
            if (
                max(np.max(hep_significance), np.max(b8_significance))
                > max_significance
            ):
                max_significance = max(
                    np.max(hep_significance), np.max(b8_significance)
                )
            for component, significance in zip(
                ["8B", "hep"], [b8_significance, hep_significance]
            ):
                # Append zeros to significance to match the length of this_plot "Energy"
                energy_array = np.array(
                    this_plot[(this_plot["Component"] == component)]["Energy"].to_list()
                )
                significance = np.concatenate(
                    (np.zeros(len(energy_array) - len(significance)), significance)
                )
                fig.add_trace(
                    go.Scatter(
                        x=energy_array,
                        y=significance,
                        mode="lines+markers",
                        line_shape="hvh",
                        legendgroup=2,
                        legendgrouptitle=dict(
                            text="Significance",
                            font=dict(size=16),
                        ),
                        name=f"{component} {np.sqrt(np.sum(significance**2)):.1f}σ",
                        line=dict(color=component_color.get(component, "grey")),
                    ),
                    row=2,
                    col=1,
                )
            # Add vertical line at args.threshold
            fig.add_vline(
                args.threshold,
                line=dict(color="grey", dash="dash"),
                row=1,
                col=1,
            )
            if args.stacked:
                fig.update_layout(barmode="stack")

            fig = format_coustom_plotly(
                fig, tickformat=(".0f", ".1e"), legend={"x": 0.72, "y": 0.99}
            )
            fig.update_xaxes(range=[8, 22])
            fig.update_xaxes(title_text="Reconstructed Energy (MeV)", row=2, col=1)
            # Set row 1 yaxis to log
            fig.update_yaxes(
                type="log",
                range=[0.5, 10],
                tickformat=".0e",
                title_text="Counts per Energy (100·kT·year·MeV)⁻¹",
                row=1,
                col=1,
            )
            # Set row 2 yaxis to linear and tickformat to .1f
            fig.update_yaxes(
                type="linear",
                range=[0, max_significance * 1.5],
                tickformat=".1f",
                title_text="Significance (σ)",
                row=2,
                col=1,
            )

            save_figure(
                fig,
                f"{save_path}/{args.folder.lower()}",
                config,
                name,
                None,
                f"{energy_label}_{fiducial_label}Fiducial_Significance"
                + (f"_Stacked" if args.stacked else ""),
                rm=args.rewrite,
                debug=args.debug,
            )
