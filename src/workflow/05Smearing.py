import sys

sys.path.insert(0, "../../")

from lib import *

data_path = f"{root}/data/workflow/smearing"
save_path = f"{root}/images/workflow/smearing"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_signal"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {"workflow": "SMEARING"}

run, output = load_multi(configs, preset=user_input["workflow"], debug=args.debug)
rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "TruthDriftTime",
        "DEFAULT_ADJCL_ENERGY_TIME": "TruthAdjClDriftTime",
    },
    workflow=user_input["workflow"],
    debug=args.debug,
)

filtered_run, mask, output = compute_filtered_run(
    run, configs, presets=[user_input["workflow"]], debug=args.debug
)
rprint(output)
data = filtered_run["Reco"]

output_dict = {}
fit = {
    "color": "grey",
    "spec_type": "max",
    "print": False,
}

for config in configs:
    info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
    detector_mass = 1e9  # kT
    for name in configs[config]:
        list_hist = []
        ###################################
        # Generate Smearing df for plotting
        ###################################
        for energy in ["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"]:
            eff_flux = get_detected_solar_spectrum(
                bins=true_energy_centers, mass=detector_mass, components=["b8", "hep"]
            )
            eff_flux_b8 = get_detected_solar_spectrum(
                bins=true_energy_centers, mass=detector_mass, components=["b8"]
            )
            eff_flux_hep = get_detected_solar_spectrum(
                bins=true_energy_centers, mass=detector_mass, components=["hep"]
            )

            for (idx, energy_bin), nhit in product(
                enumerate(true_energy_centers), nhits[:10]
            ):
                this_filter = (
                    (data["SignalParticleK"] > (energy_bin - ebin / 2))
                    * (data["SignalParticleK"] < (energy_bin + ebin / 2))
                    * (data["NHits"] >= nhit)
                )  # Filtering genereted neutrinos in 1GeV energy bin

                hist, bin_edges = np.histogram(
                    data[energy][this_filter], bins=true_energy_edges
                )

                hist = hist / np.sum(hist)
                flux = hist * eff_flux[idx]
                fluxb8 = hist * eff_flux_b8[idx]
                fluxhep = hist * eff_flux_hep[idx]

                list_hist.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Version": info["VERSION"],
                        "Name": name,
                        "Generator": 1,
                        "NHits": nhit,
                        "TrueEnergy": energy_bin,
                        "Hist": hist,
                        "Flux": flux,
                        "FluxB8": fluxb8,
                        "FluxHEP": fluxhep,
                        "EnergyLabel": energy,
                        "Energy": true_energy_centers,
                        "RMS": np.sqrt(
                            np.mean(
                                ((data[energy][this_filter] - energy_bin) / energy_bin)
                                ** 2
                            )
                        ),
                    }
                )
                fit["func"] = "gauss"
                output_dict[f"{energy}_{energy_bin:.2f}"] = {
                    "LABEL": energy,
                    "NHITS": nhit,
                    "ENERGY": energy_bin,
                    "RMS": np.sqrt(
                        np.mean(
                            ((data[energy][this_filter] - energy_bin) / energy_bin) ** 2
                        )
                    ),
                }

        # Finished the loop for all energies and nhits
        this_path = f"{root}/config/{config}/{name}/{config}_calib/"
        if not os.path.exists(this_path):
            os.makedirs(this_path)

        with open(
            f"{this_path}{config}_{name}_mono_energy_resolution_nhits_{nhit}.json", "w"
        ) as f:
            json.dump(output_dict, f)

        df = pd.DataFrame(list_hist)
        # display(df)
        fig = px.line(
            df[df["NHits"] < 4],
            x="TrueEnergy",
            y="RMS",
            facet_col="NHits",
            color="EnergyLabel",
            color_discrete_sequence=compare,
            line_shape="hvh",
            render_mode="SVG",
        )
        fig.update_layout(
            title="Energy Resolution",
            yaxis_title="Energy Resolution RMS (Reco - True) / True",
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Counts")),
        )
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 0.5]),
            legend_title="Reco. Algorithm",
            add_units=False,
            title=f"CC Neutrino Energy Resolution vs NHit Threshold {config}",
        )
        fig.update_xaxes(title="True Neutrino Energy (MeV)")

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Calibration_RMS",
            rm=args.rewrite,
            debug=args.debug,
        )

        fig1 = make_subplots(
            rows=3,
            cols=4,
            subplot_titles=(
                "TotalEnergy",
                "SelectedEnergy",
                "SolarEnergy",
                "ClusterEnergy",
            ),
        )
        for (j, comp), (i, energy) in product(
            enumerate(["", "B8", "HEP"]),
            enumerate(
                ["TotalEnergy", "SelectedEnergy", "SolarEnergy", "ClusterEnergy"]
            ),
        ):
            for k, nhit in enumerate(nhits[:10]):
                this_df = df[(df["EnergyLabel"] == energy) * (df["NHits"] == nhit)]

                smearing_matrix = pd.DataFrame(
                    list(this_df["Flux" + comp]),
                    columns=true_energy_centers,
                    index=true_energy_centers,
                ).T

                smearing_matrix.to_pickle(
                    f"{root}/config/{config}/{name}/{config}_calib/{config}_{energy+comp}_Smearing_NHits_{nhit}.pkl"
                )

                smearing_matrix = smearing_matrix.fillna(0)

                if nhit == 3:
                    fig1.add_trace(
                        go.Heatmap(
                            z=np.log10(smearing_matrix.values),
                            x=smearing_matrix.columns,
                            y=smearing_matrix.index,
                            colorscale="turbo",
                            coloraxis="coloraxis",
                            colorbar=dict(title="log10(Flux) Hz/MeV"),
                        ),
                        row=j + 1,
                        col=i + 1,
                    )

                if nhit < 4 and comp == "":
                    exploded_df = explode(
                        this_df,
                        ["Flux", "Hist", "Energy"],
                        keep=["TrueEnergy", "Version"],
                    )

                    exploded_df["TrueEnergy"] = exploded_df["TrueEnergy"].astype(float)
                    # rprint(this_df.groupby("Version")["Flux"].sum())
                    fig = px.bar(
                        exploded_df,
                        x="Energy",
                        y="Flux",
                        log_y=False,
                        color="TrueEnergy",
                        barmode="stack",
                        template="presentation",
                        facet_col="Version",
                        color_continuous_scale="turbo",
                    )

                    # 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr','ylorrd'
                    fig = format_coustom_plotly(
                        fig,
                        log=(False, False),
                        tickformat=(None, ".1s"),
                        matches=(None, None),
                    )
                    fig.update_layout(bargap=0)
                    fig.update_layout(
                        coloraxis=dict(
                            colorscale="Turbo", colorbar=dict(title="Energy")
                        ),
                        showlegend=False,
                        title="Reconstruction Smearing",
                        xaxis1_title="Reco Electron Energy (MeV)",
                        yaxis1_title="Norm.",
                    )

                    save_figure(
                        fig,
                        save_path,
                        config,
                        name,
                        filename=f"{energy}_Smearing_NHits{nhit}_Hist",
                        rm=args.rewrite,
                        debug=args.debug,
                    )

        fig1.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig1.update_yaxes(title_text="Reco Energy (MeV)")
        fig1 = format_coustom_plotly(
            fig1, title=f"Smearing Matrix NHits {nhit} {config}"
        )
        save_figure(
            fig1,
            save_path,
            config,
            name,
            filename=f"Solar_Smearing_Matrix_NHits3",
            rm=args.rewrite,
            debug=args.debug,
        )
