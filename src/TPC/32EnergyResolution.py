import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/TPC/resolution/"
data_path = f"{root}/data/TPC/resolution/"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

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
    "--name", type=str, help="The name of the configuration", default="marley_official"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "SMEARING",
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "TruthDriftTime",
        "DEFAULT_ADJCL_ENERGY_TIME": "TruthAdjClDriftTime",
    },
    workflow=user_input["workflow"],
    debug=user_input["debug"],
)
filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=[user_input["workflow"]],
    params={("Reco", "TrueMain"): ("equal", True)},
    debug=user_input["debug"],
)
rprint(output)

data = filtered_run["Reco"]

# Plot the calibration workflow

fit = {
    "color": "grey",
    "opacity": 1,
    "print": False,
    "show": False,
}

RMS_data = []
for config in configs:
    fig2 = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("NHit Threshold 1", "NHit Threshold 2", "NHit Threshold 3"),
    )
    for name, (jdx, (variable, variable_label)), (kdx, nhit) in product(
        configs[config],
        enumerate(
            zip(
                ["Cluster", "Total", "Selected", "Solar"],
                ["Cluster", "Total", "Selected", "Solar"],
            )
        ),
        enumerate(nhits[:3]),
    ):

        fig1 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Energy Smearing", "Energy Resolution"),
        )

        miny = np.min(data[f"{variable}Energy"][data["NHits"] >= nhit])
        maxy = np.max(data[f"{variable}Energy"][data["NHits"] >= nhit])
        minx = np.min(data["SignalParticleK"][data["NHits"] >= nhit])
        maxx = np.max(data["SignalParticleK"][data["NHits"] >= nhit])

        miny_idx = np.where(miny < true_energy_edges)[0][0]
        maxy_idx = np.where(maxy > true_energy_edges)[0][-1]
        minx_idx = np.where(minx < true_energy_edges)[0][0]
        maxx_idx = np.where(maxx > true_energy_edges)[0][-1]

        # rprint(
        #     f"Min {variable_label} Energy: {miny:.1f} MeV, Max {variable_label} Energy: {maxy:.1f} MeV\nMin True Energy: {minx:.1f} MeV, Max True Energy: {maxx:.1f} MeV"
        #     f"Found indices: {miny_idx}, {maxy_idx}, {minx_idx}, {maxx_idx} for {variable_label} Energy and True Energy"
        # )

        x, y, h = get_hist2d(
            data["SignalParticleK"][data["NHits"] >= nhit],
            data[f"{variable}Energy"][data["NHits"] >= nhit],
            per=None,
            norm=False,
            acc=(
                true_energy_edges[minx_idx:maxx_idx],
                true_energy_edges[miny_idx:maxy_idx],
            ),
        )
        h = h / np.max(h)
        # Change 0 entries in h for Nan
        h = np.where(h == 0, np.nan, h)
        fig1.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
            ),
            row=1,
            col=1,
        )

        RMS = []
        RMS_error = []
        for energy_bin in true_energy_centers:
            idx = np.where(
                (
                    data["SignalParticleK"][data["NHits"] >= nhit]
                    > energy_bin - true_ebin / 2
                )
                & (
                    data["SignalParticleK"][data["NHits"] >= nhit]
                    < energy_bin + true_ebin / 2
                )
            )
            rms = np.sqrt(
                np.mean(
                    np.power(
                        (
                            data["SignalParticleK"][data["NHits"] >= nhit][idx]
                            - data[f"{variable}Energy"][data["NHits"] >= nhit][idx]
                        )
                        / data["SignalParticleK"][data["NHits"] >= nhit][idx],
                        2,
                    )
                )
            )

            # Compute an associated error on the RMS dependent on the number of events in the bin
            RMS.append(rms)
            error = np.sqrt(
                np.mean(
                    np.power(
                        (
                            data["SignalParticleK"][data["NHits"] >= nhit][idx]
                            - data[f"{variable}Energy"][data["NHits"] >= nhit][idx]
                        )
                        / data["SignalParticleK"][data["NHits"] >= nhit][idx],
                        2,
                    )
                )
                / np.sqrt(len(idx[0]))
            )

            RMS_error.append(error)

        # RMS = np.where(np.isnan(RMS), 0, RMS)
        # RMS_error = np.where(np.isnan(RMS_error), 0, RMS_error)

        # Add error bars
        for (
            fig,
            title,
            color,
            showlegend,
            row,
            col,
        ) in zip(
            [fig1, fig2],
            ["RMS (True - Reco) / True", f"{variable_label}Energy"],
            ["black", compare[jdx]],
            [True, kdx == 0],
            [1, 1],
            [2, kdx + 1],
        ):
            fig.add_trace(
                go.Scatter(
                    x=true_energy_centers,
                    y=RMS,
                    mode="lines",
                    line_shape="hvh",
                    marker=dict(color=color, size=5),
                    name=title,
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )

            RMS_data.append(
                {
                    "config": config,
                    "name": name,
                    "nhit": nhit,
                    "energy": f"{variable}Energy",
                    "energy_centers": true_energy_centers,
                    "RMS": RMS,
                    "RMS_error": RMS_error,
                }
            )

        # Draw grey error bands
        fig1.add_trace(
            go.Scatter(
                x=true_energy_centers,
                y=np.add(RMS, [-x for x in RMS_error]),
                mode="lines",
                line_shape="hvh",
                line=dict(color="grey", width=0),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig1.add_trace(
            go.Scatter(
                x=true_energy_centers,
                y=np.add(RMS, RMS_error),
                mode="lines",
                line_shape="hvh",
                line=dict(color="grey", width=0),
                fillcolor="rgba(128, 128, 128, 0.5)",
                fill="tonexty",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig1 = format_coustom_plotly(
            fig1,
            matches=("x", None),
            tickformat=(".1f", ".1f"),
            title=f"{variable_label}Energy - NHit Threshold {nhit} - {config} {name}",
            legend_title="Data",
            legend=dict(
                y=0.01,
                x=0.56,
            ),
        )

        fig1.update_layout(
            coloraxis=dict(colorscale="turbo", colorbar=dict(title="Norm.")),
            xaxis1_title="True Neutrino Energy (MeV)",
            xaxis2_title="True Neutrino Energy (MeV)",
            yaxis1_title=f"Reco Neutrino Energy (MeV)",
            yaxis2_title=f"RMS (True - Reco) / True",
            yaxis2_range=[0, 0.5],
        )

        save_figure(
            fig1,
            save_path,
            config,
            name,
            filename=f"{variable_label}Energy_Resolution_NHits{nhit}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        # fig1.show()

    fig2 = format_coustom_plotly(
        fig2,
        tickformat=(".1f", ".1f"),
        title=f"Low Energy Resolution {config}",
        legend_title="Reco. Algorithm",
    )
    fig2.update_xaxes(
        title="True Neutrino Energy (MeV)",
    )
    fig2.update_yaxes(
        title=f"RMS (True - Reco) / True",
        # Set axis range
        range=[0, 0.5],
    )
    save_figure(
        fig2,
        save_path,
        config,
        name,
        filename=f"Neutrino_Energy_Resolution",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
    # fig2.show()
    save_pkl(
        RMS_data,
        data_path,
        config,
        name,
        filename=f"Neutrino_Energy_Resolution_NHits",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
