import sys

sys.path.insert(0, "../../")

from lib import *

data_path = f"{root}/data/workflow/reconstruction/"
save_path = f"{root}/images/workflow/reconstruction"
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

user_input = {"workflow": "RECONSTRUCTION"}

run, output = load_multi(configs, preset=user_input["workflow"], debug=args.debug)
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

acc = get_default_acc(len(data["Generator"]))
fit = {
    "color": "grey",
    "opacity": 0,
    "print": True,
    "range": (0, 10),
    "show": True,
}

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=("Electron", "Gamma", "Electron+Gamma")
        )

        fit["threshold"] = 0.7
        fit["bounds"] = ([-10], [0])
        fit["spec_type"] = "intercept"
        fig, popt_int, perr_int = get_hist2d_fit(
            data["SignalParticleK"],
            data["ElectronK"],
            fig=fig,
            idx=(1, 1),
            acc=100,
            fit=fit,
            zoom=True,
            debug=args.debug,
        )

        x, y, h = get_hist2d(
            x=data["SignalParticleK"],
            y=data["GammaK"],
            density=True,
            acc=acc,
        )
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            col=2,
            row=1,
        )

        fit["func"] = "linear"
        fit["spec_type"] = "max"
        fit["bounds"] = ([0.1, -10], [1.1, 0])
        fig, popt_true, perr_true = get_hist2d_fit(
            data["SignalParticleK"],
            data["GammaK"] + data["ElectronK"],
            fig=fig,
            idx=(1, 3),
            acc=acc,
            fit=fit,
            zoom=True,
            debug=args.debug,
        )

        x, y, h = get_hist2d(
            x=data["SignalParticleK"],
            y=data["Energy"],
            density=True,
            acc=acc,
            debug=args.debug,
        )
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            col=1,
            row=2,
        )

        x, y, h = get_hist2d(
            x=data["SignalParticleK"][data["TotalAdjClEnergy"] > 0],
            y=data["TotalAdjClEnergy"][data["TotalAdjClEnergy"] > 0],
            density=True,
            acc=acc,
            debug=args.debug,
        )
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            col=2,
            row=2,
        )

        fit["spec_type"] = "max"
        fit["trimm"] = (2, int(acc / 4))
        fig, total_popt, total_perr = get_hist2d_fit(
            data["SignalParticleK"],
            data["Energy"] + data["TotalAdjClEnergy"],
            fig,
            idx=(2, 3),
            acc=acc,
            fit=fit,
            zoom=True,
            debug=args.debug,
        )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title="Reco Cluster - Neutrino Energy Reconstruction",
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm")),
            showlegend=False,
            yaxis1_title="True Electron Energy (MeV)",
            yaxis2_title="True Gamma Energy (MeV)",
            yaxis3_title="Visible Energy (MeV)",
            yaxis4_title="Main Cluster Energy (MeV)",
            yaxis5_title="Adj. Cluster Energy (MeV)",
            yaxis6_title="Reco Energy (MeV)",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"True&Reco_Neutrino_Energy",
            rm=args.rewrite,
            debug=args.debug,
        )

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=("Electron", "Gamma", "Electron+Gamma")
        )

        x, y, h = get_hist2d(
            x=data["ElectronK"],
            y=data["Energy"],
            acc=acc,
            nanz=True,
            zoom=True,
            density=True,
        )
        # h = h / h.max()
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            row=1,
            col=1,
        )

        x, y, h = get_hist2d(
            x=data["GammaK"],
            y=data["TotalAdjClEnergy"],
            acc=(true_energy_edges, true_energy_edges),
            nanz=True,
            zoom=True,
            density=True,
        )
        # h = h / h.max()
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            row=1,
            col=2,
        )

        x, y, h = get_hist2d(
            x=data["GammaK"] + data["ElectronK"],
            y=data["Energy"] + data["TotalAdjClEnergy"],
            acc=acc,
            nanz=True,
            zoom=True,
            density=True,
        )
        # h = h / h.max()
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=h.T,
                coloraxis="coloraxis",
                colorscale="Turbo",
            ),
            row=1,
            col=3,
        )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title=f"CC Neutrino Energy Reconstruction {config}",
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm.")),
            showlegend=False,
            xaxis1_title="True Electron Energy (MeV)",
            xaxis2_title="True Gamma Energy (MeV)",
            xaxis3_title="Visible Energy (MeV)",
            yaxis1_title="Reco Main Cluster Energy (MeV)",
            yaxis2_title="Reco Adj. Cluster Energy (MeV)",
            yaxis3_title="Reco Total Energy (MeV)",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"True&Reco_Daughter_Energy",
            rm=args.rewrite,
            debug=args.debug,
        )

        data["ClusterEnergy"] = data["Energy"]

        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "TotalEnergy",
                "SelectedEnergy",
                "SolarEnergy",
                "ClusterEnergy",
            ),
        )

        for idx, energy in enumerate(
            ["TotalEnergy", "SelectedEnergy", "SolarEnergy", "ClusterEnergy"]
        ):
            x, y, h = get_hist2d(
                data["SignalParticleK"],
                data[energy],
                acc=acc,
                density=False,
            )
            # Substitute h == 0 with np.nan
            h = np.where(h == 0, np.nan, h)
            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=h.T,
                    coloraxis="coloraxis",
                    colorscale="Turbo",
                ),
                col=idx + 1,
                row=1,
            )

            save_pkl(
                data["SignalParticleK"],
                data_path,
                config,
                name,
                filename=f"Truth_Neutrino_{energy}_hist1D",
                rm=args.rewrite,
                debug=args.debug,
            )
            save_pkl(
                data[energy],
                data_path,
                config,
                name,
                filename=f"Reco_Neutrino_{energy}_hist1D",
                rm=args.rewrite,
                debug=False,
            )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title="Reco Cluster - Neutrino Energy Reconstruction",
        )

        fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm")),
            showlegend=False,
            yaxis1_title="Reco Energy (MeV)",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Reco_Neutrino_Energy_Raw",
            rm=args.rewrite,
            debug=args.debug,
        )

        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "TotalEnergy",
                "SelectedEnergy",
                "SolarEnergy",
                "ClusterEnergy",
            ),
        )

        fit_dict = {}
        for idx, energy in enumerate(
            ["TotalEnergy", "SelectedEnergy", "SolarEnergy", "ClusterEnergy"]
        ):
            fit["spec_type"] = "max"
            fit["func"] = params["SAMPLE_FIT"][energy]
            if fit["func"] == "quadratic":
                fit["bounds"] = ([-np.inf, -np.inf, -10], [1, 1.1, 10])
            elif fit["func"] == "linear":
                fit["bounds"] = ([0.1, -10], [1.1, 10])

            fit["trimm"] = (
                int(acc * params["MIN_SAMPLE_TRIM"][energy]),
                int(acc * params["MAX_SAMPLE_TRIM"][energy]),
            )

            fig, total_popt, total_perr = get_hist2d_fit(
                data["SignalParticleK"],
                data[energy],
                fig,
                idx=(1, idx + 1),
                acc=acc,
                fit=fit,
                nanz=True,
                zoom=True,
                debug=True,
            )
            if total_popt is None:
                total_popt = [np.nan] * len(fit["bounds"][0])
                total_perr = [np.nan] * len(fit["bounds"][0])

            if fit["func"] == "quadratic":
                fit_dict[energy] = {"popt": total_popt, "perr": total_perr}

            elif fit["func"] == "linear":
                fit_dict[energy] = {
                    "popt": (1e-10, *total_popt),
                    "perr": (1e-10, *total_perr),
                }
            elif fit["func"] == "slope1":
                fit_dict[energy] = {
                    "popt": (1e-10, 1, total_popt[0]),
                    "perr": (1e-10, 0, total_perr[0]),
                }

        fig = format_coustom_plotly(
            fig,
            matches=("x", "y"),
            title="Reco Cluster - Neutrino Energy Reconstruction",
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm")),
            showlegend=False,
            yaxis1_title="Reco Energy (MeV)",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Reco_Neutrino_Energy_Norm_Hist2D_Fit",
            rm=args.rewrite,
            debug=args.debug,
        )

        energy_calibrartion_dict = {
            "TRUE": {"ENERGY_AMP": popt_true[0], "INTERSECTION": popt_true[1]},
            "CLUSTER": {
                "ENERGY_CURVATURE": fit_dict["ClusterEnergy"]["popt"][0],
                "ENERGY_CURVATURE_ERROR": fit_dict["ClusterEnergy"]["perr"][0],
                "ENERGY_AMP": fit_dict["ClusterEnergy"]["popt"][1],
                "ENERGY_AMP_ERROR": fit_dict["ClusterEnergy"]["perr"][1],
                "INTERSECTION": fit_dict["ClusterEnergy"]["popt"][2],
                "INTERSECTION_ERROR": fit_dict["ClusterEnergy"]["perr"][2],
            },
            "SOLAR": {
                "ENERGY_CURVATURE": fit_dict["SolarEnergy"]["popt"][0],
                "ENERGY_CURVATURE_ERROR": fit_dict["SolarEnergy"]["perr"][0],
                "ENERGY_AMP": fit_dict["SolarEnergy"]["popt"][1],
                "ENERGY_AMP_ERROR": fit_dict["SolarEnergy"]["perr"][1],
                "INTERSECTION": fit_dict["SolarEnergy"]["popt"][2],
                "INTERSECTION_ERROR": fit_dict["SolarEnergy"]["perr"][2],
            },
            "SELECTED": {
                "ENERGY_CURVATURE": fit_dict["SelectedEnergy"]["popt"][0],
                "ENERGY_CURVATURE_ERROR": fit_dict["SelectedEnergy"]["perr"][0],
                "ENERGY_AMP": fit_dict["SelectedEnergy"]["popt"][1],
                "ENERGY_AMP_ERROR": fit_dict["SelectedEnergy"]["perr"][1],
                "INTERSECTION": fit_dict["SelectedEnergy"]["popt"][2],
                "INTERSECTION_ERROR": fit_dict["SelectedEnergy"]["perr"][2],
            },
            "TOTAL": {
                "ENERGY_CURVATURE": fit_dict["TotalEnergy"]["popt"][0],
                "ENERGY_CURVATURE_ERROR": fit_dict["TotalEnergy"]["perr"][0],
                "ENERGY_AMP": fit_dict["TotalEnergy"]["popt"][1],
                "ENERGY_AMP_ERROR": fit_dict["TotalEnergy"]["perr"][1],
                "INTERSECTION": fit_dict["TotalEnergy"]["popt"][2],
                "INTERSECTION_ERROR": fit_dict["TotalEnergy"]["perr"][2],
            },
        }

        # Save as json file
        if not os.path.exists(f"{root}/config/{config}/{name}/{config}_calib/"):
            os.makedirs(f"{root}/config/{config}/{name}/{config}_calib/")
        with open(
            f"{root}/config/{config}/{name}/{config}_calib/{config}_{name}_energy_calibration.json",
            "w",
        ) as f:
            json.dump(energy_calibrartion_dict, f)

        rprint(
            f"-> Saved reco energy fit parameters to {root}/config/{config}/{name}/{config}_calib/{config}_energy_calibration.json"
        )

        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "TotalEnergy",
                "SelectedEnergy",
                "SolarEnergy",
                "ClusterEnergy",
            ),
        )

        for jdx, energy in enumerate(
            ["TotalEnergy", "SelectedEnergy", "SolarEnergy", "ClusterEnergy"]
        ):
            # Inverse function of polinomyal with degree 2
            def inverse_quadratic(x, a, b, c):
                return (-b + np.sqrt(b**2 - 4 * a * (c - x))) / (2 * a)

            h = []
            for idx, ebin in enumerate(true_energy_edges[:-1]):
                this_filter = (data["SignalParticleK"] > true_energy_edges[idx]) * (
                    data["SignalParticleK"] < true_energy_edges[idx + 1]
                )
                # Quadratic function correction
                a = inverse_quadratic(
                    data[energy][this_filter],
                    fit_dict[energy]["popt"][0],
                    fit_dict[energy]["popt"][1],
                    fit_dict[energy]["popt"][2],
                )
                hist, bins = np.histogram(
                    a,
                    bins=true_energy_edges,
                    density=False,
                )
                hist = hist / np.sum(hist)
                h.append(hist)

            h = np.array(h)
            h = np.where(h == 0, np.nan, h)
            fig.add_trace(
                go.Heatmap(
                    x=true_energy_centers,
                    y=true_energy_centers,
                    z=h.T,
                    coloraxis="coloraxis",
                    colorscale="Turbo",
                ),
                col=jdx + 1,
                row=1,
            )

            save_pkl(
                h.T,
                data_path,
                config,
                name,
                filename=f"Neutrino_{energy}_hist2D",
                rm=args.rewrite,
                debug=args.debug,
            )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title="Reco Cluster - Neutrino Energy Reconstruction",
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Norm")),
            showlegend=False,
            yaxis1_title="Effective Reco Energy (MeV)",
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Reco_Neutrino_Energy_Norm_Hist2D",
            rm=args.rewrite,
            debug=args.debug,
        )

        for zdx, hits in enumerate(nhits[:3]):
            fig = make_subplots(
                rows=1,
                cols=4,
                subplot_titles=(
                    "TotalEnergy",
                    "SelectedEnergy",
                    "SolarEnergy",
                    "ClusterEnergy",
                ),
            )
            for jdx, energy in enumerate(
                ["TotalEnergy", "SelectedEnergy", "SolarEnergy", "ClusterEnergy"]
            ):

                for idx, energy_bin in enumerate([8, 12, 16, 20]):
                    this_filter = (data["SignalParticleK"] > (energy_bin - 0.5)) * (
                        data["SignalParticleK"]
                        < (energy_bin + 0.5) * (data["NHits"] >= hits)
                    )  # Filtering genereted neutrinos in 1GeV energy bin

                    # this_data = (
                    #     data[energy][this_filter] - fit_dict[energy]["popt"][1]
                    # ) / fit_dict[energy]["popt"][0]
                    this_data = inverse_quadratic(
                        data[energy][this_filter],
                        fit_dict[energy]["popt"][0],
                        fit_dict[energy]["popt"][1],
                        fit_dict[energy]["popt"][2],
                    )

                    hist, bins = np.histogram(this_data, bins=true_energy_edges)
                    hist = hist / np.sum(hist)

                    rms = np.sqrt(
                        np.mean(
                            (
                                (data["SignalParticleK"][this_filter] - this_data)
                                / data["SignalParticleK"][this_filter]
                            )
                            ** 2
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=true_energy_centers,
                            y=hist,
                            mode="lines",
                            line_shape="hvh",
                            line=dict(color=colors[2 * idx - 2]),
                            name=f"{energy_bin:.0f} MeV",
                            showlegend=True if jdx == 0 else False,
                        ),
                        row=1,
                        col=1 + jdx,
                    )
                    fig.add_vline(
                        x=energy_bin,
                        line_width=1,
                        annotation_text=f"{100*rms:.0f} %",
                        annotation_position="bottom right",
                        line_dash="dash",
                        line_color="grey",
                        row=1,
                        col=1 + jdx,
                    )

            fig = format_coustom_plotly(
                fig,
                title=f"Neutrino Energy Reconstruction (#NHits > {hits}) - {config}",
                legend_title="True Energy",
                ranges=([4, 30], [-0.025, 0.35]),
            )
            fig.add_vline(
                x=30,
                line_width=0,
                annotation_text=f"<b>RMS</b>",
                annotation_position="bottom left",
            )
            fig.update_xaxes(title_text="Reco Neutrino Energy (MeV)")
            fig.update_layout(
                xaxis_title="Neutrino Energy (MeV)",
                yaxis1_title="Norm.",
                showlegend=True,
            )
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Reconstruction_NHits{hits}",
                rm=args.rewrite,
                debug=args.debug,
            )
