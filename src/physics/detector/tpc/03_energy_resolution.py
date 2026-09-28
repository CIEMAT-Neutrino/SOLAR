import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/TPC/resolution/neutrino"
data_path = f"{root}/output/data/TPC/resolution/neutrino"

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

RMS_data = []
gauss_data = []
for label, params in zip(
    ["True", "Reco", "None"],
    [
        {
            "DEFAULT_ENERGY_TIME": "Time",
            "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime",
        },
        {},
        {
            "DEFAULT_ENERGY_TIME": "AverageDriftTime",
            "DEFAULT_ADJCL_ENERGY_TIME": "AdjClAverageDriftTime",
        },
    ],
):
    this_run = compute_reco_workflow(
        run,
        configs,
        params=params,
        workflow=user_input["workflow"],
        debug=user_input["debug"],
    )

    this_filtered_run, mask, output = compute_filtered_run(
        this_run,
        configs,
        params={("Reco", "TrueMain"): ("equal", True)},
        presets=[user_input["workflow"]],
        signal = "marley" in args.name,
        debug=user_input["debug"],
    )
    rprint(output)

    data = this_filtered_run["Reco"]

    # Plot the calibration workflow

    fit = {
        "color": "grey",
        "opacity": 1,
        "print": False,
        "show": False,
    }

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", {}, output, debug=args.debug
        )
        fig2 = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("NHit Threshold 1", "NHit Threshold 2", "NHit Threshold 3"),
        )
        # MainK is the backtracked main-track energy (a Reco-tree branch, not a pure
        # truth one). Its natural truth counterpart is whichever particle the sample's
        # primary/signal particle actually is: for marley (electron recoil) that is
        # ElectronK, built from TSignalK/TSignalPDG==11/TSignalMother==0 (see
        # compute_particle_energies in lib/signal.py); for a mono-energetic gamma
        # calibration sample the primary *is* the gamma, so SignalParticleK -- the
        # truth K.E. of the injected/signal particle, generic across samples -- is
        # already the correct reference (ElectronK would be ~0, no electron primary).
        # Neither is SignalParticleK-for-cluster-variables' meaning (true total event
        # energy) nor ClusterEnergy (a different, calorimetric reco quantity).
        variable_columns = {
            "Cluster": "ClusterEnergy",
            "Total": "TotalEnergy",
            "Selected": "SelectedEnergy",
            "Solar": "SolarEnergy",
            "MainK": "MainK",
        }
        variable_truth = {
            "Cluster": "SignalParticleK",
            "Total": "SignalParticleK",
            "Selected": "SignalParticleK",
            "Solar": "SignalParticleK",
            "MainK": "ElectronK",
        }
        for name, (jdx, variable), (kdx, nhit) in product(
            configs[config],
            enumerate(
                ["Cluster", "Total", "Selected", "Solar", "MainK"],
            ),
            enumerate(nhits[:3]),
        ):
            column = variable_columns[variable]
            truth = variable_truth[variable]
            if variable == "MainK" and "gamma" in name.lower():
                truth = "SignalParticleK"

            if variable == "MainK":
                # Fractional (Truth - MainK) / Truth residual: a single Gaussian
                # fit to this distribution gives a resolution (sigma) directly
                # comparable across NHit thresholds/Drift labels, and is more
                # robust to non-Gaussian tails than the plain per-energy-bin RMS
                # above, since the fit only responds to the core of the peak.
                valid = data["NHits"] >= nhit
                truth_arr = np.asarray(data[truth])[valid]
                reco_arr = np.asarray(data[column])[valid]
                nz = truth_arr != 0
                residual = (truth_arr[nz] - reco_arr[nz]) / truth_arr[nz]
                residual = residual[np.isfinite(residual)]

                # Bin the residual by hand rather than via get_hist1d/generate_bins:
                # for a near-perfect match (e.g. MainK vs SignalParticleK on the
                # mono-energetic gamma sample) most residuals are exactly 0, so the
                # data range can collapse to zero width and break auto-ranging.
                res_edges = hist_y = hist_counts = res_centers = hist_sigma = None
                if len(residual) > 0:
                    lo, hi = np.percentile(residual, [1, 99])
                    if hi <= lo:
                        pad = max(abs(lo), abs(hi), 1e-3) * 0.1
                        lo, hi = lo - pad, hi + pad
                    res_edges = np.linspace(lo, hi, 61)
                    hist_counts, _ = np.histogram(residual, bins=res_edges)
                    res_centers = 0.5 * (res_edges[:-1] + res_edges[1:])

                    # Normalize to a probability density (area = 1) so the marley
                    # (n~10^5) and gamma (n~10^5, but ~all in one bin) distributions,
                    # and different NHit cuts, sit on a common y-scale.
                    bin_width = res_edges[1] - res_edges[0]
                    n_total = np.sum(hist_counts)
                    hist_y = hist_counts / (n_total * bin_width)
                    hist_sigma = np.where(
                        hist_counts > 0,
                        np.sqrt(hist_counts) / (n_total * bin_width),
                        np.inf,
                    )

                fig3 = make_subplots(rows=1, cols=1)
                if res_centers is not None:
                    fig3.add_trace(
                        go.Scatter(
                            x=res_centers,
                            y=hist_y,
                            line=dict(shape="hvh"),
                            mode="lines",
                            name="(Truth - MainK) / Truth",
                        )
                    )

                if (
                    res_centers is not None
                    and len(res_centers) > 4
                    and np.count_nonzero(hist_y) >= 4
                ):
                    # fit_hist1d's built-in "gauss" initial guess seeds sigma from
                    # std(counts) rather than std(the data), which regularly fails
                    # to converge here; seed curve_fit ourselves from the residual
                    # array directly instead.
                    def _gauss_model(x, a, mu, sigma):
                        return gauss(x, [a, mu, sigma])

                    p0 = [
                        np.max(hist_y),
                        np.mean(residual),
                        max(np.std(residual), 1e-3),
                    ]
                    bounds = ([0, -np.inf, 1e-6], [np.inf, np.inf, np.inf])
                    try:
                        popt, pcov = curve_fit(
                            _gauss_model,
                            res_centers,
                            hist_y,
                            p0=p0,
                            sigma=hist_sigma,
                            bounds=bounds,
                            maxfev=10000,
                        )
                    except (RuntimeError, ValueError):
                        popt, pcov = curve_fit(
                            _gauss_model,
                            res_centers,
                            hist_y,
                            p0=p0,
                            bounds=bounds,
                            maxfev=10000,
                        )
                    perr = np.sqrt(np.diag(pcov))
                    fit_labels = ["Amplitude", "Mean", "Sigma"]

                    fig3.add_trace(
                        go.Scatter(
                            x=res_centers,
                            y=_gauss_model(res_centers, *popt),
                            mode="lines",
                            line=dict(dash="dash", color="red"),
                            name=f"Gauss fit: sigma={popt[2]:.3f}+/-{perr[2]:.3f}",
                        )
                    )

                    gauss_data.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "Drift": label,
                            "#Hits": nhit,
                            "Variable": column,
                            "Truth": truth,
                            "Values": res_centers,
                            "ValuesUnit": "(Truth - Reco) / Truth",
                            "Counts": hist_y,
                            "CountsUnit": "",
                            "RawCounts": hist_counts,
                            # No live callable is stored here (unlike lib.fitting.gauss
                            # itself) so this pkl can be unpickled outside this repo --
                            # a function object pickles by module reference, which an
                            # external reader can't resolve. FitFunctionFormula is the
                            # portable description of the fit: gauss(x) = a * exp(...).
                            "FitFunctionLabel": "Gaussian",
                            "FitFunctionFormula": "a * exp(-0.5 * ((x - mu) / sigma)^2)",
                            "Params": popt,
                            "ParamsLabel": fit_labels,
                            "ParamsFormat": [".1f", ".3f", ".3f"],
                            "ParamsError": perr,
                            "ParamsUnit": [
                                "",
                                "",
                                "",
                            ],
                            "p0": popt[0],
                            "p1": popt[1],
                            "p2": popt[2],
                            "Amplitude": popt[0],
                            "Mean": popt[1],
                            "MeanError": perr[1],
                            "Resolution": popt[2],
                            "ResolutionError": perr[2],
                            "Fitted": True,
                        }
                    )
                elif len(residual) > 0:
                    # Too few populated bins for a Gaussian fit -- typically because
                    # the residual is a near-delta distribution (e.g. MainK matches
                    # SignalParticleK to float precision for ~all gamma events).
                    # Record the raw mean/std instead of silently dropping the row.
                    gauss_data.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "Drift": label,
                            "#Hits": nhit,
                            "Variable": column,
                            "Truth": truth,
                            "Values": res_centers,
                            "ValuesUnit": "(Truth - Reco) / Truth",
                            "Counts": hist_y,
                            "CountsUnit": "",
                            "RawCounts": hist_counts,
                            "FitFunction": None,
                            "FitFunctionLabel": "None (too few populated bins)",
                            "FitFunctionFormula": None,
                            "Params": None,
                            "ParamsLabel": ["Amplitude", "Mean", "Sigma"],
                            "ParamsFormat": [".1f", ".3f", ".3f"],
                            "ParamsError": None,
                            "ParamsUnit": [
                                "",
                                "",
                                "",
                            ],
                            "p0": None,
                            "p1": float(np.mean(residual)),
                            "p2": float(np.std(residual)),
                            "Amplitude": None,
                            "Mean": float(np.mean(residual)),
                            "MeanError": float(
                                np.std(residual) / np.sqrt(len(residual))
                            ),
                            "Resolution": float(np.std(residual)),
                            "ResolutionError": None,
                            "Fitted": False,
                            "ExactMatchFraction": float(
                                np.mean(np.abs(residual) < 1e-6)
                            ),
                        }
                    )

                fig3 = format_coustom_plotly(
                    fig3,
                    title=f"{column} Resolution Fit - NHit {nhit} - {label} - {config} {name}",
                )
                fig3.update_layout(
                    xaxis_title="(Truth - MainK) / Truth",
                    yaxis_title="Counts",
                )
                save_figure(
                    fig3,
                    save_path,
                    config,
                    name,
                    filename=f"{column}_{label}ResolutionFit_NHits{nhit}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

            fig1 = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Energy Smearing", "Energy Resolution"),
            )

            miny = np.min(data[column][data["NHits"] >= nhit])
            maxy = np.max(data[column][data["NHits"] >= nhit])
            minx = np.min(data[truth][data["NHits"] >= nhit])
            maxx = np.max(data[truth][data["NHits"] >= nhit])

            miny_idx = np.where(miny < reco_energy_edges)[0][0]
            maxy_idx = np.where(maxy > reco_energy_edges)[0][-1]
            minx_idx = np.where(minx < reco_energy_edges)[0][0]
            maxx_idx = np.where(maxx > reco_energy_edges)[0][-1]

            x, y, h = get_hist2d(
                data[truth][data["NHits"] >= nhit],
                data[column][data["NHits"] >= nhit],
                per=None,
                norm=False,
                acc=(
                    reco_energy_edges[minx_idx:maxx_idx],
                    reco_energy_edges[miny_idx:maxy_idx],
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
            for energy_bin in reco_energy_centers:
                idx = np.where(
                    (
                        data[truth][data["NHits"] >= nhit]
                        > energy_bin - reco_ebin / 2
                    )
                    & (
                        data[truth][data["NHits"] >= nhit]
                        < energy_bin + reco_ebin / 2
                    )
                )
                rms = np.sqrt(
                    np.mean(
                        np.power(
                            (
                                data[truth][data["NHits"] >= nhit][idx]
                                - data[column][data["NHits"] >= nhit][idx]
                            )
                            / data[truth][data["NHits"] >= nhit][idx],
                            2,
                        )
                    )
                )

                # Compute an associated error on the RMS dependent on the number of events in the bin
                RMS.append(float(rms))
                error = np.sqrt(
                    np.mean(
                        np.power(
                            (
                                data[truth][data["NHits"] >= nhit][idx]
                                - data[column][data["NHits"] >= nhit][idx]
                            )
                            / data[truth][data["NHits"] >= nhit][idx],
                            2,
                        )
                    )
                    / np.sqrt(len(idx[0]))
                )

                RMS_error.append(float(error))

            RMS_data.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Drift": label,
                    "#Hits": nhit,
                    "Variable": column,
                    "Values": reco_energy_centers,
                    "RMS": RMS,
                    "RMSError": RMS_error,
                }
            )

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
                ["RMS (True - Reco) / True", column],
                ["black", compare[jdx]],
                [True, kdx == 0],
                [1, 1],
                [2, kdx + 1],
            ):
                fig.add_trace(
                    go.Scatter(
                        x=reco_energy_centers,
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

            # Draw grey error bands
            fig1.add_trace(
                go.Scatter(
                    x=reco_energy_centers,
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
                    x=reco_energy_centers,
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
                title=f"{column} - NHit Threshold {nhit} - {config} {name}",
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
                filename=f"{column}_{label}Resolution_NHits{nhit}",
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
            filename=f"Neutrino_Energy_{label}Resolution",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        # fig2.show()
save_pkl(
    RMS_data,
    data_path,
    config,
    name,
    filename=f"Neutrino_Energy_Resolution",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

# Gaussian-fit resolution results (MainK only), following the df_fit/Charge_Correction_Factor
# template from src/physics/calibration/01_correction.py: a row per (Drift label, NHit)
# with the fit function/params/errors, saved as its own pkl via save_df.
save_df(
    pd.DataFrame(gauss_data),
    data_path,
    config,
    name,
    filename=f"MainK_Resolution_GaussianFit",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)
