import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/vertex/resolution"
data_path = f"{root}/output/data/vertex/resolution"


def double_gaussian(x, a_core, mu, sigma_core, a_tail, sigma_tail):
    """Double Gaussian: narrow core (intrinsic resolution) + broad tail component."""
    return (
        a_core * np.exp(-0.5 * ((x - mu) / abs(sigma_core)) ** 2)
        + a_tail * np.exp(-0.5 * ((x - mu) / abs(sigma_tail)) ** 2)
    )


def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / abs(c)) ** 2)


def exponential_decay(x, a, b, c):
    return a * np.exp(-abs(x) / abs(c)) - b


def gaussian_plus_sym_exp(x, a, mu, sigma, b, tau):
    return a * np.exp(-0.5 * ((x - mu) / abs(sigma)) ** 2) + b * np.exp(-abs(x - mu) / abs(tau))


def _fit_resolution(hx, edges, name, p0_sigma=5, label=None):
    """Fit resolution histogram; returns (fit_function_label, popt, perr, bin_centers)."""
    _xc = 0.5 * (edges[1:] + edges[:-1])
    _w = np.where(hx > 0, 1.0 / np.sqrt(hx), np.inf)
    if "marley" in name.lower():
        try:
            ff = "DoubleGaussian"
            popt, pcov = curve_fit(
                double_gaussian, _xc, hx,
                p0=[np.max(hx), 0, p0_sigma, np.max(hx) * 0.2, 4 * p0_sigma],
                sigma=_w,
                bounds=([0, -20, 0.1, 0, 0.1], [np.max(hx) * 2, 20, 20, np.max(hx), 100]),
            )
            if abs(popt[2]) > abs(popt[4]):
                popt[0], popt[2], popt[3], popt[4] = popt[3], popt[4], popt[0], popt[2]
                _pe = np.sqrt(np.diag(pcov))
                _pe[0], _pe[2], _pe[3], _pe[4] = _pe[3], _pe[4], _pe[0], _pe[2]
                pcov = np.diag(_pe ** 2)
        except Exception:
            ff = "Gaussian"
            popt, pcov = curve_fit(gaussian, _xc, hx, p0=[np.max(hx), 0, p0_sigma], sigma=_w)
    else:
        ff = "Gaussian"
        popt, pcov = curve_fit(gaussian, _xc, hx, p0=[np.max(hx), 0, p0_sigma], sigma=_w)
    if label:
        rprint(f"[cyan][Fit] {label}: {ff}  σ_core = {popt[2]:.2f} cm[/cyan]")
    return ff, popt, np.sqrt(np.diag(pcov)), _xc


def _get_fit_meta(fit_function):
    if fit_function == "DoubleGaussian":
        return {
            "func": double_gaussian,
            "formula": r"A_\mathrm{core}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma_\mathrm{core}^2}\right)+A_\mathrm{tail}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma_\mathrm{tail}^2}\right)",
            "labels": ["Amp. Core", "Mean", "Sigma Core", "Amp. Tail", "Sigma Tail"],
            "units": ["", "cm", "cm", "", "cm"],
            "formats": [".1f", ".1f", ".2f", ".1f", ".2f"],
        }
    elif fit_function == "GaussianPlusSymExp":
        return {
            "func": gaussian_plus_sym_exp,
            "formula": r"A \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) + B \exp\!\left(-\frac{|x-\mu|}{\tau}\right)",
            "labels": ["Amp.", "Mean", "Sigma", "ExpAmp.", "Tau"],
            "units": ["", "cm", "cm", "", "cm"],
            "formats": [".1f", ".1f", ".2f", ".1f", ".2f"],
        }
    elif fit_function == "Gaussian":
        return {
            "func": gaussian,
            "formula": r"A \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
            "labels": ["Amp.", "Mean", "Sigma"],
            "units": ["", "cm", "cm"],
            "formats": [".1f", ".1f", ".2f"],
        }
    else:
        return {
            "func": exponential_decay,
            "formula": r"A \exp\!\left(-\frac{|x|}{\tau}\right) - B",
            "labels": ["Amp.", "Offset", "DecayConst"],
            "units": ["", "cm", "cm"],
            "formats": [".1f", ".1f", ".2f"],
        }


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
parser.add_argument(
    "--purity_threshold",
    type=float,
    default=None,
    help="OpFlash purity threshold for high-purity selection (default: 0.1 for vd, 0.5 for hd)",
)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "VERTEXING",
    "label": {
        "marley": "Neutrino",
        "neutron": "Neutron",
        "gamma": "Gamma",
        None: "Particle",
    },
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=["ANALYSIS"],
    signal="marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

reco_df = npy2df(run, "Reco", debug=False)

hist_list, purity_list, sigma_list = [], [], []

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        max_hist = 0
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        purity_threshold = (
            args.purity_threshold
            if args.purity_threshold is not None
            else 0.5
        )
        rprint(f"[cyan]Purity threshold ({info['GEOMETRY']}): {purity_threshold}[/cyan]")
        this_reco_df = reco_df[
            (reco_df["Geometry"] == info["GEOMETRY"])
            & (reco_df["Version"] == info["VERSION"])
            & (reco_df["Name"] == name)
        ]

        fig1 = make_subplots(rows=1, cols=3, subplot_titles=["RecoX", "RecoY", "RecoZ"])
        for idx, (coord, error) in enumerate(
            zip(["X", "Y", "Z"], ["ErrorX", "ErrorY", "ErrorZ"])
        ):
            #############################################################################
            ########################### Vertexing Error Plot ############################
            #############################################################################

            hx, edges = np.histogram(
                this_reco_df[error][this_reco_df["MatchedOpFlashPur"] > purity_threshold],
                bins=np.arange(-20, 20.5, 0.5),
                density=False,
            )
            fit_function, popt, perr_tmp, _ = _fit_resolution(hx, edges, name, p0_sigma=2, label=f"{coord} overall")

            perr = perr_tmp
            popt_overall, perr_overall = popt, perr
            _xc = 0.5 * (edges[1:] + edges[:-1])
            fit_y = _get_fit_meta(fit_function)["func"](_xc, *popt)

            # Find percentage of events within 1, 2 and 3 sigma
            sigma1 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 1 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 1 * popt[2]))
                ]
            ) / len(this_reco_df[error])
            sigma2 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 3 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 3 * popt[2]))
                ]
            ) / len(this_reco_df[error])
            sigma3 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 5 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 5 * popt[2]))
                ]
            ) / len(this_reco_df[error])

            # Make a 2D histogram of the purity vs the error
            if coord == "X":
                drift_sample = []
                drift_sample_error = []
                for idx, (purity_label, purity_idx) in enumerate(
                    zip(
                        ["No-Match", "Background", "Low-Purity", "High-Purity"],
                        [
                            (this_reco_df["MatchedOpFlashPur"] <= 0)
                            * (this_reco_df["MatchedOpFlashPE"] <= 0),
                            (this_reco_df["MatchedOpFlashPur"] == 0)
                            * (this_reco_df["MatchedOpFlashPE"] >= 0),
                            (this_reco_df["MatchedOpFlashPur"] > 0)
                            * (this_reco_df["MatchedOpFlashPur"] <= purity_threshold),
                            this_reco_df["MatchedOpFlashPur"] > purity_threshold,
                        ],
                    )
                ):
                    drift_sample.append(np.sum(purity_idx) / len(this_reco_df))
                    drift_sample_error.append(
                        np.sqrt(np.sum(purity_idx)) / len(this_reco_df)
                        if np.sum(purity_idx) > 0
                        else 0
                    )
                    h, this_edges = np.histogram(
                        this_reco_df[error][purity_idx],
                        density=False,
                        bins=np.arange(
                            -info["DETECTOR_MAX_X"], info["DETECTOR_MAX_X"], 0.5
                        ),
                    )
                    _fit_meta = _get_fit_meta(fit_function)
                    purity_list.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "Coordinate": coord,
                            "Label": purity_label,
                            "Percentage": 100 * drift_sample[-1],
                            "PercentageError": 100 * drift_sample_error[-1],
                            "Counts": h,
                            "CountsError": np.sqrt(h),
                            "Density": np.asarray(h)
                            / np.sum(h)
                            / (this_edges[1] - this_edges[0]),
                            "DensityError": np.sqrt(h)
                            / np.sum(h)
                            / (this_edges[1] - this_edges[0]),
                            "Values": 0.5 * (this_edges[1:] + this_edges[:-1]),
                            "FitFunction": _fit_meta["func"],
                            "FitFunctionLabel": fit_function,
                            "FitFunctionFormula": _fit_meta["formula"],
                            "Params": popt,
                            "ParamsError": perr,
                            "ParamsLabel": _fit_meta["labels"],
                            "ParamsUnit": _fit_meta["units"],
                            "ParamsFormat": _fit_meta["formats"],
                        }
                    )
                    # Print the drift sample percentages
                    rprint(
                        f"{purity_label}\t-> {100*drift_sample[idx]:.2f}% +/- {100*drift_sample_error[idx]:.2f}%"
                    )

            h, edges = np.histogram(
                this_reco_df[error],
                bins=edges,
                density=True,
            )
            h_total, edges_total = np.histogram(
                this_reco_df[error],
                bins=np.arange(
                    (
                        info[f"DETECTOR_MIN_{coord}"]
                        if coord != "Z"
                        else -info[f"DETECTOR_MAX_{coord}"]
                    ),
                    info[f"DETECTOR_MAX_{coord}"] + 10,
                    10,
                ),
                density=True,
            )

            _fit_meta = _get_fit_meta(fit_function)
            hist_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Energy": None,
                    "CoordinateBin": None,
                    "Coordinate": coord,
                    "Error": np.asarray(h),
                    "Values": 0.5 * (edges[1:] + edges[:-1]),
                    "Sigma": popt[2],
                    "SigmaError": perr[2],
                    "FitFunctionLabel": fit_function,
                    "FitFunctionFormula": _fit_meta["formula"],
                    "Params": popt,
                    "ParamsError": perr,
                    "ParamsLabel": _fit_meta["labels"],
                    "ParamsUnit": _fit_meta["units"],
                    "ParamsFormat": _fit_meta["formats"],
                }
            )

            if np.max(h) > max_hist:
                max_hist = np.max(h)

            _res_edges = np.arange(-20, 20.5, 0.5)
            _purity_mask = this_reco_df["MatchedOpFlashPur"] > purity_threshold

            # Energy scan
            _es_energies, _es_sigma, _es_sigma_err = [], [], []
            for energy in np.arange(6, 31, 2):
                _emask = (
                    (this_reco_df["SignalParticleK"] > (energy - 1))
                    & (this_reco_df["SignalParticleK"] < energy + 1)
                )
                _edf = this_reco_df[_emask]
                if len(_edf) < 50:
                    continue
                _evals = _edf[error][_purity_mask[_emask]] if coord == "X" else _edf[error]
                hx, edges = np.histogram(_evals, bins=_res_edges, density=True)
                if np.sum(hx) == 0:
                    continue
                try:
                    fit_function, popt, perr, _xc = _fit_resolution(hx, edges, name)
                except Exception:
                    continue
                _es_energies.append(energy)
                _es_sigma.append(popt[2])
                _es_sigma_err.append(perr[2])
                _fit_meta = _get_fit_meta(fit_function)
                hist_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Energy": energy,
                        "CoordinateBin": None,
                        "Coordinate": coord,
                        "Error": np.asarray(hx),
                        "Values": _xc,
                        "Sigma": popt[2],
                        "SigmaError": perr[2],
                        "FitFunction": _fit_meta["func"],
                        "FitFunctionLabel": fit_function,
                        "FitFunctionFormula": _fit_meta["formula"],
                        "Params": popt,
                        "ParamsError": perr,
                        "ParamsLabel": _fit_meta["labels"],
                        "ParamsUnit": _fit_meta["units"],
                        "ParamsFormat": _fit_meta["formats"],
                    }
                )

            # Coordinate scan: resolution vs true position along each axis
            _coord_min = info[f"DETECTOR_MIN_{coord}"] if coord != "Z" else -info[f"DETECTOR_MAX_{coord}"]
            _coord_max = info[f"DETECTOR_MAX_{coord}"]
            _n_cbins = 10
            _cbw = (_coord_max - _coord_min) / _n_cbins
            _coord_centers = np.linspace(_coord_min + _cbw / 2, _coord_max - _cbw / 2, _n_cbins)
            _cs_bins, _cs_sigma, _cs_sigma_err = [], [], []

            for coord_bin in _coord_centers:
                _cmask = (
                    (this_reco_df[f"SignalParticle{coord}"] >= coord_bin - _cbw / 2)
                    & (this_reco_df[f"SignalParticle{coord}"] < coord_bin + _cbw / 2)
                )
                if coord == "X":
                    _cmask &= _purity_mask
                _cdf = this_reco_df[_cmask]
                if len(_cdf) < 50:
                    continue
                hx, edges = np.histogram(_cdf[error], bins=_res_edges, density=True)
                if np.sum(hx) == 0:
                    continue
                try:
                    fit_function, popt, perr, _xc = _fit_resolution(hx, edges, name)
                except Exception:
                    continue
                _cs_bins.append(coord_bin)
                _cs_sigma.append(popt[2])
                _cs_sigma_err.append(perr[2])
                _fit_meta = _get_fit_meta(fit_function)
                hist_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Energy": None,
                        "CoordinateBin": coord_bin,
                        "Coordinate": coord,
                        "Error": np.asarray(hx),
                        "Values": _xc,
                        "Sigma": popt[2],
                        "SigmaError": perr[2],
                        "FitFunction": _fit_meta["func"],
                        "FitFunctionLabel": fit_function,
                        "FitFunctionFormula": _fit_meta["formula"],
                        "Params": popt,
                        "ParamsError": perr,
                        "ParamsLabel": _fit_meta["labels"],
                        "ParamsUnit": _fit_meta["units"],
                        "ParamsFormat": _fit_meta["formats"],
                    }
                )

            sigma_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Coordinate": coord,
                    "Error": np.asarray(h_total),
                    "Values": np.asarray(0.5 * (edges_total[1:] + edges_total[:-1])),
                    "Zoom": np.asarray(h),
                    "ZoomBins": np.asarray(0.5 * (edges[1:] + edges[:-1])),
                    "NoMatch": 100 * drift_sample[0],
                    "Background": 100 * drift_sample[1],
                    "LowPurity": 100 * drift_sample[2],
                    "HighPurity": 100 * drift_sample[3],
                    "Sigma": popt_overall[2],
                    "SigmaError": perr_overall[2],
                    "Sigma1": sigma1,
                    "Sigma3": sigma2,
                    "Sigma5": sigma3,
                    "Mean": np.mean(this_reco_df[error]),
                    "Median": np.median(this_reco_df[error]),
                    "STD": np.std(this_reco_df[error]),
                    "STDError": np.std(this_reco_df[error]) / np.sqrt(len(this_reco_df[error])),
                    "EnergyScanBins": np.asarray(_es_energies),
                    "EnergyScanSigma": np.asarray(_es_sigma),
                    "EnergyScanSigmaError": np.asarray(_es_sigma_err),
                    "CoordScanBins": np.asarray(_cs_bins),
                    "CoordScanSigma": np.asarray(_cs_sigma),
                    "CoordScanSigmaError": np.asarray(_cs_sigma_err),
                }
            )

        df = pd.DataFrame(hist_list)
        df = explode(df, ["Error", "Values"])
        df["Error"] = df["Error"].astype(float)
        df["Values"] = df["Values"].astype(float)

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & df["Energy"].isna() & df["CoordinateBin"].isna()]
            # print(this_df)
            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=default[j], width=2),
                    # Select the second entry in params which is the sigma
                    name=f"{coord} Sigma: {this_df['Params'].values[0][2]:.2f} cm",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Density", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            log=(False, True),
            tickformat=(None, ".1s"),
            ranges=([-20, 20], [-3, 0]),
            title=f"Error Distribution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.68, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="Vertex True - Reco (cm)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error_LogY",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & df["Energy"].isna() & df["CoordinateBin"].isna()]

            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=default[j], width=2),
                    name=f"{coord}",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Density", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            log=(False, False),
            ranges=([-20, 20], [0, 1.1 * max_hist]),
            title=f"Error Distribution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.68, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig = make_subplots(rows=1, cols=3)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):
            if coord == "X":
                this_reco_matrix = {
                    "Config": config,
                    "Name": name,
                    f"{coord}": this_reco_df[f"SignalParticle{coord}"],
                    f"Error{coord}": this_reco_df[f"Error{coord}"],
                    "Energy": this_reco_df["SignalParticleK"],
                    "Matched": this_reco_df["MatchedOpFlashPE"] > 0,
                }
                save_pkl(
                    pd.DataFrame(this_reco_matrix),
                    data_path,
                    config,
                    name,
                    None,
                    filename= f"Vertex_Matrix_{coord}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

            h, x, y = np.histogram2d(
                this_reco_df[f"SignalParticle{coord}"],
                this_reco_df[f"Error{coord}"],
                bins=[100, 100],
                density=True,
            )

            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=np.log10(h.T),
                    coloraxis="coloraxis",
                ),
                row=1,
                col=j + 1,
            )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title=f"Error Heatmap - {config}",
            add_watermark=False,
        )

        # Add title to colorbar
        fig.update_coloraxes(colorbar=dict(title="log(Density)"))

        for j, coord in enumerate(["X", "Y", "Z"]):
            fig.update_yaxes(title_text="Reco - True (cm)", row=1, col=1 + j)
            fig.update_xaxes(title_text=f"True {coord} (cm)", row=1, col=1 + j)

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Matrix",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Make a plot of the error vs purity heatmaps
        purity_df = pd.DataFrame(purity_list)
        cols = ["Counts", "CountsError", "Density", "DensityError", "Values"]
        this_purity_df = explode(purity_df, cols)
        for col in cols:
            this_purity_df[col] = this_purity_df[col].astype(float)

        fig = make_subplots(rows=1, cols=1)
        maxy = 0
        for k, purity_label in enumerate(this_purity_df["Label"].unique()[::-1]):
            this_df = this_purity_df[
                (this_purity_df["Coordinate"] == "X")
                & (this_purity_df["Label"] == purity_label)
            ]

            if purity_label == "High-Purity":
                df_x = df[(df["Coordinate"] == "X") & (df["Energy"].isna())]
                _popt_full = df_x["Params"].values[0]
                _fit_label = df_x["FitFunctionLabel"].values[0]
                _fit_func = _get_fit_meta(_fit_label)["func"]
                _xplot = 0.5 * (edges[1:] + edges[:-1])
                fig.add_trace(
                    go.Scatter(
                        x=_xplot,
                        y=_fit_func(_xplot, *_popt_full),
                        mode="lines",
                        line_shape="spline",
                        line=dict(color="red", dash="dash"),
                        name=f"Fit ({_fit_label}): Sigma {_popt_full[2]:.1f} cm",
                    ),
                    row=1,
                    col=1,
                )
            if np.max(this_df["Counts"]) > maxy:
                maxy = np.max(this_df["Counts"])

            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Counts"],
                    error_y=dict(type="data", array=this_df["CountsError"]),
                    mode="lines+markers",
                    line_shape="hvh",
                    line=dict(color=default[k], width=2),
                    name=f"{purity_label}: {this_df['Percentage'].values[0]:.1f}%",
                ),
                row=1,
                col=1,
            )
            # Add gaussian fit for the high purity sample

        # fig.update_layout(legend_title_text="Energy (MeV)")
        fig.update_xaxes(title_text="Vertex X True - Reco (cm)")
        fig.update_yaxes(title_text="Counts")
        fig = format_coustom_plotly(
            fig,
            # add_watermark=False,
            title=f"X Error vs OpFlash Purity - {config}",
            legend=dict(x=0.72, y=0.99),
            legend_title="Matched OpFlash",
            log=(False, True),
            tickformat=(None, ".1s"),
            ranges=(None, [0, np.log10(1.5 * maxy)]),
        )

        for rangex in (None, [-20, 20]):
            if rangex is not None:
                fig.update_xaxes(range=rangex)

            save_figure(
                fig,
                save_path,
                config,
                name,
                None,
                filename=(
                    f"Vertex_Error_Purity"
                    if rangex is None
                    else f"Vertex_Error_Purity_Zoom"
                ),
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        fig = make_subplots(rows=1, cols=1)
        _sigma_df = pd.DataFrame(sigma_list)
        for j, coord in enumerate(["X", "Y", "Z"]):
            _row = _sigma_df[
                (_sigma_df["Coordinate"] == coord)
                & (_sigma_df["Config"] == config)
                & (_sigma_df["Name"] == name)
            ]
            if len(_row) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_row["EnergyScanBins"].values[0],
                    y=_row["EnergyScanSigma"].values[0],
                    error_y=dict(type="data", array=_row["EnergyScanSigmaError"].values[0]),
                    mode="lines+markers",
                    line_shape="spline",
                    line=dict(color=default[j % len(default)], width=2),
                    name=f"{coord}",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Resolution (cm)", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            ranges=([6, 30], [0, 4]),
            title=f"Vertex Resolution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.82, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Resolution",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Resolution vs true coordinate position (one subplot per axis)
        fig = make_subplots(rows=1, cols=3, subplot_titles=["X (Drift)", "Y", "Z"])
        sigma_df = pd.DataFrame(sigma_list)
        for j, coord in enumerate(["X", "Y", "Z"]):
            row_data = sigma_df[
                (sigma_df["Coordinate"] == coord)
                & (sigma_df["Config"] == config)
                & (sigma_df["Name"] == name)
            ]
            if len(row_data) == 0:
                continue
            _bins = row_data["CoordScanBins"].values[0]
            _sig  = row_data["CoordScanSigma"].values[0]
            _serr = row_data["CoordScanSigmaError"].values[0]
            if len(_bins) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_bins,
                    y=_sig,
                    error_y=dict(type="data", array=_serr),
                    mode="lines+markers",
                    line_shape="spline",
                    line=dict(color=default[j % len(default)], width=2),
                    name=coord,
                ),
                row=1,
                col=j + 1,
            )
            fig.update_xaxes(title_text=f"True {coord} (cm)", row=1, col=j + 1)
            fig.update_yaxes(title_text="Resolution (cm)", row=1, col=j + 1)

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            ranges=(None, [0, 4]),
            title=f"Vertex Resolution vs Position - {config} {name}",
            legend_title="Coordinate",
            legend=dict(x=0.82, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))),
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Resolution_vs_Position",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Save the sigma list to a dataframe
        for filename, df in zip(
            ["Resolution", "Purity_Match_Resolution", "Vertex_Resolution"],
            [
                pd.DataFrame(hist_list),
                pd.DataFrame(purity_list),
                pd.DataFrame(sigma_list),
            ],
        ):
            save_df(
                df=df,
                path=data_path,
                config=config,
                name=name,
                subfolder=None,
                filename=filename,
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
