import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/PDS/opflash"
data_path = f"{root}/data/PDS/opflash"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the OpFlash distributions of the signal"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6",
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

user_input = {"workflow": "OPFLASH", "rewrite": args.rewrite, "debug": args.debug}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)

run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)


def linear_cut(x, a, b):
    y = a - a * b * x / time_array[-1]  # Linear attenuation
    return y


def quadratic_cut(x, a, b, c):
    y = (
        a - a * b * x / time_array[-1] + c * (x / time_array[-1]) ** 2
    )  # Quadratic attenuation
    return y


def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c


def MinPECutCharge(df, charges):
    # For each charge value x, compute the parameters a, b, c from light_map_df
    parameter_dict = {}
    for parameter in df["Parameter"].unique():
        p0 = df[df["Parameter"] == parameter]["p0"]
        p1 = df[df["Parameter"] == parameter]["p1"]
        p2 = df[df["Parameter"] == parameter]["p2"]
        parameter_dict[parameter] = (
            p0.values[0] * charges**2 + p1.values[0] * charges + p2.values[0]
        )
    a = parameter_dict["Amplitude"]
    b = parameter_dict["Attenuation"]
    c = parameter_dict["Correction"]
    return a, b, c


def MinPECut(df, charges, times):
    a, b, c = MinPECutCharge(df, charges)

    # Vectorized computation
    return 10 ** (
        np.array(a)
        - np.array(a) * np.array(b) * times / time_array[-1]
        + np.array(c) * (times / time_array[-1]) ** 2
    )


for config in configs:
    if config in ["vd_1x8x14_3view_30deg", "hd_1x2x6"]:
        rprint(f"Processing PE matching for {config}...")
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
        )
        charge_array = np.linspace(
            np.percentile(run["Reco"]["Charge"], 5),
            np.percentile(run["Reco"]["Charge"], 95),
            10,
        )
        charge_bin = charge_array[1] - charge_array[0]
        time_array = np.linspace(0, np.percentile(run["Reco"]["Time"], 90), 20)
        time_bin = time_array[1] - time_array[0]
        # run["Reco"]["MatchedOpFlashPE"][run["Reco"]["MatchedOpFlashPE"] == -1e6] = 0
        light_data = []
        light_map = []

        for name in configs[config]:
            df_list = []
            for var, drift, charge in product(
                ["MatchedOpFlashPE"], time_array, charge_array
            ):
                this_run, mask, output = compute_filtered_run(
                    run,
                    configs,
                    params={
                        ("Reco", "Name"): ("equal", name),
                        ("Reco", "TrueMain"): ("equal", True),
                        ("Reco", var): ("bigger", 0),
                        ("Reco", "MatchedOpFlashPlane"): ("equal", 0),
                        ("Reco", "Time"): (
                            "between",
                            (drift - time_bin / 2, drift + time_bin / 2),
                        ),
                        ("Reco", "Charge"): (
                            "between",
                            (charge - charge_bin / 2, charge + charge_bin / 2),
                        ),
                    },
                    debug=user_input["debug"],
                )
                # rprint(output)
                mean = (
                    np.mean(this_run["Reco"][var])
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )
                median = (
                    np.median(this_run["Reco"][var])
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )
                lower_quantile = (
                    np.quantile(this_run["Reco"][var], 0.16)
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )
                upper_quantile = (
                    np.quantile(this_run["Reco"][var], 0.84)
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )
                center_quantile = (
                    np.quantile(this_run["Reco"][var], 0.50)
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )
                STD = (
                    np.std(this_run["Reco"][var])
                    if len(this_run["Reco"][var]) > 0
                    else np.nan
                )

                df_list.append(
                    {
                        "Config": config,
                        "Name": name,
                        "Drift (us)": drift + time_bin / 2,
                        "Charge (ADC x tick)": charge + charge_bin / 2,
                        "Mean": mean,
                        "Median": median,
                        "Lower": lower_quantile,
                        "Upper": upper_quantile,
                        "Center": center_quantile,
                        "STD": STD,
                    }
                )
            df = pd.DataFrame(df_list)

            # Plot
            fig = make_subplots(1, 1)
            jdx = 0
            for idx, charge in enumerate(df["Charge (ADC x tick)"].unique()):
                # Only plot the 3 entries in charge_array that are evenly spaced
                if idx % (len(charge_array) // 3) != 1:
                    continue
                # Given the log of a linear equation that attenuates PE_0 * PE_attenuation over a time distance of time_array[-1], find the fit parameter PE_attenuation
                this_df = df[df["Charge (ADC x tick)"] == charge]

                fig.add_trace(
                    go.Scatter(
                        x=this_df["Drift (us)"],
                        y=this_df["Lower"],
                        mode="lines",
                        line=dict(
                            color=colors[jdx],
                            width=2,
                            dash="dot",
                        ),
                        showlegend=False,
                        legendgroup="Quantile",
                        legendgrouptitle=dict(text="Limit", font=dict(size=14)),
                        name="Lower Quantile",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=this_df["Drift (us)"],
                        y=this_df["Upper"],
                        mode="lines",
                        line=dict(width=2, dash="dash", color=colors[jdx]),
                        fill="tonexty",
                        fillcolor=colors[jdx]
                        .replace("1)", "0.1)")
                        .replace("rgb", "rgba"),
                        showlegend=True,
                        legendgrouptitle=dict(text="Limit", font=dict(size=14)),
                        legendgroup="Quantile",
                        name="16-84%",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=this_df["Drift (us)"],
                        y=this_df["Median"],
                        mode="lines",
                        line=dict(color=colors[jdx], width=2),
                        marker=dict(size=6),
                        name=f"{charge + charge_bin/2:.1f}",
                        showlegend=True,
                        legendgroup=1,
                        legendgrouptitle=dict(text="Charge", font=dict(size=14)),
                    )
                )
                jdx += 1

            fig = format_coustom_plotly(
                fig,
                log=(False, True),
                title=f"MainOpFlash PE vs Drift - {config}",
            )
            # Add x and y axis titles
            fig.update_xaxes(title_text="Drift (us)")
            fig.update_yaxes(title_text="Median OpFlashPE (PE)", rangemode="tozero")
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename="MainOpFlashPE_vs_Drift",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            for cut, modulator, jdx in zip(
                ["Upper", "Upper", "Upper", "Center", "Lower", "Lower", "Lower"],
                (
                    [14, 4, 1, 1, 0.175, 0.5, 1]
                    if config == "vd_1x8x14_3view_30deg"
                    else [2, 1.2, 1, 1, 0.1, 0.5, 1]
                ),
                [0, 1, 2, 0, 0, 1, 2],
            ):

                fig = make_subplots(1, 2)
                attenuation = []
                amplitude = []
                correction = []
                for kdx, charge in enumerate(df["Charge (ADC x tick)"].unique()):
                    # Given the log of a linear equation that attenuates PE_0 * PE_attenuation over a time distance of time_array[-1], find the fit parameter PE_attenuation
                    # print(df)
                    this_df = df[
                        (df["Charge (ADC x tick)"] == charge)
                        * (df["Name"] == name)
                        * (df["Config"] == config)
                    ]
                    # Fit only the mean values
                    fit_x = this_df["Drift (us)"].to_numpy()
                    # print(this_df)
                    fit_y = np.log10(modulator * this_df[cut].to_numpy())
                    # rprint(len(fit_x), fit_x, len(fit_y), fit_y)

                    # print(fit_y)
                    popt, pcov = curve_fit(
                        quadratic_cut,
                        fit_x[3:] if config == "vd_1x8x14_3view_30deg" else fit_x[1:],
                        fit_y[3:] if config == "vd_1x8x14_3view_30deg" else fit_y[1:],
                        p0=(
                            [np.max(fit_y), 0, 0]
                            if config == "vd_1x8x14_3view_30deg"
                            else [np.max(fit_y), 0.5, 0]
                        ),
                        bounds=(
                            ([-np.inf, -10, -np.inf], [np.max(fit_y), 0, np.inf])
                            if config == "vd_1x8x14_3view_30deg"
                            else ([0, -10, -np.inf], [np.max(fit_y) * 1.1, 10, np.inf])
                        ),
                    )
                    perr = np.sqrt(np.diag(pcov))

                    amplitude.append(popt[0])
                    attenuation.append(popt[1])
                    correction.append(popt[2])

                    fit_y_vals = quadratic_cut(fit_x, *popt)
                    fig.add_trace(
                        go.Scatter(
                            x=fit_x,
                            y=fit_y_vals,
                            mode="lines",
                            line=dict(color=colors[kdx % len(colors)], dash="dash"),
                            showlegend=False,
                            name=f"Fit {charge + charge_bin/2:.1f}",
                            legendgroup=1,
                        ),
                        col=1,
                        row=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=this_df["Drift (us)"],
                            y=np.log10(modulator * this_df[cut].to_numpy()),
                            mode="markers",
                            line=dict(color=colors[kdx % len(colors)], width=2),
                            marker=dict(size=6),
                            name=f"{charge + charge_bin/2:.1f}",
                            legendgroup=1,
                        ),
                        col=1,
                        row=1,
                    )

                    light_data.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "Cut": cut,
                            "Modulator": modulator,
                            "Charge": int(charge),
                            "Time": fit_x,
                            "PE": 10**fit_y,
                            "FitFunction": quadratic_cut,
                            "FitFunctionLabel": "Light-Map",
                            "FitFunctionFormula": r"$y = a - a \cdot b \cdot \frac{x}{t_{max}} + c \cdot \left(\frac{x}{t_{max}}\right)^2$",
                            "Params": popt,
                            "ParamsError": perr,
                            "ParamsLabels": ["Amplitude", "Attenuation", "Correction"],
                            "ParamsFormat": [".2f", ".2f", ".2f"],
                        }
                    )

                # Draw the values for popt[0] and popt[1] as a function of charge in the second plot
                for idx, (value, label) in enumerate(
                    zip(
                        [amplitude, attenuation, correction],
                        ["Amplitude", "Attenuation", "Correction"],
                    )
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=charge_array,
                            y=value,
                            mode="markers",
                            line=dict(color=default[idx % len(default)], width=2),
                            marker=dict(size=6),
                            name=label,
                            legendgroup=2,
                            legendgrouptitle_text="Parameter",
                        ),
                        col=2,
                        row=1,
                    )
                    # Add quadratic fit to the second plot
                    fit_x = charge_array
                    fit_y = np.array(value)

                    popt, pcov = curve_fit(
                        quadratic_function,
                        fit_x[1:] if label != "Attenuation" else fit_x[2:],
                        fit_y[1:] if label != "Attenuation" else fit_y[2:],
                        p0=[0, 0, 0],
                        # bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]) if label=="Amplitude" else ([-np.inf, -1, -np.inf], [np.inf, 0, np.inf]) if label=="Attenuation" else ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    )
                    perr = np.sqrt(np.diag(pcov))
                    fit_y_vals = popt[0] * fit_x**2 + popt[1] * fit_x + popt[2]

                    fig.add_trace(
                        go.Scatter(
                            x=fit_x,
                            y=fit_y_vals,
                            mode="lines",
                            line=dict(color=default[idx % len(default)], dash="dash"),
                            showlegend=False,
                            name=label,
                            legendgroup=2,
                        ),
                        col=2,
                        row=1,
                    )
                    # print(
                    #     f"Config: {config}, Reco: {name}, Parameter: {label}, p0: {popt[0]:.2e}, p1: {popt[1]:.2e}, p2: {popt[2]:.2e}"
                    # )

                    light_map.append(
                        {
                            "Config": config,
                            "Name": name,
                            "Cut": cut,
                            "Parameter": label,
                            "Modulator": modulator,
                            "Charge": fit_x,
                            "PE": fit_y,
                            "FitFunction": quadratic_function,
                            "FitFunctionLabel": f"Quadratic",
                            "FitFunctionFormula": r"$y = a \cdot x^2 + b \cdot x + c$",
                            "Params": popt,
                            "ParamsError": perr,
                            "ParamsLabels": ["A", "B", "C"],
                            "ParamsFormat": [".1e", ".1e", ".2f"],
                            "p0": popt[0],
                            "p1": popt[1],
                            "p2": popt[2],
                        }
                    )

                fig = format_coustom_plotly(
                    fig,
                    tickformat=(None, ".1f"),
                    matches=(None, None),
                    legend_title="Charge (ADC x tick)",
                    title=f"MainOpFlash PE vs Drift - Modulated by {modulator} - {config}",
                )
                # Add x and y axis titles
                fig.update_xaxes(title_text="Drift (us)", row=1, col=1)
                fig.update_xaxes(title_text="Charge (ADC x tick)", row=1, col=2)
                fig.update_yaxes(title_text=f"log{subscript(10)}(PE)", row=1, col=1)
                fig.update_yaxes(title_text="Fit Parameter Value", row=1, col=2)
                save_figure(
                    fig,
                    f"{save_path}",
                    config,
                    name,
                    None,
                    filename=f"MainOpFlashPE_vs_Drift_{cut}_Modulator_{jdx}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

            light_data_df = pd.DataFrame(light_data)
            light_map_df = pd.DataFrame(light_map)

            for light_df, df_name in zip(
                [light_data_df, light_map_df],
                ["Light_Map_Fitting_Data", "Light_Map_Fitting_Parameters"],
            ):
                save_df(
                    light_df,
                    data_path,
                    config,
                    name,
                    filename=df_name,
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

            fig = make_subplots(1, 3)
            # Read the df from data_path as pkl
            for ldx, cut in enumerate(light_map_df["Cut"].unique()):
                if cut == "Center":
                    continue
                cut_df = pd.read_pickle(
                    f"{data_path}/{config}/{name}/{config}_{name}_Light_Map_Fitting_Parameters.pkl"
                )
                cut_df = cut_df[(cut_df["Cut"] == cut)]

                for kdx, reduction in enumerate(cut_df["Modulator"].unique()):
                    this_df = cut_df[cut_df["Modulator"] == reduction]
                    this_run, mask, output = compute_filtered_run(
                        run,
                        configs,
                        params={
                            ("Reco", "Name"): ("equal", name),
                            ("Reco", "TrueMain"): ("equal", True),
                            ("Reco", "MatchedOpFlashPE"): ("bigger", 0),
                        },
                        debug=user_input["debug"],
                    )

                    this_run["Reco"]["MinPECut"] = MinPECut(
                        this_df,
                        this_run["Reco"]["Charge"],
                        this_run["Reco"]["Time"],
                    )

                    if cut == "Lower":
                        # if reduction > 0:
                        #     continue
                        this_run["Reco"]["PECut"] = (
                            this_run["Reco"]["MatchedOpFlashPE"]
                            < this_run["Reco"]["MinPECut"]
                        )
                        print(
                            f"Percentage of events after {cut} with {reduction}: {np.sum(this_run['Reco']['PECut'])/len(this_run['Reco']['PECut'])*100:.2f}%"
                        )
                        event_cut = (
                            np.sum(this_run["Reco"]["PECut"])
                            / len(this_run["Reco"]["PECut"])
                            * 100
                        )

                    else:
                        # if reduction < 0:
                        #     continue
                        this_run["Reco"]["PECut"] = (
                            this_run["Reco"]["MatchedOpFlashPE"]
                            > this_run["Reco"]["MinPECut"]
                        )
                        print(
                            f"Percentage of events after {cut} with {reduction}: {np.sum(this_run['Reco']['PECut'])/len(this_run['Reco']['PECut'])*100:.2f}%"
                        )
                        event_cut = (
                            np.sum(this_run["Reco"]["PECut"])
                            / len(this_run["Reco"]["PECut"])
                            * 100
                        )
                        # Inverse cut to show the inverse effect
                        this_run["Reco"]["PECut"] = ~this_run["Reco"]["PECut"]

                    for (idx, coord), (jdx, cut_idx) in product(
                        enumerate(["X", "Y", "Z"]),
                        enumerate(
                            [
                                # np.ones(len(this_run["Reco"]["PECut"]), dtype=bool),
                                this_run["Reco"]["PECut"],
                            ]
                        ),
                    ):
                        h_total, bins = np.histogram(
                            this_run["Reco"][f"SignalParticle{coord}"],
                            bins=np.arange(
                                info[f"DETECTOR_MIN_{coord}"],
                                info[f"DETECTOR_MAX_{coord}"] + 20,
                                20,
                            ),
                            # density=True,
                        )
                        h, bins = np.histogram(
                            this_run["Reco"][f"SignalParticle{coord}"][cut_idx],
                            bins=np.arange(
                                info[f"DETECTOR_MIN_{coord}"],
                                info[f"DETECTOR_MAX_{coord}"] + 20,
                                20,
                            ),
                            # density=True,
                        )
                        h = h / h_total
                        # rprint(f"Percentage of events cut: {np.sum(cut)/len(cut)*100:.2f}%")
                        bin_centers = 0.5 * (bins[1:] + bins[:-1])
                        fig.add_trace(
                            go.Scatter(
                                x=bin_centers,
                                y=h,
                                mode="lines",
                                line=dict(color=compare[ldx]),
                                line_dash=(
                                    "solid"
                                    if kdx == 0
                                    else (
                                        "dash"
                                        if kdx == 1
                                        else (
                                            "dot"
                                            if kdx == 2
                                            else "dashdot" if kdx == 3 else "longdash"
                                        )
                                    )
                                ),
                                line_shape="hvh",
                                name=f"Modulation {reduction:.2f}: ({event_cut:.1f}%)",
                                legendgroup=ldx,
                                legendgrouptitle=dict(
                                    text=(
                                        f"PE < MaxPE"
                                        if cut == "Upper"
                                        else "PE < MinPE"
                                    )
                                ),
                                showlegend=True if idx == 0 else False,
                            ),
                            row=1,
                            col=idx + 1,
                        )

            fig = format_coustom_plotly(
                fig,
                title=f"PECut Coordinate Scan - {config}",
                legend_title="Cuts",
                log=(False, False),
                matches=(None, None),
                tickformat=(None, ".1f"),
                ranges=(None, (0, 1.1)),
            )
            fig.update_xaxes(title_text=f"Coordinate X (cm)", row=1, col=1)
            fig.update_xaxes(title_text=f"Coordinate Y (cm)", row=1, col=2)
            fig.update_xaxes(title_text=f"Coordinate Z (cm)", row=1, col=3)
            fig.update_yaxes(title_text="Masked Sample Size (%)", row=1, col=1)
            save_figure(
                fig,
                f"{save_path}",
                config,
                name,
                None,
                filename=f"PECut_Coordinate_Scan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

# truth_df = npy2df(run, "Truth", debug=user_input["debug"])

# for config in configs:
#     info, params, output = get_param_dict(
#         f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
#     )
#     for name, (y_label, y_axis_title, y_variables) in product(
#         configs[config],
#         zip(
#             ["Num", "PE"],
#             ["Number of Adj. OpFlashes", "PE of Adj. OpFlashes"],
#             (
#                 ["TotalOpFlashSameGenNum", "TotalOpFlashBkgNum"],
#                 ["TotalOpFlashSameGenPE", "TotalOpFlashBkgPE"],
#             ),
#         ),
#     ):
#         fig = make_subplots(
#             rows=1,
#             cols=2,
#             subplot_titles=("Signal", "Background"),
#         )
#         table_list = []
#         for (jdx, limit), (idx, (label, variable)) in product(
#             enumerate(info["OPFLASH_RADIUS"]),
#             enumerate(
#                 zip(
#                     ["Signal", "Background"],
#                     y_variables,
#                 ),
#             ),
#         ):
#             per_99 = np.percentile(truth_df[f"{variable}Radius{limit}"], 99)
#             hist, bins = np.histogram(
#                 truth_df[f"{variable}Radius{limit}"],
#                 bins=(
#                     np.arange(0, np.max(truth_df[f"{variable}Radius{limit}"]) + 1, 1)
#                     if y_label == "Num"
#                     else np.arange(1.5, per_99, 100)
#                 ),
#             )
#             hist = hist / np.sum(hist)
#             fig.add_trace(
#                 go.Scatter(
#                     x=bins,
#                     y=hist,
#                     mode="lines",
#                     line_shape="hvh",
#                     showlegend=True if idx == 0 else False,
#                     line=dict(color=colors[jdx], width=2),
#                     name=f"{limit}",
#                 ),
#                 row=1,
#                 col=1 + idx,
#             )
#             table_list.append(
#                 {
#                     "Type": label,
#                     "Radius (cm)": f"{limit}",
#                     "Mean": np.mean(truth_df[f"{variable}Radius{limit}"]),
#                     "STD": np.std(truth_df[f"{variable}Radius{limit}"]),
#                 }
#             )
#         fig = format_coustom_plotly(
#             fig,
#             matches=(None, "y"),
#             tickformat=(".0f", ".1s"),
#             legend_title="Radius (cm)",
#             title=f"Radial OpFlashNum (Signal vs Background) - {config}",
#             log=(False, True),
#         )
#         fig.update_xaxes(title_text=y_axis_title)
#         fig.update_yaxes(title_text="Fraction of Events", row=1, col=1)
#         # fig.show()
#         save_figure(
#             fig,
#             save_path,
#             config,
#             name,
#             filename=f"Signal_OpFlash{y_label}_RadialScan",
#             rm=user_input["rewrite"],
#             debug=user_input["debug"],
#         )

#         # display(pd.DataFrame(table_list).set_index("Radius (cm)").T)
#         save_df(
#             pd.DataFrame(table_list),
#             data_path,
#             config,
#             name,
#             filename=f"Signal_OpFlash{y_label}_RadialScan",
#             rm=user_input["rewrite"],
#             debug=user_input["debug"],
#         )

#         fig = make_subplots(
#             rows=1,
#             cols=2,
#             subplot_titles=("Signal", "Background"),
#         )
#         table_list = []
#         plane_names = info["OPFLASH_PLANES"]
#         plane_names[None] = "Total"
#         plane_ids = list(info["OPFLASH_PLANES"].keys())
#         for (jdx, plane), (idx, (label, variable)) in product(
#             enumerate(plane_ids),
#             enumerate(
#                 zip(
#                     ["Signal", "Background"],
#                     y_variables,
#                 ),
#             ),
#         ):
#             if np.percentile(truth_df[f"{variable}"], 99) < 1.5:
#                 print(f"{variable} is empty for {config} - {name}, skipping...")
#                 continue

#             if plane is None:
#                 per_99 = np.percentile(truth_df[f"{variable}"], 99)
#                 hist, bins = np.histogram(
#                     truth_df[f"{variable}"],
#                     bins=(
#                         np.arange(0, np.max(truth_df[f"{variable}"]) + 1, 1)
#                         if y_label == "Num"
#                         else np.arange(1.5, per_99 + 100, 100)
#                     ),
#                 )
#             else:
#                 per_99 = np.percentile(truth_df[f"{variable}Plane{plane}"], 99)
#                 hist, bins = np.histogram(
#                     truth_df[f"{variable}Plane{plane}"],
#                     bins=(
#                         np.arange(0, np.max(truth_df[f"{variable}Plane{plane}"]) + 1, 1)
#                         if y_label == "Num"
#                         else np.arange(1.5, per_99 + 100, 100)
#                     ),
#                 )
#             hist = hist / np.sum(hist)
#             fig.add_trace(
#                 go.Scatter(
#                     x=bins,
#                     y=hist,
#                     mode="lines",
#                     line_shape="hvh",
#                     showlegend=True if idx == 0 else False,
#                     line=dict(color=colors[jdx], width=2),
#                     name=f"{plane_names[plane]}",
#                 ),
#                 row=1,
#                 col=1 + idx,
#             )
#             table_list.append(
#                 {
#                     "Type": label,
#                     "Plane": plane,
#                     "Mean": (
#                         np.mean(truth_df[f"{variable}Plane{plane}"])
#                         if plane is not None
#                         else np.mean(truth_df[f"{variable}"])
#                     ),
#                     "STD": (
#                         np.std(truth_df[f"{variable}Plane{plane}"])
#                         if plane is not None
#                         else np.std(truth_df[f"{variable}"])
#                     ),
#                 }
#             )
#         fig = format_coustom_plotly(
#             fig,
#             matches=(None, "y"),
#             tickformat=(".0f", ".1s"),
#             legend_title="Plane",
#             title=f"Radial OpFlashNum (Signal vs Background) - {config}",
#             log=(False, True),
#         )
#         fig.update_xaxes(title_text="Number of Adj. OpFlashes")
#         fig.update_yaxes(title_text="Fraction of Events", row=1, col=1)
#         # fig.show()
#         save_figure(
#             fig,
#             save_path,
#             config,
#             name,
#             filename=f"Signal_OpFlash{y_label}_PlaneScan",
#             rm=user_input["rewrite"],
#             debug=user_input["debug"],
#         )

#         # display(pd.DataFrame(table_list).set_index("Plane (cm)").T)
#         save_df(
#             pd.DataFrame(table_list),
#             data_path,
#             config,
#             name,
#             filename=f"Signal_OpFlash{y_label}_PlaneScan",
#             rm=user_input["rewrite"],
#             debug=user_input["debug"],
#         )

#         for x_axis_title, x_variable in zip(
#             ("Drift Distance (cm)", "Coordinate Y (cm)", "Coordinate Z (cm)"),
#             ("SignalParticleX", "SignalParticleY", "SignalParticleZ"),
#         ):
#             fig = make_subplots(
#                 rows=1,
#                 cols=2,
#                 subplot_titles=("Signal", "Background"),
#             )

#             for (jdx, limit), (idx, (label, variable)) in product(
#                 enumerate(info["OPFLASH_RADIUS"]),
#                 enumerate(
#                     zip(
#                         ["Signal", "Background"],
#                         y_variables,
#                     )
#                 ),
#             ):
#                 values = []
#                 stds = []
#                 coord = x_variable[-1]
#                 x_axis = np.arange(
#                     info[f"DETECTOR_MIN_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
#                     info[f"DETECTOR_MAX_{coord}"],
#                     params[f"DEFAULT_{coord}_BIN"] / 2,
#                 )
#                 for drift in x_axis:
#                     this_truth_df = truth_df[
#                         (
#                             truth_df[x_variable]
#                             > drift - params[f"DEFAULT_{coord}_BIN"] / 2
#                         )
#                         & (
#                             truth_df[x_variable]
#                             < drift + params[f"DEFAULT_{coord}_BIN"] / 2
#                         )
#                     ]
#                     values.append(np.mean(this_truth_df[f"{variable}Radius{limit}"]))
#                     stds.append(np.std(this_truth_df[f"{variable}Radius{limit}"]))

#                 fig.add_trace(
#                     go.Scatter(
#                         x=x_axis,
#                         y=values,
#                         # error_y = dict(type='data', array=stds, visible=True),
#                         mode="lines",
#                         line_shape="hvh",
#                         showlegend=True if idx == 0 else False,
#                         line=dict(color=colors[jdx], width=2),
#                         name=f"{limit}",
#                     ),
#                     row=1,
#                     col=1 + idx,
#                 )

#             fig = format_coustom_plotly(
#                 fig,
#                 log=(False, True),
#                 tickformat=(None, ".1s"),
#                 legend_title="Radius (cm)",
#                 title=f"OpFlash{y_label} (Signal vs Background) - {config}",
#             )
#             fig.update_xaxes(title_text=x_axis_title)
#             fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
#             # fig.show()
#             save_figure(
#                 fig,
#                 save_path,
#                 config,
#                 name,
#                 filename=f"Signal_OpFlash{y_label}_{coord}Scan_Radius",
#                 rm=user_input["rewrite"],
#                 debug=user_input["debug"],
#             )

#             fig = make_subplots(
#                 rows=1,
#                 cols=2,
#                 subplot_titles=("Signal", "Background"),
#             )

#             for (jdx, plane), (idx, (label, variable)) in product(
#                 enumerate(plane_ids),
#                 enumerate(
#                     zip(
#                         ["Signal", "Background"],
#                         y_variables,
#                     )
#                 ),
#             ):
#                 values = []
#                 stds = []
#                 coord = x_variable[-1]
#                 x_axis = np.arange(
#                     info[f"DETECTOR_MIN_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
#                     info[f"DETECTOR_MAX_{coord}"],
#                     params[f"DEFAULT_{coord}_BIN"] / 2,
#                 )
#                 for drift in x_axis:
#                     this_truth_df = truth_df[
#                         (
#                             truth_df[x_variable]
#                             > drift - params[f"DEFAULT_{coord}_BIN"] / 2
#                         )
#                         & (
#                             truth_df[x_variable]
#                             < drift + params[f"DEFAULT_{coord}_BIN"] / 2
#                         )
#                     ]
#                     if plane is None:
#                         values.append(np.mean(this_truth_df[f"{variable}"]))
#                         stds.append(np.std(this_truth_df[f"{variable}"]))
#                     else:
#                         values.append(np.mean(this_truth_df[f"{variable}Plane{plane}"]))
#                         stds.append(np.std(this_truth_df[f"{variable}Plane{plane}"]))

#                 fig.add_trace(
#                     go.Scatter(
#                         x=x_axis,
#                         y=values,
#                         # error_y = dict(type='data', array=stds, visible=True),
#                         mode="lines",
#                         line_shape="hvh",
#                         showlegend=True if idx == 0 else False,
#                         line=dict(color=colors[jdx], width=2),
#                         name=f"{plane_names[plane]}",
#                     ),
#                     row=1,
#                     col=1 + idx,
#                 )

#             fig = format_coustom_plotly(
#                 fig,
#                 log=(False, True),
#                 tickformat=(None, ".1s"),
#                 legend_title="Plane",
#                 title=f"OpFlash{y_label} (Signal vs Background) - {config}",
#             )
#             fig.update_xaxes(title_text=x_axis_title)
#             fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
#             # fig.show()
#             save_figure(
#                 fig,
#                 save_path,
#                 config,
#                 name,
#                 filename=f"Signal_OpFlash{y_label}_{coord}Scan_Plane",
#                 rm=user_input["rewrite"],
#                 debug=user_input["debug"],
#             )
