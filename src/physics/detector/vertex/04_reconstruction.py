import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


def double_gaussian(x, a_core, mu, sigma_core, a_tail, sigma_tail):
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


save_path = f"{root}/output/images/vertex/reconstruction"
data_path = f"{root}/output/data/vertex/reconstruction"


def position_mask(
    run,
    info,
    reference = 'Reco',
    energy: Optional[float] = 10,
    tolerances: Optional[list[int]] = [1],
    position: float = 0,
    coordinate_bin: Optional[float] = 10,
    coordinate: Optional[str] = "X",
    combined_scan_coordinate: Optional[str] = "X",
    sigma: bool = True,
    radius_thresholds: Optional[dict] = None,
):
    def _energy_sigma(coord_label: str) -> float:
        _emask = df["Energy"] == energy if energy is not None else df["Energy"].isna()
        sigma_row = df[
            (df["Coordinate"] == coord_label)
            & _emask
            & df["CoordinateBin"].isna()
        ]["Sigma"].values
        return float(sigma_row[0]) if len(sigma_row) > 0 else np.nan

    if tolerances is None:
        tolerances = [1]

    half_bin = coordinate_bin / 2 if coordinate_bin is not None else 0.0

    true_mask_dict = {}
    reco_mask_dict = {}
    for tolerance in tolerances:
        true_mask = np.ones(len(run[reference]["SignalParticleX"]), dtype=bool)
        reco_mask = np.ones(len(run["Reco"]["SignalParticleX"]), dtype=bool)
        scan_coordinate = coordinate if coordinate is not None else combined_scan_coordinate

        if scan_coordinate is None:
            # Combined 3D efficiency with no scan axis selected.
            pass
        elif scan_coordinate == "X" and info["GEOMETRY"] == "hd":
            true_mask = true_mask * (
                (
                    np.absolute(run[reference][f"SignalParticle{scan_coordinate}"])
                    > position - half_bin
                )
                * (
                    np.absolute(run[reference][f"SignalParticle{scan_coordinate}"])
                    <= position + half_bin
                )
            )
            reco_mask = reco_mask * (
                (
                    np.absolute(run["Reco"][f"SignalParticle{scan_coordinate}"])
                    < position + half_bin
                )
                * (
                    np.absolute(run["Reco"][f"SignalParticle{scan_coordinate}"])
                    >= position - half_bin
                )
            )
        else:
            true_mask = true_mask * (
                (
                    run[reference][f"SignalParticle{scan_coordinate}"]
                    > position - half_bin
                )
                * (
                    run[reference][f"SignalParticle{scan_coordinate}"]
                    <= position + half_bin
                )
            )
            reco_mask = reco_mask * (
                (
                    run["Reco"][f"SignalParticle{scan_coordinate}"]
                    < position + half_bin
                )
                * (
                    run["Reco"][f"SignalParticle{scan_coordinate}"]
                    >= position - half_bin
                )
            )

        if sigma:
            if coordinate is None:
                reco_error = np.sqrt(
                    run["Reco"]["ErrorX"] ** 2
                    + run["Reco"]["ErrorY"] ** 2
                    + run["Reco"]["ErrorZ"] ** 2
                )
                if radius_thresholds is not None:
                    reco_mask = reco_mask * (reco_error < radius_thresholds[tolerance])
                else:
                    sigma_axes = np.asarray([_energy_sigma(axis) for axis in ["X", "Y", "Z"]])
                    has_valid_sigma = np.all(np.isfinite(sigma_axes))
                    sigma_3d = np.sqrt(np.sum(sigma_axes**2)) if has_valid_sigma else np.nan
                    reco_mask = reco_mask * (
                        reco_error < tolerance * sigma_3d if has_valid_sigma else False
                    )
            else:
                axis_sigma = _energy_sigma(f"{coordinate}")
                reco_mask = reco_mask * (
                    np.absolute(run["Reco"][f"Error{coordinate}"])
                    < tolerance * axis_sigma if np.isfinite(axis_sigma) else False
                )
        else:
            if coordinate is None:
                reco_error = np.sqrt(
                    run["Reco"]["ErrorX"] ** 2
                    + run["Reco"]["ErrorY"] ** 2
                    + run["Reco"]["ErrorZ"] ** 2
                )
                reco_mask = reco_mask * (reco_error < tolerance)
            else:
                reco_mask = reco_mask * (
                    np.absolute(run["Reco"][f"Error{coordinate}"]) < tolerance
                )

        true_mask_dict[tolerance] = true_mask
        reco_mask_dict[tolerance] = reco_mask

    return true_mask_dict, reco_mask_dict


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
parser.add_argument(
    "--combined-scan-coordinate",
    type=str,
    default="X",
    choices=["X", "Y", "Z", "NONE"],
    help="Scan axis for combined 3D reconstruction efficiency (X, Y, Z, or NONE)",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--purity_threshold", type=float, default=0.5,
    help="MatchedOpFlashPur threshold for the high-purity 3D reference sample (default: 0.5)",
)

args = parser.parse_args()
config = args.config
name = args.name
combined_scan_coordinate = (
    None
    if args.combined_scan_coordinate.upper() == "NONE"
    else args.combined_scan_coordinate.upper()
)
purity_threshold = args.purity_threshold

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
    params={("Reco", "TrueMain"): ("equal", True)},
    presets=["ANALYSIS"],
    signal = "marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

# Extend to match resolution scan range (02_vertex.py uses np.arange(6, 31, 2))
lowe_energy_centers = np.arange(6, 31, 2)
lowe_energy_centers = np.append(np.array([None]), lowe_energy_centers)

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    fig = make_subplots(rows=1, cols=1)
    position_list = []
    analysis_info = load_analysis_info(str(root))
    for name in configs[config]:
        hist_list = []
        missing_energies = []
        missing_sigma_warnings = set()
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))

        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        df = pickle.load(
        open(
            f"{root}/output/data/vertex/resolution/{config}/{name}/{config}_{name}_Resolution.pkl",
            "rb",
            )
        )
        print( df.head())
        df = df.fillna(np.nan)

        # Pre-compute 3D quantile thresholds from high-purity matched sample.
        # threshold(n) = quantile(r_hp, erf(n/√2)) — e.g. n=3 → 99.73% coverage.
        from scipy.special import erf as _erf
        _reco = run["Reco"]
        _purity_base = (
            (_reco["Geometry"] == info["GEOMETRY"])
            & (_reco["Version"] == info["VERSION"])
            & (_reco["Name"] == name)
            & (_reco["MatchedOpFlashPur"] > purity_threshold)
        )
        _3d_error_all = np.sqrt(
            _reco["ErrorX"] ** 2 + _reco["ErrorY"] ** 2 + _reco["ErrorZ"] ** 2
        )
        _3d_quantile_thresholds = {}
        for _eq in lowe_energy_centers:
            if _eq is None:
                _qmask = _purity_base
            else:
                _qmask = _purity_base & (
                    (_reco["SignalParticleK"] >= _eq - lowe_ebin / 2)
                    & (_reco["SignalParticleK"] < _eq + lowe_ebin / 2)
                )
            _r_hp = _3d_error_all[_qmask]
            if len(_r_hp) < 10:
                _3d_quantile_thresholds[_eq] = None
                continue
            _3d_quantile_thresholds[_eq] = {
                s: float(np.quantile(_r_hp, _erf(s / np.sqrt(2))))
                for s in analysis_info["VERTEX_RESOLUTION_SIGMAS"]
            }
        rprint(
            f"[cyan][3D quantile thresholds at energy=None]: "
            f"{_3d_quantile_thresholds.get(None)}[/cyan]"
        )

        for energy, coord in product(lowe_energy_centers, ["X", "Y", "Z", None]):
            if energy not in df["Energy"].values and energy is not None:
                missing_energies.append(energy)
                continue
            
            counts, counts_error = {True: {}, False: {}}, {True: {}, False: {}}
            efficiency, efficiency_error = {True: {}, False: {}}, {True: {}, False: {}}

            for sigma, tolerance in zip(
                (
                    [True] * len(analysis_info["VERTEX_RESOLUTION_SIGMAS"])
                    + [False] * len(analysis_info["VERTEX_RESOLUTION_TOLERANCES"])
                ),
                (
                    analysis_info["VERTEX_RESOLUTION_SIGMAS"]
                    + analysis_info["VERTEX_RESOLUTION_TOLERANCES"]
                ),
            ):
                counts[sigma][tolerance] = []
                counts_error[sigma][tolerance] = []
                efficiency[sigma][tolerance] = []
                efficiency_error[sigma][tolerance] = []

            scan_coordinate = coord if coord is not None else combined_scan_coordinate

            if scan_coordinate is None:
                coordinate_array = np.array([0])
            elif scan_coordinate == "X" and info["GEOMETRY"] == "hd":
                coordinate_array = np.arange(
                    0,
                    info[f"DETECTOR_MAX_{scan_coordinate}"]
                    + info[f"DETECTOR_GAP_{scan_coordinate}"]
                    + params[f"DEFAULT_{scan_coordinate}_BIN"],
                    params[f"DEFAULT_{scan_coordinate}_BIN"],
                )

            else:
                coordinate_array = np.arange(
                    info[f"DETECTOR_MIN_{scan_coordinate}"] - info[f"DETECTOR_GAP_{scan_coordinate}"],
                    info[f"DETECTOR_MAX_{scan_coordinate}"]
                    + info[f"DETECTOR_GAP_{scan_coordinate}"]
                    + params[f"DEFAULT_{scan_coordinate}_BIN"],
                    params[f"DEFAULT_{scan_coordinate}_BIN"],
                )

            true_mask, reco_mask = {}, {}
            reference = "Reco"
            for position, (tolerances, sigma), reference in product(
                coordinate_array,
                zip(
                    [
                        analysis_info["VERTEX_RESOLUTION_SIGMAS"],
                        analysis_info["VERTEX_RESOLUTION_TOLERANCES"],
                    ],
                    [True, False],
                ),
                ['Reco']
            ):
                if sigma:
                    _emask = df["Energy"] == energy if energy is not None else df["Energy"].isna()
                    _missing_sigma = False
                    if coord is None:
                        if _3d_quantile_thresholds.get(energy) is None:
                            warning_key = (energy, coord)
                            if warning_key not in missing_sigma_warnings:
                                output += (
                                    f"[yellow][WARNING][/yellow] Insufficient high-purity events for 3D quantile threshold at energy={energy}. "
                                    "Padding with NaN.\n"
                                )
                                missing_sigma_warnings.add(warning_key)
                            _missing_sigma = True
                    else:
                        axis_sigma_vals = df[
                            (df["Coordinate"] == coord)
                            & _emask
                            & df["CoordinateBin"].isna()
                        ]["Sigma"].values
                        if len(axis_sigma_vals) == 0:
                            warning_key = (energy, coord)
                            if warning_key not in missing_sigma_warnings:
                                output += (
                                    f"[yellow][WARNING][/yellow] Missing sigma entry for coordinate={coord} at energy={energy}. "
                                    "Padding with NaN.\n"
                                )
                                missing_sigma_warnings.add(warning_key)
                            _missing_sigma = True
                    if _missing_sigma:
                        for _tol in tolerances:
                            counts[sigma][_tol].append(0)
                            counts_error[sigma][_tol].append(0)
                            efficiency[sigma][_tol].append(np.nan)
                            efficiency_error[sigma][_tol].append(np.nan)
                        continue
                
                true_mask[sigma], reco_mask[sigma] = position_mask(
                    run=run,
                    info=info,
                    reference=reference,
                    energy=energy,
                    tolerances=tolerances,
                    position=position,
                    coordinate_bin=(
                        params[f"DEFAULT_{scan_coordinate}_BIN"]
                        if scan_coordinate is not None
                        else None
                    ),
                    coordinate=coord,
                    combined_scan_coordinate=combined_scan_coordinate,
                    sigma=sigma,
                    radius_thresholds=(
                        _3d_quantile_thresholds.get(energy)
                        if coord is None and sigma
                        else None
                    ),
                )
                if energy is None:
                    true_energy_mask = np.ones(
                        len(run[reference]["SignalParticleK"]), dtype=bool
                    )
                    reco_energy_mask = np.ones(
                        len(run["Reco"]["SignalParticleK"]), dtype=bool
                    )
                else:
                    true_energy_mask = (
                        run[reference]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run[reference]["SignalParticleK"] < energy + lowe_ebin / 2)
                    reco_energy_mask = (
                        run["Reco"]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run["Reco"]["SignalParticleK"] < energy + lowe_ebin / 2)

                for tolerance in tolerances:
                    this_true_mask = true_mask[sigma][tolerance] * (
                        (run[reference]["Geometry"] == info["GEOMETRY"])
                        * (run[reference]["Version"] == info["VERSION"])
                        * (run[reference]["Name"] == name)
                        * true_energy_mask
                    )
                    this_reco_mask = reco_mask[sigma][tolerance] * (
                        (run["Reco"]["Geometry"] == info["GEOMETRY"])
                        * (run["Reco"]["Version"] == info["VERSION"])
                        * (run["Reco"]["Name"] == name)
                        * reco_energy_mask
                    )
                    counts[sigma][tolerance].append(sum(this_reco_mask))
                    counts_error[sigma][tolerance].append(np.sqrt(sum(this_reco_mask)))
                    efficiency[sigma][tolerance].append(
                        100
                        * (
                            sum(this_reco_mask) / sum(this_true_mask)
                            if sum(this_true_mask) > 0
                            else 0
                        )
                    )
                    efficiency_error[sigma][tolerance].append(
                        100
                        * (
                            np.sqrt(sum(this_reco_mask)) / sum(this_true_mask)
                            if sum(this_true_mask) > 0
                            else 0
                        )
                    )

            for sigma, tolerance in zip(
                (
                    [True] * len(analysis_info["VERTEX_RESOLUTION_SIGMAS"])
                    + [False] * len(analysis_info["VERTEX_RESOLUTION_TOLERANCES"])
                ),
                (
                    analysis_info["VERTEX_RESOLUTION_SIGMAS"]
                    + analysis_info["VERTEX_RESOLUTION_TOLERANCES"]
                ),
            ):
                if energy in missing_energies:
                    continue
                values_axis = scan_coordinate
                position_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Variable": coord,
                        "Energy": energy,
                        "Tolerance": tolerance,
                        "Reference": reference,
                        "Values": (
                            2 * info[f"DETECTOR_MAX_{values_axis}"] - coordinate_array
                            if (
                                values_axis is not None
                                and config == "hd_1x2x6_lateralAPA"
                                and values_axis == "X"
                            )
                            else (
                                coordinate_array + info[f"DETECTOR_MAX_{values_axis}"]
                                if (
                                    values_axis is not None
                                    and info["GEOMETRY"] == "vd"
                                    and values_axis == "X"
                                )
                                else coordinate_array
                            )
                        ),
                        "Counts": counts[sigma][tolerance],
                        "CountsError": counts_error[sigma][tolerance],
                        "Efficiency": efficiency[sigma][tolerance],
                        "EfficiencyError": efficiency_error[sigma][tolerance],
                        "Sigma": sigma,
                    }
                )

    df_position = pd.DataFrame(position_list)

    # Keep Variable=None in saved outputs, but use an explicit label for plotting/faceting.
    df_position_plot = df_position.copy()
    df_position_plot["VariablePlot"] = df_position_plot["Variable"].apply(
        lambda value: "XYZ (3D)" if value is None else value
    )

    for sigma in [True, False]:
        fig = px.line(
            df_position_plot[
                (df_position_plot["Energy"].isna())
                * (df_position_plot["Sigma"] == sigma)
            ].explode(["Values", "Efficiency", "EfficiencyError"]),
            x="Values",
            y="Efficiency",
            error_y="EfficiencyError",
            markers=True,
            line_shape="spline",
            facet_col="VariablePlot",
            color="Tolerance",
            color_discrete_sequence=default,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=f"Vertex Reconstruction Efficiency - {config}",
            legend_title="Sigmas" if sigma else "Tolerance (cm)",
            matches=(None, None),
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(title_text="Reconstruction Efficiency (%)", row=1, col=1)

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Reconstruction_Efficiency_{'Sigma' if sigma else 'Tolerance'}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    _plot_energies = [6, 10, 14, 18, 22, 26, 30]
    for tolerance in analysis_info["VERTEX_RESOLUTION_SIGMAS"]:
        fig = px.line(
            df_position_plot[
                (df_position_plot["Energy"].isin(_plot_energies))
                & (df_position_plot["Sigma"] == True)
                & (df_position_plot["Tolerance"] == tolerance)
            ].explode(["Values", "Efficiency", "EfficiencyError"]),
            x="Values",
            y="Efficiency",
            error_y="EfficiencyError",
            markers=True,
            line_shape="spline",
            facet_col="VariablePlot",
            color="Energy",
            color_discrete_sequence=colors,
        )
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=f"Reconstruction Efficiency Sigma{tolerance} - {config}",
            legend_title="Energy (MeV)",
            matches=(None, None),
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(title_text="Vertex Reconstruction Efficiency", row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        fig.update_layout(
            legend=dict(
                traceorder="normal",
                itemsizing="constant",
            )
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Reconstruction_Efficiency_Energy_Sigma{tolerance}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    # Append efficiency-vs-energy summary rows into the same dataframe.
    # Variable = "EnergyX" / "EnergyY" / "EnergyZ" / "EnergyXYZ"; Energy = None;
    # Values = array of energy bin centers; Efficiency/EfficiencyError = aligned arrays.
    all_energies = sorted(df_position["Energy"].dropna().unique())
    _summary_accum = {}
    for energy in all_energies:
        for coord in [None]:
            this_df = df_position[
                (df_position["Energy"] == energy)
                & df_position["Variable"].isna()
                & (df_position["Sigma"] == True)
            ]
            if len(this_df) == 0:
                continue
            coord_label = "Energy"
            for tolerance in this_df["Tolerance"].unique():
                tol_df = this_df[this_df["Tolerance"] == tolerance]
                efficiency_values = np.asarray(
                    [val for row in tol_df["Efficiency"].values for val in np.asarray(row)
                     if np.isfinite(val)]
                )
                n = len(efficiency_values)
                key = (coord_label, tolerance)
                if key not in _summary_accum:
                    _summary_accum[key] = {"Energy": [], "Efficiency": [], "EfficiencyError": []}
                _summary_accum[key]["Energy"].append(energy)
                _summary_accum[key]["Efficiency"].append(
                    np.mean(efficiency_values) if n > 0 else np.nan
                )
                _summary_accum[key]["EfficiencyError"].append(
                    np.std(efficiency_values) / np.sqrt(n) if n > 0 else np.nan
                )

    _nan_arr = lambda n: np.full(n, np.nan)
    summary_rows = [
        {
            "Geometry": info["GEOMETRY"],
            "Config": config,
            "Name": name,
            "Variable": coord_label,
            "Energy": None,
            "Tolerance": tolerance,
            "Reference": "Reco",
            "Values": np.asarray(arrays["Energy"], dtype=float),
            "Counts": _nan_arr(len(arrays["Energy"])),
            "CountsError": _nan_arr(len(arrays["Energy"])),
            "Efficiency": np.asarray(arrays["Efficiency"]),
            "EfficiencyError": np.asarray(arrays["EfficiencyError"]),
            "Sigma": True,
        }
        for (coord_label, tolerance), arrays in _summary_accum.items()
    ]
    df_position = pd.concat(
        [df_position, pd.DataFrame(summary_rows)], ignore_index=True
    )

    save_df(
        df_position,
        data_path,
        config,
        name,
        filename="Vertex_Reconstruction_Efficiency",
        rm=user_input["rewrite"],
        filetype="pkl",
        debug=user_input["debug"],
    )
