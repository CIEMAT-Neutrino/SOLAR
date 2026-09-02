import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/vertex/fiducial"
data_path = f"{root}/output/data/vertex/fiducial"


def fiducial_mask(
    run,
    info,
    reference="Truth",
    fiducial: int = 100,
    inverse: bool = False,
    coordinate: Optional[list[str]] = ["X", "Y", "Z"],
):
    if coordinate is None:
        coordinate = ["X", "Y", "Z"]

    def axis_region(values, coord):
        """Region test on one axis: inside the fiducial box if inverse, else outside."""
        if coord == "X":
            if info["VERSION"] == "hd_1x2x6_lateralAPA":
                return values > fiducial if inverse else values < fiducial

            if info["VERSION"] in ["hd_1x2x6_centralAPA", "hd_1x2x6"]:
                edge = info["DETECTOR_MAX_X"] - fiducial
                return (
                    np.absolute(values) < edge
                    if inverse
                    else np.absolute(values) > edge
                )

            if info["GEOMETRY"] == "vd":
                edge = info["DETECTOR_MAX_X"] - fiducial
                return values < edge if inverse else values > edge

            rprint(
                f"[red]ERROR[/red] Unknown geometry {info['GEOMETRY']} and version {info['VERSION']} for fiducial cut in X"
            )
            return np.ones(len(values), dtype=bool)

        if coord == "Y":
            edge = info["DETECTOR_MAX_Y"] - fiducial
            return np.absolute(values) < edge if inverse else np.absolute(values) > edge

        if coord == "Z":
            low = info["DETECTOR_MIN_Z"] + fiducial
            high = info["DETECTOR_MAX_Z"] - fiducial
            return (
                (values > low) * (values < high)
                if inverse
                else (values > high) + (values < low)
            )

        return np.ones(len(values), dtype=bool)

    def combine(tree, prefix, size):
        """Combine the per-axis region tests of one position variable.

        inverse=True selects the inner fiducial box, so the axes intersect (AND).
        inverse=False selects the outer veto shell, whose combination is the De
        Morgan dual: outside along ANY axis (OR). ANDing three shells would keep
        only the 8 corners of the detector.
        """
        mask = np.ones(size, dtype=bool) if inverse else np.zeros(size, dtype=bool)
        for coord in coordinate:
            this_axis = axis_region(tree[f"{prefix}{coord}"], coord)
            mask = mask * this_axis if inverse else mask + this_axis
        return mask

    n_true = len(run[reference]["SignalParticleX"])
    n_reco = len(run["Reco"]["SignalParticleX"])

    # Each position variable is combined across axes on its own before the reco and
    # true requirements intersect. Combining them per-axis instead would demand that
    # the reco vertex leave the box through the same face as the true vertex.
    true_mask = combine(run[reference], "SignalParticle", n_true)
    reco_mask = combine(run["Reco"], "Reco", n_reco) * combine(
        run["Reco"], "SignalParticle", n_reco
    )

    return true_mask, reco_mask


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
    "--drop_default_recox",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Reject clusters whose RecoX carries the flash-matching failure default. "
        "compute_main_variables replaces the upstream -1e6 sentinel with "
        "+/-DETECTOR_MAX_X (sign taken from the true SignalParticleX), which parks "
        "every failed match inside the outer |X| > MAX_X - fiducial shell and "
        "credits it as a success there. Local to this script."
    ),
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
    params={("Reco", "TrueMain"): ("equal", True)},
    presets=["ANALYSIS"],
    signal = "marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

# Redefine the np.ndarray lowe_energy_centers to have an extra None value at the beginning by using np.insert
lowe_energy_centers = np.append(
    np.array([None]),
    lowe_energy_centers,
)

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    fig = make_subplots(rows=1, cols=1)
    for name in configs[config]:
        fiducial_list = []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        # Clusters whose flash match failed reach this script with RecoX already
        # overwritten by compute_main_variables: the upstream -1e6 sentinel becomes
        # +/-DETECTOR_MAX_X, with the sign read off the true SignalParticleX. Those
        # land at |RecoX| = MAX_X, i.e. inside every outer shell the inverse=False
        # branch scans, so each failed match is scored as a containment success and
        # the shell efficiency is inflated by the whole failure rate. The same
        # comparison also catches unphysical drift overruns.
        valid_recox = np.ones(len(run["Reco"]["RecoX"]), dtype=bool)
        if args.drop_default_recox:
            # Match the two sentinel constants exactly. A wider |RecoX| >= MAX_X test
            # would also discard clusters that merely overrun the nominal half-width,
            # which is a large, legitimate population wherever RecoX does not share
            # the config's X convention.
            _recox = run["Reco"]["RecoX"]
            valid_recox = (_recox != info["DETECTOR_MAX_X"]) * (
                _recox != info["DETECTOR_MIN_X"]
            )
            rprint(
                f"[cyan][INFO][/cyan] Default RecoX rejected: "
                f"{100 * (1 - np.mean(valid_recox)):.2f}% "
                f"(exactly {info['DETECTOR_MIN_X']} or {info['DETECTOR_MAX_X']} cm) | "
                f"for reference, |RecoX| > MAX_X holds for "
                f"{100 * np.mean(np.absolute(_recox) > info['DETECTOR_MAX_X']):.2f}% "
                f"and |RecoX| == MAX_X for "
                f"{100 * np.mean(np.absolute(_recox) == info['DETECTOR_MAX_X']):.2f}%"
            )

        for reference, inverse, coord, energy in product(
            ["Reco", "Truth"], [True, False], ["X", "Y", "Z", None], lowe_energy_centers
        ):
            counts = []
            counts_error = []
            efficiency = []
            efficiency_error = []
            for fiducial in np.arange(0, 220, 20):
                true_mask, reco_mask = fiducial_mask(
                    run, info, reference, fiducial, inverse,
                    [coord] if coord is not None else ["X", "Y", "Z"],
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

                this_true_mask = true_mask * (
                    (run[reference]["Geometry"] == info["GEOMETRY"])
                    * (run[reference]["Version"] == info["VERSION"])
                    * (run[reference]["Name"] == name)
                    * true_energy_mask
                )
                this_reco_mask = reco_mask * (
                    (run["Reco"]["Geometry"] == info["GEOMETRY"])
                    * (run["Reco"]["Version"] == info["VERSION"])
                    * (run["Reco"]["Name"] == name)
                    * reco_energy_mask
                )

                # A default RecoX never counts as a reconstructed vertex. With
                # reference="Reco" the denominator is drawn from the same tree, so the
                # cluster leaves both terms and the ratio becomes a containment
                # efficiency conditional on a successful flash match. With
                # reference="Truth" the denominator is the truth tree and keeps it, so
                # the failure stays in the sample and the ratio is the absolute
                # efficiency including flash-matching loss.
                this_reco_mask = this_reco_mask * valid_recox
                if reference == "Reco":
                    this_true_mask = this_true_mask * valid_recox

                counts.append(sum(this_reco_mask))
                counts_error.append(np.sqrt(sum(this_reco_mask)))
                efficiency.append(
                    100 * sum(this_reco_mask) / sum(this_true_mask)
                    if 100 * sum(this_reco_mask) / sum(this_true_mask) < 100
                    else 100
                )
                efficiency_error.append(
                    100 * np.sqrt(sum(this_reco_mask)) / sum(this_true_mask)
                    if sum(this_true_mask) > 0
                    else 0
                )

            fiducial_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": coord,
                    "Energy": energy,
                    "Values": np.arange(0, 220, 20),
                    "Counts": counts,
                    "CountsError": counts_error,
                    "Efficiency": efficiency,
                    "EfficiencyError": efficiency_error,
                    "Inverse": inverse,
                    "Reference": reference,
                }
            )

    df_fiducial = pd.DataFrame(fiducial_list)
    df_fiducial = df_fiducial.fillna(np.nan)
    for inverse, reference in product([True, False], ["Truth"]):
        this_fiducial_df = df_fiducial[
            (df_fiducial["Inverse"] == inverse)
            * (df_fiducial["Reference"] == reference)
        ]
        fig = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].isna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=True,
            line_shape="hvh",
            # error_y="EfficiencyError",
            color="Variable",
            labels={
                "Values": "Fiducial Cut (cm)",
                "Efficiency": (
                    "Fiducialization Efficiency (%)"
                    if not inverse
                    else "Reconstruction Efficiency (%)"
                ),
            },
            color_discrete_sequence=default,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")

        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=(
                f"Fiducialization Efficiency - {config}"
                if not inverse
                else f"Vertex Reconstruction Efficiency vs Fiducial Cut - {config}"
            ),
            legend_title="Variable",
            legend=dict(y=0.06, x=0.82),
        )

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=(
                f"Vertex_Fiducial_Efficiency_{reference}"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_{reference}"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    for inverse, reference in product([True, False], ["Truth", "Reco"]):
        this_fiducial_df = df_fiducial[
            (df_fiducial["Inverse"] == inverse)
            * (df_fiducial["Reference"] == reference)
        ]
        fig = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].notna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=True,
            line_shape="hvh",
            # error_y="EfficiencyError",
            color="Energy",
            facet_col="Variable",
            color_discrete_sequence=colors,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")

        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=(
                f"Fiducialization Efficiency vs Energy - {config}"
                if not inverse
                else f"Vertex Reconstruction Efficiency vs Fiducial Cut and Energy - {config}"
            ),
            legend_title="Energy (MeV)",
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(
            title_text=(
                "Fiducialization Efficiency (%)"
                if not inverse
                else "Reconstruction Efficiency (%)"
            ),
            row=1,
            col=1,
        )

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
            filename=(
                f"Vertex_Fiducial_Efficiency_Energy_{reference}"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_Energy_{reference}"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    for this_df, df_filename, df_type in zip(
        [df_fiducial],
        [
            "Fiducial_Efficiency",
        ],
        ["pkl"],
    ):
        save_df(
            this_df,
            data_path,
            config,
            name,
            filename=df_filename,
            rm=user_input["rewrite"],
            filetype=df_type,
            debug=user_input["debug"],
        )
