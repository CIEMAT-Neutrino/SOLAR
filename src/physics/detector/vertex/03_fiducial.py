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
    split: bool = False,
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
    reco_pos_mask = combine(run["Reco"], "Reco", n_reco)
    reco_true_mask = combine(run["Reco"], "SignalParticle", n_reco)
    reco_mask = reco_pos_mask * reco_true_mask

    if split:
        # Containment purity needs the reconstructed and true positions tested
        # separately: its denominator is the reco-selected sample alone, whereas
        # reco_mask above already requires both to land in the region.
        return true_mask, reco_mask, reco_pos_mask, reco_true_mask
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
    "--require_true_in_detector",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Require the true vertex to lie inside the DETECTOR_MIN/MAX volume declared "
        "by the config. Productions that generate in a buffer beyond the active "
        "volume otherwise leave unreconstructable vertices in the denominator, "
        "capping the shell efficiencies. Local to this script."
    ),
)
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
        purity_list = []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        # Only the drift coordinate depends on the optical match: RecoX is the cluster
        # time referred to the matched flash t0, while RecoY/RecoZ are wire positions
        # and stand independently of it. So this mask gates X and the combined XYZ
        # scan only -- applying it to a standalone Y or Z would report an X flash
        # failure as a Y/Z reconstruction failure.
        valid_x = np.ones(len(run["Reco"]["RecoX"]), dtype=bool)
        if args.drop_default_recox:
            # MatchedOpFlashPlane names the PDS plane that supplied the match, not its
            # quality. HD only ever yields plane -1 (no flash) or 0, so the
            # QUALITY_CUTS.OPFLASH_PLANE == 0 test used by lib/fiducial.py and
            # signal/01_fiducialize.py costs it nothing. VD spreads matches over planes
            # 0-4 and planes 1-4 reconstruct X as well as plane 0 does (>94% within
            # 10 cm), so that same test discards ~22% of sound VD clusters. PE > 0 is
            # the geometry-independent equivalent: a failed match always carries
            # plane == -1 and PE == 0, so this excludes exactly the no-flash case.
            flash_ok = run["Reco"]["MatchedOpFlashPE"] > 0

            # compute_main_variables rewrites the upstream -1e6 RecoX sentinel to
            # +/-DETECTOR_MAX_X, taking the sign from the TRUE SignalParticleX. That
            # parks every failed match at |RecoX| = MAX_X, inside every outer shell
            # the inverse=False branch scans, so it scores as a containment success
            # and inflates the shell efficiency by the whole failure rate. Compare
            # against the two constants exactly: a wider |RecoX| >= MAX_X test also
            # discards clusters that merely overrun the nominal half-width.
            recox_ok = (run["Reco"]["RecoX"] != info["DETECTOR_MAX_X"]) * (
                run["Reco"]["RecoX"] != info["DETECTOR_MIN_X"]
            )

            valid_x = flash_ok * recox_ok
            rprint(
                f"[cyan][INFO][/cyan] X-only flash gate rejects "
                f"{100 * (1 - np.mean(valid_x)):.2f}% of clusters "
                f"(no valid flash: {100 * (1 - np.mean(flash_ok)):.2f}%, "
                f"default RecoX: {100 * (1 - np.mean(recox_ok)):.2f}%). "
                f"Y and Z are not gated."
            )

        # A cluster whose flash match failed still lands in the outer shell by chance
        # at the rate the shell occupies the X range (27.8% of HD at fiducial=100), and
        # is scored as a containment success there. That accidental term floors the
        # efficiency at the shell's geometric size no matter how badly the match went.
        # MatchedOpFlashPur > 0 -- the criterion the flash-matching efficiency curves
        # themselves use -- selects exactly the population that is not accidental, so
        # the gated rows drop the floor and read the match efficiency directly.
        #
        # The FlashMatch column records which rows carry that requirement:
        #   "None" -> no purity requirement anywhere.
        #   "X"    -> purity on X and the combined scan only. Y and Z come from wire
        #             positions and do not depend on the match, so they stay ungated
        #             and their "X" rows repeat their "None" rows.
        #   "All"  -> purity on every coordinate, so Y and Z are also conditioned on a
        #             successfully matched event. Use this for a table that treats all
        #             three axes alike; "X" mixes a gated X with ungated Y/Z.
        # These are the strings "None"/"X"/"All", not Python None, so that the column
        # stays plain text and filters as == "None" rather than needing .isna().
        flash_matched_ok = run["Reco"]["MatchedOpFlashPur"] > 0

        # Some productions generate vertices in a buffer beyond the active volume the
        # config declares (the vd _nominal sample overruns Y and Z by ~57 cm and X by
        # ~45 cm, ~5-8% of truth entries per axis). Those vertices sit outside the
        # detector, so no cluster can ever be reconstructed at them, yet the shells are
        # built from the config bounds and still count them in the denominator. For the
        # Y shell that puts ~36% of the denominator outside the detector and caps the
        # efficiency near 64% however good the reconstruction is. Requiring the true
        # vertex to lie inside the declared volume restores a fiducializable denominator.
        def in_detector(tree):
            inside = np.ones(len(tree["SignalParticleX"]), dtype=bool)
            for axis in ["X", "Y", "Z"]:
                values = tree[f"SignalParticle{axis}"]
                inside = (
                    inside
                    * (values >= info[f"DETECTOR_MIN_{axis}"])
                    * (values <= info[f"DETECTOR_MAX_{axis}"])
                )
            return inside

        true_in_detector = {tree: in_detector(run[tree]) for tree in ["Truth", "Reco"]}
        if args.require_true_in_detector:
            rprint(
                "[cyan][INFO][/cyan] Truth vertices outside the declared volume: "
                f"Truth {100 * (1 - np.mean(true_in_detector['Truth'])):.2f}%, "
                f"Reco {100 * (1 - np.mean(true_in_detector['Reco'])):.2f}% (rejected)"
            )

        for reference, inverse, coord, energy, flash_matched in product(
            ["Reco", "Truth"],
            [True, False],
            ["X", "Y", "Z", None],
            lowe_energy_centers,
            ["None", "X", "All"],
        ):
            counts = []
            counts_error = []
            efficiency = []
            efficiency_error = []
            purity, purity_error = [], []
            in_migration, in_migration_error = [], []
            reco_selected = []
            for fiducial in np.arange(0, 220, 20):
                true_mask, reco_mask, reco_pos_mask, reco_true_mask = fiducial_mask(
                    run, info, reference, fiducial, inverse,
                    [coord] if coord is not None else ["X", "Y", "Z"],
                    split=True,
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

                if args.require_true_in_detector:
                    this_true_mask = this_true_mask * true_in_detector[reference]
                    this_reco_mask = this_reco_mask * true_in_detector["Reco"]

                # Gate X and the combined XYZ scan on a usable drift coordinate, and
                # leave standalone Y/Z alone. With reference="Truth" the denominator
                # comes from the truth tree and keeps the failures, so the ratio is the
                # absolute efficiency and X tracks the flash-matching performance. With
                # reference="Reco" both terms are drawn from the same tree and the
                # cluster leaves both, giving containment conditional on a good match.
                drift_coord = coord in ["X", None]
                apply_purity = flash_matched == "All" or (
                    flash_matched == "X" and drift_coord
                )
                if drift_coord or apply_purity:
                    this_gate = valid_x if drift_coord else np.ones(len(valid_x), bool)
                    if apply_purity:
                        this_gate = this_gate * flash_matched_ok
                    this_reco_mask = this_reco_mask * this_gate
                    if reference == "Reco":
                        this_true_mask = this_true_mask * this_gate

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

                # Containment purity of the reconstructed sample. Unlike the
                # efficiency above, whose denominator is the true population, this
                # normalises to what the reconstruction actually selects:
                #   P_FV  = N(true in region & reco in region) / N(reco in region)
                #   f_in  = N(true outside     & reco in region) / N(reco in region)
                #         = 1 - P_FV
                # f_in is the fraction of the reconstructed fiducial sample that
                # migrated in from outside. Both terms are drawn from the Reco tree,
                # so this is independent of the Truth/Reco reference split.
                _sel = reco_pos_mask * (
                    (run["Reco"]["Geometry"] == info["GEOMETRY"])
                    * (run["Reco"]["Version"] == info["VERSION"])
                    * (run["Reco"]["Name"] == name)
                    * reco_energy_mask
                )
                if args.require_true_in_detector:
                    _sel = _sel * true_in_detector["Reco"]
                if drift_coord or apply_purity:
                    _sel = _sel * this_gate
                _n_sel = sum(_sel)
                _n_contained = sum(_sel * reco_true_mask)
                reco_selected.append(_n_sel)
                if _n_sel > 0:
                    _p = _n_contained / _n_sel
                    # Binomial: the numerator is a subset of the denominator, so the
                    # Poisson form used for the efficiency above would overstate it.
                    _err = 100 * np.sqrt(max(_p * (1 - _p), 0.0) / _n_sel)
                    purity.append(100 * _p)
                    in_migration.append(100 * (1 - _p))
                else:
                    _err = 0.0
                    purity.append(np.nan)
                    in_migration.append(np.nan)
                purity_error.append(_err)
                in_migration_error.append(_err)

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
                    "FlashMatch": flash_matched,
                }
            )

            # Purity is built entirely from the Reco tree, so it does not vary with
            # the Truth/Reco reference. Emit it once to keep the pkl unambiguous.
            if reference == "Reco":
                purity_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Variable": coord,
                        "Energy": energy,
                        "Values": np.arange(0, 220, 20),
                        "Counts": reco_selected,
                        "Purity": purity,
                        "PurityError": purity_error,
                        "InMigration": in_migration,
                        "InMigrationError": in_migration_error,
                        "Inverse": inverse,
                        "FlashMatch": flash_matched,
                    }
                )

    df_fiducial = pd.DataFrame(fiducial_list)
    df_fiducial = df_fiducial.fillna(np.nan)
    # Plots keep the ungated rows; FlashMatch=True is carried in the pkl only.
    df_fiducial_plot = df_fiducial[df_fiducial["FlashMatch"] == "None"]
    for inverse, reference in product([True, False], ["Truth"]):
        this_fiducial_df = df_fiducial_plot[
            (df_fiducial_plot["Inverse"] == inverse)
            * (df_fiducial_plot["Reference"] == reference)
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
        this_fiducial_df = df_fiducial_plot[
            (df_fiducial_plot["Inverse"] == inverse)
            * (df_fiducial_plot["Reference"] == reference)
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

    df_purity = pd.DataFrame(purity_list)

    for this_df, df_filename, df_type in zip(
        [df_fiducial, df_purity],
        [
            "Fiducial_Efficiency",
            "Fiducial_Purity",
        ],
        ["pkl", "pkl"],
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
