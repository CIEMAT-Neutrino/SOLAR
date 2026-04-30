import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


save_path = f"{root}/images/analysis/hep"
data_path = f"{root}/data/analysis/hep"
for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


def get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace):
    if args.nhits is not None and args.ophits is not None and args.adjcls is not None:
        return int(args.nhits), int(args.ophits), int(args.adjcls)

    sigma_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_highest_HEP.pkl"
    )
    if not os.path.exists(sigma_path):
        return None

    sigma = pd.read_pickle(sigma_path)
    try:
        ref_plot = sigma[(config, name, energy)]
    except KeyError:
        return None

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
    return int(nhits_value), int(ophits_value), int(adjcl_value)

parser = argparse.ArgumentParser(
    description="Compare Gaussian, Asimov, and ProfileLikelihood HEP exposure curves"
)
parser.add_argument("--analysis", type=str, default="HEP")
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Reduced")
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--signal_uncertainty", type=float, default=0.04)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
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
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

comparison_styles = {
    ("Asimov", "Raw"): dict(color="black", dash="dot", width=2),
    ("Asimov", "Smoothed"): dict(color="black", dash="solid", width=3),
    ("Gaussian", "Raw"): dict(color=compare[1], dash="dot", width=2),
    ("Gaussian", "Smoothed"): dict(color=compare[1], dash="solid", width=3),
    ("ProfileLikelihood", "Raw"): dict(color=compare[2], dash="dot", width=2),
    ("ProfileLikelihood", "Smoothed"): dict(color=compare[2], dash="solid", width=3),
}

reference_variables = ["Asimov", "Gaussian", "ProfileLikelihood"]

for config, name, energy in product(args.config, args.name, args.energy):
    selection = get_selection_cuts(config, name, energy, args)
    if selection is None:
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}."
        )
        continue
    nhits_value, ophits_value, adjcl_value = selection

    exposure_file = (
        Path(data_path)
        / config
        / name
        / args.folder.lower()
        / f"{config}_{name}_HEP_Exposure.pkl"
    )
    if not exposure_file.exists():
        rprint(
            f"[yellow][WARNING][/yellow] Missing saved HEP exposure data for {config} {name} {energy}. Run 13HEPExposurePlot.py first."
        )
        continue

    exposure_df = pd.read_pickle(exposure_file)
    required_columns = [
        "Config",
        "Name",
        "EnergyLabel",
        "Variable",
        "Exposure",
        "Significance",
        "RawSignificance",
    ]
    missing_columns = [col for col in required_columns if col not in exposure_df.columns]
    if missing_columns:
        rprint(
            f"[yellow][WARNING][/yellow] Missing columns {missing_columns} in {exposure_file}."
        )
        continue

    exposure_rows = exposure_df.loc[
        (exposure_df["Config"] == config)
        & (exposure_df["Name"] == name)
        & (exposure_df["EnergyLabel"] == energy)
        & (exposure_df["Variable"].isin(["Asimov", "Gaussian"]))
    ].copy()

    for column, value in [
        ("NHits", nhits_value),
        ("OpHits", ophits_value),
        ("AdjCl", adjcl_value),
    ]:
        if column in exposure_rows.columns:
            exposure_rows = exposure_rows.loc[exposure_rows[column] == int(value)].copy()

    profile_rows = pd.DataFrame()
    profile_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{energy}_{args.analysis}_ProfileLikelihood.pkl"
    )
    if os.path.exists(profile_path):
        profile_df = pd.read_pickle(profile_path)
        profile_rows = profile_df.loc[
            (profile_df["Config"] == config)
            & (profile_df["Name"] == name)
            & (profile_df["Energy"] == energy)
            & (profile_df["NHits"] == int(nhits_value))
            & (profile_df["OpHits"] == int(ophits_value))
            & (profile_df["AdjCl"] == int(adjcl_value))
        ].copy()

    if exposure_rows.empty and profile_rows.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing exposure comparison inputs for {config} {name} {energy} after cut filtering NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    fig = make_subplots(rows=1, cols=1, subplot_titles=("",))

    significance_max = 0.0
    for variable in reference_variables:
        row = exposure_rows.loc[exposure_rows["Variable"] == variable]
        if variable == "ProfileLikelihood":
            row = profile_rows
        if row.empty:
            continue

        xvals = np.asarray(row["Exposure"].values[0], dtype=float)
        if variable == "ProfileLikelihood":
            yvals = np.nan_to_num(
                np.asarray(row["ProfileLikelihood"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            yraw = np.nan_to_num(
                np.asarray(row["RawProfileLikelihood"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            err_plus = np.nan_to_num(
                np.asarray(row["ProfileLikelihood+Error"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            err_minus = np.nan_to_num(
                np.asarray(row["ProfileLikelihood-Error"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_upper = err_plus
            y_lower = err_minus
        else:
            yvals = np.nan_to_num(
                np.asarray(row["Significance"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            yraw = np.nan_to_num(
                np.asarray(row["RawSignificance"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_upper = None
            y_lower = None
            if (
                "SignificanceError+" in row.columns
                and "SignificanceError-" in row.columns
                and len(row["SignificanceError+"].values) > 0
                and len(row["SignificanceError-"].values) > 0
            ):
                err_plus = np.nan_to_num(
                    np.asarray(row["SignificanceError+"].values[0], dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                err_minus = np.nan_to_num(
                    np.asarray(row["SignificanceError-"].values[0], dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if err_plus.size == yvals.size and err_minus.size == yvals.size:
                    y_upper = yvals + err_plus
                    y_lower = yvals - err_minus

        significance_max = max(significance_max, float(np.max(yvals)), float(np.max(yraw)))
        if y_upper is not None:
            significance_max = max(significance_max, float(np.max(y_upper)))

        style_raw = comparison_styles[(variable, "Raw")]
        style_smoothed = comparison_styles[(variable, "Smoothed")]

        add_reference_pair_traces(
            fig,
            x=xvals,
            y_raw=yraw,
            y_smoothed=yvals,
            name=variable,
            raw_style=style_raw,
            smoothed_style=style_smoothed,
            row=1,
            col=1,
            legend="legend",
            legendgrouptitle="Reference",
            showlegend_raw=False,
            showlegend_smoothed=True,
            line_shape="linear",
            y_upper=y_upper,
            y_lower=y_lower,
        )

    fig = format_coustom_plotly(
        fig,
        tickformat=(".1f", ".0e"),
        add_units=False,
        legend_title=f"{energy}",
        title=f"HEP Discovery Comparison - {args.folder} - {config}",
        matches=(None, None),
    )

    fig.update_yaxes(
        tickformat=".1f",
        dtick=1,
        range=[0, max(1.0, 1.1 * significance_max)] if args.zoom else [0, 6],
        title="Significance (sigma)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[-1, args.exposure],
        zeroline=False,
        title="Exposure (kT·year)",
        row=1,
        col=1,
    )

    # for sigma, cl in zip([1, 2, 3, 4, 5], [0.6827, 0.9545, 0.9973, 0.9999, 0.999999]):
    #     fig.add_hline(y=sigma, line_dash="dash", line_color="grey")
    #     fig.add_annotation(
    #         x=max(0.5, args.exposure * 0.1),
    #         y=sigma + 0.2,
    #         text=f"{100 * cl:.2f}% CL",
    #         showarrow=False,
    #     )

    fig.update_layout(
        legend_title_text="",
        legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.7)"),
    )

    figure_name = f"{energy}_HEP_Exposure_Comparison"
    if args.threshold is not None:
        figure_name += f"_Threshold_{args.threshold:.0f}"

    save_figure(
        fig,
        save_path,
        config=config,
        name=None,
        subfolder=args.folder.lower(),
        filename=figure_name,
        rm=args.rewrite,
        debug=args.plot,
    )
