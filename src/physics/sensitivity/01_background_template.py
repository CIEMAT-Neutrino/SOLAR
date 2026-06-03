import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

from lib.root import Sensitivity_Fitter
from lib.oscillation import get_oscillation_datafiles
from lib.oscillation_backends import get_nadir_pdf_nufast

save_path = f"{root}/images/analysis/sensitivity/templates"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and args.name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis",
    default=30,
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help="Oscillation backend. 'file' uses pre-computed pkl files for the nadir axis; 'prob3'/'nufast' derive it from analysis/physics.json.",
)

args = parser.parse_args()
if args.debug:
    rprint(args)
# smoothing_config = get_smoothing_config(
#     str(root), analysis_name="SENSITIVITY", dimensions="2d", stage="significance"
# )
smoothing_config_1d = get_smoothing_config(
    str(root), analysis_name="SENSITIVITY", dimensions="1d", stage="significance"
)
# smoothing_config = dict(smoothing_config)
# smoothing_config["params"] = dict(smoothing_config.get("params", {}))
# smoothing_config["params"]["sigma_y"] = 0.0
smoothing_info = smoothing_metadata(smoothing_config_1d)


def _load_best_cut_map(info: dict, args):
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    tried = []
    for analysis in candidates:
        filepath = (
            f"{info['PATH']}/{analysis}/{args.folder.lower()}/{args.config}/{args.name}/"
            f"{args.config}_{args.name}_highest_{analysis}.pkl"
        )
        tried.append(filepath)
        if os.path.exists(filepath):
            if args.debug:
                rprint(f"[cyan][INFO][/cyan] Using best-cut map from {analysis}")
            return pickle.load(open(filepath, "rb"))

    rprint(
        "[yellow][WARNING][/yellow] Unable to load any best-cut map. Checked:\n"
        + "\n".join(tried)
    )
    return None


def _project_1d_to_2d(hist_1d, oscillation_df, nadir_pdf=None):
    """Project a 1D energy spectrum into a nadir-weighted 2D template.

    nadir_pdf: pre-computed normalised weights over oscillation_df.index.
    If None, loads from nadir.root via get_nadir_angle() (file backend).
    """
    hist_1d = np.asarray(hist_1d, dtype=float)
    hist2d = np.tile(hist_1d / len(oscillation_df), (len(oscillation_df), 1))

    if nadir_pdf is not None:
        rebin_nadir = np.asarray(nadir_pdf, dtype=float)
    else:
        nadir = get_nadir_angle()
        interp_nadir = interp1d(*nadir)
        rebin_nadir = interp_nadir(oscillation_df.index)

    hist2d = rebin_nadir * hist2d.T
    norm = np.sum(hist2d)
    if norm > 0:
        hist2d = np.sum(hist_1d) * hist2d.T / norm
    else:
        hist2d = hist2d.T
    return hist2d

for path in [save_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

plot_df = pd.DataFrame()
oscillation_df = pd.DataFrame()
background_samples = []
dm2_list, sin13_list, sin12_list = [], [], []

analysis_info = load_analysis_info(str(root))
info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
fiducials = json.loads(open(f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())

detector_mass = get_full_detector_mass(args.config, info)

df_list = []
background_samples = []
for bkg, filepath in load_available_background_dataframes(str(root), "SENSITIVITY", args.folder, args.config, args.energy):
    bkg_df = pd.read_pickle(filepath)
    df_list.append(bkg_df)
    background_samples.append(bkg)

plot_df = pd.concat(df_list, ignore_index=True)

plot_df = explode(
    plot_df, ["Counts", "Counts/Energy", "Error", "Energy"], debug=args.debug
)
plot_df["Counts"] = plot_df["Counts"].replace(0, np.nan)
plot_df["Counts/Energy"] = plot_df["Counts/Energy"].replace(0, np.nan)

if args.oscillation_backend == "file":
    (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
        dm2=None,
        sin13=None,
        sin12=None,
        path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
        ext="pkl",
    )
    for dm2, sin13, sin12 in product(dm2_list, sin13_list, sin12_list):
        oscillation_df = pd.read_pickle(
            f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
        )
else:
    nadir_edges = np.linspace(-1.0, 1.0, analysis_info["NADIR_BINS"] + 1)
    nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])
    oscillation_df = pd.DataFrame(index=nadir_centers)

# Pre-compute nadir PDF for _project_1d_to_2d — use NuFast-Earth Solar_Weight when
# backend is "nufast" so the run stays fully file-free; otherwise interpolate nadir.root.
_latitude_deg = analysis_info.get("DUNE_LATITUDE_DEG", 44.35)
_nadir_centers_arr = np.asarray(oscillation_df.index, dtype=float)
if args.oscillation_backend == "nufast":
    _nadir_pdf_weights = get_nadir_pdf_nufast(_nadir_centers_arr, _latitude_deg)
    _nadir_plot_x = _nadir_centers_arr
    _nadir_plot_y = _nadir_pdf_weights
else:
    _nadir_raw = get_nadir_angle()
    _nadir_plot_x = _nadir_raw[0]
    _nadir_plot_y = _nadir_raw[1]
    _nadir_pdf_weights = None  # _project_1d_to_2d will call get_nadir_angle() itself

# Panel 2: standalone nadir time-fraction distribution p(cos θ_z) at DUNE latitude
_nadir_fig = make_subplots(rows=1, cols=1)
_nadir_fig.add_trace(go.Scatter(
    x=list(_nadir_plot_x), y=list(_nadir_plot_y),
    mode="lines", fill="tozeroy",
    line=dict(color="steelblue", width=2),
    name="Time-Fraction",
), row=1, col=1)
_nadir_fig = format_coustom_plotly(
    _nadir_fig, title=f"Nadir Time-Fraction at DUNE (lat. {_latitude_deg}°)",
)
_nadir_fig.update_xaxes(title="cos(η) Zenith Angle")
_nadir_fig.update_yaxes(title="Time-Fraction per Bin")
save_figure(
    _nadir_fig, save_path, config=args.config, name=args.name, subfolder=args.folder.lower(),
    filename="NadirDistribution", rm=args.rewrite, debug=args.plot,
)

cut_entries = []
if args.nhits is not None and args.adjcls is not None and args.ophits is not None:
    cut_entries = [
        {
            "NHits": int(args.nhits),
            "AdjCl": int(args.adjcls),
            "OpHits": int(args.ophits),
        }
    ]
else:
    fastest_sigma = _load_best_cut_map(info, args)
    if fastest_sigma is not None:
        cut_entries = [
            {
                "NHits": int(value["NHits"]),
                "AdjCl": int(value["AdjCl"]),
                "OpHits": int(value["OpHits"]),
            }
            for value in fastest_sigma.values()
        ]
    else:
        available = sorted(
            {
                (int(row["NHits"]), int(row["AdjCl"]), int(row["OpHits"]))
                for _, row in plot_df[["NHits", "AdjCl", "OpHits"]].drop_duplicates().iterrows()
            },
            key=lambda x: (x[0], x[2], x[1]),
        )
        cut_entries = [
            {"NHits": nh, "AdjCl": ad, "OpHits": op}
            for nh, ad, op in available
        ]
        rprint(
            f"[yellow][WARNING][/yellow] Falling back to {len(cut_entries)} cut triplets discovered from background data"
        )

for idx, cut in enumerate(cut_entries):
    if args.energy is not None:
        energy = args.energy
    else:
        energy = args.energy

    total = np.zeros(len(sensitivity_rebin) - 1)
    total_error = np.zeros(len(sensitivity_rebin) - 1)

    nhits = int(cut["NHits"])
    adjcl = int(cut["AdjCl"])
    ophits = int(cut["OpHits"])

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=(
            [
                "Background Components",
                "Raw 2D Background",
                "Smoothed 2D Background",
                "Residual (Smoothed - Raw)",
            ]
        ),
    )

    component_energy = np.asarray(sensitivity_rebin_centers)

    for bkg in background_samples:
        this_df = plot_df[
            (plot_df["Component"] == bkg)
            * (plot_df["NHits"] == nhits)
            * (plot_df["OpHits"] == ophits)
            * (plot_df["AdjCl"] == adjcl)
        ]

        if this_df.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Empty dataframe for {bkg} with NHits{nhits} OpHits{ophits} AdjCl{adjcl}"
            )
            continue

        x = np.asarray(list(this_df["Energy"].values))
        component_energy = x
        y = np.asarray(list(this_df["Counts"].values))
        y_error = np.asarray(list(this_df["Error"].values))
        y = np.nan_to_num(y)
        y_error = np.nan_to_num(y_error)

        component_smoothing_config_1d = get_component_smoothing_config(
            smoothing_config_1d, bkg
        )
        y_smoothed = smooth_histogram_with_config(y, component_smoothing_config_1d)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=bkg,
                line_shape="hvh",
                line=dict(color=this_df["Color"].values[0], dash="dot", width=2),
                opacity=0.45,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_smoothed,
                mode="lines",
                name=bkg,
                line_shape="hvh",
                line=dict(color=this_df["Color"].values[0], dash="solid", width=3),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        # print(total)
        # print(y)
        total = total + np.array(y)
        total_error = total_error + y_error**2

    total_smoothed = smooth_histogram_with_config(total, smoothing_config_1d)

    if idx == 0:
        # Panel 5: standalone 1D background spectrum b(E_reco) for the best-selected cut
        _bkg_1d_fig = make_subplots(rows=1, cols=1)
        for _bkg in background_samples:
            _bkg_slice = plot_df[
                (plot_df["Component"] == _bkg)
                & (plot_df["NHits"] == nhits)
                & (plot_df["OpHits"] == ophits)
                & (plot_df["AdjCl"] == adjcl)
            ]
            if _bkg_slice.empty:
                continue
            _bkg_y = np.nan_to_num(np.asarray(list(_bkg_slice["Counts"].values)))
            _bkg_1d_fig.add_trace(go.Scatter(
                x=np.asarray(list(_bkg_slice["Energy"].values), dtype=float),
                y=_bkg_y,
                mode="lines", name=_bkg, line_shape="hvh",
                line=dict(color=_bkg_slice["Color"].values[0], width=2),
            ), row=1, col=1)
        _bkg_1d_fig.add_trace(go.Scatter(
            x=component_energy, y=total_smoothed,
            mode="lines", name="Total (smoothed)", line_shape="hvh",
            line=dict(color="black", dash="solid", width=3),
        ), row=1, col=1)
        _bkg_1d_fig = format_coustom_plotly(
            _bkg_1d_fig,
            title=f"1D Background Spectrum {args.config} {energy}",
            legend_title="Component",
        )
        _bkg_1d_fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)")
        _bkg_1d_fig.update_yaxes(title="Counts per Energy (kt·yr·MeV)⁻¹", type="log", range=[-1, 7])
        save_figure(
            _bkg_1d_fig, save_path, config=args.config, name=args.name, subfolder=args.folder.lower(),
            filename=f"Background1D_{energy}", rm=args.rewrite, debug=args.plot,
        )

    bkg_hist = _project_1d_to_2d(total, oscillation_df, nadir_pdf=_nadir_pdf_weights)
    smoothed_bkg_hist = _project_1d_to_2d(total_smoothed, oscillation_df, nadir_pdf=_nadir_pdf_weights)

    bkg_hist[args.exposure * detector_mass * bkg_hist < 1] = 0.0
    smoothed_bkg_hist[args.exposure * detector_mass * smoothed_bkg_hist < 1] = 0.0
    residual_bkg_hist = smoothed_bkg_hist - bkg_hist
    
    if args.debug:
        print(f"Check Counts: {np.sum(total)} - {np.sum(bkg_hist)}")
        print(f"Smoothed counts: {np.sum(smoothed_bkg_hist)} using {smoothing_info['SmoothingMethod']}")
        rprint(
            f"[cyan][INFO][/cyan] Saving sensitivity background template for {args.config} {args.folder} {energy} with NHits{nhits} AdjCl{adjcl} OpHits{ophits}"
        )
        
    save_pkl(
        args.exposure * detector_mass * smoothed_bkg_hist,
        f"{info['PATH']}/SENSITIVITY",
        config=args.config,
        name=f"background",
        subfolder=f"{args.folder.lower()}/{energy}",
        filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.debug,
    )

    fig.add_trace(
        go.Heatmap(
            z=np.log10(np.where(bkg_hist > 0, bkg_hist, np.nan)),
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="Turbo",
            colorbar=dict(title="log(Counts)"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=np.log10(np.where(smoothed_bkg_hist > 0, smoothed_bkg_hist, np.nan)),
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="Turbo",
            showscale=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Heatmap(
            z=residual_bkg_hist,
            x=sensitivity_rebin_centers,
            y=oscillation_df.index,
            colorscale="RdBu",
            zmid=0.0,
            showscale=False,
        ),
        row=1,
        col=4,
    )

    fig.add_trace(
        go.Scatter(
            x=component_energy,
            y=total,
            error_y=dict(type="data", array=np.sqrt(total_error), visible=True),
            mode="lines",
            name="Total Raw",
            line_shape="hvh",
            line=dict(color="black", dash="dot", width=2),
            opacity=0.45,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=component_energy,
            y=total_smoothed,
            mode="lines",
            name="Total Smoothed",
            line_shape="hvh",
            line=dict(color="black", dash="solid", width=3),
        ),
        row=1,
        col=1,
    )

    add_histogram_style_legend_traces(
        fig,
        row=1,
        col=1,
        legend="legend2",
    )

    fig = format_coustom_plotly(
        fig,
        title=f"{energy} Background {args.config}",
        log=(False, False),
        matches=("x", None),
        tickformat=(".1f", ".0e"),
        legend_title="Component",
        debug=args.debug,
    )

    fig.update_layout(
        legend2=dict(x=0.12, y=0.94, bgcolor="rgba(255,255,255,0.7)"),
    )

    fig.update_xaxes(
        title=f"Reconstructed Energy (MeV)",
    )

    fig.update_yaxes(
        title=f"Counts per Energy (kT·year·MeV)⁻¹",
        type="log",
        range=[-1, 7],
        row=1,
        col=1,
    )

    save_figure(
        fig,
        f"{save_path}",
        config=args.config,
        name=args.name,
        subfolder=args.folder.lower(),
        filename=f"Background_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
        rm=args.rewrite,
        debug=args.plot,
    )

    if args.energy is not None and isinstance(args.energy, str):
        break
