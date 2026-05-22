import os
import re
import sys
from glob import glob as glob_files
from typing import Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


save_path = f"{root}/images/analysis/sensitivity/templates"


parser = argparse.ArgumentParser(
    description="Plot precomputed sensitivity templates without regenerating them"
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, default="marley")
parser.add_argument(
    "--reference",
    type=str,
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--folder",
    type=str,
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--energy",
    type=str,
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument("--dm2", type=float, default=None)
parser.add_argument("--sin13", type=float, default=None)
parser.add_argument("--sin12", type=float, default=None)
parser.add_argument(
    "--template",
    type=str,
    choices=["signal", "background", "all"],
    default="all",
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=None,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=None,
)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help="Oscillation backend used when computing templates. Determines fallback for nadir axis when pkl is absent.",
)
args = parser.parse_args()


if not os.path.exists(f"{save_path}/{args.folder.lower()}"):
    os.makedirs(f"{save_path}/{args.folder.lower()}")


analysis_info = load_analysis_info(str(root))
info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())


def load_best_cut_map() -> Optional[dict]:
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    for analysis in candidates:
        filepath = (
            f"{info['PATH']}/{analysis}/{args.folder.lower()}/{args.config}/{args.name}/"
            f"{args.config}_{args.name}_highest_{analysis}.pkl"
        )
        if os.path.exists(filepath):
            return pickle.load(open(filepath, "rb"))
    return None


def select_cuts() -> Optional[Tuple[int, int, int]]:
    if args.nhits is not None and args.ophits is not None and args.adjcls is not None:
        return int(args.nhits), int(args.ophits), int(args.adjcls)

    best_map = load_best_cut_map()
    if best_map is None:
        return None

    key = (args.config, args.name, args.energy)
    selected = best_map.get(key)
    if selected is None and len(best_map) > 0:
        selected = next(iter(best_map.values()))
    if selected is None:
        return None

    return int(selected["NHits"]), int(selected["OpHits"]), int(selected["AdjCl"])


def load_background_template(nhits: int, ophits: int, adjcl: int):
    exact = (
        f"{info['PATH']}/SENSITIVITY/{args.config}/background/{args.folder.lower()}/{args.energy}/"
        f"{args.config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl"
    )
    if os.path.exists(exact):
        return exact, np.asarray(pd.read_pickle(exact), dtype=float)

    pattern = (
        f"{info['PATH']}/SENSITIVITY/{args.config}/background/{args.folder.lower()}/{args.energy}/"
        f"{args.config}_background_NHits*_AdjCl*_OpHits*.pkl"
    )
    matches = sorted(glob_files(pattern))
    if not matches:
        return None, None

    def parse(filepath: str):
        base = os.path.basename(filepath)
        match = re.search(r"NHits(\d+)_AdjCl(\d+)_OpHits(\d+)", base)
        if match is None:
            return 10**9, 10**9, 10**9
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    best = min(
        matches,
        key=lambda path: (
            abs(parse(path)[0] - nhits),
            abs(parse(path)[1] - adjcl),
            abs(parse(path)[2] - ophits),
        ),
    )
    return best, np.asarray(pd.read_pickle(best), dtype=float)


def load_signal_template(nhits: int, ophits: int, adjcl: int):
    dm2 = analysis_info["SOLAR_DM2"] if args.dm2 is None else float(args.dm2)
    sin13 = analysis_info["SIN13"] if args.sin13 is None else float(args.sin13)
    sin12 = analysis_info["SIN12"] if args.sin12 is None else float(args.sin12)

    exact = (
        f"{info['PATH']}/SENSITIVITY/{args.config}/{args.name}/{args.folder.lower()}/{args.energy}/"
        f"{args.config}_{args.name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"
        f"_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
    )
    if os.path.exists(exact):
        return exact, np.asarray(pd.read_pickle(exact), dtype=float)

    pattern = (
        f"{info['PATH']}/SENSITIVITY/{args.config}/{args.name}/{args.folder.lower()}/{args.energy}/"
        f"{args.config}_{args.name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_*_sin13_*_sin12_*.pkl"
    )
    matches = sorted(glob_files(pattern))
    if not matches:
        return None, None

    first = matches[0]
    return first, np.asarray(pd.read_pickle(first), dtype=float)


def load_oscillation_axis() -> np.ndarray:
    dm2 = analysis_info["SOLAR_DM2"] if args.dm2 is None else float(args.dm2)
    sin13 = analysis_info["SIN13"] if args.sin13 is None else float(args.sin13)
    sin12 = analysis_info["SIN12"] if args.sin12 is None else float(args.sin12)
    osc_path = (
        f"{info['PATH']}/data/OSCILLATION/pkl/rebin/"
        f"osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
    )
    if args.oscillation_backend != "file" or not os.path.exists(osc_path):
        nadir_bins  = analysis_info.get("NADIR_BINS",  40)
        nadir_range = analysis_info.get("NADIR_RANGE", [-1.0, 1.0])
        nadir_edges   = np.linspace(nadir_range[0], nadir_range[1], nadir_bins + 1)
        nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])
        return nadir_centers.astype(float)
    oscillation_df = pd.read_pickle(osc_path)
    return np.asarray(list(oscillation_df.index), dtype=float)


cuts = select_cuts()
if cuts is None:
    rprint(
        f"[yellow][WARNING][/yellow] Unable to resolve cut triplet for {args.config} {args.name} {args.energy}."
    )
    raise SystemExit(0)

nhits_value, ophits_value, adjcl_value = cuts

background_path, background_template = (None, None)
signal_path, signal_template = (None, None)

if args.template in ["background", "all"]:
    background_path, background_template = load_background_template(
        nhits_value, ophits_value, adjcl_value
    )

if args.template in ["signal", "all"]:
    signal_path, signal_template = load_signal_template(
        nhits_value, ophits_value, adjcl_value
    )

panel_specs = []
if background_template is not None:
    panel_specs.append(("Background", background_template, "Turbo", True))
if signal_template is not None:
    panel_specs.append(("Signal", signal_template, "Turbo", True))
if background_template is not None and signal_template is not None:
    panel_specs.append(("|Signal - Background|", signal_template - background_template, "Turbo", True))

if not panel_specs:
    rprint(
        f"[yellow][WARNING][/yellow] No template payloads found for selected cuts NHits{nhits_value} OpHits{ophits_value} AdjCl{adjcl_value}."
    )
    raise SystemExit(0)

azimuth_axis = load_oscillation_axis()

log_values = []
for title, template_data, _, use_log in panel_specs:
    values = np.asarray(template_data, dtype=float)
    if use_log:
        source = values if title != "|Signal - Background|" else np.abs(values)
        if np.any(source > 0):
            log_values.append(np.log10(source[source > 0]))
log_min = float(min(np.nanmin(values) for values in log_values)) if log_values else None
log_max = float(max(np.nanmax(values) for values in log_values)) if log_values else None

fig = make_subplots(
    rows=1,
    cols=len(panel_specs),
    subplot_titles=tuple([item[0] for item in panel_specs]),
)

for idx, (title, template_data, colorscale, use_log) in enumerate(panel_specs, start=1):
    zvals = np.asarray(template_data, dtype=float)
    source = np.abs(zvals) if title == "|Signal - Background|" else zvals
    zplot = np.log10(np.where(source > 0, source, np.nan))
    fig.add_trace(
        go.Heatmap(
            z=zplot,
            x=sensitivity_rebin_centers,
            y=azimuth_axis,
            colorscale=colorscale,
            coloraxis="coloraxis",
            zmin=log_min,
            zmax=log_max,
            showscale=(idx == 1),
        ),
        row=1,
        col=idx,
    )

fig = format_coustom_plotly(
    fig,
    title=(
        f"Sensitivity Templates {args.config} {args.name} {args.energy} "
        f"NHits{nhits_value} OpHits{ophits_value} AdjCl{adjcl_value}"
    ),
    tickformat=(".1f", ".1f"),
    matches=("x", "y"),
)

for idx in range(1, len(panel_specs) + 1):
    fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=idx, range=[sensitivity_rebin_centers.min(), sensitivity_rebin_centers.max()])
    fig.update_yaxes(title="cos(Azimuth)", row=1, col=idx, range=[-1.0, 1.0])

fig.update_layout(
    coloraxis=dict(colorbar=dict(title="log10(Counts)")),
)

figure_name = f"Sensitivity_Templates_{args.energy}_NHits{nhits_value}_AdjCl{adjcl_value}_OpHits{ophits_value}"
save_figure(
    fig,
    f"{save_path}/{args.folder.lower()}",
    config=args.config,
    name=args.name,
    subfolder=None,
    filename=figure_name,
    rm=args.rewrite,
    debug=args.plot,
)

if args.debug:
    if background_path is not None:
        rprint(f"[cyan][INFO][/cyan] Background template source: {background_path}")
    if signal_path is not None:
        rprint(f"[cyan][INFO][/cyan] Signal template source: {signal_path}")
