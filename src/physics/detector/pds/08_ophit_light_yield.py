"""
08_ophit_light_yield.py — detected photoelectrons per MeV vs drift coordinate
=============================================================================
Light yield of the signal from the "ophit" productions, which store every OpHit of
the event in MCTruthTree (SaveSignalOpHits). For each event the signal PE is the
purity-weighted sum of the OpHit charges, sum(OpHitPE * OpHitPur), so dark counts
and other uncorrelated hits do not contribute. The module saves only the OpHits clustered into
the reconstructed (solarflash) flashes, so this is the signal light the flash finding collects. It is divided by the visible energy,
the summed kinetic energy of the primary electrons, positrons and gammas
(TSignalMother == 0), which excludes the neutrino, nucleons and nuclei.

The scan follows the Drift Scan of 04_matched_efficiency.py: neutrino energy bins
(lowe_energy_centers) x drift-coordinate bins of DEFAULT_X_BIN, |X| for HD (distance
to the central APA), X shifted by DETECTOR_MAX_X for VD.

A second scan splits |SignalParticleY| (the detectors are symmetric in y) into three equal
shells up to DETECTOR_MAX_Y (Central, Intermediate, Edge), with all events of the energy range
(lowe_energy_edges) combined.

Output: output/data/PDS/ophit/<config>/<name>/<config>_<name>_OpHit_LightYield_Drift_Scan.pkl
  columns Config, Name, Geometry, Plot ("X Scan"), Energy (neutrino, MeV), Variable ("X"),
  Coordinate (cm), LightYield (mean PE/MeV), Error (standard error of the mean),
  Median, MedianError, Lower, Upper (16 and 84 % quantiles), STD, PE (mean signal PE),
  VisibleEnergy (mean, MeV), Counts (events), Relative (LightYield over that of all
  energies in the drift bin), RelativeError, RelativeMedian (same with medians),
  RelativeMedianError.
        <config>_<name>_OpHit_LightYield_Region_Scan.pkl
  same statistics with Region, YMin, YMax (cm) in place of Energy.

Run
---
  python3 src/physics/detector/pds/08_ophit_light_yield.py --config hd_1x2x6 --name marley_ophit
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/PDS/ophit"
data_path = f"{root}/output/data/PDS/ophit"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description="Signal PE per MeV vs drift coordinate from the ophit productions")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley_ophit")
parser.add_argument("--chunk", type=int, help="Events per chunk for the OpHit sums", default=20000)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
config, name = args.config, args.name

info, params, output = get_param_dict(f"{root}/config/{config}/{config}", {}, "", debug=args.debug)
sample = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}/Truth'
if not os.path.isfile(f"{sample}/OpHitPE.npy"):
    sys.exit(f"ERROR: {sample}/OpHitPE.npy not found (not an ophit production)")


def load(branch):
    return np.load(f"{sample}/{branch}.npy", mmap_mode="r")


# Event-level sums, chunked: the OpHit and TSignal branches are (events x hits) arrays of several GB.
n_events = len(load("Event"))
signal_pe = np.zeros(n_events, dtype=np.float64)
visible_energy = np.zeros(n_events, dtype=np.float64)
for start in range(0, n_events, args.chunk):
    stop = min(start + args.chunk, n_events)
    pe = np.asarray(load("OpHitPE")[start:stop], dtype=np.float64)
    pur = np.nan_to_num(np.asarray(load("OpHitPur")[start:stop], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    signal_pe[start:stop] = np.sum(pe * pur, axis=1)
    k = np.asarray(load("TSignalK")[start:stop], dtype=np.float64)
    pdg = np.asarray(load("TSignalPDG")[start:stop])
    mother = np.asarray(load("TSignalMother")[start:stop])
    visible = (mother == 0) & np.isin(np.abs(pdg), [11, 22])
    visible_energy[start:stop] = np.sum(k * visible, axis=1)
    if args.debug:
        rprint(f"  events {stop}/{n_events}")

energy = np.asarray(load("SignalParticleK"), dtype=np.float64)
drift = np.asarray(load("SignalParticleX"), dtype=np.float64)
valid = visible_energy > 0
light_yield = np.divide(signal_pe, visible_energy, out=np.full(n_events, np.nan), where=valid)
rprint(f"[cyan]{config} {name}: {n_events} events, {int(valid.sum())} with visible energy[/cyan]")

coord = np.abs(drift) if info["GEOMETRY"] == "hd" else drift
height = np.abs(np.asarray(load("SignalParticleY"), dtype=np.float64))  # detectors are symmetric in y
drift_values = np.arange(
    0 if info["GEOMETRY"] == "hd" else info["DETECTOR_MIN_X"],
    info["DETECTOR_MAX_X"] + params["DEFAULT_X_BIN"],
    params["DEFAULT_X_BIN"],
)
in_energy_range = (energy > lowe_energy_edges[0]) & (energy < lowe_energy_edges[-1])


def drift_mask(value):
    return (coord > value - params["DEFAULT_X_BIN"] / 2) & (coord < value + params["DEFAULT_X_BIN"] / 2)


def light_yield_stats(mask):
    values = light_yield[mask]
    counts = len(values)
    median = np.median(values) if counts > 0 else np.nan
    lower = np.quantile(values, 0.16) if counts > 0 else np.nan
    upper = np.quantile(values, 0.84) if counts > 0 else np.nan
    return {
        "LightYield": np.mean(values) if counts > 0 else np.nan,
        "Error": np.std(values) / np.sqrt(counts) if counts > 1 else np.nan,
        "Median": median,
        # robust error of the median: 1.2533 x (16-84 % half width) / sqrt(N); the VD membrane
        # modules give a long high tail (events in front of a module), which makes the mean unstable
        "MedianError": 1.2533 * (upper - lower) / 2 / np.sqrt(counts) if counts > 1 else np.nan,
        "Lower": lower,
        "Upper": upper,
        "STD": np.std(values) if counts > 0 else np.nan,
        "PE": np.mean(signal_pe[mask]) if counts > 0 else np.nan,
        "VisibleEnergy": np.mean(visible_energy[mask]) if counts > 0 else np.nan,
        "Counts": counts,
    }


# Neutrino-energy scan. Relative: ratio to the light yield of all events of the energy range
# (lowe_energy_edges) in the same drift bin, which removes the drift dependence.
light_yield_plot = []
for this_energy in lowe_energy_centers:
    for value in drift_values:
        reference = light_yield_stats(valid & in_energy_range & drift_mask(value))
        stats = light_yield_stats(
            valid
            & (energy > this_energy - lowe_ebin / 2)
            & (energy < this_energy + lowe_ebin / 2)
            & drift_mask(value)
        )
        light_yield_plot.append(
            {
                "Name": name,
                "Plot": "X Scan",
                "Energy": this_energy,
                "Variable": "X",
                "Coordinate": value + info["DETECTOR_MAX_X"] if info["GEOMETRY"] == "vd" else value,
                **stats,
                "Relative": stats["LightYield"] / reference["LightYield"],
                "RelativeError": stats["Error"] / reference["LightYield"],
                "RelativeMedian": stats["Median"] / reference["Median"],
                "RelativeMedianError": stats["MedianError"] / reference["Median"],
            }
        )

light_yield_df = pd.DataFrame(light_yield_plot)
light_yield_df["Geometry"] = info["GEOMETRY"]
light_yield_df["Config"] = config
light_yield_df["Name"] = name
print(light_yield_df)
save_df(
    light_yield_df,
    data_path,
    config,
    name,
    filename="OpHit_LightYield_Drift_Scan",
    rm=args.rewrite,
    debug=args.debug,
)

# Height scan: |y| split in three equal shells (central, intermediate, edge), energy range combined.
region_edges = np.linspace(0, info["DETECTOR_MAX_Y"], 4)
region_plot = []
for region, low, high in zip(["Central", "Intermediate", "Edge"], region_edges[:-1], region_edges[1:]):
    upper = np.inf if region == "Edge" else high  # events slightly beyond DETECTOR_MAX_Y go to the edge shell
    for value in drift_values:
        region_plot.append(
            {
                "Name": name,
                "Plot": "X Scan",
                "Region": region,
                "YMin": low,
                "YMax": high,
                "Variable": "X",
                "Coordinate": value + info["DETECTOR_MAX_X"] if info["GEOMETRY"] == "vd" else value,
                **light_yield_stats(valid & in_energy_range & drift_mask(value) & (height >= low) & (height < upper)),
            }
        )

region_df = pd.DataFrame(region_plot)
region_df["Geometry"] = info["GEOMETRY"]
region_df["Config"] = config
region_df["Name"] = name
print(region_df)
save_df(
    region_df,
    data_path,
    config,
    name,
    filename="OpHit_LightYield_Region_Scan",
    rm=args.rewrite,
    debug=args.debug,
)

fig = make_subplots(rows=1, cols=1)
for jdx, this_energy in enumerate(np.unique(light_yield_df["Energy"])):
    this_df = light_yield_df[light_yield_df["Energy"] == this_energy]
    fig.add_trace(
        go.Scatter(
            x=this_df["Coordinate"],
            y=this_df["LightYield"],
            error_y=dict(type="data", array=this_df["Error"]),
            legendgroup=0,
            legendgrouptitle=dict(text=f"Energy (+- {lowe_ebin/2} MeV)", font=dict(size=13)),
            mode="lines+markers",
            name=f"{this_energy}",
            line=dict(color=colors[jdx % len(colors)]),
        )
    )
fig = format_coustom_plotly(fig, title=f"OpHit Light Yield vs Drift - {config} {name}")
fig.update_xaxes(title_text="Drift Coordinate (cm)")
fig.update_yaxes(title_text="Signal PE / Visible Energy (PE/MeV)", rangemode="tozero")
save_figure(
    fig,
    save_path,
    config,
    name,
    filename="OpHit_LightYield_Drift_Scan",
    rm=args.rewrite,
    debug=args.debug,
)
