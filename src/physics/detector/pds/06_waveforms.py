"""
06_waveforms.py — PDS waveform characterisation
===============================================
Characterises the raw PDS waveforms saved by the 2026-09 SolarNuAna waveform productions
(duneana smanthey/solar_waveform_analysis). For every flash the module stores the
waveform of its main (highest-PE) OpHit channel:

  VD (dunevd10kt, 10 ppm Xe): raw::OpDetWaveform from `opdigi10ppm` (WaveformDigitizerSim,
      standard_daphne): pedestal 100 ADC, 13-bit, analytic SPE pulse (13 ns rise, 386 ns
      fall, ~9 ADC/PE), 320-tick readout windows with 20 pre-trigger ticks, extended when
      the channel retriggers.
  HD (dune10kt, pure LAr): raw waveforms from the 2026-09-28 PIC re-run (the FNAL/official
      HD productions have none: solar_reco drops the deconvolved `opdec` product).
      OpDetDigitizerDUNE (dunefd_opdigi_unganged): pedestal 1500 ADC, 1000-tick windows with
      100 pre-trigger ticks, SPE shape from data (SPE_DAPHNE2_FBK_2022.dat).
  The SPE response differs between the two, so shape differences are not only the
  scintillation profile; the lowest PE bins (SPE-dominated) measure each response.
  Both run at 62.5 MHz (16 ns ticks). Digitiser constants are in DIGITIZERS below.

Two sources are read, chunked straight from the ROOT file:

  Truth    MCTruthTree OpFlashWaveform: every flash of the event (only in *_waveform /
           *_opwaveform files). The waveform timestamp is not stored here, so only
           single-window waveforms (<= SINGLE_WINDOW ticks) are used, whose trigger sits at
           the pre-trigger position.
  Matched  SolarNuAnaTree MatchedOpFlashWaveform: the flash matched to each cluster (also in
           *_flash files). The flash time relative to the waveform timestamp locates the
           pulse (searched in [-40, +120] ticks, since the flash time is PE-weighted), so long
           retriggered windows are usable too. These waveforms are mostly long (median ~680
           ticks): high-light flashes retrigger the channel.

Each waveform is baseline-subtracted (median of the first BASELINE_TICKS), its pulse peak is
found near the expected position, and it is cut to [-PRE, +POST) ticks around the peak.
Pulse-shape parameters (pulse_shape): PeakTime = peak - onset (10 % leading-edge crossing,
interpolated), RiseTime 10-90 %, T50/T90/T99 = time from the onset containing 50/90/99 % of the
positive-lobe charge within 4.8 us (only pulses whose waveform covers that window), FWHM,
undershoot depth and time, lobe charge. Per-waveform undershoot at low PE is mostly noise;
use the template value. Also: baseline and its RMS, amplitude, SNR, integral, prompt fraction (F_prompt, the charge
within [-3, +6] ticks of the peak over the full integral, i.e. the first ~100 ns), late
fraction (> 1 us after the peak), 10-90 % rise time, time to fall to 1/2 and 1/e of the
amplitude, ADC/PE of the main hit, saturation (digitiser range) and a secondary-pulse flag (a later pulse
rising above the running tail minimum by SECONDARY_RATIO x amplitude, or 4 sigma of the noise).

Outputs (low weight, for the plotting repo): output/data/PDS/waveform/<config>/<name>/
  Waveform_Templates   mean/STD/SEM waveform per category (amplitude- and integral-normalised),
                       peak-aligned, time in us; clean (unsaturated) and all
  Waveform_Fits        two-exponential fit of each clean amplitude-normalised template tail
  Waveform_Features    histograms of every feature per category
  Waveform_Summary     medians (and 16/84 % quantiles for timing) / fractions per category
  Waveform_Template_Summary  the same pulse-shape parameters measured on each average template
  Waveform_Table_Slim  per-waveform features, random subset (<= SLIM_ROWS per source)
  Waveform_Examples    a few raw peak-aligned waveforms per category

Categories: Source (Truth/Matched) x Signal (flash Pur > 0 / == 0) x MaxPE bin x Plane, and
for Matched also the generator of the cluster.

Run
---
  python3 src/physics/detector/pds/06_waveforms.py --config vd_1x8x14_3view_30deg_nominal --name marley_waveform
  python3 src/physics/detector/pds/06_waveforms.py --config vd_1x8x14_3view_30deg_shielded --name radiological_flash --sources Matched
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
import awkward as ak
import uproot
from scipy.optimize import curve_fit

save_path = f"{root}/output/images/PDS/waveform"
data_path = f"{root}/output/data/PDS/waveform"
for path in [save_path, data_path]:
    os.makedirs(path, exist_ok=True)

parser = argparse.ArgumentParser(description="PDS waveform characterisation")
parser.add_argument("--config", type=str, default="vd_1x8x14_3view_30deg_nominal")
parser.add_argument("--name", type=str, default="marley_waveform")
parser.add_argument("--sources", nargs="+", default=["Truth", "Matched"], choices=["Truth", "Matched"])
parser.add_argument("--bkg_fraction", type=float, default=0.05,
                    help="Fraction of background (Pur == 0) truth flashes kept; signal flashes are all kept")
parser.add_argument("--truth_step", type=int, default=5, help="Truth events per chunk")
parser.add_argument("--reco_step", type=int, default=20000, help="Clusters per chunk")
parser.add_argument("--max_entries", type=int, default=None, help="Stop after this many entries per tree (testing)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
config, name = args.config, args.name

TICK = 0.016  # us; ClockSpeedOptical 62.5 MHz for dunefd / dunefdvd (dunecore detectorclocks_dune.fcl)
# Per geometry: saturation, baseline samples, trigger position in a single readout window,
# longest single (not retriggered) window, and nominal pedestal (duneopdet fcl, v10_08_01).
DIGITIZERS = {
    "vd": {"ADC_MAX": 2**13 - 1, "BASELINE_TICKS": 15, "TRIGGER_TICK": 20, "SINGLE_WINDOW": 340, "PEDESTAL": 100},
    "hd": {"ADC_MAX": 2**14 - 1, "BASELINE_TICKS": 80, "TRIGGER_TICK": 100, "SINGLE_WINDOW": 1000, "PEDESTAL": 1500},
}
PRE, POST = 40, 500    # template window around the peak [ticks]: -0.64 us .. +8.0 us (VD single windows end earlier: NaN)
INTEGRATION = 300      # charge-containment window from the onset [ticks] = 4.8 us, recorded by both detectors
SECONDARY_RATIO = 0.3
PE_BINS = [1.5, 5, 20, 100, 1000, np.inf]
PE_LABELS = ["1.5-5", "5-20", "20-100", "100-1000", ">1000", "All"]
SLIM_ROWS = 20_000
MIN_GENERATOR_N = 500  # per-generator categories with fewer clean waveforms are not saved
N_EXAMPLES = 8
rng = np.random.default_rng(42)

info, params, output = get_param_dict(f"{root}/config/{config}/{config}", {}, "", debug=args.debug)
filename = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}.root'
if not os.path.isfile(filename):
    sys.exit(f"ERROR: {filename} not found")
planes = {int(k): v for k, v in info["OPFLASH_PLANES"].items()}
digitizer = DIGITIZERS[info["GEOMETRY"]]
ADC_MAX, BASELINE_TICKS = digitizer["ADC_MAX"], digitizer["BASELINE_TICKS"]
TRIGGER_TICK, SINGLE_WINDOW = digitizer["TRIGGER_TICK"], digitizer["SINGLE_WINDOW"]
generators = list(json.load(open(f"{root}/config/import/generator_order.json"))[info["GEOMETRY"]][info["VERSION"]].keys())

f = uproot.open(filename)
folder = [k.split(";")[0] for k, c in f.classnames().items() if c == "TDirectory"][0]
n_inputs = f[f"{folder}/ConfigTree"].num_entries
# MatchedOpFlashTime has OpFlashTimeOffset subtracted (18.1 us in HD, 0 in VD); waveform
# timestamps do not.
config_tree = f[f"{folder}/ConfigTree"]
time_offset = float(config_tree["OpFlashTimeOffset"].array(library="np")[0]) if "OpFlashTimeOffset" in config_tree.keys() else 0.0
base_row = {"Geometry": info["GEOMETRY"], "Config": config, "Name": name, "NInputs": n_inputs}
offsets_rel = np.arange(-PRE, POST)
time_axis = offsets_rel * TICK


# ── Waveform processing ──────────────────────────────────────────────────────
def gather(content, starts, lengths, first, n):
    """Samples [first, first + n) of each waveform as a 2D float array, NaN outside the waveform."""
    local = first[:, None] + np.arange(n)[None, :]
    inside = (local >= 0) & (local < lengths[:, None])
    out = content[starts[:, None] + np.clip(local, 0, np.maximum(lengths[:, None] - 1, 0))].astype(np.float32)
    out[~inside] = np.nan
    return out


def process(waveforms, expected, search_lo, search_hi):
    """Baseline, peak and peak-aligned window for a jagged array of waveforms.

    expected: approximate pulse position (tick) per waveform; the peak is searched in
    [expected + search_lo, expected + search_hi)."""
    lengths = ak.to_numpy(ak.num(waveforms)).astype(np.int64)
    content = ak.to_numpy(ak.flatten(waveforms)).astype(np.int32)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    # Mean of the pre-trigger samples, excluding single-PE spikes (~9 ADC above the median):
    # the median of integer ADC values alone biases the baseline by up to 0.5 ADC, which is
    # 5 % of a single-PE amplitude and shows up as a negative template tail.
    base_win = gather(content, starts, lengths, np.zeros(len(lengths), dtype=np.int64), BASELINE_TICKS)
    median = np.nanmedian(base_win, axis=1)
    quiet = np.where(base_win <= median[:, None] + 5, base_win, np.nan)
    baseline = np.nanmean(quiet, axis=1)
    baseline_rms = np.nanstd(quiet, axis=1)

    search = gather(content, starts, lengths, expected + search_lo, search_hi - search_lo) - baseline[:, None]
    has_search = np.any(np.isfinite(search), axis=1)
    peak = expected + search_lo + np.nanargmax(np.where(np.isfinite(search), search, -np.inf), axis=1)

    window = gather(content, starts, lengths, peak - PRE, PRE + POST)
    raw_max = np.nanmax(window, axis=1)
    window = window - baseline[:, None]
    # Samples before the waveform start are NaN; a few pre-peak samples are enough for the onset
    return window, baseline, baseline_rms, raw_max, has_search & (peak >= 8)


def pulse_shape(norm):
    """Timing of amplitude-normalised, peak-aligned pulses (rows; the peak is at column PRE).

    Onset: 10 % crossing of the leading edge (between the pre-peak minimum and the peak), linearly interpolated between the last sample
    below 10 % and the next one. PeakTime = peak - onset. The charge is accumulated from the
    onset over a fixed INTEGRATION window (4.8 us, so that a small baseline error or late
    uncorrelated single-PE pulses cannot grow without bound); its maximum inside that window
    is the charge of the positive lobe (for VD the whole pulse; the HD SPE has an undershoot,
    after which the cumulative charge decreases). T50/T90/T99 are the times from the onset at
    which 50/90/99 % of that charge is reached. ChargeContained is False when the waveform
    does not cover the whole integration window (VD single windows, pulses near the end)."""
    n, m = norm.shape
    # Levels are taken above the lowest point before the peak (clipped at 0), so a pulse
    # sitting on the tail of an earlier one (common in averaged high-PE templates) still has
    # a defined leading edge.
    floor = np.clip(np.nanmin(np.where(np.isfinite(norm[:, :PRE + 1]), norm[:, :PRE + 1], np.inf), axis=1), 0, 0.9)
    floor = np.where(np.isfinite(floor), floor, 0.0)
    level10 = floor + 0.1 * (1 - floor)
    level90 = floor + 0.9 * (1 - floor)
    rising = norm[:, :PRE + 1][:, ::-1]
    below10, below90 = rising < level10[:, None], rising < level90[:, None]
    i10 = np.where(below10.any(axis=1), PRE - np.argmax(below10, axis=1), -1)
    i90 = np.where(below90.any(axis=1), PRE - np.argmax(below90, axis=1), -1)
    ok = i10 >= 0
    rows = np.arange(n)
    lo = norm[rows, np.clip(i10, 0, m - 1)]
    hi = norm[rows, np.clip(i10 + 1, 0, m - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        onset = np.where(ok, i10 + np.clip((level10 - lo) / (hi - lo), 0, 1), np.nan)
    rise = np.where(ok & (i90 >= 0), (i90 - i10), np.nan)

    cols = np.arange(m)[None, :]
    start = np.where(ok, np.floor(onset), PRE).astype(int)[:, None]
    finite = np.isfinite(norm)
    in_window = (cols >= start) & (cols < start + INTEGRATION)
    charge = np.where(in_window & finite, norm, 0.0)
    cum = np.cumsum(charge, axis=1)
    qmax = cum.max(axis=1)
    contained = ok & (start[:, 0] + INTEGRATION <= m) & np.all(finite | ~in_window, axis=1)
    out = {}
    for frac in [50, 90, 99]:
        reach = cum >= (frac / 100) * qmax[:, None]
        idx = np.where(reach.any(axis=1), np.argmax(reach, axis=1), np.nan)
        out[f"T{frac}"] = (idx - onset) * TICK

    above_half = norm >= 0.5
    first_half = np.argmax(above_half, axis=1)
    after = above_half[:, PRE:]
    last_half = PRE + np.where((~after).any(axis=1), np.argmax(~after, axis=1), m - PRE)
    tail = np.where(finite[:, PRE:], norm[:, PRE:], np.inf)
    under_min = tail.min(axis=1)
    under_t = tail.argmin(axis=1)
    return {
        "PeakTime": (PRE - onset) * TICK,
        "RiseTime": rise * TICK,
        **out,
        "FWHM": (last_half - first_half) * TICK,
        "Undershoot": np.where(np.isfinite(under_min), np.minimum(under_min, 0), np.nan),
        "UndershootTime": np.where(under_min < 0, under_t * TICK, np.nan),
        "ChargeContained": contained,
        "LobeCharge": qmax * TICK,  # us, in units of the amplitude
    }


def features(window, baseline, baseline_rms, raw_max):
    amp = window[:, PRE]
    safe_amp = np.where(amp > 0, amp, np.nan)
    integral = np.nansum(window[:, PRE - 5:PRE + INTEGRATION], axis=1)
    prompt = np.nansum(window[:, PRE - 3:PRE + 7], axis=1)
    late = np.nansum(window[:, PRE + int(1.0 / TICK):PRE + INTEGRATION], axis=1)
    safe_int = np.where(integral > 0, integral, np.nan)
    norm = window / safe_amp[:, None]

    falling = norm[:, PRE:]
    below_half = falling < 0.5
    below_e = falling < np.exp(-1)
    t_half = np.where(below_half.any(axis=1), np.argmax(below_half, axis=1), np.nan)
    t_e = np.where(below_e.any(axis=1), np.argmax(below_e, axis=1), np.nan)
    # Secondary pulse: a later pulse rising above the running minimum of the tail by more than
    # PILEUP_RATIO of the amplitude (or 4 sigma of the baseline noise for small pulses).
    tail = norm[:, PRE:]
    running_min = np.fmin.accumulate(np.where(np.isfinite(tail), tail, np.inf), axis=1)
    rise = np.where(np.isfinite(tail), tail - running_min, -np.inf)
    threshold = np.maximum(SECONDARY_RATIO, 4 * baseline_rms / safe_amp)
    secondary = np.nanmax(rise[:, 3:], axis=1) > threshold
    return {
        "Baseline": baseline,
        "BaselineRMS": baseline_rms,
        "Amplitude": amp,
        "Integral": integral,
        "FPrompt": prompt / safe_int,
        "FLate": late / safe_int,
        **pulse_shape(norm),
        "SNR": amp / np.where(baseline_rms > 0, baseline_rms, np.nan),
        "FallHalfTime": t_half * TICK,
        "FallETime": t_e * TICK,
        "Saturated": raw_max >= ADC_MAX,
        "SecondaryPulse": secondary,
    }


FEATURE_BINS = {
    "Baseline": np.linspace(digitizer["PEDESTAL"] - 20, digitizer["PEDESTAL"] + 30, 101),
    "BaselineRMS": np.linspace(0, 10, 101),
    "Amplitude": np.logspace(0, 4, 81),
    "Integral": np.logspace(0, 6, 121),
    "FPrompt": np.linspace(0, 1, 101),
    "FLate": np.linspace(0, 1, 101),
    "RiseTime": np.arange(-0.008, 0.33, TICK),
    "PeakTime": np.arange(0, 0.5, TICK / 4),
    "T50": np.arange(0, 4.8 + TICK, TICK),
    "T90": np.arange(0, 4.8 + TICK, TICK),
    "T99": np.arange(0, 4.8 + TICK, TICK),
    "FWHM": np.arange(-0.008, 2, TICK),
    "Undershoot": np.linspace(-0.6, 0, 61),
    "LobeCharge": np.logspace(-2, 1, 61),
    "SNR": np.logspace(0, 4, 81),
    "FallHalfTime": np.arange(-0.008, 4.5, TICK),
    "FallETime": np.arange(-0.008, 4.5, TICK),
    "ADCperPE": np.linspace(0, 20, 101),
    "IntegralPerPE": np.logspace(0, 4, 81),
}


FLAGS = ["Saturated", "SecondaryPulse", "ChargeContained"]  # Saturated is over all waveforms, the rest over clean ones
CONTAINMENT_FEATURES = ["T50", "T90", "T99", "LobeCharge"]
QUANTILE_FEATURES = ["PeakTime", "RiseTime", "T50", "T90", "T99", "FWHM", "Undershoot"]


class Accumulator:
    """Per-category template sums, feature histograms, summary values and examples."""

    def __init__(self):
        self.templates = {}   # key -> [sum_a, sumsq_a, n_a, sum_i, sumsq_i, n_i]
        self.hists = {}
        self.values = {}      # key -> dict of lists for the summary
        self.examples = {}

    def add(self, key, window, feats, clean):
        amp = feats["Amplitude"][:, None]
        integ = feats["Integral"][:, None]
        for selection, mask in [("Clean", clean), ("All", np.ones(len(window), dtype=bool))]:
            if not np.any(mask):
                continue
            by_amp = window[mask] / amp[mask]
            by_int = window[mask] / (integ[mask] * TICK)  # per us, so templates integrate to ~1
            t = self.templates.setdefault((*key, selection), [np.zeros(PRE + POST) for _ in range(6)])
            for i, arr in enumerate([by_amp, by_int]):
                t[3 * i] += np.nansum(arr, axis=0)
                t[3 * i + 1] += np.nansum(arr**2, axis=0)
                t[3 * i + 2] += np.sum(np.isfinite(arr), axis=0)
        contained = clean & feats["ChargeContained"]
        for feat, edges in FEATURE_BINS.items():
            # Charge-containment times only from pulses fully inside the readout window
            values = feats[feat][contained if feat in CONTAINMENT_FEATURES else clean]
            values = values[np.isfinite(values)]
            h = self.hists.setdefault((*key, feat), np.zeros(len(edges) - 1))
            h += np.histogram(values, bins=edges)[0]
        v = self.values.setdefault(key, {k: [] for k in list(FEATURE_BINS) + FLAGS})
        for k in v:
            mask = contained if k in CONTAINMENT_FEATURES else clean
            v[k].append(np.asarray(feats[k])[mask] if k not in FLAGS
                        else np.asarray(feats[k]) if k == "Saturated" else np.asarray(feats[k])[clean])
        ex = self.examples.setdefault(key, [])
        if len(ex) < N_EXAMPLES:
            for i in np.flatnonzero(clean)[: N_EXAMPLES - len(ex)]:
                ex.append(window[i])

    def template_rows(self, labels):
        rows = []
        for key, (sa, qa, na, si, qi, ni) in self.templates.items():
            for norm, s, q, n in [("Amplitude", sa, qa, na), ("Integral", si, qi, ni)]:
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = s / n
                    std = np.sqrt(np.maximum(q / n - mean**2, 0))
                rows.append({**base_row, **dict(zip(labels + ["Selection"], key)), "Norm": norm,
                             "Time": time_axis, "Mean": mean, "STD": std, "SEM": std / np.sqrt(np.maximum(n, 1)),
                             "N": int(n[PRE])})
        return rows

    def hist_rows(self, labels):
        rows = []
        for key, hist in self.hists.items():
            feat = key[-1]
            edges = FEATURE_BINS[feat]
            total = hist.sum()
            rows.append({**base_row, **dict(zip(labels + ["Variable"], key)), "Values": (edges[1:] + edges[:-1]) / 2,
                         "Edges": edges, "Counts": hist, "CountsError": np.sqrt(hist),
                         "Density": hist / (total * np.diff(edges)) if total > 0 else hist, "Entries": int(total)})
        return rows

    def summary_rows(self, labels):
        rows = []
        for key, v in self.values.items():
            row = {**base_row, **dict(zip(labels, key))}
            flags = {k: np.concatenate(v[k]) for k in FLAGS}
            row["NWaveforms"] = int(len(flags["SecondaryPulse"]))
            for k in FLAGS:
                row[f"{k}Fraction"] = float(np.mean(flags[k])) if row["NWaveforms"] else np.nan
            row["NClean"] = int(sum(len(a) for a in v["Amplitude"]))
            for feat in FEATURE_BINS:
                x = np.concatenate(v[feat]) if v[feat] else np.array([])
                x = x[np.isfinite(x)]
                row[f"{feat}Median"] = float(np.median(x)) if len(x) else np.nan
                row[f"{feat}Mean"] = float(np.mean(x)) if len(x) else np.nan
                row[f"{feat}STD"] = float(np.std(x)) if len(x) else np.nan
                if feat in QUANTILE_FEATURES:
                    for q in [16, 84]:
                        row[f"{feat}P{q}"] = float(np.percentile(x, q)) if len(x) else np.nan
            rows.append(row)
        return rows

    def example_rows(self, labels):
        return [{**base_row, **dict(zip(labels, key)), "Time": time_axis, "Waveforms": np.asarray(ex)}
                for key, ex in self.examples.items() if len(ex)]


def pe_bin(max_pe):
    return np.digitize(max_pe, PE_BINS) - 1  # -1 below 1.5 PE, len(PE_BINS)-1 above the last edge


def fill_categories(acc, window, feats, keys_base, pe_idx, plane_names, breakdown=True):
    """keys_base: list of per-waveform label tuples (without PE/plane). Without breakdown only
    the PE "All" / plane "Total" category is filled (used for the per-generator split)."""
    # Secondary pulses are not rejected: with Xe doping a low-light flash is a train of delayed
    # single-PE pulses on the main channel, so they are part of the scintillation profile.
    clean = (~feats["Saturated"]) & (feats["Amplitude"] > 0)
    keys = np.asarray(["|".join(map(str, k)) for k in keys_base])
    for base in np.unique(keys):
        in_base = keys == base
        base_key = tuple(base.split("|"))
        for b, label in enumerate(PE_LABELS if breakdown else ["All"]):
            in_pe = in_base & ((pe_idx == b) if label != "All" else (pe_idx >= 0))
            for plane_name in ["Total"] + (sorted(set(plane_names)) if breakdown else []):
                m = in_pe & ((plane_names == plane_name) if plane_name != "Total" else True)
                if np.any(m):
                    acc.add((*base_key, label, plane_name), window[m], {k: v[m] for k, v in feats.items()}, clean[m])


def add_ratios(feats, max_pe):
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["ADCperPE"] = feats["Amplitude"] / max_pe
        feats["IntegralPerPE"] = feats["Integral"] / max_pe
    return feats


def slim_frame(tables, source):
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)
    if len(df) > SLIM_ROWS:
        df = df.iloc[np.sort(rng.choice(len(df), SLIM_ROWS, replace=False))].reset_index(drop=True)
    df[df.select_dtypes("float64").columns] = df.select_dtypes("float64").astype(np.float32)
    df.insert(0, "Source", source)
    for k, v in base_row.items():
        df.insert(0, k, v)
    return df


# ── Truth: all flashes of the event ──────────────────────────────────────────
results = {}
truth_tree = f[f"{folder}/MCTruthTree"]
reco_tree = f[f"{folder}/SolarNuAnaTree"]
if "Truth" in args.sources and "OpFlashWaveform" in truth_tree.keys():
    acc, tables = Accumulator(), []
    n_valid = n_used = 0
    stop = truth_tree.num_entries if args.max_entries is None else min(args.max_entries, truth_tree.num_entries)
    for chunk in truth_tree.iterate(["OpFlashWaveform", "OpFlashWaveformValid", "OpFlashMaxPE", "OpFlashPur", "OpFlashPlane"],
                                    step_size=args.truth_step, entry_stop=stop, library="ak"):
        valid = ak.flatten(chunk["OpFlashWaveformValid"])
        wf = ak.flatten(chunk["OpFlashWaveform"], axis=1)
        pur = ak.to_numpy(ak.flatten(chunk["OpFlashPur"]))
        max_pe = ak.to_numpy(ak.flatten(chunk["OpFlashMaxPE"]))
        plane = ak.to_numpy(ak.flatten(chunk["OpFlashPlane"]))
        lengths = ak.to_numpy(ak.num(wf))
        keep = ak.to_numpy(valid) & (lengths >= TRIGGER_TICK + 100) & (lengths <= SINGLE_WINDOW)
        keep &= (pur > 0) | (rng.random(len(pur)) < args.bkg_fraction)
        n_valid += int(ak.sum(valid))
        if not np.any(keep):
            continue
        idx = np.flatnonzero(keep)
        window, baseline, rms, raw_max, ok = process(wf[idx], np.full(len(idx), TRIGGER_TICK), -5, 40)
        idx, window, baseline, rms, raw_max = idx[ok], window[ok], baseline[ok], rms[ok], raw_max[ok]
        n_used += len(idx)
        feats = add_ratios(features(window, baseline, rms, raw_max), max_pe[idx])
        signal = np.where(pur[idx] > 0, "Signal", "Background")
        plane_names = np.asarray([planes.get(p, "Unknown") for p in plane[idx]])
        fill_categories(acc, window, feats, [("Truth", s) for s in signal], pe_bin(max_pe[idx]), plane_names)
        tables.append(pd.DataFrame({"Signal": signal, "Plane": plane_names, "MaxPE": max_pe[idx], "Pur": pur[idx],
                                    **{k: v for k, v in feats.items()}}))
        rprint(f"  truth: {n_used} waveforms used / {n_valid} valid")
    results["Truth"] = (acc, ["Source", "Signal", "PEBin", "Plane"], slim_frame(tables, "Truth"))
elif "Truth" in args.sources:
    rprint(f"[yellow]{config} {name}: no OpFlashWaveform in MCTruthTree, skipping Truth[/yellow]")

# ── Matched: the flash matched to each cluster ───────────────────────────────
if "Matched" in args.sources and "MatchedOpFlashWaveform" in reco_tree.keys():
    acc, tables = Accumulator(), []
    n_valid = n_used = 0
    stop = reco_tree.num_entries if args.max_entries is None else min(args.max_entries, reco_tree.num_entries)
    for chunk in reco_tree.iterate(["MatchedOpFlashWaveform", "MatchedOpFlashWaveformValid", "MatchedOpFlashWaveformTime",
                                    "MatchedOpFlashTime", "MatchedOpFlashMaxPE", "MatchedOpFlashPur", "MatchedOpFlashPlane",
                                    "Generator"], step_size=args.reco_step, entry_stop=stop, library="ak"):
        lengths = ak.to_numpy(ak.num(chunk["MatchedOpFlashWaveform"]))
        valid = ak.to_numpy(chunk["MatchedOpFlashWaveformValid"]) & (lengths > 0)
        n_valid += int(valid.sum())
        if not np.any(valid):
            continue
        idx = np.flatnonzero(valid)
        c = chunk[idx]
        expected = np.round((ak.to_numpy(c["MatchedOpFlashTime"]) + time_offset
                             - ak.to_numpy(c["MatchedOpFlashWaveformTime"])) / TICK).astype(np.int64)
        # The flash time is the PE-weighted mean of its OpHit times, not the main hit's peak time;
        # on single-window waveforms the main pulse sits between -30 and +70 ticks from it.
        window, baseline, rms, raw_max, ok = process(c["MatchedOpFlashWaveform"], expected, -40, 120)
        c, window, baseline, rms, raw_max = c[ok], window[ok], baseline[ok], rms[ok], raw_max[ok]
        n_used += len(window)
        max_pe = ak.to_numpy(c["MatchedOpFlashMaxPE"])
        pur = ak.to_numpy(c["MatchedOpFlashPur"])
        gen = ak.to_numpy(c["Generator"])
        feats = add_ratios(features(window, baseline, rms, raw_max), max_pe)
        signal = np.where(pur > 0, "Signal", "Background")
        gen_names = np.asarray([generators[g] if 0 <= g < len(generators) else "Unknown" for g in gen])
        plane_names = np.asarray([planes.get(p, "Unknown") for p in ak.to_numpy(c["MatchedOpFlashPlane"])])
        fill_categories(acc, window, feats, [("Matched", s, "All") for s in signal], pe_bin(max_pe), plane_names)
        fill_categories(acc, window, feats, [("Matched", s, g) for s, g in zip(signal, gen_names)], pe_bin(max_pe), plane_names,
                        breakdown=False)
        tables.append(pd.DataFrame({"Signal": signal, "Generator": gen_names, "Plane": plane_names, "MaxPE": max_pe, "Pur": pur,
                                    **{k: v for k, v in feats.items()}}))
        rprint(f"  matched: {n_used} waveforms used / {n_valid} valid / {stop} clusters")
    if n_valid == 0:
        rprint(f"[yellow]{config} {name}: no valid matched waveforms (HD productions do not carry them)[/yellow]")
    else:
        results["Matched"] = (acc, ["Source", "Signal", "Generator", "PEBin", "Plane"], slim_frame(tables, "Matched"))

if not results:
    sys.exit(0)


# ── Template fits: two exponentials + flat level on the tail from 2 ticks after the peak ──
def two_exp(t, a1, tau1, a2, tau2, c):
    # c absorbs the flat level of uncorrelated single-PE activity on the channel
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def fit_template(mean):
    t = time_axis[PRE + 2:]
    y = mean[PRE + 2:]
    ok = np.isfinite(y)
    if ok.sum() < 20:
        return {}
    try:
        popt, pcov = curve_fit(two_exp, t[ok], y[ok], p0=[0.8, 0.1, 0.2, 1.0, 0.0],
                               bounds=([0, 0.005, 0, 0.05, -0.5], [2, 1.0, 2, 20, 0.5]), maxfev=20000)
    except (RuntimeError, ValueError):
        return {}
    a1, tau1, a2, tau2, c = popt
    if tau1 > tau2:
        a1, tau1, a2, tau2 = a2, tau2, a1, tau1
    err = np.sqrt(np.diag(pcov))
    return {"A1": a1, "Tau1": tau1, "A2": a2, "Tau2": tau2, "Offset": c, "Tau1Error": err[1], "Tau2Error": err[3],
            "SlowFraction": a2 * tau2 / (a1 * tau1 + a2 * tau2), "FitRange": (t[0], t[-1])}


# ── Save ─────────────────────────────────────────────────────────────────────
frames = {"Templates": [], "Fits": [], "Features": [], "Summary": [], "Examples": [], "Table_Slim": []}
for source, (acc, labels, slim) in results.items():
    templates = acc.template_rows(labels)
    frames["Templates"] += templates
    for row in templates:
        if row["Norm"] == "Amplitude" and row["Selection"] == "Clean" and row["N"] >= 20:
            fit = fit_template(row["Mean"])
            if fit:
                frames["Fits"].append({**{k: row[k] for k in list(base_row) + labels + ["Selection", "N"]}, **fit})
    frames["Features"] += acc.hist_rows(labels)
    frames["Summary"] += acc.summary_rows(labels)
    frames["Examples"] += acc.example_rows(labels)
    if not slim.empty:
        frames["Table_Slim"].append(slim)

small_generators = {
    tuple(r[k] for k in ["Source", "Signal", "Generator"])
    for r in frames["Summary"] if r.get("Generator", "All") != "All" and r["NClean"] < MIN_GENERATOR_N
}
# Pulse-shape parameters of the average (clean, amplitude-normalised) templates themselves
template_shape = []
for row in frames["Templates"]:
    if row["Norm"] == "Amplitude" and row["Selection"] == "Clean" and row["N"] > 0:
        shape = pulse_shape(np.asarray(row["Mean"], dtype=float)[None, :])
        template_shape.append({k: v for k, v in row.items() if k not in ["Time", "Mean", "STD", "SEM", "Norm", "Selection"]}
                              | {k: (bool(v[0]) if k == "ChargeContained" else float(v[0])) for k, v in shape.items()})
frames["Template_Summary"] = template_shape

for kind, rows in frames.items():
    if kind != "Table_Slim":
        rows = [r for r in rows if tuple(r.get(k, "All") for k in ["Source", "Signal", "Generator"]) not in small_generators]
    df = pd.concat(rows, ignore_index=True) if kind == "Table_Slim" and rows else pd.DataFrame(rows)
    save_df(df, data_path, config, name, filename=f"Waveform_{kind}", rm=args.rewrite, debug=args.debug)

summary = pd.DataFrame(frames["Summary"])
cols = [c for c in ["Source", "Signal", "Generator", "NWaveforms", "NClean", "FPromptMedian", "FLateMedian",
                    "RiseTimeMedian", "FallETimeMedian", "ADCperPEMedian", "SecondaryPulseFraction", "SaturatedFraction"] if c in summary]
this = summary[(summary["PEBin"] == "All") & (summary["Plane"] == "Total")][cols]
rprint(this.to_string())
fits = pd.DataFrame(frames["Fits"])
if not fits.empty:
    rprint(fits[(fits["PEBin"] == "All") & (fits["Plane"] == "Total")][
        [c for c in ["Source", "Signal", "Generator", "N", "Tau1", "Tau2", "SlowFraction"] if c in fits]].to_string())

# ── Figure: clean amplitude-normalised templates, all PE, all planes ─────────
templates = pd.DataFrame(frames["Templates"])
sel = templates[(templates["Norm"] == "Amplitude") & (templates["Selection"] == "Clean")
                & (templates["PEBin"] == "All") & (templates["Plane"] == "Total")]
fig = make_subplots(rows=1, cols=1)
for idx, (_, row) in enumerate(sel.iterrows()):
    label = " ".join(str(row[k]) for k in ["Source", "Signal", "Generator"] if k in row and isinstance(row[k], str))
    fig.add_trace(go.Scatter(x=row["Time"], y=row["Mean"], mode="lines", name=f"{label} ({row['N']})",
                             line=dict(color=default[idx % len(default)], width=2)))
fig = format_coustom_plotly(fig, title=f"Waveform templates - {config} {name}", log=(False, True), legend_title="Category")
fig.update_xaxes(title_text="Time from peak (us)")
fig.update_yaxes(title_text="Normalised amplitude", range=[-3, 0.1])
save_figure(fig, save_path, config, name, filename="Waveform_Templates", rm=args.rewrite, debug=args.debug)
