"""
02_background_spectra.py — Background Truth Spectra
====================================================
Pipeline step 0Y (first): loads truth-level gamma/neutron flux spectra from
surface-activity ROOT files, applies shielding corrections, rebins, and saves
validated spectra as pkl for downstream PDF computation.

Outputs (per-config)
--------------------
  {data_path}/{config}/spectra/{config}_background_spectra.pkl
    dict keyed by (surface_id, particle_type, particle_origin) →
    (x_mev: ndarray, y_flux: ndarray)  [counts MeV⁻¹ s⁻¹ cm⁻²]
    Shielding corrections from SPECTRA.SHIELDING already applied.

Outputs (aggregate)
-------------------
  {data_path}/background_spectra_summary.pkl
        DataFrame with one row per spectrum and both legacy plus readable columns:
        config, surface_id, particle_type, particle_origin, area_cm2,
        x_mev, y_flux, Config, Geometry, Component, Particle, ParticleOrigin,
        Energy, Counts / Flux, Flux, SpectrumType.
        Surface rows keep the per-surface truth spectra; combined rows are added
        with Component=None and surface_id=None so downstream scanners can read a
        single table without losing the combined spectra.

  {figure_path}/shielding_comparison_{corrected_cfg}_{origin}.png
    For each SPECTRA.SHIELDING_COMPARISONS entry: reference vs. uncorrected
    vs. corrected spectra (area-weighted, summed over all surfaces).

  {figure_path}/all_productions_combined_by_type.png
    Side-by-side gamma/neutron combined spectra across COMPARISON_CONFIGS.

Run
---
  python3 02_background_spectra.py [--no-rewrite] [--no-debug] [--no-plot]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

# ── Paths ───────────────────────────────────────────────────────────────────
figure_path = f"{root}/images/background"
data_path   = f"{root}/data/background"
for _p in [figure_path, data_path]:
    os.makedirs(_p, exist_ok=True)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compute and validate truth-level background spectra"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# ── Load analysis config ─────────────────────────────────────────────────────
analysis_info  = load_analysis_info(str(root))
spectra_cfg    = analysis_info.get("SPECTRA", {})
truth_pipeline = analysis_info.get("TRUTH_PIPELINE", {})
rebin_width    = int(spectra_cfg.get("REBIN_WIDTH", 100))
shielding_cfg  = spectra_cfg.get("SHIELDING", {})
truth_path     = analysis_info.get("DATA_PATHS", {}).get(
    "TRUTH", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth"
)
configs_to_run = analysis_info.get("DEFAULT_CONFIGS", [])
names          = truth_pipeline.get("BACKGROUND_NAMES", ["gamma", "neutron"])

# ── Static import data ───────────────────────────────────────────────────────
surface_names = json.load(open(f"{root}/import/surfaces.json"))
areas         = json.load(open(f"{root}/import/surface_areas.json"))
for geo, surfs in areas.items():
    for sid, expr in surfs.items():
        areas[geo][sid] = eval(expr)

activity_files = json.load(open(f"{root}/import/surface_activity.json"))

if args.debug:
    rprint(f"[cyan][INFO][/cyan] Configs: {configs_to_run}")
    rprint(f"[cyan][INFO][/cyan] Particle types: {names}")
    rprint(f"[cyan][INFO][/cyan] Rebin width: {rebin_width} bins")
    rprint(f"[cyan][INFO][/cyan] Truth ROOT path: {truth_path}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def rebin_spectrum(x, y, bin_width):
    """Sum y into blocks of bin_width, keep first x of each block."""
    x_out, y_out = [], []
    for i in range(0, len(x), bin_width):
        x_out.append(x[i])
        y_out.append(np.sum(y[i : i + bin_width]))
    return np.array(x_out), np.array(y_out)


def apply_shielding(config, particle_origin, x_mev, y, cfg, debug=False):
    """
    Apply per-config/origin shielding reduction from SPECTRA.SHIELDING.
    Returns y (unchanged) if no shielding entry exists for this config/origin.
    """
    entry = cfg.get(config, {}).get(particle_origin)
    if entry is None:
        return np.array(y)
    factor    = entry["reduction_factor"]
    threshold = entry["threshold_mev"]
    if debug:
        rprint(
            f"[cyan][SHIELDING][/cyan] {config}/{particle_origin}: "
            f"×{factor:.6f} above {threshold} MeV"
        )
    y_out = np.array(y, dtype=float)
    y_out[np.asarray(x_mev) >= threshold] *= factor
    return y_out


def load_spectrum_from_root(file_path, particle_type):
    """Open ROOT TGraph, return (x_kev, y_raw) as numpy arrays."""
    rf    = TFile.Open(file_path)
    graph = rf.Get("Gammas" if particle_type == "gamma" else "Neutrons")
    n     = graph.GetN()
    x     = np.array([graph.GetX()[i] for i in range(n)])
    y     = np.array([graph.GetY()[i] for i in range(n)])
    return x, y


def clean_component_name(particle_origin):
    """Drop the trailing particle suffix from a particle-origin label."""
    parts = str(particle_origin).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(particle_origin)


# ── Main loop: load and process spectra ──────────────────────────────────────
# all_plots_raw: shielding-uncorrected spectra (retained for comparison plots only)
# all_plots:     shielding-corrected spectra (written to pkl; used by all downstream steps)
all_plots_raw = {}
all_plots     = {}

for config in configs_to_run:
    if config not in activity_files:
        rprint(
            f"[yellow][WARNING][/yellow] {config} not in surface_activity.json — skip"
        )
        continue

    geometry = config.split("_")[0].lower()
    all_plots_raw[config] = {}
    all_plots[config]     = {}

    for surface_id, origins in activity_files[config].items():
        if config == "hd_1x2x6_centralAPA" and surface_id == "0":
            continue  # no data for this combination

        for particle_origin, (this_file, weight) in origins.items():
            particle_type = particle_origin.lower().split("_")[-1]
            if particle_type not in names:
                continue

            file_path = f"{truth_path}/{this_file}.root"
            if args.debug:
                rprint(f"  Loading {config}/{surface_id}/{particle_origin}")

            x_kev, y_raw = load_spectrum_from_root(file_path, particle_type)

            if np.sum(y_raw) == 0:
                rprint(
                    f"[yellow][WARNING][/yellow] Zero-sum spectrum for "
                    f"{config}/{surface_id}/{particle_origin} — skip"
                )
                continue

            # Normalize to weight (fraction of total activity)
            y_norm = y_raw * weight / np.sum(y_raw)

            # Rebin to reduce resolution
            x_rb, y_rb = rebin_spectrum(x_kev, y_norm, rebin_width)

            if len(x_rb) < 2:
                rprint(
                    f"[yellow][WARNING][/yellow] Too few bins after rebin for "
                    f"{config}/{surface_id}/{particle_origin} — skip"
                )
                continue

            # Convert: x keV→MeV,  y → counts MeV⁻¹ s⁻¹ cm⁻² (density)
            x_mev  = 1e-3 * x_rb
            dx_mev = np.diff(x_mev)[0]
            y_flux = 1e3 * y_rb / dx_mev  # /keV → /MeV factor

            key = (surface_id, particle_type, particle_origin)
            # Raw (uncorrected) — kept only for shielding comparison plots
            all_plots_raw[config][key] = (x_mev, y_flux)
            # Corrected — apply shielding from JSON; this is what goes to pkl
            y_corrected = apply_shielding(
                config, particle_origin, x_mev, y_flux, shielding_cfg, debug=args.debug
            )
            all_plots[config][key] = (x_mev, y_corrected)

    rprint(
        f"[green][OK][/green] {config}: {len(all_plots[config])} spectrum entries loaded"
    )

    # ── Save corrected spectra pkl for downstream (03_background_pdf) ────────
    save_pkl(
        all_plots[config],
        data_path,
        config,
        None,
        subfolder="spectra",
        filename="background_spectra",
        rm=args.rewrite,
        debug=True,
    )


# ── Shielding correction comparison plots ────────────────────────────────────
# Driven by SPECTRA.SHIELDING_COMPARISONS in backgrounds.json.
# Each entry: {"corrected": <config>, "reference": <config>}
# For each affected origin in SPECTRA.SHIELDING[corrected_config]:
#   - Collect area-weighted sum over all surfaces (raw and corrected).
#   - Plot reference / before-correction / after-correction.
#   - Print correction summary: factor, threshold, flux before/after, retention.

if args.plot:
    _shielding_comparisons = spectra_cfg.get("SHIELDING_COMPARISONS", [])
    for _comp in _shielding_comparisons:
        _corr_cfg = _comp.get("corrected")
        _ref_cfg  = _comp.get("reference")
        if _corr_cfg not in all_plots:
            rprint(
                f"[yellow][WARNING][/yellow] Shielding comparison: "
                f"'{_corr_cfg}' not loaded — skip"
            )
            continue
        _shield_entries = shielding_cfg.get(_corr_cfg, {})
        if not _shield_entries:
            continue

        _geo_corr = _corr_cfg.split("_")[0].lower()
        _geo_ref  = _ref_cfg.split("_")[0].lower() if _ref_cfg else None

        for _p_origin, _entry in _shield_entries.items():
            _factor    = _entry["reduction_factor"]
            _threshold = _entry["threshold_mev"]

            # ── Area-weighted sum over all surfaces for one origin ────────────
            def _collect_combined(plots_dict, cfg, geo):
                _xo, _yo = None, None
                for (_sid, _pt, _po), (_xv, _yv) in plots_dict.get(cfg, {}).items():
                    if _po != _p_origin:
                        continue
                    _aw     = areas.get(geo, {}).get(_sid, 0.0)
                    _scaled = np.array(_yv) * _aw
                    if _yo is None:
                        _xo, _yo = np.array(_xv), _scaled.copy()
                    else:
                        _n = min(len(_yo), len(_scaled))
                        _yo, _xo = _yo[:_n] + _scaled[:_n], _xo[:_n]
                return _xo, _yo

            _x_raw, _y_raw = _collect_combined(all_plots_raw, _corr_cfg, _geo_corr)
            _x_cor, _y_cor = _collect_combined(all_plots,     _corr_cfg, _geo_corr)
            _x_ref, _y_ref = (
                _collect_combined(all_plots, _ref_cfg, _geo_ref)
                if _ref_cfg and _ref_cfg in all_plots else (None, None)
            )

            if _y_raw is None or _y_cor is None:
                rprint(
                    f"[yellow][WARNING][/yellow] No spectra for "
                    f"{_corr_cfg}/{_p_origin} — skip comparison"
                )
                continue

            # ── Correction summary ───────────────────────────────────────────
            _thr_idx     = int(np.searchsorted(_x_raw, _threshold))
            _flux_before = float(np.trapezoid(_y_raw[_thr_idx:], _x_raw[_thr_idx:]))
            _flux_after  = float(np.trapezoid(_y_cor[_thr_idx:], _x_cor[_thr_idx:]))
            _retention   = _flux_after / _flux_before if _flux_before > 0 else float("nan")

            rprint(f"\n[bold]Shielding correction:[/bold] {_corr_cfg} / {_p_origin}")
            rprint(f"  Factor      : ×{_factor:.6f}")
            rprint(f"  Threshold   : {_threshold} MeV")
            rprint(f"  Flux before : {_flux_before:.4e}  (area-wt, above threshold)")
            rprint(f"  Flux after  : {_flux_after:.4e}")
            rprint(f"  Retention   : {_retention * 100:.2f}%")

            # ── Plot ─────────────────────────────────────────────────────────
            _fig_sh, _ax_sh = plt.subplots(figsize=(10, 6))
            if _ref_cfg and _y_ref is not None:
                _ax_sh.plot(
                    _x_ref, _y_ref,
                    label=f"Reference ({_ref_cfg.split('_')[0].upper()})",
                    color="C0", drawstyle="steps-mid",
                )
            _ax_sh.plot(
                _x_raw, _y_raw,
                label=f"Before correction ({_corr_cfg.split('_')[0].upper()})",
                color="k", linestyle="--", drawstyle="steps-mid",
            )
            _ax_sh.plot(
                _x_cor, _y_cor,
                label=f"After correction (×{_factor:.4f} > {_threshold} MeV)",
                color="C1", drawstyle="steps-mid",
            )
            _ax_sh.axvline(
                _threshold, color="gray", linestyle="dashed", alpha=0.7,
                label=f"Threshold {_threshold} MeV",
            )
            _ax_sh.set_yscale("log")
            _ax_sh.set_xlabel("Energy (MeV)")
            _ax_sh.set_ylabel("Area-weighted flux (counts · MeV⁻¹ · s⁻¹)")
            _ax_sh.set_title(f"Shielding correction — {_p_origin} — {_corr_cfg}")
            _ax_sh.legend(fontsize=8)
            _ax_sh.grid(True, which="both", ls="--", lw=0.5)
            _fig_sh.tight_layout()
            _sh_fname = f"shielding_comparison_{_corr_cfg}_{_p_origin}.png"
            _fig_sh.savefig(f"{figure_path}/{_sh_fname}", dpi=150)
            plt.close(_fig_sh)
            rprint(f"Saved: {figure_path}/{_sh_fname}")


# ── Build spectra summary DataFrame ──────────────────────────────────────────
# Replaces individual per-surface/per-origin plots (points 9–12).
# One row per (config, surface_id, particle_type, particle_origin);
# x_mev and y_flux stored as numpy arrays.

_spectra_records = []
for config in configs_to_run:
    if config not in all_plots:
        continue
    _geo = config.split("_")[0].lower()
    for (surface_id, particle_type, particle_origin), (x_mev, y_flux) in all_plots[config].items():
        _area = areas.get(_geo, {}).get(surface_id, 0.0)
        _spectra_records.append({
            "config":          config,
            "surface_id":      surface_id,
            "particle_type":   particle_type,
            "particle_origin": particle_origin,
            "area_cm2":        _area,
            "x_mev":           np.array(x_mev),
            "y_flux":          np.array(y_flux),
            "Config":          config,
            "Geometry":        _geo,
            "Component":       clean_component_name(particle_origin),
            "Particle":        particle_type,
            "ParticleOrigin":  particle_origin,
            "Energy":          np.array(x_mev),
            "Counts / Flux":   np.array(y_flux),
            "Flux":            np.array(y_flux),
        })
_spectra_df = pd.DataFrame(_spectra_records)
rprint(
    f"\n[bold]Spectra DataFrame:[/bold] {len(_spectra_df)} rows "
    f"({len(configs_to_run)} configs)"
)


# ── Combined spectra per config ───────────────────────────────────────────────
# combined_spectra[config][particle_type] = (y_combined, x_mev)
# Area-weighted sum over all origins/surfaces, normalised per kT.

_truth_pipeline     = analysis_info.get("TRUTH_PIPELINE", {})
_comparison_configs = _truth_pipeline.get("COMPARISON_CONFIGS", configs_to_run)
_flux_regions_mev   = _truth_pipeline.get("FLUX_REGIONS_MEV", [[0, 2.7], [2.7, 8], [8, 14]])
_flux_reference     = _truth_pipeline.get("FLUX_REFERENCE_CONFIG", "vd_1x8x14_3view_30deg_nominal")
_color_map = {
    "vd_1x8x14_3view_30deg_nominal":     "k",
    "vd_1x8x14_3view_30deg_shielded":    "C0",
    "vd_1x8x14_3view_30deg_optimistic":  "C2",
    "hd_1x2x6_centralAPA":               "C1",
    "hd_1x2x6_lateralAPA":               "C3",
}

combined_spectra = {}
for config in configs_to_run:
    if config not in all_plots:
        continue
    geometry      = config.split("_")[0].lower()
    _info         = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_workspace_detector_mass(config, _info)

    combined_spectra[config] = {}
    for (surface_id, particle_type, particle_origin), (x, y) in all_plots[config].items():
        if particle_type not in names:
            continue
        area_weight = areas.get(geometry, {}).get(surface_id, 0.0)
        y_scaled    = np.array(y) / detector_mass * area_weight  # counts/(MeV·s·kT)

        if particle_type not in combined_spectra[config]:
            combined_spectra[config][particle_type] = (y_scaled, np.array(x))
        else:
            prev_y, prev_x = combined_spectra[config][particle_type]
            n = min(len(prev_y), len(y_scaled))
            combined_spectra[config][particle_type] = (prev_y[:n] + y_scaled[:n], prev_x[:n])


# ── Build combined_df ─────────────────────────────────────────────────────────
_combined_records = []
for config, spectra in combined_spectra.items():
    for particle_type, (y_comb, x_comb) in spectra.items():
        _combined_records.append({
            "config":          config,
            "particle_type":   particle_type,
            "x_mev":           np.array(x_comb),
            "y_flux_per_s_kT": np.array(y_comb),
        })
_combined_df = pd.DataFrame(_combined_records)


# ── Build summary_df ──────────────────────────────────────────────────────────
_summary_records = []
for row in _spectra_records:
    _summary_records.append({**row, "SpectrumType": "surface"})
for row in _combined_records:
    _summary_records.append({
        "config":          row["config"],
        "surface_id":      None,
        "particle_type":   row["particle_type"],
        "particle_origin": None,
        "area_cm2":        None,
        "x_mev":           row["x_mev"],
        "y_flux":          row["y_flux_per_s_kT"],
        "Config":          row["config"],
        "Geometry":        row["config"].split("_")[0].lower(),
        "Component":       None,
        "Particle":        row["particle_type"],
        "ParticleOrigin":  None,
        "Energy":          row["x_mev"],
        "Counts / Flux":   row["y_flux_per_s_kT"],
        "Flux":            row["y_flux_per_s_kT"],
        "SpectrumType":    "combined",
    })
_summary_df = pd.DataFrame(_summary_records)


# ── Build flux_df (point 14: integrated flux table) ──────────────────────────
_flux_records = []
for config, spectra in combined_spectra.items():
    for particle_type, (y_comb, x_comb) in spectra.items():
        x_arr = np.array(x_comb)
        y_arr = np.array(y_comb) * 60 * 60 * 24 * 365.25  # → counts/(MeV·kT·yr)
        for low, high in _flux_regions_mev:
            mask = (x_arr >= low) & (x_arr <= high)
            flux = float(np.trapezoid(y_arr[mask], x_arr[mask])) if np.any(mask) else 0.0
            _flux_records.append({
                "config":         config,
                "particle":       particle_type,
                "region_mev":     f"{low}–{high}",
                "flux_per_yr_kT": flux,
            })

_flux_df = pd.DataFrame(_flux_records)
if not _flux_df.empty:
    _ref_rows = _flux_df[_flux_df["config"] == _flux_reference]
    _ref_idx  = _ref_rows.set_index(["particle", "region_mev"])["flux_per_yr_kT"]

    def _red_pct(row):
        key = (row["particle"], row["region_mev"])
        if row["config"] == _flux_reference or key not in _ref_idx.index:
            return float("nan")
        ref = _ref_idx[key]
        return (1 - row["flux_per_yr_kT"] / ref) * 100 if ref != 0 else float("nan")

    _flux_df["reduction_pct"] = _flux_df.apply(_red_pct, axis=1)

rprint("\n[bold]Integrated flux (counts · kT⁻¹ · yr⁻¹):[/bold]")
rprint(_flux_df.to_string(index=False))


# ── Save aggregate summary pkl ────────────────────────────────────────────────
save_pkl(
    _summary_df,
    data_path,
    None,
    None,
    filename="background_spectra_summary",
    rm=args.rewrite,
    debug=True,
)


# ── All-productions comparison plot (point 13) ───────────────────────────────
# Gamma and neutron side-by-side across COMPARISON_CONFIGS.
# Configs driven by TRUTH_PIPELINE.COMPARISON_CONFIGS in backgrounds.json.
if args.plot and combined_spectra:
    fig_ap, axes_ap = plt.subplots(1, 2, figsize=(16, 6))
    for col, particle_type in enumerate(["gamma", "neutron"]):
        ax = axes_ap[col]
        for config in _comparison_configs:
            if config not in combined_spectra:
                continue
            if particle_type not in combined_spectra[config]:
                continue
            y_comb, x_comb = combined_spectra[config][particle_type]
            ax.plot(
                x_comb,
                y_comb * 60 * 60 * 24 * 365.25,
                label=config,
                color=_color_map.get(config, "gray"),
                drawstyle="steps-post",
            )
        ax.set_xlim(0, 14)
        ax.set_ylim(1e0, 5e16)
        ax.set_yscale("log")
        ax.set_xlabel("True Energy (MeV)")
        ax.set_ylabel("Counts (MeV · kT · year)⁻¹")
        ax.set_title(f"Combined {particle_type.capitalize()} Spectra")
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    fig_ap.savefig(f"{figure_path}/all_productions_combined_by_type.png", dpi=150)
    plt.close(fig_ap)
    rprint(f"Saved: {figure_path}/all_productions_combined_by_type.png")


rprint("[bold green]02_background_spectra complete.[/bold green]")
