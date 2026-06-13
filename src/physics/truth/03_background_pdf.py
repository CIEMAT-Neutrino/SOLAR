"""
03_background_pdf.py — Background Momentum PDF Builder
=======================================================
Pipeline step 0Y: builds surface-resolved momentum PDFs for gamma/neutron
backgrounds.  Selects one of two backends:

  truth  (default) — derives PDFs from truth-level ROOT TGraph flux spectra.
                      Lightweight; no MC simulation data required.
                      Logic mirrors the former 04_external_background.py.

  legacy           — derives PDFs from reconstructed MC particle hits.
                      Requires MCParticle_* simulation output directories.
                      Logic mirrors the former 03_background_surface.py.

The default backend is read from TRUTH_PIPELINE.PDF_BACKEND in
analysis/backgrounds.json and can be overridden with --backend on the CLI.

Both backends write identical output formats:

  {DATA_PATHS.TRUTH (truth) | DATA_PATHS.LEGACY (legacy)}/{config}/
    {config}_{name}_pdf.pkl        — (hist_s, bins_s) per-surface normalised PDFs
                                     bins in MeV/c; hist dimensionless (integral=1)
    {config}_{name}_exposure.pkl   — dict[surface_id → exposure metadata]
      truth backend keys:
        counts [\mathrm{events \cdot kT^{-1} \cdot yr^{-1}}], exposure [\mathrm{kT \cdot yr}, normalized=1],
        detector_mass [\mathrm{kT}], detector_time [\mathrm{yr}, normalized=1]
        + *_unit companions (LaTeX math strings) for each numeric key
      legacy backend keys:
        counts [raw MC particle count], exposure [\mathrm{kT \cdot yr}],
        detector_mass [\mathrm{kT}], detector_time [\mathrm{yr}]
        + *_unit companions (LaTeX math strings) for each numeric key

  legacy backend only:
    {config}_{name}_histograms.pkl — raw {(surface_id, variable): (h, bins)}
                                     consumed by 02_background_spectra.py for plots

Configs and particle names are read from analysis/backgrounds.json
(DEFAULT_CONFIGS and TRUTH_PIPELINE.BACKGROUND_NAMES); no CLI args needed.

Run
---
  python3 03_background_pdf.py [--backend {truth,legacy}]
                               [--no-rewrite] [--no-debug] [--no-plot]
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

# ── Shared helpers ───────────────────────────────────────────────────────────
def _hist_median(h, bins):
    """Approximate median from a normalised histogram."""
    centers = (bins[:-1] + bins[1:]) / 2
    total   = np.sum(h)
    if total == 0 or len(centers) == 0:
        return float(centers[len(centers) // 2]) if len(centers) else 0.0
    cdf = np.cumsum(h) / total
    return float(np.interp(0.5, cdf, centers))


def compute_weights(p_sim, surface_id_sim, hist, bins, alpha_truth):
    """Evaluate PDF weights for a set of simulated particle momenta."""
    n_s = 0
    w   = np.zeros_like(p_sim, dtype=float)
    for s in np.unique(surface_id_sim):
        mask = surface_id_sim == s
        if s < 0 or s >= len(hist) or len(hist[s]) == 0:
            continue
        n_s    += 1
        p_s     = p_sim[mask]
        bin_idx = np.clip(np.digitize(p_s, bins[s]) - 1, 0, len(hist[s]) - 1)
        w[mask] = alpha_truth[s] * hist[s][bin_idx]
    return n_s * w


# ── CLI ──────────────────────────────────────────────────────────────────────
_analysis_info_pre = load_analysis_info(str(root))
_default_backend   = _analysis_info_pre.get("TRUTH_PIPELINE", {}).get("PDF_BACKEND", "truth")

parser = argparse.ArgumentParser(
    description="Build background momentum PDFs (truth or legacy MC backend)"
)
parser.add_argument(
    "--backend",
    type=str,
    choices=["truth", "legacy"],
    default=_default_backend,
    help=(
        "'truth': PDFs from ROOT TGraph flux spectra (default). "
        "'legacy': PDFs from reconstructed MC particle hits."
    ),
)
parser.add_argument(
    "--binwidth",
    type=float,
    default=None,
    help="Momentum bin width in MeV (default: auto)",
)
parser.add_argument(
    "--pdf_floor",
    type=float,
    default=None,
    help="Minimum PDF value to avoid zeros (default: PDF.FLOOR from backgrounds.json)",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# ── Load analysis config ─────────────────────────────────────────────────────
analysis_info  = load_analysis_info(str(root))
pdf_cfg        = analysis_info.get("PDF", {})
spectra_cfg    = analysis_info.get("SPECTRA", {})
truth_pipeline = analysis_info.get("TRUTH_PIPELINE", {})
shielding_cfg  = spectra_cfg.get("SHIELDING", {})

truth_path  = analysis_info.get("DATA_PATHS", {}).get(
    "TRUTH", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth"
)
legacy_path = analysis_info.get("DATA_PATHS", {}).get(
    "LEGACY", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/legacy"
)

pdf_floor        = args.pdf_floor if args.pdf_floor is not None else pdf_cfg.get("FLOOR", 1e-8)
binwidth_factor  = pdf_cfg.get("BINWIDTH_FACTOR", 0.02)
gamma_p_thresh   = pdf_cfg.get("GAMMA_MOMENTUM_THRESHOLD_MEV", 4.0)
val_sample       = int(pdf_cfg.get("VALIDATION_SAMPLE", 1000))
val_scale        = int(pdf_cfg.get("VALIDATION_SCALE", 1000))

configs_to_run = analysis_info.get("DEFAULT_CONFIGS", [])
names          = truth_pipeline.get("BACKGROUND_NAMES", ["gamma", "neutron"])

figure_path = f"{root}/images/background"
os.makedirs(figure_path, exist_ok=True)

# ── Static import data ────────────────────────────────────────────────────────
areas          = json.load(open(f"{root}/import/surface_areas.json"))
for geo, surfs in areas.items():
    for sid, expr in surfs.items():
        areas[geo][sid] = eval(expr)

activity_files = json.load(open(f"{root}/import/surface_activity.json"))
surface_names  = json.load(open(f"{root}/import/surfaces.json"))
surface_pos    = json.load(open(f"{root}/import/surface_positions.json"))

PARTICLE_MASS = {"electron": 0.511, "gamma": 0.0, "neutron": 939.565}

rprint(
    f"[bold]03_background_pdf[/bold] — backend=[cyan]{args.backend}[/cyan] "
    f"configs={configs_to_run} names={names}"
)


# ═════════════════════════════════════════════════════════════════════════════
# TRUTH BACKEND (ROOT TGraph spectra → PDFs)
# ═════════════════════════════════════════════════════════════════════════════

def _kinetic_to_momentum(k_kev, particle_type):
    """Convert kinetic energy (keV) to momentum (MeV/c)."""
    k_mev = 1e-3 * np.asarray(k_kev, dtype=float)
    if particle_type == "gamma":
        return k_mev
    elif particle_type in ("electron", "neutron"):
        return np.sqrt(2 * PARTICLE_MASS[particle_type] * k_mev)
    raise ValueError(f"Unknown particle type for E→p: {particle_type}")


def run_truth_backend(config: str, name: str):
    """Build momentum PDFs from truth-level ROOT TGraph flux spectra."""
    if config not in activity_files:
        rprint(f"[yellow][WARNING][/yellow] {config} not in surface_activity.json — skip")
        return

    rprint(f"\n[bold]{config} / {name}[/bold]  (truth backend)")

    data          = {}
    hist_s        = []
    bins_s        = []
    alpha_truth   = []
    exposure_dict = {}
    bin_edges     = np.array([])

    info          = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_workspace_detector_mass(config, info)

    config_areas = areas[config.split("_")[0]]
    config_files = activity_files[config]

    # ── Step 1: build per-surface combined spectra ───────────────────────────
    integrated_flux  = {sid: 0.0 for sid in config_files}
    combined_spectra = {sid: [] for sid in config_files}

    for surface_id, origins in config_files.items():
        for particle_origin, (this_file, weight) in origins.items():
            particle_type = particle_origin.lower().split("_")[-1]
            if name not in particle_type.lower():
                continue

            file_path = f"{truth_path}/{this_file}.root"
            rf        = TFile.Open(file_path)
            graph     = rf.Get("Gammas" if particle_type == "gamma" else "Neutrons")

            n     = graph.GetN()
            k_kev = np.array([graph.GetX()[i] for i in range(n)])
            y_raw = np.array([graph.GetY()[i] for i in range(n)])

            y = y_raw * weight / np.sum(y_raw)
            y = y * (60 * 60 * 24 * 365.25)   # s → year
            y = y / detector_mass              # → per kT·year

            shield = shielding_cfg.get(config, {}).get(particle_origin)
            if shield is not None:
                factor  = shield["reduction_factor"]
                thr_kev = shield["threshold_mev"] * 1e3
                rprint(
                    f"  [cyan][SHIELDING][/cyan] {particle_origin}: "
                    f"×{factor:.6f} above {shield['threshold_mev']} MeV"
                )
                y = np.where(k_kev >= thr_kev, y * factor, y)

            area_weight = config_areas[surface_id]
            integrated_flux[surface_id] += float(np.trapezoid(y, k_kev) * area_weight)

            p_mev = _kinetic_to_momentum(k_kev, particle_type)

            if len(combined_spectra[surface_id]) == 0:
                combined_spectra[surface_id] = (y * area_weight, p_mev)
            else:
                prev_y, prev_p = combined_spectra[surface_id]
                combined_spectra[surface_id] = (prev_y + y * area_weight, prev_p)

        if "gamma" in name and len(combined_spectra[surface_id]) > 0:
            spec_y, mom_p = combined_spectra[surface_id]
            mask = mom_p >= gamma_p_thresh
            combined_spectra[surface_id] = (spec_y[mask], mom_p[mask])

    # ── Step 2: histogram each surface spectrum → (h, bins) ─────────────────
    for variable, (surface_label, (surface_value, surface_id)) in product(
        ["ParticleP"],
        surface_pos[info["GEOMETRY"]].items(),
    ):
        surface_id_int = int(surface_id)
        surface_id_str = str(surface_id)
        if surface_id_int < 0:
            continue

        if len(combined_spectra[surface_id_str]) > 0:
            spec_y, spec_p = combined_spectra[surface_id_str]

            if args.binwidth is not None:
                bw = args.binwidth
            else:
                rms_p = np.sqrt(np.mean(spec_p ** 2))
                bw    = binwidth_factor * rms_p

            bin_edges = np.arange(spec_p.min() - bw, spec_p.max() + bw, bw)
            h, bins   = np.histogram(spec_p, bins=bin_edges, weights=spec_y)
            data[(surface_id_str, "ParticleP")] = (h, bins)
        else:
            rprint(
                f"  [yellow][WARNING][/yellow] No spectra for surface "
                f"{surface_label} — zero PDF"
            )
            h, bins, bin_edges = np.array([]), np.array([]), np.array([])
            data[(surface_id_str, "ParticleP")] = (h, bins)

        # Per-surface momentum PDF plot
        if args.plot:
            fig = make_subplots(rows=1, cols=1)
            if len(h) > 0:
                fig.add_trace(go.Scatter(
                    x=(bins[:-1] + bins[1:]) / 2, y=h,
                    mode="lines", name=name, line_shape="hvh",
                ))
            fig.update_yaxes(title_text="Events / (kT·years)")
            fig.update_xaxes(title_text="Particle Momentum (MeV/c)")
            fig = format_coustom_plotly(
                fig, log=(False, True), title=f"Surface {surface_label}",
                tickformat=(".1f", ".0e"), add_watermark=False,
            )
            save_figure(
                fig, figure_path, config, name, subfolder="weights",
                filename=f"{surface_label.replace(' ', '_')}_particlep_PDF",
                rm=args.rewrite, debug=args.plot,
            )

        counts = int(np.sum(h * np.diff(bins))) if len(h) > 0 else 0
        exposure_dict[surface_id_int] = {
            "detector_mass":      detector_mass,
            "detector_mass_unit": r"\mathrm{kT}",
            "detector_time":      1,
            "detector_time_unit": r"\mathrm{yr} \ \text{(normalized)}",
            "exposure":           1,
            "exposure_unit":      r"\mathrm{kT \cdot yr} \ \text{(normalized)}",
            "surface_label":      surface_label,
            "surface":            surface_id_str,
            "counts":             counts,
            "counts_unit":        r"\mathrm{events \cdot kT^{-1} \cdot yr^{-1}}",
            "name":               name,
        }
        rprint(f"  Surface {surface_label} — rate: {counts:.2e} events/(kT·years)")

    # ── Step 3: normalise → shape PDFs ──────────────────────────────────────
    for surface_label, (surface_value, surface_id) in surface_pos[info["GEOMETRY"]].items():
        surface_id_int = int(surface_id)
        surface_id_str = str(surface_id)
        if surface_id_int < 0:
            continue

        h, bins = data[(surface_id_str, "ParticleP")]
        if len(h) == 0:
            hist_s.append(np.array([]))
            bins_s.append(np.array([]))
            continue

        hist  = np.maximum(h, pdf_floor)
        hist /= np.sum(hist * np.diff(bins))
        hist_s.append(hist)
        bins_s.append(bins)

    # ── Save PDFs and exposure ───────────────────────────────────────────────
    save_pkl(
        exposure_dict, truth_path, config, None,
        filename=f"{name}_exposure", rm=args.rewrite, debug=args.debug,
    )
    save_pkl(
        (hist_s, bins_s), truth_path, config, None,
        filename=f"{name}_pdf", rm=args.rewrite, debug=args.debug,
    )

    # ── Step 4: alpha_truth ──────────────────────────────────────────────────
    total_counts = sum(exp["counts"] for exp in exposure_dict.values())
    if total_counts > 0:
        alpha_truth = [exp["counts"] / total_counts for exp in exposure_dict.values()]
    else:
        alpha_truth = [0.0] * len(exposure_dict)
    rprint(f"  Alpha truth: {[f'{a:.3f}' for a in alpha_truth]}")

    # ── Step 5: timing + PDF round-trip validation ───────────────────────────
    max_sid = max(exposure_dict.keys(), default=-1)
    if max_sid < 0:
        return

    s_test = np.concatenate([np.full(val_sample, s, dtype=float) for s in range(max_sid + 1)])
    if "gamma" in name:
        p_test = np.random.uniform(0, 14, val_sample * 5)
    else:
        p_test = np.sqrt(2 * PARTICLE_MASS["neutron"] * np.random.uniform(0, 14, val_sample * 5))

    t0 = time.time()
    w  = compute_weights(p_test, s_test.astype(int), hist_s, bins_s, alpha_truth)
    t1 = time.time()
    rprint(
        f"  PDF eval: {val_scale * (t1 - t0) / 60:.2f} min for "
        f"{val_sample * val_scale} samples (extrapolated)"
    )

    if args.plot and len(bin_edges) > 0:
        for surface_label, (surface_value, surface_id) in surface_pos[info["GEOMETRY"]].items():
            surface_id_int = int(surface_id)
            surface_id_str = str(surface_id)
            if surface_id_int < 0:
                continue

            h_orig, bins_orig = data[(surface_id_str, "ParticleP")]
            if len(bins_orig) == 0:
                continue

            mask_s    = s_test.astype(int) == surface_id_int
            h_wt, _   = np.histogram(p_test[mask_s], bins=bin_edges, weights=w[mask_s])
            denom     = np.sum(h_wt * np.diff(bin_edges))
            counts    = exposure_dict.get(surface_id_int, {}).get("counts", 1)
            y_check   = counts * h_wt / denom if denom > 0 else h_wt

            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(
                x=bins_orig, y=y_check,
                mode="lines", name="Weighted by PDF", line_shape="hvh",
            ))
            fig.add_trace(go.Scatter(
                x=bins_orig, y=h_orig,
                mode="lines", name="Original PDF", line_shape="hvh",
            ))
            fig.update_yaxes(title_text="Events / (kT·years)")
            fig.update_xaxes(title_text="ParticleP (MeV/c)")
            fig = format_coustom_plotly(
                fig, title=f"Surface {surface_label} — PDF Check",
                tickformat=(".1f", ".0e"), log=(False, True),
            )
            save_figure(
                fig, figure_path, config, name,
                filename=f"{surface_label.replace(' ', '_')}_pdf_check",
                rm=args.rewrite, debug=args.plot,
            )


# ═════════════════════════════════════════════════════════════════════════════
# LEGACY BACKEND (MC particle hits → PDFs)
# ═════════════════════════════════════════════════════════════════════════════

def run_legacy_backend(config: str, name: str):
    """Build momentum PDFs from reconstructed MC particle hits."""
    rprint(f"\n[bold]{config} / {name}[/bold]  (legacy backend)")

    configs_map  = {config: [name]}
    user_input   = {"workflow": "BACKGROUND", "rewrite": args.rewrite, "debug": args.debug}

    run, output = load_multi(
        configs_map, load_all=True, name_prefix="MCParticle_", debug=user_input["debug"]
    )
    run, output, _ = compute_particle_surface(
        run, configs_map, {}, ["Reco"], "Particle", False, output, user_input["debug"]
    )
    rprint(output)

    info          = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    surfaces      = json.load(open(f"{root}/import/surface_positions.json"))
    detector_time = 2 * info["TIMEWINDOW"] * len(run["Truth"]["Event"]) / 60 / 60 / 24 / 365.25
    detector_mass = get_workspace_detector_mass(config, info)
    detector_exp  = detector_mass * detector_time

    rprint(
        f"Loaded {len(run['Config']['Geometry'])} files, "
        f"{len(run['Truth']['Event'])} events, "
        f"{len(run['Reco']['Event'])} particles"
    )
    rprint(f"Detector exposure: {detector_exp:.3f} kT·years")

    data          = {}
    A             = []
    hist_s        = []
    bins_s        = []
    alpha_truth   = []
    exposure_dict = {}

    for variable, (surface_label, (surface_value, surface_id)) in product(
        ["ParticleP", "ParticleX", "ParticleY", "ParticleZ"],
        surfaces[info["GEOMETRY"]].items(),
    ):
        if int(surface_id) < 0:
            continue

        filtered_run, mask, output = compute_filtered_run(
            run, configs_map,
            params={("Reco", "SignalParticleSurface"): ("equal", surface_id)},
            output=output, debug=user_input["debug"],
        )

        if len(filtered_run["Reco"]["ParticleP"]) > 0:
            if variable == "ParticleP":
                if args.binwidth is None:
                    valid = filtered_run["Reco"][variable][~np.isnan(filtered_run["Reco"][variable])]
                    q75, q25 = np.percentile(valid, [75, 25])
                    bw = 2 * (q75 - q25) / np.cbrt(len(valid))
                else:
                    bw = args.binwidth
                vals = filtered_run["Reco"][variable]
                bins = np.arange(vals.min(), vals.max() + bw, bw)
            else:
                vals = filtered_run["Reco"][variable]
                bins = np.linspace(vals.min(), vals.max(), 100)

            h, bins = np.histogram(filtered_run["Reco"][variable], bins=bins, density=True)
            data[(surface_id, variable)] = (h, bins)
        else:
            rprint(
                f"[yellow][WARNING][/yellow] No particles for surface "
                f"{surface_label} / {variable} — empty PDF"
            )
            data[(surface_id, variable)] = (np.array([]), np.array([]))

        if variable == "ParticleP":
            counts = len(filtered_run["Reco"]["ParticleP"])
            exposure_dict[int(surface_id)] = {
                "detector_mass":      detector_mass,
                "detector_mass_unit": r"\mathrm{kT}",
                "detector_time":      detector_time,
                "detector_time_unit": r"\mathrm{yr}",
                "exposure":           detector_exp,
                "exposure_unit":      r"\mathrm{kT \cdot yr}",
                "surface_label":      surface_label,
                "surface":            surface_id,
                "counts":             counts,
                "counts_unit":        r"\mathrm{events} \ \text{(raw MC)}",
                "name":               name,
            }
            rprint(
                f"  Surface {surface_label} — counts: {counts}, "
                f"rate: {counts / detector_exp:.2e} events/(kT·years)"
            )

    # Normalise → shape PDFs
    for surface_label, (surface_value, surface_id) in surfaces[info["GEOMETRY"]].items():
        if int(surface_id) < 0:
            continue

        h, bins = data[(surface_id, "ParticleP")]
        if len(h) == 0:
            hist_s.append(np.array([]))
            bins_s.append(np.array([]))
            continue

        hist  = np.maximum(h, pdf_floor)
        hist /= np.sum(hist * np.diff(bins))
        hist_s.append(hist)
        bins_s.append(bins)

        surface_id_int = int(surface_id)
        if surface_id_int == 0:
            A.append(info["PRODUCTION_SIZE_Y"] * info["PRODUCTION_SIZE_Z"])
        elif surface_id_int in [1, 2]:
            A.append(info["PRODUCTION_SIZE_X"] * info["PRODUCTION_SIZE_Z"])
        elif surface_id_int in [3, 4]:
            A.append(info["PRODUCTION_SIZE_X"] * info["PRODUCTION_SIZE_Y"])

    rprint(f"Surface areas: {A}")

    save_pkl(
        exposure_dict, legacy_path, config, None,
        filename=f"{name}_exposure", rm=args.rewrite, debug=args.debug,
    )
    save_pkl(
        (hist_s, bins_s), legacy_path, config, None,
        filename=f"{name}_pdf", rm=args.rewrite, debug=args.debug,
    )
    # histograms.pkl consumed by 02_background_spectra.py for PDF plots
    save_pkl(
        data, legacy_path, config, None,
        filename=f"{name}_histograms", rm=args.rewrite, debug=args.debug,
    )

    total_counts = sum(exp["counts"] for exp in exposure_dict.values())
    for exp in exposure_dict.values():
        alpha_truth.append(exp["counts"] / total_counts if total_counts > 0 else 0.0)
    rprint(f"Alpha truth: {[f'{a:.3f}' for a in alpha_truth]}")

    if not args.plot or not data:
        return

    _detector_exp = (
        list(exposure_dict.values())[0]["exposure"] if exposure_dict else 1.0
    )

    # ── Per-surface / per-variable PDF plots with median vline ───────────────
    for _variable in ["ParticleP", "ParticleX", "ParticleY", "ParticleZ"]:
        for _surf_label, (_surf_val, _surf_id) in surfaces[info["GEOMETRY"]].items():
            _sid = int(_surf_id)
            if _sid < 0:
                continue
            # Only plot the variable most relevant to each surface orientation
            _should = (
                (_sid == 0      and _variable in ["ParticleP", "ParticleX"])
                or (_sid in [1, 2] and _variable == "ParticleY")
                or (_sid in [3, 4] and _variable == "ParticleZ")
            )
            if not _should:
                continue
            _key = (_surf_id, _variable)
            if _key not in data:
                continue
            _h, _bins = data[_key]
            if len(_h) == 0:
                continue

            _median = _hist_median(_h, _bins)
            _fig = make_subplots(rows=1, cols=1)
            _fig.add_trace(go.Scatter(
                x=(_bins[:-1] + _bins[1:]) / 2,
                y=_h / _detector_exp,
                mode="lines", name=name, line_shape="hvh",
            ))
            _fig.add_vline(
                x=_median, line_dash="dash", line_color="red",
                annotation_text=f"Median ≈ {_median:.2f}",
                annotation_position="top right",
            )
            _fig.update_yaxes(title_text="Events / (kT·years)")
            _fig.update_xaxes(title_text=_variable)
            _fig = format_coustom_plotly(
                _fig, log=(False, _variable == "ParticleP"),
                title=f"Surface {_surf_label}",
                tickformat=(".1f", ".0e"), add_watermark=False,
            )
            save_figure(
                _fig, figure_path, config, name,
                filename=f"{_surf_label.replace(' ', '_')}_{_variable.lower()}_PDF",
                rm=args.rewrite, debug=args.plot,
            )

    # ── PDF round-trip validation plots ──────────────────────────────────────
    _max_sid = max(exposure_dict.keys(), default=-1)
    if _max_sid < 0 or not hist_s:
        return

    _s_test = np.concatenate(
        [np.full(val_sample, s, dtype=float) for s in range(_max_sid + 1)]
    )
    _p_test = np.random.uniform(0, 14, val_sample * 5)
    _w = compute_weights(_p_test, _s_test.astype(int), hist_s, bins_s, alpha_truth)

    for _surf_label, (_surf_val, _surf_id) in surfaces[info["GEOMETRY"]].items():
        _sid = int(_surf_id)
        if _sid < 0:
            continue
        _key = (_surf_id, "ParticleP")
        if _key not in data:
            continue
        _h_orig, _bins_orig = data[_key]
        if len(_bins_orig) == 0:
            continue
        if _sid >= len(bins_s) or len(bins_s[_sid]) == 0:
            continue

        _mask_s      = _s_test.astype(int) == _sid
        _h_wt, _     = np.histogram(
            _p_test[_mask_s], bins=bins_s[_sid],
            weights=_w[_mask_s],
        )
        _bin_centers = (bins_s[_sid][:-1] + bins_s[_sid][1:]) / 2
        _denom       = np.sum(_h_wt * np.diff(bins_s[_sid]))
        _counts      = exposure_dict.get(_sid, {}).get("counts", 1)
        _y_check     = _counts * _h_wt / _denom if _denom > 0 else _h_wt

        _fig = make_subplots(rows=1, cols=1)
        _fig.add_trace(go.Scatter(
            x=_bin_centers, y=_y_check,
            mode="lines", name="Weighted by PDF", line_shape="hvh",
        ))
        _fig.add_trace(go.Scatter(
            x=_bin_centers, y=_h_orig / _detector_exp,
            mode="lines", name="Original PDF", line_shape="hvh",
        ))
        _fig.update_yaxes(title_text="Events / (kT·years)")
        _fig.update_xaxes(title_text="ParticleP (MeV/c)")
        _fig = format_coustom_plotly(
            _fig, title=f"Surface {_surf_label} — PDF Check",
            tickformat=(".1f", ".0e"), log=(False, True),
            ranges=(None, (-3, None)),
        )
        save_figure(
            _fig, figure_path, config, name,
            filename=f"{_surf_label.replace(' ', '_')}_pdf_check",
            rm=args.rewrite, debug=args.plot,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═════════════════════════════════════════════════════════════════════════════
for config, name in product(configs_to_run, names):
    if args.backend == "truth":
        run_truth_backend(config, name)
    else:
        run_legacy_backend(config, name)

rprint(
    f"\n[bold green]03_background_pdf complete "
    f"(backend={args.backend}).[/bold green]"
)
