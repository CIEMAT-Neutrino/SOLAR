"""
0XProcessOscillationFiles.py — Oscillation Template Generation
==============================================================
Pipeline step 0X (pre-processing): builds the nadir-weighted P_ee oscillation
templates that the "file" backend reads at analysis time.

Three modes selected via --backend:

  file   (default / backwards-compat)
    Scans the existing ROOT oscillogram files in {OSC_PATH}/root/, rebins any
    that don't yet have a matching pkl/rebin/ entry, and saves them.
    Pass --rewrite to force re-rebin of all ROOT files (use when best-fit
    parameters have changed and all pkl files need refreshing).

  prob3 / nufast
    Generates oscillograms from first principles using the external bindings
    in external/Prob3plusplus and external/NuFast-Earth respectively.
    The grid is taken from OSCILLATION_GRID in analysis/physics.json.
    Saves both raw pkl (high-res nadir) and rebin pkl (analysis-resolution).
    Pass --rewrite to overwrite files that already exist on disk.

Output layout
-------------
  {OSC_PATH}/pkl/raw/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl
  {OSC_PATH}/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl

Run
---
  # Rebin all ROOT files (file backend, first time):
  python3 0XProcessOscillationFiles.py

  # Refresh all rebin pkl after changing best-fit params:
  python3 0XProcessOscillationFiles.py --rewrite

  # Generate a full grid from scratch using NuFast:
  python3 0XProcessOscillationFiles.py --backend nufast --rewrite

  # Generate using Prob3++, skip saving raw pkl:
  python3 0XProcessOscillationFiles.py --backend prob3 --no-save-raw
"""

import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.oscillation import make_oscillation_grid, rebin_df, get_nadir_angle
from lib.oscillation_backends import (
    compute_prob3,
    compute_nufast,
    get_nadir_pdf_file,
    get_nadir_pdf_nufast,
    combine_day_night,
)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate and/or rebin oscillation probability templates"
)
parser.add_argument(
    "--backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help=(
        "'file': rebin existing ROOT files to pkl (backwards-compat). "
        "'prob3'/'nufast': generate pkl from first-principles backends."
    ),
)
parser.add_argument(
    "--path",
    type=str,
    default=None,
    help="Override oscillation data directory (default: from config/{config}_config.json PATH)",
)
parser.add_argument(
    "--save-raw",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save the high-resolution raw pkl alongside the rebinned pkl (prob3/nufast only).",
)
parser.add_argument(
    "--rewrite",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Overwrite existing pkl files. "
        "For 'file' backend: re-rebin all ROOT files even if rebin pkl exists. "
        "For prob3/nufast: regenerate all grid points even if pkl files exist."
    ),
)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# ── Analysis config ───────────────────────────────────────────────────────────
analysis_info = load_analysis_info(str(root))

# Oscillation data path — prefer CLI override, then config PATH, then hardcoded default
if args.path is not None:
    osc_path = args.path.rstrip("/") + "/"
else:
    # Try to resolve from any available config file
    _cfg_glob = sorted(glob.glob(f"{root}/config/*/*.json"))
    _osc_path_from_cfg = None
    for _cfg_file in _cfg_glob:
        try:
            _info = json.loads(open(_cfg_file).read())
            if "PATH" in _info:
                _osc_path_from_cfg = _info["PATH"] + "/data/OSCILLATION/"
                break
        except Exception:
            pass
    osc_path = _osc_path_from_cfg or "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/"

rprint(f"[cyan][INFO][/cyan] Oscillation data path: {osc_path}")
rprint(f"[cyan][INFO][/cyan] Backend: {args.backend}   Rewrite: {args.rewrite}")

# Output directories
for _subdir in ["pkl/raw", "pkl/rebin"]:
    _dir = os.path.join(osc_path, _subdir)
    os.makedirs(_dir, exist_ok=True)


def _pkl_name(dm2, sin13, sin12, subfolder):
    """Return full path to the pkl file for a given parameter triplet."""
    return os.path.join(
        osc_path, "pkl", subfolder,
        f"osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl",
    )


def _remove_if_rewrite(path):
    """Delete a file when --rewrite is active."""
    if args.rewrite and os.path.exists(path):
        os.remove(path)
        if args.debug:
            rprint(f"  [yellow][REWRITE][/yellow] Deleted {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: file — rebin existing ROOT files
# ═══════════════════════════════════════════════════════════════════════════════
if args.backend == "file":
    root_pattern = os.path.join(osc_path, "root", "*_dm2_*_sin13_*_sin12_*.root")
    root_files   = sorted(glob.glob(root_pattern))

    if not root_files:
        rprint(f"[yellow][WARNING][/yellow] No ROOT files found in {osc_path}root/")
        sys.exit(0)

    rprint(f"Found {len(root_files)} ROOT oscillation file(s).")

    # Parse existing rebin pkl to know what's already done
    def _parse_triplets(file_list):
        """Extract (dm2, sin13, sin12) floats from a list of file paths."""
        triplets = set()
        for fpath in file_list:
            base = os.path.basename(fpath).rsplit(".", 1)[0]
            parts = base.split("_")
            # format: osc_probability_dm2_<dm2>_sin13_<sin13>_sin12_<sin12>
            try:
                idx_dm2   = parts.index("dm2")
                idx_sin13 = parts.index("sin13")
                idx_sin12 = parts.index("sin12")
                triplets.add((
                    float(parts[idx_dm2   + 1]),
                    float(parts[idx_sin13 + 1]),
                    float(parts[idx_sin12 + 1]),
                ))
            except (ValueError, IndexError):
                pass
        return triplets

    existing_rebin = _parse_triplets(
        glob.glob(os.path.join(osc_path, "pkl", "rebin", "*.pkl"))
    )
    root_triplets  = _parse_triplets(root_files)

    # With --rewrite: force re-rebin of all ROOT triplets
    if args.rewrite:
        to_process = root_triplets
        for dm2, sin13, sin12 in to_process:
            _remove_if_rewrite(_pkl_name(dm2, sin13, sin12, "rebin"))
            _remove_if_rewrite(_pkl_name(dm2, sin13, sin12, "raw"))
    else:
        to_process = root_triplets - existing_rebin

    if not to_process:
        rprint("[green][OK][/green] All ROOT files already rebinned. Nothing to do.")
        rprint("  (Pass --rewrite to force refresh)")
        sys.exit(0)

    rprint(f"Processing {len(to_process)} ROOT file(s) → pkl/rebin/")

    for dm2, sin13, sin12 in track(
        sorted(to_process), description="Rebinning ROOT → pkl"
    ):
        dm2_f   = float("%.3e" % dm2)
        sin13_f = sin13
        sin12_f = float("%.3e" % sin12)
        rebin_path = _pkl_name(dm2_f, sin13_f, sin12_f, "rebin")
        raw_path   = _pkl_name(dm2_f, sin13_f, sin12_f, "raw")

        if args.debug:
            rprint(f"  Processing dm2={dm2_f:.3e} sin13={sin13_f:.3e} sin12={sin12_f:.3e}")

        try:
            oscillation_df_dict = get_oscillation_map(
                path=osc_path,
                dm2=dm2_f,
                sin13=sin13_f,
                sin12=sin12_f,
                auto=False,
                rebin=True,
                rw=args.rewrite,
                output="df",
                save=True,
                ext="root",
                debug=args.debug,
            )
            rprint(f"  [green][OK][/green] Rebinned → {rebin_path}")
        except Exception as exc:
            rprint(f"  [red][ERROR][/red] dm2={dm2_f:.3e} sin13={sin13_f:.3e} sin12={sin12_f:.3e}: {exc}")

    rprint(f"[bold green]File backend: rebinning complete.[/bold green]")


# ═══════════════════════════════════════════════════════════════════════════════
# MODES: prob3 / nufast — generate from first principles
# ═══════════════════════════════════════════════════════════════════════════════
else:
    # Build parameter grid from OSCILLATION_GRID in physics.json
    dm2_list, sin13_list, sin12_list = make_oscillation_grid(analysis_info)
    n_total = len(dm2_list)
    rprint(f"Grid from OSCILLATION_GRID: {n_total} parameter point(s)")

    # Energy and nadir grids
    e_range     = analysis_info.get("OSC_ENERGY_RANGE", [0, 30])
    e_bins      = int(analysis_info.get("OSC_ENERGY_BINS", 120))
    nadir_bins_raw = int(analysis_info.get("ROOT_NADIR_BINS", 2000))  # high-res for raw pkl
    latitude_deg   = float(analysis_info.get("DUNE_LATITUDE_DEG", 44.35))

    energy_edges_raw = np.linspace(e_range[0], e_range[1], e_bins + 1)
    nadir_edges_raw  = np.linspace(-1.0, 1.0,  nadir_bins_raw + 1)
    nadir_centers_raw = 0.5 * (nadir_edges_raw[1:] + nadir_edges_raw[:-1])

    # Nadir PDF at raw resolution
    try:
        nadir_pdf_raw = get_nadir_pdf_file(
            path=osc_path, nadir_centers=nadir_centers_raw
        )
        rprint(f"[cyan][INFO][/cyan] Nadir PDF loaded from nadir.root ({nadir_bins_raw} bins)")
    except Exception as _exc:
        rprint(
            f"[yellow][WARNING][/yellow] nadir.root not found ({_exc}); "
            f"falling back to NuFast Solar_Weight for nadir PDF"
        )
        nadir_pdf_raw = get_nadir_pdf_nufast(nadir_centers_raw, latitude_deg)

    # Track skipped/done counts
    n_skipped = 0
    n_done    = 0
    n_failed  = 0

    compute_fn = compute_prob3 if args.backend == "prob3" else compute_nufast

    for dm2, sin13, sin12 in track(
        zip(dm2_list, sin13_list, sin12_list),
        description=f"Generating oscillograms [{args.backend}]",
        total=n_total,
    ):
        dm2_f   = float("%.3e" % dm2)
        sin13_f = sin13
        sin12_f = float("%.3e" % sin12)

        raw_path   = _pkl_name(dm2_f, sin13_f, sin12_f, "raw")
        rebin_path = _pkl_name(dm2_f, sin13_f, sin12_f, "rebin")

        # Skip if both outputs already exist and --rewrite not set
        raw_exists   = os.path.exists(raw_path)
        rebin_exists = os.path.exists(rebin_path)

        if not args.rewrite and rebin_exists and (not args.save_raw or raw_exists):
            if args.debug:
                rprint(
                    f"  [dim]Skip dm2={dm2_f:.3e} sin13={sin13_f:.3e} sin12={sin12_f:.3e} "
                    f"(already exists)[/dim]"
                )
            n_skipped += 1
            continue

        _remove_if_rewrite(raw_path)
        _remove_if_rewrite(rebin_path)

        if args.debug:
            rprint(
                f"  [{args.backend}] dm2={dm2_f:.3e} sin13={sin13_f:.3e} sin12={sin12_f:.3e}"
            )

        try:
            # ── Compute at high-res nadir for raw pkl fidelity ────────────────
            if args.backend == "nufast":
                osc = compute_nufast(
                    dm2_f, sin13_f, sin12_f,
                    energy_edges_raw, nadir_edges_raw,
                    latitude_deg=latitude_deg,
                )
            else:
                osc = compute_prob3(
                    dm2_f, sin13_f, sin12_f,
                    energy_edges_raw, nadir_edges_raw,
                )

            # Nadir-weighted DataFrame (same format as process_oscillation_map output)
            raw_df = combine_day_night(osc, nadir_pdf_raw)

            # ── Save raw pkl ──────────────────────────────────────────────────
            if args.save_raw:
                import pickle
                with open(raw_path, "wb") as _f:
                    pickle.dump(raw_df, _f)
                if args.debug:
                    rprint(f"    Saved raw pkl ({raw_df.shape[0]}×{raw_df.shape[1]})")

            # ── Rebin to analysis resolution and save ─────────────────────────
            rebinned_df = rebin_df(
                raw_df,
                save_path=rebin_path,
                show=False,
                save=True,
                debug=args.debug,
            )
            if args.debug:
                rprint(
                    f"    Saved rebin pkl ({rebinned_df.shape[0]}×{rebinned_df.shape[1]})"
                )

            n_done += 1

        except Exception as exc:
            rprint(
                f"  [red][ERROR][/red] dm2={dm2_f:.3e} sin13={sin13_f:.3e} "
                f"sin12={sin12_f:.3e}: {exc}"
            )
            n_failed += 1

    rprint(
        f"\n[bold green]{args.backend} backend complete.[/bold green] "
        f"Done: {n_done}  Skipped: {n_skipped}  Failed: {n_failed} / {n_total} total"
    )
    if n_failed > 0:
        rprint(
            f"[yellow][WARNING][/yellow] {n_failed} point(s) failed. "
            f"Check that external/{('NuFast-Earth' if args.backend == 'nufast' else 'Prob3plusplus')}/python "
            f"binding is built."
        )
