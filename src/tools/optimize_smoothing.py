"""
Compute optimal Gaussian smoothing sigma for each background component using
analytic bandwidth rules on rebinned histogram data.

Default strategy: Silverman's rule-of-thumb (Strategy 4).

  sigma_Silverman = 1.06 * min(std, IQR/1.34) * N^(-1/5)   [energy units]
  sigma_bins      = sigma_Silverman / bin_width

Scott's rule is also available:
  sigma_Scott = 1.059 * std * N^(-1/5)                      [energy units]

Output: data/smoothing/{config}/{name}/{folder}_{energy}_{analysis}_sigma.json
  {
    "config": ..., "name": ..., "energy": ..., "analysis": ..., "strategy": ...,
    "components": {
      "gamma": {"sigma_bins": 1.23, "n_events": 5000, "sigma_mev": 2.46, "bin_width_mev": 2.0},
      ...
    },
    "recommended_sigma": 1.05   # median across smoothed components
  }

Usage:
  python3 scripts/optimize_smoothing.py \\
      --config hd_1x2x6_centralAPA --name marley \\
      --energy SolarEnergy --folder Truncated --analysis HEP
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parents[2]))
from lib import root, load_analysis_info, load_available_background_dataframes  # noqa: E402
try:
    from rich import print as rprint
except ImportError:
    rprint = print


SIGNAL_PATH_TEMPLATE = (
    "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal"
    "/{folder}/{analysis}/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
)
BACKGROUND_PATH_TEMPLATE = (
    "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background"
    "/{folder}/{analysis}/{config}/{component}/{config}_{component}_{energy}_Rebin.pkl"
)


# ---------------------------------------------------------------------------
# Bandwidth estimators
# ---------------------------------------------------------------------------

def _weighted_stats(counts: np.ndarray, centers: np.ndarray):
    """Return (n, mean, std, iqr_estimate) for a binned histogram."""
    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n = float(np.nansum(counts))
    if n <= 0.0:
        return n, 0.0, 0.0, 0.0
    mu = float(np.nansum(centers * counts) / n)
    variance = float(np.nansum(counts * (centers - mu) ** 2) / n)
    std = float(np.sqrt(max(variance, 0.0)))
    cdf = np.cumsum(counts) / n
    q25 = float(np.interp(0.25, cdf, centers))
    q75 = float(np.interp(0.75, cdf, centers))
    iqr_estimate = (q75 - q25) / 1.34
    return n, mu, std, iqr_estimate


def silverman_sigma_bins(counts: np.ndarray, centers: np.ndarray) -> float:
    """Silverman bandwidth in bin units from a binned 1-D histogram."""
    if len(centers) < 2:
        return 0.0
    bin_width = float(centers[1] - centers[0])
    if bin_width <= 0.0:
        return 0.0
    n, _, std, iqr_est = _weighted_stats(counts, centers)
    if n <= 1.0:
        return 0.0
    effective_std = min(std, iqr_est) if iqr_est > 0.0 else std
    h = 1.06 * effective_std * n ** (-0.2)
    return max(0.0, h / bin_width)


def scott_sigma_bins(counts: np.ndarray, centers: np.ndarray) -> float:
    """Scott bandwidth in bin units from a binned 1-D histogram."""
    if len(centers) < 2:
        return 0.0
    bin_width = float(centers[1] - centers[0])
    if bin_width <= 0.0:
        return 0.0
    n, _, std, _ = _weighted_stats(counts, centers)
    if n <= 1.0:
        return 0.0
    h = 1.059 * std * n ** (-0.2)
    return max(0.0, h / bin_width)


STRATEGIES = {"silverman": silverman_sigma_bins, "scott": scott_sigma_bins}


# ---------------------------------------------------------------------------
# Histogram extraction
# ---------------------------------------------------------------------------

def _best_counts_row(df: pd.DataFrame, component: str):
    """Return the row with highest total MCCounts for a given component.

    Uses Mean=='Mean' rows only to avoid double-counting Day/Night splits.
    """
    sub = df.loc[
        (df["Component"].astype(str).str.lower() == component.lower())
        & (df["Mean"].astype(str) == "Mean")
    ]
    if sub.empty:
        return None
    mc_totals = sub["MCCounts"].apply(lambda x: float(np.nansum(np.asarray(x, dtype=float))))
    return sub.iloc[mc_totals.argmax()]


def compute_component_sigma(df: pd.DataFrame, component: str, strategy_fn):
    """Compute bandwidth sigma for one component from a Rebin dataframe."""
    row = _best_counts_row(df, component)
    if row is None:
        return None
    counts = np.asarray(row["MCCounts"], dtype=float)
    centers = np.asarray(row["Energy"], dtype=float)
    if len(counts) == 0 or np.nansum(counts) == 0:
        return None
    sigma_bins = strategy_fn(counts, centers)
    n = float(np.nansum(counts))
    bin_width = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    return {
        "sigma_bins": round(sigma_bins, 4),
        "sigma_mev": round(sigma_bins * bin_width, 4),
        "bin_width_mev": round(bin_width, 4),
        "n_events": int(round(n)),
    }


# ---------------------------------------------------------------------------
# analysis.json patcher
# ---------------------------------------------------------------------------

def _patch_analysis_json(config: str, name: str, analysis: str, recommended_sigma: float) -> bool:
    """Write recommended_sigma into config/analysis/smoothing.json:
    1. SMOOTHING.CONFIG_OVERRIDES.{config}.{name}.{analysis} — per-config/name record
       (runtime sigma delivered via SOLAR_SMOOTHING_SIGMA_{ANALYSIS} env var by orchestrator).
    2. SMOOTHING.ANALYSES.{analysis}.STAGES.*.dimensions.1d.sigma — global fallback default.
    """
    analysis_json_path = Path(str(root)) / "config" / "analysis" / "smoothing.json"
    if not analysis_json_path.exists():
        return False
    try:
        data = json.loads(analysis_json_path.read_text())
    except Exception:
        return False

    sigma_val = round(float(recommended_sigma), 4)

    # Per-config/name override record
    (
        data.setdefault("SMOOTHING", {})
        .setdefault("CONFIG_OVERRIDES", {})
        .setdefault(config, {})
        .setdefault(name, {})[analysis.upper()]
    ) = {"recommended_sigma": sigma_val}

    # Global fallback: update STAGES sigma for this analysis
    stages = (
        data.get("SMOOTHING", {})
        .get("ANALYSES", {})
        .get(analysis.upper(), {})
        .get("STAGES", {})
    )
    for stage_cfg in stages.values():
        dims = stage_cfg.get("dimensions", {})
        if "1d" in dims:
            dims["1d"]["sigma"] = sigma_val

    try:
        analysis_json_path.write_text(json.dumps(data, indent=2))
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Main per-(config, name, energy, analysis) computation
# ---------------------------------------------------------------------------

def run_optimization(config: str, name: str, energy: str, folder: str,
                     analysis: str, strategy: str, output_dir: Path,
                     rewrite: bool, patch: bool = False):
    out_path = output_dir / f"{folder.lower()}_{energy}_{analysis}_sigma.json"
    if out_path.exists() and not rewrite:
        rprint(f"[cyan][INFO][/cyan] Skipping (exists): {out_path}")
        return json.loads(out_path.read_text())

    strategy_fn = STRATEGIES[strategy]
    result = {
        "config": config,
        "name": name,
        "energy": energy,
        "analysis": analysis,
        "folder": folder.lower(),
        "strategy": strategy,
        "components": {},
    }

    # --- background components ---
    for component, filepath in load_available_background_dataframes(
        str(root), analysis, folder, config, energy
    ):
        try:
            bkg_df = pd.read_pickle(filepath)
        except Exception as exc:
            rprint(f"[yellow][WARNING][/yellow] Cannot read {filepath}: {exc}")
            continue
        stats = compute_component_sigma(bkg_df, component, strategy_fn)
        if stats is not None:
            result["components"][component] = stats

    # --- signal components (smoothing rarely applied, but included for completeness) ---
    sig_path = SIGNAL_PATH_TEMPLATE.format(
        folder=folder.lower(), analysis=analysis.upper(),
        config=config, name=name, energy=energy,
    )
    if os.path.exists(sig_path):
        try:
            sig_df = pd.read_pickle(sig_path)
            for component in sig_df["Component"].astype(str).unique():
                stats = compute_component_sigma(sig_df, component, strategy_fn)
                if stats is not None:
                    result["components"][component] = stats
        except Exception as exc:
            rprint(f"[yellow][WARNING][/yellow] Cannot read signal {sig_path}: {exc}")

    if not result["components"]:
        rprint(
            f"[yellow][WARNING][/yellow] No histogram data found for "
            f"{config} {name} {energy} {analysis}. Skipping."
        )
        return None

    # recommended_sigma: max over smoothed background components.
    # Max is the right aggregate for a single global sigma: ensures the component
    # with worst statistics (lowest N → highest Silverman sigma) gets adequate smoothing.
    analysis_info = load_analysis_info(str(root))
    smoothing_cfg = analysis_info.get("SMOOTHING", {})
    smoothed_components = set(
        str(c).lower()
        for c in smoothing_cfg.get("ANALYSES", {})
        .get(analysis.upper(), {})
        .get("STAGES", {})
        .get("SIGNIFICANCE", {})
        .get("components", [])
    )
    bg_sigmas = [
        v["sigma_bins"]
        for k, v in result["components"].items()
        if k in smoothed_components and v["sigma_bins"] > 0.0
    ]
    result["recommended_sigma"] = round(float(np.max(bg_sigmas)), 4) if bg_sigmas else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    component_summary = ", ".join(
        f"{k}={v['sigma_bins']:.2f}"
        for k, v in result["components"].items()
        if k in smoothed_components
    )
    rprint(
        f"[green][SMOOTHING][/green] {config} {name} {energy} {analysis}: "
        f"recommended_sigma={result['recommended_sigma']:.3f} bins "
        f"({component_summary})"
    )

    if patch:
        patched = _patch_analysis_json(config, name, analysis, result["recommended_sigma"])
        if patched:
            rprint(
                f"[green][SMOOTHING][/green] Patched config/analysis/smoothing.json: "
                f"SMOOTHING.ANALYSES.{analysis.upper()}.STAGES.*.dimensions.1d.sigma "
                f"= {result['recommended_sigma']:.4f}"
            )
        else:
            rprint(
                f"[yellow][WARNING][/yellow] Could not patch config/analysis/smoothing.json for {analysis}."
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--name", nargs="+", default=["marley"])
    parser.add_argument(
        "--energy", nargs="+",
        default=["SolarEnergy"],
        choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    )
    parser.add_argument(
        "--folder", type=str, default="Truncated",
        choices=["Reduced", "Truncated", "Nominal"],
    )
    parser.add_argument(
        "--analysis", nargs="+",
        default=["DayNight", "HEP", "Sensitivity"],
        choices=["DayNight", "HEP", "Sensitivity"],
    )
    parser.add_argument(
        "--strategy", type=str, default="silverman",
        choices=["silverman", "scott"],
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (default: {root}/output/data/smoothing)",
    )
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--patch", action=argparse.BooleanOptionalAction, default=False,
        help="If set, write recommended_sigma back into config/analysis/smoothing.json.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir) if args.output_dir else Path(str(root)) / "data" / "smoothing"

    for config, name, energy, analysis in product(
        args.config, args.name, args.energy, args.analysis
    ):
        out_dir = base_dir / config / name
        run_optimization(config, name, energy, args.folder, analysis,
                         args.strategy, out_dir, args.rewrite, patch=args.patch)


if __name__ == "__main__":
    main()
