"""
Backend comparison: Prob3++ vs NuFast-Earth oscillogram numerical and timing benchmark.

Usage:
    python3 scripts/compare_backends.py [--dm2 6e-5] [--sin13 0.021] [--sin12 0.303]
                                         [--n_repeats 5] [--save_plots]

Outputs:
    output/backend_comparison/  — heatmaps, difference maps, timing bar chart, report
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.lib_default import load_analysis_info
from lib.lib_osc_backends import (
    compute_prob3, compute_nufast,
    get_nadir_pdf_file, get_nadir_pdf_nufast,
    combine_day_night, OscResult,
)


def parse_args():
    p = argparse.ArgumentParser(description="Compare Prob3++ and NuFast-Earth oscillograms")
    p.add_argument("--dm2",       type=float, default=6.0e-5)
    p.add_argument("--sin13",     type=float, default=0.021)
    p.add_argument("--sin12",     type=float, default=0.303)
    p.add_argument("--n_repeats", type=int,   default=5,
                   help="Number of timing repetitions per backend")
    p.add_argument("--save_plots", action="store_true", default=True)
    p.add_argument("--out_dir",   default="output/backend_comparison")
    return p.parse_args()


def time_backend(fn, n_repeats):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    info = load_analysis_info(".")
    e_range = info.get("OSC_ENERGY_RANGE", [0, 30])
    e_bins  = info.get("OSC_ENERGY_BINS", 120)
    n_bins  = info.get("NADIR_BINS", 40)
    lat_deg = info.get("DUNE_LATITUDE_DEG", 44.35)

    energy_edges = np.linspace(e_range[0], e_range[1], e_bins + 1)
    nadir_edges  = np.linspace(-1.0, 1.0, n_bins + 1)
    e_centers    = 0.5 * (energy_edges[1:] + energy_edges[:-1])
    n_centers    = 0.5 * (nadir_edges[1:]  + nadir_edges[:-1])

    dm2, sin13, sin12 = args.dm2, args.sin13, args.sin12

    print_section(f"Parameters: dm2={dm2:.3e}  sin13={sin13:.3e}  sin12={sin12:.3e}")
    print(f"  Grid: {e_bins} energy bins × {n_bins} nadir bins")
    print(f"  Timing: {args.n_repeats} repeats each")

    # ── Compute oscillograms ─────────────────────────────────────────────────
    print_section("Computing oscillograms")

    print("  Prob3++  ...", end=" ", flush=True)
    t_p3 = time_backend(
        lambda: compute_prob3(dm2, sin13, sin12, energy_edges, nadir_edges),
        args.n_repeats
    )
    r_p3 = compute_prob3(dm2, sin13, sin12, energy_edges, nadir_edges)
    print(f"done  ({t_p3.mean()*1e3:.1f} ± {t_p3.std()*1e3:.1f} ms)")

    print("  NuFast-Earth ...", end=" ", flush=True)
    t_nf = time_backend(
        lambda: compute_nufast(dm2, sin13, sin12, energy_edges, nadir_edges, lat_deg),
        args.n_repeats
    )
    r_nf = compute_nufast(dm2, sin13, sin12, energy_edges, nadir_edges, lat_deg)
    print(f"done  ({t_nf.mean()*1e3:.1f} ± {t_nf.std()*1e3:.1f} ms)")

    # ── Nadir PDFs ───────────────────────────────────────────────────────────
    try:
        nadir_pdf_file  = get_nadir_pdf_file(nadir_centers=n_centers)
        has_file_pdf    = True
    except Exception:
        has_file_pdf    = False
    nadir_pdf_nufast = get_nadir_pdf_nufast(n_centers, lat_deg)

    # ── Combined oscillograms ────────────────────────────────────────────────
    nadir_pdf_use = nadir_pdf_file if has_file_pdf else nadir_pdf_nufast
    df_p3 = combine_day_night(r_p3, nadir_pdf_use)
    df_nf = combine_day_night(r_nf, nadir_pdf_use)

    # ── Numerical comparison ──────────────────────────────────────────────────
    print_section("Numerical comparison (raw P_ee, night DataFrame)")
    diff_night = r_p3.night.values - r_nf.night.values
    diff_day   = r_p3.day.values   - r_nf.day.values

    print(f"  Night P_ee:  max|ΔP_ee|={np.abs(diff_night).max():.5f}  "
          f"mean|ΔP_ee|={np.abs(diff_night).mean():.5f}  "
          f"RMS={np.sqrt((diff_night**2).mean()):.5f}")
    print(f"  Day P_ee:    max|ΔP_ee|={np.abs(diff_day).max():.5f}  "
          f"mean|ΔP_ee|={np.abs(diff_day).mean():.5f}  "
          f"RMS={np.sqrt((diff_day**2).mean()):.5f}")

    diff_combined = df_p3.values - df_nf.values
    print(f"  Combined:    max|ΔP_ee|={np.abs(diff_combined).max():.6f}  "
          f"mean|ΔP_ee|={np.abs(diff_combined).mean():.6f}")

    # Compare against file fixture if available
    fixture_path = os.path.join("tests", "fixtures", "oscillogram_default.pkl")
    if os.path.exists(fixture_path) and dm2 == 6.0e-5 and sin13 == 0.021 and sin12 == 0.303:
        ref = pd.read_pickle(fixture_path)
        print(f"\n  vs file fixture (shape {ref.shape}):")
        if df_p3.shape == ref.shape:
            diff_ref_p3 = df_p3.values - ref.values
            diff_ref_nf = df_nf.values - ref.values
            print(f"    prob3 vs file:   max|Δ|={np.abs(diff_ref_p3).max():.6f}  "
                  f"mean|Δ|={np.abs(diff_ref_p3).mean():.6f}")
            print(f"    nufast vs file:  max|Δ|={np.abs(diff_ref_nf).max():.6f}  "
                  f"mean|Δ|={np.abs(diff_ref_nf).mean():.6f}")
        else:
            print(f"    Shape mismatch: computed={df_p3.shape}, fixture={ref.shape}")

    # ── Timing summary ───────────────────────────────────────────────────────
    print_section("Timing summary")
    speedup = t_p3.mean() / t_nf.mean()
    print(f"  Prob3++:      {t_p3.mean()*1e3:.1f} ms/call  (n={args.n_repeats})")
    print(f"  NuFast-Earth: {t_nf.mean()*1e3:.1f} ms/call")
    print(f"  Speedup:      NuFast-Earth is {speedup:.1f}× {'faster' if speedup > 1 else 'slower'} than Prob3++")

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.save_plots:
        print_section(f"Saving plots to {args.out_dir}/")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Oscillogram comparison  dm2={dm2:.2e}  sin13={sin13:.3f}  sin12={sin12:.3f}")

        night_pee_p3 = r_p3.night.values
        night_pee_nf = r_nf.night.values

        vmax = max(night_pee_p3.max(), night_pee_nf.max())

        im0 = axes[0].imshow(night_pee_p3, aspect='auto', origin='lower',
                             extent=[e_centers.min(), e_centers.max(),
                                     n_centers.min(), n_centers.max()],
                             vmin=0, vmax=vmax, cmap='turbo')
        axes[0].set_title("Prob3++  P_ee(E, cos η)")
        axes[0].set_xlabel("Energy (MeV)")
        axes[0].set_ylabel("cos(η)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(night_pee_nf, aspect='auto', origin='lower',
                             extent=[e_centers.min(), e_centers.max(),
                                     n_centers.min(), n_centers.max()],
                             vmin=0, vmax=vmax, cmap='turbo')
        axes[1].set_title("NuFast-Earth  P_ee(E, cos η)")
        axes[1].set_xlabel("Energy (MeV)")
        plt.colorbar(im1, ax=axes[1])

        diff_abs = np.abs(diff_night)
        im2 = axes[2].imshow(diff_abs, aspect='auto', origin='lower',
                             extent=[e_centers.min(), e_centers.max(),
                                     n_centers.min(), n_centers.max()],
                             cmap='RdBu_r')
        axes[2].set_title(f"|ΔP_ee|  max={diff_abs.max():.4f}  mean={diff_abs.mean():.4f}")
        axes[2].set_xlabel("Energy (MeV)")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "oscillogram_comparison.png"), dpi=120)
        plt.close()

        # Timing bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        backends = ["Prob3++", "NuFast-Earth"]
        means    = [t_p3.mean() * 1e3, t_nf.mean() * 1e3]
        stds     = [t_p3.std()  * 1e3, t_nf.std()  * 1e3]
        bars     = ax.bar(backends, means, yerr=stds, capsize=5,
                          color=["#2196F3", "#FF5722"])
        ax.set_ylabel("Time per call (ms)")
        ax.set_title(f"Backend timing  ({args.n_repeats} repeats, {e_bins}×{n_bins} grid)")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val:.1f} ms", ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "timing_comparison.png"), dpi=120)
        plt.close()

        # Energy slices: P_ee(E) for a few nadir bins
        fig, ax = plt.subplots(figsize=(10, 5))
        slice_indices = [0, len(n_centers)//4, len(n_centers)//2, 3*len(n_centers)//4, -1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(slice_indices)))
        for idx, col in zip(slice_indices, colors):
            n_val = n_centers[idx]
            ax.plot(e_centers, night_pee_p3[idx, :], '-',  color=col, label=f"Prob3++ cos η={n_val:.2f}")
            ax.plot(e_centers, night_pee_nf[idx, :], '--', color=col)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("P_ee")
        ax.set_title("P_ee energy slices (solid=Prob3++, dashed=NuFast-Earth)")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "energy_slices.png"), dpi=120)
        plt.close()

        print(f"  Saved: oscillogram_comparison.png, timing_comparison.png, energy_slices.png")

    # ── Write text report ────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Oscillogram Backend Comparison Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Parameters: dm2={dm2:.3e}  sin13={sin13:.3e}  sin12={sin12:.3e}\n")
        f.write(f"Grid: {e_bins} energy × {n_bins} nadir\n\n")
        f.write(f"Night P_ee differences (Prob3++ - NuFast-Earth):\n")
        f.write(f"  max |ΔP_ee| = {np.abs(diff_night).max():.6f}\n")
        f.write(f"  mean|ΔP_ee| = {np.abs(diff_night).mean():.6f}\n")
        f.write(f"  RMS         = {np.sqrt((diff_night**2).mean()):.6f}\n\n")
        f.write(f"Timing (ms per call, n={args.n_repeats}):\n")
        f.write(f"  Prob3++:      {t_p3.mean()*1e3:.2f} ± {t_p3.std()*1e3:.2f} ms\n")
        f.write(f"  NuFast-Earth: {t_nf.mean()*1e3:.2f} ± {t_nf.std()*1e3:.2f} ms\n")
        f.write(f"  Speedup:      {speedup:.2f}x ({'NuFast-Earth faster' if speedup>1 else 'Prob3++ faster'})\n")
    print(f"\n  Report saved: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
