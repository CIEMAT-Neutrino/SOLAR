"""
Visual comparison of oscillation backends: file fixture vs Prob3++ vs NuFast-Earth.

Generates plots saved to output/backend_visual/.
Run with:
    python3 -m pytest tests/test_backend_visual.py -v -s

KEY CONVENTION:
  The library pkl files store  P_ee(E, cosη) × nadir_PDF(cosη)  (nadir-weighted).
  Raw P_ee from backends is in [0.18, 0.58]; fixture is in [0, 0.022].
  Comparing raw P_ee to fixture is apples-to-oranges.
  All fixture comparisons use the COMBINED (nadir-weighted) oscillogram.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.oscillation_backends import (
    compute_prob3,
    compute_nufast,
    get_nadir_pdf_nufast,
    combine_day_night,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DM2, SIN13, SIN12 = 6.0e-5, 0.021, 0.303
LATITUDE_DEG = 44.35

ENERGY_EDGES = np.linspace(0.0, 30.0, 121)
NADIR_EDGES  = np.linspace(-1.0, 1.0, 41)
E_CENTERS    = 0.5 * (ENERGY_EDGES[1:] + ENERGY_EDGES[:-1])
N_CENTERS    = 0.5 * (NADIR_EDGES[1:]  + NADIR_EDGES[:-1])

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "oscillogram_default.pkl")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "..", "output", "backend_visual")

PNFS_ROOT     = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION"
PNFS_PKL      = os.path.join(
    PNFS_ROOT, "pkl", "rebin",
    "osc_probability_dm2_6.000e-05_sin13_2.100e-02_sin12_3.030e-01.pkl",
)
PNFS_ROOT_FILE = os.path.join(
    PNFS_ROOT, "root",
    "osc_probability_dm2_6.000e-05_sin13_2.100e-02_sin12_3.030e-01.root",
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def computed_oscillograms():
    r_p3 = compute_prob3( DM2, SIN13, SIN12, ENERGY_EDGES, NADIR_EDGES)
    r_nf = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES, NADIR_EDGES, LATITUDE_DEG)
    nadir_pdf = get_nadir_pdf_nufast(N_CENTERS, LATITUDE_DEG)
    df_p3 = combine_day_night(r_p3, nadir_pdf)
    df_nf = combine_day_night(r_nf, nadir_pdf)
    return {
        "r_p3": r_p3, "r_nf": r_nf,
        "df_p3": df_p3, "df_nf": df_nf,
        "nadir_pdf": nadir_pdf,
    }


def _load_reference():
    """Return (array [n_nadir, n_energy], label) for the best available reference.
    The reference is a nadir-weighted oscillogram (same scale as combine_day_night output).
    """
    if os.path.exists(PNFS_PKL):
        ref = pd.read_pickle(PNFS_PKL)
        return ref.values, "File library (PNFS)"
    if os.path.exists(FIXTURE_PATH):
        ref = pd.read_pickle(FIXTURE_PATH)
        return ref.values, "File fixture (snapshot)"
    return None, None


# ── Test 1: Raw P_ee heatmaps (backends only, no fixture) ─────────────────────

def test_oscillogram_heatmaps(computed_oscillograms):
    """
    Raw P_ee(E, cosη) heatmaps from each backend plus absolute difference.

    NOTE: The library pkl files are nadir-WEIGHTED and cannot be compared to
    raw P_ee directly (different scales: raw ~0.5, weighted ~0.02).
    Fixture comparison is in test_combined_oscillogram.

    Columns: Prob3++ | NuFast-Earth | |Prob3 − NuFast|
    """
    p3 = computed_oscillograms["r_p3"].night.values   # [n_nadir, n_energy]
    nf = computed_oscillograms["r_nf"].night.values

    diff_p3_nf = np.abs(p3 - nf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Raw P_ee(E, cosη)   dm²={DM2:.2e}  sin²θ₁₃={SIN13:.3f}  sin²θ₁₂={SIN12:.3f}\n"
        "(library pkl files store nadir-WEIGHTED P_ee — see test_combined_oscillogram for comparison)",
        fontsize=10,
    )

    extent = [E_CENTERS.min(), E_CENTERS.max(), N_CENTERS.min(), N_CENTERS.max()]
    kw     = dict(aspect="auto", origin="lower", extent=extent)
    vmax   = max(p3.max(), nf.max())

    im0 = axes[0].imshow(p3, vmin=0, vmax=vmax, cmap="turbo", **kw)
    axes[0].set_title("Prob3++")
    plt.colorbar(im0, ax=axes[0], label="P_ee")

    im1 = axes[1].imshow(nf, vmin=0, vmax=vmax, cmap="turbo", **kw)
    axes[1].set_title("NuFast-Earth")
    plt.colorbar(im1, ax=axes[1], label="P_ee")

    im2 = axes[2].imshow(diff_p3_nf, cmap="RdBu_r", **kw)
    axes[2].set_title(
        f"|Prob3 − NuFast|  (raw P_ee)\nmax={diff_p3_nf.max():.4f}  mean={diff_p3_nf.mean():.5f}"
    )
    plt.colorbar(im2, ax=axes[2], label="|ΔP_ee|")

    for ax in axes:
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("cos(η)")
        ax.axhline(0.0, ls="--", color="white", lw=0.8, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "raw_pee_heatmaps.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [raw heatmaps] {path}")

    assert diff_p3_nf.max() < 0.02, f"Backends differ > 2%: max|ΔP_ee|={diff_p3_nf.max():.4f}"


# ── Test 2: Combined oscillogram vs library fixture ────────────────────────────

def test_combined_oscillogram(computed_oscillograms):
    """
    Nadir-weighted P_ee for both backends vs library fixture.

    Both quantities are  P_ee(E, cosη) × nadir_PDF(cosη)  — same scale [0, 0.03].
    This is the correct apples-to-apples comparison.

    Columns: Prob3++×PDF | NuFast×PDF | |Prob3−NuFast| | |Prob3−library| | |NuFast−library|
    """
    df_p3 = computed_oscillograms["df_p3"]
    df_nf = computed_oscillograms["df_nf"]

    ref_vals, ref_label = _load_reference()

    extent = [E_CENTERS.min(), E_CENTERS.max(), N_CENTERS.min(), N_CENTERS.max()]
    kw     = dict(aspect="auto", origin="lower", extent=extent)

    ncols = 5 if (ref_vals is not None and ref_vals.shape == df_p3.shape) else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(
        "Nadir-weighted P_ee  P_ee(E,cosη)×w(cosη)  [same scale as library pkl]\n"
        f"dm²={DM2:.2e}  sin²θ₁₃={SIN13:.3f}  sin²θ₁₂={SIN12:.3f}",
        fontsize=10,
    )

    vmax = max(df_p3.values.max(), df_nf.values.max())

    im0 = axes[0].imshow(df_p3.values, vmin=0, vmax=vmax, cmap="turbo", **kw)
    axes[0].set_title("Prob3++ × nadir PDF")
    plt.colorbar(im0, ax=axes[0], label="w·P_ee")

    im1 = axes[1].imshow(df_nf.values, vmin=0, vmax=vmax, cmap="turbo", **kw)
    axes[1].set_title("NuFast-Earth × nadir PDF")
    plt.colorbar(im1, ax=axes[1], label="w·P_ee")

    diff_backends = np.abs(df_p3.values - df_nf.values)
    im2 = axes[2].imshow(diff_backends, cmap="RdBu_r", **kw)
    axes[2].set_title(
        f"|Prob3 − NuFast| (weighted)\nmax={diff_backends.max():.5f}  mean={diff_backends.mean():.6f}"
    )
    plt.colorbar(im2, ax=axes[2], label="|Δ(w·P_ee)|")

    if ncols == 5:
        diff_p3_ref = np.abs(df_p3.values - ref_vals)
        diff_nf_ref = np.abs(df_nf.values - ref_vals)

        im3 = axes[3].imshow(diff_p3_ref, cmap="RdBu_r", **kw)
        axes[3].set_title(
            f"|Prob3 − {ref_label}|\nmax={diff_p3_ref.max():.5f}  mean={diff_p3_ref.mean():.6f}"
        )
        plt.colorbar(im3, ax=axes[3], label="|Δ(w·P_ee)|")

        im4 = axes[4].imshow(diff_nf_ref, cmap="RdBu_r", **kw)
        axes[4].set_title(
            f"|NuFast − {ref_label}|\nmax={diff_nf_ref.max():.5f}  mean={diff_nf_ref.mean():.6f}"
        )
        plt.colorbar(im4, ax=axes[4], label="|Δ(w·P_ee)|")

        print(f"\n  Prob3 vs {ref_label}: max={diff_p3_ref.max():.5f}  mean={diff_p3_ref.mean():.6f}")
        print(f"  NuFast vs {ref_label}: max={diff_nf_ref.max():.5f}  mean={diff_nf_ref.mean():.6f}")

    for ax in axes:
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("cos(η)")
        ax.axhline(0.0, ls="--", color="white", lw=0.8, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "combined_oscillogram.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [combined] {path}")

    # Backends must agree within 0.1% after nadir weighting
    assert diff_backends.max() < 1e-3, f"Combined P_ee backends differ: max={diff_backends.max():.6f}"

    # vs library: tolerance accounts for nadir PDF difference (nufast vs nadir.root)
    if ncols == 5:
        assert diff_p3_ref.mean() < 5e-3, f"Prob3 vs library mean diff too large: {diff_p3_ref.mean():.5f}"
        assert diff_nf_ref.mean() < 5e-3, f"NuFast vs library mean diff too large: {diff_nf_ref.mean():.5f}"


# ── Test 3: Energy slices + residuals ─────────────────────────────────────────

def test_energy_slices(computed_oscillograms):
    """
    P_ee(E) at 8 nadir slices for Prob3++ vs NuFast-Earth.
    Bottom panels: |residual| per slice.
    """
    p3 = computed_oscillograms["r_p3"].night.values
    nf = computed_oscillograms["r_nf"].night.values

    ref_vals, ref_label = _load_reference()
    # Reference is weighted; to compare raw P_ee vs reference per nadir bin
    # we need to de-weight — but reference uses a different nadir PDF.
    # So we only compare p3 vs nf in raw slices.

    slice_idx = np.round(np.linspace(0, len(N_CENTERS) - 1, 8)).astype(int)
    colors     = plt.cm.coolwarm(np.linspace(0, 1, len(slice_idx)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Energy slices at fixed nadir    dm²={DM2:.2e}  sin²θ₁₂={SIN12:.3f}",
        fontsize=11,
    )

    ax_pee   = axes[0, 0]
    ax_night = axes[0, 1]
    ax_res   = axes[1, 0]
    ax_rat   = axes[1, 1]

    for idx, col in zip(slice_idx, colors):
        n_val = N_CENTERS[idx]
        lbl   = f"cosη={n_val:+.2f}"
        ax_pee.plot(E_CENTERS, p3[idx],  "-",  color=col, lw=1.5, label=f"Prob3 {lbl}")
        ax_pee.plot(E_CENTERS, nf[idx],  "--", color=col, lw=1.0, alpha=0.8)

        if n_val < 0:
            ax_night.plot(E_CENTERS, p3[idx],  "-",  color=col, lw=1.5, label=f"Prob3 {lbl}")
            ax_night.plot(E_CENTERS, nf[idx],  "--", color=col, lw=1.0, alpha=0.8)

        res = np.abs(p3[idx] - nf[idx])
        ax_res.plot(E_CENTERS, res, color=col, lw=1.2, label=lbl)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(p3[idx] > 1e-6, nf[idx] / p3[idx], 1.0)
        ax_rat.plot(E_CENTERS, ratio, color=col, lw=1.2, label=lbl)

    for ax, title in [
        (ax_pee,   "All nadir  (solid=Prob3++, dashed=NuFast)"),
        (ax_night, "Night only (cosη < 0)"),
        (ax_res,   "|Prob3++ − NuFast|  (raw P_ee)"),
        (ax_rat,   "NuFast / Prob3++  ratio per nadir slice"),
    ]:
        ax.set_xlabel("Energy (MeV)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)

    ax_pee.set_ylabel("P_ee")
    ax_night.set_ylabel("P_ee")
    ax_res.set_ylabel("|ΔP_ee|")
    ax_rat.set_ylabel("ratio")
    ax_rat.axhline(1.0, ls="--", color="gray", lw=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "energy_slices.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [energy slices] {path}")


# ── Test 4: Day / Night asymmetry ─────────────────────────────────────────────

def test_day_night_asymmetry(computed_oscillograms):
    """
    Predicted day/night P_ee asymmetry for DUNE.

      A_DN(E) = 2 * (P_ee_day(E) - P_ee_night(E)) / (P_ee_day(E) + P_ee_night(E))

    where P_ee_day / P_ee_night are computed by splitting the nadir PDF at cosη = 0
    and normalising each half to sum = 1 before weighting.

    Computes for: Prob3++, NuFast-Earth, and (if available) file fixture.
    Saves:
      day_night_asymmetry.png  — A_DN(E) comparison
      day_night_pee.png        — P_ee_day and P_ee_night separately
    """
    r_p3      = computed_oscillograms["r_p3"]
    r_nf      = computed_oscillograms["r_nf"]
    nadir_pdf = computed_oscillograms["nadir_pdf"]

    day_mask   = N_CENTERS >  0.0
    night_mask = N_CENTERS <= 0.0

    def _day_night_pee(night_df_vals, nadir_pdf):
        """Compute P_ee_day(E) and P_ee_night(E) from a full nadir grid."""
        w_day   = nadir_pdf[day_mask]
        w_night = nadir_pdf[night_mask]
        w_day   = w_day   / w_day.sum()   if w_day.sum()   > 0 else w_day
        w_night = w_night / w_night.sum() if w_night.sum() > 0 else w_night
        pee_day   = (night_df_vals[day_mask,   :] * w_day[:, None]).sum(axis=0)
        pee_night = (night_df_vals[night_mask, :] * w_night[:, None]).sum(axis=0)
        return pee_day, pee_night

    def _asymmetry(pee_day, pee_night):
        denom = pee_day + pee_night
        return np.where(denom > 0, 2 * (pee_day - pee_night) / denom, 0.0)

    pee_day_p3,  pee_night_p3  = _day_night_pee(r_p3.night.values, nadir_pdf)
    pee_day_nf,  pee_night_nf  = _day_night_pee(r_nf.night.values, nadir_pdf)
    adn_p3 = _asymmetry(pee_day_p3, pee_night_p3)
    adn_nf = _asymmetry(pee_day_nf, pee_night_nf)

    # Reference: de-weight the fixture to recover per-bin P_ee, then split
    ref_vals, ref_label = _load_reference()
    has_ref = ref_vals is not None and ref_vals.shape == r_p3.night.values.shape
    if has_ref:
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_pee_full = np.where(nadir_pdf[:, None] > 1e-12,
                                    ref_vals / nadir_pdf[:, None], 0.0)
        pee_day_ref, pee_night_ref = _day_night_pee(ref_pee_full, nadir_pdf)
        adn_ref = _asymmetry(pee_day_ref, pee_night_ref)

    # ── Plot 1: A_DN(E) ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"DUNE Day/Night P_ee Asymmetry   dm²={DM2:.2e}  sin²θ₁₂={SIN12:.3f}  lat={LATITUDE_DEG}°N",
        fontsize=11,
    )

    ax_adn  = axes[0]
    ax_diff = axes[1]

    ax_adn.plot(E_CENTERS, adn_p3 * 100, "-",  color="#2196F3", lw=2.0, label="Prob3++")
    ax_adn.plot(E_CENTERS, adn_nf * 100, "--", color="#FF5722", lw=1.5, label="NuFast-Earth")
    if has_ref:
        ax_adn.plot(E_CENTERS, adn_ref * 100, ":", color="#4CAF50", lw=1.5, label=ref_label)

    ax_adn.axhline(0, ls="--", color="gray", lw=0.8, alpha=0.5)
    ax_adn.set_xlabel("Energy (MeV)")
    ax_adn.set_ylabel("A_DN = 2(P_day − P_night)/(P_day + P_night)  [%]")
    ax_adn.set_title("Day/Night P_ee Asymmetry vs Energy")
    ax_adn.legend()
    ax_adn.grid(True, alpha=0.25)

    diff_adn = np.abs(adn_p3 - adn_nf)
    ax_diff.plot(E_CENTERS, diff_adn * 100, color="#9C27B0", lw=1.5, label="|Prob3 − NuFast|")
    if has_ref:
        ax_diff.plot(E_CENTERS, np.abs(adn_p3 - adn_ref) * 100,
                     "--", color="#2196F3", lw=1.2, label=f"|Prob3 − {ref_label}|")
        ax_diff.plot(E_CENTERS, np.abs(adn_nf - adn_ref) * 100,
                     "--", color="#FF5722", lw=1.2, label=f"|NuFast − {ref_label}|")

    ax_diff.set_xlabel("Energy (MeV)")
    ax_diff.set_ylabel("|ΔA_DN|  [%]")
    ax_diff.set_title("Asymmetry difference between backends")
    ax_diff.legend()
    ax_diff.grid(True, alpha=0.25)

    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "day_night_asymmetry.png")
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [asymmetry] {path1}")

    # ── Plot 2: P_ee_day and P_ee_night separately ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Day and Night P_ee(E) for DUNE", fontsize=11)

    axes[0].plot(E_CENTERS, pee_day_p3,  "-",  color="#2196F3", lw=2.0, label="Prob3++")
    axes[0].plot(E_CENTERS, pee_day_nf,  "--", color="#FF5722", lw=1.5, label="NuFast-Earth")
    if has_ref:
        axes[0].plot(E_CENTERS, pee_day_ref, ":", color="#4CAF50", lw=1.5, label=ref_label)
    axes[0].set_title("P_ee day  (cosη > 0, normalized)")
    axes[0].set_ylabel("P_ee")

    axes[1].plot(E_CENTERS, pee_night_p3,  "-",  color="#2196F3", lw=2.0, label="Prob3++")
    axes[1].plot(E_CENTERS, pee_night_nf,  "--", color="#FF5722", lw=1.5, label="NuFast-Earth")
    if has_ref:
        axes[1].plot(E_CENTERS, pee_night_ref, ":", color="#4CAF50", lw=1.5, label=ref_label)
    axes[1].set_title("P_ee night  (cosη ≤ 0, normalized)")
    axes[1].set_ylabel("P_ee")

    axes[2].plot(E_CENTERS, (pee_day_p3 - pee_night_p3) * 100, "-",
                 color="#2196F3", lw=2.0, label="Prob3++")
    axes[2].plot(E_CENTERS, (pee_day_nf - pee_night_nf) * 100, "--",
                 color="#FF5722", lw=1.5, label="NuFast-Earth")
    if has_ref:
        axes[2].plot(E_CENTERS, (pee_day_ref - pee_night_ref) * 100, ":",
                     color="#4CAF50", lw=1.5, label=ref_label)
    axes[2].axhline(0, ls="--", color="gray", lw=0.8, alpha=0.5)
    axes[2].set_title("P_ee_day − P_ee_night  [%]")
    axes[2].set_ylabel("ΔP_ee [%]")

    for ax in axes:
        ax.set_xlabel("Energy (MeV)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "day_night_pee.png")
    plt.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [day/night P_ee] {path2}")

    # Print summary numbers
    nonzero = E_CENTERS > 5.0  # B8 spectrum starts ~5 MeV
    print(f"\n  A_DN at E > 5 MeV (NuFast Solar_Weight PDF):")
    print(f"    Prob3++:      mean={adn_p3[nonzero].mean()*100:.3f}%  "
          f"max={adn_p3[nonzero].max()*100:.3f}%  min={adn_p3[nonzero].min()*100:.3f}%")
    print(f"    NuFast-Earth: mean={adn_nf[nonzero].mean()*100:.3f}%  "
          f"max={adn_nf[nonzero].max()*100:.3f}%  min={adn_nf[nonzero].min()*100:.3f}%")
    if has_ref:
        print(f"    {ref_label}: mean={adn_ref[nonzero].mean()*100:.3f}%")
    print(f"\n  Backend A_DN difference at E > 5 MeV: max={diff_adn[nonzero].max()*100:.4f}%")

    # Sanity checks
    # Both backends should predict night enhancement (P_ee_night > P_ee_day) in some range
    # (Earth MSW regeneration increases nu_e survival at night)
    assert pee_night_p3.max() > pee_day_p3.min(), "Expected some night enhancement from Prob3++"
    assert pee_night_nf.max() > pee_day_nf.min(), "Expected some night enhancement from NuFast"
    # Backends should agree on asymmetry within 0.5% absolute
    assert diff_adn[nonzero].max() < 0.005, \
        f"A_DN backends disagree > 0.5% at E>5MeV: max={diff_adn[nonzero].max()*100:.4f}%"


# ── Test 5: Nadir PDF comparison ───────────────────────────────────────────────

def test_nadir_pdf(computed_oscillograms):
    """Plot nadir PDF from NuFast Solar_Weight vs file fixture values."""
    nadir_pdf = computed_oscillograms["nadir_pdf"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(N_CENTERS, nadir_pdf, width=np.diff(NADIR_EDGES),
           color="#2196F3", alpha=0.7, label="NuFast Solar_Weight (40 bins)")

    fixture_c = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "nadir_pdf_centers.npy"))
    fixture_v = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "nadir_pdf_values.npy"))
    fv_norm   = fixture_v / fixture_v.sum()
    ax.plot(fixture_c, fv_norm * len(fixture_c) / len(N_CENTERS),
            "r-", lw=1.5, alpha=0.8, label="nadir.root (2000 bins, rescaled)")

    ax.axvline(0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("cos(η)")
    ax.set_ylabel("Normalised weight")
    ax.set_title(f"Nadir PDF — DUNE latitude {LATITUDE_DEG}°N  (sum={nadir_pdf.sum():.6f})")
    ax.legend()
    ax.grid(True, alpha=0.25)

    path = os.path.join(OUT_DIR, "nadir_pdf.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"\n  [nadir PDF] {path}")

    assert abs(nadir_pdf.sum() - 1.0) < 1e-10, "Nadir PDF not normalised"


# ── Test 6: Timing vs grid density ────────────────────────────────────────────

def test_timing_vs_grid_density():
    """
    Benchmark both backends across a range of (n_nadir, n_energy) grid sizes.
    Produces:
      timing_vs_total_points.png  — absolute time + speedup vs total grid points
      timing_heatmap.png          — 2D heatmap: nadir bins × energy bins
    """
    grid_sizes = [
        (4,  10), (4,  30), (4,  60), (4,  120),
        (10, 10), (10, 30), (10, 60), (10, 120),
        (20, 10), (20, 30), (20, 60), (20, 120),
        (40, 10), (40, 30), (40, 60), (40, 120),
    ]
    N_REPEATS = 3

    results = []
    print()
    print(f"  {'n_nadir':>8}  {'n_energy':>8}  {'Prob3++ (ms)':>14}  {'NuFast (ms)':>12}  {'speedup':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*8}")

    for n_nadir, n_energy in grid_sizes:
        e_edges = np.linspace(0.0, 30.0, n_energy + 1)
        n_edges = np.linspace(-1.0,  1.0, n_nadir  + 1)

        times_p3 = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            compute_prob3(DM2, SIN13, SIN12, e_edges, n_edges)
            times_p3.append(time.perf_counter() - t0)

        times_nf = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            compute_nufast(DM2, SIN13, SIN12, e_edges, n_edges)
            times_nf.append(time.perf_counter() - t0)

        p3_ms  = np.mean(times_p3) * 1e3
        nf_ms  = np.mean(times_nf) * 1e3
        p3_std = np.std(times_p3)  * 1e3
        nf_std = np.std(times_nf)  * 1e3
        spd    = p3_ms / nf_ms if nf_ms > 0 else float("nan")

        results.append({
            "n_nadir": n_nadir, "n_energy": n_energy,
            "total":   n_nadir * n_energy,
            "p3_ms":   p3_ms,  "nf_ms":   nf_ms,
            "p3_std":  p3_std, "nf_std":  nf_std,
            "speedup": spd,
        })
        print(f"  {n_nadir:>8}  {n_energy:>8}  {p3_ms:>11.1f}ms  {nf_ms:>9.1f}ms  {spd:>8.2f}×")

    totals  = [r["total"]   for r in results]
    p3_ms   = [r["p3_ms"]   for r in results]
    nf_ms   = [r["nf_ms"]   for r in results]
    p3_std  = [r["p3_std"]  for r in results]
    nf_std  = [r["nf_std"]  for r in results]
    speedup = [r["speedup"] for r in results]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Backend timing vs grid density  (3 repeats each)", fontsize=11)

    ax0.errorbar(totals, p3_ms, yerr=p3_std, marker="o", lw=1.5,
                 label="Prob3++", color="#2196F3", capsize=4)
    ax0.errorbar(totals, nf_ms, yerr=nf_std, marker="s", lw=1.5,
                 label="NuFast-Earth", color="#FF5722", capsize=4)
    ax0.set_xlabel("Total grid points  (n_nadir × n_energy)")
    ax0.set_ylabel("Time per call (ms)")
    ax0.set_title("Absolute timing")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(totals, speedup, "o-", color="#4CAF50", lw=1.5)
    ax1.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax1.fill_between(totals, 1.0, speedup,
                     where=[s > 1 for s in speedup], alpha=0.15, color="#4CAF50")
    ax1.set_xlabel("Total grid points")
    ax1.set_ylabel("Prob3++ time / NuFast time")
    ax1.set_title("Speedup factor  (> 1 means NuFast-Earth is faster)")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "timing_vs_total_points.png")
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()

    unique_nadirs   = sorted(set(r["n_nadir"]  for r in results))
    unique_energies = sorted(set(r["n_energy"] for r in results))
    nn, ne = len(unique_nadirs), len(unique_energies)

    p3_grid  = np.full((nn, ne), np.nan)
    nf_grid  = np.full((nn, ne), np.nan)
    spd_grid = np.full((nn, ne), np.nan)

    for r in results:
        i = unique_nadirs.index(r["n_nadir"])
        j = unique_energies.index(r["n_energy"])
        p3_grid[i, j]  = r["p3_ms"]
        nf_grid[i, j]  = r["nf_ms"]
        spd_grid[i, j] = r["speedup"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Timing (ms) as function of  n_nadir (rows) × n_energy (cols)", fontsize=11)
    vmax = max(np.nanmax(p3_grid), np.nanmax(nf_grid))

    for ax, grid, title in [(axes[0], p3_grid, "Prob3++ (ms)"), (axes[1], nf_grid, "NuFast-Earth (ms)")]:
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(ne)); ax.set_xticklabels(unique_energies, fontsize=8)
        ax.set_yticks(range(nn)); ax.set_yticklabels(unique_nadirs,   fontsize=8)
        ax.set_xlabel("n_energy bins")
        ax.set_ylabel("n_nadir bins")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="ms")
        for i in range(nn):
            for j in range(ne):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if grid[i, j] > vmax * 0.5 else "black")

    im2 = axes[2].imshow(spd_grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0)
    axes[2].set_xticks(range(ne)); axes[2].set_xticklabels(unique_energies, fontsize=8)
    axes[2].set_yticks(range(nn)); axes[2].set_yticklabels(unique_nadirs,   fontsize=8)
    axes[2].set_xlabel("n_energy bins")
    axes[2].set_ylabel("n_nadir bins")
    axes[2].set_title("Speedup  Prob3 / NuFast")
    plt.colorbar(im2, ax=axes[2], label="×")
    for i in range(nn):
        for j in range(ne):
            if not np.isnan(spd_grid[i, j]):
                axes[2].text(j, i, f"{spd_grid[i,j]:.1f}×", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "timing_heatmap.png")
    plt.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n  [timing] {path1}")
    print(f"  [timing] {path2}")

    mean_speedup = np.nanmean(spd_grid)
    assert mean_speedup > 1.0, f"NuFast unexpectedly slower: mean speedup={mean_speedup:.2f}"


# ── Test 7: Library load vs compute timing ────────────────────────────────────

def test_library_vs_compute_timing():
    """
    Compare time to obtain an oscillogram from each source:
      - Local pkl  : pd.read_pickle(fixture)                        (disk, fast)
      - PNFS pkl   : pd.read_pickle(PNFS pkl)                       (network, slow)
      - PNFS ROOT  : uproot.open + histogram → DataFrame            (network + processing)
      - Prob3++    : compute_prob3() at full analysis grid (40×120)
      - NuFast     : compute_nufast() at full analysis grid

    All methods produce a (40×120) oscillogram ready for analysis use.
    Saves: library_vs_compute_timing.png
    """
    import uproot
    from scipy import interpolate

    N_REPEATS    = 5
    e_edges      = np.linspace(0.0, 30.0, 121)
    n_edges      = np.linspace(-1.0, 1.0, 41)
    n_centers    = 0.5 * (n_edges[1:] + n_edges[:-1])
    nadir_pdf    = get_nadir_pdf_nufast(n_centers, LATITUDE_DEG)

    timings = {}

    # ── Local fixture pkl ──────────────────────────────────────────────────────
    t = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        pd.read_pickle(FIXTURE_PATH)
        t.append(time.perf_counter() - t0)
    timings["Local pkl\n(fixture)"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

    # ── PNFS pkl ───────────────────────────────────────────────────────────────
    if os.path.exists(PNFS_PKL):
        t = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            pd.read_pickle(PNFS_PKL)
            t.append(time.perf_counter() - t0)
        timings["PNFS pkl\n(pre-rebinned)"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

    # ── PNFS ROOT: uproot load + histogram → DataFrame ────────────────────────
    if os.path.exists(PNFS_ROOT_FILE):
        def _load_root():
            with uproot.open(PNFS_ROOT_FILE) as f:
                hist = f["hsurv;1"]
                arr  = hist.to_hist().to_numpy()
            data        = arr[0][:, :-1]   # [300 E, 1000 nadir night]
            e_centers_r = 0.5 * (arr[1][1:] + arr[1][:-1])
            n_edges_r   = arr[2][:-1]
            n_centers_r = 0.5 * (n_edges_r[1:] + n_edges_r[:-1])
            df1 = pd.DataFrame(data, index=1e3 * e_centers_r, columns=n_centers_r)
            df2 = pd.DataFrame(
                arr[0][:, -1][:, np.newaxis] * np.ones((len(e_centers_r), len(n_centers_r))),
                index=1e3 * e_centers_r, columns=1 + n_centers_r,
            )
            return df1.join(df2).T   # → [2000 nadir, 300 E]

        t = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            _load_root()
            t.append(time.perf_counter() - t0)
        timings["PNFS ROOT\n(raw load)"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

        # ROOT + full nadir convolution + rebin (matches process_oscillation_map)
        def _process_root():
            df = _load_root()   # [2000 nadir, 300 E]
            nadir_data_raw = df.index.values.astype(float)
            nadir_pdf_root = get_nadir_pdf_nufast(nadir_data_raw, LATITUDE_DEG)
            df_weighted    = df.mul(nadir_pdf_root, axis=0)
            # Rebin to 40×120 (same bins as analysis)
            n_out = pd.cut(df_weighted.index, bins=n_edges, labels=n_centers, include_lowest=True)
            return df_weighted.groupby(n_out, observed=True).sum()

        t = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            _process_root()
            t.append(time.perf_counter() - t0)
        timings["PNFS ROOT\n+convolve+rebin"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

    # ── Prob3++ compute ────────────────────────────────────────────────────────
    t = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        r  = compute_prob3(DM2, SIN13, SIN12, e_edges, n_edges)
        combine_day_night(r, nadir_pdf)
        t.append(time.perf_counter() - t0)
    timings["Prob3++\n(compute+combine)"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

    # ── NuFast-Earth compute ───────────────────────────────────────────────────
    t = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        r  = compute_nufast(DM2, SIN13, SIN12, e_edges, n_edges, LATITUDE_DEG)
        combine_day_night(r, nadir_pdf)
        t.append(time.perf_counter() - t0)
    timings["NuFast-Earth\n(compute+combine)"] = (np.mean(t) * 1e3, np.std(t) * 1e3)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(f"  {'Method':<30}  {'Mean (ms)':>10}  {'Std (ms)':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    for label, (mean_ms, std_ms) in timings.items():
        label_1line = label.replace("\n", " ")
        print(f"  {label_1line:<30}  {mean_ms:>10.2f}  {std_ms:>10.2f}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels = list(timings.keys())
    means  = [v[0] for v in timings.values()]
    stds   = [v[1] for v in timings.values()]

    # Colour-code by category
    colors = []
    for lbl in labels:
        if "Local" in lbl:    colors.append("#4CAF50")
        elif "PNFS" in lbl:   colors.append("#FF9800")
        elif "Prob3" in lbl:  colors.append("#2196F3")
        else:                 colors.append("#FF5722")

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels)), 5))
    bars = ax.bar(range(len(labels)), means, yerr=stds,
                  color=colors, capsize=5, width=0.6, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Time per call (ms)")
    ax.set_title(
        f"Oscillogram acquisition time  (40×120 grid, {N_REPEATS} repeats)\n"
        "Green=local disk | Orange=PNFS network | Blue=Prob3++ | Red=NuFast",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "library_vs_compute_timing.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [library timing] {path}")


# ── Test 8: Oscillation-weighted signal KDE vs legacy pkl reconstruction ───────

def test_osc_weight_kde():
    """
    Compare oscillation-weighted signal KDE produced by:
      - Prob3++ backend  (lib_weights._compute_osc_kde_and_exposure)
      - NuFast-Earth backend
      - Legacy pkl reconstruction: same KDE algorithm applied to the ROOT-derived
        PNFS oscillogram pkl (raw, non-weighted).  Skipped if PNFS unavailable.

    These three should produce the same KDE shape and event rate:
      Prob3++ vs NuFast counts:  < 1% relative difference
      Night counts > day counts for every (backend, comp) pair (Earth regeneration)

    Saves:
      osc_weight_kde_comparison.png  — KDE grid: 3 nadir slices × 3 components
      osc_weight_counts.png          — expected event rates per 1 kT·yr
    """
    from lib.weights import (
        _precompute_osc_result, _compute_osc_kde_and_exposure,
        _load_osc_pkl_as_pee, SECONDS_PER_YEAR,
    )
    from lib.oscillation_backends import get_nadir_pdf_nufast
    from lib.solar import get_detected_solar_spectrum
    from sklearn.neighbors import KernelDensity
    from lib import energy_centers as ec, ebin

    _PNFS_PKL_RAW = os.path.join(
        PNFS_ROOT, "pkl", "raw",
        "osc_probability_dm2_6.000e-05_sin13_2.100e-02_sin12_3.030e-01.pkl",
    )

    comps         = ["comb", "b8", "hep"]
    nadir_slices  = ["mean", "day", "night"]
    e_eval   = np.linspace(0, 30, 300)

    # ── On-the-fly backends ───────────────────────────────────────────────────
    kde_results = {}
    for backend in ["prob3", "nufast"]:
        osc, nadir_pdf = _precompute_osc_result(DM2, SIN13, SIN12, backend)
        pee_2d    = osc.night.values
        n_centers = osc.night.index.values
        for comp in comps:
            for nadir_slice in nadir_slices:
                kde, exp = _compute_osc_kde_and_exposure(
                    pee_2d, n_centers, nadir_pdf, comp, nadir_slice
                )
                kde_results[(backend, comp, nadir_slice)] = {
                    "kde_curve": np.exp(kde.score_samples(e_eval[:, None])),
                    "counts":    exp["counts"],
                }

    # ── Legacy: reconstruct KDE from raw PNFS pkl using the same algorithm ────
    has_legacy = os.path.exists(_PNFS_PKL_RAW)
    if has_legacy:
        raw_df          = pd.read_pickle(_PNFS_PKL_RAW)
        n_centers_raw   = raw_df.index.values.astype(float)
        raw_e           = raw_df.columns.values.astype(float)
        pee_raw         = raw_df.values.astype(float)      # (n_nadir, n_energy_raw)
        nadir_pdf_leg   = get_nadir_pdf_nufast(n_centers_raw, LATITUDE_DEG)
        raw_ebin        = float(raw_e[1] - raw_e[0]) if len(raw_e) > 1 else ebin
        _comp_map       = {"comb": ["b8", "hep"], "b8": ["b8"], "hep": ["hep"]}

        for comp in comps:
            eff_flux_raw = get_detected_solar_spectrum(
                bins=raw_e, mass=1e9, components=_comp_map[comp]
            )
            for nadir_slice in nadir_slices:
                if nadir_slice == "mean":
                    mask = np.ones(len(n_centers_raw), dtype=bool)
                elif nadir_slice == "day":
                    mask = n_centers_raw > 0.0
                else:
                    mask = n_centers_raw <= 0.0

                w = nadir_pdf_leg[mask]
                w_sum = w.sum()
                pee_e = (
                    (pee_raw[mask, :] * w[:, None]).sum(axis=0) / w_sum
                    if w_sum > 0 else np.zeros(len(raw_e))
                )
                weighted = np.clip(eff_flux_raw * SECONDS_PER_YEAR * pee_e, 0.0, None)

                kde_leg = KernelDensity(
                    bandwidth=raw_ebin, kernel="gaussian", algorithm="kd_tree"
                ).fit(raw_e[:, None], sample_weight=weighted)

                kde_results[("legacy", comp, nadir_slice)] = {
                    "kde_curve": np.exp(kde_leg.score_samples(e_eval[:, None])),
                    "counts":    float(np.sum(weighted)),
                }

    # ── Plot 1: KDE grid  (3 rows = nadir_slice, 3 cols = component) ─────────────
    colors            = {"prob3": "#2196F3", "nufast": "#FF5722", "legacy": "#4CAF50"}
    labels_nice       = {"prob3": "Prob3++", "nufast": "NuFast-Earth", "legacy": "Legacy pkl"}
    ls_map            = {"prob3": "-", "nufast": "--", "legacy": ":"}
    comp_nice         = {"comb": "Combined (b8+hep)", "b8": "⁸B only", "hep": "hep only"}
    nadir_slice_nice  = {"mean": "Mean (all nadir)", "day": "Day (cosη > 0)", "night": "Night (cosη ≤ 0)"}
    backends_plot = ["prob3", "nufast"] + (["legacy"] if has_legacy else [])

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey="col")
    fig.suptitle(
        f"Oscillation-weighted signal KDE per 1 kT·yr  [log scale]\n"
        f"dm²={DM2:.2e}  sin²θ₁₂={SIN12:.3f}  sin²θ₁₃={SIN13:.3f}\n"
        "Blue=Prob3++  |  Red=NuFast-Earth  |  Green=Legacy pkl  "
        "  (dashed = NuFast, dotted = legacy)",
        fontsize=10,
    )

    for row, nadir_slice in enumerate(nadir_slices):
        for col, comp in enumerate(comps):
            ax = axes[row, col]
            for backend in backends_plot:
                key = (backend, comp, nadir_slice)
                if key not in kde_results:
                    continue
                cnts  = kde_results[key]["counts"]
                curve = kde_results[key]["kde_curve"]
                ax.plot(e_eval, np.clip(curve, 1e-10, None),
                        linestyle=ls_map[backend], color=colors[backend], lw=1.5,
                        label=f"{labels_nice[backend]}  ({cnts:.0f} ev/kT·yr)")
            if row == 0:
                ax.set_title(comp_nice[comp], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{nadir_slice_nice[nadir_slice]}\nKDE density", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2, which="both")
            ax.set_xlim(0, 30)

    # Set column-wise y floor: 4 decades below each column's peak
    for col in range(3):
        col_peak = max(
            kde_results[(b, comps[col], ns)]["kde_curve"].max()
            for b in backends_plot for ns in nadir_slices
            if (b, comps[col], ns) in kde_results
        )
        for row in range(3):
            axes[row, col].set_ylim(bottom=col_peak * 1e-4)

    for ax in axes[-1, :]:
        ax.set_xlabel("Energy (MeV)")

    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "osc_weight_kde_comparison.png")
    plt.savefig(path1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [osc KDE] {path1}")

    # ── Plot 2: Expected event rate bar chart ─────────────────────────────────
    combo_labels = [f"{ns[:3]}/{comp}" for ns in nadir_slices for comp in comps]
    x            = np.arange(len(combo_labels))
    width        = 0.8 / len(backends_plot)

    fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(combo_labels)), 5))
    for i, backend in enumerate(backends_plot):
        counts = [
            kde_results.get((backend, comp, nadir_slice), {}).get("counts", 0)
            for nadir_slice in nadir_slices
            for comp in comps
        ]
        ax.bar(x + (i - len(backends_plot) / 2 + 0.5) * width, counts,
               width=width * 0.9, color=colors[backend], alpha=0.8,
               label=labels_nice[backend])

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Expected events / kT·yr")
    ax.set_title(
        f"Signal event rate after oscillation weighting per 1 kT·yr  [log scale]\n"
        f"dm²={DM2:.2e}  sin²θ₁₂={SIN12:.3f}  sin²θ₁₃={SIN13:.3f}",
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "osc_weight_counts.png")
    plt.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [osc counts] {path2}")

    # ── Print summary table ───────────────────────────────────────────────────
    print()
    print(f"  {'backend':<14} {'comp':<6} {'nadir':<7} {'ev/kT·yr':>10}")
    print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*10}")
    for backend in backends_plot:
        for comp in comps:
            for nadir_slice in nadir_slices:
                key = (backend, comp, nadir_slice)
                if key in kde_results:
                    print(f"  {backend:<14} {comp:<6} {nadir_slice:<7} {kde_results[key]['counts']:>10.1f}")

    # ── Assertions ────────────────────────────────────────────────────────────
    for comp in comps:
        p3_mean  = kde_results[("prob3",  comp, "mean") ]["counts"]
        nf_mean  = kde_results[("nufast", comp, "mean") ]["counts"]
        p3_day   = kde_results[("prob3",  comp, "day")  ]["counts"]
        nf_day   = kde_results[("nufast", comp, "day")  ]["counts"]
        p3_night = kde_results[("prob3",  comp, "night")]["counts"]
        nf_night = kde_results[("nufast", comp, "night")]["counts"]

        rel = abs(p3_mean - nf_mean) / p3_mean
        assert rel < 0.01, (
            f"{comp}/mean: Prob3++({p3_mean:.1f}) vs NuFast({nf_mean:.1f}) "
            f"differ {rel*100:.2f}% — expected < 1%"
        )
        assert p3_night > p3_day, (
            f"Prob3++ {comp}: night({p3_night:.1f}) should exceed day({p3_day:.1f})"
        )
        assert nf_night > nf_day, (
            f"NuFast {comp}: night({nf_night:.1f}) should exceed day({nf_day:.1f})"
        )

    if has_legacy:
        for comp in comps:
            p3   = kde_results[("prob3",   comp, "mean")]["counts"]
            leg  = kde_results[("legacy",  comp, "mean")]["counts"]
            rel  = abs(p3 - leg) / p3
            assert rel < 0.05, (
                f"{comp}: Prob3++({p3:.1f}) vs legacy({leg:.1f}) "
                f"differ {rel*100:.2f}% — expected < 5%"
            )
