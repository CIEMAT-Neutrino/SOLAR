"""
lib/study.py — Study variant context derived from CLI args.

Centralises the study_suffix / template_suffix / save_subfolder derivation
that was previously replicated across every analysis and plot script.

Usage
-----
    from lib import study_context          # available via `from lib import *`

    ctx = study_context(args)
    # ctx.study_suffix     e.g.  "_charge_Q100"  or  ""
    # ctx.template_suffix  same as study_suffix when the variant changes template contents
    # ctx.save_subfolder   e.g.  "truncated/charge_Q100"  or  "truncated/default"
    # ctx.rebin_label(energy)  →  "SolarEnergy_Rebin_charge_Q100"  or  "SolarEnergy_Rebin"

Template suffix is set (equals study_suffix) whenever the variant changes which
pre-scaled template pkl files are needed:
  - charge_threshold > 0   → different event selection
  - dm2_override set       → different oscillation weights
  - exposure != default    → templates are scaled by exposure_yr × mass_kT at creation
                             time, so a different exposure produces different absolute counts
  - truth_fiducial set     → fiducial mask uses truth positions, a different event selection
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from typing_extensions import NotRequired, TypedDict


_DEFAULT_EXPOSURE = 30.0  # canonical reference exposure in years


@dataclass(frozen=True)
class StudyContext:
    study_suffix: str     # "_<label>" or ""
    template_suffix: str  # = study_suffix when variant changes template contents, else ""
    save_subfolder: str   # "<folder>/<label>" or "<folder>/default"

    def rebin_label(self, energy: str) -> str:
        """Rebin pkl filename stem, labeled when template_suffix is set (charge/dm2/exposure variants)."""
        if self.template_suffix:
            return f"{energy}_Rebin{self.study_suffix}"
        return f"{energy}_Rebin"


def study_context(args, folder: Optional[str] = None) -> StudyContext:
    """
    Build a StudyContext from parsed CLI args.

    Parameters
    ----------
    args   : argparse.Namespace — must expose study_label; optionally charge_threshold.
    folder : explicit folder string; falls back to args.folder when None.
    """
    label            = getattr(args, "study_label",     None) or ""
    charge_threshold = getattr(args, "charge_threshold", 0)   or 0
    dm2_override     = getattr(args, "dm2",              None)
    exposure         = getattr(args, "exposure",         None)
    truth_fiducial   = bool(getattr(args, "truth_fiducial", False))
    membrane_veto    = bool(getattr(args, "membrane_veto", True))
    folder_str       = (folder or getattr(args, "folder", "")).lower()

    study_sfx    = f"_{label}" if label else ""
    # Template pkls must be labeled whenever their stored counts differ from the default:
    #   charge_threshold  → different event selection changes the spectrum shape
    #   dm2_override      → different oscillation weights change the spectrum
    #   exposure != default → templates are pre-scaled as exposure_yr × detector_mass_kT × rate,
    #                         so a different exposure produces a different absolute-count array
    #   truth_fiducial    → fiducial mask built from SignalParticleX/Y/Z instead of RecoX/Y/Z,
    #                       a different event selection. Backgrounds have no signal particle, so
    #                       an unlabeled write would zero the nominal gamma/neutron Rebin pkls.
    #   membrane_veto off → accepts VD plane 1-4 optical matches (~22% more VD clusters), a
    #                       different event selection. Without this the membrane study wrote its
    #                       veto-off counts to the nominal Rebin path, so every later study
    #                       reading that path silently inherited veto-off physics.
    has_rebin_variant = (
        (charge_threshold > 0)
        or (dm2_override is not None)
        or (exposure is not None and exposure != _DEFAULT_EXPOSURE)
        or truth_fiducial
        or not membrane_veto
    )
    template_sfx = study_sfx if has_rebin_variant else ""
    subfolder    = f"{folder_str}/{label}" if label else f"{folder_str}/default"

    return StudyContext(
        study_suffix=study_sfx,
        template_suffix=template_sfx,
        save_subfolder=subfolder,
    )


# ---------------------------------------------------------------------------
# Study variant registry
# ---------------------------------------------------------------------------
# Single source of truth for the Chapter 9 study variants. run_studies.py
# orchestrates them; the plot scripts (exposure_plot / significance_plot /
# oscillogram_plot) resolve `--study all` against the same table so the two
# can never drift apart.


# ---------------------------------------------------------------------------
# Variant schema
# ---------------------------------------------------------------------------

class StudyVariant(TypedDict):
    skip_rebin:           bool                         # reuse existing Rebin DataFrames
    skip_best_cuts:       bool                         # skip 04_best_cuts.py phase
    label:                NotRequired[Optional[str]]   # None → folder label used instead
    folder:               NotRequired[Optional[str]]   # None → use --folder from CLI
    fiducialization:      NotRequired[bool]            # default False (skip fiducialization)
    energy_override:      NotRequired[Optional[str]]   # single energy replacing CLI --energy
    analysis_override:    NotRequired[Optional[List[str]]]  # override --analysis for this variant only
    ignore_energy_window: NotRequired[bool]            # pass --ignore_energy_window to run_sensitivity.py
    skip_best_sigmas:     NotRequired[bool]            # pass --skip_best_sigmas (use nominal best cuts)
    extra:                NotRequired[List[str]]       # verbatim flags appended last


# ---------------------------------------------------------------------------
# Study variant definitions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# One-knob policy (applies to EVERY variant below)
# ---------------------------------------------------------------------------
# A study changes exactly one thing and holds the rest of the chain at nominal,
# so its number is comparable to the default and reproducible run to run. We are
# measuring what the knob does, not hunting the best achievable result per variant.
#
#   skip_best_cuts:   True   -> reuse the nominal cut optimisation (04_best_cuts.py)
#   skip_best_sigmas: True   -> reuse the nominal smoothing sigmas
#   fiducialization:  omitted -> reuse the nominal BestFiducials.json
#                               (set True only when the variant's Fiducial_Scan.pkl
#                                cannot exist yet, e.g. a new energy estimator)
#   skip_rebin:       per variant -- False only when the variant changes the
#                     histograms themselves (new energy estimator, charge cut,
#                     different dm2, truth fiducialisation, membrane planes).
#
# Re-optimising cuts per variant would confound the knob with a re-tuned analysis:
# a variant could look better purely because its cuts were re-fit, not because the
# physics improved. Hold the cuts, move one knob, read the difference.
#
# NOTE: skip_best_cuts does NOT suppress 01_daynight.py / 01_hep.py -- each variant
# always computes its own significance grid. See run_sensitivity.py --skip_best_cuts.
STUDY_VARIANTS: dict[str, list[StudyVariant]] = {
    # 9.1.1 — histogram metric / smoothing comparison
    # Raw vs Smoothed results are part of the default pipeline (--all_metrics).
    # No separate study runs needed — extract directly from default output pkls.
    # 9.1.2 — uncertainty impacts
    "unc": [
        # Signal uncertainty — HEP only (DayNight uses σ_sig=0 by statistical design)
        # Scan bracketing the default 30%: tighter (20%) and looser (40%)
        {"label": "unc_sig20", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["HEP"],         "extra": ["--signal_uncertainty", "0.20"]},
        {"label": "unc_sig40", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["HEP"],         "extra": ["--signal_uncertainty", "0.40"]},
        # Signal uncertainty — Sensitivity only; scan bracketing default unc_sig4
        {"label": "unc_sig0",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.00"]},
        {"label": "unc_sig2",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.02"]},
        {"label": "unc_sig6",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.06"]},
        # Background uncertainty — DayNight + Sensitivity; effect enters when σ_bkg²·N_bkg > 1.
        # σ_bkg² · N_bkg > 1  →  N_bkg > 1/σ_bkg²  (6%→278, 4%→625, 2%(default)→2500 events)
        {"label": "unc_bkg0",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--background_uncertainty", "0.00"]},
        {"label": "unc_bkg4",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--background_uncertainty", "0.04"]},
        {"label": "unc_bkg6",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--background_uncertainty", "0.06"]},
        # Background uncertainty with fixed background normalization (physically correct fitting)
        # These variants demonstrate proper behavior: contours loosen with increased σ_bkg
        {"label": "unc_bkg4_nobkgfit", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--background_uncertainty", "0.04", "--no-fit_background"]},
        {"label": "unc_bkg6_nobkgfit", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--background_uncertainty", "0.06", "--no-fit_background"]},
        # Signal uncertainty with fixed background normalization for comparison
        {"label": "unc_sig6_nobkgfit",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.06", "--no-fit_background"]},
    ],
    # 9.1.2 — nuisance parameter decomposition (Sensitivity only)
    # Default profile is 'full' (sin²θ₁₃ + energy scale). Variants isolate each nuisance.
    # DayNight Asimov is σ_bkg-invariant; no sin²θ₁₃/escale enter the LLR.
    # HEP ProfileLikelihood handles its own nuisances inside the PL fit.
    # skip_best_sigmas=True: cuts already optimised at 'full' profile; reuse them here.
    "nuisance": [
        {"label": "nuisance_nominal", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "nominal"]},
        {"label": "nuisance_sin13",   "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "marginalize_sin13"]},
        {"label": "nuisance_escale",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "energy_scale"]},
    ],
    # 9.2.1 — energy variable: energy_override replaces CLI --energy for this variant
    # fiducialization=True required — Fiducial_Scan.pkl for these energies may not exist
    "energy": [
        {"label": "energy_spk",   "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "fiducialization": True, "energy_override": "SignalParticleK", "ignore_energy_window": True},
        {"label": "energy_maink", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "fiducialization": True, "energy_override": "MainK",           "ignore_energy_window": True},
    ],
    # 9.2.2 — fiducialization (folder provides isolation; no study_label needed)
    "fiduc": [
        {"folder": "Nominal",   "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True},
        {"folder": "Reduced",   "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True},
        {"folder": "Truncated", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True},
    ],
    # 9.2.3 — charge threshold scan
    # AdjCl energy features are recomputed with AdjClCharge > Q before the Rebin pkl is
    # written, so the energy axis itself reflects the charge cut — not just event selection.
    # SelectedEnergy (= Energy + SelectedAdjClEnergy) is used as the analysis metric:
    # it is a direct calorimetric sum that needs no BDT retraining.
    #
    # charge_Q0 is the reference point for this scan, NOT the unlabeled default. The default
    # uses SolarEnergy (BDT-based), so a delta taken against it would fold the charge cut
    # together with an energy-estimator swap and break the one-knob rule. charge_Q0 is the
    # same SelectedEnergy estimator with no charge cut, which isolates the knob: quote charge
    # deltas as Q_n - Q0. It carries no --charge_threshold, so study_context leaves
    # template_suffix empty and it reads/writes the unlabeled SelectedEnergy Rebin pkls —
    # that is deliberate, those are exactly the Q=0 histograms. skip_rebin=False because
    # those unlabeled SelectedEnergy Rebins do not exist yet and must be produced for signal
    # and every background sample.
    "charge": [
        {"label": "charge_Q0",   "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "energy_override": "SelectedEnergy"},
        {"label": "charge_Q50",  "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold",  "50"]},
        {"label": "charge_Q100", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold", "100"]},
        {"label": "charge_Q500", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold", "500"]},
    ],
    # 9.2.4 — background model normalization (folder provides isolation)
    "bkgmodel": [
        {"folder": "Nominal", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True},
        {"folder": "Reduced", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True},
    ],
    # 9.1.3 — oscillation best-fit point: solar (Δm²₂₁=6e-5) vs reactor (Δm²₂₁=7.54e-5)
    # Solar variant reuses nominal Rebin pkls (skip_rebin=True); reactor variant regenerates
    # Rebin pkls with the reactor dm2 point (skip_rebin=False) using a labeled filename to
    # avoid overwriting the nominal solar-dm2 Rebin.  Background Rebin pkls are dm2-independent
    # (backgrounds use Truth weights, not oscillation weights) and are reused unchanged.
    # Sensitivity stage is skipped for both variants: the Score is invariant to Δm²₂₁ because
    # the discrimination is always computed between solar and reactor dm² templates regardless
    # of which point the signal MC was simulated at (Score(oscpoint_reactor) = Score(default)).
    "oscpoint": [
        {"label": "oscpoint_solar",   "skip_rebin": True,  "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["DayNight", "HEP"]},
        {"label": "oscpoint_reactor", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--dm2", "7.54e-5"], "analysis_override": ["DayNight", "HEP"]},
    ],
    # 9.2.2 / 9.2.3 — truth x-fiducialisation vs reco flash-matching
    # Runs full fiducialization with SignalParticleX/Y/Z instead of RecoX/Y/Z.
    # Produces BestFiducials_fiduc_truth.json and labeled Rebin pkls.
    # Background scans always use nominal coordinates (no truth position available).
    "fiduc_truth": [
        {
            "label": "fiduc_truth",
            "skip_rebin": False,
            "skip_best_cuts": True,
            "skip_best_sigmas": True,
            "fiducialization": True,
            "extra": ["--truth_fiducial"],
        },
    ],
    # 9.2.5 — background gamma model: ClusterEnergy vs TotalEnergy comparison
    # Variant 1: ClusterEnergy (sum of cluster hit charge × calibration) as direct calorimetric proxy
    # Variant 2: TotalEnergy as comparison baseline (bkg_gamma included)
    # ClusterEnergy is already computed in the reco workflow; no new simulation needed.
    # fiducialization=True required — Fiducial_Scan.pkl for these energies may not exist.
    "bkg_gamma": [
        {
            "label": "bkg_gamma_cluster",
            "skip_rebin": False,
            "skip_best_cuts": True,
            "skip_best_sigmas": True,
            "fiducialization": True,
            "energy_override": "ClusterEnergy",
            "ignore_energy_window": True,
        },
        {
            "label": "bkg_gamma_total",
            "skip_rebin": False,
            "skip_best_cuts": True,
            "skip_best_sigmas": True,
            "fiducialization": True,
            "energy_override": "TotalEnergy",
            "ignore_energy_window": True,
        },
    ],
    # 9.2.6 — membrane veto: which optical planes may supply the TPC-PDS match.
    # QUALITY_CUTS.OPFLASH_PLANE == 0 keeps cathode (VD) / APA (HD) matches only, and
    # stays the default everywhere. HD reports no other plane, so the veto is free
    # there; VD also reports Membrane 1/2 and Front/EndCap (planes 1-4), which carry
    # ~22% of its clusters and reconstruct the drift coordinate essentially as well as
    # the cathode does (>94% of them within 10 cm of truth, against 96.8% for plane 0).
    # This variant lifts the veto so the membrane-matched signal and background events
    # enter the analysis, and measures what they are worth downstream.
    # Only the "off" arm runs: the "on" arm is the default pipeline, so compare against
    # the unlabeled default outputs. VD-only in practice -- an HD run reproduces the
    # default bit for bit and is useful mainly as a null check.
    #
    # Everything except the event selection is held at nominal, so the comparison
    # isolates the membrane events themselves rather than a re-tuned analysis:
    #   fiducialization omitted (default False) -> reuse the nominal BestFiducials.json
    #   skip_best_cuts=True                     -> reuse the nominal cut optimisation
    #   skip_best_sigmas=True                   -> reuse the nominal smoothing sigmas
    # skip_rebin stays False because the Rebin pkls are the one thing that must change:
    # they carry the histograms, and admitting the membrane planes changes which signal
    # and background events fill them. 03_analysis.py runs over every sample, signal and
    # background alike, and the sensitivity background templates are built from those
    # same Rebin pkls, so both sides pick the change up.
    "membrane_veto": [
        {
            "label": "membrane_veto_off",
            "skip_rebin": False,
            "skip_best_cuts": True,
            "skip_best_sigmas": True,
            "extra": ["--no-membrane_veto"],
        },
    ],
}

ALL_GROUPS: list[str] = list(STUDY_VARIANTS.keys())


def all_study_labels(
    analysis: Optional[str] = None,
    include_default: bool = True,
) -> List[Optional[str]]:
    """
    Every study label defined in STUDY_VARIANTS, in registry order.

    Used by the plot scripts to expand `--study all` so a single invocation
    covers the whole variant matrix without the caller enumerating labels by
    hand (and silently missing the ones added since they last looked).

    Parameters
    ----------
    analysis : "DayNight" | "HEP" | "Sensitivity" | None
        When given, variants carrying an `analysis_override` that excludes this
        analysis are dropped -- e.g. `unc_sig20` is HEP-only and `nuisance_*`
        Sensitivity-only, so asking for DayNight labels never returns them.
        None keeps every label regardless of analysis.
    include_default : bool
        Prepend None (the unlabeled nominal run) to the list.

    Notes
    -----
    Folder-isolated groups (`fiduc`, `bkgmodel`) define no "label" -- they vary
    --folder instead -- so they are absent here by construction. Iterate
    --folder Nominal/Reduced/Truncated to cover them.
    """
    labels: List[Optional[str]] = [None] if include_default else []
    for variants in STUDY_VARIANTS.values():
        for variant in variants:
            label = variant.get("label")
            if not label:
                continue
            override = variant.get("analysis_override")
            if analysis is not None and override and analysis not in override:
                continue
            if label not in labels:
                labels.append(label)
    return labels


def resolve_study_labels(
    study_args: Optional[List[str]],
    study_label: Optional[str],
    analysis: Optional[str] = None,
) -> List[Optional[str]]:
    """
    Turn the plot scripts' --study / --study_label flags into labels to iterate.

    "all"     -> every label for this analysis, nominal run included
    "default" -> None (the unlabeled nominal run)
    absent    -> [study_label], preserving single-study behaviour
    """
    if not study_args:
        return [study_label]
    if any(s.lower() == "all" for s in study_args):
        return all_study_labels(analysis=analysis)
    return [None if s == "default" else s for s in study_args]
