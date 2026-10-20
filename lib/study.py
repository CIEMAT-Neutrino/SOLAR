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

import os
from dataclasses import dataclass
from typing import List, Optional

from typing_extensions import NotRequired, TypedDict

from src.utils import get_project_root


# An analysis has TWO reference exposures, both analysis dependent and both defined in
# config/analysis/config.json (see lib/defaults.py): the exposure it is run to
# (ANALYSIS_EXPOSURES, 30 yr) and the livetime its numbers are quoted at
# (EVALUATION_EXPOSURE_YEARS, 20 yr for DayNight/HEP, 30 yr for Sensitivity).
# Scripts fill --exposure from one or the other -- 01_daynight.py and the template
# stages from the first, significance_plot.py and the cutflow from the second -- so a
# variant is only a template variant when the exposure matches NEITHER. Comparing
# against a single reference makes has_rebin_variant spuriously True for every study
# run by the other half of the pipeline -- e.g. unc_bkg/charge_Q0/bkg_gamma_*, which
# must never get a labeled Rebin, or the Sensitivity templates, which would then be
# read from an {energy}_<label>/ subfolder nothing ever wrote.
def _reference_exposures_for(analysis: Optional[object]) -> tuple:
    """The exposures in years that count as 'default' for this analysis."""
    from .defaults import get_analysis_exposure, get_evaluation_exposure

    root_str = str(get_project_root())
    return (
        float(get_analysis_exposure(root_str, analysis, stage="PRIMARY", fallback=30.0)),
        float(get_evaluation_exposure(root_str, analysis, fallback=20.0)),
    )


def warn_if_not_best_cut(cut, best_map, key, *, stage: str, reason: str = "") -> bool:
    """Warn when `stage` works at a cut that is not the best cut for `key`.

    `cut` is any mapping with NHits/AdjCl/OpHits; `best_map` the highest_*.pkl dict
    ({(config, name, energy): {"NHits": ..., ...}}) or None. Returns True when a
    non-best cut is in use. A cut scan (many cuts, before 04_best_cuts.py) is not a
    deviation and must not call this; an explicitly requested cut is.
    """
    from rich import print as rprint

    keys = ("NHits", "AdjCl", "OpHits")
    used = tuple(int(cut[k]) for k in keys)
    _fmt = lambda c: f"NHits{c[0]} AdjCl{c[1]} OpHits{c[2]}"
    _why = f" ({reason})" if reason else ""

    if not best_map:
        rprint(
            f"[yellow][WARNING][/yellow] {stage}: using {_fmt(used)}{_why} with no best-cut map "
            "available — cannot verify this is the optimised cut."
        )
        return True

    entry = best_map.get(key)
    if entry is None:
        # Energy-override variants hold the cuts of the single (config, name, *) entry.
        same = [v for k, v in best_map.items() if k[:2] == key[:2]]
        entry = same[0] if len(same) == 1 else None
    if entry is None:
        rprint(
            f"[yellow][WARNING][/yellow] {stage}: using {_fmt(used)}{_why}; the best-cut map has "
            f"no entry for {key} — cannot verify this is the optimised cut."
        )
        return True

    best = tuple(int(entry[k]) for k in keys)
    if used != best:
        rprint(
            f"[yellow][WARNING][/yellow] {stage}: using {_fmt(used)}{_why}, which is NOT the best "
            f"cut {_fmt(best)} selected by 04_best_cuts.py for {key}. Results are evaluated at a "
            "sub-optimal working point and are not comparable to best-cut numbers."
        )
        return True
    return False


@dataclass(frozen=True)
class StudyContext:
    study_suffix: str     # "_<label>" or ""
    template_suffix: str  # = study_suffix when variant changes template contents, else ""
    save_subfolder: str   # "<folder>/<label>" or "<folder>/default"
    # "_<label>" for truth-fiducial variants, else "". Suffix of the event-level exports of
    # 03_analysis.py (Ref arrays, FiducializationMask, AnalysisMask, analysis_cuts): those
    # variants share energy and folder with the nominal run, so unlabeled exports would
    # replace the nominal reco-fiducial masks with truth-position ones.
    ref_suffix: str = ""

    def rebin_label(self, energy: str) -> str:
        """Rebin pkl filename stem, labeled when template_suffix is set (charge/dm2/exposure variants)."""
        if self.template_suffix:
            return f"{energy}_Rebin{self.study_suffix}"
        return f"{energy}_Rebin"


def study_context(args, folder: Optional[str] = None, analysis: Optional[str] = None) -> StudyContext:
    """
    Build a StudyContext from parsed CLI args.

    Parameters
    ----------
    args     : argparse.Namespace — must expose study_label; optionally charge_threshold.
    folder   : explicit folder string; falls back to args.folder when None.
    analysis : analysis this script belongs to ("DayNight"/"HEP"/"Sensitivity"); falls back
               to args.analysis. It selects the reference exposures the variant check compares
               against, so single-analysis scripts that have no --analysis flag must pass it
               explicitly — otherwise a script's own analysis default reads as a non-default
               exposure and every study gets a spurious labeled template.
    """
    label            = getattr(args, "study_label",     None) or ""
    charge_threshold = getattr(args, "charge_threshold", 0)   or 0
    dm2_override     = getattr(args, "dm2",              None)
    exposure         = getattr(args, "exposure",         None)
    analysis         = analysis or getattr(args, "analysis", None)
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
        or variant_labels_rebin(label)
        or (dm2_override is not None)
        or (exposure is not None and exposure not in _reference_exposures_for(analysis))
        or truth_fiducial
        or not membrane_veto
    )
    template_sfx = study_sfx if has_rebin_variant else ""
    subfolder    = f"{folder_str}/{label}" if label else f"{folder_str}/default"

    return StudyContext(
        study_suffix=study_sfx,
        template_suffix=template_sfx,
        save_subfolder=subfolder,
        # --no-update_best_fiducials runs (SOLAR_FIDUCIAL_STEM) label their event-level exports as well
        ref_suffix=study_sfx if (truth_fiducial or os.environ.get("SOLAR_FIDUCIAL_STEM")) else "",
    )


def variant_labels_rebin(label: Optional[str]) -> bool:
    """True for a registry variant whose extra flags change the Rebin contents (labeled_rebin)."""
    if not label:
        return False
    return any(v.get("label") == label and v.get("labeled_rebin", False) for vs in STUDY_VARIANTS.values() for v in vs)


def study_variant_extra(label: Optional[str]) -> List[str]:
    """The verbatim "extra" flags of the STUDY_VARIANTS entry with this label ([] if none)."""
    if not label:
        return []
    for variants in STUDY_VARIANTS.values():
        for variant in variants:
            if variant.get("label") == label:
                return list(variant.get("extra", []))
    return []


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
    reference_energy:     NotRequired[str]             # hold this energy's BestFiducials volume (--reference_energy) instead of scanning
    skip_best_sigmas:     NotRequired[bool]            # pass --skip_best_sigmas (use nominal best cuts)
    skip_templates:       NotRequired[bool]            # pass --skip-templates (reuse existing template pkls)
    excluded_configs:     NotRequired[List[str]]       # configs this variant does not run for
    extra:                NotRequired[List[str]]       # verbatim flags appended last
    labeled_rebin:        NotRequired[bool]            # the variant changes the Rebin contents through its extra flags
    superseded:           NotRequired[bool]            # replaced by a later default; runs only when named with --variant


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
        {"label": "unc_sig20", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["HEP"],         "extra": ["--signal_uncertainty", "0.20"]},
        {"label": "unc_sig40", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["HEP"],         "extra": ["--signal_uncertainty", "0.40"]},
        # Signal uncertainty — Sensitivity only; scan bracketing default unc_sig4
        {"label": "unc_sig0",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.00"]},
        {"label": "unc_sig2",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.02"]},
        {"label": "unc_sig6",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--signal_uncertainty", "0.06"]},
        # Background uncertainty — DayNight + HEP. Sensitivity is excluded: the pull fit pins the
        # background normalisation to sigma_data ~ B^-1/2 ~ 5e-7 at the nominal cut, four orders of
        # magnitude tighter than any 0-6% prior, so unc_bkg0/4/6 give identical Sensitivity results
        # (see [[sensitivity-background-uncertainty-degenerate]]). Effect enters for DayNight when
        # sigma_bkg^2 * N_bkg > 1  ->  N_bkg > 1/sigma_bkg^2  (6%->278, 4%->625, 2%(default)->2500 events).
        {"label": "unc_bkg0",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["DayNight", "HEP"], "extra": ["--background_uncertainty", "0.00"]},
        {"label": "unc_bkg4",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["DayNight", "HEP"], "extra": ["--background_uncertainty", "0.04"]},
        {"label": "unc_bkg6",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["DayNight", "HEP"], "extra": ["--background_uncertainty", "0.06"]},
    ],
    # 9.1.2 — nuisance parameter decomposition (Sensitivity only)
    # Default profile is 'full' (sin²θ₁₃ + energy scale). Variants isolate each nuisance.
    # DayNight Asimov is σ_bkg-invariant; no sin²θ₁₃/escale enter the LLR.
    # HEP ProfileLikelihood handles its own nuisances inside the PL fit.
    # skip_best_sigmas=True: cuts already optimised at 'full' profile; reuse them here.
    "nuisance": [
        {"label": "nuisance_nominal", "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "nominal"]},
        {"label": "nuisance_sin13",   "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "marginalize_sin13"]},
        {"label": "nuisance_escale",  "skip_rebin": True, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": True, "analysis_override": ["Sensitivity"], "extra": ["--nuisance_profiles", "energy_scale"]},
    ],
    # 9.2 — energy variable: energy_override replaces CLI --energy for this variant
    # fiducialization=True required — Fiducial_Scan.pkl for these energies may not exist
    # analysis_override=["DayNight"] used to restrict this study: with the analysis energy
    # window applied, SignalParticleK/MainK have no events in the HEP (14-30 MeV) or
    # Sensitivity windows, so 04_best_cuts.py / cutflow_plot.py had no best-cuts entry to key on
    # and hard-failed with a KeyError. Lifted 2026-09-20 now that ignore_energy_window bypasses
    # the window; if HEP/Sensitivity fail again with that KeyError, restore the restriction.
    #
    # skip_best_sigmas was True (held at the SolarEnergy-tuned reference cut) until
    # 2026-09-18: on hd_1x2x6_centralAPA that read 0.04 sigma for energy_spk against a 5.47
    # sigma default, not because SignalParticleK is a poor DayNight discriminant (its own best
    # cut reaches double digits sigma) but because the reference cut is not tuned for it and,
    # at that cut, radiological piles ~6e7 weighted counts into the truth-energy window with
    # no MC-support gate protecting the comparison (see [[energy-study-held-cut-near-zero]]).
    # Same remedy as fiduc_truth (see below): free the cut search so each energy estimator is
    # read at its own optimum, now that the MC-support gate (adaptive_mc_thresholds, applied
    # in 01_daynight.py/01_hep.py/04_best_cuts.py since 2026-09-18) keeps that optimum honest.
    # 2026-09-21: the fiducial volume is now HELD at the SolarEnergy reference (reference_energy). The un-cut
    # fiducial scan on a truth-energy variable is background-dominated (central DayNight SignalParticleK
    # ~0.02 sigma at every volume; MainK 2 vs SolarEnergy 3.7), so its argmax was noise (20/0/20) and
    # the study read 0.63 sigma vs 5.47 -- a fiducial artefact, not the estimator. One knob: energy only.
    "energy": [
        {"label": "energy_spk",   "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False, "fiducialization": True, "energy_override": "SignalParticleK", "ignore_energy_window": True, "reference_energy": "SolarEnergy", "extra": ["--max_sparse_weight_fraction", "0.25"]},
        {"label": "energy_maink", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False, "fiducialization": True, "energy_override": "MainK",           "ignore_energy_window": True, "reference_energy": "SolarEnergy", "extra": ["--max_sparse_weight_fraction", "0.25"]},
    ],
    # 9.4 — charge threshold scan
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
    # 9.5 — background model normalization (folder provides isolation)
    # --reference_folder Truncated: the Nominal/Reduced runs hold the Truncated fiducial volumes
    # (BestFiducials.json) and the Truncated DayNight/HEP/Sensitivity best cuts, so only the
    # background model differs from the reference. A folder with more background (Nominal) can
    # then only come out worse, and one with less (Reduced) only better. skip_rebin/skip_templates
    # False: the spectra and templates must be rebuilt at the held volume and cut.
    # excluded_configs: vd_1x8x14_3view_30deg_shielded does not run this study (2026-09-22, by request).
    "bkgmodel": [
        {"label": "bkgmodel_nominal", "folder": "Nominal", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": False,
         "excluded_configs": [], "extra": ["--reference_folder", "Truncated"]},
        {"label": "bkgmodel_reduced", "folder": "Reduced", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "skip_templates": False,
         "excluded_configs": ["vd_1x8x14_3view_30deg_shielded"], "extra": ["--reference_folder", "Truncated"]},
        # Pessimistic: the Truncated selection with the unscaled external gamma/neutron rates (the other folders
        # divide them by background_model_factors, 12.57 for HD and VD). The full chain (fiducial volume,
        # cuts, sigmas) is re-optimised for this background level.
        {"label": "bkgmodel_pessimistic", "folder": "Pessimistic", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False,
         "skip_best_sigmas": False, "skip_templates": False},
    ],
    # 9.1.3 — oscillation best-fit point: solar (Δm²₂₁=6e-5) vs reactor (Δm²₂₁=7.54e-5)
    # Solar variant reuses nominal Rebin pkls (skip_rebin=True); reactor variant regenerates
    # Rebin pkls with the reactor dm2 point (skip_rebin=False) using a labeled filename to
    # avoid overwriting the nominal solar-dm2 Rebin.  Background Rebin pkls are dm2-independent
    # (backgrounds use Truth weights, not oscillation weights) and are reused unchanged.
    # Sensitivity stage is skipped for both variants: the Score is invariant to Δm²₂₁ because
    # the discrimination is always computed between solar and reactor dm² templates regardless
    # of which point the signal MC was simulated at (Score(oscpoint_reactor) = Score(default)).
    # S/B classifier working point above the 12 MeV gate: 90% instead of the calibrated 95% CC completeness
    # (user, 2026-10-07: a study for the VD Shielded HEP analysis only). Fiducial volume and cuts held at
    # the nominal Truncated ones; only the Rebin pkls (labeled) and the HEP significance are rebuilt.
    # Analysis without the signal/background classifier (user, 2026-10-15, comparison page): SolarEnergy keeps the
    # de-excitation offset for every cluster and the Sensitivity fit has no score categories; the neutron-capture veto
    # is off as well (user). Full chain re-optimised (volume, cuts, sigmas), as the 2026-10-08 diag_sboff.
    # Simplified S/B + neutron-capture set (2026-10-18, user; cluster-level scan SOLAR-sb _runs/sbeval_20261017/sbscan):
    # no S/B decision below the 12 MeV gate, and/or the veto classifier over the Day-Night window 6-18 MeV (model
    # neutron_capture_6_18, 03b --target ncap --ncap_energy_max 18). Full chains re-optimised.
    "sbsimple": [
        {"label": "sb_simple", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--no-sb_low_threshold", "--ncveto_model", "neutron_capture_6_18"]},
        {"label": "sb_gateonly", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--no-sb_low_threshold"]},
        {"label": "ncveto_618", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--ncveto_model", "neutron_capture_6_18"]},
        # the previous default (2026-10-04 to 2026-10-18): SB_THRESHOLD below the 12 MeV gate as well (config SB_LOW_THRESHOLD
        # false since 2026-10-18, so sb_gateonly equals the default from then on)
        {"label": "sb_lowthr", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--sb_low_threshold"]},
        # the simplified set with the HEP background of hep_looseshape_win (no tail extrapolation)
        {"label": "sb_simple_hepwin", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-sb_low_threshold", "--ncveto_model", "neutron_capture_6_18", "--no-hep_tail_extrapolation",
                   "--hep_bkg_shape", "loose_window"]},
    ],
    "sboff": [
        {"label": "sb_off", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--no-sb_classifier", "--no-ncveto"]},
        # the neutron-capture veto on its own: classifier off, veto as in the default (2026-10-17, user)
        {"label": "sb_off_veto", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--no-sb_classifier"]},
    ],
    # Every no-offset cluster (S/B background-like or neutron-capture veto) removed instead of kept without the offset
    # (user, 2026-10-19, after the cut-flow display variant mcfilter_noflag). Full chain re-optimised (volume, cuts, sigmas).
    "sbremove": [
        {"label": "sb_remove", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--remove_no_offset"]},
        # the same with the radiological component of the Sensitivity background template smoothed (as rad_smooth; user,
        # 2026-10-19). Sensitivity only; full chain re-optimised.
        {"label": "sb_remove_radsmooth", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"],
         "extra": ["--remove_no_offset", "--radiological_smoothing"]},
        # sb_remove at the default fiducial volumes (no volume scan; cuts and sigmas re-optimised), to separate the volume
        # change from the removal itself (user, 2026-10-19)
        {"label": "sb_remove_fixfid", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--remove_no_offset"]},
        # final candidate (user, 2026-10-19): sb_remove with the radiological Sensitivity smoothing, and the fiducial volume
        # scanned without the radiological background (02_best_fiducial.py --exclude_radiological_from_scan: dropped from the
        # significance sum and the MC-support gate). All analyses; full chain re-optimised.
        {"label": "sb_remove_rs_noradfid", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True,
         "extra": ["--remove_no_offset", "--radiological_smoothing", "--exclude_radiological_from_scan"]},
        # fallback for the Sensitivity if the full smoothing costs too much (user, 2026-10-19): the radiological smoothing
        # stopped at its MC endpoint (as rad_smooth_end), otherwise as sb_remove_rs_noradfid. Sensitivity only.
        {"label": "sb_remove_rse_noradfid", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"],
         "extra": ["--remove_no_offset", "--radiological_smoothing_endpoint", "--exclude_radiological_from_scan"]},
        # final candidate 2 (2026-10-19): the default fiducial volumes (as sb_remove_fixfid, which reproduces rerun 38 in HD
        # Central Day-Night, 5.50 vs 5.52) with the radiological smoothing stopped at its MC endpoint (full smoothing costs
        # HD Central Sensitivity 5.70 -> 3.09). Sensitivity only: Day-Night / HEP are sb_remove_fixfid (the smoothing knob
        # only reaches the Sensitivity background template).
        {"label": "sb_remove_rse_fixfid", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"],
         "extra": ["--remove_no_offset", "--radiological_smoothing_endpoint"]},
        # the default (no-offset removal, no radiological smoothing) with the fiducial volumes re-optimised instead of held at
        # the rerun-38 volumes (final_run.sh -F): which volumes the optimiser would choose under the rerun-39 settings (2026-10-20)
        {"label": "fid_rescan", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": []},
    ],
    # Radiological treatment (user, 2026-10-20): the default is the per-bin MC floor 3 (backgrounds.json
    # RADIOLOGICAL_MIN_MC_PER_BIN) with the regular radiological smoothing in every stage (smoothing.json,
    # lib/smoothing.radiological_smoothing_mode); the hep profile likelihood stays on raw spectra.
    # SUPERSEDED by that default (kept to read their old labelled outputs; their results ran on the rerun-38/39 defaults,
    # floor 2 and the radiological background unsmoothed, so rerunning them now measures something else):
    #   rad_smooth (Sensitivity template smoothed, 2026-10-15), rad_smooth_end (the same stopped at the MC endpoint,
    #   2026-10-19), rad_floor3 / rad_floor3_smooth / rad_floor3_smooth_end (floor 3 without / with / with endpoint-stopped
    #   Sensitivity smoothing, 2026-10-20), rad_unified_smooth (floor 3 + regular smoothing everywhere = the default since
    #   2026-10-20; validated configuration, SOLAR-sb bf1446a).
    "radsmooth": [
        {"label": "rad_smooth", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"], "extra": ["--radiological_smoothing"]},
        {"label": "rad_smooth_end", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"], "extra": ["--radiological_smoothing_endpoint"]},
        {"label": "rad_floor3", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"], "extra": ["--radiological_min_mc", "3"]},
        {"label": "rad_floor3_smooth", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"], "extra": ["--radiological_min_mc", "3", "--radiological_smoothing"]},
        {"label": "rad_floor3_smooth_end", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["Sensitivity"], "extra": ["--radiological_min_mc", "3", "--radiological_smoothing_endpoint"]},
        {"label": "rad_unified_smooth", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--radiological_min_mc", "3", "--radiological_smoothing_unified"]},
        # rad_default_check: the default with labelled outputs (validation of the promoted default against rad_unified_smooth,
        # 2026-10-20); runs only when named.
        {"label": "rad_default_check", "superseded": True, "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False,
         "skip_best_sigmas": False, "skip_templates": False, "labeled_rebin": True, "extra": []},
        # Variations of the default, all analyses, full chain re-optimised:
        # rad_unified: the radiological smoothing stopped at its MC endpoint in every stage (floor 3 kept).
        {"label": "rad_unified", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--radiological_min_mc", "3", "--radiological_smoothing_endpoint"]},
        # rad_off: the rerun-39 radiological treatment (floor 2, radiological never smoothed, summed-background smoothing in
        # the fiducial figure of merit) on the current analysis.
        {"label": "rad_off", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--radiological_min_mc", "2", "--no-radiological_smoothing"]},
    ],
    # HEP without the external-tail extrapolation (HEP_TAIL_EXTRAPOLATION off; user, 2026-10-16, comparison page).
    # HEP only, full chain re-optimised.
    "hepnotail": [
        {"label": "hep_notail", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"], "extra": ["--no-hep_tail_extrapolation"]},
    ],
    # Fiducial and HEP-energy comparisons (2026-10-16, user; run with --no-update_best_fiducials):
    # fid_oneband: one fiducial volume over the full analysis window instead of one per energy band; full chain re-optimised.
    "fidband": [
        {"label": "fid_oneband", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--single_fiducial_band"]},
        # fid_zlim: beam-direction margin limited to the vertical reach of the grid (user, 2026-10-16: test whether F_z is
        # picked only because its range is longer); full chain re-optimised.
        {"label": "fid_zlim", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_z_limit_y"]},
        # fid_exp30 (user, 2026-10-17): fiducial figure of merit at 30 yr (the analysis benchmark) instead of 100 yr.
        {"label": "fid_exp30", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_exposure", "30"]},
        # fid_hepbands22 (user, 2026-10-17): HEP fiducial bands 14-22 and 22-30 MeV instead of 14-20 and 20-30; HEP only.
        {"label": "fid_hepbands22", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"], "extra": ["--fiducial_bands", "HEP=14-22,22-30"]},
        # fid_bandmin1 / fid_bandrel5 (user, 2026-10-17): a band keeps its own volume only if its best fiducial-stage
        # significance is >= 1 (absolute) / >= 5 % of the full-window one; otherwise it takes the full-window volume.
        {"label": "fid_bandmin1", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_band_min_z", "abs:1.0"]},
        {"label": "fid_bandrel5", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_band_min_z", "rel:0.05"]},
        # fid_highband: the volume chosen for the highest energy band applied to every event (user, 2026-10-17: HD Central
        # HEP, where the full-window volume is the 14-20 MeV band's); full chain re-optimised.
        {"label": "fid_highband", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "high"]},
        # fid_bestband (user, 2026-10-17): the volume of the band with the highest fiducial-stage significance applied to
        # every event (one volume per analysis); full chain re-optimised.
        {"label": "fid_bestband", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "best"]},
        # Single-volume candidates for Day-Night (user, 2026-10-17: one volume per analysis without losing HD Central
        # Day-Night). fid_lowband: the 6-12 MeV band volume for every event; fid_v*: fixed volumes X_Y_Z in cm (HD Central:
        # X/Y of the low band with the high-band Z, the midpoint, the high-band X/Y with the low-band Z; VD Top: both bands'
        # margins combined). Run with --analysis DayNight on the config named in the comment.
        # fid_radfloor / fid_radfloor_one (2026-10-17): the fiducial figure of merit ignores radiological bins backed by fewer
        # than 2 simulated clusters, as 03_analysis.py then did (RADIOLOGICAL_MIN_MC_PER_BIN, 3 since 2026-10-20: these
        # studies now loosen the fiducial floor to 2); with the default bands, or with one volume over the full window.
        {"label": "fid_radfloor", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_bin_mc_floor", "radiological:2"]},
        {"label": "fid_radfloor_one", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_bin_mc_floor", "radiological:2", "--single_fiducial_band"]},
        # fid_oldrule (2026-10-18): the fiducial rule before 5a97226 (per-band volumes, no radiological floor) on the current
        # analysis (no-offset neutron-capture veto), to separate the effect of the veto from that of the fiducial rule.
        {"label": "fid_oldrule", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True,
         "extra": ["--fiducial_bands", "DAYNIGHT=6-12,12-18;HEP=14-20,20-30;SENSITIVITY=10-20,20-30", "--fiducial_bin_mc_floor", "none"]},
        {"label": "fid_v100_300_60", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:100,300,60"]},
        {"label": "fid_v0_300_0", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:0,300,0"]},
        # fid_noveto (user, 2026-10-18): fiducial volumes chosen on SolarEnergy before the neutron-capture veto shift
        # (SolarEnergyNoVeto); cuts and significances on the default SolarEnergy (veto applied). Full chain re-optimised.
        {"label": "fid_noveto", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_energy_branch", "SolarEnergyNoVeto"]},
        # fid_dn616 (user, 2026-10-17): one Day-Night volume, optimised on 6-16 MeV (the 16-18 MeV bins, almost free of
        # background, otherwise pull the optimum to the loosest volume) and applied to every event. Day-Night only.
        {"label": "fid_dn616", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["DayNight"],
         "extra": ["--fiducial_bands", "DAYNIGHT=6-16", "--fiducial_force_band", "low"]},
        # fid_bandmax: the intersection of the band volumes (largest margin of the bands per coordinate) for every event.
        {"label": "fid_bandmax", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "max"]},
        {"label": "fid_lowband", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "low"]},
        {"label": "fid_v140_300_60", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:140,300,60"]},
        {"label": "fid_v80_240_160", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:80,240,160"]},
        {"label": "fid_v0_200_260", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:0,200,260"]},
        {"label": "fid_v100_60_480", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_force_band", "vol:100,60,480"]},
        # fid_fullspec: one volume chosen on the full reconstructed spectrum (no analysis window, no bands), the energy
        # range the cut scan and the significances use; full chain re-optimised.
        {"label": "fid_fullspec", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "ignore_energy_window": True, "extra": ["--single_fiducial_band"]},
    ],
    # fid_nosb: fiducial volumes scanned on SolarEnergyOffset (SolarEnergy without the signal/background classifier
    # stage); band assignment, cuts and significances on the default SolarEnergy; full chain re-optimised.
    "fidnosb": [
        {"label": "fid_nosb", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--fiducial_energy_branch", "SolarEnergyOffset"]},
    ],
    # hep_plain_opt: HEP on SolarEnergy (no HEP-only S/B working point) with volume and cuts re-optimised (hep_plain holds them).
    "hepplain": [
        {"label": "hep_plain_opt", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"], "extra": ["--hep_energy", "SolarEnergy"]},
    ],
    # Selection bias from background MC statistics (2026-10-16, diagnostics; run with --no-update_best_fiducials).
    # mcfloor: a cut is eligible only if gamma and neutron each keep >= MC_FLOOR simulated clusters in the analysis
    # window and fiducial volume; full chain re-optimised.
    "mcfloor": [
        {"label": f"mcfloor_{n}", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "extra": ["--mc_floor", str(n)]}
        for n in (10, 25, 50)
    ],
    # bkgsub: HD Lateral / VD Shielded gamma and neutron cut to the VD Top generated statistics by keeping a random
    # subset of whole jobs (F = VD Top generated / own generated, per sample); three seeds, full chain re-optimised.
    "bkgsub": [
        {"label": f"bkgsub_s{seed}", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "excluded_configs": ["vd_1x8x14_3view_30deg_nominal"],
         "extra": ["--bkg_subsample", f"match=vd_1x8x14_3view_30deg_nominal:{seed}"]}
        for seed in (1, 2, 3)
    ],
    # HEP without the tail extrapolation and with the likelihood bins restricted to those with simulated gamma/neutron
    # clusters at the cut (2026-10-16, user: simplify HEP, keep VD Shielded above VD Top). HEP only, full re-optimisation.
    "hepextmask": [
        {"label": "hep_extmask", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-hep_tail_extrapolation", "--hep_bin_mask", "external"]},
    ],
    # HEP without the tail extrapolation, external backgrounds with the shape of their loosest cut normalised to each
    # cut (2026-10-16, user: simpler HEP, HD Central near the thesis, VD Shielded above VD Top). HEP only, full re-optimisation.
    "heplooseshape": [
        {"label": "hep_looseshape", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-hep_tail_extrapolation", "--hep_bkg_shape", "loose"]},
        # the same on SolarEnergy (the DayNight/Sensitivity energy) instead of SolarEnergyHEP (user, 2026-10-16)
        {"label": "hep_looseshape_plain", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-hep_tail_extrapolation", "--hep_bkg_shape", "loose", "--hep_energy", "SolarEnergy"]},
        # shape from the loosest OpHits/AdjCl cut at the same NHits (2026-10-17, user: the energy-correlated cut stays fixed)
        {"label": "hep_looseshape_nh", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-hep_tail_extrapolation", "--hep_bkg_shape", "loose_nhits"]},
        # loosest-cut shape inside the HEP window only, normalised to the cut's own window counts (2026-10-17; the
        # full-spectrum normalisation of hep_looseshape / hep_looseshape_nh underestimates the window background)
        {"label": "hep_looseshape_win", "fiducialization": True, "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "skip_templates": False, "labeled_rebin": True, "analysis_override": ["HEP"],
         "extra": ["--no-hep_tail_extrapolation", "--hep_bkg_shape", "loose_window"]},
    ],
    "sbhep": [
        {"label": "sb_hep90", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True,
         "analysis_override": ["HEP"], "labeled_rebin": True,
         "excluded_configs": ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal"],
         "extra": ["--sb_completeness_high", "0.90"]},
    ],
    # Alternative HEP energy (user, 2026-10-08: kept as an option next to SolarEnergyHEP): the HEP-trained S/B
    # classifier of 03b_signal_background.py --target hep decides the offset above 12 MeV. HEP only; fiducial
    # volume and cuts held at the nominal Truncated ones, as sb_hep90.
    "hepsbmodel": [
        {"label": "hep_sbmodel", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True,
         "analysis_override": ["HEP"], "labeled_rebin": True,
         "extra": ["--hep_energy", "SolarEnergyHEPModel"]},
        # HEP without the HEP-only S/B working point (the SolarEnergy of DayNight/Sensitivity).
        {"label": "hep_plain", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True,
         "analysis_override": ["HEP"], "labeled_rebin": True,
         "extra": ["--hep_energy", "SolarEnergy"]},
    ],
    # Neutron-capture veto (config NEUTRON_CAPTURE_VETO) for DayNight and Sensitivity, 2026-10-09: default volume,
    # cuts re-optimised with the veto on. Optional group; adopted per config (ENABLED_CONFIGS) where neither
    # analysis gets worse.
    "ncveto": [
        {"label": "ncveto", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "analysis_override": ["DayNight", "Sensitivity"], "labeled_rebin": True, "extra": ["--ncveto"]},
        {"label": "ncveto_off", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "analysis_override": ["DayNight", "Sensitivity"], "labeled_rebin": True, "extra": ["--no-ncveto"]},
    ],
    # MC-statistics requirements loosened (2026-10-09, user: how much looser to reach thesis-like results). Default:
    # fiducial gate 2x100 essential background MC events, DayNight/HEP cut-scan gate 20, per-sample 1 MeV bins with
    # fewer than 2 simulated events zeroed. Full chain (volume, cuts) re-optimised under each setting. Diagnostic only.
    "mcgate": [
        {"label": "mcgate_50", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False, "fiducialization": True,
         "labeled_rebin": True, "extra": ["--fiducial_mc_threshold", "50", "--daynight_mc_threshold", "10", "--hep_mc_threshold", "10"]},
        {"label": "mcgate_20", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False, "fiducialization": True,
         "labeled_rebin": True, "extra": ["--fiducial_mc_threshold", "20", "--daynight_mc_threshold", "5", "--hep_mc_threshold", "5",
                                          "--mc_filter_threshold", "1"]},
        {"label": "mcgate_0", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False, "fiducialization": True,
         "labeled_rebin": True, "extra": ["--fiducial_mc_threshold", "0", "--daynight_mc_threshold", "0", "--hep_mc_threshold", "0",
                                          "--mc_filter_threshold", "1"]},
    ],
    # HEP with a stricter per-bin MC filter (03_analysis: per-sample 1 MeV bins with fewer simulated events are zeroed),
    # cuts re-optimised, default volumes (2026-10-10, diagnostic).
    "mcfilter": [
        {"label": "hep_mcf3", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "analysis_override": ["HEP"], "labeled_rebin": True, "extra": ["--mc_filter_threshold", "3"]},
        {"label": "hep_mcf5", "skip_rebin": False, "skip_best_cuts": False, "skip_best_sigmas": False,
         "analysis_override": ["HEP"], "labeled_rebin": True, "extra": ["--mc_filter_threshold", "5"]},
    ],
    "oscpoint": [
        {"label": "oscpoint_solar",   "skip_rebin": True,  "skip_best_cuts": True, "skip_best_sigmas": True, "analysis_override": ["DayNight", "HEP"]},
        {"label": "oscpoint_reactor", "skip_rebin": False, "skip_best_cuts": True, "skip_best_sigmas": True, "extra": ["--dm2", "7.54e-5"], "analysis_override": ["DayNight", "HEP"]},
    ],
    # 9.1 — truth x-fiducialisation vs reco flash-matching
    # Runs full fiducialization with per-sample truth positions instead of RecoX/Y/Z:
    #   marley → SignalParticleX/Y/Z, gamma/radiological → EndX/Y/Z, neutron → MainX/Y/Z
    # Produces BestFiducials_fiduc_truth.json and labeled Rebin pkls.
    # Truth is used for everything position related: the fiducial cut runs on the true
    # position, and the matched flash must be pure and agree with it (truth_match_purity in
    # backgrounds.json) -- the rejection the reco fiducial cut gets by accident from wrong
    # flash matches is made explicit.
    #
    # Deliberate SECOND exception to the one-knob policy (2026-09-18, on top of the volume
    # re-optimisation already granted above): skip_best_cuts/skip_best_sigmas are False, so
    # every analysis picks its own best NHits/AdjCl/OpHits on the truth-fiducial spectra
    # instead of being held at the reco-fiducial reference cut. Holding the reference cut
    # measured "truth positions with the cut tuned for reco positions", which is not what
    # this study asks and does not have to be monotone: hd_1x2x6_lateralAPA's optimal truth
    # volume (small, X60/Y60/Z20) has both more true signal AND more radiological background
    # than the reco volume at the SAME topological cut (background grows ~7% faster), so the
    # held comparison read worse even though the purity cut is working as intended. Full cut
    # freedom is a superset of the held-cut configuration, so this study can no longer score
    # below the default by construction (worst case it reconverges near the reco optimum).
    # Costs a full 04_best_cuts.py scan per config; DayNight/HEP add no extra cost since
    # 01_daynight.py / 01_hep.py already compute the full per-cut grid regardless of this flag.
    "fiduc_truth": [
        {
            "label": "fiduc_truth",
            "skip_rebin": False,
            "skip_best_cuts": False,
            "skip_best_sigmas": False,
            "fiducialization": True,
            "extra": ["--truth_fiducial"],
        },
    ],
    # fiduc_truth_refvol (2026-09-19): the same truth positions and background match cut as fiduc_truth, but
    # at the REFERENCE fiducial volumes (BestFiducials.json, no fiducialization stage) with free cuts. The
    # fiducial-volume scan scores raw counts without the analysis cuts, so its own truth optimum is not
    # always the best volume downstream (VD HEP: truth volume 40/0/0 vs reco 0/0/40); this variant isolates
    # what perfect position knowledge buys at an unchanged volume. Optional diagnostic since 2026-09-20
    # (OPTIONAL_GROUPS): with the truth-containment cut fiduc_truth is at or above the default in every
    # row, so it is no longer needed to define fiduc_truth or to rescue a row; it is excluded from --all.
    "fiduc_truth_refvol": [
        {
            "label": "fiduc_truth_refvol",
            "skip_rebin": False,
            "skip_best_cuts": False,
            "skip_best_sigmas": False,
            "extra": ["--truth_fiducial", "--fiducial_from_reco"],
        },
    ],
    # 9.3 — background gamma model: ClusterEnergy vs TotalEnergy comparison
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
    # 9.6 — membrane veto: which optical planes may supply the TPC-PDS match.
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
    # legacy_fit — the single legacy-fit comparison (Sensitivity only).
    # The default analysis is --fit_method pull on --flyweight templates. This variant feeds the
    # SAME templates and nominal settings through the legacy nested-minimiser fitter, so the fit
    # method is the only knob. Its chi2 grids go to results/<profile>_legacy/; the best-cut map,
    # contour tables and figures are isolated by the label. Expect hours per exposure on HD
    # (background-dominated) and validation-gate warnings -- that is what it is there to show.
    "legacy_fit": [
        {
            "label": "legacy_fit",
            "skip_rebin": True,
            "skip_best_cuts": True,
            "skip_best_sigmas": True,
            "skip_templates": True,
            "analysis_override": ["Sensitivity"],
            "extra": ["--fit_method", "legacy"],
        },
    ],
}

ALL_GROUPS: list[str] = list(STUDY_VARIANTS.keys())
# Diagnostic groups: runnable by name (--study <group>) but left out of --all and of the stale-rerun queues.
OPTIONAL_GROUPS: list[str] = ["fidnosb", "fidband", "hepplain", "fiduc_truth_refvol", "hepsbmodel", "ncveto", "mcgate", "mcfilter", "sboff", "radsmooth", "hepnotail", "mcfloor", "bkgsub", "hepextmask", "heplooseshape", "sbsimple", "sbremove"]  # hepsbmodel: needs 03b --target hep
DEFAULT_GROUPS: list[str] = [g for g in ALL_GROUPS if g not in OPTIONAL_GROUPS]


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
    Folder-isolated groups define no "label" -- they vary --folder instead --
    so they are absent here by construction. As of the bkgmodel study update,
    all active study groups now use explicit labels.
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
