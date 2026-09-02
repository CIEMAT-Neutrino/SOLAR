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
from typing import Optional


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
    has_rebin_variant = (
        (charge_threshold > 0)
        or (dm2_override is not None)
        or (exposure is not None and exposure != _DEFAULT_EXPOSURE)
        or truth_fiducial
    )
    template_sfx = study_sfx if has_rebin_variant else ""
    subfolder    = f"{folder_str}/{label}" if label else f"{folder_str}/default"

    return StudyContext(
        study_suffix=study_sfx,
        template_suffix=template_sfx,
        save_subfolder=subfolder,
    )
