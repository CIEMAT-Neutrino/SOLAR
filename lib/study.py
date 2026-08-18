"""
lib/study.py — Study variant context derived from CLI args.

Centralises the study_suffix / template_suffix / save_subfolder derivation
that was previously replicated across every analysis and plot script.

Usage
-----
    from lib import study_context          # available via `from lib import *`

    ctx = study_context(args)
    # ctx.study_suffix     e.g.  "_charge_Q100"  or  ""
    # ctx.template_suffix  same as study_suffix only when charge_threshold > 0
    # ctx.save_subfolder   e.g.  "truncated/charge_Q100"  or  "truncated/default"
    # ctx.rebin_label(energy)  →  "SolarEnergy_Rebin_charge_Q100"  or  "SolarEnergy_Rebin"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StudyContext:
    study_suffix: str     # "_<label>" or ""
    template_suffix: str  # = study_suffix when charge_threshold > 0, else ""
    save_subfolder: str   # "<folder>/<label>" or "<folder>/default"

    def rebin_label(self, energy: str) -> str:
        """Rebin pkl filename stem, labeled only for charge variants."""
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
    folder_str       = (folder or getattr(args, "folder", "")).lower()

    study_sfx    = f"_{label}" if label else ""
    # Label Rebin pkls whenever the variant changes their contents:
    # charge_threshold modifies selection; dm2_override changes oscillation weights.
    has_rebin_variant = (charge_threshold > 0) or (dm2_override is not None)
    template_sfx = study_sfx if has_rebin_variant else ""
    subfolder    = f"{folder_str}/{label}" if label else f"{folder_str}/default"

    return StudyContext(
        study_suffix=study_sfx,
        template_suffix=template_sfx,
        save_subfolder=subfolder,
    )
