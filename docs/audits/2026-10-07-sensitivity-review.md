# Sensitivity audit (2026-10-07)

This is an isolated validation, not a replacement for the production sensitivity results.
The numerical A/B tests used `sb-classifier` at `e572b48`; subsequent classifier
commits, including `fcf73bf`, require fresh templates and a new comparison.
It uses HD Central, `marley_final`, the Truncated background model, SolarEnergy,
30 and 10 detector-years, and the four-nuisance pull fit (signal/background
normalisation, energy scale, and sin²θ₁₃). Scores are the mean of the two
wrong-hypothesis χ² values; they are **not** sigma values.

| Cut (NHits/AdjCl/OpHits) | 30-year Score | 10-year Score |
| --- | ---: | ---: |
| Current 1/3/12, freshly rebuilt signal | 3.488591 | 1.334021 |
| Candidate 1/4/14, freshly rebuilt signal | 4.271234 | 1.608027 |

The candidate improves Score by 22.4% at 30 years and 20.5% at 10 years.
Both 14,702-point fits passed the finite, Asimov, profile-bound and smoothness
validation gates. Only seven nearby cuts were tested; 1/4/14 is not established
as the global optimum. Its 6–20 MeV background MC support is 9 gamma, 234 neutron
and 484 radiological events. Gamma's weighted contribution is small, but the
low gamma count warrants a robustness check.

The thesis quotes HD Central Scores of 5.73 (30 years) and 2.11 (10 years),
corresponding to sqrt(Score) = 2.39 and 1.45 sigma equivalents. The candidate
remains 25.5% and 23.8% below those Scores. This is not a controlled comparison:
the thesis used different cuts/fiducial volumes, and Chapter 8 marks its signal
templates as provisional because their nadir PDF differed from the background's.

Findings and follow-up:

- The production 1/3/12 flyweight base was dated 2026-10-06 17:44, before the
  current fiducial JSON (23:13). Rebuilding it reduced its unoscillated weighted
  sum from 19,560.82 to 12,284.64 per year. The fitter must reject such a stale
  base; cut searches must likewise rebuild reference templates after input changes.
- `04_best_cuts.py` scores candidates with signal/background normalisation only,
  while `06_significance.py` profiles energy scale and sin²θ₁₃ too. Its objective
  and limited coarse-to-fine search need not select the best final-fit cut.
  Re-score finalists with the final nuisance model and run a wider cut search.
- The energy-scale nuisance interpolated binned counts as point samples. A
  count-preserving redistribution is numerically correct, but at 1/3/12 it
  changed the 30-year Score from 3.488591 to 3.478099 (no recovery).
- Turning off the signal/background classifier in a matched-template test gave
  3.487035 at 30 years, versus 3.488591 with it: no meaningful recovery.
- Combining the 2% background systematic with Poisson error in quadrature
  changes the HD Central fiducial objective, but did not change its selected
  volume in the stored fiducial scan.

Audit templates and fits are isolated under PNFS `SolarEnergy_audit_current`,
`SolarEnergy_audit_cutcheck` and `SolarEnergy_audit_nosb`. Do not treat their
best-cut pickle files as production selections. The production templates,
best cuts and reported thesis figures should be regenerated together after the
input-freshness and numerical fixes, then checked across all configurations.
