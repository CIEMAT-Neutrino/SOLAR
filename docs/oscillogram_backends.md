# Oscillogram Backends

Two backends can compute the electron-neutrino survival probability map P_ee(E, cos η)
used by the SOLAR analysis. Both are selected via `OSCILLATION_BACKEND` in `import/analysis.json`.

## Backends

| Backend key | Source | Status |
|---|---|---|
| `"file"` | Pre-computed ROOT→pkl library (original) | Default, always available |
| `"prob3"` | Prob3++ `BargerPropagator`, `external/Prob3plusplus/` | Reproduces library files |
| `"nufast"` | NuFast-Earth `Probability_Engine`, `external/NuFast-Earth/` | Fastest, independent |

## Unit and Convention Table

| Quantity | Internal Python (oscillation.py) | Prob3++ C++ | NuFast-Earth C++ |
|---|---|---|---|
| Energy axis | MeV | GeV (÷1000 at call) | GeV (÷1000 at call) |
| Nadir axis | cos(η) (downward direction) | cos(η) ✓ | cos(zenith_ν) ✓ compatible |
| Night condition | cos(η) < 0 | cos(η) < 0 ✓ | cosz < 0 ✓ same |
| Δm²₂₁ (solar) | `SOLAR_DM2` | `dmsq12` ✓ | `Dmsq21` ✓ |
| Δm²₃₁ (atmospheric) | `ATM_DM2` = 2.5e-3 | `DM2` = 2.5e-3 ✓ | `Dmsq31` ✓ |
| sin²(θ₁₂) | `SIN12` | `ssth12` ✓ | `s12sq` ✓ |
| sin²(θ₁₃) | `SIN13` | `ssth13` ✓ | `s13sq` ✓ |
| sin²(θ₂₃) | fixed 0.5 | fixed 0.5 | fixed 0.5 |
| δ_CP | fixed 0 | fixed 0 | fixed 0 |

**NuFast-Earth sign flip:** when passing nadir centers to `Set_Spectra`, use `coszs = -nadir_centers`.

## DataFrame Contract

Both `"prob3"` and `"nufast"` backends must produce `OscResult` (see `lib/oscillation_backends.py`)
with components that, after `combine_day_night()`, yield a DataFrame identical in structure to
the pre-computed pkl files:

```
df.index   → cos(η) centers, shape (40,), range [-0.975, 0.975]
df.columns → E_MeV centers,  shape (120,), range [0.125, 29.875]
df.values  → P_ee ∈ [0, 1]
```

Fixture for regression: `tests/fixtures/oscillogram_default.pkl`
Parameters: dm2=6.0e-5, sin13=0.021, sin12=0.303

## Solar Matter Effect (Prob3++ backend)

Production fractions computed via analytic `ssth()` (adiabatic MSW in Sun):

```
rhoY = 0.090  # kg/cm³, production density × electron fraction
A    = 1.53e-4 * rhoY * E_MeV    (MSW potential)
f_2  = ssth(E_MeV, dm2, sin12, sin13) * (1 - sin13)
f_3  = sin13
P_ee = (1-f_2-f_3)*|<e|ν_1>|² + f_2*|<e|ν_2>|² + f_3*|<e|ν_3>|²
```

## Nadir PDF

| Backend | Nadir PDF source |
|---|---|
| `"file"`, `"prob3"` | `nadir.root` loaded by `get_nadir_pdf_file()` |
| `"nufast"` | `Solar_Weight(eta, latitude)` analytic (SURF lat = 44.35°N) |

Fixture: `tests/fixtures/nadir_pdf_centers.npy`, `tests/fixtures/nadir_pdf_values.npy`

## Validation Tolerances

| Comparison | Max |ΔP_ee| | Notes |
|---|---|---|
| `prob3` vs `file` | < 0.005 | Same algorithm, should be near-exact |
| `nufast` vs `prob3` | < 0.02 | Different matter-effect implementation |
| Either backend, significance | < σ_stat | End-to-end regression gate |
