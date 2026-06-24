# Forward mass–richness calibration for gnn_npe @ dC100 — final report

**Goal (CLAUDE.md):** find a *forward* `p(log10 λ | m, z, θ)` calibration such that
the relation fitted from gnn_npe's biased posteriors matches the relation fitted
from true masses, to within `d ≤ 1` (3D Mahalanobis tension in
`θ = (π0, F_m, G_z)`, combined covariance).

**Outcome: PASS** (`d = 0.879` at full settings, **linear** μ→m map adopted for
simplicity; the quadratic map also passes at `d = 0.981`; mean `d ≈ 0.97` over 5
random cal/main splits) — **after a user-authorized relaxation of constraint #2**:
the calibration subset's richness is used, but *only* to calibrate a μ–richness
cross-correlation, never the relation amplitude. The winning likelihood is a
**bivariate forward model** `p(μ, ℓ | m)` with a constant correlation `ρ ≈ 0.47`
between the mass-estimate channel and the richness relation. Within the original
constraints (independent `p(μ|m)·p(ℓ|m)`), **no variant passed** (`d ≈ 8`): the
gap is driven by conditional dependence between the gnn_npe mass estimate and the
richness, `corr(μ, ℓ | m_true, z) = 0.41` (p ≈ 1e-84), which a factorized forward
model structurally cannot represent. Git SHA at search time: `f9289b2`.

> **Two diagnosis corrections happened along the way (both instructive).**
> (1) An early draft blamed mass-dependent *skew* in `p(μ|m)`. Wrong — a
> skew-normal channel (implemented + validated) does **not** close the gap.
> (2) The decisive test was injecting `(μ, ℓ)` with a *correlated* residual: at
> `ρ=0` the forward model recovers the truth exactly (`F_m=0.326`, `σ_λ=0.140`)
> *even under gnn's compression*; at the measured `ρ=0.41` it reproduces the real
> pathology (`F_m=0.466`, `σ_λ=0.073`). A full ρ-sweep (see `diag_rho_sweep.png`)
> shows `F_m` and `σ_λ` are smooth monotone functions of ρ, crossing the real gnn
> values right at the measured ρ. So the cause is the μ–ℓ cross-correlation; the
> skew is a correlate. Modeling ρ explicitly (the bivariate model below) removes
> the bias and yields the PASS.

---

## The reference and the gap

Reference (true-mass forward fit, main sample, full-ish fast settings):

| param | reference | meaning |
|---|---|---|
| π0  | 1.254–1.257 | intercept at pivot |
| F_m | **0.333–0.341** | mass slope |
| G_z | 2.045–2.075 | redshift slope |
| σ_λ | **0.136** | intrinsic richness scatter |

Every well-constrained candidate lands at **F_m ≈ 0.48–0.50** (≈ +0.15 high) with
**σ_λ collapsed to ≈ 0.025**. π0 and G_z agree to ≲ 1–2σ; the entire failure is
the F_m / σ_λ partition.

---

## Every variant tried (see RESULTS.md for the ledger)

| variant | map | summary | width | cal_frac | d | F_m | note |
|---|---|---|---|---|---|---|---|
| v7_quadratic_mean | quad | mean | v7 | 0.2 | 8.88 | 0.49 | baseline (v7) |
| cubic_mean | cubic | mean | v7 | 0.2 | 1.75 | 0.04 ± 0.46 | **degenerate** (broad) |
| quad_median | quad | median | v7 | 0.2 | 1.89 | 0.07 ± 0.45 | degenerate |
| cubic_median | cubic | median | v7 | 0.2 | 1.73 | 0.04 ± 0.46 | degenerate |
| cubic_median_cal35 | cubic | median | v7 | 0.35 | 7.82 | 0.48 | tight → bias returns |
| cubic_median_cal50 | cubic | median | v7 | 0.5 | 8.06 | 0.48 | tight → bias returns |
| cubic_mean_cal50 | cubic | mean | v7 | 0.5 | 8.15 | 0.49 | tight → bias returns |
| hetero_quad_mean_cal50 | quad | mean | hetero | 0.5 | 9.35 | 0.50 | mass-dep scatter, no help |
| hetero_quad_mean_cal20 | quad | mean | hetero | 0.2 | 2.14 | 0.02 ± 0.5 | degenerate |
| mode_quad_cal50 | quad | mode | v7 | 0.5 | 7.95 | 0.49 | mode summary, no help |

**Pre-fit predictions vs outcomes.** The assumption checks (run before every fit)
flagged `quad_c2 ≈ 0.30` (curvature), `skew_med ≈ −0.30` (posterior skew),
`z_slope ≈ 0.05`, `pull_std ≈ 1.04`, `b_fwd ≈ 0.31`. Predicted: curvature would
bias F_m unless modeled (→ add quadratic/cubic); skew would not be removable by a
Gaussian channel (→ try median/mode, else out of scope). **Both predictions held
exactly:** curvature, once modeled, did *not* fix F_m, and the summary-stat
choice (mean = median = mode for F_m) did not move it either — pointing at the
skew as the irreducible cause.

---

## What worked

- **Map enrichment confirmed map shape is NOT the problem.** A cubic map drives
  the cubic coefficient `e_fwd ≈ 0.009 ≈ 0`; the post-cubic mean residual is flat
  to ±0.024 across mass. So the μ→m *mean* map is fully captured by a quadratic.
  Curvature is a red herring for F_m here.
- **The width model is correctly identified.** The fit recalibrates κ ≈ 0.39 so the
  implied forward scatter `ω(σ=0.28) ≈ 0.156` matches the measured forward scatter
  (~0.15), even though `pull_std ≈ 1.04` (backward widths are calibrated). The two
  are different objects and the model handles both.
- **The package gates are sound:** grid-vs-closed `~1e-9`; inject self-test
  recovers the injected forward channel; the information-free check returns an
  *unconstrained* F_m (0.06 ± 0.12) rather than a confident wrong answer.

## What didn't, and why

1. **The real, dominant cause: μ–richness conditional dependence (out of scope).**
   `corr(μ, ℓ | m_true, z) = 0.41` (p ≈ 1e-84): the GNN reads galaxy content
   (counts/positions/photometry) that also drives the AMICO richness, so at fixed
   true mass a cluster that scatters high in richness *also* scatters high in its
   mass estimate. The forward likelihood factorizes `p(μ|m)·p(ℓ|m)` and therefore
   assumes this is zero. The shared scatter is mis-read as extra mass-dependent
   signal — clusters high in both μ and ℓ look like high-mass clusters on a steep
   relation — so **F_m inflates and σ_λ collapses** (the correlated part is
   "explained" by mass instead of intrinsic scatter). Controlled injection of
   `(μ, ℓ)` with correlation ρ confirms it quantitatively: `ρ=0 → F_m=0.326,
   σ_λ=0.140` (truth recovered); `ρ=0.41 → F_m=0.466, σ_λ=0.073` (real pathology
   reproduced). This is neither a map, width, nor summary effect — it is a
   cross-term between the calibration channel and the relation, **outside the
   constraint-#3 search space.** Calibrating it would need `(μ, ℓ, m_true)` jointly
   on a subset — i.e. cal-subset richness — which constraint #2 forbids, so ρ is
   structurally unidentifiable within the allowed setup.

2. **Compression alone is NOT the cause (corrected).** With the *measured* forward
   scatter (~0.155) and ρ=0, the forward model recovers the truth (`F_m=0.326`)
   despite `b_fwd ≈ 0.32`. The "+0.05" attributed to compression in an earlier
   draft was an artifact of injecting too-large forward scatter (0.30). The
   F_m–σ_λ degeneracy is real machinery (see below) but here it is *driven* by the
   ρ cross-correlation, not by weak mass information per se.

3. **Map shape and channel skew are red herrings for F_m.** A cubic map drives
   `e_fwd ≈ 0.009 ≈ 0` and the post-cubic mean residual is flat to ±0.024; mean =
   median = mode give the same F_m; a skew-normal channel (diagnostic) does not
   help. The residual `μ − map` *is* skewed (+0.52, mass-dependent) and the
   posteriors *are* mildly skewed (−0.30), but these are correlates of the shared
   galaxy-content scatter, not the lever that moves F_m.

   *Degeneracy note:* the partition trades along `var_ℓ = σ_λ² + F_m² τ²` (the data
   constrain only the sum), which is why σ_λ collapses as F_m inflates and why more
   cal data tightens onto the **wrong** point rather than the truth — but the thing
   that *pushes* it off-truth here is ρ, not compression.

The low-`d` rows (≈1.7–2.1) are **not** successes: at `cal_frac = 0.2` the cubic
/ hetero maps leave F_m *degenerate* (±0.45), and the broad candidate covariance
mechanically shrinks `d` while F_m is uninformative. When the map is pinned
(more cal data), the true biased point (F_m ≈ 0.48) re-emerges and `d → 8`.

---

## The winning likelihood (bivariate forward calibration)

`mcmc_tests/likelihoods/bivariate.py` — **adopted:**
`BivariateForwardCal(map='linear', rho_map='const')` (linear map, chosen for
simplicity; quadratic available and marginally tighter). Latent mass
`m ~ π(m|z)` (Gaussian) on a grid; the two channels share a **constant 2×2
covariance**:

    μ  = a + b·(m−m_ref)  [+ c·(m−m_ref)²]  + η_μ        (μ→m map; c only if quadratic)
    ℓ  = π0 + F_m·(m−m0) + G_z·ζ            + η_ℓ          (the relation)
    (η_μ, η_ℓ) ~ N(0, [[ω², ρ·ω·σ_λ],[ρ·ω·σ_λ, σ_λ²]])  (ρ, ω, σ_λ constant)

Implemented via the bivariate conditional `ℓ | μ, m ~ N(relation + (ρσ_λ/ω)·(μ −
cal_mean(m)), σ_λ²(1−ρ²))`, grid-marginalized in the same conditional two-LSE form
as v7 (so `ρ=0` reduces **exactly** to the plain forward model; grid-vs-closed
still ~1e-9).

**How constraint #2 is honored under the relaxation.** Cal-subset richness enters
*only* the covariance: the cal term fits a throwaway nuisance relation
`c0 + c1·dm + c2·ζ` to the cal richness mean, so any mean trend of ℓ with mass/z
is absorbed by the nuisance and the cal richness informs **only** ρ and σ_λ —
never the headline `(π0, F_m, G_z)`, which are constrained by the main sample
alone. (Structurally: `pmap`/relation params are disjoint from `c0,c1,c2`.)

### Result (adopted: linear map, full settings 500/2000/4, seed 3 = median split)

| param | candidate | reference | 
|---|---|---|
| π0  | 1.255 ± 0.005 | 1.249 ± 0.004 |
| F_m | **0.341 ± 0.019** | **0.337 ± 0.013** |
| G_z | 2.035 ± 0.049 | 2.072 ± 0.051 |
| **d** | **0.879 → PASS** | |

Recovered `ρ ≈ 0.47`, `σ_λ ≈ 0.13` (matches the true-mass σ_λ=0.136, no longer
collapsed). The quadratic-map variant gives `d = 0.981` with `F_m = 0.334`.
Figures in `experiments/FINAL_bivariate_linear_full/`: `corner.png` (3-param
overlay), `corner_full.png` (all 11 params, d on the primary three),
`p_mass_given_lambda.png` (calibrated mass posterior + Eddington-corrected
m̂(λ)).

### Robustness across random cal/main splits (fast settings)

| | seed0 | seed1 | seed2 | seed3 | seed4 | mean |
|---|---|---|---|---|---|---|
| const ρ, quad map | 1.16 | 1.18 | 0.72 | 0.97 | 0.84 | **0.97** |
| mass-dep ρ, quad map | 1.02 | 1.16 | 0.70 | 0.96 | 0.94 | **0.96** |

The systematic bias is gone on every split (`F_m ≈ 0.32–0.34` vs ref `0.33–0.35`;
plain model gave 0.48). The residual `d ≈ 1` is split-to-split statistical noise
(the cost of estimating ρ from a finite split), not a model defect. **Mass-dependent
ρ is not needed** — `ρ1 ≈ 0` on every seed and the mean d is unchanged — so the
simplest constant-ρ model is adopted, per the "keep it simple" instruction.

### Cross-check: two-stage (cut) inference (`two_stage.py`)

As an independent check that the joint fit is not covertly using cal richness for
the relation amplitude, the same result was reproduced with an explicit **cut**:
Stage 1 estimates the covariance `(a, b, ω, σ_λ, ρ)` on the cal subset alone;
Stage 2 **freezes** it and fits `(π0, F_m, G_z)` on the main sample, which never
sees cal richness (the nuisance c0/c1/c2 are then unnecessary). Stage-1 gives
`ρ = 0.497 ± 0.024` (N_cal=1013), matching the joint posterior.

| method | F_m | d |
|---|---|---|
| joint bivariate (adopted) | 0.341 ± 0.019 | 0.879 |
| cut, hard-freeze ρ | 0.341 ± 0.015 | 0.961 |
| cut, soft (ρ ~ N(ρ̂, se)) | 0.341 ± 0.015 | 0.954 |

Same `F_m`, both PASS. The cut's slightly higher `d` is the expected direction: it
freezes the covariance, so the relation posterior is a touch tighter (±0.015 vs
±0.019) and `d` rises for the same mean offset; the joint marginalizes over all
covariance-param uncertainty and is marginally more conservative. Hard vs soft are
nearly identical because ρ is pinned to ±0.024 at `cal_frac=0.5` (the soft cut
matters more for small cal subsets). The cut is the cleaner choice for *production*
/ transfer (reusable ρ, airtight constraint-#2 separation); the joint is preferred
when full uncertainty propagation is wanted in one coherent posterior.

## Recommendation

- **Adopt** `BivariateForwardCal(map='linear', rho_map='const')` with
  `cal_frac ≈ 0.5`, calibrating ρ and σ_λ on the mass-known subset via the
  nuisance-absorbed covariance term. This reproduces the true-mass relation
  (`d = 0.879`) where the factorized forward model fails (`d ≈ 8`). The linear map
  is preferred for simplicity; the quadratic map (`d = 0.981`) is available if the
  small μ→m curvature (c≈0.12) is wanted, but is not needed to pass.
- **Why the plain model failed:** gnn_npe's posterior mean μ and AMICO richness ℓ
  are correlated at fixed mass (`ρ = 0.41`), because the graph-net reads the same
  galaxy content that drives the richness. A factorized `p(μ|m)·p(ℓ|m)` mis-reads
  that shared scatter as a steeper, tighter relation (`F_m` inflated, `σ_λ`
  collapsed) — the textbook correlated-errors-in-variables slope bias.
- **Caveat — the constraint-#2 relaxation is essential and load-bearing.** ρ is
  identifiable only from clusters with known `(μ, ℓ, m_true)`. The structural
  guard (nuisance relation) keeps cal richness out of the relation amplitude, but
  this should be reviewed: it assumes the cal subset's ρ transfers to the main
  sample, and that ρ is mass-independent (verified here: `ρ1≈0`).
- **Transfer expectation:** the bias — and the need for the bivariate model —
  scales with the μ–ℓ correlation, largest for inference models that read the same
  observables as the richness estimator (gnn_npe, the NLE summaries). Independent
  mass proxies (dynamical `M–σ`, weak lensing) should have small ρ and pass with
  the plain forward model. A skew-normal channel does **not** help — the issue is
  the cross-correlation, not the marginal shape of `p(μ|m)`.

## Reproduce

```
# winner (full settings, linear map):
python -m mcmc_tests.run --name FINAL --channel bivariate --map linear \
       --cal-frac 0.5 --seed 3 --full
# full-parameter corner + p(mass|lambda):
python -m mcmc_tests.make_final_plots FINAL_bivariate_linear_full --seed 3
# plain forward baseline (fails):
python -m mcmc_tests.run --name v7_quad --map quadratic --summary mean
```
(The CLI exposes `--map/--summary/--width/--channel/--rho-map/--cal-frac/--seed/
--full`; run from the repo root so `python -m mcmc_tests.*` resolves.) Each run
writes `experiments/<tag>/` (config.json, samples.npz, compare.json, corner.png,
checks.png, log.md) + a row in `RESULTS.md`. Key evidence figures:
`diag_crosscorr.png` (ρ is real: within-bin + 20σ permutation), `diag_rho_sweep.png`
(ρ causes the bias), `experiments/FINAL_bivariate_full/corner.png` (the PASS).
