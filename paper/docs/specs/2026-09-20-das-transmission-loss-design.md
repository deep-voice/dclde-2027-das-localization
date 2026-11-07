# Transmission loss on the DAS fin-whale data — closing Q-path

_Design, 2026-09-20, branch `sound-speed-research`. Executes item 1 of
[`paper/docs/das-finwhale-soundspeed-plan-a-findings.md`](../das-finwhale-soundspeed-plan-a-findings.md) §6,
the follow-on that Plan A named as the highest-value next step. Plan A's spec is
[`2026-09-15-sound-speed-research-design.md`](2026-09-15-sound-speed-research-design.md); this
document does not supersede it, it discharges one of its open items._

---

## 1. The question

Plan A left Q-path open in a specific way. Its §3.3 established that **trapped modes exist**
wherever ray theory called the geometry shadowed, which falsified the spec's §2.5B argument for
abandoning the straight-line model — but existence is not arrival. `soundspeed/modes.py` computes
whether a trapped mode *exists*; it says nothing about whether that mode is *excited* by a source
at −30 m, or whether it *survives* 5–40 km of propagation across a shelf-to-abyssal margin.

> **The question this design answers.** At 20 Hz, over 5–40 km on the Oregon cables, is the first
> arrival consistent with a direct water path, or with a trapped / bottom-interacting path — and
> can this dataset distinguish them at all?

The verdict decides whether any of Plan A's spec §7.4 (depth observability) is interpretable.
Plan A's findings §5 stated that Plan B needs a new justification and a new null; this experiment
is the prerequisite that supplies or withholds them.

**"Undecidable" is a permitted verdict** and must be reported as such if the power is not there.
This is not a formality: §7 below defines what would make it the right answer in advance, so that
it cannot be reached for after seeing the result.

---

## 2. What is already built and must not be rebuilt

| Existing | What it gives | Where |
|---|---|---|
| `modes()`, `ModeSet`, `Seabed` | trapped-mode existence, count, shapes `phi`, wavenumbers `kr`; validated against the Pekeris waveguide | `soundspeed/modes.py` |
| `SoundSpeedProfile`, `hycom_south_deep` | the 0–1500 m profile covering 98.4% of picks | `soundspeed/profile.py`, `reanalysis.py` |
| `solve()`, `Solution.best`, `.separation_m` | multi-basin position estimates and the flip-vs-delta contract | `experiments/solve.py` |
| `load_picks`, `attach_xyz`, `load_cables` | 11,117 picks over 628 calls with channel `(px, py, pz)` | `src/dclde_localization/physics/picks.py`, `geometry.py` |
| GMRT bathymetry, decoded and cross-validated against cable depths | water depth on a local XY grid | `viz/bathymetry.py`, `fixtures/gmrt_bathy.nc` (committed) |
| `write_results`, `load_config`, the config driver | provenance-headed result tables, one dispatch per config | `experiments/runner.py`, `paper/scripts/run_experiment.py` |

`Seabed.alpha` already exists and is **deliberately not consumed** by `modes()`, documented as
"carried for downstream use". This design is that downstream use.

`src/dclde_localization/physics/forward.py` stays frozen, as in Plan A.

---

## 3. The data, and its denominators

Measured with the canonical loader (`attach_xyz(load_picks(), load_cables("fixtures"))`), not
from the raw snapshot CSV:

| Quantity | Value |
|---|---|
| Picks | **11,117** |
| Calls (the `call` key from `dclde_localization.physics.picks.call_key`, **not** `call_id`, which is not unique across files) | **628** |
| Picks per call | median **14.0**, mean 17.7, min 3, max 61 |
| Calls with picks on both cables | **166 / 628** |
| Channel depth `pz` | −83.8 m to −1,554.9 m (median −484.1) |
| Pick SNR | median 13.54 dB, p25 10.46, p75 17.12, min 0.00, max 31.50 |
| Picks with **exactly** zero SNR | **31** |

SNR versus channel depth, which is the second observable §6.3 uses:

| Depth band | n | mean SNR |
|---|---:|---:|
| 0–200 m | 947 | 14.96 |
| 200–500 m | 5,168 | 14.59 |
| 500–1,000 m | 3,986 | 12.97 |
| 1,000–1,600 m | 1,016 | 11.98 |

Two facts about these columns that every downstream step depends on:

- **`dist` is distance along the cable, not source–receiver range.** Range must be computed from a
  solved source position to each channel's `(px, py, pz)`. §6.1 handles the dependency this creates.
- **`snr` comes from a denoised strain product** (`Denoised_SNR_*.nc`), with unknown and possibly
  range-dependent denoiser gain. **No step in this design may use its absolute level.** Only shape
  — versus range, versus depth — is admissible evidence.

### 3.1 Eligibility

Every data-side result is computed over the **basin-stable** calls, the same gate Plan A's §3.5
used, applied the same way and reported with the same split: calls whose solution flips basins are
excluded from pooled statistics and counted separately, never merged in. `min_picks = 8` applies as
elsewhere. Each table states its own surviving n; the numbers are not assumed to match §3.5's
459 / 489, because the eligibility here additionally requires a usable bathymetry path (§5.1).

---

## 4. The two hypotheses

A modal picture has no clean "direct ray" object, so the discriminator is between two models,
each of which predicts a received level as a function of range and receiver depth:

- **H-geometric** — the model currently in use. Geometric spreading plus water absorption. No
  bottom loss, no mode stripping. This is the null: it is what the straight-ray `ConstantC` /
  `PathAverage` family implies about amplitude.
- **H-modal** — adiabatic mode sum over the real bathymetry, with per-mode sediment attenuation
  and mode stripping.

Both are then projected onto the fiber (§5.4) before comparison, because the measurement is axial
strain, not pressure.

Neither is "the truth". The experiment asks which is **less inconsistent with the observed shape**,
and whether the difference clears the noise.

---

## 5. The physics unit — `soundspeed/tl.py`

New module. Depth sign convention: **positive down inside `soundspeed/`**, as everywhere else in
that package, converted only through `to_down` / `to_repo`.

### 5.1 `bathymetry.py` — moved, then extended

`viz/bathymetry.py`'s grid decode and sampler move to
`src/dclde_localization/physics/bathymetry.py`; `viz` imports them from there. One definition, and
`das_finwhale` does not reach into the visualizer package. `viz`'s existing tests must still pass
unchanged — that is the check that the move was behaviour-preserving.

New function: **water depth sampled along a source→receiver path** on the local XY grid, returning
depth at `n_steps` points plus the horizontal range at each. A path that leaves the grid, or
crosses a no-data hole, returns an explicit failure the caller must handle — it must not silently
interpolate across the gap, and calls whose path fails are excluded from the data comparison and
counted.

### 5.2 Modal attenuation

Sediment attenuation enters as a first-order perturbation of the eigenvalue by the complex sound
speed in the sediment, giving each mode its own `Im(k_r)`. The implementation must **cite the
equation it implements** (Jensen et al., _Computational Ocean Acoustics_, the modal-attenuation
perturbation result in its normal-mode chapter) and validate it against a case with an independent
answer (§8).

One conversion is pinned here because it is a classic error site. `Seabed.alpha` is in
**dB per wavelength**, as its docstring states. Therefore

```
alpha_dB_per_m     = alpha_dB_per_wavelength * freq_hz / c_sediment
alpha_nepers_per_m = alpha_dB_per_m / (20 / ln 10)      # 20/ln10 = 8.685889...
```

A test asserts this conversion directly against hand-computed values, and asserts that
`alpha = 0` yields **exactly** zero modal attenuation for every mode — the guard that a
non-attenuating seabed cannot silently acquire loss.

### 5.3 Excitation and the mode sum

Source excitation is the mode shape at source depth, `phi_m(z_s)`. This is the term that decides
whether a source at −30 m drives a mode trapped in a deep guide: a mode with a node near 30 m is
effectively not excited however well it is trapped. It must be reported per mode, not folded
silently into a total, because "the modes survive but the whale does not excite them" and "the
modes are not there" are different findings with different consequences.

Range-independent mode-sum transmission loss uses the standard far-field form,

```
p(r, z) ~ (i e^{-i pi/4} / (rho(z_s) sqrt(8 pi r))) * SUM_m  phi_m(z_s) phi_m(z) e^{i kr_m r} / sqrt(kr_m)
TL(r, z) = -20 log10 |p(r, z) / p_ref_1m|
```

with `phi` the density-reduced, orthonormal shapes `ModeSet.phi` already provides. The sum is
**coherent**; an incoherent sum is also computed and carried alongside, because the coherent sum's
interference nulls are sharper than this geometry's uncertainty justifies and a verdict must not
rest on a null's position.

### 5.4 Adiabatic range stepping and mode stripping

Over a range-varying path, each mode's phase accumulates as `∫ kr_m(r') dr'` along the path, with
the amplitude factor `sqrt(kr_m(0) / kr_m(r))`, re-solving the local mode set at each step against
the local water depth from §5.1 and the profile truncated to it.

**Mode stripping is the mechanism this whole approach exists to capture.** A mode is counted as
stripped when it ceases to be trapped in the local guide — `modes()` already answers that exactly
via `n_modes_expected`, the truncation-free cutoff count, which its docstring certifies as a
physical statement rather than a solver outcome. Once stripped, a mode contributes nothing further;
it does not re-appear if the guide deepens again. Every table reports how many modes were stripped
along each path.

`ModeSet.converged = False` must be handled per that dataclass's own documented rules, which
distinguish "count exact, shapes under-resolved" (common and expected near the trapping threshold,
must **not** be discarded, since discarding it biases away exactly the marginal regime of interest)
from "a shape is missing entirely". A path step of the second kind invalidates that path; the
count of such steps is reported rather than absorbed.

### 5.5 The fiber projection

DAS measures axial strain, so the predicted quantity is not pressure. The response carries the
axial projection `cos²θ`, with `θ` the angle between the horizontal propagation direction and the
local cable tangent, the tangent taken from the committed channel geometry.

This is not a refinement — it varies strongly and systematically with geometry along these cables,
and omitting it would put a large range-correlated error into the prediction and invite it to be
read as propagation physics. The gauge-length response is a second-order term at 20 Hz (wavelength
~74 m at 1,480 m/s against a gauge length of order 10 m); it is **out of scope**, and the write-up
says so rather than leaving a reader to assume it was included.

### 5.6 Seabed sweep

Geoacoustics are unknown at this site, so they are swept and the verdict's sensitivity across the
sweep is the uncertainty — the convention `regime_shadow_zone.yaml` already established. The sweep
starts from that config's three seabeds (`c` = 1550 / 1700 / 1900 m/s, `rho` = 1.5 / 1.8 / 2.1,
`alpha` = 0.05 / 0.15 / 0.40 dB/λ) and widens them at both ends to cover soft mud through
sand/gravel.

**Each swept entry must carry its literature source in the config's notes.** Hamilton's geoacoustic
compilations are the standard reference for the ranges; if no Oregon-margin-specific source is
found at implementation time, the config must say that plainly and the sweep width then stands in
for the site uncertainty — it must not be presented as site knowledge.

---

## 6. The experiment unit — `experiments/propagation.py`

Shaped like `experiments/regime.py`, driven by YAML through
`paper/scripts/run_experiment.py`, writing provenance-headed tables through
`write_results`. Three results, cheapest first, each standing alone.

### 6.1 Ranges, and the circularity in getting them

Range requires a source position, and source positions come from `solve()` — which uses a
propagation model, which is what is under test. The dependency is real and is handled, not ignored:

- Positions come from the **existing `ConstantC` solutions**, stated as an input rather than a
  result.
- The sensitivity is **already measured**: Plan A §3.5 found a depth-dependent effective `c` moves
  basin-stable positions a median **9.0–9.5 m**. As a covariate at 5–40 km that is negligible, and
  the argument is quantitative rather than asserted.
- Basin-flipping calls are excluded (§3.1), because for those the position difference is a median
  1,273 m and the argument above does not hold.

The write-up must state this ordering explicitly. A reader must not be able to mistake the data
comparison for an independent test of the positions.

### 6.2 Result 1 — predicted transmission loss

Pure forward calculation, no data: TL versus range for each hypothesis, over the seabed sweep, at
the geometries `regime_shadow_zone.yaml` already uses, extended to the observed depth range.
Reports per row: modes existing, modes excited at `z_s`, modes stripped, TL at each range, and the
coherent/incoherent pair.

This alone answers the literal form of the findings' item 1 — *does a mode trapped at these
geometries survive 5–40 km and get excited by a source at −30 m*.

### 6.3 Result 2 — observed maximum detection range per call

The most robust data contact available, requiring no calibration whatsoever: for each eligible
call, the greatest source–receiver range at which a pick exists, against each hypothesis's
predicted detection range under the sweep.

Its weakness is stated with it: maximum observed range is bounded by cable extent and by which
channels were examined, not by physics alone, so it can only falsify a hypothesis that predicts
detection ranges **shorter** than what is observed. It cannot distinguish two hypotheses that both
predict detection beyond the cable. That asymmetry is a property of the test and is reported as
one.

### 6.4 Result 3 — censored joint fit of SNR on range and depth

Pick SNR regressed jointly on source–receiver range **and** receiver depth, with a **free intercept
per call** (absorbing unknown source level and any per-call denoiser gain), pooling the shape across
calls. Range and receiver depth are correlated on a shelf-to-abyssal cable and must be separated by
the joint fit; neither coefficient may be reported alone.

**The detection threshold is modelled as censoring.** Picks exist only where SNR crossed a
detection threshold, so the observed sample is truncated, and the truncation bites hardest at long
range — exactly where the hypotheses differ. An uncensored fit would flatten the apparent decay and
bias the comparison toward whichever hypothesis decays more slowly. §8 requires a test in which the
uncensored fit is **demonstrably** biased on synthetic data and the censored fit recovers the known
slope; without that, the correction is untested decoration.

The 31 exactly-zero-SNR picks get a **stated rule** — they are treated as censored observations at
the threshold rather than dropped, and the table reports the result both ways so the rule's
influence is visible.

**The comparison statistic, fixed here rather than chosen later.** Each hypothesis supplies a
predicted `(range, depth)` shape; each is offered to the same per-call-intercept censored fit as a
fixed regressor, and the hypotheses are compared by the difference in fitted log-likelihood, with a
confidence interval obtained by resampling **whole calls** (not individual picks, which are
correlated within a call). The interval is reported across the seabed sweep, and its width is what
§7's "undecidable" branch reads. No statistic other than this one may be introduced after seeing
the data.

---

## 7. What counts as an answer — fixed in advance

Written before any result exists, so that the verdict cannot be chosen after seeing it.

- **Direct path.** H-geometric reproduces the observed range and depth shapes within their
  uncertainty, and H-modal does not, across the whole seabed sweep.
- **Bottom-interacting / trapped.** H-modal reproduces them and H-geometric does not, across the
  whole sweep, **and** Result 2 does not contradict it.
- **Undecidable.** Either both reproduce the observed shapes within uncertainty, or the answer
  changes sign across the seabed sweep, or the fitted uncertainty on the discriminating
  coefficient spans both predictions. This is the expected outcome if the shapes differ by less
  than the scatter the confounds in §9 can produce, and reporting it is a result, not a failure.

Whichever it is, the write-up states the consequence for Plan A's spec §7.4 explicitly: whether
depth observability is now interpretable, and under what null.

---

## 8. Testing

Unit-tested to this repo's standard — data loaders and anything that decides a number. The
load-bearing cases, distinguished from mechanics:

1. **Reduction.** `adiabatic_tl` over a **constant-depth** path must reduce exactly to the
   range-independent `mode_sum_tl`. This is the single strongest check on the range-stepping.
2. **Independent oracle.** Mode-sum TL in an isovelocity Pekeris waveguide checked against an
   independently computed answer — the pattern Task 10 used for `modes()` itself, not a
   re-derivation of the same code.
3. **Attenuation conversion.** dB/λ → nepers/m against hand-computed values, and `alpha = 0`
   yielding exactly zero attenuation for every mode.
4. **Stripping monotonicity.** A monotonically shallowing path loses modes monotonically and never
   regains one.
5. **Censoring has teeth.** On synthetic data with a known slope and a known threshold, the naive
   uncensored fit is biased in the expected direction by a margin the test asserts, and the
   censored fit recovers the known slope. A test that only shows the censored fit works would pass
   trivially on a correction that does nothing.
6. **Excitation.** A source placed at a mode's node is not coupled to it; a source at its antinode
   is maximally coupled.
7. **The move is behaviour-preserving.** `viz`'s existing bathymetry tests pass unchanged after
   §5.1.

Plus the smoke-test rule from `CLAUDE.md`: the config-driven path must run end to end on a small
subset inside the smoke budget.

---

## 9. Limitations, carried into the write-up rather than discovered by a reader

1. **Adiabatic assumes gentle slopes; the Oregon margin is steep.** This bounds mode coupling, it
   does not resolve it. A full coupled-mode or parabolic-equation treatment is out of scope, and
   the verdict inherits this as a stated limit.
2. **Per-channel fiber coupling varies.** If coupling correlates with burial depth or seafloor
   type, it correlates with range along a shelf-to-abyssal cable — a **bias**, not merely scatter.
   Nothing in this design removes it; it is a reason "undecidable" may be the honest verdict.
3. **Denoised SNR has no calibration.** Only shape is used. Any result quoting an absolute level
   would be wrong by construction.
4. **One profile product.** `hycom_south_deep` was never validated in situ at its own location;
   Plan A's spec §6.4 two-product bar remains unmet here as it was there, and every
   profile-dependent conclusion inherits that caveat.
5. **Gauge-length response excluded** (§5.5).
6. **Positions are an input, not a result** (§6.1).

---

## 10. Reporting rules

Inherited from `CLAUDE.md` and from Plan A, restated because Plan A's review found each of them
violated at least once:

- Every rate and every count **states its denominator and n**, at the point of use.
- **Basin-flipping calls are never merged** into a pooled statistic, in the tables or in any prose
  restatement of them. Plan A's review caught this being violated one layer above where it was
  enforced; here it is a rule on the prose as well as the code.
- Every table regenerates from a **committed config** through
  `paper/scripts/run_experiment.py`, and the claim that it does is **verified by
  regenerating and diffing**, not asserted. Plan A's whole-branch review found this claim false
  when it was merely asserted.
- Figures use seaborn, minimal in-figure text.
- Anything reported is promoted to `src/dclde/` first; notebooks explore only.

---

## 11. Scope

**In:** `bathymetry.py` (moved + path sampling), `soundspeed/tl.py`, `experiments/propagation.py`,
their configs, their result tables, their tests, and a findings write-up that answers §1 and states
the §7 verdict.

**Out:** coupled-mode and parabolic-equation propagation; gauge-length response; a second profile
product (WOA23 — still open from Plan A's findings §6 item 6); Plan A's findings items 2 (the
§7.0.2 moveout-shape test), 4 (organizer questions), and 7 (free-`c` contamination); and all of
Plan A's spec §7.4–7.6, which this experiment exists to make interpretable rather than to perform.
