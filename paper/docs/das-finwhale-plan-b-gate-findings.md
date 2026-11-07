# The Plan B gate — source depth is not recoverable from moveout on this dataset

_Written 2026-09-21, branch `sound-speed-research`. Reports the gate fixed in
[`paper/docs/specs/2026-09-21-plan-b-depth-observability-design.md`](specs/2026-09-21-plan-b-depth-observability-design.md)
§3, which is the identifiability precondition for that note's §4 and §5._

**The gate fails, and Plan B stops here.** The design note names that outcome in advance as a
complete, publishable result rather than a shortfall, and this document reports it as one. What it
establishes is a property of the measurement, not a limit on the effort spent: on this geometry, at
this pick precision, arrival-time moveout does not bound source depth well enough to improve on
simply assuming the null's −30 m.

Every number below was re-read from its committed table in `paper/results/`
before being written here (read those with `pd.read_csv(path, comment="#")`; the `#` headers carry
caveats reproduced in the relevant section rather than left for a reader to find). Anything
measured during the plan but not backed by a committed table is marked **[not table-backed]**, as
Plan A's findings do.

Tables: `gate_moveout.csv` (628 rows, one per call), `gate_moveout_summary.csv` (1 row),
`gate_moveout_path_comparison.csv` (1 row). Config: `paper/configs/gate_moveout.yaml`.

---

## 1. What the gate asked, and when it was fixed

The design note's §3 poses the question:

> **Question:** does the moveout curve change with source depth by more than the per-call pick
> scatter, on real picks?

and fixes the criterion in the abstract, deferring the numbers to the implementation plan:

> **Gate criterion, fixed here.** Proceed to §4 only if, on a pre-declared stratum of calls with
> adequate depth aperture, the depth-induced moveout shape change exceeds the per-call effective
> noise by a stated margin on a **material fraction** of calls — material meaning enough calls to
> support the stratified reporting §4 requires, not a handful of favourable ones. If it does not,
> Plan B **stops at the gate**, and that is a publishable negative.
>
> The stratum and the margin are set in the implementation plan, before the gate is run, and
> recorded there. They are not chosen after seeing the outcome.

The concrete form, copied verbatim into the config's `notes` before the run and applied unchanged:

> Stratum: `n_picks >= 8`, receiver-depth span ≥ 200 m, basin-stable, solved under the null.
> Statistic: per-call **marginal** 1σ on source depth, from profiling `x` and `y` out of the misfit
> at each candidate depth (`t0` is analytic). Proceed to depth observability only if **at least 10%
> of the stratum AND at least 50 calls** achieve σ_z ≤ 100 m.

The thresholds carry their own justifications, recorded at the same time: 100 m because the null
assumes −30 m and fin whales call shallow, so a resolution coarser than ±100 m cannot improve on
assuming the null's depth; 10% and 50 calls because that is enough to support the stratified
reporting §4 requires and enough that the outcome is not carried by a few favourable geometries.
The criterion asks whether a usable *subset* exists, not whether depth works everywhere.

That the threshold was genuinely in play is recorded too. A linearised scoping estimate made before
the plan (440 calls with ≥ 100 m span) put roughly 15–18% of calls below 100 m — above the bar. That
estimate treats the depth direction as independent of `x`, `y` and `t0` and treats a structured
residual as white, so it is optimistic by construction, and the design note's §1 forbids quoting it
as a result. The point is that the pre-registered bar was set where a plausible prior estimate
cleared it. It did not survive the joint measurement.

The null throughout is the design note's §2: `ConstantC(1480)` straight-ray propagation with source
depth fixed at −30 m.

---

## 2. Denominators

From `gate_moveout_summary.csv`. **Every rate in §3 and §4 is over the 271-call stratum and says so
at the point of use.**

| Category | Calls | Share of 628 |
|---|---:|---:|
| Total annotated calls | **628** | 100% |
| Below `min_picks = 8` | 104 | 16.6% |
| Unsolved under the null (of the 524 with enough picks) | 26 | 4.1% |
| **Basin-flipping (excluded, never merged)** | **86** | **13.7%** |
| Solved and basin-stable but receiver-depth span < 200 m | 141 | 22.5% |
| **In stratum** | **271** | **43.2%** |

628 − 104 − 26 − 86 − 271 = 141, so the categories close exactly with no silent extra filter. Total
picks in the snapshot: **11,117**.

Basin-flipping calls are excluded, as in the transmission-loss study and for the same reason: a call
whose horizontal solution jumps between mirror basins has no stable `(x, y)` for a depth profile to
be taken around. They are counted here and never merged into a stratum statistic. **Zero of the 271
stratum calls are flagged flipped** — the exclusion is complete, not partial.

Stratum composition, re-read from `gate_moveout.csv`: **156 of 271 two-cable, 115 of 271
single-cable**; median receiver-depth span **381.6 m**; per-call robust residual σ under the null,
quartiles **57.9 / 97.8 / 122.4 ms** (n = 271). (The summary's `calls_two_cable = 166` is over all
628 calls, not over the stratum; the two figures are different denominators and are not
interchangeable.)

The bidirectional walk check described in §4 ran on **all 271** eligible stratum calls
(`calls_bidirectional_checked = 271`, `calls_bidirectional_eligible = 271`), not on a sample.

---

## 3. Depth sensitivity: the result

### 3.1 The pre-declared statistic

**17 of 271 stratum calls (6.3%) reach σ_z ≤ 100 m.** That is `n_resolved_forward_only` — the
pre-declared criterion applied on its own, with no later condition intersected into it. It fails
both legs: 6.3% against a 10% bar, and 17 calls against a 50-call bar.

**After an additional condition added after the pre-registration** — described in §4, and *not* part
of the registered criterion — the count is **2 of 271 (0.74%)**. That is the summary's `n_resolved`.
It also fails both legs.

`gate_passes = False`, `passes_fraction = False`, `passes_min_calls = False`.

The two numbers are reported separately, and the pre-declared one leads, because the composite
cannot be decomposed by a reader who only sees the smaller figure. The addition did not turn a pass
into a fail; it moved 17 to 2, both of which fail. A tolerance sweep on the added condition
**[not table-backed]** gave 2 at a 10% tolerance, 3 at 20%, 3 at 50% and 9 at 100% — the ceiling of
that sweep is the unconditioned 17, against a bar of 50. No setting of the addition changes the
verdict, which is why it can be reported transparently rather than defended.

### 3.2 The σ_z distribution, and why most of it is not a number

Of the 271 stratum calls, only **42 (15.5%)** have a finite σ_z at all. Among those 42: median
**125.9 m**, quartiles **40.7 / 125.9 / 159.5 m**, min **11.2 m**, max **235.3 m**. So even the
subset where the profile bounded depth at all sits, at its median, above the 100 m target.

The remaining **229 of 271 (84.5%)** have no finite marginal width. These are not one failure mode,
and the table separates them (`sigma_z_reason`) rather than collapsing them into an
undifferentiated `inf`:

| `sigma_z_reason` | Calls (of 271) | What it means |
|---|---:|---|
| `edge_deep` | **114** (42.1%) | Profiled minimum sits at the −600 m grid floor. **The grid stopped the walk, not the data.** |
| `not_crossed` | **98** (36.2%) | Interior minimum, but Δχ² = 1 is never reached on both sides — the misfit is too flat in `z` for the noise. |
| `edge_surface` | **17** (6.3%) | Minimum at 0 m, the physical sea surface. Not a grid artifact and cannot be extended. |
| finite | 42 (15.5%) | A bounded marginal width, distributed as above. |

`edge_surface`'s 17 (6.3%) is numerically identical to §3.1's `n_resolved_forward_only` = 17 (6.3%)
below. The two are unrelated counts over the same 271-call denominator — a coincidence, not a
relationship — and are flagged here so a reader does not read one into the other.

The 114 `edge_deep` calls are a stated limitation of this run and not a hidden one: a −600 m grid
floor bounded them, and a deeper grid would move them. A probe re-profiled 20 of the 114 on a
−2000 m grid **[not table-backed — measured during review, not committed]**: 17 of 20 acquired
interior minima, between −610 and −1240 m, and **zero of 20** reached σ_z ≤ 100 m. So the grid
bounded the *location* of those minima but not their *width*, which is the quantity the criterion
tests.

That probe supports the optimistic ceiling. Applying every charitable assumption at once — forgive
all 15 of the calls §4 identifies as artifacts, and scale the probe's implied best case onto the
whole 114 with an upper confidence bound — gives **2 + 15 + ~16 = 33 calls**, against a bar of
**50**. The count leg of the criterion fails even under assumptions no honest reading would grant
simultaneously. The fraction leg, at 33/271 = 12.2%, would clear 10% only under that same stacked
charity, and only if every one of those 33 were real.

One further inflation source was checked and ruled out **[not table-backed]**: the ratio of the
robust residual σ used for the χ² scale to the L2 RMS has median 0.897 (IQR 0.74–1.02, n = 266). If
anything that *narrows* the intervals. The failure is not an over-wide noise estimate.

---

## 4. Why the 17 are artifacts, on four independent grounds

15 of the 17 pre-declared resolutions are not depth measurements. This is over-determined — four
lines of evidence, any one of which would be suggestive and which together are not reconcilable with
a genuine depth constraint.

**(i) The misfit curves themselves are direction-dependent.** Each in-stratum profile was walked
through the depth grid in both directions, deep-to-shallow (z_grid's own ascending order) and the reverse, each step warm-starting
`(x, y)` from the previous step's own optimum. For **249 of 271** stratum calls the two walks trace
the same curve to machine precision: maximum discrepancy median **5.7e-13 s**, all 249 below 1e-9 s.
For **22 of 271** they do not, disagreeing by **1.47e-4 to 1.61e-2 s** — nine orders of magnitude
away from the 249. **15 of those 22 are exactly the low-σ_z calls**, discrepancy **1.20e-3 to
1.61e-2 s**; the other 7 are non-finite in both directions and never claimed a resolution. This is a
property of the misfit *curve*, before any σ_z or tolerance enters. For **8 of the 15**, one
direction gives a finite σ_z and the other gives `inf` — a categorical disagreement that no relative
tolerance can soften.

**(ii) The disagreement is tolerance-invariant.** Five direction-inconsistent stratum calls,
including the largest curve discrepancy in the stratum, were re-profiled forward and reverse at a
1000× tighter optimiser setting (`xatol=1e-4`, `fatol=1e-16`, `maxfev=40000`, against the run's
`xatol=0.05`, `fatol=1e-13`). Every value reproduced to printed precision, recorded in the summary
CSV's `#` header and the config notes:

| Call | σ_z forward | σ_z reverse | Curve discrepancy |
|---|---:|---:|---:|
| `2021-11-04_02_02_32/2` | 11.201 | inf | 6.91e-3 |
| `2021-11-04_08_02_32/7` | 22.373 | inf | 1.61e-2 |
| `2021-11-04_08_05_02/7` | 21.557 | 225.106 | 9.73e-3 |
| `2021-11-04_02_04_12/4` | 42.122 | inf | 5.29e-3 |
| `2021-11-04_08_03_22/7` | 17.926 | 58.725 | 9.96e-3 |

So the direction-dependence is genuine multi-basin local-minimum tracking, not optimiser convergence
slop. The same five-call check also reproduced these calls' committed values exactly from raw picks
via a separately-written code path — a second, independent line of evidence beyond the run's own
regenerate-and-diff determinism check.

**(iii) They are single-cable geometry.** **16 of the 17** pre-declared resolutions come from calls
recorded on one cable only (and **all 15** of the artifact subset do), against a stratum in which
single-cable calls are the *minority*: 115 of 271. `solve.py`'s own mirror-lobe documentation
predicts exactly this multi-basin behaviour for single-arc geometry, and §5 gives the mechanism.

**(iv) Their residual scatter is anomalously small.** The 15 have per-call robust residual σ between
**5.9 and 27.1 ms**, against a stratum median of **97.8 ms** (n = 271). A narrow σ_z obtained by
dividing a flat curvature by an unusually small noise estimate is arithmetic, not resolution.

**The two survivors**, reported by name rather than as a count, since two calls cannot support a
distribution: `2021-11-04_02_03_22/7` (σ_z 40.4 m forward and reverse, single-cable, residual σ 21.3
ms, minimum at −145 m) and `2021-11-04_02_05_02/7` (σ_z 78.2 m both directions, two-cable, residual
σ 98.5 ms, minimum at −200 m). Two calls out of 271 is not a stratified result and is not offered as
one.

**The walk-consistency condition was added after the design note was written.** It is stated in the
config's notes as an addition decided before the run, but it is not part of the §3 pre-registration,
and folding it silently into `n_resolved` would have made the headline number undecomposable. It is
reported here as what it is: a diagnostic that identified 15 artifacts, on a verdict that does not
depend on it.

---

## 5. The mechanism, and why two curved cables matter

This is the most transferable thing in the document, and it is measured rather than asserted.

When receiver depth varies **monotonically** with along-cable position, source depth becomes nearly
exchangeable with a source-position shift along the cable. The two perturbations point the same
direction in travel-time space for every receiver at once, so re-minimising `x` and `y` absorbs
almost all of a depth change. Measured on the module's own synthetic fixture (`depth_profile`'s
docstring), 400 m of source-depth error leaves, after refitting `x` and `y`:

| Geometry | Unresolvable RMS residual left by 400 m of depth error |
|---|---:|
| Single arc, depth monotonic along the cable | **~23 µs** |
| Single arc, depth V-shaped along the cable | **~918 µs** |
| Two arcs at clearly different bearings | **~3,044 µs** |

A single monotonic arc is the worst case by two orders of magnitude, and the two-cable case is not
close to it: receivers approaching from different bearings cannot be reconciled by one common
horizontal shift. This is why two curved cables matter for depth specifically, beyond the mirror-side
ambiguity in `(x, y)` that `solve.py` already handles.

It also explains §4(iii) directly. 23 µs of unresolvable signal is below the optimiser's own
positional slop at any reasonable tolerance — the run's `xatol = 0.05 m` is ~34 µs of travel time —
so on a monotonic single arc the quantity being measured and the numerical noise floor are the same
size. The tolerance was tightened for this run precisely so that "local-minimum artifact" and
"convergence noise" would remain distinguishable; §4(ii) shows they are.

This is a geometric statement, not a data-quality one. More picks, better picks, or a longer
deployment would not change it for a call heard on one monotonic arc.

---

## 6. The path question, from timing

Carried alongside the gate as the design note's §3 requires: direct versus bottom-bounce curvature in
(along-cable distance, time), fitted to the same picks. From `gate_moveout_path_comparison.csv`,
n = 266 calls compared.

**235 of 266 favour the direct path, 31 of 266 favour bounce, 0 tied.**

**Correction (this section was rewritten; see below for what changed and why).** `delta_chi2 =
n_picks * (misfit_bounce**2 - misfit_direct**2) / resid_sigma**2` — `resid_sigma` in seconds
(`resid_sigma_ms`/1000) — is the module's own chi-squared convention (the same form `marginal_sigma_z`
uses for `sigma_z_m`) applied to the two path misfits, and it is the statistic that answers whether an
individual call's path preference exceeds its own noise: a positive value favours direct, negative
favours bounce, and `|delta_chi2| >= 1, 4, 9` are the 1/2/3-sigma-for-one-dof significance thresholds.
Measured on `gate_moveout.csv`, n = 266:

| Threshold | Favouring direct | Favouring bounce |
|---|---:|---:|
| 1σ (`\|Δχ²\| >= 1`) | **136 of 266 (51.1%)** | **14 of 266 (5.3%)** |
| 2σ (`\|Δχ²\| >= 4`) | **66 of 266 (24.8%)** | **7 of 266 (2.6%)** |
| 3σ (`\|Δχ²\| >= 9`) | **22 of 266 (8.3%)** | **1 of 266 (0.4%)** |

So a majority of individually-tested calls DO reach 1σ significance one way or the other (150 of 266,
56.4%; 116 of 266, 43.6%, fall short even at 1σ), and among the 150 that resolve, direct wins
decisively at every threshold — and there is a real bounce-favouring tail, up to 7 calls at 2σ and 1
at 3σ, that the design note's §3 requires be reported as a finding rather than folded away. Neither
"no individual call resolves the question" nor "every call agrees with the ensemble" is the honest
summary; both a majority direct-favouring significant subset and a small but real bounce-favouring
minority coexist in this table.

The magnitudes agree in direction. Δ misfit: median **1.42e-3 s**, mean 2.73e-3 s, range **−4.61e-3
to +2.24e-2 s**. Scaled by each call's own null-model residual σ (`delta_misfit_over_resid_sigma`,
n = 266): median **0.0153**, mean 0.0404, range **−0.223 to +0.579**. The largest single-call
preference in the dataset reaches 58% of that call's own noise. The median reaches 1.5%. **This ratio
is a SCALE comparison, not a significance test** — it divides a whole-fit RMS misfit difference by a
PER-PICK sigma, with no `n_picks` in it, so `n_exceeds_own_noise` (`|ratio| >= 1`, which is 0 of 266)
does not answer "how many calls individually resolve the path question." `delta_chi2` above is the
statistic that does.

**What was wrong in the earlier version of this section, and which way the error ran.** An earlier
draft read `n_exceeds_own_noise = 0` as "not one of the 266 calls has a direct-versus-bounce misfit
difference exceeding its own residual noise" and glossed that as "no individual call resolves the
question." The first claim was true only of the mis-scaled ratio; the gloss was unsupported once the
correctly n-scaled `delta_chi2` is read instead — 136 calls (51.1%) do individually reach 1σ, and 66
(24.8%) reach 2σ, favouring direct. The error **understated** the direct-path evidence: it reported
zero individually-significant calls where 136 exist. That is the conservative direction for the
purpose the transmission-loss work's §1.2 licence rests on (§8.8) — the correction adds support for
the direct-path conclusion already reached, not evidence against it, so §7 and §8.8 stand unchanged.

So the honest claim is now broader than the earlier draft's, and still precise: **the data lean toward
the direct path both at the ensemble level (88% of calls, sign-stable median Δmisfit) and among the
subset of individual calls that reach significance (136 of 266 at 1σ, the large majority of that
subset favouring direct) — with a real, reportable minority (up to 7 calls at 2σ) resolving toward
bounce instead.** A 235:31 ensemble split under a sign-symmetric null would be surprising on its own;
that collective statement now sits alongside, not in place of, the per-call significance picture.

**Does this agree with the transmission-loss study's amplitude-based ruling? Yes.** That study found
the straight-ray picture the better description of the observations at every seabed in its sweep,
sign-stably. This finds the direct path favoured on 88% of calls from an entirely different
observable, with over half of the individually-significant calls agreeing. The agreement is worth
something specific: **travel time is immune to the dB-calibration problem that clouds the amplitude
result.** The transmission-loss findings' §7.7 shows its observable registers a known, purely
geometric amplitude change at a fraction of the expected rate, and that the implied bias points
*toward* the straight-ray answer it delivered. Travel time carries no dB scale, so this corroboration
comes from a direction that bias cannot reach.

Its limit is narrower than "no resolving power at all" but still real: this check cannot certify the
forward model. It corroborates the earlier ruling at the ensemble level and, now, for a majority of
the individually-significant calls — but 116 of 266 calls resolve neither way even at 1σ, and a
bounce-favouring tail exists and does not vanish under the corrected statistic. And it inherits the
flat-bottom approximation in §7.1,
which is a coarser bounce model than the transmission-loss study's bathymetry-tracked paths. There IS
a disagreement to report as a finding, per the correction above: up to 7 calls (2.6%) individually
favour bounce at 2σ and 1 (0.4%) at 3σ. §3 of the design note requires this be reported rather than
resolved away, and it is reported here rather than folded into the ensemble-level agreement with the
earlier, amplitude-based answer.

---

## 7. Verdict

**`gate_passes = False`. Plan B stops at the gate.**

Design note §3 makes this a terminal, permitted outcome, and §4's closing line — "**Not resolved is a
permitted and complete outcome**, at either part, and is reported as such" — makes it a complete one.
Neither §4's synthetic recovery, nor §5's `c(z)` comparison, nor §7.4 of the Plan A spec will be
built. Building them would mean estimating depths on an objective measured not to constrain depth.

Stated plainly, without hedge: **this geometry cannot support source-depth inference at this pick
precision.** Depth is not recoverable from arrival-time moveout on this dataset. That is not a
statement about what a better optimiser, a finer grid or more careful picking would achieve; §5 shows
the dominant obstruction is geometric — on a single monotonic cable arc, source depth and along-cable
source position are very nearly the same parameter.

The claim's scope, so it is not over-read:

- It is about **arrival-time moveout under straight-ray propagation**, on **these two cables**, on
  **these 628 annotated calls / 11,117 picks**. It is not a claim about DAS depth estimation in
  general, and §5 names the geometry that would change it: two arcs at clearly different bearings, or
  one arc with non-monotonic depth.
- It is about **depth**. Nothing here bears on horizontal position, which Plan A already measured to
  be insensitive to the sound-speed treatment (a median 9.0–9.5 m shift against a 142 m noise floor).
- The amplitude route to depth was already closed by the transmission-loss study, whose receiver-depth
  coefficient of received level has an interval spanning zero in all 16 configurations. With the
  travel-time route now measured and closed as well, **both routes this dataset offers are closed**,
  and there is no third one to try.

The useful residue is §5's exchangeability measurement. It is a quantitative statement about when DAS
geometry can and cannot carry depth, it generalises past this dataset, and it would justify a cable
layout in a way no depth estimate from these data could have.

---

## 8. Limitations

**8.1 The bounce model is flat-bottom.** `bounce_travel_time` reflects off a single horizontal plane
at a per-call bottom depth (`bottom_z_m`, median −731 m, range −1,628 to −194 m over the 266 compared
calls). The real margin is a shelf-to-abyssal slope, and the transmission-loss study tracked paths
across the actual bathymetry. §6's ensemble lean toward direct should be read as a comparison against
an idealised bounce, not against the best bounce model available. Because §6 resolves nothing at the
call level, this approximation has no load-bearing consequence here — but it would if the path
comparison were ever quoted on its own. The bottom depth used is also taken along the path from the solved source to the *centroid* of that call’s own picks, which for a two-cable call can lie between the cables, on no real path — a second, unflagged approximation stacked on the first.

**8.2 The profile is L2 where the repo's solve is L1.** The depth profile minimises an L2 misfit,
because a Δχ² = 1 reading of an interval width requires a Gaussian likelihood; the L1 solve still
supplies the starting position. The choice is stated in the module docstring and the config rather
than assumed, but it means σ_z is computed under a different objective from the positions it is
profiled around, and L2 is the more outlier-sensitive of the two on a dataset with a known
mis-association tail.

**8.3 σ_z is a local curvature width, not a posterior.** It is the Δχ² = 1 half-width of the
profiled misfit around its minimum, with `x` and `y` re-minimised and `t0` eliminated analytically.
It is not a marginal posterior, it assumes a locally quadratic misfit, and — as §4(i) shows for 22 of
271 calls — the profile is not always even single-valued as a function of depth. For a multi-modal
misfit the number is a description of one basin, not of the depth's uncertainty.

**8.4 The −600 m grid bounds 114 of 271 stratum calls.** Reported in §3.2 rather than buried: those
minima sit on the grid floor, and the grid, not the data, put them there. The 20-call probe on a
−2000 m grid **[not table-backed]** found interior minima for 17 of 20 but σ_z ≤ 100 m for none, so
the limitation constrains where the minima are and not the widths the criterion tests. A full re-run
on a deeper grid was not performed.

**8.5 The residual treated as noise is partly model error.** The χ² scale uses each call's robust
residual σ under the null. Plan A measured that residual to carry a **structured 76–82 ms trend with
range** — not white scatter. Treating structured model error as noise biases the widths rather than
merely scattering them, and the direction of that bias is not established here. It is one more reason
the 42 finite σ_z values are an optimistic reading, not a pessimistic one.

**8.6 The walk-consistency condition was added after the pre-registration.** §3.1 and §4 state this
at the point of use. The composite `n_resolved = 2` is not the registered statistic;
`n_resolved_forward_only = 17` is. Both fail both legs, so the verdict does not rest on the addition,
but a reader is entitled to see which number came from where.

**8.7 Positions are an input, not a result.** Inherited from the transmission-loss study and restated
because it constrains how §6 may be read: the `(x, y)` a profile is taken around comes from `solve()`
under `ConstantC`, a propagation model of the kind under test. Basin-flipping calls, for which that
dependency is not benign, are excluded (§2). No result here independently validates the localizer.

**8.8 The §1.2 path licence travels with this, as the design note requires.** The straight-ray forward
model is the surviving null of a *comparative* ruling made on an observable that failed its own
calibration check, with the implied bias pointing toward the straight-ray answer. §6 corroborates it
from timing at the ensemble level. Neither result certifies the forward model, and a gate verdict is
conditional on the model it was computed under: what has been measured is that depth is unrecoverable
*under straight-ray propagation*. §5's geometric argument, however, does not depend on the propagation
model — the depth/along-cable exchangeability on a monotonic arc is a property of the sensitivity
structure, not of the velocity field.

---

## 9. Reproducing this

```bash
uv run python -m paper.scripts.run_experiment \
  --config paper/configs/gate_moveout.yaml
```

Wall clock ~12.5 minutes on the team box (bidirectional profiling on all 271 eligible stratum calls).
All three tables regenerate byte-identically on re-run, verified with
`pandas.testing.assert_frame_equal` on the parsed body (`comment='#'`), modulo the `# generated:`
timestamp line. `src/dclde_localization/physics/forward.py` and `paper/src/dclde_soundspeed/soundspeed/modes.py`
are frozen and were not modified by any task in this plan; verified with `git diff --exit-code`.

Tests, measured 2026-09-21: `uv run pytest paper/tests -q` — **370 passed, 1 pre-existing
failure** (`test_propagation_data_runs_end_to_end_on_a_small_subset`, another plan's smoke test, which
overran its 120 s budget at 166.0 s on this run). Not touched here. `uv run pytest viz/tests -q` —
**41 passed**.
