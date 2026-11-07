# Sound speed and propagation regime on the DAS fin-whale data — Plan A findings

_Written 2026-09-17, branch `sound-speed-research`. Reports Plan A of the research design in
[`paper/docs/specs/2026-09-15-sound-speed-research-design.md`](specs/2026-09-15-sound-speed-research-design.md),
which executes the decision gate in [`paper/docs/sound-speed-feasibility.md`](sound-speed-feasibility.md)._

Every number below was re-read before being written here, either from its committed table in
`paper/results/` (read those with `pd.read_csv(path, comment="#")`; each
carries a provenance header) or from the committed source named at the point of use. A handful
of figures come from neither — they were measured during the plan from data that is not
committed, chiefly the reference association archive under the gitignored `data/`. Those are
marked **[not table-backed]** where they appear, so no reader has to guess which is which.

---

## 1. What was asked

The feasibility note proposed that replacing the standard constant `c = 1480 m/s` with a real
depth-dependent sound-speed field would (a) matter little for 2-D single-cable localization but
(b) possibly make **source depth** observable, because the Oregon cables run shelf-to-abyssal and
rays to channels at very different depths sample very different parts of the profile.

Re-scoping the note on 2026-09-16 added a third question the note had not asked. Perturbing a
*constant* `c` tests a scale factor; it says nothing about a *gradient*, which refracts — and
refraction can delete a propagation path rather than retime it. The spec therefore carried two
open questions into Plan A:

> **Q-depth.** Does the shelf-to-abyssal depth aperture make source depth `z_s` identifiable, and
> does constant-`c` bias it?
>
> **Q-path.** Is a straight line the right *path*, not merely the right speed? If the true first
> arrival is bottom-interacting, constant-`c` models the wrong path and that affects 2-D as well
> as 3-D.

Plan A covers spec §7.0–7.3: reproduce the published baseline, characterise the propagation
regime, and run the two sound-speed ablations. Plan A was explicitly permitted to conclude that
**the sound-speed programme is aimed at the wrong term**. The depth experiment itself (§7.4–7.6)
is Plan B and was not run.

---

## 2. What was measured

### 2.1 The data, and the denominators every number below sits on

The annotation snapshot (`fixtures/das_calls.csv.gz`, both cables, all five batches):

| Quantity | Value |
|---|---|
| Annotated calls | **628** |
| Picks | **11,117** |
| Picks per call | median **14.0**, min 3, max 61 (batch 1 alone: median **14.0**, mean **18.63**) |
| Pick depths | −83.8 m to −1,554.9 m (from `geometry.load_cables`'s channel-to-depth mapping, joined onto each pick's `(cable, dist)` via `dclde_localization.physics.picks.attach_xyz` — the snapshot CSV itself carries no depth column) |
| Calls with ≥ 8 picks (`min_picks`, the eligibility floor for every solve below) | **524 / 628** |

Every experiment in this document therefore has at most 524 eligible calls, not 628, and each
reports how many of those actually produced a usable basin. Those counts differ per experiment
and are given in place.

### 2.2 The experiments and their tables

| Spec section | Question | Table |
|---|---|---|
| §7.1 | Can the published baseline be reproduced per call? | `repro_join_feasibility.csv`, `repro_baseline.csv` |
| §6 | How far is reanalysis from in-situ measurement? | `profile_validation.csv`, `hycom_profile_*.csv`, `hycom_spatial_consistency_check.csv` |
| §7.0.1 | Does the residual range trend flatten under a source-depth sweep? | `regime_depth_vs_path.csv` |
| §7.0.3 / §7.6 | Do trapped modes exist where ray theory says shadowed? | `regime_shadow_zone.csv` |
| §7.2 | Does a free scalar `c` absorb the effect? | `ablation_free_c.csv`, `ablation_free_c_flip_diagnostics.csv` |
| §7.3 | Does a depth-dependent effective `c` move positions? | `ablation_effective_c.csv`, `ablation_effective_c_flip_detail.csv` |

Not run, and therefore not reported: §7.0.2 (moveout-shape test), §7.4 (depth observability),
§7.5 (pick density), §8 (noise model and simulator).

---

## 3. Findings

### 3.1 The published baseline cannot be reproduced per call, and that is itself the finding

The released baseline artifact — Goestchel et al.'s `batch1_localizations_with_coords.csv`,
**176 rows**, batch 1 — **carries no call identifier**. The only available join key is the `utc`
timestamp. At this dataset's calling rate that key does not identify a call.

Joining each baseline row to an annotated call, requiring the pairing to be **mutually unique**
(exactly one candidate call for the row *and* exactly one candidate row for the call), over
all 176 rows:

| Tolerance | matched | ambiguous | unmatched |
|---:|---:|---:|---:|
| 90.0 s (the interface default) | **0** | 176 | 0 |
| 30.0 s | 0 | 176 | 0 |
| 10.0 s | 1 | 174 | 1 |
| 5.0 s | 4 | 154 | 18 |
| 2.0 s | 21 | 93 | 62 |
| 1.0 s | 33 | 34 | 109 |

At the default tolerance, **0 / 176 rows (0.0%)** join unambiguously. Tightening does not fix
it, it trades one failure for another: the largest matched count anywhere in the sweep is
**33 / 176 (18.8%)** at 1.0 s, bought by pushing **109 / 176 (61.9%)** outside tolerance
entirely. Before the mutuality test, every single baseline row has at least 10 candidate calls
within tolerance (median 30, max 49).

The cause is call density, not a bad threshold. The 176 baseline localizations sit against 147
distinct annotated calls in a ~10-minute window — roughly one call every 4 s — against a
timestamp whose precision cannot separate them.

An alternative rule that accepts a row only when its nearest candidate beats the second-nearest
by a margin, and no other accepted row claims the same call:

| Margin | uncontested | contested | ambiguous | unmatched |
|---:|---:|---:|---:|---:|
| 1.0 s | **21** | 57 | 98 | 0 |
| 2.0 s | 16 | 32 | 128 | 0 |
| 3.0 s | 10 | 21 | 145 | 0 |
| 5.0 s | 4 | 7 | 165 | 0 |

The committed reproduction run uses margin 1.0 s: **21 / 176 (11.9%)** uncontested, of which
4 fall below `min_picks = 8`, leaving **17 / 176 (9.7%)** solved.

On those 17 calls, the distance between our solved position and the published one is:

```
n = 17      min 6.7 m    p25 102.7    median 247.0    p75 2,034.9    p90 21,440.8    max 46,161.6 m
9 / 17 below 500 m;  4 / 17 above 15 km
```

`delta_nearest_basin_m` is numerically identical to `delta_m` for all 17 rows — none of the 17
is a case of "right position, wrong lobe"; the best basin was already the one nearest the
published position in every case. None of the 21 matched rows has a confidently-resolved
competing basin (`separation_resolved = False`, 21/21).

**What this does and does not support.** The median 247 m is not a reproduction accuracy figure
and must not be quoted as one. Three reasons, all of which push in the same direction:

1. **The 17 are a biased sample.** Margin mode selects rows that are *isolated in time*, i.e.
   drawn from sparser stretches of the recording. That is not a random 10% of the 176.
2. **Timing isolation is not identity.** A baseline row can be isolated from every other
   *annotated* call and still describe a different real vocalization — our annotations are a
   subset of everything that was calling. Some of the 15–46 km tail is plausibly residual join
   error rather than estimator error, and we cannot separate the two. Nor does
   under-determination cover the tail: those four calls carry **8, 18, 22 and 23 picks**. Against
   the median pick count of the 17 solved calls (23), one of the four is at or above it; against
   the median of all 21 matched calls (18), three of the four are. Either way the tail is not
   simply the most pick-starved solves — the single worst call (46,162 m) has 18 picks, a *low*
   misfit (1.6) and the tightest join gap of any solved row (0.018 s), while the second-worst
   (26,332 m) has 22 picks and a misfit of 25.3 against a median of 2.51 over the other 16 solved
   rows. A high misfit is weak circumstantial evidence for two real events mixed into one "call";
   the worst row shows the opposite signature. No single explanation covers both.
3. **The two estimators are not solving the same input.** Measured directly from the association
   archive (12 batch-1 `Baseline` windows, 194 `assoc_pair` items): the baseline's own
   association draws **min 324, median 4,684, mean 5,328, max 12,228** picks per candidate call
   *[not table-backed: measured during Task 7 from `paper/data/reference/associations.zip`, which is
   gitignored; restore it with `paper/scripts/fetch_reference_data.py` to re-check].*
   Ours draws a median of 14.0 (batch 1 alone: median 14.0, mean 18.63 — the mean is what an
   earlier draft of this document mislabelled as the median) from the challenge annotations. A
   close or a poor agreement is therefore not a statement about the two estimators.

**This is a property of the released artifacts, not of this codebase**, and is worth raising with
the challenge organizers independently: a published localization set with no call identifier
cannot be checked call-by-call by anyone.

### 3.2 Source depth does not explain the residual range trend

The spec's §2.5C measured a range-structured residual trend — mean pick residual rising with
source–receiver range, and rising *fastest at the highest SNR*, which rules out the obvious
confound (late picks on weak envelopes). That trend was measured with source depth pinned at
−30 m, so path error and depth error were confounded.

Sweeping assumed source depth from −5 m to −200 m with the propagation model held fixed at
`ConstantC(1480)`, so depth is the only thing that varies:

| `source_z` | solved | trend, \|r\|<1 s cut (ms) | trend, median (ms) | trend, raw (ms) |
|---:|---:|---:|---:|---:|
| −5 m | 497 | 81.8 | 16.9 | 41.7 |
| −15 m | 498 | 80.8 | 17.5 | 39.5 |
| −30 m | 498 | 79.9 | 16.7 | 38.8 |
| −60 m | 498 | 80.8 | 16.9 | 41.2 |
| −100 m | 499 | 78.4 | 16.3 | 38.3 |
| −150 m | 502 | 75.9 | 14.8 | 34.5 |
| −200 m | 500 | 75.8 | 14.9 | 35.6 |

Denominators, identical in structure at every row: 628 calls total, 104 excluded for fewer than
8 picks, 22–27 excluded because the estimator returned no usable basin, 0 solver errors,
**497–502 solved**, 10,246–10,297 picks. `trend = far_mean − near_mean` over range bins
[0, 5, 10, 20, 30, 200] km. The primary statistic drops picks with |residual| ≥ 1 s — 10 near and
8 far at every depth, 18 of the 3,833–3,874 picks in the two bins the trend is computed over
(**0.465–0.470%**) — because the residual distribution has a gross mis-association tail out to
tens of seconds; median and uncut statistics are carried alongside rather than chosen after the
fact.

**The trend does not flatten, under any of the three statistics**, across a 40× change in the
confounded parameter. None approaches zero, and each stays within a narrow band about its own
mean across the whole sweep:

| Statistic | range across the sweep | mean | spread about the mean |
|---|---|---:|---|
| cut (\|r\| < 1 s), primary | 75.8 – 81.8 ms | 79.1 | −4.2% / +3.5% |
| median | 14.8 – 17.5 ms | 16.3 | −9.1% / +7.5% |
| raw / uncut | 34.5 – 41.7 ms | 38.5 | −10.5% / +8.2% |

The two secondary statistics vary about twice as much about their means as the primary one does —
worth stating rather than rounding all three to a single reassuring figure — but none of the
three shows structure that heads toward zero at any depth.

The one thing that could be mistaken for structure is a decline in the cut trend from 81.8 ms at
−5 m to 75.8 ms at −200 m (≈7%). It is not monotonic — the sequence is 81.8, 80.8, 79.9, **80.8**,
78.4, 75.9, 75.8 — and we do not read it as signal: the naive per-depth
standard error on the trend is 5.1–5.2 ms at every depth, so the drift across the *whole* sweep
is ~1.2× a single depth's own sampling noise. That comparison is itself rough — the same ~500
calls are re-solved at every depth and are not independent draws, and no paired bootstrap was
run — so the right statement is that the drift is not resolved either way by what was measured.

**What this establishes: source depth does not explain the range trend over −5 to −200 m. It
does not identify what does.** The experiment falsifies one confound by elimination. Propagation
path, residual mis-association surviving the 1 s cut, bathymetry error and timing/clock error are
all still live and were not distinguished from one another. The operationally useful consequence
is narrow but firm: **Plan B's depth experiment cannot use "the range trend is a depth artifact"
as its premise.**

### 3.3 The ray-theoretic shadow zone does not survive a modal treatment

§2.5B of the spec claimed the straight-line path the model integrates along "does not exist" for
most of this geometry, because a downward-refracting profile gives a shallow source no direct
water-borne ray to a distant channel. At 20 Hz (λ ≈ 74 m) in 100–1,575 m of water, the guide is
1.3–21 wavelengths deep, which is exactly the regime where ray theory's sharp shadow boundary is
a claim normal modes exist to adjudicate.

Five geometries, source at 30 m depth, 20 Hz, on the `hycom_south_deep` profile, with seabed
sound speed swept over 1550 / 1700 / 1900 m/s (three values, `n_seabed = 3` per row):

| `z_rcv` | range | ray max direct range | ray says shadowed | modes (min–max over the seabed sweep) |
|---:|---:|---:|:---:|---:|
| 100 m | 5 km | 1,255 m | yes | **1 – 2** |
| 150 m | 10 km | 1,617 m | yes | **1 – 3** |
| 500 m | 20 km | 3,846 m | yes | **4 – 8** |
| 1,000 m | 30 km | 6,986 m | yes | **8 – 17** |
| 1,575 m | 40 km | 11,077 m | yes | **13 – 27** |

Ray theory calls all five shadowed. **A trapped mode exists in all five, for every seabed in the
sweep** — the minimum over the whole table is 1, never 0. The seabed sweep matters
quantitatively (13 to 27 modes at the deepest geometry) but not for the existence question,
which is stable across the entire swept range of an unknown parameter.

The mode solver is validated against the analytic Pekeris waveguide *[not table-backed: Task 10,
reproducible from `paper/tests/test_modes.py`]*: 12/12 exact on hand-chosen
configurations and 224/224 exact on a systematic sweep over water/seabed speed, depth and
frequency, with mode wavenumbers agreeing with an independently-bisected analytic dispersion
relation to 4.0e-8 rad/m.

**Bounding this, because it is the finding most likely to be over-read.** The solver computes
*existence and counting*, not arrival. It models **no modal attenuation** (`Seabed.alpha` is
carried but deliberately unused — the eigenproblem is real by construction), **no source
excitation** (whether a whale at −30 m puts energy into a given mode depends on the mode shape at
that depth), and **no transmission loss**. Near-threshold modes — the ones that decide the
marginal cases — are precisely the ones that interact hardest with the seabed and attenuate
fastest.

So, precisely:

- **Falsified:** the ray-theoretic statement that *no propagation path exists* at these
  geometries. That statement is a ray artifact of applying geometric acoustics to a
  few-wavelength-deep guide, and §2.5B should not have been written as it was.
- **Not established:** that energy actually arrives at a receiver 5–40 km away from a −30 m
  source. Concluding "the shadow zone is not real" from mode counts alone would over-read this
  module. That step needs a transmission-loss calculation, which was not shipped.

Two further bounds on the table itself. Each row computes modes for a range-*independent* guide
at the receiver's own depth; a real shoaling path's propagating mode set is bounded by its
shallowest point, so the deep rows are an upper bound on the along-path count. That bias was
checked against the existence conclusion rather than assumed: sweeping water depth on the same
profile, at least one mode is trapped down to 70 m (slowest seabed) and 40 m (the other two)
*[not table-backed: a Task 12 verification sweep, reproducible from `soundspeed/modes.py`]*,
both below the cable's ~100 m shallow end. And `all_converged` is False for the `z_rcv = 150 m`
row — the mode *shapes* there are under-resolved, though the count is not, and rows were not
filtered on convergence because doing so would bias away from the marginal regime the question
lives in.

**The profile this rests on was never validated against measurement.** See §3.6.

### 3.4 A free scalar `c` is a nuisance sink, not a sound-speed estimator

The feasibility note proposed fitting `c` as a free fifth unknown as the cheap version of the
idea, and the spec expected it to absorb most of the effect for 2-D — making it the null model
the depth experiment would have to beat.

Fitting `c` per call over a bracket of [1430, 1530] m/s by a three-pass coarse-to-fine search
(27 solves per call plus one fixed-1480 comparator; achieved resolution 0.20–0.78 m/s, 3–13×
finer than the configured 2.5 m/s tolerance). **498 calls**, of 524 eligible, of 628 annotated —
26 dropped because the estimator returned no usable basin at some point in the search.

Fitted `c`: median **1450.3 m/s**, mean 1459.2, std 33.1, IQR [1430.0, 1469.1].

The physically plausible band for water at this site, taken from the profile's own range, is
**1477.5 – 1497.5 m/s**. Against that:

- **23 / 498 (4.6%)** fitted values are physically plausible.
- **208 / 498 (41.8%)** are pinned at a search-bracket edge — 149 (29.9%) at the low edge,
  59 (11.8%) at the high edge. Real ocean sound speed does not vary by ±50 m/s across calls
  seconds apart.

Position shift against fixed 1480, split per the reporting contract and never merged:

| Group | n | median shift | p90 | max |
|---|---:|---:|---:|---:|
| Basin-stable (shift ≤ 500 m) | 311 (62.4%) | **185.8 m** | 353.9 m | 499.0 m |
| Basin-flipped (shift > 500 m) | 187 (**37.6%**) | 1,575.5 m | 6,527.7 m | 65,466.3 m |

Within the basin-stable group, stratified by what the fitted `c` actually was:

| Stable subgroup | n | median shift |
|---|---:|---:|
| `c` pinned at a bracket edge | 113 | **250.1 m** |
| `c` physically plausible | 18 | **58.7 m** |
| neither | 180 | 168.9 m |

Calls whose fitted `c` is physically plausible move **4× less** than calls whose fitted `c` is
unphysical. The free parameter is doing most of its work where its value is meaningless.
`best_misfit ≤ fixed_misfit` for all 498 rows, as it must for a strictly larger model — the
misfit improvement is real and tells us nothing about whether the parameter is physical.

**Conclusion: a free scalar `c` mostly absorbs model and association error, not sound speed. It
is not a usable null model for Plan B's depth experiment to beat.** What exactly it is absorbing
was not separated further; the edge-pinning asymmetry (low edge hit 2.5× more than high) is
consistent with a subset of calls whose real failure is on the picks side — the same gross
mis-association tail that §3.2's outlier cut removes — but that was not cross-checked and is not
claimed.

**The 37.6% flip rate overstates real ambiguity.** Re-solving all 187 flipped calls at both their
`c` values:

- The *winning* position is a confidently refined local minimum in **183 / 187** (fixed-`c` solve)
  and **181 / 187** (free-`c` solve). The reported positions are not fallback stand-ins.
- A confidently-resolved *competing* basin exists in only **20 / 187 (10.7%)** on the fixed solve;
  39/187 on the free solve; both resolved in 14/187 (7.5%); either in 45/187 (24.1%).
- Where a competing basin *is* confidently resolved it sits far away — a median **14,961 m** over
  the 20 fixed-solve resolutions, **11,015 m** over the 39 free-`c` ones, and **12,167 m** pooling
  all 59 resolved separations (45 distinct calls, 14 of which resolve on both solves). Whichever
  denominator is used, this is genuine kilometre-scale mirror ambiguity, not a close call.

So most flips are "a confident winner replaces a weakly-characterised runner-up", not two
well-characterised lobes trading rank. **Genuine large-scale mirror ambiguity is more like
11–24% (20–45 of 187), not 37.6%.**

### 3.5 The actual sound-speed correction is unresolvable at this dataset's SNR

Replacing the constant with a depth-dependent effective speed, at fixed nominal `c`, against
`ConstantC(1480)`. Denominators: 524 eligible calls; `ConstantC` solved 498, `PathAverage` 501,
`RayTrace` 501; the comparison uses only calls **both** sides solved, **n = 489**.

| Model vs constant | n | stable | flipped | median stable delta | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| `PathAverage` | 489 | 459 | 30 (6.1%) | **9.01 m** | 109.7 m | 451.4 m |
| `RayTrace` | 489 | 459 | 30 (6.1%) | **9.47 m** | 114.8 m | 451.4 m |

Both models flip **exactly the same 30 calls** (verified per call, not assumed); those 30 move a
median 1,273 m.

The scale that matters is this dataset's own pick-noise floor. Solving every call under the
baseline's own assumptions (`c` = 1480, `z_s` = −30) and taking residuals after analytic `t0`
elimination gives a per-call residual std distribution over **n = 524 calls / 10,517 picks** — the
calls with at least 8 picks — whose median (p50) is **~96 ms** (p10 30 ms, p90 150 ms). At
`c = 1480` that median is a range-error floor of **142.1 m**. *[not table-backed: measured at spec
§2.2 and named at `soundspeed/raytrace.py:133`; the residual distribution itself is not a
committed table.]* Note it is an upper bound on pick noise, since these are residuals of a
constant-`c` model and therefore contain model error as well. Against that floor,
a depth-dependent effective sound speed moves the **459 basin-stable calls of the 489 compared**
(30 calls, 6.1%, flip basins instead and move a median 1,273 m — see the table above; never merged
into this figure) by **6.3% (PathAverage) to 6.7% (RayTrace) of the noise floor**.

**That is not "small but real". It is unresolvable.** This measurement neither confirms nor
refutes that a depth-dependent effective speed matters — it establishes that *this dataset, at
this SNR and this pick density, cannot see it*. For horizontal position, the water-column
sound-speed correction the feasibility note proposed chasing is below the floor.

**The refracted-ray arm did not test refracted-ray physics.** `RayTrace` falls back to
`PathAverage` whenever no direct ray connects the query points. Measured two independent ways:

- At each call's own best-fit position: **71.85%** of picks fall back (n = 10,261 picks across
  501 calls).
- Over the 161×161 coarse search grid that actually drove the localization: **99.38%** of queries
  fall back (n = 265,975,381 query pairs across 501 calls).

So for 99.4% of the queries that produced the `RayTrace` numbers, `RayTrace` *was*
`PathAverage`, through a different code path. The two rows above are not independent
corroboration of each other, and the near-identical numbers are a consequence of that, not
evidence of agreement between two physics models. **Snell refraction through `c(z)` remains
untested on this geometry** — which is the same fact §3.3 reports from the other side: at these
ranges there mostly is no direct ray to trace.

**This section's 9.0–9.5 m figure, like §3.3's, rests on `hycom_south_deep` — one reanalysis
product, at a site never checked against in-situ measurement (§3.6).** The spec's §6.4 requires
key experiments to run against two independent products (WOA23 was the identified cross-check and
was not implemented) and treats a conclusion that differs between them as unsupported; that bar is
unmet here. Nothing above changes if `hycom_south_deep` is wrong in a way `hycom_oregon_nov`'s
0.47 m/s in-situ agreement, measured 87 km away, does not catch — but a reader relying on the
"6.3–6.7% of the noise floor" figure should carry that single-product caveat forward with it, not
just the noise-floor comparison.

### 3.6 The profile the conclusions rest on is only partly measured, and the one actually used is not measured at all

The spec's §6 inverted the intended data strategy after finding that in-situ coverage for
November 2021 is far thinner than the feasibility note claimed: reanalysis as the base,
in-situ as validation.

**Validation, over the band where in-situ data exists.** HYCOM GOFS 3.1 at the OOI mooring
(44.37, −124.95; 0–600 m; 29/40 levels non-fill, from the profile CSV's own `# source:` header)
against QARTOD-filtered OOI CE04OSPD deep-profiler
CTD casts:

| Window | QC-passing in-situ rows | Band actually validated | median \|Δc\| | p90 | max |
|---|---|---|---:|---:|---:|
| 1–6 Nov 2021 (primary) | 5,067 / 7,472 (68%) | **219–573 m** | **0.471 m/s** | 1.085 | 1.372 |
| 25 Oct – 15 Nov 2021 *[not table-backed]* | 93,901 / 129,177 (73%) | 157–572 m | 0.363 m/s | 1.365 | 2.888 |

**Only the first row is a committed table.** `profile_validation.csv` holds the primary window
alone; its 400 rows are the comparison grid, not the in-situ sample count. The second row is a
sensitivity check reported in the Task 9 run log and reproducible from
`paper/scripts/validate_hycom_profile.py`, which computes both windows but writes only the
primary one. Its `n = 400` and its three statistics are therefore not independently checkable
from this repo's committed results — treat it as a consistency signal (the agreement does not
depend on which QC-passing window is chosen) rather than as a second measurement. Sub-1 m/s
median agreement is a good result — **for that band**.

**The band is roughly a third of the problem.** The cables span 0 to −1,575 m. There is **no
QARTOD-passing in-situ measurement above ~157–219 m** (the edge is window-dependent; the raw
mechanical sweep reaches 97 m but the shallowest ~120–450 m of it is QARTOD-suspect for
temperature and is dropped — the usable bands are recorded in
`soundspeed/erddap.py::NOV_2021_STATUS`) **or below ~575 m**, anywhere on this coastline in this
period. The entire upper water column — where every
source sits and through which every ray to a shallow channel passes — is unvalidated model
output. Above and below the validated band, the profile is reanalysis and nothing else.

**And the profile the experiments actually used was never validated at all.** The validation-site
profile only reaches ~600 m, which covers 7,689/11,117 (69.2%) of real pick depths *[not
table-backed: Task 9 coverage check, recomputable via `soundspeed.reanalysis.coverage_stats`]*;
30.8% of
picks would have silently hit clamped extrapolation, landing specifically on the deep channels
that give this approach its vertical aperture. So §3.3 and §3.5 use a *different* profile,
`hycom_south_deep`, fetched at the South cable's own deepest channel (45.15, −125.11; 0–1,500 m;
35/40 levels non-fill; covers 10,935/11,117 = 98.4% of pick depths *[same source]*, the
remainder needing ≤55 m
of clamped extrapolation).

`hycom_south_deep` has **no in-situ check of its own**. Its credibility is inferred from the
validation site's 0.47 m/s agreement — **87 km away and ~1,000 m shallower**. Every result in
§3.3 and §3.5 inherits that. The tables say so in their own `profile_validated` column rather
than relying on this document.

Two weaker signals, neither a validation:

- HYCOM at the South cable's deep end vs. its midpoint, over their shared 0–350 m band (n = 400):
  median 0.488 m/s, p90 1.139, max 1.470. This is **HYCOM against HYCOM** — it bounds how much
  the choice of fetch point matters, and proves nothing about accuracy.
- Only **one** reanalysis product was used. The spec's §6.4 requires key experiments to run
  against two independent products and treats a conclusion that differs between them as
  unsupported. WOA23 was identified as the cross-check and not implemented. **That requirement is
  unmet**, and every profile-dependent conclusion here is single-product.

---

## 4. What the evidence does not support

Stated separately, because several of these were asserted in earlier drafts of this programme's
own documents:

1. **Not supported: that our estimator agrees with the published baseline to ~250 m.** That
   number is a median over 17 calls selected for timing isolation, joined by a key that cannot
   prove identity, against an estimator using 335× more picks (4,684/14, both medians — §3.1
   item 3). It is not a reproduction result.
2. **Not supported: that propagation-path error causes the residual range trend.** Only "source
   depth does not cause it" was measured. Path error, residual mis-association, bathymetry error
   and timing error were not distinguished.
3. **Not supported: that energy arrives via trapped modes at 5–40 km.** Mode *existence* was
   computed. Attenuation, source excitation and transmission loss were not.
4. **Not supported: that a free scalar `c` absorbs the sound-speed effect for 2-D.** The opposite
   was measured — the parameter mostly takes unphysical values, and the calls where it is
   physical move least.
5. **Not supported: that a depth-dependent effective speed does not matter.** It was measured, over
   the 459 basin-stable calls of the 489 compared (the other 30 flip basins instead, never merged
   into this figure), at 6.3–6.7% of this dataset's pick-noise floor, i.e. below what this data can
   resolve. "We cannot distinguish it from zero" is the correct statement, not "it is zero".
6. **Not supported: that `RayTrace` corroborates `PathAverage`.** 99.4% of its queries *were*
   `PathAverage`.
7. **Not supported: any profile-dependent conclusion at the level the spec requires.** One
   product, validated over ~a third of the depth range, at a site 87 km from the one used.

---

## 5. Verdict on Q-path and Q-depth

### Q-path — "is the sound-speed programme aimed at the wrong term?"

**Partly yes, and the replacement term is not identified.** The spec explicitly permitted this
conclusion, so it is stated plainly:

- The term the programme was aimed at — the water-column sound-speed correction to horizontal
  position — is **measurably too small to matter on this dataset**: a median 9.0–9.5 m over the
  **n = 459 basin-stable calls of 489 compared** (30 calls, 6.1%, flip basins instead and move a
  median 1,273 m — never merged into that 9.0–9.5 m figure), against a 142 m noise floor (§3.5).
  That figure also rests on one, never-in-situ-validated reanalysis product (`hycom_south_deep`;
  see §3.5's own caveat and §3.6) — the spec's two-independent-product bar is unmet, so treat
  "measurably too small" as this product's answer, not a cross-checked one. Continuing to refine
  it will not change a localization.
- A much larger, range-structured model error is present — 76–82 ms of trend across the range
  bins, an order of magnitude above the ~9 ms the profile correction supplies — and it is **not**
  source depth (§3.2). Its cause is unidentified.
- The specific mechanism §2.5B proposed for it — that no direct path exists, so the first arrival
  must be bottom-interacting — **is not supported as written** (§3.3). Trapped modes exist
  wherever ray theory calls the geometry shadowed. That does not rescue the straight-line model
  either; it means the ray-theoretic argument for abandoning it was wrong, not that the model is
  right.

So the honest verdict is narrower than either "sound speed is the lever" or "bottom interaction
is the lever": **sound speed is demonstrably not the lever for horizontal position on this data,
and we have not established what is.**

### Q-depth — "does the depth aperture make `z_s` identifiable?"

**Not answered here, by design** — that is Plan B's §7.4. What Plan A changes about it:

- The depth aperture itself still stands (spec §2.1, unchallenged by anything measured here).
- Plan B **cannot** justify itself with "the range trend is a depth artifact" (§3.2 falsifies it)
  and **cannot** use free-`c` as its null model (§3.4 disqualifies it). Both were load-bearing in
  the original framing. Plan B needs a new justification and a new null before §7.4 is
  interpretable.

---

## 6. What a follow-on plan should do

In rough order of what would change a decision per unit of effort.

1. **Close Q-path with transmission loss, not more mode counts.** §3.3 took the question as far as
   an existence calculation can. The decisive missing piece is whether a mode trapped at these
   geometries *survives* 5–40 km and is *excited* by a source at −30 m — modal attenuation with
   swept sediment `alpha`, source excitation from the mode shape at source depth, and a mode-sum
   transmission loss. Until that exists, "the first arrival is/is not a direct path" stays open,
   and it is the question that decides whether any of §7.4 is interpretable.
   **Done (2026-09-21).** Answered in
   [`paper/docs/das-finwhale-transmission-loss-findings.md`](das-finwhale-transmission-loss-findings.md):
   the verdict lands in the spec's pre-registered **direct-path** branch, **on that branch's
   comparative clause** — the observed range-and-depth shape of pick SNR is less inconsistent with
   a direct water path than with a trapped one at every seabed in the sweep, and the modal
   hypothesis additionally predicts non-detection for 3,051–3,311 of 8,660 eligible picks that were
   detected. Three caveats travel with it. The branch's *absolute* clause ("H-geometric reproduces
   the observed shapes within their uncertainty") is **untested**, because the pre-registered
   statistic is comparative and an absolute fit claim needs a calibration this observable lacks —
   so the result supports "the trapped picture fits worse", not "direct-path spreading fits". The
   comparison settles which *shape* fits under an assumed dB scale that the observable's own
   physics-anchored calibrator calls into question — and that possible miscalibration is a
   *compression*, which would bias the comparison toward the shallower fixed shape, i.e. toward
   H-geometric, the hypothesis that won; magnitude unknown, competing explanations still live.
   And amplitude is measured to carry no
   resolvable receiver-depth information, so §7.4 must reach depth through travel-time moveout
   (item 2 below) rather than through level.
2. **Run §7.0.2, the moveout-shape test.** It has no task in Plan A and was left out because
   §7.0.1 and §7.0.3 answered the same question more cheaply. They came back "not depth" and
   "not a ray shadow" respectively — i.e. ambiguous on the thing that matters — so per Plan A's
   own self-review this becomes the first task of Plan B rather than an afterthought. Direct and
   bottom-bounce arrivals trace measurably different curvature in (along-cable distance, time),
   and the picks to fit both are already on disk.
3. **Stop spending on water-column sound speed for horizontal position.** §3.5 measures it below
   the noise floor for the 459 basin-stable calls of 489 compared (30 flip basins instead, median
   1,273 m — never merged into "below the noise floor"). It may still matter for depth — that is
   §7.4's question, not this one — but the 2-D branch is closed by measurement, not by the argument
   the feasibility note used.
4. **Ask the organizers for a call identifier.** §3.1 is a property of the released artifacts. A
   published localization set joinable only by timestamp, at one call per 4 s, cannot be verified
   call-by-call by anyone. Ask for (a) a call-id mapping for the baseline CSV and (b) the answer
   to the spec's §13 question about whether any depth ground truth exists at all — if challenge
   ground truth is the baseline's own fixed-depth output, L-2's Z component is unscoreable and
   that should be said publicly.
5. **Either fix `RayTrace` or retire it.** At 99.4% fallback it is a `PathAverage` alias with a
   misleading name. Testing refracted-ray physics on this geometry requires a model that handles
   bottom-interacting paths, which is the same gap as item 1 and remains open. **Done (Task 14
   whole-branch review):** `RayTrace.last_fallback_fraction` used to be overwritten by every
   internal `travel_time` call, silently answering a different question than it advertised (the
   most recent call, usually the smallest and least representative, rather than anything
   summarising a whole solve). It is now a per-call scalar with that scope stated in its own
   docstring, backed by `RayTrace.fallback_calls` — an accumulating record of every
   `(n_queries, n_fallback)` call since construction or the last `reset_fallback_record()` — and
   `fallback_fraction_overall`, the pooled figure a solve loop or ablation should actually read.
   `experiments/ablation.py`'s `raytrace_fallback_fraction_at_best_fit` and
   `raytrace_fallback_over_search_grid` use it directly; the 71.85%/99.38% pair in §3.5 is now
   produced by that tested code path, not a one-off script recomputing it outside the class.
6. **Fix the profile evidence base before leaning on it further.** Either validate
   `hycom_south_deep` at its own location and depth range, or restrict conclusions to what the
   validated profile supports. Add the second independent product (WOA23) that §6.4 requires;
   right now no profile-dependent conclusion in this document meets the spec's own bar.
7. **Separate the free-`c` nuisance term from pick error.** §3.4's 41.8% edge-pinning is an
   unexplained signal sitting on top of every ablation. Cross-checking the edge-pinned calls
   against the gross-mis-association outliers, and widening the bracket to see whether pinning
   persists, are both cheap and would tell us whether the ablations are contaminated by
   association failure.

---

## 7. Corrections issued

This plan found several claims stated more strongly than their evidence, including in its own
source documents. Both were amended rather than silently edited; the corrections are marked in
place.

**To `paper/docs/sound-speed-feasibility.md`** — OOI availability (site code, which instruments were
profiling, the usable band), the `z = −30.0` claim, the ingestion-effort estimate, and the
constant-versus-gradient category error.

**To `paper/docs/specs/2026-09-15-sound-speed-research-design.md`** — §2.3's "~9,500 picks
per call" (measured median 4,684, range 324–12,228), §2.4's usable in-situ band, and §2.5B's
shadow-zone claim, which §3.3 overturns.

---

## 8. Reproducing this

**Corrected (Task 14 whole-branch review, Critical 1).** An earlier draft of this section claimed,
unconditionally, that every table "regenerates from committed configs ... via
`paper/src/dclde_soundspeed/experiments/`." That was false: before this review, `load_config` was
called only from `paper/tests/test_runner.py`, no `__main__`/`argparse`/`[project.scripts]`
existed anywhere under `experiments/`, `paper/configs/ablation_free_c.yaml`
raised `ValueError` if loaded through `load_config` (by design — it deliberately does not declare
`truth_model`/`inversion_model`, see its own notes), and every table below was in fact produced by
one-off `uv run python -c "..."` snippets, most of them quoted only in
`.superpowers/sdd/2026-09-16-sound-speed-plan-a/task-*-report.md`, not committed anywhere runnable.

`paper/scripts/run_experiment.py` is the fix: a real driver, `uv run python -m
paper.scripts.run_experiment --config <path>`, one dispatch per experiment config, each
backed by tested functions in `experiments/{repro,regime,ablation}.py` (`effective_c_report` and
`free_c_flip_diagnostics` were promoted from the one-off snippets specifically for this). Per table:

| Table | Regenerable offline via the driver? | Notes |
|---|---|---|
| `repro_join_feasibility.csv`, `repro_baseline.csv` | **Yes** — verified byte-identical (except the timestamp header) | needs `paper/data/reference/` restored first (`paper/scripts/fetch_reference_data.py`; a one-time git+zip fetch, not a live query) |
| `regime_depth_vs_path.csv` | **Yes** — verified byte-identical | fully offline, no external data beyond the committed picks snapshot and geometry |
| `regime_shadow_zone.csv` | **Yes** — verified byte-identical | uses the committed `hycom_south_deep` profile CSV; no network |
| `ablation_effective_c.csv`, `ablation_effective_c_flip_detail.csv` | **Yes** | uses the committed `hycom_south_deep` profile CSV; no network |
| `ablation_free_c.csv`, `ablation_free_c_flip_diagnostics.csv` | **Yes** — every value identical; row/column order differs from the committed CSV (a stale ordering from before a later refactor, not a data difference — see below) | no profile dependency (fits `ConstantC` at different `c`); ~15 minutes end to end |
| `profile_validation.csv`, `hycom_profile_oregon_nov2021.csv`, `hycom_profile_south_cable_deep.csv`, `hycom_spatial_consistency_check.csv` | **No** | need **live** network access to OOI's ERDDAP server and HYCOM's NCSS server; produced by `uv run python -m paper.scripts.validate_hycom_profile` directly, not by a YAML config, and cannot be reproduced offline by design (a fresh fetch may also disagree slightly with the committed CSV if either upstream server's data changed since 2026-09-17) |

Regeneration was run and diffed against the committed CSV (via `pandas.read_csv(path,
comment="#")`, ignoring the `#`-prefixed provenance header's timestamp) for every offline-capable
table; every value matched exactly for all six. `ablation_free_c.csv` and
`ablation_free_c_flip_diagnostics.csv` needed a row/column-order-aware comparison (sort by `call`,
compare column-by-column) rather than a direct `DataFrame.equals` — the committed CSV's column
order predates a later refactor of `fit_free_c`'s row-construction dict, so a plain regeneration
reorders `flipped` relative to `achieved_resolution_m_s`, and `free_c_flip_diagnostics` iterates
flipped calls in a different (but equally valid) order than the original one-off script did. No
numeric value differs. Two configs used to name a run they did not perform (`repro_baseline.yaml`'s
`inversion_model`, `ablation_effective_c.yaml`'s single-arm `truth_model`) — both corrected
alongside adding the driver; see each config's own notes and the commit that fixed them.

Everything runs under `uv run` from the repo root. `src/dclde_localization/physics/forward.py` is frozen
and was not modified by any of this work.

The reference data (`Ocean-Data-Lab/Goestchel_JASA_2025b`, GPL-3.0; Zenodo record 21220346) is
not committed — `data/` is gitignored. `paper/scripts/fetch_reference_data.py` re-downloads both from
their canonical sources and applies the `:`→`_` rename that NTFS requires, so a teammate restores
it with one command — but the two artifacts are **not** checked the same way. **Corrected**: an
earlier draft of this section said the script "verifies recorded SHA-256 checksums" for "both"
sources; only the association archive (a zip download, Zenodo record `21220346`) is checked
against a recorded SHA-256, because a zip has no self-verification of its own. The reference
implementation (the git repository) is not checksummed separately — git's own content addressing
(SHA-1 per object) makes a truncated or corrupted clone fail git's own integrity checks before the
fetch script ever reads a blob out of it, which is a different but equally real guarantee, not a
weaker one. See `paper/data/reference/PROVENANCE.md`, which already stated this correctly, for the
per-artifact detail this section previously got wrong.
