# Transmission loss on the DAS fin-whale data — closing Q-path

_Written 2026-09-21, branch `sound-speed-research`. Reports the design in
[`paper/docs/specs/2026-09-20-das-transmission-loss-design.md`](specs/2026-09-20-das-transmission-loss-design.md),
which discharges item 1 of [`paper/docs/das-finwhale-soundspeed-plan-a-findings.md`](das-finwhale-soundspeed-plan-a-findings.md) §6._

Every number below was re-read from its committed table in `paper/results/`
before being written here (read those with `pd.read_csv(path, comment="#")`; each carries a
provenance header whose caveats are reproduced in the relevant section rather than left for a
reader to find). Anything measured during the plan but not backed by a committed table is marked
**[not table-backed]**, as Plan A's findings do.

---

## 1. What was asked

The spec's §1 fixes the question:

> **The question this design answers.** At 20 Hz, over 5–40 km on the Oregon cables, is the first
> arrival consistent with a direct water path, or with a trapped / bottom-interacting path — and
> can this dataset distinguish them at all?

Plan A had established that **trapped modes exist** wherever ray theory called the geometry
shadowed (its §3.3), which overturned the earlier spec's §2.5B argument for abandoning the
straight-line model. That question is settled and is **not** re-litigated here. Existence is not
arrival: this document asks whether such a mode is *excited* by a 20 Hz source at −30 m and
*survives* 5–40 km across a shelf-to-abyssal margin, and whether the observed data can tell the
two pictures apart.

**The verdict criteria were fixed in the spec's §7 before any of these numbers existed**, and
§6.4 fixed the comparison statistic in advance as well, expressly forbidding the substitution of
another after seeing the data. §6 below quotes those criteria and applies them. "Undecidable" is
one of the three pre-defined branches, not a fallback.

The two hypotheses under comparison (spec §4):

- **H-geometric** — geometric spreading plus water absorption; no bottom loss, no mode stripping.
  This is what the straight-ray `ConstantC` / `PathAverage` family implies about amplitude, and it
  is the null.
- **H-modal** — an adiabatic mode sum over the real bathymetry, with per-mode sediment attenuation
  and mode stripping.

Neither is "the truth". The experiment asks which is less inconsistent with the observed shape.

**Positions are an input, not a result.** Range requires a source position, and source positions
come from `solve()` under `ConstantC` — a propagation model, i.e. the kind of thing under test.
The dependency is handled rather than ignored: positions are stated as an input, the sensitivity
is quantified from Plan A §3.5 (a depth-dependent effective `c` moves basin-stable positions a
median 9.0–9.5 m, negligible as a covariate at 5–40 km), and basin-flipping calls — for which that
argument does not hold, since they move a median 1,273 m — are excluded entirely. **No result
below is an independent test of the positions.**

---

## 2. What was measured

### 2.1 The denominators every number below sits on

From `propagation_denominators.csv` (1 row). The six call categories are asserted in code to
partition the whole snapshot, and they do: 104 + 26 + 86 + 31 + 0 + 381 = **628**.

| Category | Calls | Share of 628 |
|---|---:|---:|
| Total annotated calls | **628** | 100% |
| Below `min_picks = 8` | 104 | 16.6% |
| Unsolved | 26 | 4.1% |
| **Basin-flipping (excluded, never merged)** | **86** | **13.7%** |
| Bathymetry path failed | 31 | 4.9% |
| Geometry error | 0 | 0.0% |
| **Eligible** | **381** | **60.7%** |
| **Picks within the 381 eligible calls** | **8,660** | — |

The 86 basin-flipping calls are excluded from every pooled statistic in this document, in the
tables **and** in the prose. They are never averaged in anywhere, including in the max-range
distribution of §4 and the fits of §5.

Two properties of the observables that everything downstream depends on:

- **`dist` is distance along the cable, not source–receiver range.** Range is computed from the
  solved source position to each channel's `(px, py, pz)`.
- **`snr` comes from a denoised strain product with unknown, possibly range-dependent gain.**
  **No absolute SNR level is quoted anywhere in this document.** Only shape — versus range, versus
  depth, versus axial projection — is used as evidence. §7.7 explains why this defence is weaker
  than it looks.

### 2.2 The seabed sweep is site *uncertainty*, not site knowledge

Five seabeds are swept, spanning soft mud through sand/gravel:

| Name | `c` (m/s) | `rho` | `alpha` (dB/λ) |
|---|---:|---:|---:|
| `seabed_c1500` | 1500 | 1.4 | 0.02 |
| `seabed_c1550` | 1550 | 1.5 | 0.05 |
| `seabed_c1700` | 1700 | 1.8 | 0.15 |
| `seabed_c1900` | 1900 | 2.1 | 0.40 |
| `seabed_c2100` | 2100 | 2.3 | 0.80 |

The ranges follow the standard geoacoustic compilations for continental-margin sediments
(Hamilton, JASA 68:1313, 1980; the sediment-property tables in Jensen et al., *Computational Ocean
Acoustics*). **No Oregon-margin-specific geoacoustic measurement was found for these two cables.**
The sweep width therefore stands in for the site uncertainty. It must not be read as knowledge of
this site's seabed, and no statement below asserts what the seabed here *is*.

### 2.3 The tables

| Result | Question | Table |
|---|---|---|
| Denominators | What is every rate below over? | `propagation_denominators.csv` |
| 1 | Does a trapped mode survive 5–40 km and get excited at −30 m? | `propagation_tl.csv` |
| 2 | How far out are calls actually detected? | `propagation_max_range.csv` |
| 3 (descriptive) | How does pick SNR fall with range, depth and axial projection? | `propagation_decay_fit.csv`, `propagation_decay_fit_draws.csv` |
| **3 (pre-registered)** | **Which hypothesis's fixed shape fits better?** | **`propagation_hypothesis_test.csv`**, `propagation_hypothesis_summary.csv` |
| intermediate | H-modal's per-pick predicted shape | `propagation_modal_shapes.csv` |

---

## 3. Result 1 — a trapped mode is excited only weakly, and at the shallowest geometry it may not survive at all

From `propagation_tl.csv`, 25 rows: five idealised shelf-to-abyssal geometries × the five seabeds,
source at 30 m, 20 Hz, 48 range steps. This is a pure forward calculation with no data in it.

| Range (m) | Receiver z (m) | Water depth (m) | Seabed | `n_modes_start` | `n_modes_end` | `n_stripped` | `excitation_rms` | TL coherent (dB) | TL incoherent (dB) | TL geometric (dB) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 100 | 120 | c1500 | 1 | 0 | 1 | 0.0207 | **inf** | **inf** | 36.99 |
| 5,000 | 100 | 120 | c1550 | 2 | 1 | 1 | 0.0345 | 91.00 | 91.00 | 36.99 |
| 5,000 | 100 | 120 | c1700 | 3 | 2 | 1 | 0.0620 | 87.97 | 85.49 | 36.99 |
| 5,000 | 100 | 120 | c1900 | 3 | 2 | 1 | 0.0665 | 80.59 | 83.05 | 36.99 |
| 5,000 | 100 | 120 | c2100 | 4 | 2 | 2 | 0.0753 | 80.29 | 82.83 | 36.99 |
| 10,000 | 150 | 200 | c1500 | 3 | 1 | 2 | 0.0109 | 115.01 | 115.01 | 40.01 |
| 10,000 | 150 | 200 | c1550 | 5 | 2 | 3 | 0.0213 | 111.50 | 110.66 | 40.01 |
| 10,000 | 150 | 200 | c1700 | 8 | 3 | 5 | 0.0338 | 109.32 | 103.12 | 40.01 |
| 10,000 | 150 | 200 | c1900 | 10 | 3 | 7 | 0.0397 | 113.71 | 103.65 | 40.01 |
| 10,000 | 150 | 200 | c2100 | 11 | 4 | 7 | 0.0418 | 94.67 | 95.20 | 40.01 |
| 20,000 | 500 | 600 | c1500 | 5 | 3 | 2 | 0.0066 | 124.48 | 121.68 | 43.03 |
| 20,000 | 500 | 600 | c1550 | 10 | 5 | 5 | 0.0149 | 110.43 | 112.84 | 43.03 |
| 20,000 | 500 | 600 | c1700 | 16 | 8 | 8 | 0.0238 | 102.33 | 102.56 | 43.03 |
| 20,000 | 500 | 600 | c1900 | 20 | 10 | 10 | 0.0280 | 101.45 | 101.57 | 43.03 |
| 20,000 | 500 | 600 | c2100 | 23 | 11 | 12 | 0.0300 | 110.15 | 100.33 | 43.03 |
| 30,000 | 1,000 | 1,100 | c1500 | 6 | 5 | 1 | 0.0068 | 117.47 | 117.06 | 44.80 |
| 30,000 | 1,000 | 1,100 | c1550 | 11 | 9 | 2 | 0.0138 | 108.52 | 110.45 | 44.80 |
| 30,000 | 1,000 | 1,100 | c1700 | 19 | 15 | 4 | 0.0230 | 102.87 | 100.85 | 44.80 |
| 30,000 | 1,000 | 1,100 | c1900 | 24 | 19 | 5 | 0.0269 | 104.17 | 99.63 | 44.80 |
| 30,000 | 1,000 | 1,100 | c2100 | 27 | 21 | 6 | 0.0283 | 101.84 | 98.38 | 44.80 |
| 40,000 | 1,575 | 1,600 | c1500 | 7 | 7 | 0 | 0.0058 | 112.66 | 115.93 | 46.06 |
| 40,000 | 1,575 | 1,600 | c1550 | 13 | 13 | 0 | 0.0120 | 104.97 | 106.96 | 46.06 |
| 40,000 | 1,575 | 1,600 | c1700 | 21 | 21 | 0 | 0.0194 | 112.52 | 101.10 | 46.06 |
| 40,000 | 1,575 | 1,600 | c1900 | 27 | 27 | 0 | 0.0235 | 99.15 | 99.43 | 46.06 |
| 40,000 | 1,575 | 1,600 | c2100 | 31 | 31 | 0 | 0.0254 | 101.40 | 99.02 | 46.06 |

**`path_ok` is `True` for all 25 rows.** `n_unresolved_steps` is 0 everywhere except two rows
(5 km / c1550 and 10 km / c1900, one step each), i.e. the adiabatic stepping was essentially
clean.

Three things this table says, and one it does not.

1. **Modes exist and are stripped, heavily, at the shallow end.** At 20 km over a 1,200→600 m path
   the mode count falls from 16 to 8 (c1700) or 23 to 11 (c2100) — roughly half the guide's modes
   are gone by the receiver. At 40 km over a 1,600 m path nothing is stripped at all
   (`n_stripped` = 0 at all five seabeds). Stripping is a shallow-water phenomenon here, as
   expected.
2. **Excitation is weak everywhere.** `excitation_rms` never exceeds **0.0753** (5 km, c2100) and
   falls to **0.0058** (40 km, c1500). A 20 Hz source at 30 m couples poorly into these guides at
   every geometry and every seabed in the sweep. This is the "the modes survive but the whale does
   not drive them" outcome the spec's §5.3 insisted be reported separately from "the modes are not
   there", and it is the one that occurs.
3. **The answer changes sign at exactly one place in the sweep.** At the shallowest geometry
   (5 km, 120 m water) with the softest seabed (`seabed_c1500`), `n_modes_start` = 1 and that one
   mode is stripped: `n_modes_end` = 0 and transmission loss is **`inf`** — no arrival at all.
   Every other one of the 25 configurations gives a finite TL. So "does a trapped mode survive"
   is **yes at 24 of 25 configurations and no at 1 of 25**, and the single failure is at the
   shallow, soft-seabed corner. The sweep is uncertainty, not knowledge, so the honest reading is
   that the shallow end of the geometry is where the modal picture is fragile to a seabed we do
   not know.

**What this table does not say.** The modal and geometric columns are **not on a common reference
scale**: the modal columns carry real source excitation and receiver mode-shape factors, while
`tl_geometric_db` is a bare spreading law against an implicit unit source at 1 m with none of that
structure. The 50–70 dB gap between the columns is **not excess loss** and **not** evidence about
propagation. Do not subtract them. (This warning is in the table's own header; it is repeated here
because a reader looking at two adjacent dB columns will otherwise subtract them.)

Also, `range_m` and the water depths co-vary by construction — the geometries trace a realistic
shelf-to-abyssal path — so this is not a TL-versus-range curve at fixed geometry, and grouping by
range alone would conflate a range trend with a depth trend.

The coherent/incoherent pair is carried for the reason §5.3 of the spec gave: coherent
interference nulls are sharper than this geometry's uncertainty justifies. They differ by up to
**11.4 dB** (40 km, c1700: 112.52 vs 101.10), and **no conclusion in this document rests on a
null's position**.

---

## 4. Result 2 — observed detection ranges, and the asymmetry that limits what they can prove

From `propagation_max_range.csv`, 381 rows — one per eligible call, the 381 of §2.1, with the 86
basin-flipping calls excluded and not merged. Those 381 calls carry the 8,660 eligible picks.

| Statistic | Value (n = 381 eligible calls) |
|---|---:|
| Median greatest source–receiver range at which a pick exists | **20,830 m** |
| 25th / 75th percentile | 10,952 m / 30,333 m |
| Minimum / maximum | 2,059 m / 67,404 m |
| Calls detected beyond 5 km | 366 / 381 (**96.1%**) |
| Calls detected beyond 25 km | 164 / 381 (**43.0%**) |
| Calls detected beyond 30 km | 99 / 381 (**26.0%**) |
| Calls detected beyond 40 km | 11 / 381 (**2.9%**) |

**The asymmetry, restated in full rather than cross-referenced** (spec §6.3). Maximum observed
range is bounded by the cable's own extent and by which channels an annotator examined, not by
physics alone. This statistic can therefore **falsify a hypothesis that predicts detection ranges
shorter than what was observed**; it **cannot** distinguish two hypotheses that both predict
detection out beyond the cable. A large observed max range confirms neither hypothesis. It can
only rule one out.

It does rule one out, in the only direction it can. §5.2 below shows H-modal returning
**+inf transmission loss — i.e. predicting no detection — at 3,051–3,311 of the 8,660 eligible
picks**, all of which are picks that exist. That is exactly the falsifying direction this
statistic is capable of detecting: the hypothesis says those observations should not have
happened, and they did.

**Two qualifications on that falsification, at the point it is made rather than later.**

**(a) The mechanism behind every +inf is the one §7.1 identifies as an artefact risk.** A +inf
arises when no mode is trapped where the path starts or when **every mode has been stripped**
before the receiver. Mode stripping is exactly what the adiabatic approximation over-produces on a
steep margin: adiabatic propagation forbids energy moving between modes, so a mode that would in
reality couple into another and stay in the guide is instead counted as lost forever and never
returns. The Oregon margin is steep. So a real part of this 3,051–3,311 may be an artefact of the
approximation rather than a claim the trapped picture genuinely makes, and the falsification is
correspondingly weaker than its size suggests. It is not empty — a coupled-mode treatment would
have to recover a great deal of energy to overturn it, and the observed detections are real
either way — but a reader must not read "35.2–38.2% of the 8,660 eligible picks unexplained" as
that share of the trapped hypothesis refuted. What is refuted is *the adiabatic form of it*, which is the only form tested
(§7.1).

**(b) This substitutes a better statistic than the spec asked for, and the substitution is
flagged.** Spec §6.3 specified comparing the observed maximum range against "each hypothesis's
predicted detection range under the sweep". What is reported instead is a per-pick count of
non-predictions. That is strictly more informative — it uses every pick rather than one
per call, and it needs no threshold on what counts as a "predicted detection range" — but it is
not the statistic written down, and unlike the §6.4 hypothesis test, §6.3 carries no
pre-registration that would forbid the change. It is named here so the difference is visible
rather than assumed away.

---

## 5. Result 3 — the fit

### 5.1 The descriptive fit is *not* the pre-registered statistic

`propagation_decay_fit.csv` fits **free** slopes for range, receiver depth and axial projection,
with a free intercept per call. Its own header states, and this document repeats, that
**it is descriptive and is not the hypothesis test**. Spec §6.4 fixed the comparison in advance as
a difference in fitted log-likelihood between two hypotheses offered as *fixed* shapes, and
forbade substituting another statistic after seeing the data. Eyeballing a free slope against a
physics benchmark is precisely the substitution that was forbidden. **The verdict in §6 is built
on `propagation_hypothesis_test.csv`, not on this table.**

With that stated, the descriptive fit at the reference configuration — threshold 0, zero-SNR rule
`censored`, projection term `fitted`, n = **8,660 picks over 381 calls**, 16 censored, 200
whole-call bootstrap draws — is:

| Term | Estimate | 95% whole-call bootstrap interval |
|---|---:|---|
| `log10_range` | **−6.00** dB per decade | [−6.49, −5.51] |
| `receiver_depth_km` | **−0.78** dB per km | [−2.10, +0.73] |
| `axial_projection_db` | **+0.167** | [+0.058, +0.303] |

The three are reported together and none is quoted alone; range and receiver depth are correlated
on a shelf-to-abyssal cable and only the joint fit separates them.

The **depth coefficient spans zero** (−2.10 to +0.73 dB/km, n = 8,660 picks over 381 calls). On
this observable, received level carries no resolvable information about receiver depth once range
and axial projection are accounted for.

The projection term is not optional. Pooled across all picks, `corr(10·log10(axial_response),
log10(range_m))` is only −0.09 — but the fit has per-call intercepts, so only *within-call*
variation identifies the range slope, and the within-call correlation between projection and range
is much stronger (mean 0.36, median 0.45; 78% of the 381 eligible calls have |corr| > 0.3). Within
a call, longer-range picks are systematically more endfire and therefore sensed better. Omitting
the term biases the fitted decay shallow, and measurably does: at the same configuration, the
`omitted` variant gives `log10_range` = **−5.72** [−6.21, −5.30] against **−6.00** with the term
fitted — a **0.28 dB per decade** shallowing, in the predicted direction.

`propagation_decay_fit_draws.csv` carries the joint per-draw coefficients (8,000 rows: one per
threshold × zero-SNR rule × projection setting × draw × term) that this summary reduces to
independent per-term percentiles. Any linear combination of terms must be formed *within* each
draw before percentiles are taken across draws; the summary interval above cannot be combined
across rows.

### 5.2 The pre-registered hypothesis test

`propagation_hypothesis_test.csv`, 5 rows. Each hypothesis's predicted `(range, depth)` shape is
pinned as a **fixed** offset in the same per-call-intercept censored fit; the statistic is
Δ log-likelihood (H-modal minus H-geometric), with a 200-draw **whole-call** bootstrap interval.
Threshold 0.0, zero-SNR rule `censored`.

| Seabed | Δ log-lik (H-modal − H-geometric) | 95% interval | `n_path_refused` | `n_tl_nonfinite_standalone` | `n_predicted_standalone` (of 8,660) |
|---|---:|---|---:|---:|---:|
| `seabed_c1500` | **−1,228.1** | [−1,484.7, −1,045.1] | 0 | 3,311 | 5,349 |
| `seabed_c1550` | **−1,101.4** | [−1,348.7, −874.4] | 0 | 3,051 | 5,609 |
| `seabed_c1700` | **−1,018.0** | [−1,231.8, −838.3] | 0 | 3,051 | 5,609 |
| `seabed_c1900` | **−1,026.6** | [−1,225.5, −843.9] | 0 | 3,051 | 5,609 |
| `seabed_c2100` | **−972.3** | [−1,174.7, −795.3] | 0 | 3,051 | 5,609 |

All five Δ are **negative**, i.e. H-geometric's fixed shape fits better than H-modal's at every
seabed in the sweep. `sign_consistent_across_sweep` is `True`. Every interval excludes zero by a
wide margin: the least negative upper bound anywhere in the sweep is −795.3.

The absolute log-likelihoods, from `propagation_hypothesis_summary.csv` on the same
5,349-pick / 380-call sample: H-geometric **−14,720.1**; the five H-modal seabeds −15,948.2,
−15,821.5, −15,738.1, −15,746.7, −15,692.4. H-geometric beats the best H-modal seabed.

**Two conditions on this number, stated here and not buried.**

**(a) The Δ is conditional on the subset where both hypotheses are defined, and that conditioning
is not neutral.** H-modal produces no prediction at all for **3,051–3,311 of the 8,660 eligible
picks** (35.2% at four seabeds, 38.2% at `seabed_c1500`) — and `n_path_refused` is **0 at every
seabed**, so *not one* of those exclusions is a bathymetry-path failure. Every one is an adiabatic
solve returning **+inf transmission loss**: no trapped mode where the path starts, or every mode
stripped before the receiver. H-modal is not missing data on those picks. It is asserting that
those calls should not have been detected — and they were detected; they are picks in the
annotation snapshot. The reported Δ is computed on the shared six-way intersection of
**5,349 picks over 380 calls** (`predicted_fraction` = 0.618), with H-geometric fitted on the
*identical* sample by construction, so the Δ never compares two different samples. But the picks
H-modal cannot reach are not neutral missing data, and this table must not be read as an
unconditional comparison over all 8,660 eligible picks. Note also that the intersection costs one
call: 380 calls here against 381 in §5.1.

**(b) The Δ settles which *shape* is less inconsistent under an *assumed* dB scale — and whether
that is the physics' dB scale is exactly what §7.7 puts in question.** Both facts hold at once.
The comparison is internally consistent, decisive and sign-stable across the whole sweep; and its
physical interpretation rests on a scale for the pick SNR that this experiment's own
axial-projection coefficient calls into doubt. Neither sentence cancels the other, and §6 states
them together.

### 5.3 How much the censoring correction moved the estimate — and why it is the wrong correction

The threshold is **swept, not known**: the annotation snapshot records no detection threshold, and
observed pick SNR tails smoothly to zero with no cliff. Over the 8,660 eligible picks, the
censored count is 16 (**0.18%**) at threshold 0, 93 (**1.1%**) at 4, 241 (**2.8%**) at 6 and 664
(**7.7%**) at 8. (Over all 11,117 picks in the snapshot the corresponding counts are 31 [0.3%],
172 [1.5%], 384 [3.5%] and 1,015 [9.1%].) The fixed-threshold model is therefore an approximation
to whatever the annotators actually did, not a documented sensor property.

The range coefficient across that sweep, at zero-SNR rule `censored` with the projection term
fitted, n = 8,660 picks over 381 calls throughout:

| Threshold | `log10_range` | 95% interval | n censored (of 8,660) |
|---:|---:|---|---:|
| 0 | −6.00 | [−6.49, −5.51] | 16 |
| 4 | −5.98 | [−6.45, −5.51] | 93 |
| 6 | −5.94 | [−6.39, −5.49] | 241 |
| 8 | −5.79 | [−6.55, −5.39] | 664 |

**The correction moves the estimate very little.** Across the whole threshold sweep the range
coefficient moves **0.21 dB per decade**, well inside a single configuration's own interval width.
The zero-SNR rule moves it less still: at threshold 0, `censored` gives −6.002 over **8,660 picks
in 381 calls** and `dropped` gives −5.989 over **8,644 picks in 381 calls** — the two fits have
different denominators, because the `dropped` rule removes the 16 exactly-zero-SNR picks that fall
inside the eligible sample before fitting, and the difference between them is **0.013**. (The
snapshot holds 31 exactly-zero-SNR picks in all; the other 15 belong to calls already excluded
upstream by §2.1's denominators.) Both rules are reported in the table so the rule's influence is
visible rather than asserted away. The correction is small on this data, and saying so is more
useful than implying it rescued anything.

**But it is not the correction the spec's problem needs.** The implemented likelihood is
left-**censored** (Type-I Tobit): a row is present with a value known only as a bound. The
sampling problem the spec's §6.4 describes is left-**truncation**: sub-threshold picks were never
recorded at all, so they are *absent rows*, not present rows with bounds. Those are different
corrections for different problems. Treating present low-SNR picks as bounded observations tells
the estimator both how many such picks exist and at what range and depth — information a genuinely
truncated sample does not contain. So the threshold sweep above measures robustness to
**discarding low-SNR information**, not sensitivity to a detection threshold in the truncation
sense. The correct estimator for a truncated sample is a truncated-normal likelihood (subtracting
`log(1 − Φ((threshold − μ)/σ))` per **retained** observation, with no bounded-observation term at
all). **It is not implemented.** The numerical consequence here is small — the sweep above shows
how small — but the implemented correction does not address the stated selection bias, and a
reader must not believe it does. Recorded as a follow-up in §8, not as work done.

---

## 6. Verdict

The spec's §7 criteria, quoted as written, before any of the above existed:

> - **Direct path.** H-geometric reproduces the observed range and depth shapes within their
>   uncertainty, and H-modal does not, across the whole seabed sweep.
> - **Bottom-interacting / trapped.** H-modal reproduces them and H-geometric does not, across the
>   whole sweep, **and** Result 2 does not contradict it.
> - **Undecidable.** Either both reproduce the observed shapes within uncertainty, or the answer
>   changes sign across the seabed sweep, or the fitted uncertainty on the discriminating
>   coefficient spans both predictions.

### 6.1 The branch: **direct path**

Taking the "undecidable" branch's three triggers in the order the spec lists them, against the
pre-registered statistic:

1. **Do both reproduce the shapes within uncertainty?** No. Δ log-likelihood runs −972 to −1,228
   across the sweep with intervals whose least negative bound is −795.3. The two hypotheses are
   separated by roughly a thousand log-likelihood units on 5,349 picks over 380 calls. This is not
   a near tie.
2. **Does the answer change sign across the seabed sweep?** No. All five Δ are negative;
   `sign_consistent_across_sweep` is `True`. The magnitude varies by about a quarter across the
   sweep (−972 to −1,228) but the ordering never moves.
   **What "the answer" is the sign of, stated because this document uses the phrase twice with
   different referents.** Here it is the sign of **Δ log-likelihood** — the pre-registered
   statistic, which spec §6.4 fixes as *the* comparison and which §7's third trigger calls "the
   discriminating coefficient". That sign is stable across all five seabeds. §3's item 3 uses the
   same phrase about something else entirely: whether Result 1's *forward calculation* yields a
   surviving mode at all, which does flip at one of its 25 configurations (5 km, softest seabed).
   Result 1 is a no-data forward calculation and is not the discriminating statistic, so its flip
   does not fire this trigger. A reader checking the triggers should read only the Δ here.
3. **Does the fitted uncertainty on the discriminating coefficient span both predictions?** No.
   Every one of the five whole-call bootstrap intervals lies wholly on the H-geometric side of
   zero.

None of the three triggers fires, so "undecidable" is not the branch — and it was defined in
advance precisely so that it could not be reached for now. The bottom-interacting branch requires
H-modal to reproduce the shapes and H-geometric not to; the evidence is the exact reverse at every
seabed, and Result 2 points the same way (§4). **The evidence lands in the direct-path branch on
that branch's comparative clause: across the whole seabed sweep, the observed range-and-depth
shape of pick SNR is less inconsistent with a direct water path than with a trapped /
bottom-interacting one, and H-modal additionally predicts non-detection for 3,051–3,311 of 8,660
eligible picks that were in fact detected — a falsification whose mechanism is mode stripping,
which §4(a) and §7.1 note the adiabatic approximation over-produces on a steep margin. The
branch's absolute clause is untested — see immediately below.**

**The direct-path criterion has two clauses, and only one of them was tested.** As the spec writes
it, that branch requires that "H-geometric **reproduces** the observed range and depth shapes
**within their uncertainty**" *and* that H-modal does not. Both halves are absolute
goodness-of-fit claims. The pre-registered statistic is purely **comparative**: a difference in
fitted log-likelihood establishes that H-geometric fits *better* than H-modal, decisively and
sign-stably, and nothing more. Two hypotheses can be separated by a thousand log-likelihood units
while both fit badly, and nothing above rules that out.

**The pre-registered branches are not exhaustive over the evidence state that actually occurred,
and that is said here rather than papered over.** Direct-path's antecedent is unmet, because its
absolute clause was never tested. Undecidable's three-way disjunction is unmet, as trigger-by-
trigger above. Bottom-interacting is refuted outright. The state that occurred — *a decisive,
sign-stable ranking with neither hypothesis shown to fit in absolute terms* — has no branch
written for it, because §7 was drafted on the assumption that a comparison this decisive would
also be an absolute one. This document therefore **reports the ranking, and names the clause that
went untested, rather than forcing the evidence into a branch that does not fit it.** Naming the
direct-path branch is the closest true statement available, not a claim that its stated conditions
were all met.

**The absolute clause is untested, not met.** It could not be tested on this data, and the reason
is a property of the data rather than an omission of effort: an absolute goodness-of-fit claim
about a dB-scale observable requires knowing what that observable's dB scale *is*, and §6.2 and
§7.7 show that it is not established — the one physics-anchored calibrator available fails by an
order of magnitude. A residual scatter cannot be read as "within uncertainty" when the units of
both the residual and the uncertainty are in question. So the honest form of the ruling is: the
comparative half of the direct-path branch is satisfied, emphatically and across the whole sweep;
the absolute half is open, and no reader should take this document as having shown that
H-geometric fits the observed shapes in absolute terms. It shows that H-modal fits them worse.
This is carried into the limitations as §7.9, separately from the scale limitation, because the
two say different things.

Result 1 supplies the mechanism without circularity, since it contains no data: the excitation of
a 20 Hz source at 30 m into these guides never exceeds `excitation_rms` 0.0753 anywhere in the
25-configuration sweep, and roughly half a shallow guide's modes are stripped by 20 km. A trapped
path is available in the sense Plan A established — the modes exist — but on these numbers it is
barely driven and heavily stripped, which is precisely the configuration in which it does not win
a comparison of arrivals.

### 6.2 What the ruling is, and what it is not

**It is a ruling on shape under an assumed scale.** The Δ compares two fixed shapes against an
observable whose dB scale this experiment does not establish. The one physics-anchored calibrator
in the whole dataset — the axial-projection coefficient, whose benchmark is **2** on purely
geometric grounds — comes back at **+0.15 to +0.17** with an interval excluding both 1 and 2 at
every configuration (§7.7). That is a known, purely geometric amplitude change registering at a
small fraction of its expected rate. A decisive, sign-stable Δ under a scale that fails its own
calibrator settles which shape the data are **less inconsistent with**; it does **not** settle the
underlying propagation physics, and this document does not claim it does.

**And that possible miscalibration has a direction — toward the hypothesis that won.** This is the
most uncomfortable sentence in the document and it belongs here, where the ruling's scope is set,
rather than only in §7.7 where the mechanism is explained. In the pre-registered test each
hypothesis's shape is pinned as a **fixed** offset, i.e. with coefficient exactly 1. So if the
observable is a *compressed* version of the physical dB scale — `observed_dB = k · true_dB` with
`k < 1`, which is what the calibrator's own reading implies, since it puts the projection response
at roughly a twelfth of its geometric benchmark — then a genuinely steep world is flattened into
something that looks more like the *shallower* of the two fixed shapes. H-geometric is the
shallower shape. **A compression of this kind would therefore bias Δ systematically toward
H-geometric: the hypothesis this document declares the winner.** The ruling's direction is the
same direction the bias would produce, and nothing in this dataset separates the two.

Stated with its own limits, because overstating it would be its own error: this is a **possible**
bias of **unknown magnitude**, not a measured one. No value of `k` is estimated anywhere here; the
twelvefold figure is an order-of-magnitude reading off a single coefficient, not a calibration.
And the two competing explanations for that coefficient (§7.7) remain live and **would not** have
this effect — denoiser-flattened directivity would shrink the projection coefficient while leaving
the range slope, and so the dB scale, untouched; functional-form misspecification in the range
term is a different failure again. So the honest position is: an identified mechanism that would
favour the declared verdict exists, is unquantified, and cannot be ruled out by anything measured
here. A reader should weight the ruling accordingly, and §8's first follow-on is the test that
would settle it.

**And the exclusion is not neutral.** The Δ is conditional on the 5,349-pick / 380-call
intersection where both hypotheses are defined. That intersection excludes **3,311 of the 8,660
eligible picks at every seabed** — the six-way intersection is shared, so the excluded set is the
same everywhere, not the per-seabed count. The attribution differs by seabed and the precise form
matters: at `seabed_c1500` all 3,311 are picks that seabed itself could not predict. At the other
four, that seabed's own non-predictions number 3,051, and the remaining **260** are picks it *did*
predict which were dropped only because `seabed_c1500` failed on them. The generosity argument
survives that correction intact, because **H-geometric has zero non-finite predictions** (it
predicts all 8,660): every single excluded pick traces to *some* H-modal seabed failing, never to
H-geometric. Excluding them therefore removes H-modal failures — its own at c1500, the sweep's at
the others — from the statistic that judges H-modal, so the reported Δ, decisive as it is, is if
anything *generous* to H-modal rather than prejudicial against it. The conditioning does not
manufacture the result; it understates it. It remains true that a comparison conditioned on 61.8%
of the eligible picks is not the unconditional comparison a reader might assume, and §7 carries
that.

So the answer to the second half of §1's question — *can this dataset distinguish them at all?* —
is: **yes on the ranking, no on the underlying physics without a calibration.** The dataset has
ample power to rank the two shapes and does so unambiguously. It does not have the calibration to
convert that ranking into a statement about the true received level, nor — for the same reason —
into an absolute statement that the surviving hypothesis fits (§7.9). Nothing in this plan
supplies either.

### 6.3 Consequence for Plan A's spec §7.4 (depth observability)

Plan A's findings §5 stated that Plan B needs a new justification and a new null before §7.4 is
interpretable. This experiment supplies part of both, and withholds part.

- **Interpretable, with a caveat.** The straight-ray `ConstantC` family is not modelling the wrong
  *path* in the specific sense that was feared: the trapped alternative fits the observed amplitude
  shape worse, sign-stably across the seabed sweep, and predicts non-detection for calls that were
  detected. The concern that motivated Q-path — that a bottom-interacting first arrival would
  invalidate 2-D as well as 3-D inference — is not supported by this data. **§7.4 may proceed on
  straight-ray geometry on that basis, but the warrant is a comparative one.** Nothing here shows
  that straight-ray propagation reproduces the observed shapes in absolute terms (§6.1, §7.9); it
  shows that the trapped alternative does worse. §7.4 should state that as its footing rather than
  claim the propagation model has been validated.
- **The null.** The null for §7.4 is H-geometric straight-ray propagation, and it is now
  comparatively evidence-backed rather than assumed — conditional on the 5,349-pick subset, on the
  assumed dB scale of §6.2, and on the untested absolute clause of §7.9. That is a weaker warrant
  than "measured and calibrated", and §7.4 should carry it as such rather than citing this document
  as having closed the question outright.
- **What is withheld: amplitude cannot supply the depth signal.** The descriptive fit's
  receiver-depth coefficient is −0.78 dB/km with a 95% interval of [−2.10, +0.73] over 8,660 picks
  in 381 calls — it spans zero. Received level carries no resolvable depth information on this
  observable once range and axial projection are accounted for. If §7.4 is to make source depth
  observable it must do so through **travel-time** moveout, not through amplitude. Plan A's item 2
  (the §7.0.2 moveout-shape test) is therefore still the right next step, and is now the *only*
  remaining route to depth that this programme has not closed off.

---

## 7. Limitations

All six of the spec's §9, in this document's own words, plus three this work discovered (§7.7–7.9).

**7.1 Adiabatic propagation assumes gentle slopes; the Oregon margin is not gentle.** The
adiabatic approximation forbids energy moving between modes, which on a steep shelf break is
false. This bounds mode coupling rather than resolving it: H-modal as implemented cannot represent
a mode converting into another at a slope rather than being stripped. A coupled-mode or
parabolic-equation treatment would be required and is out of scope. The verdict inherits this:
what was tested is an adiabatic modal hypothesis, not every bottom-interacting hypothesis. This
bites hardest on the +inf non-predictions of §4 and §5.2, every one of which arises from no
trapped mode at the start or from **every mode stripped** — the outcome this approximation
manufactures most readily on a steep margin. §4(a) states that where the falsification is
asserted, and it is repeated here so a reader arriving at the limitations from either direction
meets it.

**7.2 Per-channel fibre coupling varies, and it may vary *with range*.** How well each channel's
fibre is mechanically coupled to the seafloor is unknown and not measured. If coupling correlates
with burial depth or seafloor type, it correlates with range along a cable that runs shelf to
abyssal — which makes it a **bias in the range slope, not merely scatter around it**. Nothing in
this design removes it, and no measurement here bounds it. It is a standing reason a shape
comparison on this observable could be wrong in a direction no interval above covers.

**7.3 The denoised SNR has no calibration.** Only shape is used; no absolute level is quoted
anywhere in this document. Any result quoting one would be wrong by construction. §7.7 is why this
defence is not as strong as it sounds.

**7.4 One profile product, never validated in situ at its own location, and clamped below its own
bottom on a material fraction of paths.** Everything modal here rests on `hycom_south_deep`, whose
credibility is inferred from a different HYCOM extraction agreeing with in-situ CTD to a median
0.47 m/s at a site 87 km away and shallower. Plan A's spec §6.4 bar of two independent products
(WOA23) remains unmet, exactly as it was in Plan A. Every profile-dependent number above inherits
this.

The profile itself covers 0–1,500 m. 98.4% of *picks* have a receiver shallower than 1,500 m --
but that is the wrong denominator for a normal-mode solve, which integrates over the WHOLE water
column along the path, not just the receiver's own depth. `modes()` does not extrapolate the
profile past 1,500 m; it CLAMPS `c` to isovelocity below the profile's bottom, and emits a
`UserWarning` when it does, specifically so a caller cannot pass a shallow profile under a deeper
water column and read the mode count as physics (see `soundspeed/modes.py`'s own comment). The
committed bathymetry grid reaches 2,484 m, and 16.3% of its water CELLS are deeper than 1,500 m.
Measured directly on this experiment's real per-pick paths (`compute_pick_paths`, the same
bathymetry `modal_shape_for_seabed` solves against): 272 of 8,660 picks (3.1%) have a path whose
maximum depth exceeds 1,500 m at some point between source and receiver, meaning the mode solve
along that path ran at least partly on a clamped, isovelocity-extended profile rather than the
profile HYCOM actually supplies. This PATH-based exposure (3.1% of picks) is the right denominator
for this limitation -- it is roughly double the receiver-depth figure (1.6%) because a path can
pass through deep water even when it ends at a shallow receiver. It is an untested assumption that
every H-modal prediction on those paths inherits, including the non-predictions (+inf TL) that are
load-bearing for §4's falsification of H-modal. The per-pick count of clamped solves is not
currently surfaced as its own column in `propagation_hypothesis_test.csv` -- see that table's
header note for why (recovering it exactly requires re-running the ~90-minute per-seabed modal
shape stage, which was not done for this fix; the 3.1% path-exposure figure above is the
recoverable, cheaper proxy).

**7.5 The gauge-length response is excluded.** At 20 Hz the acoustic wavelength is ~74 m at
1,480 m/s, against a gauge length of order 10 m, so the gauge-length transfer function is a
second-order term — but it is genuinely omitted, not shown to be negligible on this instrument,
and this is said here rather than leaving a reader to assume it was modelled.

**7.6 Positions are an input, not a result.** Restated from §1 because it constrains how the
verdict may be read: the ranges every fit above regresses on come from `ConstantC` solutions, so
no result here independently tests those positions, and the data comparison must not be mistaken
for validation of the localizer.

### 7.7 New: the observable fails its own physics-anchored calibrator, and an intercept does not absorb the failure

This is the most consequential thing found, and it is not in any table's numeric body — it is a
comparison between a fitted coefficient and a benchmark that physics fixes.

The axial-projection regressor enters as `10·log10(axial_response)`, where `axial_response` is
`cos²θ` with `θ` the angle between the horizontal propagation direction and the local cable
tangent. That `cos²θ` is an **amplitude** factor for axial strain: one cosine projects particle
motion onto the fibre, the second comes from the along-fibre spatial gradient. Under either dB
convention — `10·log10` of a power quantity or `20·log10` of an amplitude — the coefficient on
this regressor should therefore be **2**. It is fitted at **+0.15 to +0.17** across every
configuration of the threshold × zero-SNR-rule grid, with bootstrap intervals (e.g.
[+0.058, +0.303] at the reference configuration, n = 8,660 picks over 381 calls) that **exclude
both 1 and 2 everywhere**.

This is the only calibrator in the dataset anchored to physics rather than to a model, and it says
the observable registers a known, purely geometric amplitude change at a fraction of the expected
rate.

**Why this is worse than the uncalibrated-level caveat the spec already carried (§7.3).** The
spec's §9.3 defence is that only *shape* is used, and per-call free intercepts absorb any unknown
per-call gain. That works for an **additive** offset on the dB scale — an unknown source level, a
per-call denoiser gain. It does **not** work for a **multiplicative** gain on the dB scale. If the
observable's dB values are some factor `g` times the physical dB values, an intercept cannot
absorb `g`; `g` multiplies every slope in the regression, the range slope included. So the
suspicion this coefficient raises does not stay confined to the level. It propagates into the
range coefficient, into the depth coefficient, and into the scale on which the pre-registered Δ
log-likelihood is evaluated. That is why §6.2 is worded as it is.

**Which way it propagates, stated rather than left symmetric.** In the pre-registered test each
hypothesis's shape enters as a **fixed** offset — coefficient exactly 1 — so the test asks which
fixed shape the observed dB values match, and it is the *scale* of those values that decides.
Under a compression `observed_dB = k · true_dB` with `k < 1` — the direction the calibrator's own
reading implies, since a coefficient of 0.15–0.17 against a benchmark of 2 is a compression, not
an expansion — a steep underlying decay is flattened, and a flattened observation resembles the
**shallower** of the two candidate shapes. H-geometric is the shallower shape. So this
miscalibration, if it is real, **biases Δ toward H-geometric, the hypothesis that won**. An
identified bias pointing at the declared verdict is not a symmetric "the scale is uncertain"
caveat and is not reported as one; §6.2 states it at the ruling.

Its limits, equally plainly: the magnitude is **unknown**. No `k` is estimated anywhere in this
work, and reading one off a single coefficient would be exactly the kind of substitution §5.1
refuses. Moreover, the first competing explanation below would produce the small coefficient
**without** compressing the dB scale at all, in which case this bias does not exist. Both
possibilities are open.

**Competing explanations that were considered and are *not* ruled out**, stated so this is not
read as a settled diagnosis:

- **The denoiser may have destroyed the directivity.** Across-channel smoothing would fill in the
  broadside null in `cos²θ`, flattening the measured projection response. Crucially, that
  mechanism would shrink this coefficient **without touching the range slope**, in which case the
  multiplicative-gain worry above does not apply and only the projection term is compromised.
  Nothing measured here distinguishes this from a true compression of the whole dB scale.
- **Functional-form misspecification.** The fit pools ranges spanning spherical spreading,
  cylindrical spreading and absorption-dominated regimes under a single log-linear range term,
  while the projection column is itself a curved function of the same geometry. A misspecified
  range term can leak into a correlated curved regressor.

**One explanation *was* ruled out quantitatively: errors-in-variables.** Classical attenuation
from angular measurement noise would need **35–65 degrees** of angular error to attenuate the
coefficient **sixfold**, against a measured cable-tangent stability of **0.5–0.7 degrees**
**[not table-backed]**. Two orders of magnitude short, so the attenuation is not measurement noise
in the angle.

One bookkeeping correction to that argument, made here rather than quietly: the attenuation the
fitted coefficient actually implies against the corrected benchmark of 2 is **~12-fold**
(2 ÷ 0.15–0.17), not sixfold. The 35–65 degree figure was computed against the sixfold case and
is **not** recomputed here; it is retained as a *lower bound* on the angular error required,
because classical attenuation is monotone in the error variance — a larger attenuation demands a
larger angular error, never a smaller one. So the correct multiple makes the required error
bigger than the quoted range and the conclusion strictly stronger: errors-in-variables is deader
under the corrected multiple than under the one the calculation used. The figure is left as-is and
labelled, rather than silently rescaled, so nobody reads 35–65 degrees as a recomputed answer to
the twelvefold question.

### 7.8 New: the censoring model does not address the selection problem the spec describes

Restated from §5.3 because it belongs in the limitations a reader carries away. The implemented
likelihood is left-**censored**; the sampling problem is left-**truncation**. Sub-threshold picks
are absent rows, not present rows with bounds, so the correction as implemented answers a
different question and, at thresholds above zero, discards real low-SNR measurements rather than
recovering absent ones. The measured censored fraction is small (16 to 664 of the 8,660 eligible
picks, 0.18% to 7.7%, across the threshold sweep) and the estimates move by ~0.01 between
zero-SNR rules, so the numerical consequence on this data is minor. But the stated selection bias
is **not** corrected, and no claim in this document should be read as though it were. The
truncated-normal likelihood is the correct follow-up; it is not implemented.

### 7.9 New: the verdict rests on a comparative statistic; neither hypothesis was shown to fit in absolute terms

This is a different limitation from §7.7 and is stated separately because conflating them would
lose one of them. §7.7 says the *ranking may not translate into physics*, because the observable's
dB scale is in question. §7.9 says that even taking the scale at face value, **no absolute
goodness-of-fit was ever established for either hypothesis**.

The pre-registered statistic (spec §6.4) is a difference in fitted log-likelihood. It is
comparative by construction. It can say that H-geometric's fixed shape is a better description of
the data than H-modal's, and it does so decisively and sign-stably across the seabed sweep. It
cannot say that H-geometric's shape is a *good* description. Two hypotheses can be separated by a
thousand log-likelihood units while both are badly wrong, and this experiment contains no test
that would have detected that case.

The spec's direct-path branch is worded as an absolute claim — H-geometric "reproduces the
observed range and depth shapes within their uncertainty" — so **half of the criterion the verdict
is declared under went untested**, and §6.1 says so at the point of the ruling rather than only
here. The reason it went untested is a property of the data: an absolute fit statement about a
dB-scale observable needs the calibration §7.7 shows is absent, since a residual cannot be
compared to an uncertainty when the units of both are unresolved. No amount of additional analysis
of *these* tables would close it; a calibrated observable would.

Practical consequence: the ruling supports "the trapped/bottom-interacting picture is the worse
description of these arrivals" and does **not** support "direct-path geometric spreading describes
these arrivals". Any downstream work that needs the second claim — anything that treats
H-geometric as a validated forward model rather than as the surviving null — needs evidence this
document does not provide.

---

## 8. What a follow-on should do

1. **Calibrate the observable, or stop treating its dB values as dB.** §7.7 is now the binding
   constraint on everything amplitude-based in this programme. The cheapest discriminating test is
   whether the denoiser flattens directivity: recomputing the projection coefficient on a
   non-denoised strain product, if one can be obtained, separates "the denoiser ate the broadside
   null" from "the whole dB scale is compressed". Those two have very different consequences and
   the current data cannot tell them apart.
2. **Close the absolute half of the criterion, or say permanently that it cannot be closed**
   (§7.9). Once the observable has a scale, an absolute goodness-of-fit test on H-geometric is
   cheap — it reuses the same fitted model — and it is the difference between "the trapped picture
   fits worse" and "direct-path spreading fits". Until then no downstream work may treat
   H-geometric as a validated forward model.
3. **Implement the truncated-normal likelihood** (§7.8) and confirm that the small numerical
   consequence measured here survives the change of estimator.
4. **Run Plan A's item 2, the §7.0.2 moveout-shape test.** §6.3 makes it the only remaining route
   to source-depth observability, since amplitude is now measured to carry no resolvable depth
   information.
5. **Do not spend further effort on modal transmission loss for this question.** It has been asked
   and answered within the limits this data allows.

---

## 9. Reproducing this

Everything runs under `uv run` from the repo root. The driver is
`uv run python -m paper.scripts.run_experiment --config <path>`, one dispatch per config:
`paper/configs/propagation_tl.yaml` (Result 1) and
`paper/configs/propagation_data.yaml` (Results 2 and 3).

**The verification was split, and is reported as split. It was *not* an end-to-end re-run.**

| Table | Regenerable via the driver? | Verification actually performed |
|---|---|---|
| `propagation_tl.csv` | **Yes**, offline | Pure forward calculation from the committed `hycom_south_deep` profile and the committed geometries; no data, no RNG, no network |
| `propagation_denominators.csv` | **Yes**, offline | Deterministic bookkeeping over the committed picks snapshot and geometry; the six categories are asserted in code to sum to `calls_total` |
| `propagation_max_range.csv` | **Yes**, offline | Deterministic reduction of the same eligible frame |
| `propagation_modal_shapes.csv` | **Yes**, offline, ~44 min | **Verified on a 1,000-pick subset, not at full scale.** `modal_shape_for_seabed` was re-run on a real 1,000-pick subset in a **separate process** and compared against the corresponding rows of the full-scale run's persisted shapes: max absolute difference **~1.4e-14 dB**, i.e. *not* bit-identical, reported plainly rather than rounded away. A same-process control (two calls on the identical subset) gave exactly 0.0, so the computation is deterministic for identical inputs within one process. At \|x\| ~ 64 dB, 1.4e-14 dB is **one double-precision ULP** — one last bit, not accumulated error — which bounds any log-likelihood perturbation at ~1e-10, about **thirteen orders of magnitude** below the reported Δ of ~−1,000 and its ±200 interval. The likely cross-process mechanism is buffer alignment selecting a different SIMD kernel, or a different OpenBLAS thread count under different machine load, in the mode eigenvalue solve — not a code defect |
| `propagation_decay_fit.csv`, `propagation_decay_fit_draws.csv` | **Yes**, offline, ~22 min | Stochastic stage (RNG, a convergence stopping criterion, floating-point path dependence — the part where a reproducibility claim could actually fail). **Does not read the persisted modal shapes** -- `decay_fit_table` fits directly against the analysis frame (range, receiver depth, axial projection), so this is a fresh sweep over the full threshold x zero-SNR-rule x projection grid, not a cheap re-derivation from `propagation_modal_shapes.csv`. Re-run and diffed against the committed CSV via `pandas.testing.assert_frame_equal` on the parsed body (`comment='#'`), modulo the `# generated:` timestamp line: **identical** |
| `propagation_hypothesis_test.csv`, `propagation_hypothesis_summary.csv` | **Yes**, offline, ~64 s from persisted shapes | This is the table that actually IS cheap: `_fit_hypothesis_test` reads shapes back from the persisted `propagation_modal_shapes.csv` rather than recomputing them, so only the fit-and-bootstrap stage runs. Same literal byte-for-byte regenerate-and-diff: **identical** |

So: the expensive deterministic stage was verified on a subset to one ULP with its error bounded
analytically; the stochastic stage — the one that could plausibly fail — was verified literally.
Neither is a full 8,660-pick end-to-end re-run, and nothing here should be read as one.

`src/dclde_localization/physics/forward.py` is frozen; it was not modified by any task in this plan and
the working tree is clean against it.

Tests, all three suites measured 2026-09-21: `uv run pytest paper/tests -q` —
**349 passed**. `uv run pytest viz/tests -q` — **41 passed**, which is the check that moving
`bathymetry.py` out of `viz` was behaviour-preserving. `cd viz && npm test` — **49 passed, 0
failed**, the JavaScript side of the visualizer, unaffected by this plan but exercised so the
claim is measured rather than assumed.
