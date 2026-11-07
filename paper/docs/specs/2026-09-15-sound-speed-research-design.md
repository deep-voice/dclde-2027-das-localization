# Sound-speed structure and depth observability — research design

_Written 2026-09-15. Branch `sound-speed-research`, cut from `visualizer`._

> [!WARNING]
> **Plan A (§7.0–7.3) has been executed; report in
> [`paper/docs/das-finwhale-soundspeed-plan-a-findings.md`](../das-finwhale-soundspeed-plan-a-findings.md).**
> Three claims in this spec were measured and are wrong; each is marked **[CORRECTED 2026-09-17]**
> or **[OVERTURNED 2026-09-17]** in place rather than silently edited.
> - **§2.3** — "~9,500 picks per call" (measured median 4,684) and "the baseline's `utc` column is
>   joinable" (it is not).
> - **§2.4** — "97–575 m" is the raw mechanical sweep; the QARTOD-usable band is 216–575 m
>   (5-day window) or 154–575 m (3-week).
> - **§2.5B** — the ray-theoretic shadow zone, and with it §1's "the assumed straight-line path
>   does not exist" amendment. Trapped modes exist in every geometry ray theory calls shadowed.
>
> Plan B (§7.4–7.6) must be re-justified before it is planned: §7.0.1 falsified the premise that
> the residual range trend is a source-depth artifact, and §7.2 disqualified free-`c` as the null
> model §7.4 was to beat.

Design for executing the decision gate in [`paper/docs/sound-speed-feasibility.md`](../sound-speed-feasibility.md):
reproduce the published baseline, then test whether sound-speed structure changes DAS fin-whale
localization — and specifically whether it makes **source depth** observable at all.

## 1. What this is and is not

> [!IMPORTANT]
> **Amended 2026-09-16.** The original framing — "L-1 is settled, only depth is open" — did not
> survive scrutiny. It rested on published evidence that perturbing a *constant* `c` by ±25–50 m/s
> moves localizations only 16–31 m. That tests the value of a constant; it does not test the
> **gradient**, which is different physics: a gradient refracts, and refraction can remove a
> propagation path rather than perturb its timing. Measurement says the smooth part of the gradient
> effect is indeed small (§2.5A), but that the assumed straight-line path **does not exist** for most
> of our geometry (§2.5B), and that real residuals carry range-structured model error far larger than
> the sound-speed correction being chased (§2.5C). A third question is therefore in scope, and it may
> matter more than the original two. See §7.0.

The feasibility note's verdict on the *magnitude* of the smooth correction stands: this is **not an
accuracy-improvement project** in the sense of tuning `c`. A uniform sound-speed error is largely a
range scale factor, and the dominant single-cable geometric error is the left–right ambiguity. Two
questions are open, and the second was not anticipated:

> **Q-depth.** Two seafloor cables at similar depth make source depth nearly unobservable. The one
> available source of vertical aperture is that the cables run shelf-to-abyssal, so channels picking
> the same call sit at very different depths. Does that aperture make `z_s` identifiable — and does
> constant-`c` bias it?
>
> **Q-path.** Is a straight line the right *path*, not merely the right speed? In a downward-refracting
> upper ocean a shallow source has no direct water-borne ray to a distant channel, so the true first
> arrival would be bottom-interacting. If so, constant-`c` is not slightly wrong; it models the wrong
> path, and that affects L-1 as well as L-2.

Every deliverable here answers one of those or supports it. Point-accuracy improvement is not a goal,
and we should not claim it.

### Success criteria

This program succeeds if it produces a **defensible answer, including a negative one**. Specifically:

1. A reproduction of the published baseline good enough that differences from it are attributable.
2. A number, in metres, for how much constant-`c` biases estimated source depth — or a demonstration
   that `z_s` is not identifiable at all, in which case L-2's Z component is unscoreable and we should
   say so publicly.
3. An honest statement of which of those two the evidence supports, with its denominator.

It does **not** succeed by producing a better leaderboard position, and we should not let that
reframe it later.

## 2. Facts established before designing

Everything in this section was measured or fetched on 2026-09-15, not assumed. Several contradict the
feasibility note; see §3.

### 2.1 The depth aperture is real

Per-call span of seafloor depth across the channels that picked each call, from the committed
annotation snapshot (11,117 picks, 628 calls, all five batches):

```
median 219 m    p90 575 m    max 1085 m
span >= 100 m:  501/628 = 80%
span >= 200 m:  348/628 = 55%
span >= 400 m:  164/628 = 26%

two-cable calls (n=166):  median 372 m,  max 1085 m
one-cable calls (n=462):  median 175 m
```

North spans 0 to −589 m, South 0 to −1575 m. The `∂t/∂z_s = (z_s − z_c)/(c·R)` argument therefore has
real leverage to work with. **This is the single assumption the whole L-2 hypothesis rests on, and it
survives.**

### 2.2 Pick residuals under the published model

Solving all 628 calls on a refined grid at `c = 1480`, `z_s = −30` (the baseline's own assumptions)
and taking residuals after analytic `t0` elimination — n = 524 calls with at least 8 picks, 10,517
picks:

```
per-call residual std (s):   p10 0.030   p50 0.096   p90 0.150
pooled |residual| (s):       p50 0.050   p90 0.193   max 43.5
p50 equivalent range error at c=1480:  74 m
```

Two things matter here. The median per-call scatter is consistent with the baseline's own published
`wrms` (median 0.126 s over its 176 rows), so we are in the same regime as the reference — a useful
independent check. And `max |r| = 43.5 s` proves the residual distribution has a **heavy tail of
mis-associations**, not merely measurement noise. A Gaussian noise model would be wrong on its face.

### 2.3 Reference implementation and association output, in hand

- `Ocean-Data-Lab/Goestchel_JASA_2025b` cloned — code (`main_section{2,3,3c,3d,5}.py`), the published
  `batch1_localizations_with_coords.csv` (176 rows), cable geometry, GMRT grid.
- Zenodo `associations.zip` (record 21220346) — 180 HDF5 files, 5 batches × 3 method variants
  (`Baseline`, `FarWin`, `Gabor_Farwin`) × 12 windows, 3.67 GB uncompressed.

**Windows cannot check either out.** 32 of the repo's 45 files and all 180 archive members carry
colons in their names (`..._2021-11-04_02:00:02_...`), which NTFS forbids; `git clone` fetched the
objects and then aborted the checkout wholesale. Both are materialised through a `:`→`_` rename with a
manifest recording every mapping (`_worktree_manifest.csv`). Their scripts hardcode the colon paths as
string literals rather than constructing them, so running their pipeline here is a small patch to
those literals — kept as a tracked diff, never an in-place edit of their source.

Each association file carries, per call:

```
localizations/{north,south}/{hf,lf}/item_N = [x, y, z, wrms]
assoc_pair/{north,south}/{hf,lf}/item_N    = (2, ~9500)  [channel_idx, time_sample_idx]
hyperbolas/{north,south}/{used,rejected}
```

Two consequences. The filename gives the window, so **the baseline's `utc` column is joinable** to
call keys — per-call reproduction is possible, not just distributional comparison. And their
associated pick sets are ~9,500 (channel, time) pairs per call against ~18 in the challenge
annotations: the same calls at two very different sampling densities. Since depth leverage comes from
aperture, that difference is itself worth an experiment (§7.5).

> [!WARNING]
> **[CORRECTED 2026-09-17] — "~9,500 picks per call" is wrong.** Measured directly from
> `associations.zip` (12 batch-1 `Baseline` windows, 194 `assoc_pair` items, every item's last-axis
> shape read via `h5py`): **min 324, median 4,684, mean 5,328, max 12,228** picks per item. Roughly
> half the quoted figure. The qualitative point survives unchanged — hundreds to tens of thousands
> of picks per call against a median of 14 in the annotations (~18 over batch 1) — and §7.5's
> motivation is unaffected. The "~18" figure is batch 1; over all 628 annotated calls the median is
> **14** (min 3, max 61).
>
> Also corrected here: **the baseline's `utc` column is NOT joinable to call keys.** §2.3 says
> per-call reproduction is therefore possible. It is not — see §7.1's outcome and
> `paper/docs/das-finwhale-soundspeed-plan-a-findings.md` §3.1. Call density (one call per ~4 s) defeats
> a timestamp-only join at every tolerance: 0/176 rows join unambiguously at the 90 s default, and
> at most 21/176 (11.9%) under a timing-isolation rule.

### 2.4 In-situ sound-speed data for November 2021 — much thinner than claimed

Queried OOI ERDDAP directly for 25 Oct – 15 Nov 2021:

| Dataset | Status in Nov 2021 |
|---|---|
| `ooi-ce04ospd-dp01b-01-ctdpfl105` (Oregon Offshore **deep profiler**) | **Actively profiling.** 129,177 samples, z −575…−97 m, 479 distinct depths |
| `ooi-rs01sbpd-dp01a-01-ctdpfl104` (Slope Base **deep profiler**) | **No data.** HTTP 404, zero rows |
| `ooi-rs01sbps-sf01a-2a-ctdpfa102` (Slope Base shallow profiler) | **Parked**, z −195…−189 m over five days — a point, not a profile |
| `ooi-ce04osps-sf01b-2a-ctdpfa107` (Oregon Offshore shallow profiler) | **Parked**, z −197…−192 m |
| `ooi-rs01sbps-pc01a-4a-ctdpfa103` (Slope Base 200 m platform) | Fixed-depth point, 23,960 samples |

So exactly one instrument was profiling, over 97–575 m. The cables span 0 to −1575 m. **We have
measured T/S for roughly a third of the depth range that matters, and nothing above 97 m or below
575 m.** QARTOD aggregate flags are present and non-trivial (temperature: 412 pass / 510 suspect /
130 fail over five days), confirming the note's warning that raw values must not be used unfiltered.

> [!WARNING]
> **[CORRECTED 2026-09-17] — 97 m is the raw mechanical sweep, not the usable band.** After
> QARTOD filtering (flag ≤ 2) the usable band for `ce04ospd_deep` is **216–575 m** over the 5-day
> primary window (5,067/7,472 rows, 68%) or **154–575 m** over 25 Oct – 15 Nov (93,901/129,177
> rows, 73%); the shallowest ~120–450 m of the raw sweep is QARTOD-suspect for temperature and is
> correctly dropped. The shallow edge is window-dependent — re-check it per fetch rather than
> quoting either number. §6.2 and §6.3 below inherit this correction: "97–575 m" there should read
> "~154–575 m at best, ~216–575 m for a temporally matched window."
>
> Two consequences carried into the results. The band actually validated in Plan A was **219–573 m**
> (n = 400 comparison points; median |Δc| 0.471 m/s, p90 1.085, max 1.372 against HYCOM). And the
> profile the experiments actually ran on (`hycom_south_deep`, at the South cable's deep end) was
> **never validated at its own location** — the validation site is 87 km away and ~1,000 m
> shallower. §6.4's two-independent-products requirement is also unmet: only HYCOM was used.

This inverts the intended data strategy — see §6.

### 2.5 The gradient is not the constant (added 2026-09-16)

Reproduce with `paper/scripts/recon_propagation_regime.py`. Profile is the indicative Oregon
November one: 1491 m/s at the surface falling to 1477 m/s at 500 m, then rising — **downward
refracting** over the whole depth band the whales occupy. Rays bend toward lower sound speed, so every
ray leaving a shallow source dives.

**A — the smooth correction is small, as claimed.** Depth-averaged `c` instead of a constant 1480
changes predicted arrival time by a median 8.8 ms, max 18.8 ms (28 m of range) across representative
geometries. Against a measured per-call pick-residual std of 96 ms, that is an order of magnitude
below the noise. *The original "negligible for L-1" verdict is correct for this part of the effect.*

**B — but the assumed path does not exist.** Maximum range at which a direct, water-borne ray can
still connect a source to a receiver:

```
z_src \ z_rcv     100 m    200 m    500 m   1000 m   1575 m
   10 m            1.6      2.4      4.7      8.6     14.2   km
   30 m            1.4      2.3      4.8      9.2     15.9   km
  100 m             --      3.8      9.0     19.0     21.7   km
```

Annotated calls sit 12.5–95 km along cable, with typical source–receiver ranges of 5–40 km. At the
assumed source depth of −30 m, a direct ray reaches a 100 m channel for only 1.4 km and a 1575 m
channel for 15.9 km. **For most picks in this dataset the straight-line path the model integrates
along is not a propagation path at all**; the real first arrival must be bottom-interacting.

> [!WARNING]
> **[OVERTURNED 2026-09-17] — this paragraph's conclusion does not survive a modal treatment.**
> The paragraph is kept as written because §7.0.3 was designed to adjudicate exactly it, and it
> lost. Measured (`paper/results/regime_shadow_zone.csv`, five geometries at
> 20 Hz on `hycom_south_deep`, seabed sound speed swept over 1550/1700/1900 m/s): ray theory calls
> all five geometries shadowed, and **a trapped mode exists in every one of them, for every seabed
> in the sweep** — `n_modes_min` ranges 1 to 13 and is never 0. The solver is validated against the
> analytic Pekeris waveguide (12/12 and 224/224 exact mode counts, wavenumbers to 4e-8 rad/m).
>
> What is falsified is the sentence in bold: the *ray-theoretic* claim that no propagation path
> exists is a geometric-acoustics artifact in a guide 1.3–21 wavelengths deep. What is **not**
> established is the following clause — that the first arrival is bottom-interacting, or indeed
> that energy arrives at all. The solver computes existence and counting, with no modal
> attenuation, no source excitation and no transmission loss. "Trapped modes exist" is not "energy
> arrives at 5–40 km". Closing that gap needs a transmission-loss calculation, which Plan A did not
> ship; it is the first item on Plan B's list.
>
> §1's amendment paragraph, which asserts the straight-line path "does not exist" for most of this
> geometry on the strength of this section, inherits the same correction.

**C — and the real data shows it.** Mean residual by source–receiver range, stratified by SNR
quartile (n = 10,470 picks, gross mis-associations excluded):

```
                 0-5km   5-10km  10-20km  20-30km  30km+     trend
snr Q1 (low)      -8.0     17.5     19.8     42.0    37.6     +45.6 ms
Q2               -21.4     10.2     20.6     50.0    28.3     +49.6 ms
Q3               -32.8     13.3      8.4     56.5    43.7     +76.5 ms
Q4 (high)        -54.8     -4.7      3.6     58.0   119.4    +174.2 ms
```

The confound to exclude is that long range means low SNR, and a low-SNR envelope pick lands late.
If that were the cause, the trend would be largest in Q1 and absent in Q4. **It is the other way
round** — strongest at the highest SNR — so this is model error, not pick bias. The magnitude
(45–174 ms) is far larger than the 8.8 ms the sound-speed profile correction would supply.

**What this does not establish.** The residuals come from a solution fit to those same picks, so
anything resembling a source shift is already absorbed and what survives is a lower bound. Source
depth is fixed at −30 m here, and a wrong source depth produces short-range structure of its own, so
path error and depth error are confounded — which is not a nuisance but a connection: the same signal
carries both. And at 20 Hz (λ ≈ 74 m) in 100–500 m of water, ray theory, including the sharp shadow
zone it predicts, is marginal. Modes smear shadow zones. This measurement raises the regime question
decisively; it does not settle it.

**Literature check.** Ray tracing through a measured profile is standard in other whale-localization
work — sperm whales with depth-varying SSP, fjord localization with refraction and reflection
modelling, and general underwater localization reporting a few metres of improvement over constant
speed. **In the DAS whale literature it is unused**: all four studies tabulated in the feasibility
note fix a single constant. So the answer to "has anyone tried perturbing `c` with a non-constant
profile on DAS?" is no.

## 3. Corrections to the feasibility note

Errors in `paper/docs/sound-speed-feasibility.md` found while verifying it. Recorded here; the note will be
amended.

1. **"Paired shallow + deep CTD profilers give full-depth 0–2900 m T/S several times per day,
   verified continuous through Nov 2021"** — wrong on every clause for Slope Base. The deep profiler
   returns no data for that period and the shallow profiler was parked at ~193 m. The site code was
   also wrong: the deep profiler is under `RS01SBPD`, not `RS01SBPS`. The usable instrument is at a
   different site (CE04OSPD) and covers 97–575 m.
2. **"Every row has `z = −30.0 m` exactly"** — true of the published pair-localization CSV, but their
   single-cable output in the association files uses `z = −40.0`. The claim that published DAS
   localization always fixes depth is unaffected; the specific constant is not universal.
3. **"Roughly a day of ingestion work"** for the oceanographic side — optimistic given §2.4. A
   defensible profile now requires reanalysis plus in-situ validation rather than in-situ alone.
4. **"A uniform `c` error is a scale factor... the dominant L-1 error is geometric"** used the
   Rørstadbotnen constant-`c` perturbation to dismiss sound speed for L-1 generally. That is a
   category error: perturbing a constant tests the scale factor; it says nothing about the gradient,
   which refracts. §2.5 shows the smooth part of the gradient effect is indeed small, but that the
   straight-line path is wrong for most of this geometry and the residuals carry range-structured
   model error an order of magnitude larger than the correction the note proposed chasing.

## 4. Architecture

`src/dclde_localization/physics/forward.py` is **frozen**. It is bit-for-bit parity-tested against
`viz/template/lib/forward.mjs`, which the published visualizer executes in the browser; changing its
signature breaks a shipped cross-language invariant. All new physics layers above it.

```
src/dclde/das_finwhale/
  forward.py                 # FROZEN: constant-c travel time, t0 elimination, misfit
  geometry.py                # unchanged
  soundspeed/
    profile.py               # T/S -> c(z);  reanalysis base + in-situ validation
    erddap.py                # OOI ERDDAP client, QARTOD filtering
    models.py                # SoundSpeedModel implementations
    modes.py                 # range-independent normal-mode solver (phase 2)
  solve.py                   # estimator; returns ALL basins, not a point
  simulate.py                # synthetic calls: known source -> arrivals -> noise
  experiments/
    runner.py                # config -> result table
    repro.py                 # baseline reproduction + per-call comparison
paper/configs/*.yaml
```

### 4.1 The propagation interface

```python
class SoundSpeedModel(Protocol):
    name: str
    def travel_time(self, src_xyz, chan_xyz) -> np.ndarray: ...
```

Implementations:

| Model | Physics | Role |
|---|---|---|
| `ConstantC(c)` | straight ray, single `c` | the published baseline; the null |
| `PathAverage(profile)` | straight ray, harmonic mean of `c(z)` between endpoint depths | cheapest realistic model; **the one the note's hypothesis is about** |
| `RayTrace(profile)` | Snell refraction through `c(z)` | tests whether path-average is adequate |
| `ModalDelay(profile, seabed)` | group delay of trapped modes at 20 Hz | phase 2; tests whether ray theory holds at all |

`c` cannot be precomputed per channel and handed to `forward.travel_time`: the path average between
source depth `z_s` and channel depth `z_c` depends on `z_s`, which is the unknown being solved for.
That is why this is a model object rather than an array, and why the interface takes source and
channel positions rather than a speed.

**Equivalence test:** `ConstantC.travel_time` must equal `forward.travel_time` to machine precision.
This keeps one definition of the physics, enforced rather than promised.

### 4.2 The self-consistency guard

The way this program fails silently is by simulating and inverting with the same physics, which
recovers truth by construction and measures nothing but conditioning.

`simulate.generate()` and `experiments.runner` take `truth_model` and `inversion_model` as two
**separate required arguments**, neither defaulting to the other. Any experiment whose stated purpose
is measuring model-error bias asserts `truth_model.name != inversion_model.name` and raises otherwise.
A test constructs a config that violates this and asserts the run refuses to start.

This is architectural rather than procedural on purpose: it is the mistake most likely to invalidate
everything downstream, and it should be impossible to make by accident.

## 5. The estimator

**Unknowns:** `x, y` always free; `z_s` and `c` free per experiment; `t0` eliminated analytically
(median residual for L1, mean for L2 — already in `forward.py`).

**The estimator returns every basin, not a point.** The misfit is bimodal by construction for a single
cable. A local optimizer seeded from a coarse grid lands in whichever basin the grid happened to
favour, and a small parameter change can flip it. If an ablation reports "positions moved a median of
80 m" while 5% of calls silently jumped to the mirror lobe kilometres away, the median hides the only
interesting thing that happened.

So `solve()` returns a ranked list of refined local minima with their misfits and separations, and
**every ablation reports two numbers separately**:

- the position delta for calls that stayed in the same basin, and
- the count and fraction of calls that changed basin.

Merging those into one statistic is forbidden by the reporting contract, not by convention.

**Procedure:** coarse grid over a box wide enough to contain both lobes → enumerate local minima →
refine each with a bounded local optimizer over the free parameters → rank by misfit. The visualizer's
two-pass bracketing (`viz/template/lib/contour.mjs`) already does the grid half of this, and its
sizing logic is reused.

## 6. Sound-speed profiles — reanalysis base, in-situ validation

§2.4 forces this order. The plan cannot be "build a profile from OOI", because OOI covers 97–575 m of
a 0–1575 m problem.

1. **Base profile:** two independent products, both full-depth, at the cable locations for Nov 2021.
   T/S → `c(z)` via `gsw.sound_speed` (TEOS-10).
   - **HYCOM GOFS 3.1** (`GLBy0.08/expt_93.0`, 1/12°, 40 levels) — primary. Open NCSS subsetting, no
     account needed.
   - **WOA23** monthly 1/4° climatology — independent cross-check. Not a Nov-2021 observation; it is
     a climatological November, and results must say so.
   - **GLORYS12V1** is the better daily product but needs a Copernicus Marine account. Not in the
     critical path; add it only if someone registers. Do not block on it.
2. **Validation:** compare against the CE04OSPD in-situ profiles over 97–575 m, QARTOD-filtered. That
   band contains the sound-channel minimum, so it is the most diagnostic third we could have got.
   **Report the disagreement, in m/s, as a measured quantity** — it is the uncertainty on every
   downstream result and must not be quietly dropped.
3. **Above 97 m and below 575 m:** reanalysis only. Flag it everywhere. The upper 97 m matters most,
   because sources sit there and every ray passes through it.
4. **Sensitivity:** run the key experiments with HYCOM and with WOA23 separately. If the conclusion
   differs between two reasonable profiles, the conclusion is not supported.

Inter-equation disagreement among Chen–Millero / Del Grosso / Mackenzie is 0.05–0.5 m/s, far below the
effect being chased, so the equation choice is not a decision.

## 7. Experiments

Each is a YAML in `paper/configs/`, run by `experiments/runner.py`, emitting a
versioned result table committed next to the document that interprets it. No W&B. Every result states
its denominator: how many calls, which batches and cables, and how many were excluded and why.

### 7.0 Propagation regime — is the assumed path the real path? (added 2026-09-16)

**Not gated. Runs alongside §7.1, before the ablations**, because if the answer is "the first arrival
is bottom-interacting" then §7.2 and §7.3 are comparing two wrong models to each other.

The question §2.5 raises and cannot close: at 20 Hz, does the shadow zone ray theory predicts actually
exist, and what path does the energy take? Three strands, in order of cost:

1. **Disentangle path error from source-depth error.** §2.5C's range trend is confounded with the
   fixed −30 m assumption. Re-run the residual stratification with `z_s` free and with the source
   depth swept, and see how much of the trend survives. Cheap, uses committed data, and its outcome
   tells us whether §7.4 is measuring depth or measuring path error wearing depth's clothes.
2. **Moveout-shape test.** A direct arrival and a bottom-bounce arrival trace measurably different
   curvature in (along-cable distance, time). Fit both to the real picks and compare. This is the
   most direct evidence available from data we already hold.
3. **Modal calculation** (§7.6, now un-gated). The only way to answer whether the shadow zone is real
   at 20 Hz. A ray-theoretic shadow zone in a waveguide a few wavelengths deep is a statement modes
   routinely contradict.

**Reporting contract:** this experiment is allowed to conclude that the sound-speed programme is
mis-aimed. If bottom interaction dominates, the honest finding is that propagation-path modelling —
not water-column sound speed — is the lever on this dataset, and that is a more useful result than a
carefully-measured correction to the wrong model. It should be written up as such.

### 7.1 Baseline reproduction

Run the reference implementation (patched only for colon paths) on batch 1; compare to its own
published CSV; then run **our** estimator on the same picks and compare per call, joined through `utc`
plus window.

Acceptance: their code reproduces their CSV exactly. Our estimator's differences from theirs are
reported as a distribution, and any call differing by more than its own pick-noise-implied uncertainty
is investigated individually rather than averaged away. No pass/fail threshold is set in advance,
because a fabricated threshold would only be a way of not looking.

### 7.2 Free-`c` ablation

Add `c` as a free scalar. Report position deltas and residual change against fixed 1480, split by
basin-stable / basin-flipped, plus the distribution of fitted `c`. Expected outcome: a free scalar
absorbs most of the effect for L-1, which **closes the L-1 branch** — it does not stop the program,
because this fit is the null model that §7.4 must beat.

### 7.3 Effective-`c` ablation

`PathAverage` and `RayTrace` against `ConstantC` and against free-`c`. Same reporting.

### 7.4 Depth observability — the actual experiment

Two parts, and the second is the one that matters.

**(a) On real calls:** with `z_s` free, measure the curvature of the misfit in `z` under each model,
restricted to calls whose picks span wide `z_c`. Report identifiability stratified by depth span
(§2.1 gives the strata). This says whether the objective has a defined minimum in depth.

**(b) On synthetic calls, which is the only source of a number in metres:** place sources at known
`(x, y, z)`, generate arrivals with `truth_model`, add noise per §8, invert with
`inversion_model = ConstantC`, and measure actual depth error. Then invert with `PathAverage` and
measure how much of that error is removed.

Curvature alone cannot distinguish a sharp minimum in the right place from a sharp minimum in the
wrong place; (a) without (b) would let a confident bias pass as success.

### 7.5 Pick density

The reference's own association yields ~9,500 picks per call against ~18 in the annotations. Re-run
§7.4(a) on both to establish whether depth identifiability is limited by geometry or merely by pick
count. Cheap, and it changes what we would recommend to the challenge.

> [!WARNING]
> **[CORRECTED 2026-09-17] — "~9,500 picks per call" is wrong; see §2.3.** Measured from
> `associations.zip` (12 batch-1 `Baseline` windows, 194 `assoc_pair` items): **min 324, median
> 4,684, mean 5,328, max 12,228**. And "~18" is the batch-1 median; over all 628 annotated calls
> the median is **14** (min 3, max 61). The density contrast this experiment turns on is therefore
> roughly 300×, not 500×, which does not change the experiment's motivation — but this section is
> a **Plan B** section, so the figure a follow-on planner reads here must be the measured one.
>
> Two further constraints on planning §7.5, both established by Plan A. The association archive is
> under the gitignored `data/` and may have been deleted; `paper/scripts/fetch_reference_data.py`
> restores it. And re-running §7.4(a) "on both" requires knowing which annotated call each
> association item corresponds to — which is exactly the join §7.1 found to be infeasible from
> timestamps alone (§2.3's correction). Scope this experiment against the association filenames
> and window keys directly, not against the published CSV.

### 7.6 Modal calculation (un-gated as of 2026-09-16)

**Originally gated behind a promising §7.4. §2.5 removes that gate**: a ray-theoretic shadow zone in a
waveguide a few wavelengths deep is exactly the claim normal modes exist to adjudicate, and §7.0
depends on the answer. It is no longer a phase-2 nicety — it is on the critical path for interpreting
everything else.

The purpose is **not** a better simulator: it is to bound how wrong ray theory is at 20 Hz, where
λ ≈ 74 m and the waveguide is 1.3–7 wavelengths deep over much of this span. Seabed geoacoustics are
unknown, so sediment speed, density and attenuation are **swept over a plausible range** and the
spread reported as an uncertainty, never fixed at invented values.

If mode dispersion moves arrival times more than the water-column correction does, that is the most
important finding available here and should be the headline rather than a footnote.

**Implementation:** `arlpy` and `pyat` both shell out to Acoustics Toolbox Fortran binaries that are
not pip-installable and are hostile on Windows. We implement a range-independent normal-mode solver
directly (finite-difference Sturm–Liouville eigenproblem via `scipy`), validated against the analytic
Pekeris waveguide. That is a testable few hundred lines and more defensible than an unvalidated binary
dependency.

## 8. The noise model — where this is most likely to fool us

A synthetic experiment is worth exactly what its noise model is worth. Three ways to get it wrong, all
of which make results look better than reality:

1. **Gaussian.** §2.2 shows a heavy tail out to 43.5 s from mis-associations. Modelled as an explicit
   outlier mixture, with the contamination fraction measured rather than chosen.
2. **Homoscedastic.** Per-call residual std ranges p10 0.030 to p90 0.150 s. Per-call sigma is drawn
   from the measured distribution.
3. **Independent across channels.** This is the dangerous one. A mis-picked or biased moveout curve
   shifts a *contiguous run* of channels together. I.i.d. noise averages down as 1/√n over ~18 picks;
   correlated noise does not, so assuming independence would overstate precision — and could
   manufacture depth identifiability that does not exist. **We measure the along-cable autocorrelation
   of real residuals first and match it** (AR(1) or an exponential kernel in channel index), rather
   than assuming a structure.

**One caveat that must be carried into every result:** the residuals of §2.2 are residuals of a
*constant-`c` model*, so they contain pick error **plus** model error. Calibrating noise from them
therefore gives an upper bound on pick noise, and using it makes our synthetic experiments
conservative. That is the right direction to err, but it must be stated, not silently enjoyed.

**Simulator acceptance test (a gate, not a nicety):** synthetic calls must reproduce held-out summary
statistics of the real data — residual quantiles, along-cable autocorrelation, per-call sigma
distribution. If they do not, the simulator's error numbers are not evidence about the ocean, and
§7.4(b) does not run.

## 9. Data provenance and deletion

The 3.6 GB of reference data is temporary on this machine by agreement. `paper/scripts/fetch_reference_data.py`
re-downloads both artifacts from their canonical sources, verifies recorded SHA-256 checksums, and
applies the colon rename — so deleting the local copy costs a teammate one command rather than a
reconstruction. The manifest, checksums and source DOIs are committed; the data is not (`data/` is
gitignored).

Sources: `Ocean-Data-Lab/Goestchel_JASA_2025b` (GPL-3.0) and Zenodo record 21220346. Deletion happens
once the experiments that consume them have produced committed result tables.

## 10. Testing

Per CLAUDE.md, the load-bearing places get unit tests — here, anything whose silent failure corrupts
every downstream number:

- `ConstantC` equals `forward.travel_time` to machine precision.
- The self-consistency guard refuses a truth == inversion config.
- `PathAverage` with a constant profile reduces exactly to `ConstantC`.
- The normal-mode solver reproduces the analytic Pekeris waveguide.
- The estimator recovers a known source from noiseless synthetic arrivals to grid precision.
- The estimator returns **both** lobes on a synthetic single-cable case with a known mirror.
- QARTOD filtering rejects the documented bad ranges (salinity 33929 PSU, temperature 61 °C).
- Round-trip on the `:`→`_` rename against the manifest.

Tests that restate an implementation formula against itself are not tests; the mirror-recovery and
Pekeris cases are the ones with real content.

## 11. Risks

| Risk | Handling |
|---|---|
| Depth unobservable even with aperture | A clean negative is a deliverable (§1). Report it. |
| **The whole programme is mis-aimed** — bottom interaction, not water-column `c`, is the lever | §7.0 is designed to find this out early and is explicitly allowed to conclude it (§2.5). Cost of missing it: every later number is a precise correction to the wrong model. |
| Path error and source-depth error confounded | §7.0.1 separates them before §7.4 runs; they share one signal, so §7.4 is uninterpretable until it is done. |
| Seabed dominates the water column | §7.6 is designed to detect exactly this and report it as the headline. |
| No profile above 97 m or below 575 m | Reanalysis, flagged; two independent products cross-checked (§6.4). |
| Simulator self-consistency | Architectural guard (§4.2) plus acceptance test (§8). |
| Basin flips hidden by medians | Split reporting mandated (§5). |
| No depth ground truth ever exists | Then L-2's Z is unscoreable; that finding is itself worth publishing and worth telling the organizers. |

## 12. Delivery order — this spec yields two implementation plans

It is too much for one plan, and more importantly the second half should not be designed in detail
until the first half reports.

**Plan A — reproduction, propagation regime, and the L-1 gate (§7.0–7.3).** Needs `solve.py`,
`models.py` with `ConstantC` and `PathAverage`, the profile pipeline (§6), `repro.py`, the
fetch/provenance script (§9), **and `modes.py`, pulled forward from phase 2 because §7.0 depends on
it**. Ends with a committed result table, a verdict on whether a free scalar `c` absorbs the effect
for L-1, and — the new headline — a verdict on whether the first arrival is a direct path at all.

**Plan B — the depth experiment (§7.4–7.6).** Needs `simulate.py`, the noise model (§8) and its
acceptance gate, and `modes.py` only if §7.4 clears. Planned after Plan A reports, because the free-`c`
distribution and the measured residual autocorrelation both feed its design, and guessing them now
would mean rewriting it.

Nothing about Plan B is deferred for scope reasons — §4.2's guard and §5's split reporting land in
Plan A, because Plan B depends on them being right.

## 13. Open questions for the organizers

Not blocking §7.1–7.3; blocking the interpretation of §7.4.

1. **Is there any depth ground truth?** If challenge ground truth is the baseline's own output, GT `z`
   is a fixed −30 m assumption and L-2's Z component cannot be scored. The most important question to
   ask.
2. What is the 2027 localization metric? Unpublished; keep scoring swappable per CLAUDE.md.
3. Are uncertainty submissions scored for calibration, or only for presence? This determines whether a
   multimodal posterior is a differentiator or merely honest.
