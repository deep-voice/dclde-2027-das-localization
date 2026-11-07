# Plan B — depth observability on the DAS fin-whale data

_Design note, 2026-09-21, branch `sound-speed-research`. Short by intent: it exists to supply the
new justification and the new null that Plan A's findings §5 said Plan B lacked, and to fix the
success criteria before any code is written._

Prior art this rests on, and does not re-argue:
[Plan A findings](../das-finwhale-soundspeed-plan-a-findings.md) ·
[transmission-loss findings](../das-finwhale-transmission-loss-findings.md) ·
[Plan A spec](2026-09-15-sound-speed-research-design.md) §7.4 ·
[transmission-loss spec](2026-09-20-das-transmission-loss-design.md)

---

## 1. Why this can be attempted now

Plan A ended by withdrawing both of Plan B's supports: the range-residual trend is **not** a depth
artifact (§3.2), and free-`c` is a nuisance sink, **not** a null (§3.4). Two further results have
landed since. Together they replace what was withdrawn.

1. **The depth aperture survives, unchallenged.** Per-call span of receiver depth: median **219 m**,
   p90 575 m, max 1,085 m; **348 / 628** calls span ≥ 200 m; two-cable calls (n = 166) median 372 m.
   The leverage term `∂t/∂z_s = (z_s − z_c)/(c·R)` has real geometry to work with. This was always
   the single assumption the depth hypothesis rested on, and nothing has touched it.

2. **The path objection is answered, comparatively.** The transmission-loss study found the
   straight-ray picture fits the observations better than a trapped one at every seabed in the
   sweep, sign-stably, and the trapped picture additionally cannot predict 38% of the detections.
   So a depth solve on straight-ray geometry is no longer obviously modelling the wrong path.
   **The qualifier is load-bearing**: that ruling is comparative, it rests on an observable whose dB
   scale failed its own calibration check, and the bias that implies points toward the straight-ray
   answer. It licenses proceeding; it does not certify the forward model.

3. **Depth must be reached through travel time, not through level.** The same study measured the
   receiver-depth coefficient of received level and its interval **spans zero in all 16
   configurations**. Amplitude carries no resolvable depth information on this data. That closes one
   route and forces the other, which is the route §7.4 always intended.

**Scoping measurement, made before writing this note** (440 calls with ≥ 8 picks and ≥ 100 m of
depth span, solved under `ConstantC(1480)`): the moveout **shape** changes by a median **5.00 ms**
per 100 m of source depth (p90 14.4 ms) against a robust per-call residual σ of **81.2 ms**, which
√n averaging brings to an effective **16.4 ms**. Implied 1σ depth resolution: median **393 m**,
best decile **36 m**; **145 / 440** calls below 200 m and **246 / 440** below 500 m.

This is an **optimistic bound and must not be quoted as a result.** It treats the depth direction as
independent of `x`, `y` and `t0`, when depth trades off against range and emission time — the joint
solve will be worse. And it treats the residual as white, when Plan A showed it carries a structured
76–82 ms range trend, which biases rather than merely scatters. Its only job is to establish that
depth is not obviously hopeless, and therefore that the gate below is worth running.

## 2. The null

**`ConstantC(1480)` straight-ray propagation, with source depth fixed at the baseline's −30 m.**

Fixed by exclusion, not preference. Free-`c` is disqualified (Plan A §3.4: 41.8% edge-pinning, a
nuisance sink). A depth-dependent `c` cannot be the null because whether it helps depth is a
*question* in §5, not an assumption. And the transmission-loss study established the straight-ray
family as the better-fitting path description, so it is the incumbent a depth solve must beat.

Every depth result is reported against this null on the same calls. A depth estimate that does not
improve on fixed −30 m has not demonstrated anything, however confident its interval.

## 3. The gate — §7.0.2, run first, and permitted to stop the plan

**Question:** does the moveout curve change with source depth by more than the per-call pick
scatter, on real picks?

This is the identifiability precondition for everything downstream. It is deliberately *not* the
amplitude question — that route is closed by §1.3 — and it uses travel time, an observable the
calibration problem does not touch.

Carried alongside, as the independent travel-time check on the path: **direct versus bottom-bounce
curvature** in (along-cable distance, time), fitted to the same picks. The transmission-loss study
answered the path question from amplitude; this answers it from timing, on an observable with a
different and largely independent failure mode. Agreement strengthens both; disagreement is itself a
finding and must be reported as one rather than resolved in favour of the earlier answer.

**Gate criterion, fixed here.** Proceed to §4 only if, on a pre-declared stratum of calls with
adequate depth aperture, the depth-induced moveout shape change exceeds the per-call effective noise
by a stated margin on a **material fraction** of calls — material meaning enough calls to support
the stratified reporting §4 requires, not a handful of favourable ones. If it does not, Plan B
**stops at the gate**, and that is a publishable negative: it would mean this geometry cannot
support depth inference at this pick precision, which is a more useful statement than a depth
estimate nobody can trust.

The stratum and the margin are set in the implementation plan, before the gate is run, and recorded
there. They are not chosen after seeing the outcome.

### 3.1 Outcome — the gate FAILED, 2026-09-21

**`gate_passes = False`. Plan B stops at the gate.** §4, §5 and Plan A spec §7.4 are not built.

The concrete criterion, fixed in `paper/configs/gate_moveout.yaml` before the
run and applied unchanged: stratum of `n_picks >= 8`, receiver-depth span ≥ 200 m, basin-stable,
solved under the §2 null; pass requires **≥ 10% of the stratum AND ≥ 50 calls** at σ_z ≤ 100 m.

| Quantity | Value |
|---|---|
| Stratum (denominator for both rates) | **271** calls, of 628 annotated |
| `n_resolved_forward_only` — **the pre-declared criterion alone** | **17 / 271 = 6.3%** |
| `n_resolved` — after a walk-consistency condition added *after* this note | **2 / 271 = 0.74%** |
| Both legs | fail on either figure (6.3% < 10%; 17 < 50) |

15 of the 17 are artifacts of direction-dependent local-minimum tracking on single-cable geometry,
established on four independent grounds. The mechanism: on a cable whose receiver depth varies
monotonically along its length, source depth is very nearly exchangeable with an along-cable
source-position shift — 400 m of depth error leaves ~23 µs of residual after refitting `x` and `y`,
against ~918 µs on a V-shaped arc and ~3,044 µs on two arcs at different bearings.

The carried path check agrees with the transmission-loss study's amplitude ruling at the ensemble
level — 235 of 266 calls favour the direct path — but **0 of 266** exceed their own residual noise,
so no individual call resolves it.

Full report, with all denominators and limitations:
[`paper/docs/das-finwhale-plan-b-gate-findings.md`](../das-finwhale-plan-b-gate-findings.md).

## 4. What counts as "depth resolved"

Two parts. **Neither half alone counts**, for the reason Plan A's spec already gave: curvature
cannot distinguish a sharp minimum in the right place from a sharp minimum in the wrong place.

**(a) Identifiability, on real calls.** With `z_s` free, the misfit must have a genuine, grid-resolved
minimum in `z` whose curvature exceeds the noise — not a flat direction, not a bound-pinned edge.
Reported **stratified by depth aperture**, using the §1 strata, with the count of calls in each and
how many are identifiable. This says the objective has a defined minimum. It says nothing about
where.

**(b) Error in metres, on synthetics.** The only source of a number in metres, because — verified
object-by-object against the published dataset — **there is no depth ground truth and no
localization ground truth of any kind**. Place sources at known `(x, y, z)`, generate arrivals under
the truth model, add noise matched to the *measured* per-call residual scale rather than an invented
one, invert with the §2 null, and report the depth error distribution. Then report how much of that
error the alternative in §5 removes.

**Resolved** therefore means: identifiable on real calls in a stated stratum, **and** a synthetic
depth error whose distribution is reported in metres with its own n — not a single summary number.
A bias check is mandatory: a tight interval centred on the wrong depth is the failure mode this
two-part definition exists to catch.

**Not resolved is a permitted and complete outcome**, at either part, and is reported as such.

## 5. The optional second question

**Does `c(z)` help depth?** Only after §4, only on the same calls, and only as a comparison against
the §2 null.

Explicitly **not** in scope: producing depth maps, per-call depth catalogues, or any downstream
product built on depth estimates. Plan A closed the 2-D sound-speed branch by measurement (a median
9.0–9.5 m position shift against a 142 m noise floor); this asks the narrow question of whether a
depth-dependent profile buys anything for the *depth* component specifically, which Plan A left open.

## 6. Reporting rules, inherited and non-negotiable

Every rate states its denominator and n at the point of use. Basin-flipping calls are counted and
never merged into a pooled statistic, in tables or prose. Result tables regenerate from committed
configs, and that claim is verified by regenerating and diffing rather than asserted. Figures use
seaborn, minimal in-figure text. Anything reported is promoted to `src/dclde/` first.

Two specific to this plan:

- **No depth number is quoted without saying whether it came from real calls or synthetics.** They
  are different claims with different warrants, and there is no truth to check the real ones against.
- **The §1.2 qualifier travels with every result.** The path licence is comparative and rests on an
  uncalibrated observable. Any conclusion that would change if the straight-ray picture were wrong
  must say so.

## 7. Scope

**In:** the §3 gate; §4(a) identifiability on real calls; §4(b) synthetic recovery with a
measurement-matched noise model; the §5 comparison if §4 clears.

**Out:** depth maps and catalogues; §7.5 pick-density (the reference association cannot be joined to
our calls — Plan A §3.1 found 0/176 rows join unambiguously); any further water-column sound-speed
work for horizontal position; re-opening the path question from amplitude.
