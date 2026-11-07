# How fin-whale calls reach the Oregon cables — one-page brief

_2026-09-21. Plain summary of the transmission-loss study. Full detail, with every number's
denominator and every caveat, is in
[`das-finwhale-transmission-loss-findings.md`](das-finwhale-transmission-loss-findings.md)._

## The question

Our localization assumes each call travels in a **straight line** from whale to cable. If instead
the first sound to arrive had bounced off the seafloor, or travelled trapped inside a layer of
water, then we would be modelling the wrong path — and that would bias horizontal positions, not
just depth. Earlier work had shown such trapped paths *exist* here. This study asked whether they
actually **arrive**.

## What we did

Built a proper acoustic propagation model — sound bending in the water column, energy leaking into
the seabed, paths tracked across real seafloor topography from shelf to deep water, and the fact
that a fibre-optic cable hears sound arriving *along* it much better than sound arriving
*sideways*. Then we compared two competing pictures against the real detections:

- **Straight-line**: sound spreads out and weakens with distance, nothing more.
- **Trapped**: sound is guided inside the water layer, losing energy into the seabed along the way.

Because the seabed's composition off Oregon is unknown, we ran the comparison against **five**
possible seabeds, from soft mud to sand and gravel, and required the answer to hold for all of them.

**Data:** 628 annotated calls / 11,117 arrival times. After excluding calls with too few arrivals,
calls that would not solve, calls with ambiguous positions, and calls whose path crossed gaps in
the seafloor map, **381 calls and 8,660 arrivals** remained. Every exclusion is counted, not
quietly dropped.

## What we found

**The straight-line picture fits better — clearly, and for every seabed we tried.** The comparison
is not close and does not flip anywhere across the five seabeds. On top of that, the trapped
picture cannot account for **38% of the arrivals at all**: it predicts those calls should have been
too faint to detect. They were detected.

**Practical consequence:** the concern that motivated this work — that we might be modelling the
wrong path, corrupting horizontal positions — is not supported. Our 2-D localization can proceed on
straight-line geometry.

## Two things this does *not* settle

**1. It ranks the two pictures; it does not prove the winner is right.** The detections come from a
*denoised* product, and we have no calibration telling us how its loudness numbers relate to real
sound levels. We found one accidental check: the cable's sideways-versus-lengthways sensitivity is
fixed by geometry, so we know what it should measure — and the data register it at roughly a
**twelfth** of that. Something is compressing the loudness scale, which means "how fast sound fades
with distance" cannot be read off these numbers at face value.

Worth stating plainly: **that distortion, if real, would push the comparison toward the
straight-line answer — the one that won.** We cannot separate the two with this data. The finding
is honest as a ranking and should not be quoted as settled physics.

**2. Source depth remains untestable, and not for want of trying.** We checked the published
dataset object by object: 242 files, all arrival times, strain images and cable coordinates.
**There is no depth ground truth, and no localization ground truth of any kind.** Most likely it is
being withheld until scoring. So while the cables' curved, two-fibre layout should in principle
make depth recoverable, we cannot check any depth estimate against truth today. The depth signal we
*can* measure is statistically indistinguishable from zero.

## What we would want next

1. **A calibration** — any amplitude measure that is not the denoised product would test the
   compression directly, and it is the single thing blocking a physical reading of the result.
2. **An answer from the organizers** on whether depth truth exists and whether depth will be scored
   at all. If scoring truth is the baseline's own fixed −30 m assumption, the depth component is
   unscoreable and we should know that before building for it.
3. A treatment of sound energy leaking *between* guided paths on a steep slope, which our model
   approximates and which is the most likely reason the trapped picture fails on 38% of arrivals.

_Everything above regenerates from committed configuration files; the result tables carry their own
provenance and caveats in their headers._
