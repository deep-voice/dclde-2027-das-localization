# Sound-speed structure for DAS fin-whale localization — feasibility assessment

_Written 2026-09-10. Assesses whether using measured/modelled ocean sound-speed structure
(instead of the standard constant `c`) can improve fin-whale localization on the DCLDE 2027
Challenge L dataset (two DAS fibers off Oregon, Nov 2021)._

> [!WARNING]
> **Corrected 2026-09-17 after Plan A was executed.** Four claims in this note were wrong and are
> amended in place below, each marked **[CORRECTED 2026-09-17]** with the original text kept so a
> reader can see what changed: the OOI data-availability claim (§ Data availability), the
> `z = −30.0 m` claim (§ Why it may be enabling for L-2), the ingestion-effort estimate, and the
> constant-versus-gradient category error that runs through § Why it is negligible for L-1.
>
> The last of these is the one that matters most: perturbing a *constant* `c` tests a scale
> factor and says nothing about a *gradient*, which refracts. That error is what motivated spec
> §2.5 and §7.0.
>
> Measured outcomes of the decision gate proposed at the bottom of this note are in
> [`das-finwhale-soundspeed-plan-a-findings.md`](das-finwhale-soundspeed-plan-a-findings.md).
> Headline: steps 1–3 of the gate all returned negative or uninterpretable. The baseline cannot
> be reproduced per call (the released CSV has no call identifier); a free scalar `c` is a
> nuisance sink, not a sound-speed estimator; and the depth-dependent effective-`c` correction
> moves horizontal positions by 6.3–6.7% of this dataset's pick-noise floor, i.e. below what it
> can resolve.

## Verdict

| Sub-task | Effect of realistic sound speed | Confidence |
|---|---|---|
| **L-1** — localize in 2D from a single fiber | **Negligible.** Second-order behind the left–right ambiguity; a uniform `c` error is largely absorbable as a scale factor. | High — there is a direct published measurement (below). |
| **L-2** — localize in 3D from two fibers | **Potentially enabling.** The only proposed depth mechanism on this geometry is differential travel time across channels at very different seafloor depths — which is exactly where constant-`c` produces a *structured bias* that a scale factor cannot absorb. | Low/medium — the depth mechanism itself is unpublished and untested. |

Short version: this is not an accuracy-improvement idea, it is a **depth-observability** idea.
Pursue it only as part of attacking L-2, and only after the baseline is reproduced.

## The idea

Standard practice inverts along-cable arrival times to a source position assuming a single
constant sound speed. Every published DAS whale-localization study does this:

| Study | `c` used | Source depth |
|---|---|---|
| Goestchel, Wilcock & Abadi 2026 (our dataset's descriptor) | 1480 m/s, fixed | fixed −30 m |
| Horne, Stork & Stanek 2025 (same Nov 2021 Oregon data) | 1480 m/s | fixed 10 m |
| Bouffaut et al. 2022 (Svalbard) | 1500 m/s | fixed 30 m |
| Rørstadbotnen et al. 2023 (Svalbard, two cables) | 1440 m/s, calibrated from air guns | fixed 20 m |

The proposal: use real bathymetry plus temperature/salinity profiles for this place and time to
compute a depth-dependent (and possibly spatially varying) sound-speed field, and invert with it.

## Why it is negligible for L-1

> [!IMPORTANT]
> **[CORRECTED 2026-09-17] — the category error in this section.** Points 1 and 2 below use a
> perturbation of a *constant* `c` to dismiss sound speed for L-1 in general. That is a category
> error: perturbing a constant tests a **scale factor**; it says nothing about a **gradient**,
> which refracts, and refraction can remove a propagation path rather than retime it. See spec
> §2.5 and §3.4 for the full statement.
>
> What Plan A measured, which changes the reasoning without changing the verdict:
> - The smooth part of the gradient effect *is* small, as this section claims — but the right
>   comparison is against this dataset's pick-noise floor (~96 ms ≈ 142 m at `c` = 1480), not
>   against published perturbation results. A depth-dependent effective speed moves basin-stable
>   positions by a median 9.0–9.5 m, **6.3–6.7% of that floor** (n = 459 stable of 489 compared,
>   30 flipped). The correct statement is "unresolvable at this dataset's SNR", not "zero".
> - Point 2's proposed cheap version — fit `c` as a free parameter — **does not work**. Only
>   23/498 fitted values (4.6%) are physically plausible water at this site; 41.8% pin at the
>   search-bracket edge. A free scalar `c` absorbs model and association error, not sound speed.
> - Point 4's guess that the seabed may matter more than the water column is **untested** and
>   remains open: mode counts at 20 Hz vary by 2× across a plausible seabed sweep (1550–1900 m/s),
>   but no attenuation or transmission-loss calculation was run.

1. **There is a direct empirical measurement, and it is small.** Rørstadbotnen et al. (2023)
   perturbed `c` by ±25–50 m/s and their localizations moved only **16–31 m** — against a
   validated accuracy of ~100 m (n = 41 GPS-known air-gun shots). Sound speed was not their
   limiting factor.
2. **A uniform `c` error is a scale factor.** It maps almost entirely into range, and is therefore
   absorbable by fitting `c` as a free parameter — which, notably, *nobody currently does*. The
   cheap version of this idea (make `c` free, or set it from a depth-weighted profile) captures
   most of the available benefit without any oceanographic data at all.
3. **The dominant L-1 error is geometric, not acoustic.** A single straight iso-depth cable
   constrains only `(d₀, h)` — the source lies on a **circle of radius `h` about the cable axis**.
   Restricted to an assumed depth, that circle becomes the left–right ambiguity. Horne et al.
   describe their own objective function as having "two complementary solutions located
   symmetrically to the North and South," and resolved it using the *other* cable as a prior.
   Choosing the wrong side costs ≈ 2|y|, i.e. kilometres. That dwarfs a few hundred metres of
   sound-speed bias.
4. **The water column is not where the uncertainty is.** Over the top 1500 m the total sound-speed
   spread is ~16 m/s (≈1.1 %); the *horizontal* variation at fixed depth over 100 × 100 km in
   November is only ~2–4 m/s (≈0.2 %) — so the "spatially varying field" part of the idea is the
   weakest part. Meanwhile at 20 Hz (λ ≈ 74 m) in 100–500 m of water the waveguide is only 1.3–7
   wavelengths deep, so low modes are strongly seabed-interacting and **sediment** sound speed and
   attenuation plausibly perturb effective group speed by more than the water column does — and in
   the opposite direction.

## Why it may be enabling for L-2

Depth is the genuinely open problem. Every published method fixes source depth; verified directly
from the baseline authors' released output (`batch1_localizations_with_coords.csv`, n = 176):
**every row has `z = −30.0 m` exactly.** Their two-cable routine also runs with `fix_z=True`, so
"3D" localization does not currently exist in the literature for DAS.

> [!NOTE]
> **[CORRECTED 2026-09-17].** `z = −30.0` is true of the released **pair-localization CSV** only.
> The same authors' **single-cable** output, in the association HDF5 files, uses **`z = −40.0`**.
> The claim that published DAS localization always fixes depth is unaffected; the specific
> constant is not universal, and any experiment that compares against "the baseline's assumed
> depth" must say which of the two it means.

The reason is conditioning: each cable constrains the source to a circle about its own axis, and
two seafloor cables at *similar* depth make those circles nearly coplanar, so ∂misfit/∂z ≈ 0.
Rørstadbotnen et al. confirmed this operationally — perturbing z over 15–30 m moved the horizontal
solution by **< 1 m**. Depth needs vertical aperture, which two iso-depth cables lack.

**The one available source of vertical aperture on this dataset is bathymetry.** For the general
model `t(d) = t_e + ‖S − C(d)‖ / c`:

```
∂t/∂z_s = (z_s − z_c(d)) / (c · R)
```

This is non-zero only because the channel depth `z_c(d)` varies along the cable — and the Oregon
cables run shelf-to-abyssal across the 12.5–95 km span where calls are annotated. That is one of
very few geometries anywhere with enough along-cable depth diversity to make `z_s` observable at
all. (This mechanism is an inference from the forward model, **not** something any paper has
attempted. Treat as a hypothesis.)

**And that is precisely where constant-`c` fails structurally rather than uniformly.** The
mechanism works by comparing arrival times at channels at radically different depths. A ray to a
1500 m channel traverses the whole profile — ≈1491 m/s at the surface, through the sound-channel
minimum of ≈1477 m/s near 500 m, back up to ≈1484 m/s — while a ray to a 100 m shelf channel never
leaves the upper layer. The path-averaged sound speed therefore differs **systematically between
the very channels whose time differences carry the depth information.**

Consequences:

- Constant-`c` does not merely add noise to `z_s`; it **biases** it, and the bias is confounded
  with the signal.
- Because the effect is *differential across the aperture*, a single free scale parameter cannot
  absorb it — unlike the L-1 case.
- So the sound-speed profile stops being a marginal correction and becomes a prerequisite for the
  one sub-task nobody has solved.

Secondary benefit: Challenge L **requires uncertainty estimates** with every submission. The
baseline's uncertainty is an analytic linearized covariance (`cov = var·(GᵀWG)⁻¹` → 95 % ellipse),
which by construction cannot represent the bimodal left–right ambiguity, contributes zero variance
from sound speed (because `c` is fixed), and has no z-uncertainty at all. Their released
uncertainties: median `deltax` 57 m, p90 576 m, max 3365 m (n = 176, batch 1). An honest error
budget that propagates sound-speed, cable-geometry and pick error into a *multimodal* posterior is
a differentiator independent of whether we win on point accuracy.

## Data availability (checked)

> [!WARNING]
> **[CORRECTED 2026-09-17] — this section's headline claim and its OOI bullet are both wrong.**
> The original text is kept below, struck through, with what was actually measured underneath.
> Measured by direct ERDDAP query for 25 Oct – 15 Nov 2021 (spec §2.4, `soundspeed/erddap.py`'s
> `NOV_2021_STATUS`).

~~The oceanographic side is cheap and confirmed — roughly a day of ingestion work.~~

**What it actually took, and what it bought.** The *fetching* was not the cost. Delivering a
usable profile required: an ERDDAP client with QARTOD flag filtering **and** a guard that refuses
to build a profile from a parked instrument (two of the four candidate datasets were parked, and
would otherwise have produced a near-degenerate profile that failed three modules away); a HYCOM
NCSS client that unpacks the server's raw Int16 `scale_factor`/`add_offset` values — despite
`[unit="degC"]`/`[unit="psu"]` labels the CSV endpoint returns packed integers, and the naive
parse yields temperatures like −6483 °C silently rather than raising; **two** separate profiles at
two sites for two distinct roles (one where in-situ data exists to validate against, one deep
enough to cover the real pick depths); and a validation comparison between them. That is roughly
a day's work, so the estimate was not wrong about effort — **it was wrong about what the effort
would deliver.** The result validates HYCOM over **219–573 m of a 0–1,575 m problem**, and the
profile the experiments actually ran on was never validated at its own location at all. A
defensible profile here needs reanalysis *plus* in-situ validation, not in-situ alone, and still
leaves the upper water column unchecked.

~~**OOI Regional Cabled Array profilers**, Oregon Slope Base (`RS01SBPS`, ~44.52°N −125.39°W,
2900 m): paired shallow + deep CTD profilers give **full-depth 0–2900 m T/S several times per
day**, verified continuous through Nov 2021.~~ **Wrong on every clause.** For Nov 2021:

| Dataset | What was actually there |
|---|---|
| `ooi-rs01sbpd-dp01a-01-ctdpfl104` (Slope Base **deep** profiler) | **No data** — HTTP 404, zero rows. The site code is **`RS01SBPD`**, not `RS01SBPS`. |
| `ooi-rs01sbps-sf01a-2a-ctdpfa102` (Slope Base shallow) | **Parked** at z −195…−189 m: a point, not a profile |
| `ooi-ce04osps-sf01b-2a-ctdpfa107` (Oregon Offshore shallow) | **Parked** at z −197…−192 m |
| `ooi-ce04ospd-dp01b-01-ctdpfl105` (Oregon Offshore **deep** profiler) | **The only profiling instrument.** Raw 129,177 samples, z −575…−97 m |

  Exactly one instrument was profiling, at a different site (**CE04OSPD**) from the one named
  here. Its raw extent is 97–575 m, but **97 m is not the usable minimum**: after QARTOD filtering
  (flag ≤ 2) the usable band is **216–575 m** over a 5-day window (5,067/7,472 rows, 68%) or
  **154–575 m** over three weeks (93,901/129,177 rows, 73%) — the shallow edge is window-dependent,
  so re-check it for whatever window you fetch. **The entire upper water column has no usable
  in-situ measurement**, and neither does anything below 575 m. Access via ERDDAP `tabledap` at
  `erddap.dataexplorer.oceanobservatories.org`, OOI Data Explorer, or the M2M API.
  **Warning (this part was right):** the raw ERDDAP variable ranges are polluted (salinity max
  33929 PSU, temperature max 61 °C) — filter on the QARTOD QC flags or you will build a garbage
  profile.
- **Reanalysis** with Nov 2021 coverage: HYCOM GOFS 3.1 `GLBy0.08/expt_93.0` (1/12°, 40 levels,
  NCSS subsetting) and Copernicus GLORYS12V1 (1/12°, 50 levels, daily).
- **Climatology:** WOA23 monthly 1/4° T/S — probably sufficient on its own for a sensitivity study.
- **Argo:** effectively unavailable here — the site is inside the shelf/slope where floats do not
  park; expect ~0–3 profiles within 200 km for Oct–Dec 2021. Cross-check only.
- **Bathymetry:** NOAA BlueTopo (4–16 m), NCEI Central Oregon Coastal DEM (1/3 arc-sec), GEBCO
  2024 (15 arc-sec). Note the dataset's own `*_DAS_multicoord.csv` already gives per-channel
  lat/lon/depth, so external bathymetry only matters for the water column *above* the cable and
  for constraining plausible source depth.
- **T/S → sound speed:** Chen–Millero–Li (UNESCO), Del Grosso, or Mackenzie. Inter-equation
  disagreement is 0.05–0.5 m/s, three orders of magnitude below the effect being chased, so the
  choice is irrelevant. Use `gsw.sound_speed` (TEOS-10).
- **Regional profile, November** (indicative, ±3 m/s): surface ≈1491, 100 m ≈1480, 200 m ≈1478,
  500 m ≈1477 (sound-channel minimum), 1000 m ≈1479, 1500 m ≈1484 m/s. Post-upwelling transition:
  weaker cross-shelf gradients than summer, no surface duct, downward-refracting to ~500 m.

## How to test it cheaply — decision gate

Do these in order and stop at the first negative result.

> [!NOTE]
> **[OUTCOMES, 2026-09-17]** Steps 1–3 were executed as Plan A; full report in
> [`das-finwhale-soundspeed-plan-a-findings.md`](das-finwhale-soundspeed-plan-a-findings.md).
> Step 1: **not possible** — the released baseline CSV has no call identifier and 0/176 rows join
> unambiguously by timestamp at the default tolerance (at most 21/176 under a timing-isolation
> rule). Step 2: **negative** — a free scalar `c` is a nuisance sink (4.6% of fits physically
> plausible, 41.8% pinned at a bracket edge), not the cheap win this step expected. Step 3:
> **unresolvable** — 9.0–9.5 m median shift, 6.3–6.7% of the pick-noise floor. Steps 4–5 are
> Plan B and were not run; step 4 must find a new justification, because the range-trend
> premise it would have leaned on was falsified (source depth does not explain the trend).

1. **Reproduce the baseline first.** No sound-speed work is interpretable without a reference
   number on the same picks.
2. **Free-`c` ablation (no ocean data needed).** Add `c` as a fifth unknown to the inversion and
   report how much positions move and how residuals change, vs. fixed 1480 m/s. If a free scalar
   `c` already absorbs the effect, the profile work is unnecessary for L-1 — which is the expected
   outcome.
3. **Effective-`c` ablation.** Replace the constant with a per-channel, depth-weighted path-average
   `c` from the OOI Slope Base profiles for 1–5 Nov 2021. Report position deltas against (1) and (2).
4. **Depth-observability test — the actual experiment.** With `z_s` *free*, measure the curvature
   of the misfit in `z` (i.e. is `z_s` identifiable at all?) under (a) constant `c` and (b) the
   depth-dependent profile, restricted to calls whose picks span a wide range of `z_c(d)`. The
   question to answer is not "did accuracy improve" — there is no depth ground truth to score
   against — but **"is `z_s` identifiable, and does the profile change the answer?"**
5. Only if (4) is promising: consider modal/PE propagation (KRAKEN, RAM) and seabed geoacoustics,
   which the physics suggests may matter more than the water column.

Every step must report its denominator: how many calls, from which batches/cables, and what
fraction had picks spanning enough depth diversity to be usable.

## Open questions

- **Is there any depth ground truth at all?** If the challenge's ground-truth positions are the
  baseline's own output, then GT `z` is the fixed −30 m assumption and L-2's Z component may be
  unscoreable. This is the single most important question to ask the organizers.
- Are the denoised SNR envelopes sufficient? Carrier phase is absent, which likely rules out the
  Lloyd's-mirror route to depth (spectral nulls). The raw data is separately public
  (DOI 10.58046/5J60-FJ89) if we need multipath.
- Does the seabed dominate? If sediment-interacting modes perturb effective group speed more than
  the water column, the water-column correction is inside the error bars of a bigger unmodelled
  term.

## Sources

- Goestchel, Q., Wilcock, W. S. D. & Abadi, S. (2026). Automated association of fin whale calls for
  localization using distributed acoustic sensing. *JASA* 160(1), 135–146.
  doi:10.1121/10.0044257. Code: `Ocean-Data-Lab/Goestchel_JASA_2025b` (GPL-3.0).
- Horne, Stork & Stanek (2025). *Front. Mar. Sci.* 12:1603541. doi:10.3389/fmars.2025.1603541.
- Rørstadbotnen et al. (2023). *Front. Mar. Sci.* 10:1130898. doi:10.3389/fmars.2023.1130898.
- Bouffaut et al. (2022). *Front. Mar. Sci.* 9:901348. doi:10.3389/fmars.2022.901348.
- Pascoe, Spiesberger & Mellinger (2024). Unreported large errors in a common method for sound
  source localization of marine mammals. *JASA* 156(6), 3780–3787.
- OOI ERDDAP: `erddap.dataexplorer.oceanobservatories.org`. NCEI bathymetry:
  `ncei.noaa.gov/products/coastal-relief-model`. HYCOM: `hycom.org/dataserver/gofs-3pt1`.

> [!NOTE]
> Numbers attributed to specific papers are as reported in those papers. The
> `∂t/∂z_s` depth-aperture argument and the differential-path-average argument are our own
> inferences from the forward model and are not published; they are the hypotheses this note
> proposes testing, not established results.
