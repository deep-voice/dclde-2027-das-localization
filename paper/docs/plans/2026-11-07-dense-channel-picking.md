# T-005 — Dense-channel picking: does it cut pick noise toward ~12 ms?

*2026-11-07 · amit · status: plan, not started*

## Question

The annotation picks are sparse: a median 14 picks per call per cable, about 1.1 km apart
(`fixtures/das_calls.csv.gz`, 794 call/cable tracks). The `.nc` files hold denoised SNR images on
every channel. Can re-picking each call on the dense channels bring the per-call residual σ (median
97.8 ms over the 271-call Plan B stratum) close to ~12 ms? That is about where a 100 m depth shift
becomes visible (median |Δχ²| 0.787, commit ee89913).

## The risk to test first

Most of the 97.8 ms may not be pick noise. Plan A measured a range-structured trend of 76–82 ms
(`paper/docs/das-finwhale-soundspeed-plan-a-findings.md` §3.2). That figure is a far-bin minus
near-bin difference of pooled mean residuals, not a per-call noise component; its median-based
version is 15–17 ms. Better picks remove random scatter, not a systematic trend. Step 0 measures
the split per call before anything is downloaded.

Dense picks still have a second use even if the noise target fails: they sample the structured
residual every few metres, not every 1.1 km. That is a better probe of the open "76–82 ms
residual" question.

## Steps

0. **Go/no-go on existing picks (~2h, no download).** For each stratum call, split the residual
   under the ConstantC(1480), z = −30 m null into a smooth trend with range (low-order fit) and the
   scatter around it. Report both medians, with n.
   - Scatter ≥ 30 ms: picking is a real lever, so continue.
   - Scatter ≤ 15 ms: picks are already near the floor, so re-frame T-005 as a diagnosis of the
     structured term and tell the team before spending the 2 days.
   **Result (run 2026-11-07, verdict: GO).** All 271 stratum calls, 7,138 picks. The re-solve
   reproduces `gate_moveout.csv` `resid_sigma_ms` exactly (max difference 0.00 ms). Per-call
   robust residual σ, quartiles in ms (picks with |r| ≥ 1 s excluded from the fits, as in Plan A):

   | Residual after removing | all (n = 271) | one cable (115) | two cables (156) |
   |---|---|---|---|
   | nothing (null) | 57.9 / 97.8 / 122.4 | 19.6 / 48.3 / 100.7 | 91.4 / 106.6 / 130.3 |
   | per-cable offset | 57.9 / 89.9 / 117.6 | 19.6 / 48.3 / 100.7 | 83.5 / 102.8 / 129.1 |
   | + linear in range | 54.3 / 76.2 / 97.8 | 21.8 / 53.2 / 84.9 | 70.2 / 84.8 / 103.0 |
   | + quadratic in range | 48.0 / 73.5 / 96.3 | 22.0 / 43.4 / 78.4 | 68.5 / 82.4 / 99.4 |

   - Smooth structure takes the median from 97.8 to 73.5 ms. The ~74 ms left is above the 30 ms
     bar, so picking may still be a lever. Only 10 of 271 calls are at ≤ 15 ms and 41 at ≤ 30 ms.
   - The leftover is weakly correlated between neighbouring picks: median lag-1 autocorrelation
     along the cable 0.30 (0.38 two-cable, 0.12 one-cable).
   - It barely depends on pick SNR. Per-pick robust σ by SNR quartile (1,780 picks each):
     80.0 / 74.4 / 72.1 / 72.6 ms at median SNR 9.1 / 12.8 / 16.1 / 19.8. Timing jitter from a
     peak picker should fall roughly as 1/SNR, so this is **not** mainly SNR-limited jitter.
   - It is not a fixed channel static or cable-position error. Per-cable-bin medians estimated on
     half the calls cut held-out σ by only ~3 ms (74.8 → 71.8 ms, 2 km bins, 20 random splits),
     and make it worse across batches (81.2 → 88.2 ms, batches 1–2 vs 3–5).
   - One-cable calls look quieter because a one-cable fit has more freedom to absorb error, not
     because their picks are better.

   **What this means.** The ~74 ms is call-specific and does not shrink at higher SNR. That fits
   two explanations: inconsistent pick definitions in the annotations (e.g. a different point on
   the ~1 s pulse envelope from channel to channel), which a consistent dense re-picker would
   fix; or call- and position-dependent propagation error, which it would not. Step 3a (dense picks
   against annotation picks at the same channels) separates the two, so it moves ahead of the full
   run. Script: `paper/scripts/t005_step0_residual_split.py`; output: `paper/results/t005_step0_residual_split.csv` (per call) and `t005_step0_perpick.csv` (per pick).

1. **Read one window (~1h).** Fetch one `.nc` (0.5–1 GB) with `scripts/fetch_sample.py`, add
   `xarray` + `h5netcdf` to the environment, and record dims, time step, channel spacing and the
   variables (`SNR_lf`, `SNR_hf`). Time step and channel spacing set the best achievable pick
   precision.

   **Result (2026-11-07).** File: `detections/batch_1/Denoised_SNR_South_2021-11-04_02-00-02.nc`,
   1.01 GB, chosen because it holds 9 stratum calls (157 of its 166 picks). Downloaded with
   `aws s3 cp` from our mirror into `data/_samples/` (gitignored).
   - Variables: `SNR_hf` and `SNR_lf`, float64 on dims (dist 20,325 × time 14,000); about 5 GB in
     memory, so read rows lazily. The `utc_time` coordinate is on `time`.
   - Time: fs = 200 Hz (5 ms step), 0–69.995 s.
   - Distance: 11,998–94,998 m along the cable, every 4.084 m (every 2nd channel of dx = 2.042 m;
     `selected_channels` 5876–46525 step 2). Gauge length 51.0 m, so channels closer than ~51 m are
     not independent; decimate picks to ≥ 51 m before any χ².
   - Other attributes: `n` = 1.4682 (fibre refractive index), `scale_factor` 2.02e-13,
     `fileBeginTimeUTC` 2021-11-04_02:00:02.

   **Early look at step 3a (one window, n = 166 picks, 10 calls; not a result).** At each
   annotation pick's nearest channel (within ±2 m), the SNR peak within ±2 s of the pick:
   - Annotation picks sit on the SNR envelope peak: median peak − pick = −0.9 ms. The CSV `snr`
     column matches the image value at the pick (median 15.7 vs 16.0).
   - The envelope is broad: median FWHM 350 ms (IQR 295–420 ms).
   - Peak − pick scatter is a robust σ of 33.6 ms (plain sd 65 ms, tail to +456 ms). Per call the
     spread ranges from 22 to 115 ms.
   - So the annotations look like peak picks on a ~350 ms-wide envelope. Taking the peak of a broad,
     noisy envelope is a jittery estimator, and it would not get much better at higher SNR when the
     envelope's shape varies from channel to channel. That fits step 0. **Implication for step 2:**
     pick by cross-correlating the envelope shape across neighbouring channels (stack over the
     ~51 m gauge, align, sub-sample interpolate), not by per-channel peak.

2. **Picker (~4h).** For each annotated call, seed with its sparse picks, interpolate the moveout,
   and track the SNR peak channel by channel inside a ± window around it, with sub-sample peak
   interpolation and an SNR gate. Decimate to channels at least one coherence length apart, so the
   picks are independent and χ² is not inflated. Output: dense picks per call, same schema as
   `load_picks`.

   **Built (2026-11-07):** `src/dclde_localization/physics/dense_picks.py` uses envelope
   cross-correlation, per the step-1 finding, not per-channel peaks. Stations sit every 51 m inside
   the annotated span only; each station is the mean of its channels. Each station is aligned to an
   iterated template with sub-sample parabolic refinement. The median lag is removed each
   iteration, so the picks keep the annotations' timing convention. A station is kept if
   CC ≥ 0.6 and its lag is within 0.15 s of the running median. No propagation model is used.
   Tests: `tests/physics/test_dense_picks.py`, 5 tests, including a synthetic chevron where 35 ms of
   seed jitter is recovered to < 5 ms. That covers step 3b on synthetic noise, not on real noise.
   The full suite passes (94).

   **First run, one window (2026-11-07; n = 10 calls, South only, one-cable solves; not a result).**
   Script `paper/scripts/t005_step2_pick_window.py`, output
   `paper/results/t005_step2_south_2021-11-04_02-00-02_{calls,picks}.csv`. Medians over the 10 calls:

   | Pick set (same ConstantC(1480), z = −30 m null) | residual σ |
   |---|---|
   | annotation picks | 85.2 ms |
   | dense picks at the same stations (count and geometry fixed) | 58.8 ms |
   | all kept dense stations (median 323 per call, 87% kept, median CC 0.9) | 92.0 ms |

   The dense residual along the cable, split three ways:

   | Part | σ |
   |---|---|
   | station-to-station noise, from neighbours 51 m apart | 12.2 ms |
   | rough part (minus a ~1 km running median) | 25.3 ms |
   | smooth part at km scale | 67.8 ms |

   The smooth part's 5–95% excursion along the cable has a median of 316 ms.

   - The picker is precise. The neighbour estimate (~12 ms, a lower bound because adjacent stations
     share part of the gauge) is at the ~12 ms level the task asked about. At the annotation
     stations, swapping in dense picks lowers residual σ in 8 of 10 calls.
   - **But the residual is dominated by km-scale smooth structure (~68 ms), not pick noise.** The
     sparse annotation picks (1.1 km apart) sample that structure as if it were scatter. That is
     why step 0 saw no SNR dependence and why using every dense station raises σ (92 ms) instead of
     lowering it.
   - Two candidate causes, not yet separated: (i) model error that varies smoothly along the
     cable (propagation, bathymetry, or cable position at km scale), or (ii) a picker bias that
     drifts smoothly as the envelope shape changes along the cable (e.g. multipath). Step 0 already
     ruled out a fixed per-channel static (split-half test), so (i) would have to depend on the
     source position.
   - If (i), then on this window better picks cannot bring χ² near the 13 ms depth signal: the
     ~68 ms model term needs modelling first. That reframes T-005 toward the open "76–82 ms
     residual" question.
   - The bar registered above (median σ ≤ 30 ms) is **not met** on this window: 58.8 ms same
     stations, 92.0 ms all stations. One window, one cable, 10 calls; not offered as the step-4
     result.

   **2b. What the smooth term is (2026-11-07, same window, same 10 calls).** Script
   `paper/scripts/t005_step2b_smooth_term.py`, output
   `paper/results/t005_step2b_south_2021-11-04_02-00-02_{calls,stations,pairs}.csv`.

   *Test 1: does it depend on how the arrival is defined?* At every kept station the arrival was
   measured four ways on the same trace, and each set solved under the same null. Median σ of the
   ~1 km smooth residual:

   | xcorr | peak | centroid | onset (50% leading edge) |
   |---|---|---|---|
   | 67.8 ms | 75.6 ms | 69.5 ms | 99.7 ms |

   - The xcorr and onset smooth curves have the same shape (median r = 0.97 over the 10 calls).
   - xcorr minus onset varies smoothly by only ~22 ms. It tracks envelope width (r = 0.94), partly
     by construction, since a half-height onset moves with width.
   - So envelope-shape drift exists but is a minor part. The smooth term is in the arrival time
     itself, whatever point of the envelope is picked.

   *Test 2: does it repeat for calls from the same place?* The 10 calls fall in three position
   clusters (solved positions within ~0.3 km; probably repeat calls from three whales). Smooth
   residual curves, compared over ≥ 3 km of shared cable:

   | Pairs | n | correlation, median (range) |
   |---|---|---|
   | sources < 1 km apart | 9 | 0.86 (0.78 to 0.94) |
   | sources 3.2–6.7 km apart | 13 | 0.03 (−0.24 to 0.19) |

   - The pattern is reproducible call to call from the same place, and different from a different
     place on the same stretch of cable.
   - That rules out random picker noise, and (with step 0's split-half test) a fixed per-channel
     static.

   **Conclusion (one window; to confirm on more).** The ~68 ms smooth residual is a deterministic,
   source-position-dependent travel-time error of the straight-ray ConstantC(1480) model, at km
   scale along the cable. Candidates are propagation the model lacks (bathymetry-driven paths,
   refraction) or km-scale cable-position error that only shows up differently per source
   azimuth. Consequences:
   - Better picks alone cannot reach the ~13 ms depth signal on this data; the model term must be
     reduced first.
   - Per-call χ² that treats this term as white noise is wrong in a specific way: it is correlated
     along the cable and repeats across calls from one position.
   - It gives a handle the sparse picks never had: repeat calls from one position give replicate
     residual curves, so a position-dependent correction can be estimated and cross-validated.
   - Caveat: cluster 1 (calls 1, 3, 15, near x = 37 km) has very large smooth curves (rms
     335–434 ms over 32–36 km). Check that cluster for a wrong basin or far-range breakdown before
     generalising. **Checked:** their dense-pick solutions have two basins 6.9–7.9 km apart, beyond
     Plan B's 500 m flip threshold. So their positions are unreliable, and they are excluded below.

   **2c. Confirmed on 10 windows (2026-11-07).** Windows: South 2021-11-04 02:00:02, 08:09:12,
   12:00:02, 16:09:22 and 2021-11-03 22:17:22; North 2021-11-04 02:00:02, 02:05:02, 08:09:12,
   12:00:02, 16:11:52. That covers all 5 batches and both cables, and includes two North/South
   pairs from the same minutes. Each window gets a one-cable solve. Calls are excluded when their
   dense-pick solution is basin-ambiguous (two basins > 500 m apart) or unsolved: 31 of 94.
   Script `paper/scripts/t005_step2c_pool.py`, output `paper/results/t005_step2c_pooled_*.csv`
   (per-window outputs `t005_step2[b]_<cable>_<window>_*.csv`).

   | Pooled, unambiguous calls | median |
   |---|---|
   | residual σ, annotation picks → dense picks at the same stations (n = 56) | 70.5 → 53.9 ms; lower in 42 of 56 |
   | residual σ, all kept dense stations | 85.3 ms |
   | station-to-station noise (lower bound) | 13.0 ms |
   | rough part (below ~1 km) | 30.9 ms |
   | smooth part (km scale), xcorr / peak / centroid / onset (n = 63) | 61.2 / 72.6 / 68.6 / 75.7 ms |
   | shape agreement of xcorr and onset smooth curves | r = 0.91 |
   | xcorr − onset drift (envelope shape) | 31.5 ms |

   Smooth-curve correlation between call pairs, by source separation:

   | Separation | pairs | windows | r, median (IQR) |
   |---|---|---|---|
   | < 1 km | 38 | 8 | 0.77 (0.59 to 0.86) |
   | 1–3 km | 6 | 3 | 0.07 (−0.33 to 0.46) |
   | ≥ 3 km | 90 | 9 | 0.04 (−0.16 to 0.44) |

   Per window, the smooth part ranges from 32 to 107 ms and the neighbour noise from 11.5 to
   14.6 ms. Every window shows the same picture.

   **Conclusion.** On 63 calls across 10 windows, both cables and all 5 batches, the one-window
   result holds:
   - Pick precision is ~13 ms between neighbouring stations.
   - The residual is dominated by a ~61 ms km-scale travel-time error that is present whatever
     point of the envelope is picked.
   - That error repeats for calls from the same position (r ≈ 0.77) and not for calls ≥ 1 km
     apart (r ≈ 0.05).

   This is a source-position-dependent model error of the straight-ray ConstantC(1480) null.
   T-005's question is answered:
   - Dense picking does reach ~13 ms pick precision.
   - It cannot make depth visible until this model term is reduced.
   - The registered 30 ms bar is not met (53.9 ms same stations, 85.3 ms all stations), and the
     reason is model error, not picking.

   Not yet done: two-cable solves on the paired windows, and the full 271-call stratum (step 4).
   One-cable solves are basin-ambiguous for a third of calls, which is why 31 were excluded.

3. **Validate the picker (~2h).** (a) At the annotated channels, compare dense picks with the
   annotation picks; the spread of that difference bounds annotation error. (b) Inject a synthetic
   chevron of known moveout into a noise-only stretch and measure recovery error.
4. **Measure (~3h).** Re-solve the 271 stratum calls with the physics core
   (`src/dclde_localization/physics/`). Report the per-call residual σ distribution against
   57.9 / 97.8 / 122.4 ms, rerun the depth-signal check (ee89913), and count calls reaching 2σ
   against the current 32.
5. **Write up.** Findings doc under `paper/docs/`, log entry, T-005 → review.

**Register the gate before step 4:** success = median residual σ ≤ 30 ms over the calls the picker
handles, with the denominator stated. Depth-visible = σ ≤ 15 ms. Both numbers are proposals until
amit signs them off.

## Data and compute

- `s3://deepvoice-datasets/dclde_2027_das_finwhale_localization/detections/`: 120 files, 96.0 GiB.
  Start with the batch 1 windows that hold stratum calls. Run the full set on the team box, close to
  S3, not on a laptop.

## Does it depend on the organizers?

No. The comparison uses our own residuals, not the 176 baseline rows, so the missing call-id key
does not matter. The open question is only its value to the challenge: if the L-2 test set comes
with picks, this helps the paper and not the entry (T-002 email, T-015).
