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
2. **Picker (~4h).** For each annotated call, seed with its sparse picks, interpolate the moveout,
   and track the SNR peak channel by channel inside a ± window around it, with sub-sample peak
   interpolation and an SNR gate. Decimate to channels at least one coherence length apart, so the
   picks are independent and χ² is not inflated. Output: dense picks per call, same schema as
   `load_picks`.
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
