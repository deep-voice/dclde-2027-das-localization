# T-017 — A source-position-dependent travel-time correction from repeat calls

*2026-11-07 · amit · status: result on all 5 batches (119 of 166 two-cable calls); first pass was batches 1, 4, 5*

Follows T-005 (`plans/2026-11-07-dense-channel-picking.md`, steps 2–2d). There, the dense-pick
residual under the straight-ray ConstantC(1480), z = −30 m null was dominated by a km-scale smooth
term, ~89 ms per cable under two-cable solves. It repeated for calls from the same position and
not for calls from different positions. T-017 asks whether that term can be **learned** from repeat
calls and **transferred** to a held-out call.

## Method

Script `paper/scripts/t017_position_correction.py`. Outputs:
- `paper/results/t017_b1b4b5_{calls,loo}.csv` (leave one call out);
- `paper/results/t017_b1b4b5_lwo_{calls,loo}.csv` (leave one window out).

1. **Calls.** Every call annotated on both cables in the 14 downloaded window pairs: batch 4 all 12
   windows (2021-11-04 08:00:02–08:09:12), batch 1 02:00:02, batch 5 12:00:02. Each is dense-picked
   on both cables (T-005 picker) and solved on both cables. Of 63 calls, 62 are kept; 1 is dropped
   as a gross misfit (overall σ > 500 ms; it is the 02:00:02/6 call flagged in T-016). No call was
   unsolved or basin-ambiguous.
2. **Clusters.** Single linkage at 1 km on the two-cable position, within a batch. This gives 7
   clusters of 27, 24, 3, 2, 2, 2 and 2 calls.
   - The two large clusters are batch 4 and span 10–11 consecutive windows (about 8 minutes):
     probably two whales calling repeatedly.
   - The 27-call cluster is chained by single linkage: max span 2.73 km, median pair distance
     0.85 km. The 24-call cluster is tight: 0.57 km max, 0.16 km median.
3. **Correction.** Leave one call out. For each cable, take the median of the other members' smooth
   residual curves (per-cable median removed, ~1 km running median), interpolated onto the held-out
   call's stations. Only stations covered by at least one other member are scored (≥ 30 stations).
4. **Scores** on the held-out call:
   - per cable, smooth and total robust σ before vs after subtracting the correction;
   - a re-solve of the held-out call with the corrected times (both cables, position free).
5. **Control.** Repeat step 3 with a correction taken from the nearest other cluster in the same
   batch that is at least 3 km away. Control distances: median 9.3 km, range 3.3–46.7 km. The
   control uses the same recording period, so a time-common artifact (e.g. a clock or processing
   effect per window) would help it as much as the own-cluster correction.
6. **Leakage checks.**
   - HF and LF annotations could duplicate one physical pulse. There are no near-simultaneous calls
     in any cluster: successive calls are ≥ 18.2 s apart (median 21.2 s, n = 35 gaps).
   - A stricter variant (`--leave-window-out`) also drops every call from the held-out call's own
     70 s window.

## Results, all five batches (extension, same day)

After the first pass below, all window pairs with two-cable calls in batches 2 and 3 were
downloaded too: 21 windows, 56 calls. The script was rerun unchanged on every pair on disk (35).
Outputs: `paper/results/t017_all_{calls,loo}.csv`, `t017_all_lwo_*`, and the printed summaries
`t017_all*_summary.txt`.

- **Calls.** 119 two-cable calls: batch 1 7, batch 2 29, batch 3 27, batch 4 54, batch 5 2. Of
  these, 117 are kept; 2 are dropped (1 gross misfit, 1 unsolved or ambiguous). Batches 1 and 5 have
  only one window pair downloaded each, which is why 47 of the 166 two-cable calls are not covered.
- **Clusters.** 12 clusters: 27, 27, 24, 14, 12, 3, 2, 2, 2, 2, 1, 1. The new large clusters:
  - batch 2: 27 calls, 11 windows, 0.58 km span;
  - batch 3: 14 calls, 10 windows, 0.94 km;
  - batch 3: 12 calls, 5 windows, 0.59 km.

| Median over held-out calls | Leave one call out (n = 115) | Leave one window out (n = 107) |
|---|---|---|
| South smooth σ, own cluster | 84.9 → 38.8 ms (lower in 109/115) | 82.8 → 38.2 ms (103/107) |
| North smooth σ, own cluster | 83.6 → 38.9 ms (106/115) | 83.7 → 38.0 ms (98/107) |
| South total σ, own | 103.8 → 57.1 ms | 101.0 → 56.8 ms |
| North total σ, own | 99.1 → 59.9 ms | 100.4 → 57.8 ms |
| Re-solve, overall σ, own | 113.4 → 70.5 ms | 112.3 → 71.4 ms |
| Re-solve position shift, own | 182 m | 185 m |
| South smooth σ, **control** | 84.7 → 120.0 ms (lower in 19/97) | 83.9 → 125.1 ms (17/93) |
| North smooth σ, **control** | 66.9 → 102.7 ms (19/108) | 69.6 → 101.3 ms (18/103) |
| Re-solve, overall σ, **control** | 112.6 → 126.7 ms | 111.1 → 126.1 ms |

Per batch (leave one call out; South and North smooth σ, own cluster; re-solve σ):

| Batch | n | South smooth | North smooth | Re-solve σ | Control, smooth after (S / N) |
|---|---|---|---|---|---|
| 1 | 6 | 113.0 → 37.5 | 54.4 → 34.2 | 153.5 → 65.1 | 86.4 / 110.2 |
| 2 | 27 | 84.0 → 54.9 | 109.6 → 43.6 | 123.3 → 67.8 | 147.3 / 133.7 |
| 3 | 26 | 77.7 → 39.3 | 80.1 → 32.6 | 96.6 → 63.1 | 110.8 / 131.3 |
| 4 | 54 | 91.1 → 34.4 | 68.1 → 35.9 | 107.7 → 74.6 | 136.6 / 84.8 |
| 5 | 2 | 132.1 → 83.7 | 54.0 → 73.6 | 108.3 → 103.9 | no control |

**Batches 2 and 3 were not used to build the method, and they replicate it.** The own-cluster
correction roughly halves the smooth term on both cables; the far-cluster control makes it worse.
Leave one window out gives the same answer (batch 2 South 52.1, batch 3 South 42.4 ms after).
The re-solve position shift is smaller than in the first pass (median 182 m against ~400 m),
because batch 4's chained 27-call cluster weighs less in the pooled median.

## Results, first pass (batches 1, 4, 5)

Medians over held-out calls (all in clusters of ≥ 2):

| | Leave one call out (n = 62) | Leave one window out (n = 54) |
|---|---|---|
| South smooth σ, own-cluster correction | 92.8 → 35.9 ms (lower in 57/62) | 91.1 → 33.8 ms (51/54) |
| North smooth σ, own-cluster correction | 67.7 → 36.7 ms (56/62) | 68.1 → 34.8 ms (48/54) |
| South total σ, own | 104.1 → 51.9 ms | 101.0 → 49.8 ms |
| North total σ, own | 86.7 → 61.1 ms | 87.4 → 58.2 ms |
| Re-solve, overall σ, own | 108.7 → 74.6 ms | 107.7 → 75.7 ms |
| Re-solve position shift, own | 397 m | 416 m |
| South smooth σ, **control** | 91.9 → 130.3 ms (lower in 13/52) | 90.4 → 136.6 ms (11/48) |
| North smooth σ, **control** | 46.1 → 85.7 ms (11/59) | 46.2 → 84.8 ms (10/54) |
| Re-solve, overall σ, **control** | 108.4 → 128.4 ms | 107.7 → 126.9 ms |

The control's "before" values differ from the own-cluster ones because different stations are
covered.

Per cluster (leave one call out), South smooth σ:

| Cluster | South smooth σ | Re-solve overall σ |
|---|---|---|
| 27-call | 116.6 → 49.0 ms | 89.8 → 55.0 ms |
| 24-call | 77.6 → 27.1 ms | 119.3 → 84.7 ms |
| 2- and 3-call clusters in batches 1 and 4 (4 clusters) | from 107–205 ms to 23–46 ms | 108–160 ms to 46–91 ms |

The exception is the 2-call batch-5 cluster: South 132.1 → 83.7 ms, but North gets worse,
54.0 → 73.6 ms.

## What this shows

- **The smooth term is learnable and transfers between repeat calls from the same position.** It
  more than halves (South ~92 → ~35 ms, North ~68 → ~36 ms), also when no call from the same
  window is used.
- **It does not transfer between positions.** A correction from a cluster ≥ 3 km away makes things
  worse in most calls. So the term is not a time-common artifact or a fixed cable property. It is a
  travel-time error that depends on where the source is.
- **It matters for localization.** Re-solving with the own-cluster correction moves positions by a
  median ~400 m. That is the scale of position error the uncorrected straight-ray model can carry
  for these calls; whether the corrected position is closer to the truth cannot be checked without
  ground truth.
- **After correction, the overall residual (~75 ms) is still well above the ~13 ms pick precision.**
  The rough part below ~1 km (~31 ms in step 2c) and the inter-cable offset (T-019) remain.

## Limits

- 117 calls in 12 clusters. About 5 large clusters (probably five whales, each over 4–9 minutes)
  carry most of the result. Batches 1 and 5 are only partly covered (one window pair each).
- One-cable calls are not used, so this says nothing yet about L-1.
- An empirical correction needs repeat calls near a position. The test set is 10 minutes per
  dclde.org, so repeat calls should exist, but a self-calibrating scheme without positions is
  untested. A physical model (bathymetry-aware propagation) that predicts this term would
  generalise; this result gives such a model a target: per-cluster correction curves.

## Next

1. (Done: batches 2 and 3 replicate.) Optionally complete batches 1 and 5 (47 calls).
2. Look at the correction curves themselves: how they vary with source position and along the
   cable, and whether they line up with bathymetry (GMRT grid in `fixtures/`).
3. Iterative self-calibration: solve → cluster → correct → re-solve, scored by held-out residual
   and position stability. That is the form usable on the test set.
