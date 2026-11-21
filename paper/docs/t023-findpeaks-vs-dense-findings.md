# T-023 — The baseline find_peaks picker against the T-005 dense picker

*2026-11-21 · amit · status: result on 119 two-cable calls (all 35 window pairs on disk, 5 batches)*

The test set ships the denoised SNR images without picks (Goestchel, email 2026-11-09), so
picking is part of the challenge entry. This compares the baseline's picker with ours on the same
calls.

## The baseline picker, reproduced exactly

Goestchel's picker (`main_section5.py` in Goestchel_JASA_2025) runs, on every channel of the
denoised SNR image, `scipy.signal.find_peaks(row, distance=ipi*fs, height=th)`. The published
example picks (Goestchel_JASA_2025b/data/detections) use ipi = 3 s, th = 4 dB (North) and 5 dB
(South). Run on our copy of the challenge files for 2021-11-04 02:00:02, this gives **exactly** his
HF picks: 101239 of 101239 North, 110907 of 110907 South, with identical SNR values. So we can
produce his picks for any window ourselves.

Two side facts from his code:
- The SNR images are Gaussian-blurred with a (9, 73) kernel: about 8.5 ms in time and about 46 m
  (sigma) along the cable. Stations 51 m apart are therefore not independent, and the
  "neighbour noise" of T-005 (~13 ms) is a lower bound for any picker on these files.
- The annotation picks are **not** find_peaks peaks: only 8–9% of them (median per call) sit
  within one sample of one.

## Method

Script `paper/scripts/t023_findpeaks_vs_dense.py`; output `paper/results/t023_all_calls.csv` and
`t023_all_summary.txt`. For each call annotated on both cables:
1. Station grid every 51 m inside the annotated span; seed = interpolation of the annotation picks
   (same grid and seed as the dense picker).
2. find_peaks picks: per channel, the peak nearest to the seed within ±0.4 s; per station, the
   median over its channels, kept if at least half of them have a peak.
3. Dense picks: the T-005 picker, kept stations only.
4. Two-cable solves under the ConstantC(1480), z = −30 m null, with each picker's picks, on each
   picker's own stations and on the stations both keep ("common"). Residuals are split, per cable,
   into smooth (~1 km running median) and rough (the rest).

## Results (medians over calls)

| | Dense (T-005) | find_peaks (baseline) |
|---|---|---|
| Stations kept, North / South | 316 / 382 | 307 / 362 |
| Two-cable solve σ, common stations (n = 118) | **106.6 ms** | 130.2 ms |
| — smooth (km-scale) part | **78.9 ms** | 101.1 ms |
| — rough (sub-km) part | 26.4 ms | **22.6 ms** |
| N–S offset under a shared t0 (median; median \|·\|) | +22.2 ms; 53.9 ms | +37.7 ms; 80.9 ms |
| Two-cable solve σ, own stations | 113.5 ms (n = 117) | 137.6 ms (n = 118) |

- Total σ is lower with dense picks in 106 of 118 calls; the rough part is lower with find_peaks
  in 103 of 118.
- By call type (common stations): HF 99.3 vs 125.3 ms, LF 110.8 vs 148.1 ms.
- The two pickers' positions differ by a median of 78 m (90th percentile 486 m) on common
  stations, 118 m on own stations.
- The dense two-cable σ (113.5 ms) matches T-017's 113.4 ms on the same calls.

## What this shows

- **Our picker beats the baseline's on what matters for the solve**: about 24 ms lower residual σ
  on the same stations, in 90% of calls. The gain is entirely in the km-scale part.
- **find_peaks is slightly less noisy at short scale** (22.6 vs 26.4 ms rough). The peak of a blurred
  envelope is locally smooth; the shape alignment adds a little jitter.
- **Part of the "model error" is picker-dependent.** The smooth term is 79 ms with dense picks and
  101 ms with find_peaks. One possible mechanism, not tested: along the cable the envelope peak
  moves relative to the pulse onset (changing pulse shape, overlapping notes or multipath), or
  find_peaks takes a neighbouring peak at some stations. So the T-017 correction target depends on
  the picker.
- **The North–South offset (T-019) is picker-dependent too**: +22 ms with dense picks, +38 ms with
  find_peaks. Some of it is picking convention, not clock or model.

## Limits

- Both pickers start from the annotation seed (±0.4 s). On the test set there are no annotations,
  so the seed must come from detection and association (Goestchel's association code is public:
  `dw.assoc` in DAS4Whales, driven by `main_section3c.py` in Goestchel_JASA_2025b; its outputs for
  all five batches are on Zenodo, record 21220346, 3.4 GB, not downloaded).
- Only two-cable calls; no ground truth, so "better" here means a smaller residual under the null
  model, not a smaller position error.

## Next

1. A hybrid: find_peaks-like local precision with the dense picker's shape alignment, or a
   smoothness prior on the dense lags. The rough-part gap is ~4 ms, so the gain is small.
2. Seed the dense picker from find_peaks plus association instead of annotations, which is the
   form usable on the test set.
3. Download the Zenodo associations to reproduce the baseline end to end, and retry the
   baseline-to-annotation match (T-009) by pick-set overlap.
