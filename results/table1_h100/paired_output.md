## Per-example paired tests of Table 1 (plan 2.13)

Rebuild: all 60 runs give their reported score (largest gap 0.0047 points). Units: test examples (AIME: problems, mean of 5 tries); each method = mean over its 3 seeds. d = mean of A − B in points, 95 % percentile bootstrap over units (10,000 resamples), p = paired sign-flip test (10,000 sign vectors), Holm over the 28 tests. Ours per seed = mean ± SD of the three per-seed differences of the reported scores.

| Benchmark | A − B | Paper | Ours per seed | d | 95 % interval | units A better / B better | p | p (Holm) | Holm at 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | GEPA - Baseline | +20.00 | +21.67 ± 2.03 | +21.67 | [+17.33, +26.11] | 112 / 16 | 0.0001 | 0.0028 | rejects |
| HotpotQA | GEPA - MIPROv2 | +7.00 | +2.11 ± 4.62 | +2.11 | [-1.00, +5.11] | 54 / 40 | 0.1947 | 1.0000 | no |
| HotpotQA | GEPA+Merge - GEPA | +2.00 | -2.89 ± 1.58 | -2.89 | [-6.00, +0.33] | 37 / 54 | 0.0781 | 1.0000 | no |
| HotpotQA | GEPA+Merge - Baseline | +22.00 | +18.78 ± 2.27 | +18.78 | [+15.00, +22.89] | 102 / 11 | 0.0001 | 0.0028 | rejects |
| HotpotQA | GEPA+Merge - MIPROv2 | +9.00 | -0.77 ± 4.86 | -0.78 | [-3.56, +2.11] | 40 / 46 | 0.6531 | 1.0000 | no |
| HotpotQA | MIPROv2 - Baseline | +13.00 | +19.55 ± 2.67 | +19.56 | [+15.11, +24.11] | 107 / 22 | 0.0001 | 0.0028 | rejects |
| HoVer | GEPA - Baseline | +17.00 | +13.89 ± 3.07 | +13.89 | [+10.67, +17.22] | 99 / 13 | 0.0001 | 0.0028 | rejects |
| HoVer | GEPA - MIPROv2 | +5.00 | +5.22 ± 2.53 | +5.22 | [+2.78, +7.78] | 62 / 25 | 0.0002 | 0.0044 | rejects |
| HoVer | GEPA+Merge - GEPA | -0.66 | -0.67 ± 1.15 | -0.67 | [-3.33, +1.89] | 38 / 46 | 0.6864 | 1.0000 | no |
| HoVer | GEPA+Merge - Baseline | +16.34 | +13.22 ± 2.69 | +13.22 | [+9.89, +16.67] | 95 / 19 | 0.0001 | 0.0028 | rejects |
| HoVer | GEPA+Merge - MIPROv2 | +4.34 | +4.56 ± 2.04 | +4.56 | [+1.67, +7.44] | 53 / 25 | 0.0019 | 0.0380 | rejects |
| HoVer | MIPROv2 - Baseline | +12.00 | +8.67 ± 4.67 | +8.67 | [+5.56, +11.78] | 79 / 29 | 0.0001 | 0.0028 | rejects |
| IFBench | GEPA - Baseline | +1.71 | -1.64 ± 1.19 | -1.64 | [-5.16, +1.87] | 60 / 61 | 0.3723 | 1.0000 | no |
| IFBench | GEPA - MIPROv2 | +2.39 | -6.30 ± 4.19 | -6.29 | [-9.58, -3.00] | 46 / 84 | 0.0003 | 0.0063 | rejects |
| IFBench | GEPA+Merge - GEPA | -10.38 | +2.10 ± 3.80 | +2.10 | [-1.19, +5.39] | 74 / 63 | 0.2226 | 1.0000 | no |
| IFBench | GEPA+Merge - Baseline | -8.67 | +0.45 ± 2.67 | +0.45 | [-3.00, +4.08] | 63 / 66 | 0.8233 | 1.0000 | no |
| IFBench | GEPA+Merge - MIPROv2 | -7.99 | -4.20 ± 4.96 | -4.20 | [-7.26, -1.02] | 53 / 79 | 0.0094 | 0.1692 | no |
| IFBench | MIPROv2 - Baseline | -0.68 | +4.65 ± 3.77 | +4.65 | [+1.53, +7.88] | 71 / 43 | 0.0056 | 0.1064 | no |
| AIME | GEPA - Baseline | +4.67 | +6.89 ± 4.24 | +6.89 | [-0.44, +15.11] | 12 / 6 | 0.1024 | 1.0000 | no |
| AIME | GEPA - MIPROv2 | +12.00 | +0.45 ± 9.53 | +0.44 | [-10.00, +10.22] | 14 / 7 | 0.9683 | 1.0000 | no |
| AIME | GEPA+Merge - GEPA | +0.00 | -1.34 ± 7.69 | -1.33 | [-11.78, +8.22] | 9 / 10 | same search twice | not tested | – |
| AIME | GEPA+Merge - Baseline | +4.67 | +5.56 ± 5.75 | +5.56 | [-1.33, +13.33] | 9 / 6 | 0.1716 | 1.0000 | no |
| AIME | GEPA+Merge - MIPROv2 | +12.00 | -0.89 ± 16.60 | -0.89 | [-11.11, +9.56] | 9 / 10 | 0.9057 | 1.0000 | no |
| AIME | MIPROv2 - Baseline | -7.33 | +6.45 ± 11.19 | +6.44 | [-1.11, +14.89] | 11 / 6 | 0.1449 | 1.0000 | no |
| LiveBench | GEPA - Baseline | +3.25 | -1.56 ± 1.22 | -1.56 | [-6.43, +3.49] | 24 / 33 | 0.5371 | 1.0000 | no |
| LiveBench | GEPA - MIPROv2 | +5.35 | +1.02 ± 3.58 | +1.02 | [-3.52, +5.59] | 28 / 31 | 0.6542 | 1.0000 | no |
| LiveBench | GEPA+Merge - GEPA | +0.00 | +0.35 ± 6.16 | +0.35 | [-4.84, +5.38] | 34 / 28 | same search twice | not tested | – |
| LiveBench | GEPA+Merge - Baseline | +3.25 | -1.21 ± 7.12 | -1.21 | [-6.01, +3.47] | 35 / 31 | 0.6133 | 1.0000 | no |
| LiveBench | GEPA+Merge - MIPROv2 | +5.35 | +1.37 ± 9.73 | +1.37 | [-3.03, +5.59] | 36 / 28 | 0.5293 | 1.0000 | no |
| LiveBench | MIPROv2 - Baseline | -2.10 | -2.58 ± 2.68 | -2.58 | [-6.89, +1.66] | 25 / 30 | 0.2381 | 1.0000 | no |

## Aggregate over the five benchmarks (mean of the benchmark differences, no test)

| A − B | Paper (5 benchmarks, no PUPA) | d | 95 % interval |
|---|---|---|---|
| GEPA − Baseline | +9.33 | +7.85 | [+5.70, +10.10] |
| GEPA − MIPROv2 | +6.35 | +0.50 | [-2.06, +2.90] |
| GEPA+Merge − GEPA | -1.81 | -0.49 | [-3.06, +1.87] |
| GEPA+Merge − Baseline | +7.52 | +7.36 | [+5.19, +9.53] |
| GEPA+Merge − MIPROv2 | +4.54 | +0.01 | [-2.42, +2.48] |
| MIPROv2 − Baseline | +2.98 | +7.35 | [+5.19, +9.65] |

## Without the † runs (descriptive, no test; only the rows that contain a † run)

| Benchmark | A − B | seeds of A / of B | d | 95 % interval | d with all runs |
|---|---|---|---|---|---|
| HotpotQA | GEPA+Merge − GEPA | [0, 1] / [0, 1, 2] | -2.61 | [-6.00, +0.78] | -2.89 |
| HotpotQA | GEPA+Merge − Baseline | [0, 1] / [0, 1, 2] | +19.06 | [+15.06, +23.33] | +18.78 |
| HotpotQA | GEPA+Merge − MIPROv2 | [0, 1] / [0, 1, 2] | -0.50 | [-3.56, +2.44] | -0.78 |
| HoVer | GEPA − Baseline | [2] / [0, 1, 2] | +16.22 | [+12.11, +20.44] | +13.89 |
| HoVer | GEPA − MIPROv2 | [2] / [0, 1, 2] | +7.56 | [+3.78, +11.22] | +5.22 |
| HoVer | GEPA+Merge − GEPA | [] / [2] | no run left | | -0.67 |
| HoVer | GEPA+Merge − Baseline | [] / [0, 1, 2] | no run left | | +13.22 |
| HoVer | GEPA+Merge − MIPROv2 | [] / [0, 1, 2] | no run left | | +4.56 |
| IFBench | GEPA − Baseline | [0, 1] / [0, 1, 2] | -2.15 | [-6.07, +1.81] | -1.64 |
| IFBench | GEPA − MIPROv2 | [0, 1] / [0, 1, 2] | -6.80 | [-10.54, -3.15] | -6.29 |
| IFBench | GEPA+Merge − GEPA | [0, 1] / [0, 1] | +4.17 | [+0.17, +8.25] | +2.10 |
| IFBench | GEPA+Merge − Baseline | [0, 1] / [0, 1, 2] | +2.01 | [-1.59, +5.75] | +0.45 |
| IFBench | GEPA+Merge − MIPROv2 | [0, 1] / [0, 1, 2] | -2.64 | [-6.07, +0.85] | -4.20 |

