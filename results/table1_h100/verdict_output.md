## Our runs per cell: mean ± SD over seeds (n) [paper]

| Benchmark | Baseline | MIPROv2-Heavy | GEPA | GEPA-MERGE |
|---|---|---|---|---|
| HotpotQA | 38.78 ± 0.69 (3) [42.33] | 58.33 ± 2.00 (3) [55.33] | 60.44 ± 2.72 (3) [62.33] | 57.56 ± 2.88 (3) [64.33] |
| HoVer | 37.11 ± 1.58 (3) [35.33] | 45.78 ± 3.21 (3) [47.33] | 51.00 ± 2.33 (3) [52.33] | 50.33 ± 1.45 (3) [51.67] |
| IFBench | 34.63 ± 0.52 (3) [36.90] | 39.29 ± 3.26 (3) [36.22] | 32.99 ± 1.48 (3) [38.61] | 35.09 ± 2.70 (3) [28.23] |
| AIME | 27.78 ± 4.24 (3) [27.33] | 34.22 ± 9.53 (3) [20.00] | 34.67 ± 0.00 (3) [32.00] | 33.33 ± 7.69 (3) [32.00] |
| LiveBench | 51.48 ± 0.41 (3) [48.70] | 48.90 ± 3.08 (3) [46.60] | 49.92 ± 0.91 (3) [51.95] | 50.27 ± 6.71 (3) [51.95] |

## Runs: score (failed examples)

- HotpotQA Baseline: seed 0 = 39.33 (0); seed 1 = 38.00 (0); seed 2 = 39.00 (1)
- HotpotQA MIPROv2-Heavy: seed 0 = 56.33 (1); seed 1 = 60.33 (0); seed 2 = 58.33 (1)
- HotpotQA GEPA: seed 0 = 62.33 (1); seed 1 = 57.33 (0); seed 2 = 61.67 (6)
- HotpotQA GEPA-MERGE: seed 0 = 60.67 (1); seed 1 = 55.00 (1); seed 2 = 57.00 (6)
- HoVer Baseline: seed 0 = 38.33 (0); seed 1 = 35.33 (0); seed 2 = 37.67 (0)
- HoVer MIPROv2-Heavy: seed 0 = 42.33 (0); seed 1 = 48.67 (0); seed 2 = 46.33 (0)
- HoVer GEPA: seed 0 = 48.67 (0); seed 1 = 51.00 (0); seed 2 = 53.33 (0)
- HoVer GEPA-MERGE: seed 0 = 48.67 (0); seed 1 = 51.00 (0); seed 2 = 51.33 (0)
- IFBench Baseline: seed 0 = 35.20 (23); seed 1 = 34.18 (23); seed 2 = 34.52 (?)
- IFBench MIPROv2-Heavy: seed 0 = 35.54 (29); seed 1 = 41.50 (15); seed 2 = 40.82 (21)
- IFBench GEPA: seed 0 = 33.67 (24); seed 1 = 31.29 (19); seed 2 = 34.01 (19)
- IFBench GEPA-MERGE: seed 0 = 36.56 (20); seed 1 = 36.73 (17); seed 2 = 31.97 (45)
- AIME Baseline: seed 0 = 25.33 (40); seed 1 = 25.33 (21); seed 2 = 32.67 (28)
- AIME MIPROv2-Heavy: seed 0 = 44.67 (42); seed 1 = 26.00 (49); seed 2 = 32.00 (32)
- AIME GEPA: seed 0 = 34.67 (19); seed 1 = 34.67 (33); seed 2 = 34.67 (35)
- AIME GEPA-MERGE: seed 0 = 24.67 (25); seed 1 = 36.00 (29); seed 2 = 39.33 (21)
- LiveBench Baseline: seed 0 = 51.31 (14); seed 1 = 51.95 (15); seed 2 = 51.18 (19)
- LiveBench MIPROv2-Heavy: seed 0 = 48.48 (17); seed 1 = 52.16 (9); seed 2 = 46.05 (13)
- LiveBench GEPA: seed 0 = 50.91 (21); seed 1 = 49.11 (17); seed 2 = 49.73 (17)
- LiveBench GEPA-MERGE: seed 0 = 52.94 (13); seed 1 = 42.63 (10); seed 2 = 55.23 (13)

## Verdict per target: d = our mean − paper, 95 % interval of d, Holm over the 18 targets at 0.05

| Target | Paper | Ours (n) | d | 95 % interval | SD used | z or t | p (unadjusted) | Verdict |
|---|---|---|---|---|---|---|---|---|
| HotpotQA Baseline | 42.33 | 38.78 (3) | -3.55 | [-6.90, -0.20] | SD_samp 1.48 | -2.08 | 0.038 | consistent (no evidence of a gap) |
| HotpotQA MIPROv2 | 55.33 | 58.33 (3) | +3.00 | [-6.68, +12.68] | max(SD_run 3.54, s_pooled 2.56) | +0.73 | 0.487 | consistent (no evidence of a gap) |
| HotpotQA GEPA | 62.33 | 60.44 (3) | -1.89 | [-11.53, +7.76] | max(SD_run 3.53, s_pooled 2.56) | -0.46 | 0.658 | consistent (no evidence of a gap) |
| HotpotQA GEPA+Merge | 64.33 | 57.56 (3) | -6.77 | [-16.39, +2.85] | max(SD_run 3.52, s_pooled 2.56) | -1.66 | 0.140 | consistent (no evidence of a gap) |
| HoVer Baseline | 35.33 | 37.11 (3) | +1.78 | [-1.86, +5.42] | SD_samp 1.61 | +0.96 | 0.338 | consistent (no evidence of a gap) |
| HoVer MIPROv2 | 47.33 | 45.78 (3) | -1.55 | [-11.48, +8.38] | max(SD_run 3.64, s_pooled 2.44) | -0.37 | 0.722 | consistent (no evidence of a gap) |
| HoVer GEPA | 52.33 | 51.00 (3) | -1.33 | [-11.26, +8.60] | max(SD_run 3.64, s_pooled 2.44) | -0.32 | 0.761 | consistent (no evidence of a gap) |
| HoVer GEPA+Merge | 51.67 | 50.33 (3) | -1.34 | [-11.27, +8.59] | max(SD_run 3.64, s_pooled 2.44) | -0.32 | 0.760 | consistent (no evidence of a gap) |
| IFBench Baseline | 36.90 | 34.63 (3) | -2.27 | [-5.96, +1.42] | SD_samp 1.63 | -1.20 | 0.228 | consistent (no evidence of a gap) |
| IFBench MIPROv2 | 36.22 | 39.29 (3) | +3.07 | [-6.79, +12.92] | max(SD_run 3.61, s_pooled 2.59) | +0.74 | 0.486 | consistent (no evidence of a gap) |
| IFBench GEPA | 38.61 | 32.99 (3) | -5.62 | [-15.49, +4.25] | max(SD_run 3.61, s_pooled 2.59) | -1.35 | 0.220 | consistent (no evidence of a gap) |
| IFBench GEPA+Merge | 28.23 | 35.09 (3) | +6.86 | [-2.85, +16.57] | max(SD_run 3.56, s_pooled 2.59) | +1.67 | 0.139 | consistent (no evidence of a gap) |
| AIME Baseline | 27.33 | 27.78 (3) | +0.45 | [-8.45, +9.34] | SD_samp 3.93 | +0.10 | 0.922 | consistent (no evidence of a gap) |
| AIME MIPROv2 | 20.00 | 34.22 (3) | +14.22 | [-5.08, +33.52] | max(SD_run 4.78, s_pooled 7.07) | +1.74 | 0.125 | consistent (no evidence of a gap) |
| AIME GEPA + GEPA+Merge (one target) | 32.00 | 34.00 (6) | +2.00 | [-16.05, +20.06] | max(SD_run 5.23, s_pooled 7.07) | +0.26 | 0.801 | consistent (no evidence of a gap) |
| LiveBench Baseline | 48.70 | 51.48 (3) | +2.78 | [-3.35, +8.91] | SD_samp 2.71 | +0.89 | 0.374 | consistent (no evidence of a gap) |
| LiveBench MIPROv2 | 46.60 | 48.90 (3) | +2.30 | [-9.43, +14.02] | max(SD_run 4.20, s_pooled 4.30) | +0.46 | 0.657 | consistent (no evidence of a gap) |
| LiveBench GEPA + GEPA+Merge (one target) | 51.95 | 50.09 (6) | -1.86 | [-12.83, +9.11] | max(SD_run 4.21, s_pooled 4.30) | -0.40 | 0.701 | consistent (no evidence of a gap) |

Baselines: z = d / (SD_samp · √(1 + 1/n)), normal. Optimizer targets: t = d / (σ · √(1 + 1/n)) on 7 df, σ = max(SD_run, s_pooled), SD_run = √(SD_samp² + 3.22²); t(0.975, 7) = 2.365. n = 3, or 6 for a math GEPA pair.
