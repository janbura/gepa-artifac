## Table 3 with three seeds (plan 2.14)

Rebuild: every run gives its reported score (largest gap 0.0045 points). Missing runs: PUPA SelectBestCandidate s0, PUPA SelectBestCandidate s2, PUPA BeamSearch s0, PUPA BeamSearch s2. Test score in %, mean ± SD over the seeds [paper, one run]. Our PUPA uses gpt-oss-120b as external model and judge (plan 2.10), so its column and the four-task aggregate are not comparable with the paper.

| Method | HotpotQA | IFBench | HoVer | PUPA | Aggregate (4) | Improvement (4) | Aggregate (3, no PUPA) | Improvement (3) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 38.78 ± 0.69 [42.33] | 34.63 ± 0.52 [36.90] | 37.11 ± 1.58 [35.33] | 86.22 ± 0.49 [80.82] | 49.19 [48.84] | – | 36.84 [38.19] | – |
| SelectBestCandidate | 56.78 ± 1.57† [58.33] | 29.76 ± 1.70† [30.44] | 50.22 ± 3.17† [45.33] | 92.64 (n=1) [85.45] | 57.35 [54.89] | +8.16 [+6.04] | 45.59 [44.70] | +8.75 [+6.51] |
| BeamSearch | 54.44 ± 6.95 [57.33] | 35.83 ± 2.75 [36.39] | 51.89 ± 1.58 [41.00] | 93.41 (n=1) [81.08] | 58.89 [53.95] | +9.71 [+5.11] | 47.39 [44.91] | +10.55 [+6.72] |
| GEPA | 60.44 ± 2.72 [62.33] | 32.99 ± 1.48† [38.61] | 51.00 ± 2.33† [52.33] | 92.37 ± 1.46 [91.85] | 59.20 [61.28] | +10.02 [+12.44] | 48.14 [51.09] | +11.30 [+12.90] |

† = contains a run with lost requests or a lost test score (rule of 2026-11-11; F113).

### Per seed (seed 0 / 1 / 2)

| Method | HotpotQA | IFBench | HoVer | PUPA |
|---|---|---|---|---|
| Baseline | 39.33 / 38.00 / 39.00 | 35.20 / 34.18 / 34.52 | 38.33 / 35.33 / 37.67 | 85.94 / 85.93 / 86.79 |
| SelectBestCandidate | 57.33† / 55.00 / 58.00 | 29.76† / 31.46 / 28.06 | 50.33† / 47.00 / 53.33 | – / 92.64 / – |
| BeamSearch | 53.00 / 48.33 / 62.00 | 35.37 / 33.33 / 38.78 | 51.33 / 53.67 / 50.67 | – / 93.41 / – |
| GEPA | 62.33 / 57.33 / 61.67 | 33.67 / 31.29 / 34.01† | 48.67† / 51.00† / 53.33 | 92.86 / 90.73 / 93.53 |

### Observation 3: paired per-example tests (Holm over the 8 tests)

d = mean over test examples of A − B in points (each method: mean over its seeds); 95 % percentile bootstrap (10,000 resamples); p = paired sign-flip test (10,000 sign vectors). The interval and p cover the test examples, not the run-to-run variation; the per-seed differences show that.

| Task | A − B | Paper | Per seed | d | 95 % interval | A better / B better | p | p (Holm) | Holm at 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | GEPA - SelectBestCandidate | +4.00 | +5.00 / +2.33 / +3.67 | +3.67 | [+0.67, +6.56] | 57 / 33 | 0.0190 | 0.1330 | no |
| HotpotQA | GEPA - BeamSearch | +5.00 | +9.33 / +9.00 / -0.33 | +6.00 | [+3.11, +8.78] | 69 / 30 | 0.0001 | 0.0008 | rejects |
| IFBench | GEPA - SelectBestCandidate | +8.17 | +3.91 / -0.17 / +5.95 | +3.23 | [+0.23, +6.24] | 75 / 55 | 0.0377 | 0.2262 | no |
| IFBench | GEPA - BeamSearch | +2.22 | -1.70 / -2.04 / -4.77 | -2.83 | [-5.84, +0.23] | 44 / 71 | 0.0668 | 0.3340 | no |
| HoVer | GEPA - SelectBestCandidate | +7.00 | -1.66 / +4.00 / +0.00 | +0.78 | [-1.67, +3.33] | 44 / 41 | 0.6057 | 1.0000 | no |
| HoVer | GEPA - BeamSearch | +11.33 | -2.66 / -2.67 / +2.66 | -0.89 | [-3.33, +1.44] | 36 / 38 | 0.5272 | 1.0000 | no |
| PUPA | GEPA - SelectBestCandidate | +6.40 | -1.91 | -0.26 | [-2.64, +2.16] | 37 / 60 | 0.8287 | 1.0000 | no |
| PUPA | GEPA - BeamSearch | +10.77 | -2.68 | -1.04 | [-3.16, +1.19] | 31 / 59 | 0.3550 | 1.0000 | no |

### Mean over tasks (no test)

| A − B | Tasks | Paper | d | 95 % interval |
|---|---|---|---|---|
| GEPA − SelectBestCandidate | HotpotQA, IFBench, HoVer | +6.39 | +2.56 | [+0.92, +4.18] |
| GEPA − SelectBestCandidate | all four (PUPA not comparable) | +6.39 | +1.85 | [+0.49, +3.22] |
| GEPA − BeamSearch | HotpotQA, IFBench, HoVer | +6.18 | +0.76 | [-0.82, +2.34] |
| GEPA − BeamSearch | all four (PUPA not comparable) | +7.33 | +0.31 | [-0.99, +1.59] |
| SelectBestCandidate − BeamSearch | HotpotQA, IFBench, HoVer | -0.21 | -1.80 | [-3.56, -0.06] |
| SelectBestCandidate − BeamSearch | all four (PUPA not comparable) | +0.94 | -1.54 | [-3.01, -0.08] |

### Descriptive: SelectBestCandidate − BeamSearch per task

| Task | Paper | Per seed | d | 95 % interval |
|---|---|---|---|---|
| HotpotQA | +1.00 | +4.33 / +6.67 / -4.00 | +2.33 | [-0.67, +5.33] |
| IFBench | -5.95 | -5.61 / -1.87 / -10.72 | -6.07 | [-9.35, -2.72] |
| HoVer | +4.33 | -1.00 / -6.67 / +2.66 | -1.67 | [-4.33, +1.00] |
| PUPA | +4.37 | -0.77 | -0.77 | [-3.38, +1.85] |

### Descriptive: the tested rows without the † runs

| Task | A − B | seeds of A / of B | d | 95 % interval | d with all runs |
|---|---|---|---|---|---|
| HotpotQA | GEPA - SelectBestCandidate | [0, 1, 2] / [1, 2] | +3.94 | [+0.28, +7.56] | +3.67 |
| IFBench | GEPA - SelectBestCandidate | [0, 1] / [1, 2] | +2.72 | [-1.02, +6.55] | +3.23 |
| IFBench | GEPA - BeamSearch | [0, 1] / [0, 1, 2] | -3.34 | [-6.80, +0.20] | -2.83 |
| HoVer | GEPA - SelectBestCandidate | [2] / [1, 2] | +3.17 | [-0.50, +6.83] | +0.78 |
| HoVer | GEPA - BeamSearch | [2] / [0, 1, 2] | +1.44 | [-2.11, +4.89] | -0.89 |

### Our seed-0 SelectBestCandidate and the authors' released seed-0 run

| Task | ours, seed 0 | authors (= paper) |
|---|---|---|
| HotpotQA | 57.33 | 58.33 |
| IFBench | 29.76 | 30.44 |
| HoVer | 50.33 | 45.33 |
| PUPA | – | 85.45 (GPT-4.1-mini judge) |

### Search statistics from the run logs (descriptive; sum over the seeds)

| Task | Method | runs | iterations | proposals | accepted candidates | parent picks | picks below the best so far |
|---|---|---|---|---|---|---|---|
| HotpotQA | SelectBestCandidate | 3 | 331 | 281 | 62 | 331 | 0 (0 %) |
| HotpotQA | BeamSearch | 3 | 414 | 349 | 60 | 414 | 280 (68 %) |
| HotpotQA | GEPA | 3 | 369 | 307 | 62 | 369 | 279 (76 %) |
| IFBench | SelectBestCandidate | 3 | 306 | 164 | 31 | 306 | 0 (0 %) |
| IFBench | BeamSearch | 3 | 174 | 112 | 33 | 174 | 102 (59 %) |
| IFBench | GEPA | 3 | 155 | 103 | 33 | 155 | 106 (68 %) |
| HoVer | SelectBestCandidate | 3 | 383 | 342 | 63 | 383 | 0 (0 %) |
| HoVer | BeamSearch | 3 | 393 | 342 | 62 | 393 | 249 (63 %) |
| HoVer | GEPA | 3 | 335 | 310 | 63 | 335 | 249 (74 %) |
| PUPA | SelectBestCandidate | 1 | 79 | 35 | 20 | 79 | 0 (0 %) |
| PUPA | BeamSearch | 1 | 136 | 67 | 18 | 136 | 102 (75 %) |
| PUPA | GEPA | 3 | 276 | 146 | 57 | 276 | 214 (78 %) |

