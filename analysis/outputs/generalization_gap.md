# Validation-to-test generalization gap (seed 0)

| benchmark | model | optimizer | n_val | n_candidates | seed_val | baseline_test | seed_gap | best_val | best_id | tested_val | test | gap | gap_tested | excess_gap | val_gain | test_gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | gpt-41-mini | MIPROv2 | 300 | 15 | 40.33 | 38.00 | 2.33 | 62.33 | 10 | 62.33 | 58.00 | 4.33 | 4.33 | 2.00 | 22.00 | 20.00 |
| HotpotQA | gpt-41-mini | GEPA | 300 | 21 | 40.33 | 38.00 | 2.33 | 68.33 | 13 | 68.33 | 69.00 | -0.67 | -0.67 | -3.00 | 28.00 | 31.00 |
| HotpotQA | gpt-41-mini | GEPA+Merge | 300 | 22 | 40.33 | 38.00 | 2.33 | 67.33 | 12 | 67.33 | 65.67 | 1.66 | 1.66 | -0.67 | 27.00 | 27.67 |
| HotpotQA | gpt-41-mini | SelectBest | 300 | 20 | 40.33 | 38.00 | 2.33 | 69.67 | 18 | 69.67 | 66.00 | 3.67 | 3.67 | 1.34 | 29.34 | 28.00 |
| IFBench | gpt-41-mini | MIPROv2 | 300 | 8 | 82.39 | 47.79 | 34.60 | 90.53 | 1 | 90.53 | 49.15 | 41.38 | 41.38 | 6.78 | 8.14 | 1.36 |
| IFBench | gpt-41-mini | GEPA | 300 | 12 | 82.06 | 47.79 | 34.27 | 90.47 | 9 | 90.47 | 52.72 | 37.75 | 37.75 | 3.48 | 8.41 | 4.93 |
| IFBench | gpt-41-mini | GEPA+Merge | 300 | 12 | 81.89 | 47.79 | 34.10 | 90.03 | 7 | 90.03 | 55.95 | 34.08 | 34.08 | -0.02 | 8.14 | 8.16 |
| IFBench | gpt-41-mini | SelectBest | 300 | 12 | 81.89 | 47.79 | 34.10 | 90.50 | 10 | 90.50 | 49.49 | 41.01 | 41.01 | 6.91 | 8.61 | 1.70 |
| HoVer | gpt-41-mini | MIPROv2 | 300 | 15 | 44.67 | 46.33 | -1.66 | 51.33 | 3 | 51.33 | 48.33 | 3.00 | 3.00 | 4.66 | 6.66 | 2.00 |
| HoVer | gpt-41-mini | GEPA | 300 | 22 | 44.67 | 46.33 | -1.66 | 54.33 | 19 | 54.33 | 51.67 | 2.66 | 2.66 | 4.32 | 9.66 | 5.34 |
| HoVer | gpt-41-mini | GEPA+Merge | 300 | 22 | 44.67 | 46.33 | -1.66 | 52.67 | 17 | 52.67 | 56.67 | -4.00 | -4.00 | -2.34 | 8.00 | 10.34 |
| HoVer | gpt-41-mini | SelectBest | 300 | 22 | 44.67 | 46.33 | -1.66 | 53.67 | 10 | 53.67 | 51.00 | 2.67 | 2.67 | 4.33 | 9.00 | 4.67 |
| PUPA | gpt-41-mini | MIPROv2 | 111 | 8 | 81.19 | 78.57 | 2.62 | 88.06 | 7 | 88.06 | 83.37 | 4.69 | 4.69 | 2.07 | 6.87 | 4.80 |
| PUPA | gpt-41-mini | GEPA | 111 | 19 | 82.26 | 78.57 | 3.69 | 96.30 | 12 | 96.30 | 94.47 | 1.83 | 1.83 | -1.86 | 14.04 | 15.90 |
| PUPA | gpt-41-mini | GEPA+Merge | 111 | 20 | 82.26 | 78.57 | 3.69 | 97.60 | 11 | 97.60 | 96.46 | 1.14 | 1.14 | -2.55 | 15.34 | 17.89 |
| PUPA | gpt-41-mini | SelectBest | 111 | 19 | 81.56 | 78.57 | 2.99 | 92.87 | 18 | 92.87 | 94.03 | -1.16 | -1.16 | -4.15 | 11.31 | 15.46 |
| AIME | gpt-41-mini | MIPROv2 | 90 | 7 | 46.67 | 49.33 | -2.66 | 53.33 | 3 | 53.33 | 51.33 | 2.00 | 2.00 | 4.66 | 6.66 | 2.00 |
| AIME | gpt-41-mini | GEPA | 45 | 28 | 46.67 | 49.33 | -2.66 | 60.00 | 27 | 60.00 | 59.33 | 0.67 | 0.67 | 3.33 | 13.33 | 10.00 |
| LiveBench-Math | gpt-41-mini | MIPROv2 | 121 | 7 | 66.59 | 58.20 | 8.39 | 68.39 | 1 | 68.39 | 61.84 | 6.55 | 6.55 | -1.84 | 1.80 | 3.64 |
| LiveBench-Math | gpt-41-mini | GEPA | 121 | 14 | 65.77 | 58.20 | 7.57 | 74.56 | 5 | 74.56 | 59.43 | 15.13 | 15.13 | 7.56 | 8.79 | 1.23 |
| LiveBench-Math | gpt-41-mini | GEPA+Merge | 121 | 14 | 64.11 | 58.20 | 5.91 | 75.47 | 9 | 75.47 | 64.13 | 11.34 | 11.34 | 5.43 | 11.36 | 5.93 |
| HotpotQA | qwen3-8b | MIPROv2 | 300 | 15 | 46.33 | 42.33 | 4.00 | 57.67 | 2 | 57.67 | 55.33 | 2.34 | 2.34 | -1.66 | 11.34 | 13.00 |
| HotpotQA | qwen3-8b | GEPA | 300 | 21 | 46.33 | 42.33 | 4.00 | 60.00 | 19 | 60.00 | 62.33 | -2.33 | -2.33 | -6.33 | 13.67 | 20.00 |
| HotpotQA | qwen3-8b | GEPA+Merge | 300 | 22 | 46.33 | 42.33 | 4.00 | 62.00 | 21 | 62.00 | 64.33 | -2.33 | -2.33 | -6.33 | 15.67 | 22.00 |
| HotpotQA | qwen3-8b | SelectBest | 300 | 22 | 46.33 | 42.33 | 4.00 | 61.67 | 21 | 61.67 | 58.33 | 3.34 | 3.34 | -0.66 | 15.34 | 16.00 |
| HotpotQA | qwen3-8b | GRPO | 300 | 25 | 46.67 | 42.33 | 4.34 | 52.33 | 220 | 48.00 | 43.33 | 9.00 | 4.67 | 4.66 | 5.66 | 1.00 |
| IFBench | qwen3-8b | MIPROv2 | 300 | 8 | 57.11 | 36.90 | 20.21 | 82.75 | 7 | 82.75 | 36.22 | 46.53 | 46.53 | 26.32 | 25.64 | -0.68 |
| IFBench | qwen3-8b | GEPA | 300 | 12 | 56.39 | 36.90 | 19.49 | 79.44 | 2 | 79.44 | 38.61 | 40.83 | 40.83 | 21.34 | 23.05 | 1.71 |
| IFBench | qwen3-8b | GEPA+Merge | 300 | 12 | 56.86 | 36.90 | 19.96 | 80.94 | 2 | 80.94 | 28.23 | 52.71 | 52.71 | 32.75 | 24.08 | -8.67 |
| IFBench | qwen3-8b | SelectBest | 300 | 12 | 57.03 | 36.90 | 20.13 | 73.33 | 4 | 73.33 | 30.44 | 42.89 | 42.89 | 22.76 | 16.30 | -6.46 |
| IFBench | qwen3-8b | GRPO | 300 | 25 | 57.00 | 36.90 | 20.10 | 63.47 | 460 | 59.25 | 35.88 | 27.59 | 23.37 | 7.49 | 6.47 | -1.02 |
| HoVer | qwen3-8b | MIPROv2 | 300 | 15 | 35.00 | 35.33 | -0.33 | 45.00 | 7 | 45.00 | 47.33 | -2.33 | -2.33 | -2.00 | 10.00 | 12.00 |
| HoVer | qwen3-8b | GEPA | 300 | 22 | 35.00 | 35.33 | -0.33 | 53.00 | 21 | 53.00 | 52.33 | 0.67 | 0.67 | 1.00 | 18.00 | 17.00 |
| HoVer | qwen3-8b | GEPA+Merge | 300 | 23 | 35.00 | 35.33 | -0.33 | 52.33 | 22 | 52.33 | 51.67 | 0.66 | 0.66 | 0.99 | 17.33 | 16.34 |
| HoVer | qwen3-8b | SelectBest | 300 | 21 | 35.00 | 35.33 | -0.33 | 50.67 | 15 | 50.67 | 45.33 | 5.34 | 5.34 | 5.67 | 15.67 | 10.00 |
| HoVer | qwen3-8b | GRPO | 300 | 25 | 35.33 | 35.33 | 0.00 | 39.00 | 500 | 39.00 | 38.67 | 0.33 | 0.33 | 0.33 | 3.67 | 3.34 |
| PUPA | qwen3-8b | MIPROv2 | 111 | 8 | 80.28 | 80.82 | -0.54 | 83.95 | 4 | 83.95 | 81.55 | 2.40 | 2.40 | 2.94 | 3.67 | 0.73 |
| PUPA | qwen3-8b | GEPA | 111 | 20 | 79.21 | 80.82 | -1.61 | 90.39 | 17 | 90.39 | 91.85 | -1.46 | -1.46 | 0.15 | 11.18 | 11.03 |
| PUPA | qwen3-8b | GEPA+Merge | 111 | 20 | 80.75 | 80.82 | -0.07 | 87.98 | 14 | 87.98 | 86.26 | 1.72 | 1.72 | 1.79 | 7.23 | 5.44 |
| PUPA | qwen3-8b | SelectBest | 111 | 19 | 80.15 | 80.82 | -0.67 | 90.39 | 14 | 90.39 | 85.45 | 4.94 | 4.94 | 5.61 | 10.24 | 4.63 |
| PUPA | qwen3-8b | GRPO | 111 | 25 | 78.92 | 80.82 | -1.90 | 86.64 | 300 | 84.26 | 86.66 | -0.02 | -2.40 | 1.88 | 7.72 | 5.84 |

## Summary per optimizer

`gap_below_mipro` counts the benchmark/model pairs on which the gap of the optimizer is smaller than the gap of MIPROv2 on the same pair (same for `excess_gap_below_mipro`).

| model | optimizer | n_runs | mean_gap | median_gap | mean_seed_gap | mean_excess_gap | mean_val_gain | mean_test_gain | gap_below_mipro | excess_gap_below_mipro |
|---|---|---|---|---|---|---|---|---|---|---|
| all | MIPROv2 | 10 | 11.09 | 3.66 | 6.70 | 4.39 | 10.28 | 5.89 | 0/10 | 0/10 |
| all | GEPA | 10 | 9.51 | 1.25 | 6.51 | 3.00 | 14.81 | 11.81 | 8/10 | 8/10 |
| all | GEPA+Merge | 9 | 10.78 | 1.66 | 7.55 | 3.23 | 14.91 | 11.68 | 6/9 | 6/9 |
| all | SelectBest | 8 | 12.84 | 4.30 | 7.61 | 5.23 | 14.48 | 9.25 | 5/8 | 4/8 |
| all | GRPO | 4 | 9.22 | 4.66 | 5.64 | 3.59 | 5.88 | 2.29 | 2/4 | 2/4 |
| gpt-41-mini | MIPROv2 | 6 | 10.32 | 4.51 | 7.27 | 3.05 | 8.69 | 5.63 | 0/6 | 0/6 |
| gpt-41-mini | GEPA | 6 | 9.56 | 2.24 | 7.26 | 2.30 | 13.70 | 11.40 | 5/6 | 5/6 |
| gpt-41-mini | GEPA+Merge | 5 | 8.84 | 1.66 | 8.87 | -0.03 | 13.97 | 14.00 | 4/5 | 4/5 |
| gpt-41-mini | SelectBest | 4 | 11.55 | 3.17 | 9.44 | 2.11 | 14.57 | 12.46 | 4/4 | 3/4 |
| qwen3-8b | MIPROv2 | 4 | 12.24 | 2.37 | 5.84 | 6.40 | 12.66 | 6.26 | 0/4 | 0/4 |
| qwen3-8b | GEPA | 4 | 9.43 | -0.39 | 5.39 | 4.04 | 16.48 | 12.44 | 3/4 | 3/4 |
| qwen3-8b | GEPA+Merge | 4 | 13.19 | 1.19 | 5.89 | 7.30 | 16.08 | 8.78 | 2/4 | 2/4 |
| qwen3-8b | SelectBest | 4 | 14.13 | 5.14 | 5.78 | 8.34 | 14.39 | 6.04 | 1/4 | 1/4 |
| qwen3-8b | GRPO | 4 | 9.22 | 4.66 | 5.64 | 3.59 | 5.88 | 2.29 | 2/4 | 2/4 |

Columns: `seed_val` is the full validation score of the seed program as logged by
that optimizer, `baseline_test` the test score of the Baseline run of the same
benchmark and model, `seed_gap = seed_val - baseline_test`.  `best_val` is the
best full validation score reached (GEPA family: max over candidates; MIPROv2: max
of the "Full eval scores so far" list; GRPO: max over the evaluations done every
20 steps), `best_id` identifies it (candidate index, position in the MIPROv2 list
with 0 = seed program, or training step), `tested_val` is the validation score of
the program that was evaluated on the test set (equal to `best_val` except for
GRPO, which keeps the last checkpoint), `gap = best_val - test`, `gap_tested =
tested_val - test`, `excess_gap = gap - seed_gap` (the part of the gap not already
present for the unoptimized seed program), `val_gain = best_val - seed_val`,
`test_gain = test - baseline_test`.  All scores are in percent.

Caveats: the AIME GEPA run used a 45-example validation set (benchmark
"AIMEBench2"), the AIME MIPROv2 run a 90-example one ("AIMEBench"); the GRPO
runs were tested with the final checkpoint, not the best one; only seed 0 exists,
so differences of one or two points are within evaluation noise (the validation
score of the same seed program varies by about one point between optimizer runs
of the same benchmark).
