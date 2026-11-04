# Per-instance validation changes of the best candidate relative to the seed program

| benchmark | model | optimizer | n_val | best_id | seed_val | best_val | val_gain | n_better | n_equal | n_worse | n_undetermined | frac_better | frac_equal | frac_worse | frac_changed | binary_metric | n_seed_from_log | n_settled_by_log | n_log_state_mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | gpt-41-mini | GEPA | 300 | 13 | 40.330 | 68.330 | 28.000 | 91 | 202 | 7 | 0 | 0.303 | 0.673 | 0.023 | 0.327 | True | 300 | 0 | 0 |
| HotpotQA | gpt-41-mini | GEPA+Merge | 300 | 12 | 40.330 | 67.330 | 27.000 | 89 | 203 | 8 | 0 | 0.297 | 0.677 | 0.027 | 0.323 | True | 300 | 0 | 0 |
| HotpotQA | gpt-41-mini | SelectBest | 300 | 18 | 40.330 | 69.670 | 29.340 | 94 | 200 | 6 | 0 | 0.313 | 0.667 | 0.020 | 0.333 | True | 300 | 0 | 0 |
| IFBench | gpt-41-mini | GEPA | 300 | 9 | 82.060 | 90.470 | 8.410 | 41 | 251 | 8 | 0 | 0.137 | 0.837 | 0.027 | 0.163 | False | 300 | 20 | 0 |
| IFBench | gpt-41-mini | GEPA+Merge | 300 | 7 | 81.890 | 90.030 | 8.140 | 38 | 253 | 9 | 0 | 0.127 | 0.843 | 0.030 | 0.157 | False | 300 | 18 | 0 |
| IFBench | gpt-41-mini | SelectBest | 300 | 10 | 81.890 | 90.500 | 8.610 | 42 | 250 | 8 | 0 | 0.140 | 0.833 | 0.027 | 0.167 | False | 300 | 15 | 0 |
| HoVer | gpt-41-mini | GEPA | 300 | 19 | 44.670 | 54.330 | 9.660 | 43 | 243 | 14 | 0 | 0.143 | 0.810 | 0.047 | 0.190 | True | 298 | 0 | 0 |
| HoVer | gpt-41-mini | GEPA+Merge | 300 | 17 | 44.670 | 52.670 | 8.000 | 40 | 244 | 16 | 0 | 0.133 | 0.813 | 0.053 | 0.187 | True | 298 | 0 | 0 |
| HoVer | gpt-41-mini | SelectBest | 300 | 10 | 44.670 | 53.670 | 9.000 | 42 | 243 | 15 | 0 | 0.140 | 0.810 | 0.050 | 0.190 | True | 298 | 0 | 0 |
| PUPA | gpt-41-mini | GEPA | 111 | 12 | 82.260 | 96.300 | 14.040 | 41 | 63 | 6 | 1 | 0.369 | 0.568 | 0.054 | 0.423 | False | 82 | 4 | 0 |
| PUPA | gpt-41-mini | GEPA+Merge | 111 | 11 | 82.260 | 97.600 | 15.340 | 45 | 61 | 5 | 0 | 0.405 | 0.550 | 0.045 | 0.450 | False | 85 | 1 | 0 |
| PUPA | gpt-41-mini | SelectBest | 111 | 18 | 81.560 | 92.870 | 11.310 | 32 | 69 | 5 | 5 | 0.288 | 0.622 | 0.045 | 0.333 | False | 84 | 13 | 0 |
| AIME | gpt-41-mini | GEPA | 45 | 27 | 46.670 | 60.000 | 13.330 | 10 | 31 | 4 | 0 | 0.222 | 0.689 | 0.089 | 0.311 | True | 45 | 0 | 0 |
| LiveBench-Math | gpt-41-mini | GEPA | 121 | 5 | 65.770 | 74.560 | 8.790 | 23 | 87 | 11 | 0 | 0.190 | 0.719 | 0.091 | 0.281 | False | 121 | 14 | 0 |
| LiveBench-Math | gpt-41-mini | GEPA+Merge | 121 | 9 | 64.110 | 75.470 | 11.360 | 20 | 96 | 5 | 0 | 0.165 | 0.793 | 0.041 | 0.207 | False | 121 | 19 | 0 |
| HotpotQA | qwen3-8b | GEPA | 300 | 19 | 46.330 | 60.000 | 13.670 | 56 | 229 | 15 | 0 | 0.187 | 0.763 | 0.050 | 0.237 | True | 294 | 0 | 0 |
| HotpotQA | qwen3-8b | GEPA+Merge | 300 | 21 | 46.330 | 62.000 | 15.670 | 54 | 239 | 7 | 0 | 0.180 | 0.797 | 0.023 | 0.203 | True | 294 | 0 | 0 |
| HotpotQA | qwen3-8b | SelectBest | 300 | 21 | 46.330 | 61.670 | 15.340 | 53 | 240 | 7 | 0 | 0.177 | 0.800 | 0.023 | 0.200 | True | 294 | 0 | 0 |
| IFBench | qwen3-8b | GEPA | 300 | 2 | 56.390 | 79.440 | 23.050 | 94 | 188 | 14 | 4 | 0.313 | 0.627 | 0.047 | 0.360 | False | 280 | 27 | 0 |
| IFBench | qwen3-8b | GEPA+Merge | 300 | 2 | 56.860 | 80.940 | 24.080 | 88 | 201 | 6 | 5 | 0.293 | 0.670 | 0.020 | 0.313 | False | 280 | 30 | 0 |
| IFBench | qwen3-8b | SelectBest | 300 | 4 | 57.030 | 73.330 | 16.300 | 77 | 197 | 22 | 4 | 0.257 | 0.657 | 0.073 | 0.330 | False | 280 | 32 | 0 |
| HoVer | qwen3-8b | GEPA | 300 | 21 | 35.000 | 53.000 | 18.000 | 62 | 230 | 8 | 0 | 0.207 | 0.767 | 0.027 | 0.233 | True | 298 | 0 | 0 |
| HoVer | qwen3-8b | GEPA+Merge | 300 | 22 | 35.000 | 52.330 | 17.330 | 55 | 242 | 3 | 0 | 0.183 | 0.807 | 0.010 | 0.193 | True | 298 | 0 | 0 |
| HoVer | qwen3-8b | SelectBest | 300 | 15 | 35.000 | 50.670 | 15.670 | 56 | 235 | 9 | 0 | 0.187 | 0.783 | 0.030 | 0.217 | True | 298 | 0 | 0 |
| PUPA | qwen3-8b | GEPA | 111 | 17 | 79.210 | 90.390 | 11.180 | 32 | 70 | 4 | 5 | 0.288 | 0.631 | 0.036 | 0.324 | False | 85 | 14 | 0 |
| PUPA | qwen3-8b | GEPA+Merge | 111 | 14 | 80.750 | 87.980 | 7.230 | 29 | 65 | 14 | 3 | 0.261 | 0.586 | 0.126 | 0.387 | False | 86 | 17 | 0 |
| PUPA | qwen3-8b | SelectBest | 111 | 14 | 80.150 | 90.390 | 10.240 | 30 | 68 | 3 | 10 | 0.270 | 0.613 | 0.027 | 0.297 | False | 85 | 12 | 0 |

For every GEPA-family run the best validation candidate (`best_id`) is compared
with the seed program (candidate 0) on each validation instance.  `n_better`,
`n_equal`, `n_worse` count instances where the best candidate scores higher, the
same or lower than the seed; `frac_changed` is the share of instances whose score
changed in either direction.  The comparison uses the per-instance Pareto front
stored in gepa_state.bin (the seed row of `prog_candidate_val_subscores` is not
usable, see the script); for graded metrics, instances where both programs are
below the per-instance best are settled with the seed scores from
metric_logs/val.jsonl (`n_settled_by_log`), and `n_undetermined` instances could
not be settled either way.  `n_log_state_mismatch` counts instances where the log
and the state disagree (expected 0).
