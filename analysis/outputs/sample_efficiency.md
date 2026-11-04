# Rollout budget curves and sample efficiency

Rollouts are metric calls. GEPA family: `num_metric_calls_by_discovery` from gepa_state.bin (calls spent before a candidate's full validation evaluation; the seed program sits at 0). MIPROv2: bootstrapping train calls + 35 per minibatch trial + valset size per earlier full evaluation. GRPO: 48 rollouts per training step, scores logged every 20 steps (960 rollouts), 24000 in total.

## Sample efficiency against GRPO (Qwen3 8B)

`grpo_best_val` is the best of GRPO's logged validation scores and `grpo_rollouts_needed` the rollouts at which it was first reached. `rollouts_to_match_grpo` is the first curve point of each prompt optimizer whose best-so-far validation score is at least GRPO's best; `rollouts_incl_val_eval` adds the full validation evaluation of that candidate itself. `paper_rollouts` is the number quoted in the paper for GEPA.

| bench | model | optimizer | grpo_best_val | grpo_rollouts_needed | opt_best_val | rollouts_to_match_grpo | val_score_at_match | rollouts_incl_val_eval | speedup | speedup_incl_val_eval | paper_rollouts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HotpotQABench | qwen3-8b | GEPA | 52.33 | 10560 | 60.00 | 402 | 57.33 | 702 | 26.27 | 15.04 | 243 |
| HotpotQABench | qwen3-8b | GEPA+Merge | 52.33 | 10560 | 62.00 | 672 | 53.67 | 972 | 15.71 | 10.86 |  |
| HotpotQABench | qwen3-8b | GEPA-Abl-SBC | 52.33 | 10560 | 61.67 | 636 | 56.00 | 936 | 16.60 | 11.28 |  |
| HotpotQABench | qwen3-8b | MIPROv2 | 52.33 | 10560 | 57.67 | 536 | 56.33 | 836 | 19.70 | 12.63 |  |
| IFBench | qwen3-8b | GEPA | 63.47 | 22080 | 79.44 | 330 | 67.31 | 630 | 66.91 | 35.05 | 402 |
| IFBench | qwen3-8b | GEPA+Merge | 63.47 | 22080 | 80.94 | 636 | 80.94 | 936 | 34.72 | 23.59 |  |
| IFBench | qwen3-8b | GEPA-Abl-SBC | 63.47 | 22080 | 73.33 | 348 | 70.47 | 648 | 63.45 | 34.07 |  |
| IFBench | qwen3-8b | MIPROv2 | 63.47 | 22080 | 82.75 | 510 | 74.28 | 810 | 43.29 | 27.26 |  |
| Papillon | qwen3-8b | GEPA | 86.64 | 14400 | 90.39 | 1179 | 88.63 | 1290 | 12.21 | 11.16 | 1143 |
| Papillon | qwen3-8b | GEPA+Merge | 86.64 | 14400 | 87.98 | 1734 | 87.98 | 1845 | 8.30 | 7.80 |  |
| Papillon | qwen3-8b | GEPA-Abl-SBC | 86.64 | 14400 | 90.39 | 363 | 88.21 | 474 | 39.67 | 30.38 |  |
| Papillon | qwen3-8b | MIPROv2 | 86.64 | 14400 | 83.95 |  |  |  |  |  |  |
| hoverBench | qwen3-8b | GEPA | 39.00 | 24000 | 53.00 | 306 | 40.00 | 606 | 78.43 | 39.60 | 330 |
| hoverBench | qwen3-8b | GEPA+Merge | 39.00 | 24000 | 52.33 | 306 | 44.33 | 606 | 78.43 | 39.60 |  |
| hoverBench | qwen3-8b | GEPA-Abl-SBC | 39.00 | 24000 | 50.67 | 948 | 43.33 | 1248 | 25.32 | 19.23 |  |
| hoverBench | qwen3-8b | MIPROv2 | 39.00 | 24000 | 45.00 | 1066 | 39.00 | 1366 | 22.51 | 17.57 |  |

Comparison with the paper:

- HotpotQA: GEPA reaches GRPO's best validation score (52.33) after 402 rollouts (paper: 243); GRPO needed 10560 rollouts, so GEPA used 26.3x fewer (15.0x when the matching candidate's own validation evaluation is counted).
- IFBench: GEPA reaches GRPO's best validation score (63.47) after 330 rollouts (paper: 402); GRPO needed 22080 rollouts, so GEPA used 66.9x fewer (35.0x when the matching candidate's own validation evaluation is counted).
- PUPA: GEPA reaches GRPO's best validation score (86.64) after 1179 rollouts (paper: 1143); GRPO needed 14400 rollouts, so GEPA used 12.2x fewer (11.2x when the matching candidate's own validation evaluation is counted).
- HoVer: GEPA reaches GRPO's best validation score (39.00) after 306 rollouts (paper: 330); GRPO needed 24000 rollouts, so GEPA used 78.4x fewer (39.6x when the matching candidate's own validation evaluation is counted).
- Largest ratio for GEPA: 78.4x on HoVer (paper: up to 78x).

## Rollouts at which the best candidate was found (GEPA family)

`rollouts_at_best` is the discovery count of the candidate with the highest full validation score, `budget` the rollout budget of the task.

| run | bench | model | optimizer | num_candidates | seed_val | best_val | test_score | best_candidate_idx | rollouts_at_best | budget | fraction_of_budget | last_candidate_rollouts | total_num_evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AIMEBench_CoT_GEPA_gpt-41-mini | AIMEBench | gpt-41-mini | GEPA | 28 | 46.67 | 60.00 | 59.33 | 27 | 1815 | 1839 | 0.99 | 1815 | 1860 |
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA-Abl-SBC | 20 | 40.33 | 69.67 | 66.00 | 18 | 6210 | 6871 | 0.90 | 6564 | 6876 |
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_qwen3-8b | HotpotQABench | qwen3-8b | GEPA-Abl-SBC | 22 | 46.33 | 61.67 | 58.33 | 21 | 6744 | 6871 | 0.98 | 6744 | 7044 |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA+Merge | 22 | 40.33 | 67.33 | 65.67 | 12 | 3899 | 6871 | 0.57 | 6749 | 7049 |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b | HotpotQABench | qwen3-8b | GEPA+Merge | 22 | 46.33 | 62.00 | 64.33 | 21 | 6724 | 6871 | 0.98 | 6724 | 7024 |
| HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA | 21 | 40.33 | 68.33 | 69.00 | 13 | 4326 | 6871 | 0.63 | 6624 | 6924 |
| HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b | HotpotQABench | qwen3-8b | GEPA | 21 | 46.33 | 60.00 | 62.33 | 19 | 6438 | 6871 | 0.94 | 6792 | 7092 |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_gpt-41-mini | IFBench | gpt-41-mini | GEPA-Abl-SBC | 12 | 81.89 | 90.50 | 49.49 | 10 | 3198 | 3593 | 0.89 | 3510 | 3810 |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_qwen3-8b | IFBench | qwen3-8b | GEPA-Abl-SBC | 12 | 57.03 | 73.33 | 30.44 | 4 | 1290 | 3593 | 0.36 | 3528 | 3828 |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_gpt-41-mini | IFBench | gpt-41-mini | GEPA+Merge | 12 | 81.89 | 90.03 | 55.95 | 7 | 2159 | 3593 | 0.60 | 3441 | 3741 |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_qwen3-8b | IFBench | qwen3-8b | GEPA+Merge | 12 | 56.86 | 80.94 | 28.23 | 2 | 636 | 3593 | 0.18 | 3427 | 3727 |
| IFBench_IFBenchCoT2StageProgram_GEPA_gpt-41-mini | IFBench | gpt-41-mini | GEPA | 12 | 82.06 | 90.47 | 52.72 | 9 | 2814 | 3593 | 0.78 | 3480 | 3780 |
| IFBench_IFBenchCoT2StageProgram_GEPA_qwen3-8b | IFBench | qwen3-8b | GEPA | 12 | 56.39 | 79.44 | 38.61 | 2 | 678 | 3593 | 0.19 | 3546 | 3846 |
| LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini | LiveBenchMathBench | gpt-41-mini | GEPA+Merge | 14 | 64.11 | 75.47 | 64.13 | 9 | 1245 | 1839 | 0.68 | 1777 | 1898 |
| LiveBenchMathBench_CoT_GEPA_gpt-41-mini | LiveBenchMathBench | gpt-41-mini | GEPA | 14 | 65.77 | 74.56 | 59.43 | 5 | 695 | 1839 | 0.38 | 1777 | 1898 |
| Papillon_PAPILLON_Abl-SelectBestCandidate_gpt-41-mini | Papillon | gpt-41-mini | GEPA-Abl-SBC | 19 | 81.56 | 92.87 | 94.03 | 18 | 2364 | 2426 | 0.97 | 2364 | 2475 |
| Papillon_PAPILLON_Abl-SelectBestCandidate_qwen3-8b | Papillon | qwen3-8b | GEPA-Abl-SBC | 19 | 80.15 | 90.39 | 85.45 | 14 | 1836 | 2426 | 0.76 | 2352 | 2463 |
| Papillon_PAPILLON_GEPA-MERGE_gpt-41-mini | Papillon | gpt-41-mini | GEPA+Merge | 20 | 82.26 | 97.60 | 96.46 | 11 | 1413 | 2426 | 0.58 | 2385 | 2496 |
| Papillon_PAPILLON_GEPA-MERGE_qwen3-8b | Papillon | qwen3-8b | GEPA+Merge | 20 | 80.75 | 87.98 | 86.26 | 14 | 1734 | 2426 | 0.71 | 2382 | 2493 |
| Papillon_PAPILLON_GEPA_gpt-41-mini | Papillon | gpt-41-mini | GEPA | 19 | 82.26 | 96.30 | 94.47 | 12 | 1524 | 2426 | 0.63 | 2268 | 2427 |
| Papillon_PAPILLON_GEPA_qwen3-8b | Papillon | qwen3-8b | GEPA | 20 | 79.21 | 90.39 | 91.85 | 17 | 2157 | 2426 | 0.89 | 2403 | 2514 |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_gpt-41-mini | hoverBench | gpt-41-mini | GEPA-Abl-SBC | 22 | 44.67 | 53.67 | 51.00 | 10 | 3168 | 7051 | 0.45 | 6804 | 7104 |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_qwen3-8b | hoverBench | qwen3-8b | GEPA-Abl-SBC | 21 | 35.00 | 50.67 | 45.33 | 15 | 4902 | 7051 | 0.70 | 6750 | 7056 |
| hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini | hoverBench | gpt-41-mini | GEPA+Merge | 22 | 44.67 | 52.67 | 56.67 | 17 | 5445 | 7051 | 0.77 | 6907 | 7207 |
| hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b | hoverBench | qwen3-8b | GEPA+Merge | 23 | 35.00 | 52.33 | 51.67 | 22 | 6930 | 7051 | 0.98 | 6930 | 7230 |
| hoverBench_HoverMultiHop_GEPA_gpt-41-mini | hoverBench | gpt-41-mini | GEPA | 22 | 44.67 | 54.33 | 51.67 | 19 | 6354 | 7051 | 0.90 | 7014 | 7314 |
| hoverBench_HoverMultiHop_GEPA_qwen3-8b | hoverBench | qwen3-8b | GEPA | 22 | 35.00 | 53.00 | 52.33 | 21 | 6858 | 7051 | 0.97 | 6858 | 7158 |

- Best candidate found after 72% of the budget on average (min 18%, max 99%) over 27 runs; in 6 of them the best candidate is the last one found.

## Budget accounting (GEPA family)

`train_calls` and `val_calls` are the counters on the last line of metric_logs/train.jsonl and val.jsonl (`logged_calls` adds the dev counter, which is 0 for the GEPA family), `state_total_num_evals` is the counter GEPA itself keeps in gepa_state.bin and checks against the budget before every iteration.

| run | bench | model | optimizer | budget | train_calls | val_calls | dev_calls | logged_calls | logged_minus_budget | logged_pct_of_budget | state_total_num_evals | state_minus_budget | logged_minus_state | valset_size | num_full_evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AIMEBench_CoT_GEPA_gpt-41-mini | AIMEBench | gpt-41-mini | GEPA | 1839 | 792 | 1341 | 133 | 2266 | 427 | 123.22 | 1860 | 21 | 406 | 45 | 28 |
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA-Abl-SBC | 6871 | 1089 | 6000 | 0 | 7089 | 218 | 103.17 | 6876 | 5 | 213 | 300 | 20 |
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_qwen3-8b | HotpotQABench | qwen3-8b | GEPA-Abl-SBC | 6871 | 490 | 6573 | 30 | 7093 | 222 | 103.23 | 7044 | 173 | 49 | 300 | 22 |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA+Merge | 6871 | 528 | 6635 | 0 | 7163 | 292 | 104.25 | 7049 | 178 | 114 | 300 | 22 |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b | HotpotQABench | qwen3-8b | GEPA+Merge | 6871 | 412 | 6592 | 0 | 7004 | 133 | 101.94 | 7024 | 153 | -20 | 300 | 22 |
| HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini | HotpotQABench | gpt-41-mini | GEPA | 6871 | 741 | 6299 | 0 | 7040 | 169 | 102.46 | 6924 | 53 | 116 | 300 | 21 |
| HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b | HotpotQABench | qwen3-8b | GEPA | 6871 | 893 | 6217 | 0 | 7110 | 239 | 103.48 | 7092 | 221 | 18 | 300 | 21 |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_gpt-41-mini | IFBench | gpt-41-mini | GEPA-Abl-SBC | 3593 | 318 | 3603 | 0 | 3921 | 328 | 109.13 | 3810 | 217 | 111 | 300 | 12 |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_qwen3-8b | IFBench | qwen3-8b | GEPA-Abl-SBC | 3593 | 295 | 3506 | 0 | 3801 | 208 | 105.79 | 3828 | 235 | -27 | 300 | 12 |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_gpt-41-mini | IFBench | gpt-41-mini | GEPA+Merge | 3593 | 192 | 3618 | 0 | 3810 | 217 | 106.04 | 3741 | 148 | 69 | 300 | 12 |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_qwen3-8b | IFBench | qwen3-8b | GEPA+Merge | 3593 | 115 | 3560 | 0 | 3675 | 82 | 102.28 | 3727 | 134 | -52 | 300 | 12 |
| IFBench_IFBenchCoT2StageProgram_GEPA_gpt-41-mini | IFBench | gpt-41-mini | GEPA | 3593 | 264 | 3604 | 60 | 3928 | 335 | 109.32 | 3780 | 187 | 148 | 300 | 12 |
| IFBench_IFBenchCoT2StageProgram_GEPA_qwen3-8b | IFBench | qwen3-8b | GEPA | 3593 | 350 | 3427 | 0 | 3777 | 184 | 105.12 | 3846 | 253 | -69 | 300 | 12 |
| LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini | LiveBenchMathBench | gpt-41-mini | GEPA+Merge | 1839 | 234 | 1702 | 0 | 1936 | 97 | 105.27 | 1898 | 59 | 38 | 121 | 14 |
| LiveBenchMathBench_CoT_GEPA_gpt-41-mini | LiveBenchMathBench | gpt-41-mini | GEPA | 1839 | 240 | 1702 | 65 | 2007 | 168 | 109.14 | 1898 | 59 | 109 | 121 | 14 |
| Papillon_PAPILLON_Abl-SelectBestCandidate_gpt-41-mini | Papillon | gpt-41-mini | GEPA-Abl-SBC | 2426 | 878 | 1950 | 184 | 3012 | 586 | 124.15 | 2475 | 49 | 537 | 111 | 19 |
| Papillon_PAPILLON_Abl-SelectBestCandidate_qwen3-8b | Papillon | qwen3-8b | GEPA-Abl-SBC | 2426 | 712 | 1942 | 184 | 2838 | 412 | 116.98 | 2463 | 37 | 375 | 111 | 19 |
| Papillon_PAPILLON_GEPA-MERGE_gpt-41-mini | Papillon | gpt-41-mini | GEPA+Merge | 2426 | 717 | 2048 | 172 | 2937 | 511 | 121.06 | 2496 | 70 | 441 | 111 | 20 |
| Papillon_PAPILLON_GEPA-MERGE_qwen3-8b | Papillon | qwen3-8b | GEPA+Merge | 2426 | 524 | 2088 | 162 | 2774 | 348 | 114.34 | 2493 | 67 | 281 | 111 | 20 |
| Papillon_PAPILLON_GEPA_gpt-41-mini | Papillon | gpt-41-mini | GEPA | 2426 | 713 | 1953 | 175 | 2841 | 415 | 117.11 | 2427 | 1 | 414 | 111 | 19 |
| Papillon_PAPILLON_GEPA_qwen3-8b | Papillon | qwen3-8b | GEPA | 2426 | 571 | 2046 | 183 | 2800 | 374 | 115.42 | 2514 | 88 | 286 | 111 | 20 |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_gpt-41-mini | hoverBench | gpt-41-mini | GEPA-Abl-SBC | 7051 | 580 | 6575 | 0 | 7155 | 104 | 101.47 | 7104 | 53 | 51 | 300 | 22 |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_qwen3-8b | hoverBench | qwen3-8b | GEPA-Abl-SBC | 7051 | 834 | 6290 | 0 | 7124 | 73 | 101.04 | 7056 | 5 | 68 | 300 | 21 |
| hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini | hoverBench | gpt-41-mini | GEPA+Merge | 7051 | 597 | 6629 | 0 | 7226 | 175 | 102.48 | 7207 | 156 | 19 | 300 | 22 |
| hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b | hoverBench | qwen3-8b | GEPA+Merge | 7051 | 342 | 6918 | 0 | 7260 | 209 | 102.96 | 7230 | 179 | 30 | 300 | 23 |
| hoverBench_HoverMultiHop_GEPA_gpt-41-mini | hoverBench | gpt-41-mini | GEPA | 7051 | 778 | 6577 | 0 | 7355 | 304 | 104.31 | 7314 | 263 | 41 | 300 | 22 |
| hoverBench_HoverMultiHop_GEPA_qwen3-8b | hoverBench | qwen3-8b | GEPA | 7051 | 617 | 6623 | 0 | 7240 | 189 | 102.68 | 7158 | 107 | 82 | 300 | 22 |

- Logged train+val calls minus budget: mean +260 (+8.1%), min +73, max +586; 27 of 27 runs exceed the budget.
- GEPA's own counter minus budget: mean +117, min +1, max +263; the counter can only overshoot by the last iteration (one subsample round plus one full validation evaluation) because the loop stops once it reaches the budget.
- Logged calls minus GEPA's counter: mean +143, min -69, max +537 (metric calls that GEPA did not count, e.g. retried or failed rollouts).

## Budget accounting (MIPROv2 and GRPO, for reference)

`authors_total` is the MIPROv2 rollout count the authors' notebook computes (bootstrapping + 35 per minibatch + valset size per full evaluation); the task budgets were taken from the MIPROv2 runs. For AIME the MIPROv2 search calls were logged under train/dev (val_calls is 0), so bootstrapping is taken as the logged total minus the nominal search calls.

| run | bench | model | optimizer | budget | train_calls | val_calls | dev_calls | logged_calls | logged_minus_budget | logged_pct_of_budget | authors_total | authors_minus_budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AIMEBench_CoT_MIPROv2-Heavy_gpt-41-mini | AIMEBench | gpt-41-mini | MIPROv2 | 1839 | 1710 | 0 | 129 | 1839 | 0 | 100.00 | 1708 | -131 |
| HotpotQABench_HotpotMultiHop_GRPO_qwen3-8b | HotpotQABench | qwen3-8b | GRPO | 24000 | 23824 | 7758 | 0 | 31582 | 7582 | 131.59 |  |  |
| HotpotQABench_HotpotMultiHop_MIPROv2-Heavy_gpt-41-mini | HotpotQABench | gpt-41-mini | MIPROv2 | 6871 | 57 | 6809 | 0 | 6866 | -5 | 99.93 | 6867 | -4 |
| HotpotQABench_HotpotMultiHop_MIPROv2-Heavy_qwen3-8b | HotpotQABench | qwen3-8b | MIPROv2 | 6871 | 61 | 6756 | 0 | 6817 | -54 | 99.21 | 6871 | 0 |
| IFBench_IFBenchCoT2StageProgram_GRPO_qwen3-8b | IFBench | qwen3-8b | GRPO | 24000 | 23113 | 7579 | 0 | 30692 | 6692 | 127.88 |  |  |
| IFBench_IFBenchCoT2StageProgram_MIPROv2-Heavy_gpt-41-mini | IFBench | gpt-41-mini | MIPROv2 | 3593 | 38 | 3554 | 0 | 3592 | -1 | 99.97 | 3593 | 0 |
| IFBench_IFBenchCoT2StageProgram_MIPROv2-Heavy_qwen3-8b | IFBench | qwen3-8b | MIPROv2 | 3593 | 35 | 3465 | 0 | 3500 | -93 | 97.41 | 3590 | -3 |
| LiveBenchMathBench_CoT_MIPROv2-Heavy_gpt-41-mini | LiveBenchMathBench | gpt-41-mini | MIPROv2 | 1839 | 47 | 1797 | 0 | 1844 | 5 | 100.27 | 1844 | 5 |
| Papillon_PAPILLON_GRPO_qwen3-8b | Papillon | qwen3-8b | GRPO | 24000 | 24281 | 2642 | 196 | 27119 | 3119 | 113.00 |  |  |
| Papillon_PAPILLON_MIPROv2-Heavy_gpt-41-mini | Papillon | gpt-41-mini | MIPROv2 | 2426 | 239 | 1891 | 177 | 2307 | -119 | 95.09 | 2272 | -154 |
| Papillon_PAPILLON_MIPROv2-Heavy_qwen3-8b | Papillon | qwen3-8b | MIPROv2 | 2426 | 237 | 1885 | 181 | 2303 | -123 | 94.93 | 2270 | -156 |
| hoverBench_HoverMultiHop_GRPO_qwen3-8b | hoverBench | qwen3-8b | GRPO | 24000 | 23978 | 7759 | 0 | 31737 | 7737 | 132.24 |  |  |
| hoverBench_HoverMultiHop_MIPROv2-Heavy_gpt-41-mini | hoverBench | gpt-41-mini | MIPROv2 | 7051 | 101 | 6784 | 0 | 6885 | -166 | 97.65 | 6911 | -140 |
| hoverBench_HoverMultiHop_MIPROv2-Heavy_qwen3-8b | hoverBench | qwen3-8b | MIPROv2 | 7051 | 116 | 6779 | 0 | 6895 | -156 | 97.79 | 6926 | -125 |

- GRPO: the 24000 budget counts training rollouts only (48 per step, 500 steps); logged train calls are 23113 to 24281, and the 26 validation checkpoints (val_calls) come on top of it.

