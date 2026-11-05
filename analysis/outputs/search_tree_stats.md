# Search-tree statistics per run (seed 0)

Main columns; the csv next to this file has all of them.

- acceptance_rate: iterations that added a candidate (reflective proposal or kept merge) / iterations
- accept_rate_tested: accepted reflective proposals / proposals that were scored on a minibatch
- frac_sel_seed, frac_sel_current_best: share of parent selections that picked the seed program,
  or the candidate with the best full validation score known at that time
- pareto_front_size: distinct candidates on any per-instance validation front (ties included);
  excl_ties drops instances where every candidate ties; nondominated applies the optimizer's
  dominated-program removal
- n_best_off_front: validation instances whose front does not contain the final best candidate
- log_consistent: the trace counts agree with the run_log.txt lines

| run | n_candidates | n_iterations | n_accepted | acceptance_rate | accept_rate_tested | frac_sel_seed | frac_sel_current_best | pareto_front_size | pareto_front_size_excl_ties | pareto_front_nondominated | n_best_off_front | frac_best_off_front | best_idx | best_val_score | best_depth | max_depth | best_is_last | n_merge_invoked | n_merge_attempts | n_merges_kept | log_consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_gpt-41-mini | 20 | 217 | 19 | 0.088 | 0.13 | 0.028 | 1 | 20 | 20 | 8 | 28 | 0.093 | 18 | 69.67 | 8 | 9 | False | 0 | 0 | 0 | True |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_gpt-41-mini | 12 | 71 | 11 | 0.155 | 0.314 | 0.099 | 1 | 12 | 12 | 5 | 23 | 0.077 | 10 | 90.5 | 4 | 5 | False | 0 | 0 | 0 | True |
| Papillon_PAPILLON_Abl-SelectBestCandidate_gpt-41-mini | 19 | 165 | 18 | 0.109 | 0.295 | 0.006 | 1 | 19 | 19 | 4 | 19 | 0.171 | 18 | 92.87 | 2 | 2 | True | 0 | 0 | 0 | True |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_gpt-41-mini | 22 | 102 | 21 | 0.206 | 0.25 | 0.02 | 1 | 22 | 22 | 11 | 56 | 0.187 | 10 | 53.67 | 5 | 6 | False | 0 | 0 | 0 | True |
| HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_qwen3-8b | 22 | 89 | 21 | 0.236 | 0.284 | 0.011 | 1 | 22 | 22 | 13 | 40 | 0.133 | 21 | 61.67 | 8 | 8 | True | 0 | 0 | 0 | True |
| IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_qwen3-8b | 12 | 65 | 11 | 0.169 | 0.289 | 0.169 | 1 | 12 | 12 | 9 | 54 | 0.18 | 4 | 73.33 | 3 | 4 | False | 0 | 0 | 0 | True |
| Papillon_PAPILLON_Abl-SelectBestCandidate_qwen3-8b | 19 | 111 | 18 | 0.162 | 0.305 | 0.027 | 1 | 19 | 19 | 7 | 25 | 0.225 | 14 | 90.39 | 4 | 5 | False | 0 | 0 | 0 | True |
| hoverBench_HoverMultiHop_Abl-SelectBestCandidate_qwen3-8b | 21 | 145 | 20 | 0.138 | 0.159 | 0.048 | 1 | 21 | 21 | 14 | 65 | 0.217 | 15 | 50.67 | 7 | 8 | False | 0 | 0 | 0 | True |
| AIMEBench_CoT_GEPA_gpt-41-mini | 28 | 128 | 27 | 0.211 | 0.27 | 0.156 | 0.234 | 28 | 28 | 6 | 12 | 0.267 | 27 | 60 | 4 | 4 | True | 0 | 0 | 0 | True |
| HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini | 21 | 143 | 20 | 0.14 | 0.192 | 0.063 | 0.175 | 21 | 21 | 8 | 27 | 0.09 | 13 | 68.33 | 4 | 6 | False | 0 | 0 | 0 | True |
| IFBench_IFBenchCoT2StageProgram_GEPA_gpt-41-mini | 12 | 58 | 11 | 0.19 | 0.367 | 0.052 | 0.31 | 12 | 12 | 6 | 28 | 0.093 | 9 | 90.47 | 3 | 4 | False | 0 | 0 | 0 | True |
| LiveBenchMathBench_CoT_GEPA_gpt-41-mini | 14 | 45 | 13 | 0.289 | 0.382 | 0.311 | 0.289 | 14 | 14 | 9 | 21 | 0.174 | 5 | 74.56 | 3 | 4 | False | 0 | 0 | 0 | False |
| Papillon_PAPILLON_GEPA_gpt-41-mini | 19 | 117 | 18 | 0.154 | 0.34 | 0.111 | 0.333 | 19 | 19 | 3 | 10 | 0.09 | 12 | 96.3 | 3 | 4 | False | 0 | 0 | 0 | True |
| hoverBench_HoverMultiHop_GEPA_gpt-41-mini | 22 | 133 | 21 | 0.158 | 0.176 | 0.143 | 0.18 | 22 | 22 | 11 | 56 | 0.187 | 19 | 54.33 | 5 | 6 | False | 0 | 0 | 0 | True |
| HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b | 21 | 166 | 20 | 0.12 | 0.152 | 0.139 | 0.241 | 21 | 21 | 8 | 44 | 0.147 | 19 | 60 | 5 | 5 | False | 0 | 0 | 0 | True |
| IFBench_IFBenchCoT2StageProgram_GEPA_qwen3-8b | 12 | 87 | 11 | 0.126 | 0.268 | 0.264 | 0.333 | 12 | 12 | 8 | 44 | 0.147 | 2 | 79.44 | 2 | 4 | False | 0 | 0 | 0 | True |
| Papillon_PAPILLON_GEPA_qwen3-8b | 20 | 70 | 19 | 0.271 | 0.388 | 0.157 | 0.329 | 20 | 20 | 5 | 22 | 0.198 | 17 | 90.39 | 3 | 6 | False | 0 | 0 | 0 | True |
| hoverBench_HoverMultiHop_GEPA_qwen3-8b | 22 | 105 | 21 | 0.2 | 0.226 | 0.038 | 0.133 | 22 | 22 | 9 | 58 | 0.193 | 21 | 53 | 4 | 5 | True | 0 | 0 | 0 | True |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_gpt-41-mini | 22 | 114 | 21 | 0.184 | 0.246 | 0.056 | 0.318 | 22 | 22 | 4 | 22 | 0.073 | 12 | 67.33 | 5 | 7 | False | 17 | 7 | 4 | True |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_gpt-41-mini | 12 | 46 | 11 | 0.239 | 0.381 | 0.093 | 0.279 | 12 | 12 | 6 | 27 | 0.09 | 7 | 90.03 | 3 | 4 | False | 7 | 3 | 3 | True |
| LiveBenchMathBench_CoT_GEPA-MERGE_gpt-41-mini | 14 | 43 | 13 | 0.302 | 0.382 | 0.326 | 0.233 | 14 | 14 | 9 | 22 | 0.182 | 9 | 75.47 | 2 | 4 | False | 12 | 0 | 0 | True |
| Papillon_PAPILLON_GEPA-MERGE_gpt-41-mini | 20 | 122 | 19 | 0.156 | 0.413 | 0.074 | 0.369 | 20 | 20 | 2 | 6 | 0.054 | 11 | 97.6 | 4 | 8 | False | 18 | 0 | 0 | True |
| hoverBench_HoverMultiHop_GEPA-MERGE_gpt-41-mini | 22 | 111 | 21 | 0.189 | 0.152 | 0.14 | 0.17 | 22 | 22 | 11 | 57 | 0.19 | 17 | 52.67 | 3 | 5 | False | 13 | 11 | 7 | True |
| HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b | 22 | 81 | 21 | 0.259 | 0.203 | 0.137 | 0.192 | 22 | 22 | 10 | 38 | 0.127 | 21 | 62 | 10 | 10 | True | 11 | 8 | 8 | True |
| IFBench_IFBenchCoT2StageProgram_GEPA-MERGE_qwen3-8b | 12 | 27 | 11 | 0.407 | 0.412 | 0.5 | 0.591 | 12 | 12 | 7 | 41 | 0.137 | 2 | 80.94 | 1 | 5 | False | 6 | 5 | 4 | True |
| Papillon_PAPILLON_GEPA-MERGE_qwen3-8b | 20 | 74 | 19 | 0.257 | 0.368 | 0.092 | 0.308 | 20 | 20 | 6 | 30 | 0.27 | 14 | 87.98 | 2 | 6 | False | 13 | 9 | 5 | True |
| hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b | 23 | 62 | 22 | 0.355 | 0.32 | 0.036 | 0.143 | 23 | 23 | 15 | 62 | 0.207 | 22 | 52.33 | 6 | 6 | True | 7 | 6 | 6 | True |
