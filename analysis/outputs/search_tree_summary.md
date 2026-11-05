# Search-tree statistics, mean per optimizer and model

| optimizer | model | n_runs | n_candidates | n_iterations | acceptance_rate | accept_rate_tested | frac_sel_seed | frac_sel_current_best | pareto_front_size | pareto_front_size_excl_ties | pareto_front_nondominated | n_best_off_front | frac_best_off_front | best_depth | max_depth | best_is_last | n_merge_invoked | n_merge_attempts | n_merges_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Abl-SelectBestCandidate | gpt-41-mini | 4 | 18.25 | 138.75 | 0.139 | 0.247 | 0.038 | 1 | 18.25 | 18.25 | 7 | 31.5 | 0.132 | 4.75 | 5.5 | 0.25 | 0 | 0 | 0 |
| Abl-SelectBestCandidate | qwen3-8b | 4 | 18.5 | 102.5 | 0.176 | 0.259 | 0.064 | 1 | 18.5 | 18.5 | 10.75 | 46 | 0.189 | 5.5 | 6.25 | 0.25 | 0 | 0 | 0 |
| GEPA | gpt-41-mini | 6 | 19.333 | 104 | 0.19 | 0.288 | 0.139 | 0.254 | 19.333 | 19.333 | 7.167 | 25.667 | 0.15 | 3.667 | 4.667 | 0.167 | 0 | 0 | 0 |
| GEPA | qwen3-8b | 4 | 18.75 | 107 | 0.18 | 0.258 | 0.15 | 0.259 | 18.75 | 18.75 | 7.5 | 42 | 0.171 | 3.5 | 5 | 0.25 | 0 | 0 | 0 |
| GEPA-MERGE | gpt-41-mini | 5 | 18 | 87.2 | 0.214 | 0.315 | 0.138 | 0.274 | 18 | 18 | 6.4 | 26.8 | 0.118 | 3.4 | 5.6 | 0 | 13.4 | 4.2 | 2.8 |
| GEPA-MERGE | qwen3-8b | 4 | 19.25 | 61 | 0.32 | 0.326 | 0.191 | 0.308 | 19.25 | 19.25 | 9.5 | 42.75 | 0.185 | 4.75 | 6.75 | 0.5 | 9.25 | 7 | 5.75 |
