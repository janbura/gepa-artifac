# Paper tables vs. artifact runs

Paper cells checked: 58 (Tables 1, 2 and the SelectBestCandidate row of Table 3). Exact matches: 46. Mismatches: 1. Cells without a run in the artifact: 11. Artifact runs without a paper cell: 4.

## Mismatches

Artifact score differs from the paper by more than 0.005.

| model | optimizer | benchmark | paper | artifact | diff |
|---|---|---|---|---|---|
| GPT-4.1 mini | GEPA | LiveBench-Math | 64.13 | 59.43 | -4.70 |

## Paper cells with no run in the artifact

Only seed 0 is shipped; these runs are absent.

| model | optimizer | benchmark | paper | artifact | expected_run_folder |
|---|---|---|---|---|---|
| Qwen3 8B | Baseline | AIME-2025 | 27.33 | - | AIMEBench_CoT_Baseline_qwen3-8b |
| Qwen3 8B | Baseline | LiveBench-Math | 48.70 | - | LiveBenchMathBench_CoT_Baseline_qwen3-8b |
| Qwen3 8B | GRPO | AIME-2025 | 38.00 | - | AIMEBench_CoT_GRPO_qwen3-8b |
| Qwen3 8B | GRPO | LiveBench-Math | 51.26 | - | LiveBenchMathBench_CoT_GRPO_qwen3-8b |
| Qwen3 8B | MIPROv2 | AIME-2025 | 20.00 | - | AIMEBench_CoT_MIPROv2-Heavy_qwen3-8b |
| Qwen3 8B | MIPROv2 | LiveBench-Math | 46.60 | - | LiveBenchMathBench_CoT_MIPROv2-Heavy_qwen3-8b |
| Qwen3 8B | GEPA | AIME-2025 | 32.00 | - | AIMEBench_CoT_GEPA_qwen3-8b |
| Qwen3 8B | GEPA | LiveBench-Math | 51.95 | - | LiveBenchMathBench_CoT_GEPA_qwen3-8b |
| Qwen3 8B | GEPA+Merge | AIME-2025 | 32.00 | - | AIMEBench_CoT_GEPA-MERGE_qwen3-8b |
| Qwen3 8B | GEPA+Merge | LiveBench-Math | 51.95 | - | LiveBenchMathBench_CoT_GEPA-MERGE_qwen3-8b |
| GPT-4.1 mini | GEPA+Merge | AIME-2025 | 59.33 | - | AIMEBench_CoT_GEPA-MERGE_gpt-41-mini |

## Artifact runs with no cell in the paper tables

Scores not printed in Tables 1-3.

| model | optimizer | benchmark | artifact | run |
|---|---|---|---|---|
| GPT-4.1 mini | SelectBestCandidate | HotpotQA | 66.00 | HotpotQABench_HotpotMultiHop_Abl-SelectBestCandidate_gpt-41-mini |
| GPT-4.1 mini | SelectBestCandidate | IFBench | 49.49 | IFBench_IFBenchCoT2StageProgram_Abl-SelectBestCandidate_gpt-41-mini |
| GPT-4.1 mini | SelectBestCandidate | PUPA | 94.03 | Papillon_PAPILLON_Abl-SelectBestCandidate_gpt-41-mini |
| GPT-4.1 mini | SelectBestCandidate | HoVer | 51.00 | hoverBench_HoverMultiHop_Abl-SelectBestCandidate_gpt-41-mini |

## Exact matches

Artifact score equals the paper to two decimals.

| model | optimizer | benchmark | paper | artifact |
|---|---|---|---|---|
| Qwen3 8B | Baseline | HotpotQA | 42.33 | 42.33 |
| Qwen3 8B | Baseline | IFBench | 36.90 | 36.90 |
| Qwen3 8B | Baseline | HoVer | 35.33 | 35.33 |
| Qwen3 8B | Baseline | PUPA | 80.82 | 80.82 |
| Qwen3 8B | GRPO | HotpotQA | 43.33 | 43.33 |
| Qwen3 8B | GRPO | IFBench | 35.88 | 35.88 |
| Qwen3 8B | GRPO | HoVer | 38.67 | 38.67 |
| Qwen3 8B | GRPO | PUPA | 86.66 | 86.66 |
| Qwen3 8B | MIPROv2 | HotpotQA | 55.33 | 55.33 |
| Qwen3 8B | MIPROv2 | IFBench | 36.22 | 36.22 |
| Qwen3 8B | MIPROv2 | HoVer | 47.33 | 47.33 |
| Qwen3 8B | MIPROv2 | PUPA | 81.55 | 81.55 |
| Qwen3 8B | GEPA | HotpotQA | 62.33 | 62.33 |
| Qwen3 8B | GEPA | IFBench | 38.61 | 38.61 |
| Qwen3 8B | GEPA | HoVer | 52.33 | 52.33 |
| Qwen3 8B | GEPA | PUPA | 91.85 | 91.85 |
| Qwen3 8B | GEPA+Merge | HotpotQA | 64.33 | 64.33 |
| Qwen3 8B | GEPA+Merge | IFBench | 28.23 | 28.23 |
| Qwen3 8B | GEPA+Merge | HoVer | 51.67 | 51.67 |
| Qwen3 8B | GEPA+Merge | PUPA | 86.26 | 86.26 |
| Qwen3 8B | SelectBestCandidate | HotpotQA | 58.33 | 58.33 |
| Qwen3 8B | SelectBestCandidate | IFBench | 30.44 | 30.44 |
| Qwen3 8B | SelectBestCandidate | HoVer | 45.33 | 45.33 |
| Qwen3 8B | SelectBestCandidate | PUPA | 85.45 | 85.45 |
| GPT-4.1 mini | Baseline | HotpotQA | 38.00 | 38.00 |
| GPT-4.1 mini | Baseline | IFBench | 47.79 | 47.79 |
| GPT-4.1 mini | Baseline | HoVer | 46.33 | 46.33 |
| GPT-4.1 mini | Baseline | PUPA | 78.57 | 78.57 |
| GPT-4.1 mini | Baseline | AIME-2025 | 49.33 | 49.33 |
| GPT-4.1 mini | Baseline | LiveBench-Math | 58.20 | 58.20 |
| GPT-4.1 mini | MIPROv2 | HotpotQA | 58.00 | 58.00 |
| GPT-4.1 mini | MIPROv2 | IFBench | 49.15 | 49.15 |
| GPT-4.1 mini | MIPROv2 | HoVer | 48.33 | 48.33 |
| GPT-4.1 mini | MIPROv2 | PUPA | 83.37 | 83.37 |
| GPT-4.1 mini | MIPROv2 | AIME-2025 | 51.33 | 51.33 |
| GPT-4.1 mini | MIPROv2 | LiveBench-Math | 61.84 | 61.84 |
| GPT-4.1 mini | GEPA | HotpotQA | 69.00 | 69.00 |
| GPT-4.1 mini | GEPA | IFBench | 52.72 | 52.72 |
| GPT-4.1 mini | GEPA | HoVer | 51.67 | 51.67 |
| GPT-4.1 mini | GEPA | PUPA | 94.47 | 94.47 |
| GPT-4.1 mini | GEPA | AIME-2025 | 59.33 | 59.33 |
| GPT-4.1 mini | GEPA+Merge | HotpotQA | 65.67 | 65.67 |
| GPT-4.1 mini | GEPA+Merge | IFBench | 55.95 | 55.95 |
| GPT-4.1 mini | GEPA+Merge | HoVer | 56.67 | 56.67 |
| GPT-4.1 mini | GEPA+Merge | PUPA | 96.46 | 96.46 |
| GPT-4.1 mini | GEPA+Merge | LiveBench-Math | 64.13 | 64.13 |

## Notes

- LiveBench-Math / GEPA / GPT-4.1 mini: the paper prints 64.13, the artifact run scores 59.43. The artifact's GEPA+Merge run on LiveBench-Math scores 64.13, which is the number the paper prints for both GEPA and GEPA+Merge on that task.
- The paper reports identical GEPA and GEPA+Merge scores for the two single-module math tasks (merging cannot change a one-predictor program). The artifact has no AIME-2025 GEPA+Merge run for GPT-4.1 mini, so that paper cell (59.33) is the GEPA run's score.
- Qwen3 8B has no AIME-2025 or LiveBench-Math runs for any optimizer, so 10 cells of Table 1 cannot be checked.
- GRPO was only run with Qwen3 8B on the four non-math tasks.
- config.json keeps older names for the GEPA runs (optimizer_name GEPA8-Heavy, and benchmark_name AIMEBench2 for the AIME GEPA run); the folder names are used here.
