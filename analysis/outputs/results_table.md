# Test scores in the artifact (seed 0)

## Qwen3 8B

| optimizer | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|---|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | - | - |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | - | - |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | - | - |
| GEPA | 62.33 | 38.61 | 52.33 | 91.85 | - | - |
| GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | - | - |
| SelectBestCandidate | 58.33 | 30.44 | 45.33 | 85.45 | - | - |

## GPT-4.1 mini

| optimizer | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|---|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 49.33 | 58.20 |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 51.33 | 61.84 |
| GEPA | 69.00 | 52.72 | 51.67 | 94.47 | 59.33 | 59.43 |
| GEPA+Merge | 65.67 | 55.95 | 56.67 | 96.46 | - | 64.13 |
| SelectBestCandidate | 66.00 | 49.49 | 51.00 | 94.03 | - | - |
