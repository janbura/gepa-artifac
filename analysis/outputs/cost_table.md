# Cost of the GPT-4.1 mini runs (USD, from evaluation_result.txt)

Optimizer columns: spent during optimization (0 for Baseline). Eval columns: the final test-set evaluation of the returned program. Costs are the artifact's own accounting and are not exactly tokens times list price.

| benchmark | optimizer | optimizer_cost | optimizer_input_tokens | optimizer_output_tokens | eval_cost | input_tokens | output_tokens | total_cost |
|---|---|---|---|---|---|---|---|---|
| HotpotQA | Baseline | 0.00 | 0 | 0 | 0.57 | 767,451 | 164,418 | 0.57 |
| HotpotQA | MIPROv2 | 19.19 | 39,872,835 | 3,254,327 | 0.87 | 2,006,275 | 161,814 | 20.06 |
| HotpotQA | GEPA | 20.73 | 21,433,903 | 3,305,995 | 0.87 | 1,477,051 | 227,865 | 21.60 |
| HotpotQA | GEPA+Merge | 19.13 | 13,801,004 | 1,675,390 | 0.88 | 1,476,758 | 233,628 | 20.01 |
| HotpotQA | SelectBestCandidate | 21.22 | 22,758,237 | 3,271,188 | 1.06 | 1,783,056 | 273,244 | 22.28 |
| IFBench | Baseline | 0.00 | 0 | 0 | 0.51 | 72,409 | 90,092 | 0.51 |
| IFBench | MIPROv2 | 7.24 | 9,961,526 | 2,613,496 | 0.70 | 959,936 | 330,206 | 7.95 |
| IFBench | GEPA | 7.72 | 5,372,392 | 2,237,085 | 0.69 | 599,910 | 281,325 | 8.41 |
| IFBench | GEPA+Merge | 7.74 | 4,890,476 | 2,030,398 | 0.66 | 589,056 | 267,109 | 8.40 |
| IFBench | SelectBestCandidate | 7.17 | 5,645,812 | 2,242,556 | 0.62 | 687,073 | 273,456 | 7.80 |
| HoVer | Baseline | 0.00 | 0 | 0 | 0.72 | 832,974 | 243,359 | 0.72 |
| HoVer | MIPROv2 | 22.80 | 43,822,464 | 5,087,521 | 0.98 | 2,205,501 | 277,564 | 23.78 |
| HoVer | GEPA | 23.77 | 21,060,650 | 4,836,444 | 1.07 | 1,606,613 | 329,104 | 24.84 |
| HoVer | GEPA+Merge | 24.52 | 16,892,100 | 4,282,283 | 1.05 | 1,039,401 | 236,948 | 25.57 |
| HoVer | SelectBestCandidate | 25.34 | 23,251,485 | 4,577,157 | 1.11 | 1,689,237 | 325,723 | 26.45 |
| PUPA | Baseline | 0.00 | 0 | 0 | 0.36 | 303,519 | 132,623 | 0.36 |
| PUPA | MIPROv2 | 5.59 | 10,101,491 | 1,361,542 | 0.44 | 769,562 | 128,351 | 6.03 |
| PUPA | GEPA | 6.42 | 6,236,890 | 2,071,522 | 0.42 | 434,568 | 120,082 | 6.84 |
| PUPA | GEPA+Merge | 6.36 | 6,590,681 | 2,054,873 | 0.41 | 452,667 | 117,489 | 6.77 |
| PUPA | SelectBestCandidate | 6.87 | 5,508,035 | 1,844,881 | 0.43 | 456,736 | 139,696 | 7.30 |
| AIME-2025 | Baseline | 0.00 | 0 | 0 | 0.92 | 24,399 | 347,277 | 0.92 |
| AIME-2025 | MIPROv2 | 11.36 | 5,795,259 | 5,157,799 | 0.81 | 112,940 | 293,757 | 12.17 |
| AIME-2025 | GEPA | 18.14 | 4,518,043 | 9,923,246 | 1.12 | 119,886 | 391,026 | 19.27 |
| LiveBench-Math | Baseline | 0.00 | 0 | 0 | 0.35 | 59,566 | 189,570 | 0.35 |
| LiveBench-Math | MIPROv2 | 5.09 | 7,196,344 | 1,878,195 | 0.37 | 426,517 | 192,847 | 5.46 |
| LiveBench-Math | GEPA | 5.22 | 2,729,962 | 2,591,433 | 0.41 | 195,937 | 232,390 | 5.64 |
| LiveBench-Math | GEPA+Merge | 5.62 | 2,358,424 | 2,771,103 | 0.42 | 143,647 | 227,966 | 6.04 |

Total for all 27 runs: optimizer 277.26 USD + evaluation 18.84 USD = 296.10 USD (301,090,658 input and 75,297,363 output tokens).

## Per optimizer (summed over benchmarks)

| optimizer | optimizer_cost | eval_cost | total_cost | runs |
|---|---|---|---|---|
| Baseline | 0.00 | 3.44 | 3.44 | 6 |
| MIPROv2 | 71.27 | 4.17 | 75.44 | 6 |
| GEPA | 82.01 | 4.58 | 86.59 | 6 |
| GEPA+Merge | 63.38 | 3.42 | 66.80 | 5 |
| SelectBestCandidate | 60.61 | 3.22 | 63.83 | 4 |

## GEPA run per benchmark

| benchmark | optimizer_cost | eval_cost | total_cost |
|---|---|---|---|
| LiveBench-Math | 5.22 | 0.41 | 5.64 |
| PUPA | 6.42 | 0.42 | 6.84 |
| IFBench | 7.72 | 0.69 | 8.41 |
| AIME-2025 | 18.14 | 1.12 | 19.27 |
| HotpotQA | 20.73 | 0.87 | 21.60 |
| HoVer | 23.77 | 1.07 | 24.84 |

Cheapest full GEPA run: LiveBench-Math at 5.64 USD; most expensive: HoVer at 24.84 USD; all six GEPA runs: 86.59 USD.
