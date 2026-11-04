# Paired significance of optimizer differences on the test sets (artifact, seed 0)

diff = A - B in points; CI = 95% bootstrap over test examples; p = paired sign-flip test; p_holm = Holm correction over all rows; min_detectable_diff = smallest difference this test set can detect with 80% power. Only test-set noise is included, not seed-to-seed noise.

| benchmark      | model        | comparison           |   n_examples |   diff |   ci_low |   ci_high |   p_value |   min_detectable_diff |   n_A_better |   n_B_better |   p_holm | significant_holm   |
|:---------------|:-------------|:---------------------|-------------:|-------:|---------:|----------:|----------:|----------------------:|-------------:|-------------:|---------:|:-------------------|
| HotpotQA       | GPT-4.1 mini | GEPA - Baseline      |          300 |  31    |    25.33 |     36.67 |    0.0001 |                  8.16 |           99 |            6 |   0.0048 | True               |
| HotpotQA       | GPT-4.1 mini | GEPA - MIPROv2       |          300 |  11    |     6.33 |     15.67 |    0.0001 |                  6.83 |           45 |           12 |   0.0048 | True               |
| HotpotQA       | GPT-4.1 mini | GEPA+Merge - GEPA    |          299 |  -3.34 |    -6.35 |     -0.33 |    0.0483 |                  4.37 |            6 |           16 |   0.9701 | False              |
| HotpotQA       | GPT-4.1 mini | GEPA - SelectBest    |          300 |   3    |    -0.67 |      6.67 |    0.1646 |                  5.35 |           21 |           12 |   1      | False              |
| HotpotQA       | GPT-4.1 mini | GEPA+Merge - MIPROv2 |          299 |   7.69 |     3.68 |     12.04 |    0.0007 |                  6.02 |           33 |           10 |   0.0245 | True               |
| HotpotQA       | Qwen3 8B     | GEPA - Baseline      |          293 |  20.48 |    15.02 |     25.94 |    0.0001 |                  7.99 |           71 |           11 |   0.0048 | True               |
| HotpotQA       | Qwen3 8B     | GEPA - MIPROv2       |          292 |   7.88 |     3.08 |     12.67 |    0.002  |                  6.74 |           37 |           14 |   0.062  | False              |
| HotpotQA       | Qwen3 8B     | GEPA+Merge - GEPA    |          293 |   0.68 |    -3.41 |      4.78 |    0.8672 |                  5.74 |           19 |           17 |   1      | False              |
| HotpotQA       | Qwen3 8B     | GEPA - SelectBest    |          293 |   5.12 |     0.68 |      9.56 |    0.0346 |                  6.37 |           30 |           15 |   0.8303 | False              |
| HotpotQA       | Qwen3 8B     | GEPA+Merge - MIPROv2 |          299 |   8.7  |     4.35 |     13.38 |    0.0005 |                  6.48 |           38 |           12 |   0.0185 | True               |
| IFBench        | GPT-4.1 mini | GEPA - Baseline      |          291 |   3.95 |     0    |      8.08 |    0.0642 |                  5.69 |           25 |           14 |   1      | False              |
| IFBench        | GPT-4.1 mini | GEPA - MIPROv2       |          294 |   3.57 |    -0.34 |      7.65 |    0.1002 |                  5.75 |           26 |           16 |   1      | False              |
| IFBench        | GPT-4.1 mini | GEPA+Merge - GEPA    |          294 |   3.23 |    -0.68 |      7.14 |    0.1339 |                  5.68 |           28 |           16 |   1      | False              |
| IFBench        | GPT-4.1 mini | GEPA - SelectBest    |          294 |   3.23 |    -1.02 |      7.48 |    0.1581 |                  6.1  |           29 |           19 |   1      | False              |
| IFBench        | GPT-4.1 mini | GEPA+Merge - MIPROv2 |          294 |   6.8  |     2.04 |     11.57 |    0.0074 |                  6.95 |           42 |           20 |   0.2146 | False              |
| IFBench        | Qwen3 8B     | GEPA - Baseline      |          264 |   1.7  |    -2.84 |      6.25 |    0.522  |                  6.69 |           26 |           22 |   1      | False              |
| IFBench        | Qwen3 8B     | GEPA - MIPROv2       |          269 |   4.46 |    -1.12 |     10.04 |    0.1307 |                  7.91 |           41 |           26 |   1      | False              |
| IFBench        | Qwen3 8B     | GEPA+Merge - GEPA    |          268 | -11.75 |   -17.35 |     -6.53 |    0.0001 |                  7.73 |           17 |           51 |   0.0048 | True               |
| IFBench        | Qwen3 8B     | GEPA - SelectBest    |          267 |   8.99 |     4.12 |     13.86 |    0.0008 |                  7.07 |           41 |           14 |   0.0272 | True               |
| IFBench        | Qwen3 8B     | GEPA+Merge - MIPROv2 |          268 |  -8.21 |   -13.99 |     -2.43 |    0.0078 |                  8.42 |           26 |           48 |   0.2184 | False              |
| HoVer          | GPT-4.1 mini | GEPA - Baseline      |          300 |   5.33 |     0.67 |     10    |    0.0401 |                  6.82 |           35 |           19 |   0.9222 | False              |
| HoVer          | GPT-4.1 mini | GEPA - MIPROv2       |          300 |   3.33 |    -1.33 |      8    |    0.2015 |                  6.59 |           30 |           20 |   1      | False              |
| HoVer          | GPT-4.1 mini | GEPA+Merge - GEPA    |          300 |   5    |     0.33 |      9.67 |    0.0462 |                  6.63 |           33 |           18 |   0.9701 | False              |
| HoVer          | GPT-4.1 mini | GEPA - SelectBest    |          299 |   0.67 |    -3.34 |      4.68 |    0.876  |                  5.78 |           20 |           18 |   1      | False              |
| HoVer          | GPT-4.1 mini | GEPA+Merge - MIPROv2 |          300 |   8.33 |     3.67 |     13.33 |    0.0013 |                  6.67 |           39 |           14 |   0.0429 | True               |
| HoVer          | Qwen3 8B     | GEPA - Baseline      |          300 |  17    |    12    |     22.01 |    0.0001 |                  7.14 |           59 |            8 |   0.0048 | True               |
| HoVer          | Qwen3 8B     | GEPA - MIPROv2       |          300 |   5    |     0.67 |      9.33 |    0.033  |                  6.22 |           30 |           15 |   0.8249 | False              |
| HoVer          | Qwen3 8B     | GEPA - GRPO          |          300 |  13.67 |     8.67 |     19    |    0.0001 |                  7.44 |           55 |           14 |   0.0048 | True               |
| HoVer          | Qwen3 8B     | GEPA+Merge - GEPA    |          300 |  -0.67 |    -4.67 |      3.33 |    0.8688 |                  5.61 |           17 |           19 |   1      | False              |
| HoVer          | Qwen3 8B     | GEPA - SelectBest    |          300 |   7    |     3    |     11    |    0.0017 |                  5.88 |           31 |           10 |   0.0544 | False              |
| HoVer          | Qwen3 8B     | GEPA+Merge - MIPROv2 |          300 |   4.33 |     0.33 |      8.33 |    0.0413 |                  5.49 |           24 |           11 |   0.9222 | False              |
| PUPA           | GPT-4.1 mini | GEPA - Baseline      |          185 |  14.67 |    10.52 |     18.78 |    0.0001 |                  5.87 |           75 |           13 |   0.0048 | True               |
| PUPA           | GPT-4.1 mini | GEPA - MIPROv2       |          185 |  10.77 |     7.08 |     14.55 |    0.0001 |                  5.26 |           63 |           14 |   0.0048 | True               |
| PUPA           | GPT-4.1 mini | GEPA+Merge - GEPA    |          185 |   2.51 |    -0.21 |      5.36 |    0.0855 |                  4.02 |           21 |           12 |   1      | False              |
| PUPA           | GPT-4.1 mini | GEPA - SelectBest    |          185 |  -0.37 |    -3.33 |      2.52 |    0.8049 |                  4.18 |           18 |           19 |   1      | False              |
| PUPA           | GPT-4.1 mini | GEPA+Merge - MIPROv2 |          185 |  13.27 |     9.79 |     16.83 |    0.0001 |                  5.03 |           70 |            9 |   0.0048 | True               |
| PUPA           | Qwen3 8B     | GEPA - Baseline      |          185 |  11.15 |     6.96 |     15.37 |    0.0001 |                  5.97 |           64 |           17 |   0.0048 | True               |
| PUPA           | Qwen3 8B     | GEPA - MIPROv2       |          185 |  10.2  |     6.23 |     14.35 |    0.0001 |                  5.82 |           63 |           20 |   0.0048 | True               |
| PUPA           | Qwen3 8B     | GEPA - GRPO          |          185 |   4.76 |     0.93 |      8.45 |    0.0141 |                  5.37 |           50 |           23 |   0.3807 | False              |
| PUPA           | Qwen3 8B     | GEPA+Merge - GEPA    |          185 |  -5.4  |    -9.06 |     -1.82 |    0.004  |                  5.21 |           19 |           37 |   0.12   | False              |
| PUPA           | Qwen3 8B     | GEPA - SelectBest    |          185 |   6.65 |     2.93 |     10.41 |    0.0005 |                  5.27 |           43 |           21 |   0.0185 | True               |
| PUPA           | Qwen3 8B     | GEPA+Merge - MIPROv2 |          185 |   4.8  |     0.57 |      9.15 |    0.0287 |                  6.09 |           52 |           26 |   0.7461 | False              |
| AIME           | GPT-4.1 mini | GEPA - Baseline      |           30 |  10.67 |     0    |     22.67 |    0.0947 |                 16.05 |            7 |            4 |   1      | False              |
| AIME           | GPT-4.1 mini | GEPA - MIPROv2       |           30 |   8.67 |    -2.67 |     19.33 |    0.1856 |                 16.26 |            9 |            2 |   1      | False              |
| LiveBench-Math | GPT-4.1 mini | GEPA - Baseline      |          126 |   1.23 |    -5.04 |      7.65 |    0.7352 |                  9    |           16 |           16 |   1      | False              |
| LiveBench-Math | GPT-4.1 mini | GEPA - MIPROv2       |          126 |  -2.41 |    -7.92 |      3.07 |    0.3834 |                  7.79 |           10 |           18 |   1      | False              |
| LiveBench-Math | GPT-4.1 mini | GEPA+Merge - GEPA    |          126 |   4.7  |    -0.5  |     10.05 |    0.0767 |                  7.44 |           19 |           10 |   1      | False              |
| LiveBench-Math | GPT-4.1 mini | GEPA+Merge - MIPROv2 |          126 |   2.29 |    -3.43 |      7.91 |    0.4483 |                  8.1  |           14 |           16 |   1      | False              |

## Completeness of the per-example test logs

| benchmark      | model        | optimizer   |   reported |   log_lines |   unique_examples |   log_mean |   log_minus_reported |
|:---------------|:-------------|:------------|-----------:|------------:|------------------:|-----------:|---------------------:|
| AIME           | GPT-4.1 mini | Baseline    |      49.33 |         150 |                30 |      49.33 |                 0    |
| AIME           | GPT-4.1 mini | GEPA        |      59.33 |         151 |                30 |      60    |                 0.67 |
| AIME           | GPT-4.1 mini | MIPROv2     |      51.33 |         150 |                30 |      51.33 |                 0    |
| HotpotQA       | GPT-4.1 mini | SelectBest  |      66    |         300 |               300 |      66    |                 0    |
| HotpotQA       | Qwen3 8B     | SelectBest  |      58.33 |         300 |               300 |      58.33 |                 0    |
| HotpotQA       | GPT-4.1 mini | Baseline    |      38    |         300 |               300 |      38    |                 0    |
| HotpotQA       | Qwen3 8B     | Baseline    |      42.33 |         300 |               300 |      42.33 |                 0    |
| HotpotQA       | GPT-4.1 mini | GEPA+Merge  |      65.67 |         299 |               299 |      65.89 |                 0.22 |
| HotpotQA       | Qwen3 8B     | GEPA+Merge  |      64.33 |         300 |               300 |      64.33 |                 0    |
| HotpotQA       | GPT-4.1 mini | GEPA        |      69    |         300 |               300 |      69    |                 0    |
| HotpotQA       | Qwen3 8B     | GEPA        |      62.33 |         294 |               293 |      63.82 |                 1.49 |
| HotpotQA       | Qwen3 8B     | GRPO        |      43.33 |           0 |                 0 |     nan    |               nan    |
| HotpotQA       | GPT-4.1 mini | MIPROv2     |      58    |         300 |               300 |      58    |                -0    |
| HotpotQA       | Qwen3 8B     | MIPROv2     |      55.33 |         299 |               299 |      55.52 |                 0.19 |
| IFBench        | GPT-4.1 mini | SelectBest  |      49.49 |         294 |               294 |      49.49 |                -0    |
| IFBench        | Qwen3 8B     | SelectBest  |      30.44 |         279 |               277 |      32.31 |                 1.87 |
| IFBench        | GPT-4.1 mini | Baseline    |      47.79 |         292 |               291 |      48.28 |                 0.49 |
| IFBench        | Qwen3 8B     | Baseline    |      36.9  |         280 |               276 |      39.31 |                 2.41 |
| IFBench        | GPT-4.1 mini | GEPA+Merge  |      55.95 |         294 |               294 |      55.95 |                 0    |
| IFBench        | Qwen3 8B     | GEPA+Merge  |      28.23 |         281 |               280 |      29.64 |                 1.41 |
| IFBench        | GPT-4.1 mini | GEPA        |      52.72 |         294 |               294 |      52.72 |                 0    |
| IFBench        | Qwen3 8B     | GEPA        |      38.61 |         280 |               280 |      40.54 |                 1.93 |
| IFBench        | Qwen3 8B     | GRPO        |      35.88 |           0 |                 0 |     nan    |               nan    |
| IFBench        | GPT-4.1 mini | MIPROv2     |      49.15 |         294 |               294 |      49.15 |                -0    |
| IFBench        | Qwen3 8B     | MIPROv2     |      36.22 |         286 |               281 |      37.9  |                 1.68 |
| LiveBench-Math | GPT-4.1 mini | Baseline    |      58.2  |         126 |               126 |      58.2  |                -0    |
| LiveBench-Math | GPT-4.1 mini | GEPA+Merge  |      64.13 |         126 |               126 |      64.13 |                -0    |
| LiveBench-Math | GPT-4.1 mini | GEPA        |      59.43 |         126 |               126 |      59.43 |                -0    |
| LiveBench-Math | GPT-4.1 mini | MIPROv2     |      61.84 |         126 |               126 |      61.84 |                -0    |
| PUPA           | GPT-4.1 mini | SelectBest  |      94.03 |         196 |               185 |      94.04 |                 0.01 |
| PUPA           | Qwen3 8B     | SelectBest  |      85.45 |         196 |               185 |      84.78 |                -0.67 |
| PUPA           | GPT-4.1 mini | Baseline    |      78.57 |         196 |               185 |      79    |                 0.43 |
| PUPA           | Qwen3 8B     | Baseline    |      80.82 |         196 |               185 |      80.29 |                -0.53 |
| PUPA           | GPT-4.1 mini | GEPA+Merge  |      96.46 |         196 |               185 |      96.18 |                -0.28 |
| PUPA           | Qwen3 8B     | GEPA+Merge  |      86.26 |         196 |               185 |      86.04 |                -0.22 |
| PUPA           | GPT-4.1 mini | GEPA        |      94.47 |         196 |               185 |      93.67 |                -0.8  |
| PUPA           | Qwen3 8B     | GEPA        |      91.85 |         196 |               185 |      91.44 |                -0.41 |
| PUPA           | Qwen3 8B     | GRPO        |      86.66 |         196 |               185 |      86.68 |                 0.02 |
| PUPA           | GPT-4.1 mini | MIPROv2     |      83.37 |         196 |               185 |      82.9  |                -0.47 |
| PUPA           | Qwen3 8B     | MIPROv2     |      81.55 |         196 |               185 |      81.24 |                -0.31 |
| HoVer          | GPT-4.1 mini | SelectBest  |      51    |         299 |               299 |      51.17 |                 0.17 |
| HoVer          | Qwen3 8B     | SelectBest  |      45.33 |         300 |               300 |      45.33 |                 0    |
| HoVer          | GPT-4.1 mini | Baseline    |      46.33 |         300 |               300 |      46.33 |                 0    |
| HoVer          | Qwen3 8B     | Baseline    |      35.33 |         300 |               300 |      35.33 |                 0    |
| HoVer          | GPT-4.1 mini | GEPA+Merge  |      56.67 |         300 |               300 |      56.67 |                -0    |
| HoVer          | Qwen3 8B     | GEPA+Merge  |      51.67 |         300 |               300 |      51.67 |                -0    |
| HoVer          | GPT-4.1 mini | GEPA        |      51.67 |         300 |               300 |      51.67 |                -0    |
| HoVer          | Qwen3 8B     | GEPA        |      52.33 |         300 |               300 |      52.33 |                 0    |
| HoVer          | Qwen3 8B     | GRPO        |      38.67 |         300 |               300 |      38.67 |                -0    |
| HoVer          | GPT-4.1 mini | MIPROv2     |      48.33 |         300 |               300 |      48.33 |                 0    |
| HoVer          | Qwen3 8B     | MIPROv2     |      47.33 |         300 |               300 |      47.33 |                 0    |
