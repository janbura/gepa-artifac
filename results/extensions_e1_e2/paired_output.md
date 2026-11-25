## Extensions E1 and E2 against GEPA (plan 2.15)

E1 = GEPA-SubVal100 (search on 100 of the 300 val examples, final race of the top 5 on the other 200). E2 = GEPA-SubVal100-Gate (E1 plus the 50-example gate). GEPA = the Table 1 runs. Same rollout budget in all arms. Rebuild: every run gives its reported score (largest gap 0.0033 points).

### Test score (%, 300 test examples per run)

| Task | Method | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|---|
| HotpotQA | GEPA | 62.33 | 57.33 | 61.67 | 60.44 ± 2.72 |
| HotpotQA | E1 | 63.33 | 61.33† | 58.00 | 60.89 ± 2.69 |
| HotpotQA | E2 | 61.00 | 59.67 | 62.00 | 60.89 ± 1.17 |
| HoVer | GEPA | 48.67† | 51.00† | 53.33 | 51.00 ± 2.33 |
| HoVer | E1 | 50.00† | 51.67 | 51.67 | 51.11 ± 0.96 |
| HoVer | E2 | 51.67† | 55.67 | 51.67 | 53.00 ± 2.31 |

† = run with single requests lost in the search (rule of 2026-11-11): HoVer GEPA s0 2, HoVer GEPA s1 3, HoVer E1 s0 3, HoVer E2 s0 1, HotpotQA E1 s1 1. No test request was lost.

### Paired per-example tests (Holm over the 6 tests)

d = mean over the 300 test examples of A − B in points (each method: mean over its three seeds); 95 % percentile bootstrap (10,000 resamples); p = paired sign-flip test (10,000 sign vectors). The interval and p cover the test examples, not the run-to-run variation; the per-seed differences show that.

| Task | A − B | Per seed | d | 95 % interval | A better / B better | p | p (Holm) | Holm at 0.05 |
|---|---|---|---|---|---|---|---|---|
| HotpotQA | E1 − GEPA | +1.00 / +4.00 / -3.67 | +0.44 | [-2.11, +3.00] | 50 / 44 | 0.7962 | 1.0000 | no |
| HotpotQA | E2 − GEPA | -1.33 / +2.34 / +0.33 | +0.44 | [-2.11, +3.00] | 41 / 44 | 0.8028 | 1.0000 | no |
| HotpotQA | E2 − E1 | -2.33 / -1.66 / +4.00 | +0.00 | [-2.44, +2.44] | 41 / 44 | 1.0000 | 1.0000 | no |
| HoVer | E1 − GEPA | +1.33 / +0.67 / -1.66 | +0.11 | [-2.22, +2.56] | 45 / 43 | 1.0000 | 1.0000 | no |
| HoVer | E2 − GEPA | +3.00 / +4.67 / -1.66 | +2.00 | [-0.56, +4.56] | 47 / 36 | 0.1462 | 0.8771 | no |
| HoVer | E2 − E1 | +1.67 / +4.00 / +0.00 | +1.89 | [-0.78, +4.56] | 49 / 40 | 0.1921 | 0.9604 | no |

### Mean over the two tasks (no test)

| A − B | d | 95 % interval |
|---|---|---|
| E1 − GEPA | +0.28 | [-1.44, +2.06] |
| E2 − GEPA | +1.22 | [-0.61, +3.06] |
| E2 − E1 | +0.94 | [-0.83, +2.72] |

### Descriptive: the rows with a † run, without the † runs

| Task | A − B | seeds of A / of B | d | 95 % interval | d with all runs |
|---|---|---|---|---|---|
| HotpotQA | E1 − GEPA | [0, 2] / [0, 1, 2] | +0.22 | [-2.67, +3.17] | +0.44 |
| HotpotQA | E2 − E1 | [0, 1, 2] / [0, 2] | +0.22 | [-2.67, +3.11] | +0.00 |
| HoVer | E1 − GEPA | [1, 2] / [2] | -1.67 | [-5.50, +2.33] | +0.11 |
| HoVer | E2 − GEPA | [1, 2] / [2] | +0.33 | [-3.33, +4.00] | +2.00 |
| HoVer | E2 − E1 | [1, 2] / [1, 2] | +2.00 | [-1.17, +5.33] | +1.89 |

Checks from the saved state, all passed: in the 12 extension runs the finalists are the top 5 on the 100 search examples, the returned program is the best finalist on all 300, and the metric calls are 1 to 105 above the budget; the 311 gate decisions follow the rule (discard if more than 1 example behind the parent and no example unanswered).

### The search (descriptive; mean over the three seeds, range in brackets)

Metric calls: all calls counted against the budget. Val calls: full val passes of candidates (300 examples in GEPA, 100 in E1 and E2), gate passes of discarded children (50) and the final race (1,000). The rest are the three-example tests of parent and child.

| Task | Method | iterations | proposals | candidates (with the seed program) | gate decisions | discarded | metric calls | of which val | share on val |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | GEPA | 123 [93, 167] | 102 [77, 142] | 22 [21, 22] | – | – | 7114 [7062, 7152] | 6500 [6300, 6600] | 91 % |
| HotpotQA | E1 | 301 [238, 389] | 232 [195, 274] | 46 [43, 48] | – | – | 6959 [6944, 6970] | 5567 [5300, 5800] | 80 % |
| HotpotQA | E2 | 332 [313, 348] | 263 [237, 282] | 35 [32, 37] | 51 [48, 56] | 17 [12, 25] (34 %) | 6913 [6872, 6942] | 5333 [5250, 5450] | 77 % |
| HoVer | GEPA | 112 [101, 120] | 103 [97, 111] | 22 [22, 22] | – | – | 7220 [7182, 7266] | 6600 [6600, 6600] | 91 % |
| HoVer | E1 | 267 [230, 319] | 233 [201, 276] | 47 [45, 49] | – | – | 7131 [7106, 7156] | 5733 [5500, 5900] | 80 % |
| HoVer | E2 | 313 [283, 353] | 281 [254, 321] | 34 [29, 38] | 53 [52, 53] | 20 [15, 25] (37 %) | 7071 [7064, 7076] | 5383 [5150, 5550] | 76 % |

Gate passes in which the server left an example unanswered: 0. Children the guard kept: 0.

### The returned program on the val set (descriptive)

Seen = the seed's 100 search examples; other = the other 200 val examples. GEPA searched and selected on all 300, so both columns are selection-biased for GEPA. E1 and E2 searched on the 100 and picked the best of 5 finalists on all 300, so the 200 are only lightly selected on. The test score is the unbiased comparison.

| Task | Method | seen 100 (per seed) | mean | other 200 (per seed) | mean | test mean |
|---|---|---|---|---|---|---|
| HotpotQA | GEPA | 62.0 / 54.0 / 60.0 | 58.67 | 67.0 / 59.5 / 66.0 | 64.17 | 60.44 |
| HotpotQA | E1 | 64.0 / 56.0 / 64.0 | 61.33 | 64.5 / 64.5 / 59.5 | 62.83 | 60.89 |
| HotpotQA | E2 | 57.0 / 61.0 / 64.0 | 60.67 | 58.5 / 63.5 / 66.0 | 62.67 | 60.89 |
| HoVer | GEPA | 47.0 / 52.0 / 46.0 | 48.33 | 50.5 / 49.5 / 55.5 | 51.83 | 51.00 |
| HoVer | E1 | 55.0 / 55.0 / 52.0 | 54.00 | 48.0 / 51.5 / 56.0 | 51.83 | 51.11 |
| HoVer | E2 | 55.0 / 61.0 / 50.0 | 55.33 | 49.0 / 55.0 / 52.5 | 52.17 | 53.00 |

### Best score on the seed's 100 search examples against metric calls (descriptive; mean over the seeds)

For every arm: the best score on these 100 examples among the candidates found so far. GEPA scored these examples as part of its 300, but did not select on them alone, and E1 and E2 did, so this comparison favours E1 and E2. The maximum over more candidates of a 100-example score is also higher by chance alone.

| Task | Method | 1,000 calls | 2,000 calls | 3,000 calls | 4,000 calls | 5,000 calls | 5,871 calls | end of search | calls to reach GEPA's final score (per seed) |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | GEPA | 50.3 | 56.3 | 58.3 | 59.3 | 59.3 | 59.3 | 60.3 | 3,498 / 1,938 / 2,694 |
| HotpotQA | E1 | 56.7 | 59.7 | 60.7 | 60.7 | 62.0 | 62.0 | 62.0 | 1,382 / 1,870 / 2,086 |
| HotpotQA | E2 | 56.3 | 59.7 | 61.0 | 61.0 | 62.0 | 62.7 | 62.7 | never / 266 / 1,040 |
| HoVer | GEPA | 41.7 | 46.7 | 46.7 | 48.7 | 50.3 | 50.3 | 50.3 | 3,594 / 1,644 / 4,092 |
| HoVer | E1 | 50.3 | 50.3 | 51.0 | 52.0 | 53.3 | 54.7 | 54.7 | 184 / 378 / 2,130 |
| HoVer | E2 | 47.7 | 52.3 | 54.3 | 55.0 | 55.7 | 55.7 | 55.7 | 904 / 1,160 / 366 |

GEPA's final score = the score of the program GEPA returned on these 100 examples (not the best 100-example score among its candidates).

### The final race (descriptive)

| Task | Method | seed | finalists: seen 100 | finalists: other 200 | returned | best on the 100 | their other-200 scores (returned / best on the 100) |
|---|---|---|---|---|---|---|---|
| HotpotQA | E1 | 0 | 65 / 65 / 64 / 64 / 64 | 63.0 / 61.0 / 61.0 / 58.5 / 64.5 | 25 | 19 | 64.5 / 63.0 |
| HotpotQA | E1 | 1 | 57 / 56 / 56 / 56 / 55 | 63.5 / 64.5 / 59.0 / 62.0 / 62.5 | 16 | 18 | 64.5 / 63.5 |
| HotpotQA | E1 | 2 | 64 / 60 / 59 / 59 / 59 | 59.5 / 59.5 / 58.0 / 60.5 / 58.5 | 37 | 37 | 59.5 / 59.5 |
| HotpotQA | E2 | 0 | 59 / 59 / 58 / 57 / 56 | 55.5 / 56.0 / 54.0 / 58.5 / 53.0 | 31 | 32 | 58.5 / 55.5 |
| HotpotQA | E2 | 1 | 62 / 61 / 60 / 60 / 60 | 60.5 / 63.5 / 63.0 / 61.5 / 63.5 | 8 | 29 | 63.5 / 60.5 |
| HotpotQA | E2 | 2 | 67 / 64 / 64 / 64 / 63 | 60.5 / 63.5 / 66.0 / 65.5 / 59.5 | 16 | 28 | 66.0 / 60.5 |
| HoVer | E1 | 0 | 57 / 55 / 55 / 53 / 52 | 46.5 / 48.0 / 48.0 / 47.5 / 44.5 | 40 | 31 | 48.0 / 46.5 |
| HoVer | E1 | 1 | 55 / 55 / 54 / 54 / 54 | 51.5 / 49.5 / 47.5 / 49.5 / 49.5 | 31 | 31 | 51.5 / 51.5 |
| HoVer | E1 | 2 | 52 / 48 / 47 / 46 / 46 | 56.0 / 44.5 / 52.5 / 50.5 / 50.5 | 42 | 42 | 56.0 / 56.0 |
| HoVer | E2 | 0 | 55 / 55 / 54 / 54 / 53 | 49.0 / 48.0 / 46.0 / 49.5 / 45.0 | 25 | 25 | 49.0 / 49.0 |
| HoVer | E2 | 1 | 61 / 59 / 56 / 55 / 55 | 55.0 / 49.0 / 52.0 / 55.0 / 53.5 | 24 | 24 | 55.0 / 55.0 |
| HoVer | E2 | 2 | 51 / 51 / 50 / 50 / 49 | 46.0 / 51.5 / 49.5 / 52.5 / 53.0 | 11 | 8 | 52.5 / 46.0 |

The race returned another program than the best on the 100 in 7 of 12 runs; there the returned program is +3.14 points better on the other 200 on average (it was chosen with these 200, so part of this is selection). Over all 60 finalists the score on the 100 search examples is +1.63 points above the score on the other 200.

### Cost per job (two runs sharing one H100; descriptive)

GEPA jobs: GEPA and GEPA+Merge of one task and seed. Extension jobs: E1 and E2 of one task and seed.

| Task | Jobs | hours (per seed) | mean | SBU, mean | generated tokens, mean (M) | requests, mean |
|---|---|---|---|---|---|---|
| HotpotQA | GEPA + GEPA+Merge | 5.8 / 5.4 / 7.2 | 6.1 | 1177 | 17.3 | 32,308 |
| HotpotQA | E1 + E2 | 11.9 / 9.6 / 11.9 | 11.1 (1.82× GEPA) | 2138 | 22.8 | 38,713 |
| HoVer | GEPA + GEPA+Merge | 6.8 / 7.2 / 6.5 | 6.8 | 1305 | 23.3 | 35,532 |
| HoVer | E1 + E2 | 11.2 / 10.4 / 10.7 | 10.8 (1.58× GEPA) | 2067 | 27.9 | 41,253 |

