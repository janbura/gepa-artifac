# Extension runs E1 and E2, Qwen3-8B

`GEPA-SubVal100` (E1) and `GEPA-SubVal100-Gate` (E2) on HotpotQA and HoVer, seeds 0, 1 and 2 (REPRODUCTION_PLAN.md
section 2.15, method fixed on 2026-11-22 and 2026-11-23 before any full run). The control arm is not repeated here: it
is the GEPA runs of Table 1 (`../table1_h100/`).

- E1 searches on a fixed 100 of the 300 val examples and races its top 5 candidates on the other 200.
- E2 is E1 plus a gate: a child is first scored on 50 of the 100 search examples and discarded if it trails its
  parent there by more than 1 example.
- Both use the paper's rollout budgets (6,871 and 7,051 metric calls), the race included.

Layout: `seed_<s>/<Benchmark>_<Program>_<Optimizer>_qwen3-8b/`, as written by `scripts/run_experiments.py` and packed
by `scripts/snellius/pack.sh`. Large text files are gzipped. The score is line 2 of
`evaluation_results/evaluation_result.txt`. `final_race.json` holds the search examples, the finalists with their
scores on the other 200 val examples, and the returned program. The check report of each job is
`../checks/<job>.md`. Server logs stay on Snellius (course account, login `scur0918`,
`~/repro_archive/arbor_logs/<job>/`).

All jobs ran on the course account from login `scur0918` between 2026-11-23 and 2026-11-25, commit `5c6d954`, one
H100 per job. The E1 and E2 runs of one task and seed shared one job and its server.

| Job | Runs | Wall-clock | SBU | Check |
|---|---|---|---|---|
| 29627113 | HotpotQA seed 0 | 11 h 53 | 2,282 | OK |
| 29627118 | HoVer seed 0 | 11 h 10 | 2,143 | CHECK (E1 3 and E2 1 lost requests) |
| 29655234 | HotpotQA seed 1 | 9 h 36 | 1,843 | CHECK (E1 1 lost request) |
| 29655237 | HoVer seed 1 | 10 h 26 | 2,002 | OK |
| 29655240 | HotpotQA seed 2 | 11 h 56 | 2,290 | OK |
| 29655243 | HoVer seed 2 | 10 h 42 | 2,055 | OK |

The lost requests are single client connection errors in the search (as F104). Smoke tests before the full runs:
29593517 (E1) and 29595345 (E2, before the gate guard), both capped and on an A100; their scores are not results.

## Analysis

`python analysis/extensions_paired.py results --out results/extensions_e1_e2/paired_numbers.json`, where `results`
holds `table1_h100/`, `extensions_e1_e2/` and `checks/`. Output of 2026-11-25: `paired_output.md` and
`paired_numbers.json`. The stop rule did not apply: every run rebuilds to its reported score (largest gap 0.0033
points). The script was written after the 12 scores were known and follows the method of section 2.15.

- E1 - GEPA over the two tasks: +0.28 [-1.44, +2.06]. E2 - GEPA: +1.22 [-0.61, +3.06]. E2 - E1: +0.94 [-0.83, +2.72].
- Holm (6 tests) rejects nothing; the smallest p before the correction is 0.15 (HoVer E2 - GEPA, +2.00).
- The extensions make 2.3 to 2.7 times as many proposals and take 1.6 to 1.8 times the GPU time.

Verdict and discussion: `REPRODUCTION_REPORT.md` section 12.
