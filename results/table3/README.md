# Table 3 runs (candidate selection), Qwen3-8B

SelectBestCandidate and BeamSearch for Table 3 of the paper, seeds 0, 1 and 2 (REPRODUCTION_PLAN.md section 2.14,
method fixed on 2026-11-20 before any Table 3 run). Baseline and GEPA are not repeated here. They are the Table 1
runs (`../table1_h100/`) and the PUPA block (`../pupa_gptoss/`).

- SelectBestCandidate is the authors' configuration (`Abl-SelectBestCandidate`).
- BeamSearch is not in the released code (F109). `Abl-BeamSearch` is our own reading of the paper's one-sentence
  definition, so this row is a replication, not a reproduction of the authors' code.
- PUPA is judged by gpt-oss-120b (plan section 2.10), so its numbers are not comparable with the paper's.

Layout: `seed_<s>/<Benchmark>_<Program>_<Optimizer>_qwen3-8b/`, as written by `scripts/run_experiments.py`. Large
text files are gzipped. The score is line 2 of `evaluation_results/evaluation_result.txt`. The check report of each
job is `../checks/<job>.md`. Server logs are too big for git and stay on Snellius (account `smeczynski1`, folder
`gepa-table3/archive/arbor_logs/<job>/`).

All jobs ran on account `smeczynski1` between 2026-11-21 and 2026-11-22. Each job used one H100, except PUPA, which
had gpt-oss-120b on a second H100. The SelectBestCandidate and BeamSearch runs of one task and seed shared one job
and its server.

## Analysis

`python analysis/table3_paired.py results --out results/table3/paired_numbers.json`, where `results` holds
`table1_h100/`, `pupa_gptoss/` and `table3/`. Output of 2026-11-22 11:53, run on a `git archive` of commit `49597d9`
with the files the script reads: `paired_output.md` and `paired_numbers.json`. The stop rule did not apply: every
run rebuilds to its reported score (largest gap 0.0045 points).

- Over HotpotQA, IFBench and HoVer, GEPA − SelectBestCandidate is +2.56 [+0.92, +4.18] (paper +6.39), and GEPA −
  BeamSearch is +0.76 [−0.82, +2.34] (paper +6.18).
- Holm (8 tests) rejects only HotpotQA GEPA − BeamSearch: +6.00 [+3.11, +8.78] (paper +5.00).

## Runs

| run | job | score | paper (one run) | check |
|---|---|---|---|---|
| seed_0 HotpotQA SelectBestCandidate | 29551537 | 57.33 | 58.33 | CHECK: one val request of the search lost to a client connection error (as F104); kept, † (search 1 / test 0); it could not change the search |
| seed_0 HotpotQA BeamSearch | 29551537 | 53.00 | 57.33 | job CHECK because of SelectBestCandidate; this run lost no request |
| seed_1 HotpotQA SelectBestCandidate | 29549311 | 55.00 | 58.33 | OK |
| seed_1 HotpotQA BeamSearch | 29549311 | 48.33 | 57.33 | OK |
| seed_2 HotpotQA SelectBestCandidate | 29549315 | 58.00 | 58.33 | OK |
| seed_2 HotpotQA BeamSearch | 29549315 | 62.00 | 57.33 | OK |
| seed_0 IFBench SelectBestCandidate | 29549308 | 29.76 | 30.44 | CHECK: test example 133 scored 0 after an NLTK thread error (F113); kept, † (search 0 / test 1) |
| seed_0 IFBench BeamSearch | 29549308 | 35.37 | 36.39 | job CHECK because of SelectBestCandidate; this run lost nothing |
| seed_1 IFBench SelectBestCandidate | 29549313 | 31.46 | 30.44 | OK |
| seed_1 IFBench BeamSearch | 29549313 | 33.33 | 36.39 | OK |
| seed_2 IFBench SelectBestCandidate | 29562374 | 28.06 | 30.44 | job CHECK because of BeamSearch; this run lost nothing |
| seed_2 IFBench BeamSearch | 29562374 | 38.78 | 36.39 | CHECK: the authors' `AlphabetLoopChecker` raised on test example 205, which scored 0 as in their code (F114); counted without a mark; one over-long reflection prompt rejected (allowed) |
| seed_0 HoVer SelectBestCandidate | 29551538 | 50.33 | 45.33 | CHECK: one test request lost to a client connection error (as F104); kept, † (search 0 / test 1) |
| seed_0 HoVer BeamSearch | 29551538 | 51.33 | 41.00 | job CHECK because of SelectBestCandidate; this run lost no request |
| seed_1 HoVer SelectBestCandidate | 29549312 | 47.00 | 45.33 | OK |
| seed_1 HoVer BeamSearch | 29549312 | 53.67 | 41.00 | OK |
| seed_2 HoVer SelectBestCandidate | 29561727 | 53.33 | 45.33 | OK |
| seed_2 HoVer BeamSearch | 29561727 | 50.67 | 41.00 | OK |
| seed_1 PUPA SelectBestCandidate | 29549314 | 92.64 | 85.45 (GPT-4.1-mini judge) | OK |
| seed_1 PUPA BeamSearch | 29549314 | 93.41 | 81.08 (GPT-4.1-mini judge) | OK |

Not here:
- **PUPA seed 0 failed twice (29549310, 29572300).** Both times the gpt-oss server stopped answering, after about 4 h
  and 3 h 40 min (F112). Both runs of each job are void. After the second failure, the user stopped PUPA at seed 1,
  so PUPA seed 2 was not run.
- **Jobs that failed before any result:**
  - smoke test 29528892: Arbor's start-up wait (F111);
  - smoke test 29535007: the released run script gives GEPA's arguments only to names that contain "GEPA" (F110);
  - 29549306 / 29549307: Arbor's start-up wait (F111). These two were rerun unchanged as 29551537 / 29551538.
- **Smoke test 29549303** passed. Its 700-call runs are a pipeline check, not results.

Compute: 18,864 SBU for all Table 3 jobs (sacct, 2026-11-22), under the cap of 21,000. The two void PUPA jobs used
3,084 of them.
