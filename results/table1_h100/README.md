# Table 1 runs (all on one H100 per job), Qwen3-8B

Only runs that count in Table 1 are here: every one ran on one H100 with vLLM 0.9.1 (decision of 2026-11-09 in
REPRODUCTION_PLAN.md section 2.4). The A100 runs are the hardware check: `../a100_hardware_check/` (optimizer runs) and
`../b_baselines/seed_*` (A100 Baselines). They never enter the table.

Layout: `seed_<s>/<Benchmark>_<Program>_<Optimizer>_qwen3-8b/` as written by `scripts/run_experiments.py`. Large text
files are gzipped; `optimizer_logs.tgz` holds MIPROv2's evaluated programs. The score is line 2 of
`evaluation_results/evaluation_result.txt`; the check report of each job is `../checks/<job>.md`. Server request and
response logs are too big for git and stay on Snellius in `~/repro_archive/arbor_logs/<job>/`.

Where the runs came from on Snellius: IFBench, AIME and LiveBench-Math MIPROv2 and GEPA runs from `~/repro_runs`;
the new H100 Baselines and the HotpotQA/HoVer MIPROv2 and GEPA runs from `~/repro_h100`; the IFBench seed-2 Baseline
from the H100 test `~/h100_pilot` (job 28971148, also kept as `../b_baselines/h100_test/`).

Final verdict (pre-registered rule): `python analysis/table1_verdict.py results/table1_h100`

| run | job | score | paper | check |
|---|---|---|---|---|
| seed_0 IFBench MIPROv2 | 28994839 | 35.54 | 36.22 | OK, inside the seed-0 gate |
| seed_0 LiveBench-Math MIPROv2 | 28994837 | 48.48 | 46.60 | OK, inside |
| seed_0 AIME MIPROv2 | 28994833 | 44.67 | 20.00 | gate PAUSE (F97); technically clean; the user chose to continue AIME |
| seed_1 AIME MIPROv2 | 29014358 | 26.00 | 20.00 | OK |
| seed_2 AIME MIPROv2 | 29014360 | 32.00 | 20.00 | OK |
| seed_1 LiveBench-Math MIPROv2 | 28995006 | 52.16 | 46.60 | OK (no gate for seeds 1-2) |
| seed_2 LiveBench-Math MIPROv2 | 28995016 | 46.05 | 46.60 | OK |
| seed_1 IFBench MIPROv2 | 28995008 | 41.50 | 36.22 | OK (no gate for seeds 1-2) |
| seed_2 IFBench MIPROv2 | 29114232 | 40.82 | 36.22 | OK (third attempt; F99, F102) |
| seed_0 HotpotQA MIPROv2 | 29015003 | 56.33 | 55.33 | OK, inside |
| seed_1 HotpotQA MIPROv2 | 29015005 | 60.33 | 55.33 | OK |
| seed_2 HotpotQA MIPROv2 | 29015007 | 58.33 | 55.33 | OK |
| seed_0 HoVer MIPROv2 | 29015009 | 42.33 | 47.33 | OK, inside |
| seed_1 HoVer MIPROv2 | 29015012 | 48.67 | 47.33 | OK |
| seed_2 HoVer MIPROv2 | 29015014 | 46.33 | 47.33 | OK |
| seed_2 IFBench Baseline | 28971148 | 34.52 | 36.90 | OK (H100 test run, reused, not rerun) |
| seed_0 LiveBench-Math GEPA | 29003749 | 50.91 | 51.95 | OK, inside |
| seed_0 LiveBench-Math GEPA+Merge | 29003749 | 52.94 | 51.95 | OK, inside (0 merges: one predictor) |
| seed_1 LiveBench-Math GEPA | 29014519 | 49.11 | 51.95 | OK |
| seed_1 LiveBench-Math GEPA+Merge | 29014519 | 42.63 | 51.95 | OK (0 merges) |
| seed_2 LiveBench-Math GEPA | 29014522 | 49.73 | 51.95 | OK |
| seed_2 LiveBench-Math GEPA+Merge | 29014522 | 55.23 | 51.95 | OK (0 merges) |
| seed_0 IFBench GEPA | 28994883 | 33.67 | 38.61 | OK, inside (one over-long reflection rejected, allowed) |
| seed_0 IFBench GEPA+Merge | 28994883 | 36.56 | 28.23 | OK, inside (1 merge) |
| seed_1 IFBench GEPA | 29014525 | 31.29 | 38.61 | OK |
| seed_1 IFBench GEPA+Merge | 29014525 | 36.73 | 28.23 | OK (4 merges) |
| seed_2 IFBench GEPA | 29114238 | 34.01 | 38.61 | CHECK: lost requests search 13 / test 0 (server batch failures, F106), kept (rule of 2026-11-11), † |
| seed_2 IFBench GEPA+Merge | 29114238 | 31.97 | 28.23 | CHECK: lost requests search 0 / test 34 (F106): 34 of 294 test examples without an answer, counted as wrong (score up to 11.56 points too low), kept, † (6 merges) |
| seed_0 AIME GEPA | 29014355 | 34.67 | 32.00 | OK, inside |
| seed_0 AIME GEPA+Merge | 29014355 | 24.67 | 32.00 | OK, inside (0 merges: one predictor) |
| seed_1 AIME GEPA | 29030593 | 34.67 | 32.00 | OK |
| seed_1 AIME GEPA+Merge | 29030593 | 36.00 | 32.00 | OK (0 merges) |
| seed_2 AIME GEPA | 29030597 | 34.67 | 32.00 | OK |
| seed_2 AIME GEPA+Merge | 29030597 | 39.33 | 32.00 | OK (0 merges) |
| seed_0 HotpotQA GEPA | 29015016 | 62.33 | 62.33 | OK, inside |
| seed_0 HotpotQA GEPA+Merge | 29015016 | 60.67 | 64.33 | OK, inside (8 merges) |
| seed_1 HotpotQA GEPA | 29030578 | 57.33 | 62.33 | OK |
| seed_1 HotpotQA GEPA+Merge | 29030578 | 55.00 | 64.33 | OK (10 merges) |
| seed_2 HotpotQA GEPA | 29030583 | 61.67 | 62.33 | job CHECK because of GEPA+Merge; this run lost no request |
| seed_2 HotpotQA GEPA+Merge | 29030583 | 57.00 | 64.33 | CHECK: lost requests search 1 / test 0 (F104), kept (rule of 2026-11-11), † (9 merges) |
| seed_0 HoVer GEPA | 29015028 | 48.67 | 52.33 | CHECK: lost requests search 2 / test 0 (F104), kept, †; inside the gate |
| seed_0 HoVer GEPA+Merge | 29015028 | 48.67 | 51.67 | CHECK: lost requests search 1 / test 0 (F104), kept, †; inside the gate (12 merges) |
| seed_1 HoVer GEPA | 29030587 | 51.00 | 52.33 | CHECK: lost requests search 3 / test 0 (F104), kept, † |
| seed_1 HoVer GEPA+Merge | 29030587 | 51.00 | 51.67 | CHECK: lost requests search 1 / test 0 (F104), kept, † (10 merges) |
| seed_2 HoVer GEPA | 29030590 | 53.33 | 52.33 | job CHECK because of GEPA+Merge; this run lost no request |
| seed_2 HoVer GEPA+Merge | 29030590 | 51.33 | 51.67 | CHECK: lost requests search 2 / test 0 (F104), kept, † (7 merges) |
| seed_0 HotpotQA Baseline | 29014995 | 39.33 | 42.33 | OK, inside |
| seed_1 HotpotQA Baseline | 29014995 | 38.00 | 42.33 | OK |
| seed_2 HotpotQA Baseline | 29014995 | 39.00 | 42.33 | OK |
| seed_0 HoVer Baseline | 29014995 | 38.33 | 35.33 | OK, inside |
| seed_1 HoVer Baseline | 29014995 | 35.33 | 35.33 | OK |
| seed_2 HoVer Baseline | 29014995 | 37.67 | 35.33 | OK |
| seed_0 IFBench Baseline | 29014997 | 35.20 | 36.90 | OK, inside |
| seed_1 IFBench Baseline | 29014999 | 34.18 | 36.90 | OK |
| seed_0 AIME Baseline | 29014997 | 25.33 | 27.33 | OK, inside |
| seed_1 AIME Baseline | 29014999 | 25.33 | 27.33 | OK |
| seed_2 AIME Baseline | 29015001 | 32.67 | 27.33 | OK |
| seed_0 LiveBench-Math Baseline | 29014997 | 51.31 | 48.70 | OK, inside |
| seed_1 LiveBench-Math Baseline | 29014999 | 51.95 | 48.70 | OK |
| seed_2 LiveBench-Math Baseline | 29015001 | 51.18 | 48.70 | OK |
