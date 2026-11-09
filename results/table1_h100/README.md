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
| seed_1 LiveBench-Math MIPROv2 | 28995006 | 52.16 | 46.60 | OK (no gate for seeds 1-2) |
| seed_2 LiveBench-Math MIPROv2 | 28995016 | 46.05 | 46.60 | OK |
| seed_2 IFBench Baseline | 28971148 | 34.52 | 36.90 | OK (H100 test run, reused, not rerun) |
