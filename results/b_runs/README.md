# B's optimizer runs (MIPROv2-Heavy, GEPA, GEPA+Merge), Qwen3-8B

One folder per run, `seed_<s>/<Benchmark>_<Program>_<Optimizer>_qwen3-8b/`, copied from `~/repro_runs/experiment_runs` on Snellius
(checkout `~/gepa-repro`, commit 6ea0b87). Large text files are gzipped; `optimizer_logs.tgz` holds MIPROv2's evaluated programs.
The score is line 2 of `evaluation_results/evaluation_result.txt`. The check report of each job is in `../checks/<job>.md`.
The server request/response logs of these jobs are too big for git; they are kept on Snellius in `~/repro_archive/arbor_logs/<job>/`.

| run | job | score | paper | check |
|---|---|---|---|---|
| seed_0 HotpotQA MIPROv2 | 28994825 (A100) | 60.33 | 55.33 | OK, inside the seed-0 gate |
| seed_0 HoVer MIPROv2 | 28994830 (A100) | 47.00 | 47.33 | OK, inside |
| seed_0 IFBench MIPROv2 | 28994839 (H100) | 35.54 | 36.22 | OK, inside |
| seed_0 LiveBench-Math MIPROv2 | 28994837 (H100) | 48.48 | 46.60 | OK, inside |
| seed_0 AIME MIPROv2 | 28994833 (H100) | 44.67 | 20.00 | CHECK: the score gate paused the benchmark (d +24.67, z +5.33); technically clean (F97) |
| seed_1 LiveBench-Math MIPROv2 | 28995006 (H100) | 52.16 | 46.60 | OK (no gate for seeds 1-2) |
| seed_2 LiveBench-Math MIPROv2 | 28995016 (H100) | 46.05 | 46.60 | OK |
