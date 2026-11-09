# A100 runs kept as a hardware check (not Table 1 values)

These runs used one A100 (40 GB) per job. On 2026-11-09 the user decided that every Table 1 cell comes from an H100 run;
the A100 runs are kept and reported only as a hardware check (with F90 in REPRODUCTION_PLAN.md).

| run | job | score | paper |
|---|---|---|---|
| seed_0 HotpotQA MIPROv2 | 28994825 | 60.33 | 55.33 |
| seed_0 HoVer MIPROv2 | 28994830 | 47.00 | 47.33 |

The A100 Baselines of all five benchmarks (seeds 0-2) are in `../b_baselines/seed_*`. HotpotQA MIPROv2 seed 1 on
A100 (job 28998534) was still running on 2026-11-09 and belongs here when it is done.
