# A100 runs kept as a hardware check (not Table 1 values)

These runs used one A100 (40 GB) per job. On 2026-11-09 the user decided that every Table 1 cell comes from an H100 run;
the A100 runs are kept and reported only as a hardware check (with F90 in REPRODUCTION_PLAN.md).

| run | job | score | paper |
|---|---|---|---|
| seed_0 HotpotQA MIPROv2 | 28994825 | 60.33 | 55.33 |
| seed_0 HoVer MIPROv2 | 28994830 | 47.00 | 47.33 |
| seed_1 HotpotQA MIPROv2 | 28998534 | 56.33 | 55.33 |

The A100 Baselines of all five benchmarks (seeds 0-2) are in `../b_baselines/seed_*`. HotpotQA MIPROv2 seed 1 on
A100 (job 28998534) finished on 2026-11-09 at 18:36 (check OK, 12 h 45, no server errors).

Comparison with the H100 MIPROv2 runs (descriptive, as planned): F103 in REPRODUCTION_PLAN.md. Every A100 value
lies inside the range of the three H100 seeds (HotpotQA 56.33-60.33, HoVer 42.33-48.67).
