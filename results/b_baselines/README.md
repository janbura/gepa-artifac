> **2026-11-09:** these Baselines ran on A100s. Table 1 now uses H100 runs only, so they are the hardware check, not table values (see `../table1_h100/README.md`). The exception is `h100_test/` (IFBench seed 2 on an H100), which is also copied to `../table1_h100/seed_2/` as the Table 1 value of that cell.

# B's Baseline runs (Qwen3-8B, one A100, vLLM 0.9.1)

Copied from `~/final_baselines` on Snellius (without the DSPy caches). See section 1, the run log and F67-F71 in REPRODUCTION_PLAN.md.

| Folder | What | Jobs |
|---|---|---|
| seed_0, seed_1, seed_2 | final AIME and LiveBench-Math Baselines, client/forwarder/worker timeouts 7200 s, no request dropped | 28957882, 28957884, 28941948 |
| truncated_600s | runs of 2026-11-05/06 that lost requests to Arbor's 600 s forwarder timeout (F67), kept as evidence, not results | 28929002, 28929003, 28941949 |
| vllm085 | IFBench seed 0 on vLLM 0.8.5 (not counted) | 28908434 |

IFBench seeds 0-2 are rerunning with a 4 h limit (F69), their folders will be added here.

Each run folder has config.json, evaluation_results/ (the score), metric_logs/test.jsonl (one line per scored example, failed examples are missing) and the run logs. run_log_stderr.txt.gz is the raw stderr, gzipped because of the progress bars.

## Arbor request logs and dataset revisions

`arbor_logs/<job id>/inference_{requests,responses}.jsonl.gz` are the Arbor server's logs of the three final math jobs (28957882 seed 0, 28957884 seed 1, 28941948 seed 2): one line per request and per answer, so "no request dropped" can be checked (199/199, 201/201, 201/201 lines) and the raw answers are available for the serving-behaviour analysis.

Datasets as cached on Snellius for these runs (Hugging Face revisions): `MathArena/aime_2025` @ c94da77eb22bbd6439e62a323bec18493a421302, `AI-MO/aimo-validation-aime` @ 13f9e12f613e720c2a2b2f345dd04b998a29494d, `livebench/math` @ bb66571c8ccf32d3df9e6f48b920d3770ff4aacb. The LiveBench-Math split is a seeded shuffle of the whole dataset, so a newer revision would change the test set.

Failed examples per run (test size minus the progress bar's denominator in run_log.txt): AIME 19 / 21 / 25 of 150, LiveBench-Math 21 / 12 / 20 of 126. All are answers that never reached the output fields (136 of the 140 failing completions still have an unclosed `<think>`, 96 are longer than 20,000 characters).

## Added 2026-11-08

- IFBench Baselines seeds 0, 1, 2 (jobs 28969642, 28969643, 28969644; A100): 34.69, 33.33, 36.22 (failed 23, 27, 21 of 294).
- HotpotQA and HoVer Baselines seed 0 (job 28978612; A100; the checkout and directory of all later optimizer runs): 39.33 and 38.00.
- `h100_test/`: IFBench seed 2 again on an H100 (job 28971148), 34.52; a hardware check, not part of the table.
- `arbor_logs/<job>/` has the server logs of these jobs, `../checks/<job>.md` the post-run check reports.
