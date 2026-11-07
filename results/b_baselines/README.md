# B's Baseline runs (Qwen3-8B, one A100, vLLM 0.9.1)

Copied from `~/final_baselines` on Snellius (without the DSPy caches). See section 1, the run log and F67-F71 in REPRODUCTION_PLAN.md.

| Folder | What | Jobs |
|---|---|---|
| seed_0, seed_1, seed_2 | final AIME and LiveBench-Math Baselines, client/forwarder/worker timeouts 7200 s, no request dropped | 28957882, 28957884, 28941948 |
| truncated_600s | runs of 2026-11-05/06 that lost requests to Arbor's 600 s forwarder timeout (F67), kept as evidence, not results | 28929002, 28929003, 28941949 |
| vllm085 | IFBench seed 0 on vLLM 0.8.5 (not counted) | 28908434 |

IFBench seeds 0-2 are rerunning with a 4 h limit (F69), their folders will be added here.

Each run folder has config.json, evaluation_results/ (the score), metric_logs/test.jsonl (one line per scored example, failed examples are missing) and the run logs. run_log_stderr.txt.gz is the raw stderr, gzipped because of the progress bars.
