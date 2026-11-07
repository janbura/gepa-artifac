# GEPA reproduction plan

Reproduction of *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* ([arXiv 2507.19457](https://arxiv.org/abs/2507.19457)) with this artifact, on Snellius.

- **Objective:** Table 1 of the paper (Qwen3-8B) on the five benchmarks that need no OpenAI key: HotpotQA, HoVer, IFBench, AIME-2025 and LiveBench-Math, with Baseline, MIPROv2, GEPA and GEPA+Merge. Every cell gets **3 seeds (0, 1, 2), reported as mean ± SD**, served by the authors' vLLM 0.9.1 (F62).
- **Out of scope for now:** PUPA (its program and judge call `gpt-4.1-mini`, F14), the full GRPO row (about 13k SBU per benchmark; a log analysis and one limited run are in §2.7), Table 2 (GPT-4.1-mini), Table 3 and the figures.
- **Team:** A (fork owner) runs HotpotQA and HoVer; B runs IFBench, AIME-2025 and LiveBench-Math. A benchmark's whole chain stays in its owner's experiment directory (F20).
- **Budget:** the shared course account `gpuuva090` (about 980k SBU on 2026-11-04, shared with the rest of the course) expires **2026-12-18**. Our own share is still unconfirmed. Used so far: A about 440 SBU, B about 1,300 SBU before his final Baseline jobs. One A100-hour is 128 SBU.
- F-numbers point to the findings in §6. The phase plan used until 2026-11-05 is in git history (`60b8e91`).

_Last updated: 2026-11-07._

---

## 1. Results: Table 1, Qwen3-8B, no API key

Test-set score in %: mean ± SD over seeds, the number of seeds in parentheses, the paper's single run in brackets. One A100 per job, vLLM 0.9.1 unless noted.

| Variant | HotpotQA (A) | HoVer (A) | IFBench (B) | AIME-2025 (B) | LiveBench-Math (B) |
|---|---|---|---|---|---|
| Baseline | **41.22 ± 2.22** (3) [42.33] | **36.89 ± 1.17** (3) [35.33] | rerunning, F69 [36.90] | **30.67 ± 0.67** (3) [27.33] | **49.50 ± 0.64** (3) [48.70] |
| MIPROv2 | – [55.33] | – [47.33] | – [36.22] | – [20.00] | – [46.60] |
| GEPA | – [62.33] | – [52.33] | – [38.61] | – [32.00] | – [51.95] |
| GEPA+Merge | – [64.33] | – [51.67] | – [28.23] | – [32.00] | – [51.95] |
| *Rollout budget, MIPROv2 and GEPA* | *6,871* | *7,051* | *3,593* | *1,839* | *1,839* |

- **A's Baselines per seed** (failed examples in parentheses): HotpotQA 42.33 (6), 42.67 (0), 38.67 (5); HoVer 35.67 (0), 37.00 (0), 38.00 (0). Jobs 28923152, 28927513 and 28927510, all in `~/vllm091_runs`.
- **B's Baselines per seed** (failed examples in parentheses; all are format failures, no request was dropped): AIME 30.00 (19/150), 31.33 (21/150), 30.67 (25/150); LiveBench-Math 50.20 (21/126), 48.93 (12/126), 49.37 (20/126) (failed = test size minus the progress bar's denominator; the failures are unfinished thinking at the 16,384-token limit, F84). Jobs 28957882 (seed 0), 28957884 (seed 1) and 28941948 (seed 2), all with the timeouts of F67, `~/final_baselines`; result folders in `results/b_baselines/seed_*`. IFBench: all three runs were stopped by Slurm's 1 h 30 limit at 94–96% (F69) and are rerunning with 4 h (jobs 28969642–44). The truncated runs of 2026-11-05/06 are in `results/b_baselines/truncated_600s/`, the vLLM 0.8.5 IFBench run in `results/b_baselines/vllm085/`.
- **What a Baseline seed is:** the seed doesn't change the test set or the prompts, but each seed has its own run and cache directory, so the model samples every answer afresh. The paper's single run lies 0.5 SD (HotpotQA), 1.3 SD (HoVer) and 1.2 SD (LiveBench-Math) from our mean, and 5 SD for AIME, where one problem is 3.33 points (F84); any two HotpotQA/HoVer runs agree on 82–90% of examples (F65).
- **Not counted:** A's two runs on vLLM 0.8.5 (HotpotQA 39.67 and 40.67, HoVer 35.67 and 34.33; §5). B's runs of 2026-11-05/06 (jobs 28929002, 28929003 and 28941949) lost 7–37% of their requests to a third timeout (F67) and were redone. B's earlier seed-1 and vLLM 0.9.1 runs were fresh runs, not cache replays as first thought (F55, corrected), but they lost requests to the same timeouts.
- **The paper's other Table 1 cells:** PUPA 80.82 / 81.55 / 91.85 / 86.26 (Baseline, MIPROv2, GEPA, GEPA+Merge), and a GRPO row of 43.33, 38.67, 35.88, 38.00 and 51.26 in the column order above (PUPA 86.66). The paper prints the same GEPA and GEPA+Merge score for both math benchmarks (F61).

---

## 2. Plan for the optimizer runs (B, 2026-11-07; replaces the section of 2026-11-06)

Written after checking our Baselines against the authors' runs and the code (findings F67–F83). Numbers marked "expected" come from the cost model in §2.6; they are re-estimated after the first MIPROv2 and GEPA runs and after the H100 pilot.

### 2.1 Objective and scope

- **Table 1 of the paper, Qwen3-8B, without an API key:** HotpotQA, HoVer (A), IFBench, AIME-2025, LiveBench-Math (B), with Baseline, MIPROv2-Heavy, GEPA and GEPA+Merge. **Three seeds (0, 1, 2) per cell, reported as mean ± SD with the per-seed values and the number of failed examples**, plus per-example paired significance tests (`analysis/significance.py`). The paper has one run per cell, so our spread is itself a result.
- **GRPO** is not reproduced in full (24,000 rollouts, about 13k SBU per benchmark). We do (a) an analysis of the authors' four shipped GRPO runs, which already explains the small gains (§2.7), (b) a measurement of the learning signal on our setup without training, and (c) one limited run of about 10,000 rollouts on HoVer, if the budget allows, as a decision for the supervisor.
- **Out of scope:** PUPA (needs gpt-4.1-mini, F14), Table 2 (GPT-4.1-mini, retired 2027-01-14, F16), Table 3 except the SelectBestCandidate ablation if time allows, the figures that need runs we do not have.
- **Everything is run with the artifact as the authors ran it**: their DSPy and Arbor forks, their vLLM 0.9.1 (F62), their sampling settings, their budgets. Our changes are infrastructure only and listed in §2.3: four timeouts (F67, F70, F72), one GPU per server, the per-job compile cache (F64), and no W&B (F66). Three behaviours of the artifact that change answers are kept and reported (§2.3).
- **Extensions come after Table 1** (§2.8), but the runs are set up so that they need no rerun: every run keeps its per-example logs, Arbor's request/response logs, timings and SBU.

### 2.2 What we have, and what we verified

| | HotpotQA (A) | HoVer (A) | IFBench (B) | AIME-2025 (B) | LiveBench-Math (B) |
|---|---|---|---|---|---|
| Baseline, 3 seeds | 41.22 ± 2.22 (paper 42.33) | 36.89 ± 1.17 (35.33) | rerunning with 4 h jobs (F69) | 30.67 ± 0.67 (27.33) | 49.50 ± 0.64 (48.70) |
| Failed examples per seed | 6, 0, 5 | 0, 0, 0 | – | 19, 21, 25 of 150 | 21, 12, 20 of 126 |
| Worker timeouts (void a run, F72) | up to 5 per run: A recounts | 0 | – | 0 | 0 |
| Authors' Qwen run to compare with | yes (all optimizers) | yes | yes | none (only GPT runs) | none |

- B's six final math folders were checked file by file: every score recomputes from `metric_logs/test.jsonl`; program, instructions and optimizer config are byte-identical to the authors' configs; the test examples are the authors' at the same indices; the only errors in the logs are DSPy format failures; every request got an answer (Arbor's request and response logs are in `results/b_baselines/arbor_logs/`). The failures are answers that never left the thinking phase (136 of 140 have an unclosed `<think>`, 96 are longer than 20,000 characters), i.e. unfinished thinking at the 16,384-token limit; the authors' Qwen math failure rate is unknown because no run is shipped. AIME is effectively about 50 independent answers over 30 problems (the five copies of a problem are mostly cache hits, F53), so one problem moves the score by 3.33 points; the paper's 27.33 is 5 SD below our mean, which is one problem.
- The dataset revisions are pinned in `results/b_baselines/README.md` (LiveBench-Math's split is a seeded shuffle of the whole dataset, so a newer revision would change the test set).
- A's HotpotQA Baselines had up to 5 worker timeouts per run; a worker timeout can put another example's text into an answer (F72). A counts them (`grep -c 'Timeout for chat request' logs/arbor-<job>.log` for 28923152, 28927513, 28927510); runs with any are redone with the timeout patch (about 40 SBU each). Because the client timeout is now in `run_experiments.py` and is part of the DSPy cache key (F73), A's Baselines are rerun anyway before his MIPROv2 runs, so that each chain shares one cache.

### 2.3 What went wrong so far, and the rules it gave us

| Mistake | Who | Cost | Rule now | Finding |
|---|---|---|---|---|
| First Qwen job on an H100 without Arbor installed in its venv | B | 29 SBU | the job script checks the Arbor environment and logs its vLLM version | F41 |
| Jobs on `gpu_mig` (A100 slices): vLLM cannot fit the 32,768-token context | B | 34 SBU | full GPUs only | F51 |
| An H100 job estimated to start 4 days later; cancelled | B | a day of waiting | 1-GPU A100 jobs; H100 only where the KV cache matters | F47 |
| `--dry_run` crashed on IFBench, and a dry run in the real directory would mark the real run as done | A, B | hours | dry runs and pilots in their own experiment directory; fix on `feat/pilot-caps` | F18, F50 |
| Set environment variables in front of `sbatch`; two "vLLM 0.9.1" jobs served 0.8.5 | A | 2 runs, ~70 SBU | per-run variables at the start of the command line, job-wide ones with `--export`; the job log prints the vLLM version | F63 |
| Two vLLM 0.9 jobs started in the same second and broke each other's compile cache | A | 75 SBU, no result | `VLLM_CACHE_ROOT` per job | F64 |
| A GEPA test without a server looped 2,083 times | A | login-node time | watchdog in the job script; `timeout` for pilots | F57 |
| Package pins left 58 packages at the wrong version; a reinstall removed the spaCy model | A | a day | full pin list; NLTK/spaCy data fetched once | F49, F48 |
| Called the first math runs clean although a 600 s forwarder limit was dropping answers (HTTP 500 that our checks did not count); seeds 0 and 1 had to be redone | B | ~900 SBU, 2 days | four timeouts raised (client 7200, forwarder 7200, worker 21600, chunk call 600); after every run: requests = responses and 0 timeouts/5xx/chunk errors in the Arbor log | F54, F56, F67, F70 |
| Blamed the math gap on format randomness; it was the truncated runs | B | wrong F52 | complete runs only; the three seeds agree within 1.3 points | F52 (updated) |
| Claimed a shared DSPy cache (wrong); one rerun replayed its own folder's cache | B | 1 void job | before a rerun move both `experiment_runs/seed_s/<run>` and `experiment_cache_dirs/seed_s/<run>` | F55 (corrected) |
| Did not resize the Slurm limits after raising the timeouts; three IFBench runs killed at 94–96 % | B | ~580 SBU | IFBench 4 h; optimizer jobs sized in §2.6 with margin | F69 |
| Job script killed every `vllm_serve` of the user on the node | B | none (never co-located) | kill only the job's own worker | F70 |
| Counted failed examples from error lines / raw log lines (23/13 instead of 21/12) | B | wrong table cells | failed = test size − the progress bar's denominator in `run_log.txt` (one rule for both) | F84 |
| Worker timeouts fired inside chunk calls and mixed answers between requests in the truncated runs | B (and the authors' optimizer runs) | invalid runs | worker timeout longer than the job; a run with any `Timeout for chat request` is void | F72 |
| Added the client timeout, which broke the cache replays | B | 1 h | `GEPA_REPLAY_CACHE=1` in the replay scripts; one `create_lm` per chain | F73 |

**Kept on purpose, because the authors ran the same code (report, do not fix):** `top_k=20` never reaches vLLM (F71); each 64-token continuation goes through Qwen3's chat template and puts an empty think block before the model's partial reasoning (F58, F71; all 24 of the authors' Qwen logs show the chunked pooling); DSPy resubmits stragglers and scores the first copy to finish (F71); one bad continuation fails a whole batch of up to 32 requests (F68; in 11 of the authors' 24 runs); MIPROv2 evaluates with 8 threads and GEPA with the node's CPU count although the command line says 32 (F76, F76, §2.5); GRPO trains on reconstructed completions behind an empty think block (§2.7). These are the subject of an extension (§2.8), not of the reproduction.

### 2.4 Pre-flight and post-run checks (both of us, every job)

Before `sbatch`:
1. `git pull --rebase`, commit, note the commit (the job log prints it, with "+ uncommitted changes" if dirty).
2. `bash scripts/snellius/apply_arbor_timeouts.sh` once per checkout; the job refuses to start without the four edits and the client timeout (a688f06).
3. The command line comes from `scripts/snellius/make_command.sh <bench> <prog> <opt> --use_cache_from_opt <source> --seed s`, prefixed with `GEPA_EXPERIMENT_DIR=<the chain's directory>`; `--use_cache_from_opt` is MIPROv2-Heavy ← Baseline, GEPA/GEPA-MERGE ← MIPROv2-Heavy (the run asserts it, F82).
4. The source run of the chain has `evaluation_results/` in that directory and seed; the target run folder and its cache folder do not exist (F18, F55).
5. One optimizer run per job with its own time limit (§2.6); `--dependency=afterok:<source job>`; `bash scripts/snellius/sbu_guard.sh <hours × rate>`.

After the job:
1. Job log: `arbor venv: … (vllm 0.9.1)`, `timeouts: client 7200 s …`, `server ready`, exit code 0 for each command, no `=== watchdog`, no `already exists. Skipping`.
2. Arbor log: `grep -cE 'Timeout for chat request|httpx.ReadTimeout|500 Internal Server Error|504 Gateway|LLM result|chunk chat after|Timeout waiting for worker' logs/arbor-<job>.log` is 0; `inference_requests.jsonl` and `inference_responses.jsonl` have the same number of lines.
3. Run folder: `evaluation_results/evaluation_result.txt` is new (mtime), `optimized_program/` exists; failed examples = test size − the last `Average Metric: x / N` denominator in `run_log.txt`; `run_log_stderr.txt` has only `Failed to parse` errors.
4. MIPROv2: `Default program score` and `Returning best identified program with score` in `run_log_stderr.txt` (if equal, the cell is the Baseline, F77); `optimizer_logs/evaluated_programs` has 81 / 41 / 34 (AIME 28) entries. GEPA: `total_num_evals` in `gepa_state.bin` (the counted budget), lines of `instruction_proposer_inpouts.jsonl` (reflections), `prog_candidates/`.
5. A row in §5 with commit, vLLM version, GPU type, wall-clock, SBU, score, failed examples, requests, counted budget; the result folder (without caches) into `results/`.

### 2.5 The run plan

**Who runs what (decided 2026-11-07).** Split by optimizer, not by benchmark: A runs MIPROv2-Heavy on all five benchmarks, B runs GEPA and GEPA+Merge on all five. Each person keeps the Baselines they already have. Hand-over, because a chain only works inside one experiment directory (F20, F82): (1) B copies his Baseline run folders and their cache folders (`experiment_runs/seed_s/<run>` and `experiment_cache_dirs/seed_s/<run>`) for IFBench, AIME and LiveBench-Math into A's directory before A's MIPROv2 runs on them; (2) after each MIPROv2 run, A copies its run folder and cache folder into a group-readable place (`/scratch-shared/<A>/handover/`, `chmod -R g+rX`, group `gpuuva090_access`), and B copies them into his directory before the GEPA job of that benchmark and seed. Both copies keep the same seed and the same `create_lm` (F73). Scratch has no confirmed retention policy (F8), so copy into home right away.

**Chains and directories.** Each benchmark and seed is one chain in one experiment directory: A's `~/vllm091_runs` (after the Baseline reruns of §2.2), B's `~/final_baselines`. The chain exists only to copy DSPy caches (the GEPA budget is hard-coded, F74), but it makes MIPROv2's base evaluation and GEPA's base val pass identical to the Baseline run and it is what the authors did.

**Jobs.** One optimizer run per job, chained with `--dependency=afterok`, except the GEPA-family runs of one benchmark, which share one server:
1. MIPROv2-Heavy, seeds 0, 1, 2: three separate jobs per benchmark, started as soon as the Baseline of that seed is final. MIPROv2 must run alone on its server (its proposer uses other temperatures, which Arbor serves only when the main pool is empty, F82).
2. GEPA and GEPA+Merge, seeds 0, 1, 2: one job per benchmark with six background runs (`&` lines and `wait` in the commands file) on one server, after the three MIPROv2 jobs. GEPA is latency-bound (3-example minibatches, one reflection at a time), so six runs cost little more than one (F3); they all use the same sampling signature.
3. The IFBench chain starts when the three IFBench Baselines are final (jobs 28969642–44).

**Threads.** The command line says 32, but the artifact gives MIPROv2's evaluations 8 threads and GEPA's val passes the node's CPU count (F76, F76), and the authors ran with these effective values. We keep them: MIPROv2 at 8 threads stays under the A100's KV-cache limit even on math, and GEPA's val passes are what the server sharing is for. This is noted in the write-up as an artifact behaviour.

**GPU type.** A100 for everything by default. B's math GEPA-family jobs are the exception: 1,700–1,900 answers of up to 16k tokens per run at high concurrency hit the A100's 137k-token KV cache (130–175 tok/s instead of about 1,000). The H100 pilot (job 28971148, IFBench Baseline seed 2 in `~/h100_pilot`) measures the H100's speed; if it is at least 500 tok/s, B's GEPA-family jobs go to `gpu_h100`, otherwise they run on A100 with longer limits (§2.6). GPU type is recorded per run; it is infrastructure, not method.

**AIME budget note.** Our AIME val set has 45 examples, below MIPROv2's 50-example minibatch threshold, so MIPROv2 runs 27 full-val trials and about 1,300 metric calls, while GEPA keeps the artifact's 1,839 (the paper's budget row). We keep the artifact as it is and report the mismatch (F75).

**What A does first.** Pull `feat/pilot-caps` (timeouts in the code, preflight in the job script, the Arbor patch); apply the patch; count worker timeouts in his Baseline jobs; rerun the six Baselines with the timeouts (about 240 SBU, 6 jobs of 20 min); confirm the F60 reading (no effect under `track_scores_on='val'`, F81) so the decision is closed; then start MIPROv2 on HotpotQA and HoVer, seed 0 first.

**What B does first.** IFBench Baselines (queued), H100 pilot (queued), then MIPROv2 for AIME and LiveBench-Math seed 0 on A100 as soon as a GPU is free (their Baselines are final), IFBench after its Baselines; GEPA-family jobs after the pilot's verdict.

**Order and why.** Seed 0 of every chain starts first, so that the first MIPROv2 and GEPA runs calibrate the cost model before seeds 1 and 2 are submitted; a wrong assumption then costs one run, not three.

### 2.6 Cost and time (expected; pessimistic ×1.3)

Basis (recomputed by `analysis/cost_model.py`, which prints this table from the inputs below): the authors' request and token counts per run (notes §3), 80–95 % of predictor calls fresh for MIPROv2 (the authors' HotpotQA MIPROv2 log is a restarted run and is not used, F79), our measured speeds (585 tok/s short answers, 130–175 tok/s long answers on one A100 with 32 threads; about 50 tok/s per stream), B's partial IFBench MIPROv2 run (11 of 41 trials in 2 h 43 min at 8 threads).

| Run type | Hours per run (A100) | SBU per run | Runs | SBU |
|---|---|---|---|---|
| MIPROv2-Heavy, any benchmark (8 threads, latency-bound) | 9–11 | 1,150–1,400 | 15 | 17,000–21,000 |
| GEPA family, HotpotQA, 6 runs packed | 23–26 for the six | 3,000–3,300 | 1 job | 3,000–3,300 |
| GEPA family, HoVer, 6 runs packed | 34–37 for the six | 4,300–4,700 | 1 job | 4,300–4,700 |
| GEPA family, IFBench, 6 runs packed, H100 | 15–20 for the six | 2,900–3,800 | 1 job | 2,900–3,800 (A100: 82 h, 10,500) |
| GEPA family, AIME, 6 runs packed, H100 | 20–25 for the six | 3,800–4,800 | 1 job | 3,800–4,800 (A100: 98 h, 12,500) |
| GEPA family, LiveBench-Math, 6 runs packed, H100 | 20–25 for the six | 3,800–4,800 | 1 job | 3,800–4,800 (A100: 98 h, 12,500) |
| **Table 1 total** | | | 45 runs | **35,000–42,000** expected; 55,000 pessimistic; 60,000–70,000 if B's GEPA jobs stay on A100 |
| A's Baseline reruns | 6 × 20 min | 40 | 6 | 240 |
| GRPO: signal measurement (HoVer, HotpotQA), 10-step smoke test | | | | 600 |
| GRPO: HoVer, 208 steps (10,000 rollouts), 2 × H100 | 5–7 | | 1 | 1,800–2,700 |
| GRPO: HotpotQA, 208 steps, 2 × H100 (optional) | 10–15 | | 1 | 3,700–5,600 |
| gpt-oss-120b, 1 seed, 5 benchmarks × 4 variants, 1 × H100 (optional) | ~25 + setup | | 20 | ~7,000 |
| Reserve for reruns (25 % of Table 1) | | | | 9,000–11,000 |

Budget: Table 1 with reserve 44,000–53,000 SBU expected; with GRPO (HoVer) and gpt-oss about 54,000–63,000; the 100k holds even in the pessimistic case for Table 1 (about 70,000) if gpt-oss is dropped. Used so far: about 4,700 including the queued IFBench jobs and the pilot.

Job time limits: MIPROv2 16 h; packed GEPA-family jobs 36 h (HotpotQA), 48 h (HoVer), 30 h on H100 or 5 days on A100 (B's benchmarks). A run writes its result only at the end, so a job that hits its limit loses the running step (a rerun replays its cache, F80).

Calendar: queue waits are 1–1.5 days on `gpu_a100` and about 3 days on `gpu_h100` today (rolling reboots). Per benchmark: MIPROv2 jobs (parallel) → packed GEPA job, about 5–7 days; all five benchmarks in parallel. Expected: A's chains from 2026-11-09 (after the Baseline reruns), B's math MIPROv2 from 2026-11-08, B's GEPA-family jobs after the pilot (2026-11-11), Table 1 complete **2026-11-21 to 2026-11-28**; a week later in the pessimistic case. GRPO and extensions follow until the budget expiry 2026-12-18. The course deadline is to be confirmed.

### 2.7 GRPO: what the authors' logs show, and what we do

From the four shipped Qwen GRPO runs (HotpotQA, HoVer, IFBench, PUPA; scripts in `analysis/` after clean-up):
- The validation curves are flat within noise. HotpotQA: 46.67 before training, 25 later validations with mean 48.36 and SD 1.84 (the binomial SE of one 300-example validation is 2.88), best 52.33 at step 220, which is the selected checkpoint; test 43.33 vs 42.33. HoVer: 35.33 → mean 36.16, best 39.0 at step 500; test 38.67 vs 35.33. IFBench: 57.0 → mean 59.4, best 63.47 at step 460; test 35.88 vs 36.90 (val and test come from different files). The best-validation gains (+5.7, +3.7, +6.5) are what the maximum of 25 noisy validations of an unchanged model gives; the per-example validation means over the first vs last five validations moved by +0.1, +0.8 and +3.5. There is no early stopping in the code: it keeps the best of 26 checkpoints.
- Most groups carry no signal: 65.7 % (HotpotQA), 56.4 % (HoVer) and 76.0 % (IFBench) of the 12-rollout groups have identical rewards, so their advantages are exactly zero; only 35 / 44 / 24 % of the 24,000 rollouts are in a group with a signal. PUPA, the only continuous reward, has 38 % zero groups and the only clear gain. Training reward rises on the 150 training examples (HotpotQA 45 → 54 % over 13 epochs) while validation stays flat.
- The trainer learns on a reconstruction of the parsed output fields behind an empty `<think></think>` block, not on the sampled text with its reasoning (mean trained completion 129–344 tokens); all four runs show KL spikes of 1e3–5e6; every module of a 4-module program gets the program-level reward. The in-run test evaluations of the HotpotQA and IFBench GRPO runs failed (checkpoint server did not start, score 0/300 and 0/294 in the logs); the paper's 43.33 and 35.88 come from an evaluation that is not shipped.
- Hardware per the logs: vLLM with tensor parallel 2 on two 80 GB GPUs plus one training GPU; 12–24 h; per step 61–129 s of rollouts plus a validation every 20 steps.

Plan: (1) report the analysis above as a finding of the reproduction (no SBU); (2) measure the zero-advantage fraction on our server for HoVer and HotpotQA (12 samples per train example, about 200 SBU) to show it is a property of the task and the model, not of the authors' run; (3) a 10-step smoke test of the GRPO path (GRPO environment from `pyproject_grpo.toml` with the `datasets` pin F40, flash-attn, the training YAML, W&B key; about 400 SBU); (4) HoVer, 208 steps = 10,000 rollouts, 11 validations, on two H100s (about 2,000–2,700 SBU), reporting the test score of the selected, the base and the last checkpoint; (5) HotpotQA the same if the budget allows (3,700–5,600). IFBench is skipped (its val/test come from different files and 76 % of its groups have no signal). Decision for the supervisor: (4) and (5) are the parts that cost.

### 2.8 Extensions prepared by the way we run

Each run keeps `metric_logs/test.jsonl`, Arbor's request and response logs, timings and SBU, so these need no rerun:
1. **Variance and significance of Table 1** (3 seeds, paired tests): already part of the objective; the paper has one run per cell.
2. **Serving effects on the scores** (the kept artifact behaviours, §2.3): evaluate the final Qwen programs once with vLLM's own OpenAI server (no chunking, `top_k` applied, no straggler resubmission) and compare per example; one test pass per cell, about 1 h each, 3,000–4,000 SBU for all cells on A100.
3. **The GRPO analysis** of §2.7 and the limited run.
4. **gpt-oss-120b as a second open model** (117B parameters, 5.1B active, MXFP4; fits one H100; vLLM ≥ 0.10.1 in its own environment; served with vLLM's server, not Arbor, because Arbor's chunked continuation will not work with the harmony format): one seed of Baseline + MIPROv2 + GEPA + GEPA+Merge on the five benchmarks, about 7,000 SBU if its answers are as long as Qwen's; a feasibility test first (serve it, run one Baseline, about 400 SBU). Medium engineering risk; only after Table 1.
5. **The released GEPA library vs the artifact** (`gepa-ai/gepa` and `dspy.GEPA` re-implement the method): one run per benchmark with the library on the same budget, if time allows.
6. **The reproducibility findings** (timeouts, cache keys, chunking, the missing GRPO test evaluations) as a section of the write-up.

### 2.9 Open decisions

- **A:** Baseline reruns with the timeouts (recommended, 240 SBU) or keep the current Baselines and accept that MIPROv2 regenerates the base pass; the F60 reading.
- **Both:** keep the artifact's effective thread counts (8 for MIPROv2, CPU count for GEPA) as planned, or pass 32 everywhere (speed only, one config line).
- **B:** H100 for the math GEPA-family jobs after the pilot.
- **Supervisor:** GRPO steps (4) and (5) of §2.7; gpt-oss-120b; the course deadline.

---

## 3. Running a benchmark

Every Qwen3-8B job goes through `scripts/snellius/qwen_job.sh` on one A100. This checklist is built from the problems in F50–F73.

**Environments** (once per person)

- uv 0.13.7 (`curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh`), with `UV_CACHE_DIR=/scratch-shared/$USER/uv-cache UV_LINK_MODE=copy` in `~/.bashrc`.
- `bash setup_gepa_repo.sh` (DSPy fork `62dc3b6`, Arbor fork `113fc35`, F46), then `uv sync --locked`: all 285 package versions of an original run (F49).
- Arbor with the authors' vLLM 0.9.1: `uv venv --no-config --python 3.12.9 ~/arbor-envs/vllm091`, then `uv pip sync --no-config --python ~/arbor-envs/vllm091/bin/python envs/arbor-vllm091-requirements.txt` (F62). The older 0.8.5.post1 environment (`gepa_artifact/utils/arbor/.venv`, from `envs/arbor-uv.lock`) is what a job uses without its 2nd argument.
- Weights: `huggingface-cli download qwen/qwen3-8b` from an Arbor environment (16 GB, revision `b968826`, F11). Datasets: `uv run python -m scripts.generate_launch_commands > launch_commands` once on the login node (F48). HotpotQA and HoVer need the BM25 index, built once (F23).
- The authors' runs are in `reference_runs_data/` (A's checkout), never in `experiment_runs_data/` (F19).

**Before submitting**

1. Commit first: the job log records the commit, or "+ uncommitted changes". Don't `git switch`, `pull` or `merge` while a job runs; it reads the code and the commands file from the checkout.
2. Serve with the authors' vLLM 0.9.1: `qwen_job.sh <commands> $HOME/arbor-envs/vllm091` (F62). Without the 2nd argument the job serves 0.8.5.post1.
3. Build each command with `make_command.sh <benchmark> <program> <optimizer>`, which checks the names. Put per-run variables at the start of the line (`GEPA_EXPERIMENT_DIR=… GEPA_MAX_METRIC_CALLS=…`), never in front of `sbatch` (F63). Real chain runs add `--use_cache_from_opt` (MIPROv2 ← Baseline, GEPA family ← MIPROv2-Heavy) and `--seed s`; pilots leave out `--use_cache_from_opt` (F44).
4. One experiment directory per serving setup and purpose. A chain runs in the directory whose Baseline was served by the same vLLM, because MIPROv2 copies the Baseline's cache and GEPA copies MIPROv2's (F20). A run that already finished in a directory is skipped.
5. `mkdir -p logs`, then `bash scripts/snellius/sbu_guard.sh <hours × 128>` and `sbatch -p gpu_a100 -t <limit> -J <name> scripts/snellius/qwen_job.sh …`. Chain the steps with `--dependency=afterok:<job id>`. Lines ending in `&` share one server, and `wait` waits for them.

**After every job**

1. In `logs/<name>-<id>.out`: `arbor venv: … (vllm 0.9.1)`, `job settings:`, `server ready`, each command's exit code, and no watchdog line (F57).
2. `SEED=<s> GEPA_EXPERIMENT_DIR=<dir> bash scripts/snellius/check_run.sh <run name>`: score, failed examples (test size − unique examples logged), parse failures.
3. Every request answered: `inference_requests.jsonl` and `inference_responses.jsonl` in `logs/arbor_requests_<id>/*/` have the same number of lines (F55, F67). And `grep -cE 'Timeout for chat request|httpx.ReadTimeout|500 Internal Server Error|504 Gateway|LLM result|chunk chat after|Timeout waiting for worker' logs/arbor-<id>.log` is 0 (F54, F67, F68, F70). A worker timeout also mixes answers between requests, so a run with one is void (F72).
4. A row in §5 with commit, vLLM version, SBU, score, failed examples and request count.

**Problems we hit**

| Problem | Symptom | Handled by | Finding |
| --- | --- | --- | --- |
| Environment set in front of `sbatch` dropped | The log shows the default `arbor venv` or `<unset>` settings | 2nd argument; variables at the start of command lines | F63 |
| Two vLLM 0.9 jobs start together | "vLLM failed to compile the model" | Per-job `VLLM_CACHE_ROOT`; the warm-up stops at the first fatal message | F64 |
| GEPA loops when every call fails | Thousands of "No feedback samples, skipping" iterations, budget unused | Watchdog in `qwen_job.sh`; `timeout` for pilots | F57 |
| Arbor's 300 s worker, 600 s forwarder and 600 s client limits | HTTP 504, HTTP 500 with `httpx.ReadTimeout` in the Arbor log, or `APITimeoutError`; the example scores 0 | All three raised: client and forwarder 7,200 s, worker 21,600 s (B since 2026-11-06) | F54, F56, F67 |
| Worker timeout mixes answers between requests | `Timeout for chat request` in the Arbor log; answers that jump mid-sentence to another example | Worker timeout longer than the job; a run with one is void | F72 |
| 60 s limit per chunk call | `timeout for continue-chunk chat after 60s`; later answers swapped | Raised to 600 s (B) | F70 |
| IFBench longer than the job limit | `TIMEOUT` in sacct, no `evaluation_results` | `-t 04:00:00` for IFBench | F69 |
| Timeout in the DSPy cache key | Replays or chain steps miss the cache | `GEPA_REPLAY_CACHE=1` in the replay scripts; one `create_lm` per chain | F73 |
| A rerun replays its own run folder's DSPy cache | Far fewer requests reach the server than the run needs | Move `experiment_runs/seed_<s>/<run>` and `experiment_cache_dirs/seed_<s>/<run>` before a rerun | F55 (corrected) |
| Format failures | Fewer unique examples logged than the test set has | Keep the parser; report the count | F52, F65 |
| An example logged twice in `test.jsonl` | Line count above the unique count | `check_run.sh` counts unique examples | F65 |
| `--dry_run` crash | `AttributeError: … dev_set` | Fixed on `feat/pilot-caps` | F50 |
| `gpu_mig` too small | vLLM can't fit the 32,768-token context | Full GPUs only | F51 |
| Queue | `PENDING (Priority)` for hours; H100 for days | 1-GPU jobs on `gpu_a100` | F47 |

**A's benchmarks** (B adds IFBench, AIME-2025 and LiveBench-Math the same way)

| | HotpotQA | HoVer |
| --- | --- | --- |
| Program (`--prog_name`) and calls per example | `HotpotMultiHop`, 4 | `HoverMultiHop`, 4 |
| Test examples | 300 | 300 |
| Output tokens per call, mean / p90 | 422 / 611 | 473 / 651 |
| Baseline test pass on one A100, 32 threads | 14–19 min; a 1-run job 34–41 SBU | 13–14 min; a 3-run job 94 SBU |
| Our Baselines (failed examples) | vLLM 0.9.1: 42.33 (6), 42.67 (0), 38.67 (5); 0.8.5: 39.67 (7), 40.67 (6) | vLLM 0.9.1: 35.67 (0), 37.00 (0), 38.00 (0); 0.8.5: 35.67 (0), 34.33 (2) |
| Authors' scores: Baseline / MIPROv2 / GEPA / GEPA+Merge / SelectBest | 42.33 / 55.33 / 62.33 / 64.33 / 58.33 | 35.33 / 47.33 / 52.33 / 51.67 / 45.33 |
| Authors' vLLM | 0.9.1; GEPA family 0.9.2 | 0.9.1 |
| Authors' reflections: GEPA / GEPA+Merge / SelectBest | 132 / 64 / 74 | 93 / 50 / 126 |
| Timeouts per Baseline run | 0–5, all retried; each one can mix answers (F72) | 0–1 (F72) |
| Retrieval | Local BM25 over 2017 Wikipedia abstracts; one index for both (F23) | Same |
| Our run directories | vLLM 0.9.1: `~/vllm091_runs` (seeds 0–2); 0.8.5: `experiment_runs_data/`, `~/vllm085_rep2_runs` | The same |

---

## 4. Measurements

| What | Who | Measured |
|---|---|---|
| Thinking mode in the authors' runs | B | On (cached `reasoning_content`, F25) |
| Authors' run times on 4-GPU servers | B | GEPA 1.2–6.4 h, MIPROv2 1.1–4.7 h, GRPO 12–24 h on 3 GPUs (F37); per run for HotpotQA and HoVer in §2.2 |
| Server start-up (job start until Arbor answers) | A, B | 114–238 s on A100 |
| Output tokens per call, mean (p90) | A, B | HotpotQA 422 (611), HoVer 473 (651), IFBench 742 (1,145), AIME 2,731 (5,572), LiveBench-Math 1,709 (3,609); Qwen tokenizer on Arbor's response log |
| Test pass per example, 32 threads, one A100 | A, B | HotpotQA 2.9–3.7 s, HoVer 2.7–2.8 s, IFBench 6.3 s, AIME 6.5 s, LiveBench-Math 15.8 s (B's first job; AIME needed only 63 calls for 150 examples, F53). A complete math run with raised timeouts takes about 2 h (F56) |
| MIPROv2 wall-clock, IFBench | B | Not finished in 3.6 h: bootstrapping 23 min, then 11 of 41 trials in 2 h 43 min; ≈ 10 h for the full run |
| GEPA iteration time with 1, 2 and 6 runs per server | A | Not measured yet (§2.3) |
| Mean GPU utilization | A, B | A's Baseline jobs 65–67% (start-up included); B's first job 86% |
| Run-to-run spread, Baseline | A | SD over 3 seeds: HotpotQA 2.22, HoVer 1.17 points; any two runs agree on 82–90% of examples |

---

## 5. Run log

One row per job; add new rows at the bottom.

| Date | Who | Job ID | Benchmark | Variant | Commit | Wall-clock | SBU | Score | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | A | 28902285 | HotpotQA, HoVer | BM25 index (1.7) + network check (4.1) | `2378040` | 6 min 18 s | ~2.5 | – | `genoa`, 24 cores, peak 7.9 GB RAM, one core in use |
| 2026-11-04 | B | 28901861 | IFBench, AIME, LiveBench-Math | Baselines (first try) | `b-tonight` | ~14 min | ~29 | – | H100; Arbor didn't start (not installed in its venv, F41); cancelled |
| 2026-11-04 | B | 28903967 | IFBench, AIME, LiveBench-Math | dry run + Baselines + MIPROv2 | `7adf8ca` | – | 0 | – | H100; cancelled while pending, estimated start was 2026-11-08 (F47) |
| 2026-11-04 | B | 28908296–98 | IFBench, AIME, LiveBench-Math | Baselines, one job each | `7adf8ca` | ~10 min each | ~34 | – | `gpu_mig` (A100 3g.20gb); vLLM doesn't fit with the authors' 32,768 context (F51); cancelled |
| 2026-11-05 | B | 28908434 | IFBench, AIME, LiveBench-Math | dry run (4.2), Baselines (4.3), MIPROv2-Heavy IFBench (4.5) | `7adf8ca` | 5 h 00 min (time limit) | 640 | 36.56 / 19.33 / 33.58 | A100 (`gcn22`), 18 cores; `7adf8ca` = `build/pin-env` before `0ec5585` + `feat/replay-and-analysis`; MIPROv2 cut at trial 12/41; B total so far 703 SBU |
| 2026-11-05 | B | 28916886 | AIME, LiveBench-Math | Baselines, seed 1 | `7adf8ca` | 43 min | 91 | 26.00 / 53.02 | A100 (`gcn40`); fresh server; parse failures far fewer than seed 0 (F52) |
| 2026-11-05 | B | 28920832 | IFBench, AIME, LiveBench-Math | Baselines, seed 1, vLLM 0.9.1 | `7adf8ca` | 83 min | 176 | 36.73 / 14.00* / 35.96* | A100; `~/arbor091`; output in `~/vllm091_runs/experiment_runs/seed_1`; 642 requests hit Arbor's 300 s timeout (504), 144 client timeouts; logged examples 278/294, 97/150, 87/126; the two math scores are invalid (F54); B total 1066 SBU |
| 2026-11-05 | B | 28925784 | AIME, LiveBench-Math | Baselines, seed 1, vLLM 0.9.1, Arbor timeout 1800 s | `7adf8ca` | 8 min | 16 | 13.33* / 37.54* | only 13 requests reached the server, the rest came from `~/.dspy_cache` including the timed-out answers of 28920832 (F55); void |
| 2026-11-05 | B | 28926616 | AIME, LiveBench-Math | Baselines, seed 1, vLLM 0.9.1, fresh DSPy cache, Arbor timeout 1800 s | `7adf8ca` | 87 min | 186 | 27.33 / 41.46 | A100; clean run (no cache hits); 0 HTTP 504 but 45 examples lost to the OpenAI client's default 600 s timeout (AIME 82/150 logged, LiveBench 104/126); of 144 completed responses 126 parsed (88 with unclosed `<think>`); B total 1268 SBU |
| 2026-11-05 | B | 28929002 | AIME, LiveBench-Math | final Baselines, seed 0: vLLM 0.9.1, fresh cache, both timeouts 1800 s | `c3ca457` | 1 h 20 min | 171 | 25.33 / 52.97 | A100; 239 requests, 152 answered; **88 dropped by Arbor's 600 s forwarding timeout (HTTP 500, F67)**: AIME 76, LiveBench-Math 12; 0 HTTP 504, 0 client timeouts; `~/final_baselines` |
| 2026-11-05 | B | 28929003 | IFBench, AIME, LiveBench-Math | final Baselines, seed 1: same settings | `c3ca457` | 2 h 12 min | 282 | 36.05 / 26.00 / 47.16 | A100; 811 requests, 692 answered; **120 dropped by the 600 s forwarding timeout (F67)**: IFBench 21, AIME 71, LiveBench-Math 28; 0 HTTP 504, 0 client timeouts; `~/final_baselines`; B total 1721 SBU |
| 2026-11-06 | B | 28941949 | IFBench | final Baseline, seed 0 on vLLM 0.9.1 (replaces 28908434) | `c3ca457` | 47 min | 100 | 30.27 | A100; 582 requests; **41 examples lost to HTTP 500**: forwarding timeout (F67) and one failed batch of 32 (F68); 4 format failures; `~/final_baselines` |
| 2026-11-07 | B | 28941948 | AIME, LiveBench-Math | **final Baselines, seed 2**; client, forwarder and worker timeouts 7200 s (F67) | `cdef468` | 2 h 16 min | 289 | 30.67 / 49.37 | A100 (`gcn14`); 201 requests, 0 unanswered; 0 ReadTimeout, 0 worker timeouts, 0 HTTP 500/504, 0 chunk-call timeouts (F70), longest chunk call 32 s; failed examples 25 / 20, all format |
| 2026-11-07 | B | 28952775 | IFBench | Baseline, seed 2, same settings | `cdef468` | 1 h 30 min (Slurm limit) | 192 | – | stopped at 276/294 examples with 18 requests still running (F69); nothing dropped before that; void |
| 2026-11-07 | B | 28957882 | AIME, LiveBench-Math | **final Baselines, seed 0**, same settings | `cdef468` | 1 h 45 min | 225 | 30.00 / 50.20 | A100 (`gcn14`, after 28941948); 199 requests, 0 unanswered, 0 dropped; longest chunk call 20 s; failed 19 / 21, all format |
| 2026-11-07 | B | 28957883 | IFBench | Baseline, seed 0, same settings | `cdef468` | 1 h 30 min (Slurm limit) | 193 | – | stopped at 281/294 examples with 12 requests still running (F69); void |
| 2026-11-07 | B | 28957884 | AIME, LiveBench-Math | **final Baselines, seed 1**, same settings | `cdef468` | 1 h 45 min | 225 | 31.33 / 48.93 | A100 (`gcn72`); 201 requests, 0 unanswered, 0 dropped; longest chunk call 25 s; failed 21 / 12, all format |
| 2026-11-07 | B | 28957885 | IFBench | Baseline, seed 1, same settings | `cdef468` | 1 h 30 min (Slurm limit) | 192 | – | stopped at 283/294 examples with 11 requests still running (F69); void; B total 3137 SBU |
| 2026-11-06 | B | 28969642–44 | IFBench | Baselines, seeds 0, 1, 2 with `-t 04:00:00`; worker timeout 21600 s, chunk-call timeout 600 s (F70) | `cdef468` | pending | ≤512 each | – | the killed attempts and their caches are in `~/final_baselines/killed_1h30/`, so nothing is replayed (F55) |
| 2026-11-07 | B | 28971148 | IFBench | Baseline seed 2 on **gpu_h100** (pilot for the KV-cache question, §2.5), `GEPA_EXPERIMENT_DIR=$HOME/h100_pilot` | `cdef468` | pending | ≤768 | – | measures tokens/s with the 94 GB card; its result also serves as a 4th IFBench sample |
| 2026-11-05 | B | 28919079 | AIME, LiveBench-Math | Baselines, seed 0, vLLM 0.9.1 | `7adf8ca` | 45 min | 96 | 26.67 / 42.71 | A100 (`gcn23`); `~/arbor091` (vllm 0.9.1, torch 2.7.0, transformers 4.53.3); output in `~/vllm091_runs`; some requests hit Arbor's 300 s timeout (504) and score 0; B total 890 SBU |
| 2026-11-05 | A | 28918117 | HotpotQA | Baseline (4.3) | `fe389b8` | 17 min 46 s | 38 | 39.67 | A100 (`gcn40`); vLLM 0.8.5.post1; 7 failed examples; 1,190 requests; server ready after 196 s |
| 2026-11-05 | A | 28918118 | HoVer | Baseline (4.3) | `fe389b8` | 17 min 0 s | 36 | 35.67 | A100 (`gcn23`); vLLM 0.8.5.post1; 0 failed; 1,201 requests; server ready after 160 s |
| 2026-11-05 | A | 28922239 | HotpotQA | Baseline, run 2 | `9d890e5` | 15 min 43 s | 34 | 40.67 | meant as vLLM 0.9.1 but served 0.8.5.post1 (F63); 6 failed; 1,189 requests; `~/vllm085_rep2_runs` |
| 2026-11-05 | A | 28922242 | HoVer | Baseline, run 2 | `9d890e5` | 16 min 20 s | 35 | 34.33 | as 28922239; 2 failed; 1,198 requests; `~/vllm085_rep2_runs` |
| 2026-11-05 | A | 28923152 | HotpotQA | Baseline, vLLM 0.9.1 | `109e564` | 19 min 3 s | 41 | 42.33 | A100 (`gcn49`); vLLM 0.9.1 confirmed in the log; 6 failed; 1,193 requests; `~/vllm091_runs` |
| 2026-11-05 | A | 28923153 | HoVer | Baseline, vLLM 0.9.1 | `109e564` | 35 min 15 s | 75 | – | failed: vLLM compile-cache collision with 28923152 on `gcn49` (F64); A total so far 261 SBU |
| 2026-11-05 | A | 28927510 | HoVer | Baseline, seeds 0, 1, 2 (vLLM 0.9.1) | `672c0bb` | 44 min 8 s | 94 | 35.67 / 37.00 / 38.00 | A100 (`gcn71`); its own compile cache worked (F64); 0 failed in all three; 3,601 requests; 0 timeouts; `~/vllm091_runs` |
| 2026-11-05 | A | 28927513 | HotpotQA | Baseline, seeds 1, 2 (vLLM 0.9.1) | `672c0bb` | 39 min 44 s | 85 | 42.67 / 38.67 | A100 (`gcn43`); 0 and 5 failed; 2,396 requests; 5 timeouts, all retried; `~/vllm091_runs`; A total ~440 SBU |

---

## 6. Findings

Findings from the code, the forks (`gepa_study` branches), the paper, the authors' logs and our own runs. Step and pilot numbers in them (e.g. "step 1.7", "pilot 4.4") refer to the phase plan used until 2026-11-05, which is in git history (`60b8e91`).

**For the optimizer runs, these matter most:**
- **Chain and directories:** F18, F19, F20 (each step copies the previous step's cache; one directory per chain; never extract the authors' runs into `experiment_runs_data/`).
- **Budget:** F36 (GEPA uses 2–23% more metric calls than its budget; report both), F21 (`--dry_run` doesn't shrink it; pilots use `GEPA_MAX_METRIC_CALLS`).
- **Speed and packing:** F3 (GEPA is limited by latency, so runs can share a server), F37 (the authors' run times), F58 (Arbor's 64-token chunks), F44 (timing-test design).
- **Safety:** F57 (a GEPA run whose calls all fail never ends; the watchdog stops it), F54 and F56 (timeouts on long answers), F55 (check the request count), F63 and F64 (the Snellius job environment, the vLLM compile cache).
- **Fidelity:** F60 (the post-paper GEPA fix), F59 (context window), F61 (identical GEPA and GEPA+Merge math cells), F62 (the authors' vLLM and server), F65 (format failures vary by run), F35 (count reflections), F39 (search-tree checks), F66 (no W&B account needed).

### A. Hardware and compute

- **F1. GPU requirements in the launch-command generator are unused.** `requires_80_gb_gpu` and `num_gpus` are computed in `scripts/generate_launch_commands.py:26-27` but never written into the commands. They're leftovers from the authors' scheduler.
  → Pick resources in the SLURM script. 80 GB only matters for GRPO training: H100 (94 GB) is fine, A100 (40 GB) is not.
- **F2. Arbor shards the model across every GPU in its YAML.** It launches vLLM with `--tensor-parallel-size <number of gpu_ids>`, `--gpu-memory-utilization 0.9` and prefix caching (Arbor `inference_manager.py`). `arbor_inference.yaml` lists 4 GPUs.
  → Qwen3-8B (about 16 GB of weights) fits on one H100, so 4 GPUs cost 4× the SBU for a modest latency gain. Use a 1-GPU YAML (step 3.2).
- **F3. GEPA is limited by latency, not throughput.** Each iteration runs a minibatch of only 3 examples (`num_dspy_examples_per_gepa_step=3`, `gepa_artifact/gepa/gepa.py:56`), then one sequential reflection call, then the minibatch again. Full 300-example val evals happen only when a candidate improves.
  → The GPU is mostly idle during GEPA. Run GEPA, GEPA-MERGE and the ablation together against one vLLM server; continuous batching serves them at almost single-run speed.
  → MIPROv2 evaluates minibatches of about 35 with 32 threads, so it uses the GPU much better.
- **F4. The inference server is launched lazily with a 32,768-token context.** When the first chat request arrives and no server is running, Arbor starts vLLM with `max_context_length: 32768` (Arbor `api/routes/inference.py:27`). The `LAUNCH_KWARGS` 8192 from `experiment_configs.py` is not used on this path. DSPy requests `max_tokens=16384` (`scripts/run_experiments.py:59`).
  → An H100 has plenty of KV-cache room. On an A100 (40 GB), long contexts would sharply limit concurrency.
- **F5. Billing and budget.** H100 costs 192 SBU/GPU-h, A100 128 SBU/GPU-h, CPU 1 SBU per core-hour (`scontrol show partition`). H100 is cheaper per generated token: about 1.5× the price for about 2× the decode bandwidth. The account is a shared course budget that expires 2026-12-18 (`accinfo`).
  → Estimate before every phase. Pilot first. Confirm my share.
- **F31. CPU allocation granularity.** On `rome`, an 8-core request is rounded up to 16 cores (`sbatch --test-only`), and memory is 1,792 MB per core. So the smallest CPU job costs 16 SBU/h and has about 28 GB RAM. GPU jobs on `gpu_h100` get 16 cores and 180 GB RAM per GPU by default (`DefCpuPerGPU=16`, `DefMemPerGPU=184320`).
  → Pack several API-bound runs into one 16-core job.

### B. Environment and dependencies

- **F6. The default environment can't serve Qwen.** The default `pyproject.toml` doesn't include `arbor-ai`, and so no vLLM. Only `pyproject_grpo.toml` adds it, along with `flash-attn`.
  → Every Qwen3-8B run needs an Arbor environment, not just GRPO. Use a separate environment in `gepa_artifact/utils/arbor` (step 1.4).
  → `mv pyproject_grpo.toml pyproject.toml` rewrites tracked files. Do it only in a separate checkout, and only for GRPO.
- **F7. No lockfile anywhere.** Neither the repo nor the Arbor fork has a `uv.lock` (the DSPy fork has its own). The forks were last changed on 2025-07-03 (Arbor) and 2025-07-11 (DSPy); the artifact's first commit is from 2025-08-04. Resolving today would pull vLLM, TRL and transformers versions newer than the forks.
  → *Check (B):* every original run recorded its exact packages in `<run>/wandb/wandb/run-*/files/requirements.txt`: Python 3.12.9, dspy 2.6.23, litellm 1.74.0.post1, openai 1.93.2, pydantic 2.11.7, ujson 5.10.0, cloudpickle 3.1.1, datasets 3.6.0.
  → Pin the main environment's `uv.lock` to these instead of using a date cut-off (step 1.3), and commit it. The Arbor environment has no recorded versions, so it keeps `--exclude-newer 2025-08-04` (step 1.4).
  → *Update (A):* the same file lists every package of the environment, and the lock now pins all of them (F49).
- **F8. Login-node facts.** System `python3` is 3.9 (the project needs ≥3.11, and uv provides it). There's no `uv` and no `git-lfs` module; `tmux` is at `/usr/bin/tmux`.
  - Home: 200 GB / 1M inodes.
  - Scratch (`/scratch-shared` → `/gpfs/scratch1/shared`): 8 TB / 3M inodes. I couldn't confirm its retention policy.
  → Keep results and environments in home. Use scratch only for things that can be re-downloaded, and for Arbor's logs.
- **F9. Arbor logs every request and response.** It appends them to `./storage/logs/<timestamp>/inference_{requests,responses}.jsonl`, relative to the directory it's started from (Arbor `cli.py:19`, `core/config.py:25`).
  → For thousands of rollouts this grows to gigabytes. Start Arbor from a per-job scratch directory, and gitignore `storage/`.
- **F10. `ArborRunner` is dead code.** It needs an `env.sh` that isn't in the repo, and `run_experiments.py:234` resolves its YAML as `utils/arbor/...` relative to the working directory, while the fork lives at `gepa_artifact/utils/arbor/`. It never runs anyway, because `launch_arbor` is forced to `False` (`run_experiments.py:160`).
  → Start Arbor yourself in the job script.
- **F11. Repo-id spelling for the model cache.** Arbor strips the `openai/arbor:` prefix and asks vLLM for `qwen/qwen3-8b`.
  → Pre-download with exactly that string so the HF cache hits. Otherwise the job may re-download about 16 GB.

### C. API keys and external services

- **F12. Both keys are required at startup.** `run_experiments.py:486-487` asserts that `OPENAI_API_KEY` and `WANDB_API_KEY` are set. The Qwen config sends its own dummy key to Arbor.
  → For Qwen3-8B on HotpotQA, HoVer, IFBench, LiveBench-Math and AIME, `OPENAI_API_KEY=dummy` is enough.
- **F13. W&B is mandatory for GEPA and GRPO.** GEPA always sets `use_wandb=True` (`run_experiments.py:301`) and calls `wandb.login(key=..., verify=True)` (`gepa_artifact/gepa/gepa_utils.py:374`). GRPO reports to W&B (`report_to: "wandb"`). Baseline and MIPROv2 don't use W&B.
  → Get a free account (step 0.3). Compute nodes need outbound internet; verify in step 4.1.
  → *Check (B):* B's `feat/replay-and-analysis` adds an opt-in `GEPA_USE_WANDB=0` that skips W&B for GEPA. After that merge, only GRPO needs a W&B key, so W&B accounts no longer block the pilots.
- **F14. PAPILLON (PUPA) can't run without OpenAI.** Both the program's "untrusted" model and the LLM judge are `openai/gpt-4.1-mini` (`gepa_artifact/benchmarks/papillon/__init__.py:9`, `papillon_utils.py:45`). These use the undated alias, while the task-LM config pins `gpt-4.1-mini-2025-04-14`.
  → No key means no PUPA; after the model's retirement it can't be reproduced as specified. The judge cost is small (tens of dollars at most).
  → **How a PUPA rollout works** (`papillon/papillon_program.py`, `papillon_utils.py`):
    1. The local model (`craft_redacted_request`, ChainOfThought) rewrites the private query into a request without PII.
    2. The untrusted `gpt-4.1-mini` answers that request. This is **OpenAI call 1**.
    3. The local model (`respond_to_query`) writes the final answer from the private query plus the external answer.
    4. The metric asks `gpt-4.1-mini` whether the answer is at least as good as the reference, twice with A/B swapped (**calls 2–3**). The answer counts as good unless the judge prefers the reference in both orders.
    5. It also counts how many of the example's PII units appear in the redacted request (**call 4**).
    6. Score = (quality + (1 − leakage fraction)) / 2.

    Only steps 1 and 3 are optimized. These OpenAI LMs use DSPy defaults (temperature 0, cache on, 3 retries, max 1,000 output tokens), so identical calls are served from the cache and cost nothing. That includes GEPA's feedback function, which re-runs the same judge.
- **F15. GEPA reflects with the task model.** `teacher_lm` is never set, so the reflection LM is the task LM (`gepa_artifact/gepa/gepa.py:542`). MIPROv2 likewise uses the task LM to propose prompts.
  → Qwen3-8B runs reflect on their own outputs, so no OpenAI calls happen outside PUPA.
- **F16. `gpt-4.1-mini` is being retired.** Secondary sources report OpenAI will remove GPT-4o, 4o-mini, 4.1 and 4.1-mini from the API on 2027-01-14 (not yet checked against OpenAI's deprecations page). The paper (appendix E.3) reports costs for all benchmarks: GEPA $86, GEPA+Merge $67, MIPROv2 $76. All of Table 2 came to under $500, including Trace and TextGrad, which aren't in this repo.
  → Table 2 and PUPA are only possible until mid-January 2027, so decide early.
  → OpenAI's own model page already marks `gpt-4.1-mini-2025-04-14` as **Deprecated** (checked 2026-11-04). Only the exact shutdown date comes from secondary sources.
- **F17. External data sources.**
  - HotpotQA creates a ColBERTv2 client for a remote server (`hotpot_program.py:7`) but actually retrieves with HoVer's local BM25 `search` (`hotpot_program.py:5`). There's no dependency on that server.
  - Datasets come from the HF Hub: `hover` and `hotpot_qa` (both `trust_remote_code=True`), `Columbia-NLP/PUPA`, `livebench/math`, `AI-MO/aimo-validation-aime`, and `MathArena/aime_2025`. The Wikipedia abstracts also come from HF.
  → Download everything on the login node first (step 1.5).
- **F29. API errors silently become wrong answers.** `create_lm` sets `num_retries: 0` and `max_tokens: 16384` (`scripts/run_experiments.py:59-60`). The evaluators tolerate many errors (`max_errors` = 100× the set size, `gepa_artifact/gepa/gepa.py:308,325`; 10× for the test eval, `run_experiments.py:420`). So a failed API call, such as an HTTP 429 rate-limit error, is scored as a failed example instead of stopping the run. OpenAI lists 5,000 RPM / 2M TPM for `gpt-4.1-mini` on its lowest tier ("Build"). 32 threads with a 16k `max_tokens` each may exceed that if requested output tokens count toward TPM.
  → Before Table 2, run a ~$1 pilot (step 4.7: Baseline on one benchmark) and check `run_log.txt` for rate-limit errors. If there are any, lower `--num_threads` (this only affects speed) or enable retries as a documented, opt-in deviation.
  → *Check (B):* seen directly with a dummy key: failed calls were scored as zero, and the run carried on. A crashed or unreachable Arbor server has the same effect, so grep `run_log.txt` for errors after every pilot and run.

### D. Orchestration and data pitfalls

- **F18. Dry runs and pilots collide with real runs.** The run directory and cache directory depend only on benchmark, program, optimizer, LM and seed; `--dry_run` doesn't change them.
  - A finished dry run leaves `evaluation_results/`, which marks the real run complete, so it gets **skipped** (`run_experiments.py:203`).
  - A GEPA run started without `--use_cache_from_opt` creates its own cache directory. The real run later fails its `cache_from_source_run.txt` assertion.
  → Every dry run and pilot goes to a separate `BASE_EXPERIMENT_DIR` (step 3.1). The exceptions are pilots 4.3 and 4.5 (the Baselines and the first MIPROv2), and the API Baselines in 4.7. Those are real runs and go to the real directory.
- **F19. The authors' archive must not go into `BASE_EXPERIMENT_DIR`.** Extracting it into `experiment_runs_data/` has the same effect as F18: all runs are skipped, or start from the authors' caches.
  → Extract into `reference_runs_data/` (step 2.2).
- **F20. Runs wait on each other.** MIPROv2 waits for Baseline, and GEPA, GEPA-MERGE and the ablation wait for MIPROv2. Each waiting run polls every 100 s, forever, until the source run has `evaluation_results/` (`run_experiments.py:118-119`), then copies its DSPy cache. The wait is triggered by the `--use_cache_from_opt` CLI flag, which the generated commands include.
  → A job submitted too early burns GPU hours while idle. Chain jobs with SLURM dependencies: Baseline → MIPROv2 → GEPA variants.
- **F21. `--dry_run` doesn't shrink the budget.** It cuts the data to 2 examples, but GEPA still gets `max_metric_calls` from the hard-coded `known_max_calls` (6,871 for HotpotQA; `experiment_configs.py:188`, `run_experiments.py:354`). MIPROv2-Heavy keeps its heavy settings.
  → Smoke tests need a cap (step 3.1).
- **F22. Launch commands.** The generator emits 72 commands (6 benchmarks × 1 program × 6 optimizers × 2 LMs, seed 0), and it downloads all datasets while building the benchmarks. The Qwen commands contain `"api_base": "http://localhost:{portnum}/v1/"` literally: the substitution at `run_experiments.py:247` is unreachable because `launch_arbor` is forced to `False`.
  → Filter the commands and fill in the port yourself (step 3.4).
- **F23. BM25 artifacts land in the source tree.** The first HoVer or HotpotQA search downloads `wiki.abstracts.2017.tar.gz` into the current directory and writes the corpus and index into `gepa_artifact/benchmarks/hover/` (`hover_program.py:41`). Only `retriever_cache/` (`hover_program.py:89`) is gitignored. The init lock only guards threads, not processes.
  → Gitignore these files (step 0.2) and build the index once before parallel jobs (step 1.7). The shared `retriever_cache` is harmless, since BM25 is deterministic.
- **F24. Data splits are fixed.** The base class caps the splits at 150 train / 300 val / 300 test, sampled with seed 1 (`gepa_artifact/benchmarks/benchmark.py:30-32,50`). Some benchmarks define smaller splits themselves:
  - PUPA: 111 / 111 / 221 (`papillon_data.py`).
  - AIME: half of `aimo-validation-aime` each for train and val, and the 30 AIME-2025 problems repeated 5× as test, giving 150 (`AIME_data.py`).
  - LiveBench-Math: about ⅓ each (`livebenchmath_data.py`).

  A seed other than 0 reshuffles train and val only; test stays fixed (`run_experiments.py`).
- **F32. The `GRPO × gpt-41-mini` commands can't work.** GRPO calls `lm.reinforce()` before its first validation (`dspy/teleprompt/grpo.py` in the fork), and the OpenAI provider isn't reinforceable. So these 6 commands fail with an assertion before any API call.
  → Drop them from the selection. The same goes for `GPT × Abl-SelectBestCandidate`: it costs about €76 and appears in no paper table (Table 3 is Qwen3-8B only).

### E. Fidelity of the reproduction

- **F25. Thinking mode was on.** The Qwen3 sampling settings (temperature 0.6, top-p 0.95, top-k 20; `experiment_configs.py:41`) are Qwen's recommended settings for thinking mode. The paper doesn't say whether thinking was enabled.
  → *Check (B):* thinking was on. The planned check couldn't work, because the metric logs store `prediction: null`. The cached raw Qwen responses show the thinking in `reasoning_content`: Arbor starts vLLM without a reasoning parser, and litellm splits the thinking off on the client side.
  → Keep Qwen3's default chat template and `max_tokens=16384`; there's nothing else to configure. Thinking dominates token counts and compute, and pilot 4.3 measures it (F33).
- **F26. Hardware differs from the authors'.** They used 4-GPU tensor-parallel servers per run on 80 GB GPUs; I use 1 H100 shared by several runs. This should affect speed only.
  → Expect statistically similar but not identical results: sampling uses temperature 0.6 (Qwen) or 1.0 (GPT-4.1-mini). The DSPy cache keeps repeated identical calls deterministic within a run.
- **F27. The paper uses one seed per cell** (`SEEDS = [0]`).
  → Differences of a few points may be noise. *Check (B):* the artifact shows a 4.7-point swing between two runs of the same method. So extra seeds for the key comparisons (step 7.4) are the first extension, not an optional item.
  → The paper's own GRPO row in Table 1 prints an aggregate of 48.91, while its cells average 48.97: a small inconsistency in the paper.
- **F28. GRPO setup per the paper.** "1xH100/A100 (80 GB memory) with separate GPUs for inference rollouts": LoRA rank 16, α 64, dropout 0.05, learning rate 1e-5, 500 steps, group size 12, batch 48, so 24,000 rollouts. This matches `arbor_train.yaml` (inference on GPUs 1 and 2, training on GPU 0) and `TRAIN_KWARGS_GRPO_DEFAULT`.
- **F30. Not everything in the paper is in this repo.** These have no implementation here:
  - Table 2: Trace (OptoPrime), TextGrad and MIPROv2-No-Demos.
  - Table 3: BeamSearch.
  - Figures 7–8: NPU/CUDA kernel experiments with GPT-4o.

  GEPA-Qwen-Opt (Table 2) means the Qwen3-8B-optimized GEPA prompts evaluated on GPT-4.1-mini without changes. It needs a small extra evaluation script.
  → The reproducible scope is Tables 1–3 minus these rows. State this in the write-up.

### F. From B's check against the reference runs (2026-11-04)

- **F33. Qwen token counts aren't in the artifact, and our runs won't record them either.** `evaluation_result.txt` shows 0 tokens for every Arbor run, and the cached responses carry no usage.
  - The cause: Arbor's server (`arbor/server/services/inference/vllm_serve.py` in the fork) returns no `usage` field.
  - `calculate_stats` sums `usage` from the LM history (`scripts/run_experiments.py:47-48`), so it finds nothing.
  → The token table from step 2.3 exists only for GPT-4.1-mini. Pilot 4.3 has to count Qwen tokens itself: from Arbor's response log (F9), with the Qwen3 tokenizer (step 3.5).
- **F34. Target numbers against the shipped runs.** 46 of 58 cells match, 1 differs, and 11 have no run.
  - The one that differs: the paper's GPT-4.1-mini LiveBench-Math GEPA cell (64.13) is the GEPA+Merge run. The shipped GEPA run scores 59.43.
  - There are no reference runs at all for the Qwen3-8B math cells.
  → B's AIME-2025 and LiveBench-Math results can only be compared with the paper's numbers, not with the authors' logs (see §1).
- **F35. The appendix's reflection-call counts match only the GEPA+Merge runs.** Plain GEPA on AIME made 100 reflection calls; the appendix table says 24.
  → Count them in our own runs from `instruction_proposer_inpouts.jsonl` in each run directory. The typo in the name is in the code itself (`gepa_artifact/gepa/gepa.py:549`).
- **F36. GEPA-family runs overshoot their budget.** Every shipped GEPA-family run used 2–23% more metric calls than `max_metric_calls`, because the last validation pass isn't cut off.
  → Ours will do the same. Report both the budget and the calls actually used, and add up to ~20% to GEPA's SBU estimate (§2.2).
- **F37. The authors' run times, from their logs.**
  - Qwen3-8B on 4-GPU servers: GEPA 1.2–6.4 h, MIPROv2 1.1–4.7 h. GRPO on 3 GPUs: 12–24 h.
  - GPT-4.1-mini runs: 1–4 h, except GEPA on AIME at 9 h.
  → The old plan's ~15 h for GEPA on HotpotQA on one H100 is on the safe side, and the GRPO estimate matches. MIPROv2 keeps the GPU busy, so on one GPU it may take up to 2–3× longer than §2 assumes; pilot 4.5 settles this.
- **F38. Baselines can be replayed for free.** The shipped DSPy caches don't depend on the API key. So a reference Baseline can be replayed on Snellius without a GPU or a key (`scripts/replay_test_eval.py`, on B's `feat/replay-and-analysis`).
  - Optimized candidates still miss the cache, for an unknown reason.
  - A final GEPA test evaluation through the API costs about $1.
  → Use the replay as a pipeline check (pilot 4.0) before the first GPU job.
- **F39. Search-tree evidence for the write-up.**
  - GEPA keeps about 19 of about 100 proposals.
  - In 64–87% of iterations, the sampled parent isn't the current best candidate. In 8 of 10 GEPA runs, the final prompt descends from such a pick.
  - Picking the best candidate per validation example would add 7–27 points over the best single candidate.
  → This supports the paper's case for Pareto-based selection. Check whether our runs show the same pattern (step 8.3).

### G. From setting up on Snellius (A, 2026-11-04)

- **F40. The GRPO pyproject drops the `datasets` pin.** `pyproject_grpo.toml` lacks `datasets==3.6.0`, so `datasets` resolves to 4.x. Version 4 removed dataset loading scripts, which `hotpot_qa` and `hover` need (`trust_remote_code=True`).
  → If GRPO comes into scope, add `datasets==3.6.0` back in the separate GRPO checkout.
- **F41. Arbor isn't installed in its own environment, and its log directory is fixed.** The fork's `pyproject.toml` has no build system, so `uv sync` installs only its dependencies, not the `arbor` package (uv warns that the project "is not packaged"). `STORAGE_PATH` is hard-coded to `./storage` and can't be set from the YAML (Arbor `core/config.py`).
  → To start Arbor from a scratch directory (F9), put the Arbor venv first on `PATH` (Arbor launches vLLM as a plain `python -m ...` subprocess) and set `PYTHONPATH=<repo>/gepa_artifact/utils/arbor`. Checked on the login node: `arbor.cli` and `vllm_serve` then import from any directory.
- **F42. Arbor gives vLLM only 5 minutes to start.** vLLM is launched on the first chat request, and Arbor's client waits `connection_timeout=300` s for it (Arbor `inference_manager.py:133`). Arbor's own HTTP server answers immediately, so "the port is open" doesn't mean the model is ready.
  → Pre-download the weights (F11), and have the job send one warm-up chat request and wait for its reply before starting `run_experiments`. Otherwise 32 threads hit a cold server, and the failures are scored as zeros (F29).
- **F43. Arbor overwrites `CUDA_VISIBLE_DEVICES`** with the YAML's `gpu_ids` (`'0'`).
  → This is only safe if SLURM confines a 1-GPU job to its own GPU. The job template should log `nvidia-smi -L` and see exactly one GPU.
- **F44. Pilot 4.4 as first written would collide with itself.** Several copies of "GEPA on HotpotQA" share one run name, so they'd share one run directory and one cache (F18). And a 600-call cap allows only about 5 iterations: the first val eval costs 300 calls, and each iteration then averages about 6 + 0.19 × 300 ≈ 63 calls (acceptance rate from F39).
  → Use the real packing mix: n = 1 is GEPA; n = 3 is GEPA, GEPA-MERGE and Abl-SelectBestCandidate; n = 6 is those three at seeds 0 and 1. Each n gets its own `GEPA_EXPERIMENT_DIR`. Use a cap of about 1,500 plus `timeout 30m` per phase, and log `wc -l` of every `instruction_proposer_inpouts.jsonl` once a minute. GEPA needs `GEPA_USE_WANDB=0` (F13), so this pilot also depends on T9.
- **F45. The Arbor environment depends on the Python version.** uv's lock for the Arbor fork resolves `vllm` 0.8.5.post1 (torch 2.6.0+cu124, transformers 4.54.1) on Python 3.12, and `vllm` 0.10.0 only below 3.12. 0.8.5.post1 is Arbor's declared minimum and the first vLLM with Qwen3 support.
  → Build it with `--python 3.12` from the shared lockfile, so both of us run the same server.
- **F46. The fork commits aren't pinned.** `setup_gepa_repo.sh` checks out the tip of `gepa_study`. On 2026-11-04 those were DSPy `62dc3b6` (2025-07-11) and Arbor `113fc35` (2025-07-03).
  → Check that both checkouts are on these commits (`git -C gepa_artifact/utils/<fork> rev-parse --short HEAD`).
- **F47. Partition state on 2026-11-04: rolling reboots.** Most `rome`, `genoa` and `gpu_h100` nodes showed `drng@` (draining for a pending reboot); `rome` had no free node. The smallest `genoa` job is 24 cores (24 SBU/h, about 42 GB RAM, `sbatch --test-only`).
  → Short CPU jobs can go to `genoa` when `rome` is full. Expect GPU jobs to queue while the H100 nodes reboot.
- **F48. IFBench downloads NLP data at run time.** Importing `utils_ifbench/instructions.py` runs `spacy.cli.download('en_core_web_sm')` (line 34), which pip-installs the model into the active venv. The model isn't in `uv.lock`, so a later `uv sync` removes it again. The instruction checkers also call `nltk.download(...)` (`punkt_tab`, `punkt`, `stopwords`, `averaged_perceptron_tagger_eng`) into `~/nltk_data`. `generate_launch_commands` imports IFBench too.
  → IFBench runs need internet on the compute node (4.1). Fetch the NLTK data once on the login node before starting parallel IFBench processes, so they don't race on the same download. Confirmed: the `uv sync --locked` after PR #2 removed `en_core_web_sm` from the venv.
  → The first time, pip's install output goes to stdout and ends up at the top of `launch_commands` (8 extra lines). Keep only the command lines: `grep '^uv run python -m scripts.run_experiments'`.
- **F49. The original runs recorded their whole environment, and it wasn't one clean install.** B's review of PR #2: the run's `requirements.txt` (`AIMEBench_CoT_GEPA_gpt-41-mini`, 2025-08-01) has 302 lines, not just the 8 versions of F7.
  - With only those 8 pinned and `exclude-newer` for the rest, 58 packages still differed. The ones our runs use are `cachetools` 5.5.2 → 6.1.0 (dspy's in-memory cache) and `regex` 2024.11.6 → 2025.7.34 (dspy's JSON adapter, which the fork's ChatAdapter no longer falls back to: `chat_adapter.py:42–50` is commented out, so B's pre-F49 runs parse the same way). The others included torch 2.3.0 → 2.7.1, transformers and chromadb; `run_experiments` imports torch only on the disabled `launch_arbor` path.
  - The last 16 lines are setuptools 80.9.0's vendored copies (`setuptools/_vendor/*.dist-info`), not installed packages. That's why `zipp`, `platformdirs`, `packaging`, `importlib_metadata` and `typing_extensions` appear twice.
  - The environment also held leftovers of a torch 2.7 install next to torch 2.3.0: `triton` 3.3.1, `nvidia-cufile-cu12` and `nvidia-cusparselt-cu12`. Because of that, the list can't be locked for Python 3.11.
  → `[tool.uv] constraint-dependencies` holds the remaining 285 versions (the spaCy model is left out, F48), and `environments` limits the lock to Python 3.12. Result: 280 packages at the original version and none different. Not installed are the leftovers above, plus `graphviz` and `pydot`, which nothing depends on. Only one run's file has been compared; B checks whether all runs' files are identical (`md5sum .../files/requirements.txt | sort | uniq -c -w32`).


### H. From B's first Qwen runs (2026-11-05)

- **F50. Upstream `--dry_run` crashes on every benchmark.** The dry-run block in `run_experiments.py` also trims `benchmark.dev_set`, which no benchmark defines, so every dry run stops with `AttributeError` before the first LM call. B hit it on IFBench after 44 s (job 28908434); A hit it on AIME while testing 3.1b. Real runs aren't affected.
  → Fixed on `feat/pilot-caps` (`872e73a`: the line is skipped when `dev_set` is missing). Pilot 4.2 and every smoke test need this fix. B measured the server start-up in job 28908434 instead.
- **F51. The authors' Qwen runs used a 32,768-token context, not 8,192.** `run_experiments.py:160` sets `launch_arbor = False`, so the `launch_kwargs` of `LM_CONFIGS` (8,192) are never sent. Arbor then starts vLLM on the first chat request with its placeholder `{"max_context_length": 32768}` (Arbor `routes/inference.py:27`). Every shipped non-GRPO `arbor_logs.txt` shows `max_model_len 32768`; the GRPO runs also launch 8,192 for training. Requests ask for `max_tokens` 16,384.
  → Our warm-up request starts vLLM the same way, so we match the authors. The 20 GB A100 slices (`gpu_mig`) can't serve this: vLLM needs 4.5 GiB of KV cache and has 1.98 GiB. Use full GPUs.
- **F52. The math gap is run-to-run variance in output-format compliance, not the model's ability.** B's seed-0 Baselines: IFBench 36.56 (paper 36.90), AIME 19.33 (27.33), LiveBench-Math 33.58 (48.70). A rerun of the two math Baselines with `--seed 1` (job 28916886; identical prompts, parameters and server) gave **AIME 26.0 and LiveBench-Math 53.02**, i.e. the paper's numbers.
  - What goes wrong: Qwen3 sometimes writes its answer without the `[[ ## reasoning ## ]]` / `[[ ## answer ## ]]` headers (plain "Answer: 13" or `\boxed{}`), or with the answer header only. The ChatAdapter then rejects the output and the example scores 0 (F29). Many of those answers are correct when read by eye.
  - How often: LiveBench-Math 46 of 119 calls in seed 0 vs 12 of 127 in seed 1; AIME 21 of 63 vs 17 of 69. A difference this large between two identical runs is more than sampling noise would give, and we can't explain it yet.
  - What it is not: an unclosed `<think>` block by itself parses fine when both headers are inside it; the authors' only unclosed thinks are answers cut by `max_tokens` (finish reason `length`). Prompts are hash-identical across runs, the model files haven't changed since 2025-05-19, both servers lack FlashInfer.
  - Setup difference to the authors: their `arbor_logs.txt` show vLLM **0.9.1/0.9.2** (July 2025); our Arbor lock gives **0.8.5.post1** (F45). Being tested: job 28919079 reruns both math Baselines with vLLM 0.9.1 (`~/arbor091`, `ARBOR_VENV` in the job template, output in `~/vllm091_runs`).
  - Result of the vLLM 0.9.1 test (job 28919079): AIME 26.67, LiveBench-Math 42.71, i.e. also at the paper's level. Both versions can reproduce the paper; run 1 was an outlier. Some long generations hit Arbor's 300 s request timeout (`batch_request_timeout_seconds`) and score 0.
  - Update 2026-11-07: with no request dropped (F67, F70), B's three seeds agree within 1.33 points on AIME and 1.27 on LiveBench-Math (SD 0.67 and 0.64). The large swings above were measured on runs that lost requests to timeouts, so they don't show the run-to-run variance of a complete run.
  → Report math cells with ≥2 seeds and the format-failure count per run. Count parse failures after every run (`check_run.sh`). Proposal: move the Arbor lock to vLLM 0.9.1 (the authors' version for 22 of 27 runs; needs `transformers<4.54`) before the optimizer runs.
- **Timeouts and cache are infrastructure, not the method (decision 2026-11-05; the 1800 s values below were not enough, see F67: all three limits are 7200 s since 2026-11-06).** Raising Arbor's `batch_request_timeout_seconds` (300 → 1800 s, F54) and the LM client `timeout` (600 → 1800 s, F56) does not change the model, prompts, sampling parameters or optimizers; it only stops answers from being thrown away and scored 0 because one A100 is slower than the authors' 4-GPU server. A run that silently drops 45 of 276 math examples is a different experiment from the paper's, so the final Baseline numbers are produced with both timeouts raised and a fresh `DSPY_CACHEDIR` per job (F55). We report these three artifact defaults as reproducibility pitfalls. Final Baseline runs: jobs 28929002 (seed 0: AIME, LiveBench-Math) and 28929003 (seed 1: IFBench, AIME, LiveBench-Math) in `~/final_baselines`; IFBench seed 0 is kept from job 28908434 (clean run, no timeouts). Note that the seed does not change the Baseline test prompts, so "seed 0 / seed 1" here are two independent samples of the same evaluation.
- **F56. With the Arbor timeout raised, the next limit is the client's 600 s default.** Clean run 28926616: 0 HTTP 504, but 45 math examples (68 AIME rows, 22 LiveBench) raised `APITimeoutError` after 600 s and are scored 0. The LM is built in `run_experiments.py` (`dspy.LM(**config, **fixed_config)`, `max_tokens=16384`) without a `timeout`, so litellm/OpenAI's default applies. On one A100 with 32 threads a 16k-token answer can easily queue longer than that. → Add `"timeout": 1800` to `fixed_config` next to `max_tokens` (and keep Arbor's `batch_request_timeout_seconds` ≥ that), or run math on more GPUs (authors: TP=4). Expect a complete math run to take ~2 h on one A100.
- **F55 (corrected 2026-11-07). A rerun in the same directory replays that run's own DSPy cache; there is no shared cache.** `run_experiments.py:135-140` sets `DSPY_CACHEDIR` to `$GEPA_EXPERIMENT_DIR/experiment_cache_dirs/seed_<s>/<run>/.dspy_cache` before DSPy is imported, so this per-run folder is the cache, and a `DSPY_CACHEDIR` exported to the job is overwritten (B's 2026-11-07 jobs wrote only there; the exported scratch folders stayed empty). The first version of this finding was wrong: job 28908434's 2,724 requests include its MIPROv2 run, and about 200 requests is what a fresh AIME + LiveBench-Math run sends (B's clean runs: 199–201). So jobs 28916886 and 28919079 were fresh runs (they lost requests to the timeouts of F54, F56 and F67). Only job 28925784 (13 requests) replayed: it reused the cache of job 28920832 in the same folder. → Before a rerun, move both `experiment_runs/seed_<s>/<run>` and `experiment_cache_dirs/seed_<s>/<run>`. Otherwise a killed run is resumed from its cache, because `run_experiments.py` renames only the run folder (to `_backup_<time>`).
  → *Check (A):* A's runs aren't affected. Every A job sent its full set of requests to the server (1,189–1,201 per HotpotQA or HoVer Baseline), each run's own cache folder holds 5.6–5.8 MB, and in A's checkout (`feat/pilot-caps`) `dspy` isn't imported before `run_experiments` sets `DSPY_CACHEDIR`. The authors' per-run caches have content too (5.7 MB per Baseline, 136 MB for HotpotQA MIPROv2, 220 MB for HotpotQA GEPA). So the early import is specific to B's checkout (`7adf8ca`).
- **F54. Arbor's 300 s request timeout can wipe out a math run.** Job 28920832 (vLLM 0.9.1, seed 1, all three B Baselines in one job) lost 53/150 AIME and 39/126 LiveBench-Math examples to HTTP 504 (642 timed-out requests; `batch_request_timeout_seconds=300` in Arbor). Scores 14.00 / 35.96 are therefore meaningless; IFBench (36.73) was unaffected. Job 28919079 with the same server had only a few 504s, so the loss depends on how many long generations land in the same batch window. → Before the next math run, raise `batch_request_timeout_seconds` in the Arbor config (or give the math benchmarks their own job) and treat any run with many 504s as void.
- **F53. AIME's 5 copies per problem are mostly not independent.** The test set is 30 problems × 5 (150 examples), but our job sent only 63 AIME requests: identical requests are answered from DSPy's cache. In our run, 2 of 26 logged problems have different scores across copies; in the authors' GPT runs 7–9 of 30 do.
  → "Average over 5 tries" is closer to one try per problem. Not yet checked why some copies still differ (probably threads that miss the cache at the same time).

### I. From A's checks of the pilot tooling (2026-11-05)

- **F57. A GEPA run whose LM calls all fail never ends.** A failed call never reaches the metric, so GEPA's metric-call count doesn't grow, and the budget check at the start of each iteration (`gepa.py:369`) never stops the run. GEPA logs "No feedback samples, skipping" and tries again. With no server running, an AIME GEPA dry run did 2,083 iterations in 5 minutes with `total_num_evals` stuck at 2, and wrote 50 MB of log (login node, `feat/pilot-caps`). Baseline and MIPROv2 runs just finish with zeros (F29).
  → `qwen_job.sh` on `feat/pilot-caps` has a watchdog. It stops the job's runs and exits 1 if Arbor exits, if vLLM fails its `/health/` check 3 times in a row, or if Arbor logs nothing for `STALL_MINUTES` (default 20) while runs are active. Tested against a fake server on the login node: crash and stall were both caught. GEPA pilots also get a `timeout`.
- **F58. Arbor generates text in 64-token chunks and rebuilds the prompt between them.** Arbor's server (`arbor/server/services/inference/vllm_serve.py`) isn't vLLM's OpenAI server.
  - It pools requests that have the same sampling parameters and generates `token_chunk_size` = 64 tokens for up to 32 of them per `llm.chat` call. New requests go before ones that are being continued.
  - Each continuation re-sends the conversation plus `{"role": "assistant", "content": <text so far>}` with `continue_final_message=True`. With Qwen3's chat template (the April and the May 2025 revisions behave the same), a response that is still thinking becomes `<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\n<text so far>`. The model sees an empty, closed think block before its own unclosed one. That fits the unclosed `<think>` of F52, which costs only 1–2 points there.
  - It's a likely reason for the slow rollouts in §4 (one generate call every 64 tokens, small batches). Not measured.
  - The chunking has been in Arbor since `d050c8f` (2025-05-31), before our `113fc35`, so the authors very likely ran with it too. To confirm, look for `[LIFECYCLE] Activated new pool` lines in the authors' `arbor_logs.txt`. B, 2026-11-07: they are in all 24 (F71).
  → Keep Arbor as it is for the reproduction. A larger chunk or vLLM's own server would probably be much faster, but it changes how text is generated. Decide with the scope (§2.4: cost against fidelity).

### J. From A's Baselines, seeds and the vLLM check (2026-11-05 and 06)

- **F59. The paper's context window differs from its own code.** Appendix E.2: "All models are allowed a context window of upto 16384 tokens for inference." The code starts vLLM with a 32,768-token context (every non-GRPO Arbor log, F51) and allows 16,384 output tokens per call (`max_tokens`).
  → We follow the code and the logs. Say so in the write-up.
- **F60. Upstream changed GEPA after the paper's runs.** Commit `29220ba` (2026-02-07, part of our base `cbefbc1`) fixes the seed program's validation subscores being overwritten by the Pareto front: both pointed to the same list (`gepa_utils.py`, now wrapped in `list(...)`). The paper's runs (July 2025) ran without the fix. Baselines and MIPROv2 aren't affected.
  → Decide before the GEPA runs: keep the fix (correct code) or revert it (the paper's behaviour).
- **F61. In all four math cells the paper prints the same score for GEPA and GEPA+Merge:** Qwen3-8B AIME 32.00 and LiveBench-Math 51.95, GPT-4.1-mini 59.33 and 64.13. Where the artifact can show it (GPT-4.1-mini LiveBench-Math), these are two different runs (59.43 and 64.13), and the paper printed the GEPA+Merge score in both cells (F34). For the Qwen math cells there are no shipped runs to check.
- **F62. The authors' serving setup, from their 24 Arbor logs.**
  - vLLM 0.9.1 in 19 runs and 0.9.2 in 5 (HotpotQA's GEPA, GEPA-MERGE and Abl-SelectBestCandidate, and two GRPO runs). None used our 0.8.5.post1 (F45).
  - Their HotpotQA Baseline server: `tensor_parallel_size=4`, a KV cache of 1.73M tokens (room for 53 full 32,768-token requests), vLLM seed 0. Ours on one A100 40 GB: 1 GPU, 133k tokens (room for 4), seed unset. Both: Flash Attention, no FlashInfer, 32,768-token context, 0.9 GPU memory, prefix caching.
  - Their runs ran 2025-07-05 (HoVer) to 2025-07-08 (HotpotQA). A benchmark's three GEPA-family runs started together.
  → `envs/arbor-vllm091-requirements.txt` (vLLM 0.9.1, torch 2.7.0, transformers 4.53.3, 165 packages) builds `~/arbor-envs/vllm091`. Resolve it with `uv pip compile --no-config`: otherwise uv also applies the main project's pins, because the Arbor fork sits inside our repo. Pass the environment as `qwen_job.sh`'s 2nd argument.
- **F63. Snellius drops the environment of `sbatch`.** `/run/slurm/conf/cli_filter.lua` sets `--export=NONE` for every job, so a variable set in front of `sbatch` never reaches it.
  - A's first two "vLLM 0.9.1" jobs (28922239, 28922242) therefore served 0.8.5.post1. They count as second 0.8.5 runs (`~/vllm085_rep2_runs`).
  - A pilot started that way would lose `GEPA_EXPERIMENT_DIR` and `GEPA_MAX_METRIC_CALLS` and run uncapped in the real directory.
  → Per-run variables go at the start of each command line, the Arbor environment is `qwen_job.sh`'s 2nd argument, and anything else goes through `sbatch --export=VAR=value`. Check the job log's `arbor venv:` and `job settings:` lines.
- **F64. vLLM 0.9 jobs that start together can break each other.** vLLM compiles the model at start-up into `~/.cache/vllm/torch_compile_cache`. Jobs 28923152 and 28923153 started in the same second on `gcn49`; the second read the first's half-written artifact ("vLLM failed to compile the model"), and the warm-up retried for 35 min (75 SBU, no result).
  → `qwen_job.sh` sets `VLLM_CACHE_ROOT` per job and stops the warm-up at the first fatal vLLM message.
- **F65. HotpotQA loses 0–7 test examples per run to format failures.**
  - A failed example: one of the program's 4 calls comes back in a format the ChatAdapter can't read, so DSPy raises, the example scores 0 and isn't logged in `test.jsonl`. There's no retry and no fallback parser. Each failed example costs 0.33 points.
  - Ours: 7 and 6 on vLLM 0.8.5; 6, 0 and 5 on 0.9.1 (seeds 0–2). Authors: 0 in the Baseline, 1 in MIPROv2, 6 in GEPA. HoVer: 0 and 2 on 0.8.5, 0 in all three 0.9.1 seeds; authors 0.
  - What Qwen wrote in run 1: another step's section name in 4 of 7 (e.g. `[[ ## answer ## ]]` in a summary step), reasoning left inside an unclosed `<think>` in 3, and broken text in 3 (a split header, "the his time working for"). Several have more than one.
  - Not timeouts: Arbor's 300 s limit returns HTTP 504 with no text, and its client retries 3 times. Our runs had 0–5 per run, none reached DSPy, and every response ended with `stop`. For long math answers it's different (F54, F56).
  - Any two HotpotQA Baseline runs, ours or the authors', give the same score on 84–90% of examples; our three 0.9.1 seeds 86–88% (HoVer 82–86%).
  - `test.jsonl` can log an example twice (HotpotQA seeds 0 and 2, HoVer's second 0.8.5 run), so count unique `idx_in_split`, as `check_run.sh` does since `b7fc043`.
  → Keep the authors' parser. Our seed 1 lost none, like the authors' Baseline, so the count varies by run rather than pointing to a setup difference. Report failed examples next to every score. Serving differences that remain: 1 vs 4 GPUs, the vLLM seed and the KV cache (F62).
- **F66. GEPA runs without a W&B account.** With `WANDB_MODE=disabled`, wandb 0.19.10's `wandb.login(key=..., verify=True)` accepts a dummy key, and `wandb.init` and `wandb.log` do nothing. Those are the only W&B calls GEPA makes (`gepa_utils.py:372–387`, `gepa.py:350–587`). `qwen_job.sh` sets `WANDB_MODE=disabled` and `GEPA_USE_WANDB=0` by default.
  → W&B is only needed for GRPO (F13).


### K. From B's Baseline reruns (2026-11-06 and 07)

- **F67. Arbor has a third timeout, 600 s in the forwarder, and it truncated B's Baselines of seeds 0 and 1.** `gepa_artifact/utils/arbor/arbor/server/services/inference/vllm_client.py:227` posts each request to the vLLM worker with `timeout=600`. After F54 (worker, 300 → 1800 s) and F56 (client, 600 → 1800 s) this became the first limit to fire: the forwarder raises `httpx.ReadTimeout`, Arbor answers HTTP 500, the client logs `InternalServerError` and the example scores 0. The checks used until now (HTTP 504 and `APITimeoutError`) don't see it.
  - Requests dropped: job 28929002 88 of 239 (AIME 76, LiveBench-Math 12); job 28929003 120 of 811 (IFBench 21, AIME 71, LiveBench-Math 28); job 28941949 41 of 582 (IFBench).
  - The worker kept generating for the dropped requests. In job 28929002 it finished 61 of the 88 after the cut-off, 45 with a normal answer and 16 at the 16,384-token limit; the other 27 were cut by the worker at 1800 s. So real answers were thrown away, the scores in §1 are too low by an unknown amount, and their agreement with the paper is not evidence of a reproduction.
  - The authors didn't hit it: their Baseline logs contain no timeout of any kind (IFBench Baseline: 0 worker timeouts, 0 HTTP 5xx, 24 answers at the token limit). Our server produces 180–245 tokens/s in total on one A100 with 32 threads (`[LIFECYCLE]` lines in the Arbor log), so a 16,384-token answer takes 16–27 min.
  → All three limits are now 7200 s. Worker (`vllm_serve.py`, `batch_request_timeout_seconds`) and forwarder (`vllm_client.py:227`) are local edits in the gitignored Arbor folder, the client `timeout` is in `run_experiments.py` (`83d1ad3` on `feat/replay-and-analysis`). A: make the same two Arbor edits. After every run check that `inference_requests.jsonl` and `inference_responses.jsonl` in Arbor's log folder have the same number of lines and that the Arbor log has no `httpx.ReadTimeout`, no `Timeout for chat request` and no `500 Internal Server Error`. Seed 2 (jobs 28941948 and 28952775) is the first run with the fix; seeds 0 and 1 have to be redone with it.
- **F68. One bad partial answer can fail a whole batch of 32 requests.** In job 28941949 vLLM rejected one request with `continue_final_message is set but the final message does not appear in the chat after applying the chat template`, and Arbor then failed all 32 requests of that batch (`LLM result mismatch in batch processing`); the client was still waiting for 14 of them. This is part of the artifact: the authors' logs show the same error in 11 of their 24 Qwen runs (409 requests in HotpotQA GEPA, 128 in IFBench GEPA), but in none of their Baselines.
  → Don't patch it. Count `LLM result mismatch` lines per run, and repeat a Baseline run that has one.
- **F67 confirmed (2026-11-07).** B's six jobs with the three 7200 s timeouts (28941948, 28952775, 28957882–85) dropped nothing. The math jobs have as many responses as requests, the IFBench jobs lack only the requests still running when Slurm stopped them (F69), and all six have 0 `httpx.ReadTimeout`, 0 worker timeouts and 0 HTTP 500/504.
- **F69. With nothing dropped, IFBench needs more than 1 h 30 on one A100.** The answers that run to the 16,384-token limit now take 63–79 min each (17–19 per run), and IFBench chains two calls per example. All three IFBench jobs were stopped by Slurm at 276–283 of 294 examples with 11–18 requests still running. A run writes `evaluation_results` only at the end, so nothing was kept. AIME + LiveBench-Math took 1 h 45 to 2 h 16 (limit 2 h 45).
  → IFBench runs with `-t 04:00:00` (jobs 28969642–44). Size the optimizer runs' limits on these times, not on the old truncated runs.
- **F70. Two more hazards in Arbor and in B's job script, found by a code audit on 2026-11-06. Neither fired in any run, and both are fixed for the IFBench reruns.** A third one did fire (F72).
  - Each 64-token chunk call has a hard-coded 60 s limit (`vllm_serve.py:870, 929, 960`). On expiry the worker's late reply stays in the pipe, and every later call reads the previous call's outputs, so answers are silently swapped between requests for the rest of the job. On the A100-40GB the KV cache holds 137,040 tokens (the authors had 1,735,712 with TP=4), so 32 long contexts are re-prefilled on every call. The longest call in B's six jobs took 32 s. The limit is now 600 s.
  - B's `slurm/qwen_job.sh` ran `pkill -u $USER -f vllm_serve` on exit, which also kills the vLLM worker of any other B job on the same node. No two B jobs ever ran on one node at the same time (sacct). The script now kills only processes with its own `SLURM_JOB_ID`, and it refuses to start if the timeout edits are missing. A's `scripts/snellius/qwen_job.sh` has no such `pkill`.
  → After every run, `grep -cE 'timeout for (first|continue)-chunk chat after|Timeout waiting for worker response|LLM result count mismatch|Timeout for chat request' logs/arbor-<job>.log` must be 0. A: make the two `vllm_serve.py` edits as well (lines 870, 929 and 960 `timeout=600.0`, line 414 `default=21600`).
- **F71. The artifact changes answers in three ways that raise no error. The authors ran the same code, so this is part of what we reproduce.** Report it, don't fix it.
  - `top_k=20` never reaches vLLM. Arbor's `OAChatCompletionRequest` has no `top_k` field, and `create_pool_signature` drops sampling keys that come in `extra_body` (`vllm_serve.py:100-112, 492-545`). vLLM therefore samples with temperature 0.6 and top_p 0.95 only.
  - The empty think block in front of the model's reasoning in every 64-token continuation, as A described in F58. Checked: rendered with the model's tokenizer on Snellius, and all 24 of the authors' Qwen logs contain `[LIFECYCLE] Activated new pool`, so they generated the same way. After `</think>` the continuation matches what was generated, and the returned text doesn't show the inserted block.
  - DSPy's `ParallelExecutor` resubmits an example once 3 or fewer are left and it has run for 120 s, and scores the copy that finishes first (`dspy/utils/parallelizer.py:24-25, 174-191`). The slowest examples at the end of a run are therefore biased toward shorter answers. B's math jobs sent 43–45 repeated prompts each; this count covers these resubmissions and the AIME copies sent before the first answer was cached.
  - Reported by the audit but not checked yet: the forwarder sends a request again from scratch after a worker 500/504 (up to 3 attempts); each chunk re-tokenises the decoded text; LiveBench-Math's AMPS_Hard check scores 0 when sympy needs more than 8 s.
- **F72. When Arbor's worker timeout fires, answer text goes to the wrong requests.** After a chunk call, `vllm_serve.py:986-991` (and the same loop at 1051) matches the results to the requests with a counter that skips requests already marked done. The timeout handler (`vllm_serve.py:1398-1401`) marks a request done while its chunk call is still running. The counter then gives every later request in that call the previous request's 64 tokens, and the last result is dropped. Nothing is logged, and an answer that still parses is scored with another example's text.
  - Seen in B's truncated job 28929002. After worker timeouts at 20:18:16, the AIME answer about the right triangle `BKLC` jumps mid-sentence to the `x1 = 25/11` sequence problem, then to the 16-chairs problem, and ends with another problem's `[[ ## answer ## ]] 7`.
  - It can happen once per `Timeout for chat request` line in the Arbor log. B's final runs: 0. B's truncated runs: 27 (28929002) and 3 (28941949). The authors' four Qwen Baselines: 0, so their Baseline scores are unaffected. Their optimizer runs: HotpotQA GEPA 53, Abl-SelectBestCandidate 28, GEPA-MERGE 18, MIPROv2 1; IFBench GEPA 52, MIPROv2 3, Abl-SelectBestCandidate 3; HoVer GEPA 3; PUPA 0. Some of their GEPA-family runs therefore worked with partly mixed answers (not checked whether during the search or the final test pass). Worth reporting.
  - A's HotpotQA Baselines had up to 5 worker timeouts per run (300 s default, §3 table), so answers finished after them may contain another example's text. The "broken text" format failures of F65 would fit this.
  → Keep the worker timeout longer than the job (B: `default=21600` at `vllm_serve.py:414`, so it can't fire). Treat a run with any `Timeout for chat request` as void. A: count them for jobs 28923152, 28927513 and 28927510 (`grep -c 'Timeout for chat request' logs/arbor-<job>.log`); a rerun with the edits costs 34–94 SBU per job (§3 table).
- **F73. The client timeout is part of DSPy's cache key.** `dspy/clients/lm.py:95` leaves only `api_key`, `api_base` and `base_url` out of the key (`dspy/clients/cache.py:95`), so B's `"timeout": 7200` in `create_lm` changed every key.
  - The replay scripts no longer found the authors' cached answers. Fixed in `7a8dbad` on `feat/replay-and-analysis`: they set `GEPA_REPLAY_CACHE=1`, which drops the timeout again, and real runs keep it. Checked: 30 AIME test examples of `AIMEBench_CoT_Baseline_gpt-41-mini` all come from the cache with their stored scores.
  - A chain reuses the Baseline's cache only if every step runs with the same `create_lm` (F20). B's chains keep the timeout in every step. A's runs have no timeout (`feat/pilot-caps`), so if A adds it before MIPROv2, MIPROv2 regenerates the Baseline's calls instead of reading them.
  → A: decide before the optimizer runs. Long reflection and MIPROv2 calls need the timeouts as much as B's math runs did (F67). Adding all of them now and rerunning the Baselines with them keeps each chain consistent and also clears F72.

---

### L. From B's checks of the optimizer code and the authors' optimizer and GRPO runs (2026-11-07)

- **F74. MIPROv2-Heavy as it runs, and where the budgets come from.** 18 few-shot sets (16 bootstrapped one example at a time), 9 instruction candidates per predictor (one is the original), 11 + 27 × predictors sequential proposal calls, then T trials of 35-example val minibatches with a full val pass every 5 trials and one at the end: T = 66 / 33 / 27 for 4 / 2 / 1 predictors, full passes 14 / 7 / 6, saved programs 81 / 41 / 34 (`mipro_optimizer_v2.py:35-39, 221-228, 506-507, 610`). Metric calls = bootstrap attempts + |val| + 35 T + |val| × F reproduces the hard-coded GEPA budgets: LiveBench 47 + 1792 = 1839, HotpotQA 61 + 6810 = 6871, IFBench 36 + 3555 = 3591 (3593); HoVer gives 6901 against 7051 (origin unclear). The AIME budget 1839 is LiveBench's number reused (`experiment_configs.py:193-201`).
- **F75. Our AIME MIPROv2 runs without minibatching.** Our AIME val set has 45 examples, below the 50-example minibatch threshold (`mipro_optimizer_v2.py:33, 243-244`), so each of the 27 trials is a full 45-example pass: about 1,300–1,400 metric calls against GEPA's 1,839. The authors' AIME MIPROv2 run (GPT) used a 90-example val set of an older split. → Keep the artifact as it is and report the mismatch.
- **F76. The optimizers do not run with the 32 threads of the command line.** `run_experiments.py:344` passes `--num_threads` only to optimizers whose init_args contain the key. MIPROv2's evaluations therefore run with DSPy's default 8 threads (bootstrapping and the proposal calls are single sequential requests; the authors' Arbor pools never exceed 9 in the first 90% of their MIPROv2 logs), and GEPA's val passes run with `os.cpu_count()` threads (authors' pools 114–192; a Snellius A100 node has 72 cores), minibatch 3, reflection 1, merge test 5. Only the final test evals use 32. → We keep these effective values (speed only; §2.5) and say so in the write-up.
- **F77. A MIPROv2 run can silently return the Baseline, and a failed proposal call kills it.** The returned program is the default unless a full val pass beat it (`mipro_optimizer_v2.py:524, 822-825`): compare 'Default program score' with 'Returning best identified program with score' in `run_log_stderr.txt`. A failed 'generate instruction' call has no try/except and aborts the run with exit 1 (`grounded_proposer.py:248-258, 418-429`); bootstrapping errors are swallowed and evaluation failures score 0.
- **F78. GEPA's counted budget.** total_num_evals = |val| (base pass) + 6 per proposal + |val| per accepted candidate (+5 per merge test, +|val| per merge that passed; `gepa.py:250, 454-472, 567-583`); the stop test is only at the top of an iteration, so the overshoot is one iteration (authors: +3.2%, +3.7%, checked in `gepa_state.bin`). 3 capture calls per skipped iteration are not counted (HotpotQA GEPA: 34 skips). Merge credits can exceed 5 merges (authors 8 and 6).
- **F79. What the copied caches give, and a misleading authors' log.** The copied MIPROv2 cache gives GEPA its base val pass for free; the copied Baseline cache gives MIPROv2 almost nothing (test prompts only). The authors' HotpotQA MIPROv2 log (4,485 requests for 28,464 predictor calls) is an outlier, most likely a restarted run with a warm cache; HoVer 88%, IFBench 94% and PUPA 79% of predictor calls were fresh. → Size MIPROv2 from 80–95% of metric calls × predictors; expect ~1,700–1,900 fresh generations plus ~34–100 reflections per math GEPA run.
- **F80. Neither optimizer resumes.** A killed run is renamed `*_backup_<time>`, its cache stays, and the rerun replays the same seeded trajectory from the cache until it runs out, re-counting the budget from 0 (`run_experiments.py:204-216`; GEPA's load path is unreachable). → One run per job with a full-length limit; results are written only at the end.
- **F81. Upstream's post-paper GEPA fix (F60) has no effect in our configuration.** `prog_candidate_val_subscores` is read only for merge parents, and candidate 0 (the only aliased entry) can never be a merge parent because merges need a common ancestor (`gepa_utils.py:83-95, 200-208`; `merge_programs.py:35-67`). → Keep the fix; the paper's behaviour is unchanged. A to confirm.
- **F82. Sharing a server, and the chain asserts.** Arbor pools requests by sampling signature and activates a new one only when the active pool is empty (`vllm_serve.py:657-680`). MIPROv2's proposer uses temperature 0.5 + U(0.01, 0.05) per proposal and 1.0 for 11 dataset-summary calls, so behind another run's 0.6 pool these calls starve and hit the client timeout: MIPROv2 runs alone on its server. GEPA-family runs (all 0.6) and Baselines can share one server, subject to the 32-slot FIFO and F68/F70/F72. `--use_cache_from_opt` must name the optimizer's own source or the run fails an assert; if the cache dir is missing it waits forever for the source run's `evaluation_results/` (F20), and a run dir with `evaluation_results/` is skipped with exit 0, so `afterok` cannot see a stale result (`run_experiments.py:109-133, 163-165, 204-219`).
- **F83. The authors' GRPO runs, from their logs (HotpotQA, HoVer, IFBench, PUPA).** Validation curves flat within noise (HotpotQA 46.67 before training, 25 later validations mean 48.36, SD 1.84, best 52.33 at step 220 = the selected checkpoint, test 43.33; HoVer 35.33 → mean 36.16, best 39.0 at step 500, test 38.67; IFBench 57.0 → mean 59.4, best 63.47 at step 460, test 35.88); the best-validation gains equal the expected maximum of 25 noisy validations (E[max of 25 N(0,1)] = 1.97 SD = +5.7 points); no early stopping in the code, it keeps the best of 26 checkpoints (`grpo.py:236, 355-366, 1054-1063`). 65.7% / 56.4% / 76.0% of the 12-rollout groups have identical rewards, i.e. zero advantage (`grpo_training.py:256-269`); only 35 / 44 / 24% of rollouts carry a signal; PUPA (continuous reward) 38% zero groups and the only clear gain. Training reward rises over 13 epochs of the same 150 examples while validation stays flat. The trainer learns on reconstructed completions behind an empty think block (`grpo.py:913-929`, 129–344 tokens), all runs show KL spikes of 1e3–5e6, every module gets the program-level reward (`grpo.py:880, 928`). The in-run test evaluations of the HotpotQA and IFBench runs failed (checkpoint server did not start; 0/300 and 0/294 in the logs): the paper's 43.33 and 35.88 are not reproducible from the shipped files. Hardware per the logs: vLLM TP=2 on two 80 GB GPUs plus one training GPU, 12–24 h. Scripts: `analysis/grpo_logs.py` (to be added from B's notes).
- **F84. Failed math examples are unfinished thinking, and AIME is 30 problems.** Of the 140 failing completions in B's final math runs, 136 still have an unclosed `<think>` and no output headers; 96 are longer than 20,000 characters. The five AIME copies of a problem share one `idx_in_split` and mostly one cached answer (49–52 distinct answers per run; the authors' GPT run 69), so an AIME run is about 50 independent answers over 30 problems and one problem moves the score by 3.33 points; problem idx 8 (the rotated parabola) fails in all 15 copies. → Count failed examples as test size − the progress bar's denominator; report AIME with this caveat.

## 7. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-04 | Run everything on Snellius; the Mac is a fallback only | One environment keeps results consistent |
| 2026-11-04 | Start with Qwen3-8B on the 5 OpenAI-free benchmarks; GRPO, PUPA and GPT-4.1-mini are optional | No API keys yet; GRPO is about 5–10× the cost of everything else combined |
| 2026-11-04 | After setup, run pilots (Phase 4) and re-estimate the compute (Phase 5) before any main run | The first-principles estimate is ±2×, and the course budget is shared and expires 2026-12-18 |
| 2026-11-04 | First scope: Table 1 on HotpotQA, HoVer, IFBench, AIME-2025 and LiveBench-Math with Baseline, MIPROv2, GEPA and GEPA+Merge | No OpenAI key yet, so no PUPA. GRPO alone is ~84% of all SBUs |
| 2026-11-04 | Split by benchmark: A has HotpotQA and HoVer; B has IFBench, AIME-2025 and LiveBench-Math. Shared branch `reproduction` | Each benchmark's chain has to stay in one experiment directory (F20); the split gives each of us a similar SBU load |
| 2026-11-04 | Adopt B's check of the plan: corrections to F7, F13, F25, F27 and F29, and new findings F33–F39 | B checked the plan against the authors' logs, caches and code |
| 2026-11-04 | Pin the main environment to the package versions recorded by the original runs (F7), instead of a date cut-off | The exact versions are available, which is closer to the original setup |
| 2026-11-04 | Phase 2 stays done on B's laptop and isn't repeated on Snellius | The reference analysis doesn't depend on where it runs |
| 2026-11-04 | Main environment: the 8 recorded versions as pins, plus `exclude-newer = "2025-08-04T00:00:00Z"` in `[tool.uv]` for all other packages *(replaced by the full pin list, see below)* | The original runs recorded only these 8 versions here; without a cut-off the other ~300 packages would resolve to 2026 versions. Keeping the cut-off in `pyproject.toml` means `uv sync --locked` needs no extra flag |
| 2026-11-04 | Arbor environment on Python 3.12 with vLLM 0.8.5.post1, built from one shared lockfile | uv resolves a different vLLM for other Python versions (F45); one lock keeps our servers identical |
| 2026-11-04 | Step 1.7 runs on `genoa` (24 cores) | `rome` had no free nodes during the rolling reboot (F47); same price per core-hour |
| 2026-11-04 | Main environment: pin all 285 package versions of an original run (`constraint-dependencies`) and lock for Python 3.12 only | B's review of PR #2 (F49): the full list exists, and with 8 pins, 58 packages still differed, including two that dspy uses during runs. Every original run used Python 3.12.9 |
| 2026-11-05 | Serve every further run with the authors' vLLM 0.9.1, passed as `qwen_job.sh`'s 2nd argument | 19 of the authors' 24 Arbor logs show 0.9.1 (F62). No measurable score difference from 0.8.5 at Baseline level, but it's the authors' setup |
| 2026-11-05 | Raise Arbor's request limit and the client's timeout to 1,800 s, and use a fresh DSPy cache per job (B) *(replaced on 2026-11-06, see below; the cache export had no effect, F55)* | Timeouts and the cache are infrastructure, not the method; see the decision in §6 H (F54–F56) |
| 2026-11-06 | Objective: Table 1 without an API key, 3 seeds per cell, reported as mean ± SD | A's Baselines vary by 1–2 points SD between seeds, and the paper's single runs fall inside that spread |
| 2026-11-06 | The phase plan is removed from this file; it's in git history (`60b8e91`) | Setup and pilots are done; what remains is filling §1 |
| 2026-11-06 | Raise all request limits (B): client and forwarder 7,200 s, worker 21,600 s, chunk call 600 s | The 1,800 s values still left a 600 s forwarder limit that dropped answers (F67), and a worker timeout mixes answers between requests (F72). None of this changes the method |
| 2026-11-07 | A run with any worker timeout is void. Before a rerun, move the run folder and its cache folder. IFBench jobs get 4 h (B) | F72; the cache is per run folder, not shared (F55, corrected); IFBench needs more than 1 h 30 (F69) |
| 2026-11-07 | Keep the dropped `top_k`, the 64-token continuations with the empty think block and DSPy's straggler resubmission, and report them (B) | The authors ran the same code (F58, F71) |
| 2026-11-07 | One optimizer run per job with a full-length limit, MIPROv2 alone on its server, a benchmark's six GEPA-family runs packed on one server (B, proposal) | No resume (F80); MIPROv2's proposer starves behind another pool (F82); GEPA is latency-bound (F3) |
| 2026-11-07 | Keep the artifact's effective thread counts (MIPROv2 8, GEPA cpu_count) and the AIME budget of 1839 (B, proposal) | The authors ran with them (F76, F75); speed only |
| 2026-11-07 | Keep upstream's F60 fix (B, A to confirm) | It has no effect under `track_scores_on='val'` (F81) |
| 2026-11-07 | GRPO: log analysis + signal measurement + one limited HoVer run instead of the full row (B, proposal for the supervisor) | The shipped logs already explain the small gains (F83); a full row costs ~13k SBU per benchmark |
| 2026-11-07 | Split the optimizer runs by optimizer: A runs MIPROv2-Heavy on all five benchmarks, B runs GEPA and GEPA+Merge on all five; runs are handed over by copying run and cache folders (§2.5) | B's choice; the chain only needs the source run's folders in the same directory |

---

## 8. Team workflow and conventions

- **Branches:** `main` stays equal to upstream. `reproduction` is the shared branch for the plan, the tooling and the results. Code goes through short `feat/...` or `fix/...` branches and a PR into `reproduction`, reviewed by the other person. Results (§1, §4, §5) go straight to `reproduction` after `git pull --rebase`, as `docs(results): ...`.
- **Findings:** add new ones on `reproduction` after pulling, with the next free number. A finding written on a long-lived branch is renumbered before it's merged (F54–F56 collided twice).
- **Ownership:** A runs HotpotQA and HoVer, B runs IFBench, AIME-2025 and LiveBench-Math, each in their own experiment directory (F20). Each of us edits only their own cells in §1 and adds their own rows at the bottom of §5.
- **Same environment:** uv 0.13.7, `uv sync --locked`, and the Arbor environment from `envs/arbor-vllm091-requirements.txt` (§3). Every §5 row records the commit (`git rev-parse --short HEAD`), the vLLM version and the requests that reached the server.
- **Budget:** both draw from `gpuuva090`. Run `scripts/snellius/sbu_guard.sh` before submitting and log each job's SBU in §5.
- **Raw data:** run directories stay in each person's home. For the final tables and figures, B packs their `experiment_runs/` into a tarball under `/scratch-shared/<user>/`, readable by group `gpuuva090_access` (everyone on the course account), and A unpacks it next to their own runs.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `chore`, `build`, ...). Upstream defaults stay unchanged: every addition is opt-in. Secrets live outside the repo (`~/.config/gepa/secrets.env`).

---

## 9. Analysis of the reference runs (B, 2026-11-04)

B ran this on a laptop, on the extracted reference data. The scripts are in `analysis/` (merged with PR #1) and take `--runs-dir`, so they also work on our runs.

| What | Result | Used in |
|---|---|---|
| Scripts in `analysis/` | Tables, budget curves, search trees, prompt sizes, generalization gap | the write-up |
| Target numbers vs shipped runs | 46 of 58 cells match, 1 differs (GPT LiveBench-Math GEPA), 11 have no run | F34, §1 |
| Thinking mode | On (cached `reasoning_content`) | F25 |
| Package versions of the original runs | Python 3.12.9, dspy 2.6.23, litellm 1.74.0.post1, openai 1.93.2, pydantic 2.11.7, ujson 5.10.0, cloudpickle 3.1.1, datasets 3.6.0 | F7, F49 |
| Run times | Qwen: GEPA 1.2–6.4 h and MIPROv2 1.1–4.7 h on 4 GPUs; GRPO 12–24 h on 3 GPUs. GPT runs 1–4 h (AIME GEPA 9 h) | F37, §2.2 |
| GPT-4.1-mini cost | All shipped GPT runs ≈ $296 by the artifact's own accounting (≈ €260, more with VAT); one GEPA run $6–25 | out of scope (Table 2) |
| Budget overshoot | GEPA-family runs use 2–23% more metric calls than `max_metric_calls` | F36 |
| Reflection calls | The appendix table matches only the GEPA+Merge runs (plain GEPA on AIME: 100 calls, not 24) | F35 |
| Search trees | ~19 of ~100 proposals kept; the parent isn't the current best in 64–87% of iterations; 8 of 10 final prompts descend from such a pick; per-example best adds 7–27 points | F39 |
| Seed noise | 4.7-point swing between two runs of the same method | F27 |
