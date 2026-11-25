# Reproducing GEPA's Tables 1 and 3 with Qwen3-8B: results and findings

Reproduction of *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*
([arXiv 2507.19457](https://arxiv.org/abs/2507.19457), v2) with the authors' artifact, on Snellius.
Status of 2026-11-25: all runs are finished and analysed (Table 1 on 2026-11-16, Table 3 on 2026-11-22, the two
extensions of §12 on 2026-11-25), and no job is queued. Each number points to its source: a section (§) or finding (F) of `REPRODUCTION_PLAN.md`, or a file under
`results/`.

## 1. Summary

- **What we ran.** Table 1 for Qwen3-8B: HotpotQA, HoVer, IFBench, AIME-2025 and LiveBench-Math, each with Baseline,
  MIPROv2, GEPA and GEPA+Merge, with three seeds per cell (60 runs). We used the authors' code, settings and budgets,
  with one H100 per job. PUPA is an extension, with gpt-oss-120b in place of GPT-4.1-mini, reported as its own block
  (12 runs). Table 3 adds the two candidate-selection rules, SelectBestCandidate and BeamSearch, with three seeds on
  HotpotQA, IFBench and HoVer and one on PUPA (20 runs). For GRPO we analysed the authors' four released runs and did
  no training runs. Compute: 73,476 SBU for B's jobs on the course account, and 18,864 SBU for Table 3 on B's own
  account.
- **Cells.** Under the match rule fixed before any optimizer result (2026-11-07), all 18 targets are consistent with
  the paper: no gap survives Holm's correction. This is weak evidence of a match: with three seeds only gaps of about
  9–14 points could be detected (80 % power), and the 95 % intervals of the optimizer targets are about ±10 points
  wide (±18 to ±19 on AIME).
- **Gains over the Baseline** reproduce on HotpotQA and HoVer (+8.7 to +21.7 points for the three optimizers). On
  IFBench, AIME and LiveBench-Math no gain is detected, as none passes Holm; the paper's GEPA gains there were small
  too (+1.7 to +4.7). The AIME estimates are positive but uncertain (+5.6 to +6.9, every interval includes zero), and
  IFBench MIPROv2 − Baseline (+4.65) has an interval clear of zero but does not pass Holm.
- **GEPA against MIPROv2**, the paper's main comparison, reproduces on HoVer only (+5.22, paper +5.00). On IFBench GEPA
  is below MIPROv2 (−6.29, in all three seeds). Over the five benchmarks the two are even: +0.50 (95 % interval −2.06
  to +2.90), against +6.35 in the paper. The interval covers only the choice of test examples; between seeds the
  difference varies a lot (on AIME −10.00 / +8.67 / +2.67).
- **GEPA+Merge against GEPA** differs on no benchmark. The paper's IFBench drop of −10.38 does not appear (+2.10).
- **Table 3 (candidate selection, §5).** SelectBestCandidate is the authors' configuration. BeamSearch is our own
  implementation, because the authors' code for it was not released.
  - GEPA's Pareto selection is ahead of always taking the best candidate on all three comparable tasks: +2.56 points
    on average (95 % interval +0.92 to +4.18; paper +6.39). No single task passes Holm.
  - GEPA is ahead of BeamSearch only on HotpotQA (+6.00, the one test of eight that passes Holm). Over the three
    tasks the difference is +0.76 (−0.82 to +2.34; paper +6.18).
  - On HoVer our two simpler rules score far above the paper's single runs (50.22 and 51.89 against 45.33 and
    41.00), while our GEPA is close to the paper's.
- **Extensions E1 and E2 (§12): no improvement.** E1 lets GEPA search on 100 of the 300 val examples and race its top
  5 candidates on the other 200. E2 adds a 50-example gate before each val pass. Both run at GEPA's rollout budget, on
  HotpotQA and HoVer with three seeds (12 runs).
  - Neither changes the test score detectably: E1 − GEPA is +0.28 points over the two tasks (95 % interval −1.44 to
    +2.06) and E2 − GEPA +1.22 (−0.61 to +3.06). None of the six tests passes Holm.
  - Both make 2.3 to 2.7 times as many proposals as GEPA and find programs that score higher on the 100 search
    examples. The gain does not carry over to the other 200 val examples or to the test set.
  - They need 1.6 to 1.8 times GEPA's GPU time, so "the same score at lower cost" does not hold either.
- **Seeds matter.** The same method moves 10–19 points between seeds (AIME MIPROv2 44.67 / 26.00 / 32.00,
  LiveBench-Math GEPA+Merge 52.94 / 42.63 / 55.23). Single runs, as in the paper, cannot separate the methods on IFBench,
  AIME and LiveBench-Math.
- **PUPA** (gpt-oss-120b as external model and judge, so not comparable with the paper's numbers). GEPA is above the
  Baseline in every seed (+6.16) and above MIPROv2 on average (+3.97; per seed +2.24 / −0.07 / +9.74), by smaller
  margins than in the paper (+11.03 and +10.30). GEPA+Merge and GEPA do not differ (−0.50, paper −5.59). 25 of the 221
  PUPA test examples are copies of train or val examples in the authors' split.
- **GRPO.** In the released runs only PUPA's GRPO gain holds on test data. The reported HotpotQA and IFBench GRPO test
  scores have no log, because their test runs failed.
- **Reproducibility.** The artifact runs as released only after infrastructure fixes, mainly four raised timeouts, and
  it changes some answers without any error (§7). In 8 of the 60 Table 1 runs requests were lost to server or
  connection failures. 3 of the 20 Table 3 runs each lost one request or one test score. These runs are kept and
  marked † (§3.4, §5.2).

## 2. Setup

- **Code.** The authors' artifact (`gepa-ai/gepa-artifact`), run from our fork (`janbura/gepa-artifac`, branch
  `feat/pilot-caps`) with the authors' DSPy and Arbor forks and vLLM 0.9.1. Our changes are infrastructure only. The
  main ones:
  - four timeouts raised (F67, F70, F72, F85);
  - one GPU per server instead of four with tensor parallelism, because Qwen3-8B fits on one GPU and four cost 4× the
    SBU (F2, F93);
  - a per-job vLLM compile cache (F64) and a per-job retriever cache (F86);
  - IFBench's spaCy model downloaded only when it is missing (F89);
  - no W&B (F66).

  Upstream's post-paper GEPA fix is kept; it cannot act in this configuration (F60, F81).
- **Model and serving.** Qwen3-8B with thinking on, sampled as the artifact does (temperature 0.6, top_p 0.95; F71). The
  context is 32,768 tokens with up to 16,384 output tokens per call, as in the code and the authors' logs. The paper's
  appendix says 16,384 tokens of context (F51, F59).
- **Hardware.** Snellius. All 60 Table 1 runs ran on H100s with one GPU per job (user's decision of 2026-11-09, §2.4).
  A GEPA and GEPA+Merge pair shares one job and its server, and so do the Baselines of one job. The earlier A100 runs
  are a separate hardware check (§3.5).
- **Seeds.** Seeds 0, 1 and 2 per cell. The test set is the same for every seed. For the optimizers, seeds 1 and 2
  also shuffle train and val (F24). The paper reports one run per cell (seed 0).
- **Budgets.** As in the artifact: 6,871 (HotpotQA), 7,051 (HoVer), 3,593 (IFBench) and 1,839 (AIME, LiveBench-Math)
  rollouts for MIPROv2 and GEPA.
- **Deviations per cell (§2.4).**
  - GEPA evaluates with as many threads as the node has CPUs (64 on an H100 node; the authors' pools reached 114–192).
    MIPROv2 uses DSPy's default of 8 threads. Both are what the code does (F76).
  - AIME MIPROv2 has a 45-example val set, below MIPROv2's minibatch threshold. It therefore runs 27 full-val trials,
    about 1,300 metric calls instead of 1,839 (F75).
- **Fixed in advance.** The match rule (2026-11-07), the verdict script (2026-11-08), the hardware check (2026-11-09),
  the rule for lost requests (2026-11-11) and the PUPA design (2026-11-12) were written down before the results they
  apply to. The exception is the Baseline part of the match rule, which was set with our Baseline results known
  (§3.1). The two paired analyses (PUPA 2026-11-15, Table 1 2026-11-16) were written when the run scores, and so the
  mean differences, were known, but before any interval or p-value was computed. The Table 3 method, budget rule and
  analysis script (2026-11-20) were fixed before any Table 3 run. Dates and deviations are in §10.

## 3. Table 1 (Qwen3-8B)

Test score in %: mean ± SD over three seeds, with the paper's single run in brackets. † marks a cell that contains
runs with lost requests (§3.4).

| Method | HotpotQA | HoVer | IFBench | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|
| Baseline | 38.78 ± 0.69 [42.33] | 37.11 ± 1.58 [35.33] | 34.63 ± 0.52 [36.90] | 27.78 ± 4.24 [27.33] | 51.48 ± 0.41 [48.70] |
| MIPROv2 | 58.33 ± 2.00 [55.33] | 45.78 ± 3.21 [47.33] | 39.29 ± 3.26 [36.22] | 34.22 ± 9.53 [20.00] | 48.90 ± 3.08 [46.60] |
| GEPA | 60.44 ± 2.72 [62.33] | 51.00 ± 2.33† [52.33] | 32.99 ± 1.48† [38.61] | 34.67 ± 0.00 [32.00] | 49.92 ± 0.91 [51.95] |
| GEPA+Merge | 57.56 ± 2.88† [64.33] | 50.33 ± 1.45† [51.67] | 35.09 ± 2.70† [28.23] | 33.33 ± 7.69 [32.00] | 50.27 ± 6.71 [51.95] |

Per seed (seed 0 / 1 / 2; † = run with lost requests):

| Method | HotpotQA | HoVer | IFBench | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|
| Baseline | 39.33 / 38.00 / 39.00 | 38.33 / 35.33 / 37.67 | 35.20 / 34.18 / 34.52 | 25.33 / 25.33 / 32.67 | 51.31 / 51.95 / 51.18 |
| MIPROv2 | 56.33 / 60.33 / 58.33 | 42.33 / 48.67 / 46.33 | 35.54 / 41.50 / 40.82 | 44.67 / 26.00 / 32.00 | 48.48 / 52.16 / 46.05 |
| GEPA | 62.33 / 57.33 / 61.67 | 48.67† / 51.00† / 53.33 | 33.67 / 31.29 / 34.01† | 34.67 / 34.67 / 34.67 | 50.91 / 49.11 / 49.73 |
| GEPA+Merge | 60.67 / 55.00 / 57.00† | 48.67† / 51.00† / 51.33† | 36.56 / 36.73 / 31.97† | 24.67 / 36.00 / 39.33 | 52.94 / 42.63 / 55.23 |

- **Failed examples count as wrong**, as in the paper. Most are format failures; on the math benchmarks, mostly
  unfinished thinking at the 16,384-token limit (F84). The number per run is in §1 of the plan.
- **AIME's 150 test examples are 30 problems asked 5 times.** Identical requests share DSPy's cache, so AIME is about
  50 independent answers, and one problem moves the score by 3.33 points (F53, §2.2).
- **The paper prints the same GEPA and GEPA+Merge score on both math benchmarks** (32.00 and 51.95). On one-predictor
  programs Merge cannot act, so the two are runs of the same search (F61). The verdict compares the mean of our six
  runs with that number.
- **A's HotpotQA and HoVer Baselines**, run separately by A, give 41.22 ± 2.22 and 36.89 ± 1.17. They are not in the
  table. The HotpotQA runs had up to 5 worker timeouts each (§2.2), which can mix answers (F72), so only the HoVer
  value is clean.

### 3.1 Verdict against the paper (fixed before the data)

Rule (§2.4, fixed 2026-11-07; script `analysis/table1_verdict.py` of 2026-11-08):
- For each target, d = our mean − the paper's value.
- Optimizer cells use an SD that includes a search part of 3.22 points (or our own pooled SD, if larger), with a
  t-test on 7 df. Baselines use the sampling SD.
- Holm over the 18 targets at 0.05. The math GEPA and GEPA+Merge cells form one target per benchmark.

Result (2026-11-16, `results/table1_h100/verdict_output.md`): **all 18 targets are consistent with the paper**, and
Holm rejects none.
- The smallest p-value is 0.038 (HotpotQA Baseline, d −3.55), above Holm's first threshold of 0.0028.
- The largest gaps are AIME MIPROv2 +14.22 (95 % interval −5.08 to +33.52), IFBench GEPA+Merge +6.86, HotpotQA
  GEPA+Merge −6.77 and IFBench GEPA −5.62.
- "Consistent" means no evidence of a gap, not equality. The intervals of the optimizer targets are ±9.6 to ±11.7
  points (±18 to ±19 on AIME), and only gaps of about 9–14 points could be detected with 80 % power (§2.4).
- The Baseline verdicts are descriptive, because the Baseline part of the rule was set with our Baseline results known.

### 3.2 Method comparisons (paired tests)

Method (§2.13, written and pushed before the run):
- Per test example (AIME: per problem), each method's score is the mean over its three seeds.
- d is the mean difference in points, with a 95 % bootstrap interval over examples and a sign-flip test.
- Holm over the 28 comparisons.
- The interval covers which test examples happened to be drawn, not the run-to-run variation of the optimizers.

Each cell gives d with the paper's difference in parentheses. **Bold** = Holm rejects at 0.05. * = on the math
benchmarks GEPA+Merge and GEPA are the same search, so this row is not tested. The last column is the mean over the
five benchmarks with its 95 % interval (the paper's value without PUPA in parentheses).

| A − B | HotpotQA | HoVer | IFBench | AIME-2025 | LiveBench-Math | Mean of the five |
|---|---|---|---|---|---|---|
| GEPA − Baseline | **+21.67** (+20.00) | **+13.89** (+17.00) | −1.64 (+1.71) | +6.89 (+4.67) | −1.56 (+3.25) | +7.85 [+5.70, +10.10] (+9.33) |
| GEPA − MIPROv2 | +2.11 (+7.00) | **+5.22** (+5.00) | **−6.29** (+2.39) | +0.44 (+12.00) | +1.02 (+5.35) | +0.50 [−2.06, +2.90] (+6.35) |
| GEPA+Merge − GEPA | −2.89 (+2.00) | −0.67 (−0.66) | +2.10 (−10.38) | −1.33* (0.00) | +0.35* (0.00) | −0.49 [−3.06, +1.87] (−1.81) |
| GEPA+Merge − Baseline | **+18.78** (+22.00) | **+13.22** (+16.34) | +0.45 (−8.67) | +5.56 (+4.67) | −1.21 (+3.25) | +7.36 [+5.19, +9.53] (+7.52) |
| GEPA+Merge − MIPROv2 | −0.78 (+9.00) | **+4.56** (+4.34) | −4.20 (−7.99) | −0.89 (+12.00) | +1.37 (+5.35) | +0.01 [−2.42, +2.48] (+4.54) |
| MIPROv2 − Baseline | **+19.56** (+13.00) | **+8.67** (+12.00) | +4.65 (−0.68) | +6.44 (−7.33) | −2.58 (−2.10) | +7.35 [+5.19, +9.65] (+2.98) |

- **Holm rejects 9 of the 28 tests:**
  - the three optimizers against the Baseline on HotpotQA and HoVer;
  - GEPA and GEPA+Merge against MIPROv2 on HoVer;
  - GEPA against MIPROv2 on IFBench, where the difference is negative.
- **Two more intervals exclude zero but do not pass Holm:** IFBench GEPA+Merge − MIPROv2 (−4.20) and IFBench
  MIPROv2 − Baseline (+4.65).
- **Why the cells agree with the paper but the comparisons do not.** Our MIPROv2 scores above the paper's on four of the
  five benchmarks, and our GEPA scores below it on four. Each gap is within the noise of its cell, but together they
  remove GEPA's lead.
- **Without the † runs** (descriptive), no difference changes sign. IFBench GEPA+Merge − GEPA becomes +4.17 [+0.17,
  +8.25], and IFBench GEPA+Merge − MIPROv2 becomes −2.64 [−6.07, +0.85] (§2.13).
- **Full table:** with p-values and per-seed differences in `results/table1_h100/paired_output.md`.

### 3.3 Run-to-run variation

- **One run per cell is not enough on these benchmarks.** Per seed, GEPA − MIPROv2 is +6.00 / −3.00 / +3.34 on HotpotQA
  and −10.00 / +8.67 / +2.67 on AIME (§2.13).
- **Sampling noise.** Measured from per-example agreement between reruns, the noise of a single run is about 1.5 points
  on HotpotQA, 1.6 on HoVer and IFBench, 2.7 on LiveBench-Math and 4.1 on AIME (§2.4, F92). The optimizers add a search
  part on top (3.22 points, estimated from the authors' GEPA and GEPA+Merge pairs).
- **AIME seed-0 pause.** AIME MIPROv2 seed 0 (44.67, 24.67 points above the paper) paused AIME under the gate fixed in
  advance (F97). The setup check found a technically clean run with the shipped settings, so AIME continued. Seeds 1
  and 2 gave 26.00 and 32.00.

### 3.4 Lost requests (†)

**HotpotQA and HoVer (F104, F105).**
- In 6 of the 12 HotpotQA and HoVer runs of GEPA and GEPA+Merge, 1–3 of about 17,000 model requests per run (10 in
  total) failed with a client-side connection error. As in the original code, the client does not retry.
- Each lost request fell in the full validation evaluation of a newly accepted candidate. It left one of that
  candidate's 300 validation examples unscored, so its validation score could be up to 0.33 points too low.
- None of these candidates came within 3 points of the final program, and no request was lost in a test evaluation.
- A replay of the logged search shows that 9 of the 10 losses could have altered later candidate selections or merges.
  So an effect on the search path cannot be ruled out; its size and sign are unknown
  (`results/checks/LOST_REQUESTS_2026-11-13.md`).

**IFBench seed 2 (F106, §2.12).**
- In the GEPA pair of seed 2, 47 requests were lost: 46 to two bursts of server batch failures (F68) and 1 to a
  connection error. 13 fell in the GEPA search and 34 in the GEPA+Merge test evaluation.
- 34 of its 294 test examples therefore count as wrong, and its 31.97 can be up to 11.56 points too low. This is the
  largest loss of the study, and the only one in the test evaluation of a run that counts (the void IFBench MIPROv2
  run of F102 also lost 2 test requests).
- A first re-score of the 34 lost 32 of them again, in the same way. The second and last attempt, one request at a
  time, answered 32 of the 34 and gives 36.39 instead of 31.97.
- Without the failure, GEPA+Merge is above GEPA on IFBench in all three seeds. Table 1 keeps 31.97†.

**The rule.** Following a rule fixed on 2026-11-11, before any run with lost requests had a score, these runs are kept
and marked †. Lost requests per run (search / test):

| Run | Lost (search / test) |
|---|---|
| HotpotQA GEPA+Merge s2 | 1 / 0 |
| HoVer GEPA s0, s1 | 2 / 0, 3 / 0 |
| HoVer GEPA+Merge s0, s1, s2 | 1 / 0, 1 / 0, 2 / 0 |
| IFBench GEPA s2 | 13 / 0 |
| IFBench GEPA+Merge s2 | 0 / 34 |

Before this rule, two IFBench MIPROv2 seed-2 runs were voided for lost requests. Both were cancelled before they had a
score (F99, F102), so no result was discarded.

**Cells without the † runs** (descriptive):

| Cell | Mean ± SD (seeds left) |
|---|---|
| HotpotQA GEPA+Merge | 57.835 ± 4.01 (2) |
| HoVer GEPA | 53.33 (1) |
| HoVer GEPA+Merge | none left (all three seeds affected) |
| IFBench GEPA | 32.48 ± 1.68 (2) |
| IFBench GEPA+Merge | 36.645 ± 0.12 (2) |

No paired comparison changes sign without them (§2.13).

### 3.5 Hardware check

The 15 Baselines were also run on A100s, and we checked them with a method fixed on 2026-11-09 (F101, F90):
- H100 − A100 difference of the Baseline means: −2.45 (HotpotQA), −0.89 (HoVer), −0.11 (IFBench), +1.98
  (LiveBench-Math) and −2.89 (AIME). Holm over the five rejects none.
- Per example, runs on different GPUs agree as often as two runs on the same GPU.
- HotpotQA and HoVer MIPROv2 on A100 lie within the range of the three H100 seeds (F103).

So the check found no GPU effect. One run per GPU type cannot exclude a shift of a few points (F90), but nothing
points to one.

## 4. PUPA with gpt-oss-120b (extension)

PUPA needs GPT-4.1-mini twice: as the untrusted external model inside PAPILLON and as the LLM judge of the metric (F14).
We have no OpenAI key, so both roles use gpt-oss-120b. It runs with vLLM 0.11.0 on a second H100, with reasoning effort
low: fixed for the external model, and chosen for the judge by a calibration on the authors' val outputs. Qwen3-8B runs as in Table 1.

The judge defines the score, so these numbers are not comparable with the paper's PUPA numbers (Baseline 80.82, MIPROv2
81.55, GEPA 91.85, GEPA+Merge 86.26). They form their own block (§2.10, `results/pupa_gptoss/`).

| Method | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| Baseline | 85.94 | 85.93 | 86.79 | 86.22 ± 0.49 |
| MIPROv2 | 90.62‡ | 90.80 | 83.79 | 88.40 ± 4.00 (without ‡: 87.30 ± 4.96) |
| GEPA | 92.86 | 90.73 | 93.53 | 92.37 ± 1.46 |
| GEPA+Merge | 91.41 | 94.37 | 89.83 | 91.87 ± 2.30 |

‡ = kept although gpt-oss gave empty answers in 1.8 % of the logged predictions, above the 1 % check limit. Every
request was answered.

Paired per-example differences (221 test examples, each method the mean over its seeds, 95 % bootstrap interval). The
paper's differences are shown for direction only, because the judge differs.

| A − B | d | 95 % interval | without ‡ | paper |
|---|---|---|---|---|
| GEPA − Baseline | +6.16 | +3.90 to +8.45 | | +11.03 |
| GEPA − MIPROv2 | +3.97 | +2.16 to +5.79 | +4.83 (+2.39 to +7.34) | +10.30 |
| GEPA+Merge − GEPA | −0.50 | −2.19 to +1.18 | | −5.59 |
| GEPA+Merge − Baseline | +5.65 | +3.38 to +7.95 | | +5.44 |
| GEPA+Merge − MIPROv2 | +3.47 | +1.55 to +5.42 | +4.80 (+2.47 to +7.19) | +4.71 |
| MIPROv2 − Baseline | +2.19 | +0.29 to +4.08 | +0.94 (−1.50 to +3.37) | +0.73 |

- **GEPA is above both the Baseline and MIPROv2 on PUPA**, unlike on most Table 1 benchmarks, by smaller margins than
  in the paper. Of the paper's three PUPA claims, only GEPA − Baseline has the same sign in every seed; GEPA − MIPROv2
  per seed is +2.24 / −0.07 / +9.74. The intervals cover the test examples, not the run-to-run variation of the
  optimizers, which is large for MIPROv2 (SD 4.00), and the method fixed in advance has no test across seeds (n = 3).
  GEPA+Merge and GEPA do not differ (paper −5.59).
- **Split overlap (F108).** The loader takes the first 443 rows of `Columbia-NLP/PUPA` (`pupa_new`) in order.
  - 25 of the 221 test examples are exact copies of train or val examples.
  - 11 repeat an earlier test example.
  - 43 share their user query with a train or val example.

  This is the authors' split, so the paper's PUPA numbers have it too. Without the 25 copies, or without the 43
  query overlaps, no paired difference moves by more than 0.71 points. The five Table 1 benchmarks have no such copies.
- **Deviation.** The per-example rebuild check that was fixed before the analysis (2026-11-15 20:45) failed in 7 of the
  12 runs, because of late optimizer answers in the test block. The rebuilt scores still matched all reported scores.
  The paired analysis ran under a changed check, written down before it was computed and approved by the user (§2.10,
  deviation of 2026-11-15 22:08).
- **Cost:** 9,492 SBU.

## 5. Table 3: candidate selection (Qwen3-8B)

Table 3 of the paper tests how GEPA picks the candidate to improve next. GEPA picks it from the Pareto front, and the
table compares this with two simpler rules:
- SelectBestCandidate always takes the best candidate;
- BeamSearch keeps a pool of the top 4.

Observation 3 says the Pareto rule wins. Its improvement over the Baseline (the mean over four tasks) is +12.44, against
+6.04 and +5.11 for the two rules (recomputed from the paper's cells). The paper has one run per cell. Our method,
budget rule and analysis script were fixed on 2026-11-20, before any Table 3 run (§2.14).

- **Runs.** 20 of the 24 planned runs: SelectBestCandidate and BeamSearch with seeds 0, 1 and 2 on HotpotQA, IFBench
  and HoVer, and seed 1 on PUPA.
  - Each run uses the budget, DSPy cache, seed and code of the GEPA run of the same task and seed.
  - The two runs of one task and seed share one server, as the GEPA pairs did (F94).
  - Baseline and GEPA are the runs of §3 and §4; they were not rerun.
  - The jobs ran on B's own Snellius account, with the same software, model files and H100 nodes (§2.14, addendum of
    2026-11-20 19:04).
- **SelectBestCandidate** is the authors' configuration.
  - Their four released seed-0 runs give the paper's row exactly (F109).
  - The released run script cannot start it, so we use the condition of the authors' replay script (F110).
- **BeamSearch is not in the released code** (F109). The paper defines it in one sentence: a pool of the top-N
  candidates, as in APO, with N = 4.
  - Our reading: the pool is the 4 candidates with the best validation score, and each iteration improves the pool
    member picked least often so far.
  - This row is therefore a replication with our own code, not a reproduction of the authors' code.
- **PUPA** uses gpt-oss-120b as external model and judge (§4). So its column, and every aggregate that includes it, is
  not comparable with the paper.
  - PUPA seed 0 failed twice, because the gpt-oss server stopped answering (F112).
  - After the second failure the user stopped PUPA at seed 1, so PUPA SelectBestCandidate and BeamSearch have one run
    each.

Test score in %: mean ± SD over the seeds, with the paper's single run in brackets. † marks a cell that contains a run
that lost one request or one test score (§5.2).

| Method | HotpotQA | IFBench | HoVer | PUPA (gpt-oss judge) |
|---|---|---|---|---|
| Baseline | 38.78 ± 0.69 [42.33] | 34.63 ± 0.52 [36.90] | 37.11 ± 1.58 [35.33] | 86.22 ± 0.49 [80.82] |
| SelectBestCandidate | 56.78 ± 1.57† [58.33] | 29.76 ± 1.70† [30.44] | 50.22 ± 3.17† [45.33] | 92.64, one run [85.45] |
| BeamSearch | 54.44 ± 6.95 [57.33] | 35.83 ± 2.75 [36.39] | 51.89 ± 1.58 [41.00] | 93.41, one run [81.08] |
| GEPA | 60.44 ± 2.72 [62.33] | 32.99 ± 1.48† [38.61] | 51.00 ± 2.33† [52.33] | 92.37 ± 1.46 [91.85] |

Improvement over the Baseline, as in the paper: the mean of the task means minus the Baseline's. The paper's values,
in brackets, are recomputed from its rounded cells, so they may differ from its printed values by 0.01.

| Method | HotpotQA, IFBench, HoVer | All four (PUPA not comparable) |
|---|---|---|
| SelectBestCandidate | +8.75 [+6.51] | +8.16 [+6.04] |
| BeamSearch | +10.55 [+6.72] | +9.71 [+5.11] |
| GEPA | +11.30 [+12.90] | +10.02 [+12.44] |

Per seed (seed 0 / 1 / 2). Baseline and GEPA per seed are in §3 and §4.

| Method | HotpotQA | IFBench | HoVer | PUPA |
|---|---|---|---|---|
| SelectBestCandidate | 57.33† / 55.00 / 58.00 | 29.76† / 31.46 / 28.06 | 50.33† / 47.00 / 53.33 | – / 92.64 / – |
| BeamSearch | 53.00 / 48.33 / 62.00 | 35.37 / 33.33 / 38.78 | 51.33 / 53.67 / 50.67 | – / 93.41 / – |

### 5.1 Observation 3 (paired tests)

Method as in §3.2, fixed in §2.14:
- Per test example, each method's score is the mean over its seeds.
- d is the mean difference in points, with a 95 % bootstrap interval over examples and a sign-flip test.
- Holm over the 8 tests: GEPA − SelectBestCandidate and GEPA − BeamSearch on each of the four tasks.
- The interval covers which test examples were drawn, not the run-to-run variation.

Each cell gives d with the paper's difference in parentheses. **Bold** = Holm rejects at 0.05. The last column is the
mean over HotpotQA, IFBench and HoVer, with its 95 % interval.

| A − B | HotpotQA | IFBench | HoVer | PUPA (not comparable) | Mean of the three |
|---|---|---|---|---|---|
| GEPA − SelectBestCandidate | +3.67 (+4.00) | +3.23 (+8.17) | +0.78 (+7.00) | −0.26 (+6.40) | +2.56 [+0.92, +4.18] (+6.39) |
| GEPA − BeamSearch | **+6.00** (+5.00) | −2.83 (+2.22) | −0.89 (+11.33) | −1.04 (+10.77) | +0.76 [−0.82, +2.34] (+6.18) |
| SelectBestCandidate − BeamSearch (not tested) | +2.33 (+1.00) | −6.07 (−5.95) | −1.67 (+4.33) | −0.77 (+4.37) | −1.80 [−3.56, −0.06] (−0.21) |

- **Pareto selection against the best candidate.** GEPA is ahead on all three comparable tasks, and the mean over them
  has an interval clear of zero (+2.56, from +0.92 to +4.18).
  - This is the direction of Observation 3, at less than half the paper's margin (+6.39).
  - No single task passes Holm. HotpotQA (+3.67, +0.67 to +6.56) and IFBench (+3.23, +0.23 to +6.24) have intervals
    clear of zero, with p 0.019 and 0.038 before correction. HoVer (+0.78) does not.
- **Pareto selection against BeamSearch.** GEPA is ahead only on HotpotQA (+6.00, +3.11 to +8.78; paper +5.00). This is
  the one test that passes Holm (adjusted p 0.0008).
  - On IFBench, BeamSearch is above GEPA in all three seeds (−2.83, from −5.84 to +0.23).
  - On HoVer no difference is detected (−0.89, from −3.33 to +1.44).
  - In the paper GEPA is ahead on these two tasks by 2.22 and 11.33. Over the three tasks we get +0.76 (−0.82 to
    +2.34), against +6.18 in the paper.
- **HoVer is the largest departure.** Our SelectBestCandidate and BeamSearch score 50.22 and 51.89, far above the
  paper's single runs (45.33 and 41.00). All six of our HoVer runs of these two rules score 47.00 or more. Our HoVer
  GEPA (51.00) is close to the paper's (52.33). So the paper's large HoVer gaps (+7.00 and +11.33) do not appear in
  our runs: both lie outside our intervals.
- **Seeds.** Per seed, 4 of these 6 comparisons change sign. For example, HotpotQA GEPA − BeamSearch is +9.33 / +9.00 /
  −0.33, and HoVer GEPA − SelectBestCandidate is −1.66 / +4.00 / 0.00. BeamSearch on HotpotQA ranges from 48.33 to
  62.00. A single run per cell, as in the paper, cannot order these rules.
- **Our seed 0 against the authors' released seed-0 runs** (SelectBestCandidate, same configuration): HotpotQA
  57.33 / 58.33, IFBench 29.76 / 30.44, HoVer 50.33 / 45.33.
- **How the rules search** (run logs, summed over the seeds).
  - SelectBestCandidate took the best candidate so far as the parent in all 1,099 of its iterations, as its rule says.
  - BeamSearch took a parent below the best in 59–75 % of the iterations, GEPA in 68–78 %.
  - All three rules accepted about the same number of new candidates (for example 60–62 on HotpotQA over three seeds).
  - So our BeamSearch explores almost as much as GEPA, which may be why it scores about as well. This was not tested.
- **Without the † runs** (descriptive). GEPA − SelectBestCandidate stays positive on every task: HotpotQA +3.94, IFBench
  +2.72, HoVer +3.17. HoVer GEPA − BeamSearch turns positive, +1.44 (−2.11 to +4.89), because only GEPA seed 2 is
  left.
- **PUPA** (gpt-oss judge, one run each for SelectBestCandidate and BeamSearch). They score 92.64 and 93.41, against
  GEPA's 92.37 ± 1.46. No difference is detected: −0.26 and −1.04, and both intervals include zero.
- **Full output:** `results/table3/paired_output.md`, with the p-values, per-seed differences and search statistics.

### 5.2 Lost requests, void runs and checks

- **† runs** (rule of 2026-11-11). Three of the 20 runs lost one request or test score each, all of them
  SelectBestCandidate seed 0:
  - HotpotQA: one validation request in the search was lost to a client connection error (as F104). It could not have
    changed the search: the candidate would have scored at most 56.67, below the best at the time (57.00), and this
    rule only takes the best.
  - HoVer: one test request was lost to a client connection error. The score can be up to 0.33 points too low.
  - IFBench: one test example scored 0, because NLTK's lazy loader is not thread-safe (F113). The score can be up to
    0.34 points too low.
- **Counted without a mark.** In IFBench BeamSearch seed 2, the authors' `AlphabetLoopChecker` raised an error on one
  test example (F114). That example scored 0, as it would in the authors' code. The effect is at most 0.34 points.
- **Void.** Both PUPA seed-0 jobs: the gpt-oss server stopped answering, after about 4 h and after 3 h 40 min (F112).
  - The first time, it logged 21 internal panics before it stopped. The second time it logged no error at all.
  - **Cause (found afterwards, F112).** Qwen sometimes repeats `-` in a markdown table up to its token limit, which
    gives about 1,000,000 dashes. gpt-oss's tokenizer (openai-harmony 0.0.8) fails at once on runs above about
    1,000,000 characters; those are the panics. On runs just under that size it needs about 8 minutes (999,000
    dashes on a laptop CPU), and meanwhile the server answers nothing. Our watchdog then stopped the job after about
    4 minutes, while gpt-oss was still tokenizing. The seed-1 job wrote no such text.
  - After the second failure the user stopped PUPA at seed 1 (2026-11-22). PUPA seed 2 was not run.
- **Before any result.**
  - Two smoke tests failed: one at the server start, because Arbor waits a fixed 300 s for vLLM (F111), and one
    because the released run script cannot start SelectBestCandidate (F110).
  - Two seed-0 jobs failed at the server start because of the same 300 s wait. They were rerun unchanged, except for
    a longer start-up wait (F111).
  - A third smoke test passed.
- **Checks.** In the 10 jobs with results, the servers answered every request they received. There were two
  exceptions, both allowed and reported:
  - one IFBench reflection prompt was over the length limit and was rejected (§2.4);
  - HTTP 500 batch failures were retried successfully (F68).

  There was no worker timeout. Every HotpotQA and HoVer run started at the expected starting score of its seed.
- **Cost:** 18,864 SBU (sacct), under a cap of 21,000 SBU. The cap was 17,000 at first. The user raised it to 20,000
  before any job, and to 21,000 after the first PUPA failure. 3,084 SBU went to the two void PUPA jobs.

## 6. GRPO (analysis of the authors' runs)

We ran no GRPO training: a faithful run needs 24,000 rollouts per benchmark. At the authors' speed (12.0–24.1 h on 3
GPUs per run) that is about 6,900–13,900 SBU per run on H100s, about 43,000 SBU for one seed of the four benchmarks with
logs, and about 129,000 for three seeds (estimate, `analysis/grpo/log_checks/cost_estimate_out.txt`; the plan's earlier
"about 13k per benchmark" is the upper end). Instead we analysed the four GRPO runs released with the artifact
(HotpotQA, HoVer, IFBench, PUPA). We used two independent sets of scripts and statistical tests (F107,
`analysis/grpo/GRPO_analysis.pdf`).

What the runs show:
- **PUPA:** +5.84 test points over the Baseline (paired bootstrap 95 % CI +2.27 to +9.52), and +6.59 on the 196 test
  examples that are not copies of train or val examples (F108).
- **HoVer:** +3.33, not significant (McNemar p = 0.19). Validation shows no trend, and the chosen checkpoint is what
  the best of 25 validations of an unchanged model gives (p = 0.45).
- **HotpotQA:** the train reward rises (45 → 54 %) while validation stays flat.
- **IFBench:** validation rises 4.0 points (p < 0.001), but the reported test score is 1.0 below the Baseline.
- **Missing test logs.** The test runs of HotpotQA and IFBench failed (0/300 and 0/294; the server with the trained
  model did not start), so the reported 43.33 and 35.88 appear only in the result files. AIME and LiveBench-Math have
  no released GRPO logs.

Possible causes in the code (not tested):
- training on text rebuilt from the parsed fields, with an empty think block, instead of the sampled text;
- 56–81 % of the 12-rollout groups with identical rewards, which give no signal;
- one program-level reward for every module;
- an 8,192-token context in training against 32,768 at test time;
- a checkpoint chosen as the best of 26 validations, not by early stopping as the paper says.

## 7. Reproducibility findings

### 7.1 Where the paper and the artifact differ

- **Context.** The paper says 16,384 tokens of context. The code and the authors' logs use a 32,768-token context with
  up to 16,384 output tokens (F51, F59). We follow the code.
- **Math GEPA cells.** The paper's four math cells show the same score for GEPA and GEPA+Merge. Where a shipped run
  exists (GPT-4.1-mini LiveBench-Math), the two runs differ (59.43 and 64.13), and the paper printed the GEPA+Merge
  score in both cells (F61, F34).
- **Shipped runs.** Of the 58 paper numbers we checked against the shipped runs, 46 match, 1 differs (GPT-4.1-mini
  LiveBench-Math GEPA) and 11 have no run. There are no shipped Qwen3-8B runs for AIME and LiveBench-Math (F34).
- **Reflection calls.** The appendix's counts match only the GEPA+Merge runs. For example, the shipped plain GEPA run on
  AIME (GPT-4.1-mini) made 100 reflection calls where the appendix says 24 (F35).
- **Budget overshoot.** GEPA-family runs use 2–23 % more metric calls than their budget, because the last validation
  pass is not cut off (F36, F78).
- **Missing methods.** The artifact has no Trace, TextGrad or MIPROv2-No-Demos (Table 2), no BeamSearch (Table 3) and
  no kernel experiments (Figures 7–8) (F30). For Table 3 we wrote our own BeamSearch from the paper's one-sentence
  definition (F109, §5).
- **Table 3's ablation.** The released run script cannot run SelectBestCandidate: it gives GEPA's arguments only to
  optimizer names that contain "GEPA". The authors' replay script uses a condition that includes SelectBestCandidate,
  so their runs did not use the released one (F110). The authors' four released SelectBestCandidate runs give the
  paper's row exactly (F109).
- **GRPO.** Of the paper's six Qwen GRPO test scores, only HoVer's and PUPA's have a test log: the HotpotQA and IFBench
  test runs failed, and AIME and LiveBench-Math have no released GRPO runs. The checkpoint is chosen by the best
  validation score (F107).
- **PUPA.** Its test set overlaps its train and val sets (F108).
- **Upstream change.** GEPA was changed after the paper's runs (commit `29220ba`, a fix of shared validation scores).
  In this configuration the fix has no effect (F60, F81).

### 7.2 Behaviours that change answers without an error

We kept these, because the authors ran the same code (F71):
- **`top_k` is dropped.** `top_k=20` never reaches vLLM, so sampling uses temperature 0.6 and top_p 0.95 only.
- **Chunked generation.** Arbor generates in 64-token chunks and rebuilds the prompt between them. Each continuation
  puts an empty think block in front of the model's reasoning (F58).
- **Resubmitted stragglers.** DSPy resubmits the last slow examples after 120 s and keeps the copy that finishes first,
  which favours shorter answers at the end of each evaluation.
- **Batch failures.** One bad partial answer makes Arbor fail its whole batch of up to 32 requests (F68). This also
  happens in 11 of the authors' 24 Qwen runs (F100), where the retries recovered every evaluation call. In our IFBench seed-2 runs
  bursts of it lost requests (F99, F102, F106).
- **Worker timeout.** When Arbor's worker timeout fires, answer text goes to the wrong requests (F72). Our raised
  timeouts keep it from firing, and a run with any worker timeout would be void.
- **Cache key.** The client timeout is part of DSPy's cache key, so changing it invalidates every cached answer (F73).
- **Silent failures.** A failed request becomes a wrong answer without stopping the run (F29), and the client does not
  retry connection errors (F104).
- **IFBench metric errors.** The IFBench metric can fail on an example without stopping the run, and the example then
  scores 0. We found two causes in Table 3:
  - NLTK's lazy corpus loader is not thread-safe (F113);
  - the authors' `AlphabetLoopChecker` raises on a response variant made only of punctuation (F114).

### 7.3 What it takes to run the artifact

- **Timeouts.** Arbor's worker timeout (300 s), the client (600 s) and the forwarder (600 s) cut long math and IFBench
  answers and score them 0 without a visible error (F54, F56, F67). Our first math Baselines lost answers this way and
  were redone. We raised all four limits: client and forwarder 21,600 s, worker 864,000 s, chunk call 600 s (F85).
- **Keys.** The code asserts an OpenAI key and a W&B key at start-up, even for Qwen-only runs, and GEPA always logs in
  to W&B (F12, F13). We run with `WANDB_MODE=disabled` and a dummy key, which W&B accepts (F66). PUPA cannot run without OpenAI (F14), and GPT-4.1-mini,
  which Table 2 uses, is reported to retire on 2027-01-14 (F16; not checked against OpenAI's own page).
- **Environment.** There is no lockfile, the fork commits are not pinned, and NLTK and spaCy data are downloaded at run
  time (F7, F46, F48, F49, F89).
- **Shared cache.** HotpotQA and HoVer share one SQLite search cache inside the checkout, which is unsafe for parallel
  jobs on different nodes (F86).
- **No resume.** A killed run starts again from the beginning, replaying its cache (F80).
- **Runs that never end.** A GEPA run whose model calls all fail, or whose reflection keeps failing, never ends (F57,
  F91).
- **Server start.** Arbor gives the vLLM server it launches a fixed 300 s to start. Weight loading plus a fresh
  compile can take longer, for example for two jobs that started together on one node. Then no run starts (F42, F111).
  For later Table 3 jobs we raised the wait to 1,800 s. This bounds only the start-up.
- **gpt-oss server stalls on long repeated text.** In 2 of the 3 PUPA jobs of Table 3, the gpt-oss-120b server (vLLM
  0.11.0) stopped answering, and our watchdog stopped the job (F112). Each stall began with a Qwen answer of about
  1,000,000 dashes, a markdown table separator repeated up to the token limit. gpt-oss's tokenizer (openai-harmony
  0.0.8, the newest release) fails at once on runs above about 1,000,000 characters. On runs just under that size it
  needs a time that grows with the square of the length: about 8 minutes for 999,000 dashes on a laptop CPU. The
  server answers nothing meanwhile (`results/checks/F112_freeze_cause_2026-11-22.md`).
- **MIPROv2 failure modes.** A MIPROv2 run can silently return the Baseline, and one failed proposal call aborts it
  (F77).
- **Speed.** The authors served Qwen3-8B with tensor parallelism over 4 GPUs. On one A100 our runs were about 3.4×
  slower (F93). On one H100 our HotpotQA and HoVer MIPROv2 runs took 8 h 50 to 9 h 56, about twice the authors' 4.7 h
  for HoVer MIPROv2 (§1, F93). An H100 is 4–5× faster than an A100 on long answers and cheaper per run for all our work (F87, F98).

### 7.4 Our own mistakes, and the checks they gave us (§2.3)

- **Answers dropped silently.** A 600 s forwarder limit dropped answers in our first math Baselines while our checks
  called them clean (about 900 SBU). Since then every run is checked for requests = responses and for zero timeouts,
  5xx errors and chunk errors.
- **Counting failed examples.** They are counted as the test size minus the progress bar's denominator, not from error
  lines (F84).
- **Compile cache.** Two vLLM jobs that started in the same second broke each other's compile cache, so each job now
  has its own (F64).
- **Reruns.** Before a rerun, both the run folder and its cache folder are moved, so nothing is replayed (F55).

## 8. Compute

**Total:** 73,476 SBU for B's jobs since 2026-11-04 (sacct of B's login, every job once, queried 2026-11-16). A's own
jobs, about 440 SBU according to the plan header, are not included.

| By partition | SBU |
|---|---|
| H100 | 63,704 |
| A100 | 9,715 |
| A100 slices (MIG) | 33 |
| CPU nodes | 23 |

| By use | SBU |
|---|---|
| PUPA | 9,492 |
| The two IFBench re-scores | 180 |
| Table 1, the A100 hardware check, void runs, pilots and checks | about 63,800 |

On 2026-11-07 we expected 35,000–42,000 SBU (55,000 pessimistic) for the 45 optimizer runs of Table 1 (§2.6). The
total above holds more than those runs:
- the Baselines;
- the A100 runs that were repeated on H100 and became the hardware check (decision of 2026-11-09);
- void runs and their reruns (dropped answers, void IFBench runs), pilots and checks.

The AIME runs also took much longer than expected (F96).

**Table 3:** 18,864 SBU on B's own Snellius account (sacct, queried 2026-11-22; cap 21,000 SBU). The plan expected
about 16,500 SBU.

| Table 3, by use | SBU |
|---|---|
| The 10 jobs with results (20 runs) | 15,463 |
| The two void PUPA seed-0 jobs (F112) | 3,084 |
| Three smoke tests | 268 |
| Two jobs that failed at the server start (F111) | 50 |

The per-job values are rounded, so they sum to 18,865.

**Extensions E1 and E2 (§12):** 13,050 SBU on the course account, login C (sacct of C's login from 2026-11-22, queried
2026-11-25; cap 18,000 SBU). The plan expected 12,300 SBU.

| Extensions, by use | SBU |
|---|---|
| The 6 jobs with results (12 runs) | 12,615 |
| Two smoke tests on one A100 each | 429 |
| Check and monitor jobs on CPU nodes | 6 |

## 9. Limitations

- **Three seeds per cell.** The verdict can only detect gaps of about 9–14 points (§2.4), and three seeds are too few to
  test optimizer differences against their run-to-run variation. The paired tests cover only the choice of test
  examples.
- **Serving.** We used one GPU per server instead of four, and GEPA ran with 64 evaluation threads instead of 114–192.
  Server load can also change scores a little (F94). These change speed and serving, not the method, and the hardware
  check found no GPU effect.
- **Splits.** Optimizer seeds 1 and 2 use their own train/val split; the paper's runs used seed 0's.
- **† runs.** Lost requests may have changed the search of 7 runs, and IFBench GEPA+Merge seed 2 lost 34 test answers.
- **PUPA.** It uses a different external model and judge, so its numbers measure the methods under that judge only.
- **Table 3.**
  - BeamSearch is our own reading of a one-sentence definition (F109), so its gap to the paper may come from the
    implementation.
  - PUPA has one run each for SelectBestCandidate and BeamSearch, and a different judge.
  - Three of the 20 runs are †, with one lost request or test score each.
- **Scope.** There are no GRPO training runs and no Table 2 (it needs GPT-4.1-mini).

## 10. Fixed in advance, decisions and deviations

| Date | Fixed or decided before the data it applies to | Where |
|---|---|---|
| 2026-11-07 | Match rule: seed-0 gate, final verdict with Holm over 18 targets | §2.4 |
| 2026-11-08 | Verdict script with its checks, before any MIPROv2 or GEPA result; over-long reflection requests are counted, not voided | §2.4 |
| 2026-11-09 | All Table 1 cells on H100 (user); A100 runs become the hardware check; hardware-check method | §2.4 |
| 2026-11-11 | Lost requests no longer void a run; runs marked †, cells also reported without them (user) | §2.4 |
| 2026-11-12 | PUPA with gpt-oss-120b as external model and judge (user); review of the design before any PUPA run | §2.10 |
| 2026-11-13 | Method of the lost-request check | §2.11 |
| 2026-11-14 | Re-score of the 34 lost IFBench test answers; second attempt one request at a time (user) | §2.12 |
| 2026-11-15 | PUPA: per-example method (20:45, after the run scores, before any interval); changed rebuild check, before computing (22:08, user) | §2.10 |
| 2026-11-16 | Paired tests of Table 1, pushed before the run (11:13; run 11:14); the cell means were already known | §2.13 |
| 2026-11-20 | Table 3: runs, budget rule (cap 17,000, raised to 20,000 by the user before any job), our BeamSearch, analysis script with its checks; all before any Table 3 run | §2.14 |
| 2026-11-21 | Table 3: Arbor's start-up wait raised to 1,800 s (F111); after the first void PUPA job, the queue order "paper tasks first" (user) and the time limits of reruns; no Table 3 result used | §2.14 |
| 2026-11-22 | Extensions E1 and E2: arms, the gate tolerance of 1 example (from a replay on the Table 1 logs), runs, checks and the analysis; before any E1 or E2 run with a full budget | §2.15 |
| 2026-11-23 | E2: a gate pass with examples the server left unanswered never discards (user), after the E2 smoke test and before any full run; budget cap 18,000 SBU and the order "seed 0 first" (user) | §2.15 |

Decisions taken after seeing a result:
- PUPA MIPROv2 seed 0 was kept and marked ‡ after its check flagged 1.8 % empty gpt-oss answers, with a rule for the
  later PUPA runs (2026-11-15 12:58, user).
- AIME continued after its seed-0 pause, once the setup was verified (F97).
- IFBench MIPROv2 seed 2 was void twice (F99, F102) and run a third time.
- The PUPA rebuild check was changed (above).
- The IFBench re-score got a second attempt (above).
- The Table 3 cap was raised to 21,000 SBU after the first PUPA failure (2026-11-21 19:42, user). At that point the user
  had seen PUPA seed 1's test scores and the validation scores of the running runs. Only planned runs could use the
  extra budget.
- IFBench SelectBestCandidate seed 0 was counted and marked † (F113). IFBench BeamSearch seed 2 was counted without a
  mark (F114). Both decisions were made under the existing rules, after the score was seen.
- PUPA was stopped at seed 1 after the second failure of seed 0 (2026-11-22, user). The user had seen the interim
  Table 3 results.

- Extensions: seeds 1 and 2 were submitted after the seed-0 test scores were seen, as planned, with no change to code
  or method. The analysis script was written after all 12 scores were known; it follows the method fixed on
  2026-11-22. Three tables of §12 are not in that method and were added afterwards: the returned program on the val
  set, the finalists' scores, and the cost per job.

The full decisions log is §7 of the plan.

## 11. Files

| Path | Contents |
|---|---|
| `REPRODUCTION_PLAN.md` | §1 Table 1 with all per-seed values; §2 plan, rules and analyses (Table 3 in §2.14); §5 run log; §6 findings F1–F114; §7 decisions log |
| `results/table1_h100/` | the 60 Table 1 runs (config, score, per-example logs, program); `verdict_output.md`, `paired_output.md`, `paired_numbers.json` |
| `results/pupa_gptoss/` | the 12 PUPA runs, analysis output, split overlap |
| `results/table3/` | the 20 Table 3 runs; `README.md`, `paired_output.md`, `paired_numbers.json` |
| `results/a100_hardware_check/`, `results/b_baselines/` | the A100 runs |
| `results/extensions_e1_e2/` | the 12 extension runs (with `final_race.json`); `README.md`, `paired_output.md`, `paired_numbers.json` |
| `results/checks/` | the check report of every job; `LOST_REQUESTS_2026-11-13.md` |
| `analysis/` | `table1_verdict.py`, `table1_paired.py`, `table3_paired.py`, `extensions_paired.py`, `gate_replay_subset.py`, `pupa_analysis.py`, `pupa_split_overlap.py`, `hardware_check_all.py`, `lost_request_impact.py` and `grpo/`, with their checks where they have them |
| `scripts/check_beam_selection.py` | the check of our BeamSearch parent selection and of which optimizers get GEPA's arguments (F110) |
| `scripts/check_search_val_subset.py`, `scripts/check_val_gate.py` | the checks of E1 (subset draw, budget, race) and of E2 (gate rule, counting, guard), without a model |
| Snellius `~/repro_archive/` (course account; C's login for §12), `gepa-table3/archive/` (B's own account) | the server logs, too large for git |

## 12. Extensions E1 and E2: a smaller search val set and a staged gate

On HotpotQA and HoVer our GEPA runs spend about 91 % of their rollouts on full val passes of new candidates, and the
paper names a smaller validation set as future work. The two extensions test whether GEPA gets a better program when it spends less
of the same budget on validation and more on proposals. Method, checks and analysis were fixed before any full run
(§2.15 of the plan).

- **GEPA** (control): the Table 1 runs. Every new candidate is scored on all 300 val examples.
- **E1** (`GEPA-SubVal100`): the search scores candidates on a fixed 100 of the 300 val examples, drawn per seed. At the
  end the 5 candidates with the best score on these 100 are scored on the other 200, and the best on all 300 is
  returned. The race costs 1,000 of the budget.
- **E2** (`GEPA-SubVal100-Gate`): E1, plus a gate. A child that passed the three-example test is first scored on 50 of
  the 100 search examples. If it has more than 1 correct answer fewer than its parent there, it is discarded.

All arms use the paper's rollout budgets (6,871 metric calls on HotpotQA, 7,051 on HoVer), the same seeds, splits and
MIPROv2 caches. 12 runs in 6 jobs on H100, 2026-11-23 to 2026-11-25, commit `5c6d954`: the E1 and E2 runs of one task
and seed shared a server, as the GEPA and GEPA+Merge runs did. Runs in `results/extensions_e1_e2/`, analysis by
`analysis/extensions_paired.py` (output in that folder).

### 12.1 Test scores

Test score in %, 300 test examples per run.

| Task | Method | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|---|
| HotpotQA | GEPA | 62.33 | 57.33 | 61.67 | 60.44 ± 2.72 |
| HotpotQA | E1 | 63.33 | 61.33† | 58.00 | 60.89 ± 2.69 |
| HotpotQA | E2 | 61.00 | 59.67 | 62.00 | 60.89 ± 1.17 |
| HoVer | GEPA | 48.67† | 51.00† | 53.33 | 51.00 ± 2.33 |
| HoVer | E1 | 50.00† | 51.67 | 51.67 | 51.11 ± 0.96 |
| HoVer | E2 | 51.67† | 55.67 | 51.67 | 53.00 ± 2.31 |

† = single requests lost in the search (HoVer GEPA 2 and 3, HoVer E1 3, HoVer E2 1, HotpotQA E1 1). No test request
was lost.

Paired per-example tests, as in §3.2: d is the mean over the 300 test examples of A − B in points, each method averaged
over its three seeds, with a 95 % bootstrap interval and a sign-flip test. Holm runs over the six tests.

| Task | A − B | Per seed | d | 95 % interval | p | p (Holm) |
|---|---|---|---|---|---|---|
| HotpotQA | E1 − GEPA | +1.00 / +4.00 / −3.67 | +0.44 | [−2.11, +3.00] | 0.80 | 1.00 |
| HotpotQA | E2 − GEPA | −1.33 / +2.34 / +0.33 | +0.44 | [−2.11, +3.00] | 0.80 | 1.00 |
| HotpotQA | E2 − E1 | −2.33 / −1.66 / +4.00 | +0.00 | [−2.44, +2.44] | 1.00 | 1.00 |
| HoVer | E1 − GEPA | +1.33 / +0.67 / −1.66 | +0.11 | [−2.22, +2.56] | 1.00 | 1.00 |
| HoVer | E2 − GEPA | +3.00 / +4.67 / −1.66 | +2.00 | [−0.56, +4.56] | 0.15 | 0.88 |
| HoVer | E2 − E1 | +1.67 / +4.00 / +0.00 | +1.89 | [−0.78, +4.56] | 0.19 | 0.96 |

| Mean over the two tasks | d | 95 % interval |
|---|---|---|
| E1 − GEPA | +0.28 | [−1.44, +2.06] |
| E2 − GEPA | +1.22 | [−0.61, +3.06] |
| E2 − E1 | +0.94 | [−0.83, +2.72] |

No test passes Holm, and none is below 0.14 before the correction. Every interval includes zero. The per-seed
differences have both signs in five of the six rows. The intervals cover the choice of test examples only, not the
variation between seeds.

Without the † runs the picture is the same (`paired_output.md`): no interval excludes zero, and HoVer E2 − GEPA drops
from +2.00 to +0.33, with one GEPA seed left.

### 12.2 What changed in the search

Mean over the three seeds, range in brackets.

| Task | Method | proposals | candidates | gate: decided / discarded | metric calls | share spent on val |
|---|---|---|---|---|---|---|
| HotpotQA | GEPA | 102 [77, 142] | 22 [21, 22] | | 7,114 | 91 % |
| HotpotQA | E1 | 232 [195, 274] | 46 [43, 48] | | 6,959 | 80 % |
| HotpotQA | E2 | 263 [237, 282] | 35 [32, 37] | 51 / 17 (34 %) | 6,913 | 77 % |
| HoVer | GEPA | 103 [97, 111] | 22 [22, 22] | | 7,220 | 91 % |
| HoVer | E1 | 233 [201, 276] | 47 [45, 49] | | 7,131 | 80 % |
| HoVer | E2 | 281 [254, 321] | 34 [29, 38] | 53 / 20 (37 %) | 7,071 | 76 % |

The extensions did what they were built for. E1 makes 2.3 times as many proposals as GEPA and keeps twice as many
candidates. E2 makes 2.6 to 2.7 times as many proposals and discards about a third of the children that reach the gate.
The share of the budget spent on validation falls less than expected, from 91 % to 80 % and 77 %, because twice as many
candidates each get a val pass and the race takes 1,000 calls.

The extra search buys a better score on the 100 search examples, and nothing beyond them. The table shows the returned
program of each arm on the seed's 100 search examples and on the other 200 val examples (mean over the seeds).

| Task | Method | 100 search examples | other 200 val examples | test |
|---|---|---|---|---|
| HotpotQA | GEPA | 58.67 | 64.17 | 60.44 |
| HotpotQA | E1 | 61.33 | 62.83 | 60.89 |
| HotpotQA | E2 | 60.67 | 62.67 | 60.89 |
| HoVer | GEPA | 48.33 | 51.83 | 51.00 |
| HoVer | E1 | 54.00 | 51.83 | 51.11 |
| HoVer | E2 | 55.33 | 52.17 | 53.00 |

On the 100 examples the extensions' programs are 2.0 to 2.7 points above GEPA's on HotpotQA and 5.7 to 7.0 on HoVer. On
the other 200 they are 1.3 to 1.5 points below GEPA's on HotpotQA and within 0.4 points on HoVer, and on the test set
they are level. GEPA selected on all 300, so
its "other 200" column is itself selection-biased upward; the test column is the fair comparison. Over the 60 race
finalists, the score on the 100 search examples is 1.63 points above the score on the other 200. So the extra
proposals mostly fitted the 100 examples.

E1 and E2 reach the score of GEPA's returned program on these 100 examples early, after 184 to 2,130 metric calls in
11 of 12 runs (GEPA itself: 1,644 to 4,092). This comparison favours the extensions, which selected on exactly these
examples, and it was not tested on test data: no run used a smaller budget.

### 12.3 The final race

The race returned another program than the best on the 100 search examples in 7 of 12 runs. There the returned program
is 3.14 points better on the other 200 on average; it was chosen with these 200, so part of this is selection. We did
not score the programs the race rejected on the test set, so the race's effect on the test score is unknown.

### 12.4 Cost

Per job, two runs sharing one H100 (GEPA with GEPA+Merge; E1 with E2), mean over the seeds.

| Task | Jobs | hours | SBU | generated tokens (M) |
|---|---|---|---|---|
| HotpotQA | GEPA + GEPA+Merge | 6.1 | 1,177 | 17.3 |
| HotpotQA | E1 + E2 | 11.1 (1.82×) | 2,138 | 22.8 |
| HoVer | GEPA + GEPA+Merge | 6.8 | 1,305 | 23.3 |
| HoVer | E1 + E2 | 10.8 (1.58×) | 2,067 | 27.9 |

The rollout budget is the same by design, but a rollout is not the unit of cost. Each proposal needs a reflection
call, and the extensions make 2.3 to 2.7 times as many. The jobs take 1.6 to 1.8 times as long and generate 1.2 to 1.3
times as many tokens. The servers also produced fewer tokens per second (566 to 755 against 710 to 984 in the GEPA
jobs), as expected when more of the work comes in three-example batches and less in 300-example passes. The
comparison is per job: the cost of E1 and E2 cannot be separated, and the GEPA jobs include a GEPA+Merge run.

### 12.5 Verdict

- **Not an improvement.** Neither extension raises the test score in a way these runs can detect. The estimates are
  +0.28 points for E1 and +1.22 for E2 over the two tasks, with intervals that include zero.
- **Not a detectable loss either.** The intervals put the average change between −1.4 and +2.1 points for E1 and
  between −0.6 and +3.1 for E2, over the choice of test examples.
- **Not cheaper.** The rule fixed in advance counted an equal test score as a positive result only at lower cost. At
  the same rollout budget the extensions cost 1.6 to 1.8 times the GPU time, so the result is not positive under that
  rule.
- **E2 on HoVer** (+2.00, two of three seeds positive, p = 0.15) is the largest effect. It is not evidence of a gain:
  it does not pass any test, and it shrinks to +0.33 without the † runs.
- **What it says about GEPA.** With Qwen3-8B on these two tasks, these runs give no sign that GEPA at the paper's
  budget is held back by the number of proposals it can afford. Doubling them, at the price of a noisier val score,
  gave programs that fit the smaller val set better and generalise no better.
- **For the paper.** The result belongs in the paper as a short negative result on the authors' own future-work
  suggestion, with the method fixed in advance. It should not be presented as an improvement of GEPA, and not as
  "the same score at lower cost".

### 12.6 Checks, deviations and limits

- **Checks.** Every run rebuilds to its reported test score (largest gap 0.0033 points). For every run the search
  examples equal the seeded draw, the finalists are the top 5 on them, the returned program is the best finalist on
  all 300, and the metric calls end 1 to 105 above the budget (GEPA: 131 to 281). All 311 gate decisions follow the rule when
  recomputed from the saved scores. The six servers answered every request, with no batch failure and no timeout.
- **The gate guard.** A smoke test of E2 lost 33 requests to two server batch failures (F68), all inside two gate
  passes, and both children were discarded for it. Before any full run we added a guard: a gate pass with examples
  the server left unanswered never discards. In the 12 runs no gate pass had such an example, so the guard never
  acted and E2 ran exactly as the gate without it would have.
- **Order of the smoke test.** The seed-0 jobs started before the guard's own smoke run had left the queue. We kept
  them, checked the first gate decisions of both E2 runs against the saved state, and cancelled the smoke run.
- **The control ran earlier.** The GEPA runs are 10 to 12 days older, from another login and commit (`6ea0b87`), and shared
  their server with GEPA+Merge instead of an extension run. Plain GEPA's code path is unchanged, and every extension run started at the seed-program score that the control
  run's log gives for its 100 search examples.
- **Limits.** Two tasks, one model, three seeds, one subset size (100), one gate size and tolerance, and no run at a
  reduced budget. Differences below about 2.5 points per task cannot be detected. The claim is about these settings.
