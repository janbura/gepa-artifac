# PUPA with gpt-oss-120b as external model and judge (plan 2.10)

PUPA needs GPT-4.1-mini twice: as the untrusted external model inside PAPILLON and as the LLM judge
of the metric. We have no OpenAI key, so both roles use `openai/gpt-oss-120b` (revision `b5c939de`,
vLLM 0.11.0 on the second H100 of each job; reasoning effort low for both roles: fixed for the external
model, chosen for the judge by the judge calibration of the pilot). Qwen3-8B runs exactly as in Table 1. The judge defines the score, so these
numbers are **not** comparable with the paper's PUPA numbers (authors' seed 0: Baseline 80.82,
MIPROv2 81.55, GEPA 91.85, GEPA+Merge 86.26). They are reported as their own block.

Code: branch `feat/pupa-gptoss`, commit `c5089eb` (`~/gepa-pupa` on Snellius), runs in `~/pupa_runs`.

## Runs (test score on the 221 PUPA test examples)

| method | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|
| Baseline | 85.94 | 85.93 | 86.79 | 86.22 ± 0.49 |
| MIPROv2-Heavy | 90.62‡ | 90.80 | 83.79 | 88.40 ± 4.00 (without ‡: 87.30 ± 4.96) |
| GEPA | 92.86 | 90.73 | 93.53 | 92.37 ± 1.46 |
| GEPA+Merge | 91.41 | 94.37 | 89.83 | 91.87 ± 2.30 |

‡ kept although gpt-oss gave empty answers in 42 of 2,346 logged predictions (1.8 %, above the 1 % check
limit); servers clean, every request answered, all 221 test examples scored (plan 2.10, addendum of
2026-11-15 12:58).

| run | job (check) | time, SBU | empty gpt-oss answers / logged predictions |
|---|---|---|---|
| pilot: judge calibration, smoke test, Baselines s0–s2 | 29248984 (29248985) | 45 min, 290 | 1 / 221, 1 / 222, 0 / 222 |
| MIPROv2 s0 / s1 / s2 | 29248986 / 29248990 / 29248994 | 3 h 42 / 3 h 30 / 4 h 04; 1,421 / 1,342 / 1,560 | 42 / 2,346 ‡, 8 / 2,335, 18 / 2,360 |
| GEPA + GEPA+Merge s0 | 29273624 (29273628) | 4 h 05, 1,570 | 24 / 2,863, 22 / 2,896 |
| GEPA + GEPA+Merge s1 | 29248992 (29248993) | 4 h 25, 1,699 | 21 / 2,961, 26 / 2,963 |
| GEPA + GEPA+Merge s2 | 29248996 (29248997) | 4 h 08, 1,588 | 17 / 2,972, 9 / 2,783 |

Every Qwen and gpt-oss request of these jobs was answered (0 timeouts, 0 batch failures, 0 judge errors).
The first pilot (29193231, 22 SBU) stopped at the pre-flight check because the `~/gepa-pupa` worktree had
no `gepa_artifact/utils/arbor`; nothing ran, and the chain was resubmitted with a link to the patched
Arbor of `~/gepa-repro` (plan, run log). All PUPA jobs together: 9,492 SBU.

## Files

- `seed_<s>/Papillon_PAPILLON_<method>_qwen3-8b/`: packed like `results/table1_h100` (config, result
  file, optimized program, metric logs, run logs, GEPA state and candidates). `program.pkl` and
  `gepa_state.bin` are pickles: only load them if you trust them. The saved programs and the logs contain
  example MongoDB connection strings and Stripe test keys that gpt-oss wrote into its answers when PUPA
  users asked for code (they occur only in the logged answers, not in the requests or the PUPA dataset).
  They are model output, not credentials of this project; GitHub's secret scanning flagged four of them.
- `pilot/`: judge calibration output, the chosen reasoning effort, and the pilot job log with the smoke test.
- `split_overlap.json`: output of `analysis/pupa_split_overlap.py` (see below).
- `analysis_output_strict_rule.txt`: output of `analysis/pupa_analysis.py` under the rule of plan 2.10,
  addendum of 2026-11-15 20:45.
- Check reports: `results/checks/<job>.md` and `results/checks/PUPA_SUMMARY_2026-11-15.md`.
- Server logs (Qwen/Arbor requests and responses, gpt-oss server, job logs, GPU use) are kept on
  Snellius in `~/repro_archive/pupa_logs/<job>/`.

## Split overlap (plan F108)

The loader takes the first 443 rows of `Columbia-NLP/PUPA` (`pupa_new`) in order. 25 of the 221 test
examples are exact copies of train or val examples (10 of train, 15 of val in the unshuffled split) and
11 repeat an earlier test example; 43 test examples share their user query with a train or val example.
This comes from the authors' data split, so it holds for their PUPA runs too. The metric logger files a
test copy of a train/val row under train or val, so every `test.jsonl` has 196 lines (197 with a
re-sent request) and 185 distinct indices; the reported scores are not affected.

## Analysis (plan 2.10)

The paper's three PUPA claims (mean of the per-seed differences ± SD): GEPA − Baseline +6.15 ± 1.18
(paper +11.03), GEPA − MIPROv2 +3.97 ± 5.13 (paper +10.30), GEPA+Merge − GEPA −0.50 ± 3.76 (paper −5.59).
All three hold in direction and all three are smaller than in the paper.

Paired per-example differences (221 test examples, pooled over seeds, 95 % bootstrap interval over examples):

| pair | mean difference | 95 % interval | without ‡ |
|---|---|---|---|
| GEPA − Baseline | +6.16 | +3.90 to +8.45 | |
| GEPA − MIPROv2 | +3.97 | +2.16 to +5.79 | +4.83 (+2.39 to +7.34) |
| GEPA+Merge − GEPA | −0.50 | −2.19 to +1.18 | |
| GEPA+Merge − Baseline | +5.65 | +3.38 to +7.95 | |
| GEPA+Merge − MIPROv2 | +3.47 | +1.55 to +5.42 | +4.80 (+2.47 to +7.19) |
| MIPROv2 − Baseline | +2.19 | +0.29 to +4.08 | +0.94 (−1.50 to +3.37) |

The interval covers the test examples of these runs, not the run-to-run variation of the optimizers.
The paired analysis runs under the changed rebuild check of plan 2.10 (deviation of 2026-11-15 22:08, the
first check failed in 7 runs because of late optimizer answers in the test block; see F108). Without the 25
test copies of train/val rows, or without the 43 query overlaps, no difference moves by more than 0.71
points. Full output: `analysis_output.txt`, numbers: `pupa_analysis.json`.

Run: `python -I -X utf8 analysis/pupa_analysis.py results/pupa_gptoss results/pupa_gptoss/split_overlap.json out.json`
(standard library only; needs the run folders, which a sparse laptop copy leaves out).
