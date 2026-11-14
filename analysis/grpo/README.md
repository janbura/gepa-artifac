# GRPO analysis (arXiv 2507.19457, Qwen3-8B)

`GRPO_analysis.pdf` (one page, source `GRPO_analysis.md` / `.html`): what the four GRPO runs shipped with the
artifact show, why GRPO does poorly in this setup, and how it could be fixed. Every number in it was recomputed
from the raw logs with the scripts below. The code statements rest on the DSPy fork (`gepa-ai/dspy`, branch
`gepa_study`, 62dc3b6) and Arbor (`Ziems/arbor`, branch `gepa_study`, 113fc35), as checked out for our runs.

Data: the artifact's `experiment_runs_data/experiment_runs/seed_0/*_GRPO_qwen3-8b` folders (HotpotQA, HoVer,
IFBench, PUPA) and the matching `*_Baseline_qwen3-8b` folders. The scripts read them as text and JSON only:
nothing from the data is executed, imported or unpickled. Standard library only.

## tests/ (are the GRPO gains real?)
`python -I tests/grpo_tests.py <seed_0 folder> <output folder> [B]`. The analysis plan is written at the top of
the script, dated before the tests were run.
- T1 selection effect: the best of 25 validations after training minus the one before, against simulations of
  an unchanged model.
- T2 trend over the 26 validation passes: first vs last five passes, and a permutation test of the pass order.
- T3 test set, GRPO vs Baseline, paired on the same examples: McNemar for HoVer, bootstrap for PUPA.
- T4 noise of one validation pass.

Results are in `results.json` and `grpo_tests_stdout.txt`. `crosscheck_hover.py` and `crosscheck_ifbench_pupa.py`
re-implement the main numbers in a second, simpler way.

## log_checks/ (recomputed log numbers)
`python -I log_checks/<script>.py <run folder> [...]`. The `claimN` labels in the output refer to:
1. validation scores (before training, later passes, best, used checkpoint);
2. groups of 12 rollouts with identical rewards;
3. train reward, first vs last 50 steps;
4. KL values;
5. the in-run test evaluation;
6. per-example validation, first vs last 5 passes;
7. completion lengths and the empty thinking block;
8. wall-clock;
9. train reward and validation per 100 steps.

Outputs: `*_out.txt`. `probe_methods.py` shows how much the share of no-signal groups depends on the way the
interleaved log lines are grouped.
