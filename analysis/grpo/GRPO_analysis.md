# GRPO in the GEPA paper: what the released runs show, and possible fixes

Reproduction of arXiv 2507.19457 (v2), Qwen3-8B. Sources: the paper, and the authors' artifact (the GRPO code and the logs of four GRPO runs: HotpotQA, HoVer, IFBench, PUPA). All log-based numbers were recomputed from the raw logs with two independent sets of scripts. Statements about the code were checked against the source.

## 1. What the paper reports

Test scores, one run per cell (no seeds or variance). GRPO uses 24,000 rollouts per task, GEPA 1,839–7,051. The paper's claim: GEPA beats GRPO by 6 points on average and reaches its best prompt with up to 35× fewer rollouts.

| | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math |
|---|---|---|---|---|---|---|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 |
| GRPO | 43.33 | 35.88 | 38.67 | 86.66 | 38.00 | 51.26 |
| GEPA | 62.33 | 38.61 | 52.33 | 91.85 | 32.00 | 51.95 |
| GRPO test log | failed | failed | yes | yes | none | none |

## 2. What the released GRPO runs show

- **PUPA**: a significant gain (one run): +5.8 test points over the Baseline (paired bootstrap, 95% CI +2.3 to +9.5).
- **HoVer**: no sign of learning. The +3.3 test gain is not significant (McNemar, p = 0.19). Validation shows no trend. The best checkpoint is what picking the best of 25 validations of an unchanged model gives (simulation, p = 0.45).
- **HotpotQA**: the training reward rises from 45 to 54% while validation stays flat. The 150 training examples are seen about 13 times, so the model may overfit them.
- **IFBench**: validation rises by 4 points (p < 0.001), but the reported test score is 1 point below the Baseline. Its validation items come from IFBench's training file.
- **The test runs of HotpotQA and IFBench failed** (0/300 and 0/294): the server with the trained model did not start, and the test step then writes a score near zero instead of stopping. The reported 43.33 and 35.88 appear only in the result files; no released log produces them. AIME and LiveBench have no released GRPO logs.
- Picking the best of 25 validations adds about 3 points (2.6–3.4) to the validation score, even without learning. Test scores are not affected.

## 3. Possible causes in the code (not tested)

1. **It trains on text the model did not write.** Answers are sampled with Qwen3's reasoning ("thinking") on, but the trainer rebuilds each answer from its parsed fields with an empty thinking block. Every logged training sample shows this. The KL term also spikes in all four runs (up to 5×10⁶); the cause is unknown.
2. **Many groups carry no signal.** On HotpotQA, HoVer and IFBench, 56–81% of the 12-answer groups get identical rewards, so their advantage is zero, and they are neither filtered nor re-sampled. PUPA's reward is graded (about 29% such groups), and PUPA is the only task with a significant test gain.
3. **One reward for every module.** All modules of a multi-step program get the program's final score.
4. **Shorter context in training.** Training and validation allow 8,192 tokens, the test 32,768. Answers cut in training count as format failures, which are rare (at most about 2% of training samples).
5. **Checkpoint choice.** The paper describes early stopping. The code instead uses the checkpoint with the best of 26 validation scores, on a noisy validation set of a few hundred examples.

## 4. How it could be fixed (possible extension)

Possible fixes:
- train on the sampled text, or sample without thinking;
- graded rewards (F1 for HotpotQA, share of gold documents for HoVer), and skipping groups without a signal;
- per-module rewards;
- the same context length everywhere;
- report the last checkpoint (or choose it on a second validation split), use several seeds, and check the test run.

A first comparison of the original and the fixed GRPO at the same budget, e.g. HoVer with 10,000 rollouts each, would cost about 4,000–5,400 SBU (estimate) as a one-seed pilot. Before that, a 200-SBU sampling test without training would show whether graded rewards give more signal.

## 5. Plan

We plan no new GRPO runs now. This analysis is the GRPO part of our reproduction. In the released runs GRPO learns little, the code shows several problems that may explain this, and only the PUPA gain is supported by the logs. A fixed GRPO is a candidate extension if time and budget allow.
