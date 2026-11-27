# Could the lost requests of F104 have changed the GEPA search?

B, 2026-11-13. Method fixed before any of the 10 losses was placed: plan §2.11 (commit `4bf00a5`, 16:05) and its addendum (`e766dd5`, 16:21). Tool: `analysis/lost_request_impact.py` (reads only the text logs, opens no pickle); splits: `analysis/lost_request_splits.json`; full output per run and per loss: `analysis/outputs/lost_request_impact.md` and `.json`.

## Result

- All 10 lost requests fell in the full val evaluation of a newly kept candidate. None fell in a minibatch, a merge subsample, a reflection call or a test evaluation.
- Each lost request left exactly one val example of that candidate without an answer (logged score 0). The candidate's val score was therefore 1/300 (0.33 points) lower than it could have been. No such candidate came close to its run's final program: the gaps are 3.33 to 11.00 points, and one example is worth 0.33.
- **9 of the 10 could have changed the later course of the search (CC).** Had the example been answered correctly, two things could have changed. In 5 cases the list from which GEPA samples the next parent would have been different. In 4 cases (GEPA+Merge) the candidate took part in a later merge call.
- **1 could not (NC):** HoVer GEPA seed 1, the request lost at 13:52:48 in the last evaluation of the search (iteration 114). Nothing was selected or merged after it, and the candidate (44.00, or 44.33 with the example right) stays far below the final program (50.33). No later request could have been answered from the cache.
- Every † run therefore has at least one CC loss, and none is "unaffected". The means without the runs that have a CC loss equal the means without all † runs:

| cell | Table 1 (all runs) | without the † runs |
|---|---|---|
| HotpotQA GEPA+Merge | 57.56 ± 2.88 (3) | 57.83 ± 4.01 (2) |
| HoVer GEPA | 51.00 ± 2.33 (3) | 53.33 (1) |
| HoVer GEPA+Merge | 50.33 ± 1.45 (3) | none: all three seeds lost requests |

What CC means: given every answer the run did receive, at least one decision of the search after the loss could have been different. It does not mean that a decision was different, and it says nothing about the size or sign of an effect on the test score. Whether the next random draws would actually have picked another program cannot be computed, because GEPA's random state is not logged.

## The 10 losses

"Size" is context for the report only and does not enter the verdict. It gives the share of list positions that hold another program at the first changed selection (the chance that the next draw alone differs), the number of later selections and merge calls, and the final program's val score.

| run | error | iter. | candidate (val score) | example | module | verdict | reason | size |
|---|---|---|---|---|---|---|---|---|
| HotpotQA GEPA+Merge s2 | 08:18:58 | 52 | 10 (61.67) | val 273 | final_answer | CC | program 10 takes part (as merge candidate or ancestor of one) in the merge call at iteration 53; it is merged at 62 | 26 selections, 5 merge calls after; iteration 54: 5 of 1,563 positions; final 12 (65.00) |
| HoVer GEPA s0 | 07:00:13 | 5 | 1 (43.00) | val 110 | create_query_hop2 | CC | the list at iteration 6 changes (program 1: 285 → 286 entries) | 96 selections after; 1 of 550 positions; final 13 (49.33) |
| HoVer GEPA s0 | 09:30:26 | 46 | 10 (44.00) | val 78 | summarize1 | CC | the list at iteration 47 changes (program 10: 236 → 237) | 55 selections after; 1 of 2,094; final 13 (49.33) |
| HoVer GEPA+Merge s0 | 07:38:11 | 16 | 5 (40.67) | val 104 | create_query_hop2 | CC | program 5 is merged at iteration 17 (1 and 5 via 0) | 101 selections, 11 merge calls after; iteration 18: 25 of 1,641; final 13 (49.33) |
| HoVer GEPA s1 | 09:21:24 | 45 | 9 (40.00) | val 78 | summarize2 | CC | the list at iteration 46 changes (programs 0, 2, 4 and 5 lose one entry each) | 69 selections after; 26 of 1,659; final 5 (50.33) |
| HoVer GEPA s1 | 09:21:24 | 45 | 9 (40.00) | val 49 | create_query_hop3 | CC | the list at iteration 46 changes (program 9: 227 → 228) | 69 selections after; 2 of 1,660; final 5 (50.33) |
| HoVer GEPA s1 | 13:52:48 | 114 | 21 (44.00) | val 237 | create_query_hop3 | **NC** | last evaluation of the search: no later selection or merge, final program unchanged, no later identical request | final 5 (50.33) |
| HoVer GEPA+Merge s1 | 09:12:39 | 57 | 9 (45.67) | val 91 | create_query_hop2 | CC | the list at iteration 58 changes (programs 0, 1, 3 and 5 lose one entry each) | 28 selections, 6 merge calls after; 26 of 1,658; final 18 (56.67) |
| HoVer GEPA+Merge s2 | 07:28:21 | 11 | 2 (43.67) | val 267 | create_query_hop3 | CC | program 2 takes part (as merge candidate or ancestor of one) in the merge call at iteration 12; it is merged at 25, 42 and 54 | 86 selections, 8 merge calls after; iteration 12: 5 of 767; final 21 (52.00) |
| HoVer GEPA+Merge s2 | 09:34:20 | 53 | 11 (44.67) | val 111 | summarize2 | CC | program 11 is merged at iteration 54 (2 and 11 via 0) | 48 selections, 3 merge calls after; iteration 55: 266 of 2,022; final 21 (52.00) |

Two of the 10 were lost in the same second, in the same evaluation (HoVer GEPA s1, 09:21:24). In all 10 tracebacks the connection broke while the client waited for the response headers (httpcore `_receive_response_headers`). This fits F104: the Arbor logs show every request they received as answered.

For context, here is how many val examples were left without an answer in the val evaluations of the new candidates of each run (format failures and lost requests; both are scored 0 by GEPA): HotpotQA GEPA 58 / 47 / 19 and GEPA+Merge 96 / 49 / 14 (seeds 0 / 1 / 2); HoVer GEPA 4 / 32 / 0 and GEPA+Merge 1 / 3 / 10. A lost request affects the search the same way as a format failure. In HotpotQA, format failures are far more frequent than lost requests. In HoVer, format failures are rare: in HoVer GEPA+Merge seed 0 the single lost request is the only unanswered val example in the val evaluations of all its new candidates.

## Check of the tool

- **The 12 HotpotQA and HoVer runs of GEPA and GEPA+Merge, seeds 0 to 2.** Every DSPy evaluation was placed (149 to 331 per run), and every logged val score, Pareto front, selection, merge pair and best program was reproduced. The script found no loss in the 6 runs without †. In the 6 † runs it found exactly the counts of the checks (1, 2, 1, 3, 1 and 2).
- **Two losses checked by hand in the raw logs** (HoVer GEPA s0 07:00:13, HoVer GEPA s1 13:52:48). In each, the error line names the example, and the next `Average Metric` line is the candidate's val evaluation. The candidate's logged val scores contain a single 0, at the index of that example in the val split of the seed.
- **The verdicts do not depend on Python's set order.** Every CC rests on a change in which programs are in the list or in the merge (not only in their order). The NC rests on there being no later selection at all.
- **The addendum** (DSPy's second copies of slow examples) mattered for none of the 10 losses: each hit the evaluation in progress and left its example without an answer. The reading "left behind by an earlier evaluation", which the rule does not require when the evaluation in progress contains the example, is reported in the full output and would not change any verdict.
- **Side check.** HoVer GEPA and GEPA+Merge seed 0 end with the same val (49.33) and test (48.67) scores. The two runs of the pair are nevertheless independent: their first proposals and minibatch scores already differ.

## Reproduce

The run folders are not in the code-only checkout. Copy the six files per run out of git (`git show HEAD:results/table1_h100/seed_X/<run>/run_log.txt.gz` and `run_log_stderr.txt.gz`) into the same `seed_X/<run>/` layout, then:

    python -I analysis/lost_request_impact.py --splits analysis/lost_request_splits.json \
        --md analysis/outputs/lost_request_impact.md --json analysis/outputs/lost_request_impact.json \
        <dir>/seed_{0,1,2}/{HotpotQABench_HotpotMultiHop,hoverBench_HoverMultiHop}_{GEPA,GEPA-MERGE}_qwen3-8b
