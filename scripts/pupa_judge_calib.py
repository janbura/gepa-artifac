# Judge check for PUPA with our own external model (plan section 2.10, addendum of 2026-11-12).
# The authors' GEPA-family runs kept their val outputs with the score their judge (GPT-4.1-mini) gave
# (analysis/pupa_calibration/authors_val_triples.jsonl). We score the same outputs with our judge at
# reasoning effort low and medium, compare with the authors' scores, and choose the effort for all PUPA
# runs with the rule fixed before any PUPA run:
#   usable effort: judge errors <= 1 % of the outputs and answers cut at max_tokens or empty <= 1 % of the calls;
#   choice: the usable effort with the smaller mean absolute difference to the authors' (mean) score,
#           a difference below 0.005 between the two efforts counts as a tie and goes to low;
#   stop (exit 1, no effort file): no usable effort, low does not use fewer tokens than medium (then the
#           setting has no effect), or the chosen effort differs from the authors' scores by more than 0.25
#           on average (then the judges measure different things: ask the user first).
# Val outputs of the authors only, no test example.
#   python -m scripts.pupa_judge_calib --data <jsonl> --out <report.json> --effort-file <file>
import argparse
import collections
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--effort-file", required=True)
parser.add_argument("--threads", type=int, default=32)
args = parser.parse_args()

if os.path.exists(args.effort_file):
    os.remove(args.effort_file)  # never keep a choice from an earlier attempt

from gepa_artifact.benchmarks.papillon import papillon_utils  # noqa: E402

rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
print(f"judge calibration on {len(rows)} authors' val outputs")


def mean_score(r):
    return sum(r["authors_scores"]) / len(r["authors_scores"])


def run_effort(effort):
    os.environ["PUPA_JUDGE_REASONING_EFFORT"] = effort
    lm = papillon_utils.external_lm("judge")
    judge = papillon_utils.LLMJudge()
    judge.set_lm(lm)

    def one(r):
        try:
            m = judge(user_query=r["user_query"], og_resp=r["target_response"], new_resp=r["response"],
                      updated_query=r["llm_request"], pii_str=r["pii_str"])
            return (m.quality + (1 - m.leakage)) / 2.0, ""
        except Exception as e:
            return None, repr(e)[:200]

    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as pool:
        res = list(pool.map(one, rows))
    reasons = collections.Counter()
    empty = 0
    tokens = []
    for h in lm.history:
        try:
            choice = h["response"].choices[0]
            reasons[choice.finish_reason] += 1
            if not (choice.message.content or "").strip():
                empty += 1
        except Exception:
            reasons["unknown"] += 1
        try:
            tokens.append(int(h["usage"]["completion_tokens"]))
        except Exception:
            pass
    ours = [s for s, _ in res]
    ok = [(s, r) for s, r in zip(ours, rows) if s is not None]
    diffs = [s - mean_score(r) for s, r in ok]
    exact = sum(any(abs(s - a) < 1e-9 for a in r["authors_scores"]) for s, r in ok)
    xs, ys = [s for s, _ in ok], [mean_score(r) for _, r in ok]
    corr = None
    if len(ok) > 2 and statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * statistics.pstdev(xs) * statistics.pstdev(ys))
    calls = len(lm.history)
    out = {
        "effort": effort,
        "outputs": len(rows),
        "judge_errors": sum(s is None for s in ours),
        "first_errors": [e for _, e in res if e][:3],
        "calls": calls,
        "finish_reasons": dict(reasons),
        "empty_answers": empty,
        "completion_tokens_median": statistics.median(tokens) if tokens else None,
        "completion_tokens_max": max(tokens) if tokens else None,
        "mean_abs_diff": statistics.mean(abs(d) for d in diffs) if diffs else None,
        "mean_offset_ours_minus_authors": statistics.mean(diffs) if diffs else None,
        "exact_share": exact / len(ok) if ok else None,
        "pearson_r": corr,
        "our_mean": statistics.mean(xs) if xs else None,
        "authors_mean": statistics.mean(ys) if ys else None,
        "seconds": round(time.time() - t0),
    }
    out["usable"] = (out["judge_errors"] <= 0.01 * len(rows)
                     and reasons["length"] + empty <= 0.01 * max(1, calls) and out["mean_abs_diff"] is not None)
    print(json.dumps(out), flush=True)
    return out


results = {e: run_effort(e) for e in ["low", "medium"]}
low, med = results["low"], results["medium"]
repeated = [r for r in rows if len(r["authors_scores"]) > 1]
report = {
    "results": results,
    "authors_self_check": {
        "outputs_scored_more_than_once": len(repeated),
        "with_different_scores": sum(max(r["authors_scores"]) - min(r["authors_scores"]) > 1e-9 for r in repeated),
    },
}
usable = [e for e in ["low", "medium"] if results[e]["usable"]]
problem = ""
if not usable:
    problem = "no usable effort (too many judge errors, cut or empty answers)"
elif (low["completion_tokens_median"] or 0) >= (med["completion_tokens_median"] or 0):
    problem = "low does not use fewer tokens than medium, so the reasoning effort setting has no effect"
if not problem:
    if len(usable) == 2 and abs(low["mean_abs_diff"] - med["mean_abs_diff"]) < 0.005:
        choice = "low"
    else:
        choice = min(usable, key=lambda e: results[e]["mean_abs_diff"])
    report["choice"] = choice
    if results[choice]["mean_abs_diff"] > 0.25:
        problem = f"the chosen effort ({choice}) differs from the authors' judge by {results[choice]['mean_abs_diff']:.3f} on average"
report["problem"] = problem
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

if problem:
    print("JUDGE CALIBRATION FAILED:", problem)
    raise SystemExit(1)
c = results[report["choice"]]
with open(args.effort_file, "w") as f:
    f.write(report["choice"] + "\n")
print(f"JUDGE CALIBRATION PASSED: effort {report['choice']}, mean abs diff {c['mean_abs_diff']:.3f}, exact {c['exact_share']:.2f}, "
      f"r {c['pearson_r'] if c['pearson_r'] is None else round(c['pearson_r'], 3)}, offset {c['mean_offset_ours_minus_authors']:+.3f}; "
      f"authors' judge differed from itself on {report['authors_self_check']['with_different_scores']} of "
      f"{report['authors_self_check']['outputs_scored_more_than_once']} repeated outputs")
