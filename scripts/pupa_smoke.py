# Quick technical test of PUPA with our own external model (plan section 2.5), run before the real
# PUPA runs. It runs PAPILLON and its judge on the first val examples and counts what would make the
# runs wrong without any error message: exceptions (PAPILLON.forward hides them and returns empty
# strings), empty answers of the external model and answers cut at max_tokens.
# The scores are printed for information only, they are not used for anything.
#   python -m scripts.pupa_smoke --port <arbor port> [--n 20]
# Exit code 0 = passed: no exception, no empty external answer, at most 1 answer cut at max_tokens.
import argparse
import collections
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import dspy

from gepa_artifact.benchmarks.papillon import PAPILLON, Papillon, untrusted_lm
from gepa_artifact.benchmarks.papillon import papillon_utils

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True)
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--threads", type=int, default=10)
args = parser.parse_args()

# Qwen3-8B with the same settings as the runs (lm_config of the commands + create_lm in run_experiments.py)
qwen = dspy.LM(
    model="openai/arbor:qwen/qwen3-8b",
    api_key="API_KEY",
    api_base=f"http://localhost:{args.port}/v1/",
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=16384,
    timeout=21600,
    num_retries=0,
)
dspy.configure(lm=qwen)

print("code from:", papillon_utils.__file__)
print("external model:", untrusted_lm.model, {k: v for k, v in untrusted_lm.kwargs.items() if k != "api_key"})
print("judge model:", papillon_utils.openai_lm.model, {k: v for k, v in papillon_utils.openai_lm.kwargs.items() if k != "api_key"})

bench = Papillon()
examples = bench.val_set[: args.n]
prog = PAPILLON(untrusted_model=untrusted_lm)


def run_one(ex):
    # the same steps as PAPILLON.forward, but without its try/except, so errors show up here
    out = {"forward_error": "", "judge_error": "", "empty_external": False}
    t0 = time.time()
    try:
        llm_request = prog.craft_redacted_request(user_query=ex.user_query).llm_request
        llm_response = untrusted_lm(llm_request)[0]
        response = prog.respond_to_query(
            related_llm_request=llm_request, related_llm_response=llm_response, user_query=ex.user_query
        ).response
    except Exception as e:
        out["forward_error"] = repr(e)[:300]
        return out
    out["empty_external"] = not str(llm_response).strip()
    pred = dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)
    try:
        m = papillon_utils.compute_metrics(ex, pred)
        out["score"] = (m.quality + (1 - m.leakage)) / 2.0
    except Exception as e:
        out["judge_error"] = repr(e)[:300]
    out["seconds"] = time.time() - t0
    return out


def finish_reasons(lm):
    reasons = collections.Counter()
    tokens = []
    for h in lm.history:
        try:
            reasons[h["response"].choices[0].finish_reason] += 1
        except Exception:
            reasons["unknown"] += 1
        try:
            tokens.append(int(h["usage"]["completion_tokens"]))
        except Exception:
            pass
    return reasons, tokens


def med(xs):
    return statistics.median(xs) if xs else 0


t_start = time.time()
with ThreadPoolExecutor(args.threads) as pool:
    results = list(pool.map(run_one, examples))
wall = time.time() - t_start

ext_reasons, ext_tokens = finish_reasons(untrusted_lm)
judge_reasons, judge_tokens = finish_reasons(papillon_utils.openai_lm)
forward_errors = [r["forward_error"] for r in results if r["forward_error"]]
judge_errors = [r["judge_error"] for r in results if r["judge_error"]]
empty = sum(r["empty_external"] for r in results)
cut = ext_reasons["length"] + judge_reasons["length"]
scores = [r["score"] for r in results if "score" in r]
secs = [r["seconds"] for r in results if "seconds" in r]

print(f"examples {len(examples)}, wall time {wall:.0f} s, seconds per example median {med(secs):.0f}")
print(f"external calls: finish reasons {dict(ext_reasons)}, completion tokens median {med(ext_tokens)}, max {max(ext_tokens, default=0)}")
print(f"judge calls: finish reasons {dict(judge_reasons)}, completion tokens median {med(judge_tokens)}, max {max(judge_tokens, default=0)}")
print(f"program errors {len(forward_errors)}, judge errors {len(judge_errors)}, empty external answers {empty}, answers cut at max_tokens {cut}")
for e in (forward_errors + judge_errors)[:5]:
    print("  error:", e)
print(f"mean score on these val examples (information only): {statistics.mean(scores) if scores else 0:.3f}")
ok = not forward_errors and not judge_errors and empty == 0 and cut <= 1
print("SMOKE TEST", "PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
