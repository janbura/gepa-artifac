# Quick technical test of PUPA with our own external model (plan section 2.10 and its addendum of
# 2026-11-12), run before the real PUPA runs. It runs PAPILLON and its judge on the val examples and
# counts what would make the runs wrong without any error message: exceptions (PAPILLON.forward hides
# them and returns empty strings), external answers that are empty or None (vLLM returns no content
# when gpt-oss stops inside its reasoning) and answers cut at max_tokens.
# The scores are printed for information only, they are not used for anything.
#   python -m scripts.pupa_smoke --port <arbor port> [--n 111] [--threads 32]
# Passed (exit 0) when the external and judge calls that were cut or came back empty are at most 1 % of
# those calls, and the examples with an exception in the program or the judge are at most 1 % of them.
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
parser.add_argument("--n", type=int, default=111)
parser.add_argument("--threads", type=int, default=32)
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
    # None would reach Qwen as the text "None" without any error
    out["empty_external"] = llm_response is None or not str(llm_response).strip()
    out["external_chars"] = len(llm_response or "")
    pred = dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)
    try:
        m = papillon_utils.compute_metrics(ex, pred)
        out["score"] = (m.quality + (1 - m.leakage)) / 2.0
    except Exception as e:
        out["judge_error"] = repr(e)[:300]
    out["seconds"] = time.time() - t0
    return out


def call_stats(lm):
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
    return reasons, empty, tokens


def med(xs):
    return statistics.median(xs) if xs else 0


t_start = time.time()
with ThreadPoolExecutor(args.threads) as pool:
    results = list(pool.map(run_one, examples))
wall = time.time() - t_start

ext_reasons, ext_empty, ext_tokens = call_stats(untrusted_lm)
judge_reasons, judge_empty, judge_tokens = call_stats(papillon_utils.openai_lm)
forward_errors = [r["forward_error"] for r in results if r["forward_error"]]
judge_errors = [r["judge_error"] for r in results if r["judge_error"]]
empty_examples = sum(r["empty_external"] for r in results)
calls = len(untrusted_lm.history) + len(papillon_utils.openai_lm.history)
bad_calls = ext_reasons["length"] + judge_reasons["length"] + ext_empty + judge_empty
bad_examples = len(forward_errors) + len(judge_errors)
scores = [r["score"] for r in results if "score" in r]
secs = [r["seconds"] for r in results if "seconds" in r]
chars = [r["external_chars"] for r in results if "external_chars" in r]

print(f"examples {len(examples)}, wall time {wall:.0f} s, seconds per example median {med(secs):.0f}")
print(f"external calls: finish reasons {dict(ext_reasons)}, empty {ext_empty}, completion tokens median {med(ext_tokens)}, "
      f"max {max(ext_tokens, default=0)}; visible answer characters median {med(chars)}, max {max(chars, default=0)}")
print(f"judge calls: finish reasons {dict(judge_reasons)}, empty {judge_empty}, completion tokens median {med(judge_tokens)}, "
      f"max {max(judge_tokens, default=0)}")
print(f"program errors {len(forward_errors)}, judge errors {len(judge_errors)}, examples with an empty or None external "
      f"answer {empty_examples}; calls cut or empty {bad_calls} of {calls}")
for e in (forward_errors + judge_errors)[:5]:
    print("  error:", e)
print(f"mean score on these val examples (information only): {statistics.mean(scores) if scores else 0:.3f}")
ok = bad_calls <= 0.01 * max(1, calls) and bad_examples <= 0.01 * len(examples) and empty_examples <= 0.01 * len(examples)
print("SMOKE TEST", "PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
