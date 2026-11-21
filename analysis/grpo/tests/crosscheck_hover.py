# crosscheck_hover.py - independent re-implementation (different code path) of
# HoVer T1 (plain per-example Bernoulli loop) and T3 (McNemar), to check grpo_tests.py.
# usage: python -I crosscheck_hover.py <HoVer GRPO folder> <HoVer Baseline folder>
import sys
import os
import json
import math
import random

g_dir, b_dir = sys.argv[1], sys.argv[2]


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


val = rows(os.path.join(g_dir, "metric_logs", "val.jsonl"))
train = rows(os.path.join(g_dir, "metric_logs", "train.jsonl"))
# pass boundaries from the val_counter: a new pass starts where train_counter jumps by > 100
starts = [0] + [i for i in range(1, len(val)) if val[i]["train_counter"] - val[i - 1]["train_counter"] > 100]
bounds = list(zip(starts, starts[1:] + [len(val)]))
print("passes:", len(bounds))
M = {}  # (split, idx) -> list of 26 outcomes
for k, (a, b) in enumerate(bounds):
    lo, hi = val[a]["step_counter"], val[b - 1]["step_counter"]
    seen = set()
    for r in val[a:b]:
        key = ("val", r["idx_in_split"])
        if key in seen:
            continue
        seen.add(key)
        M.setdefault(key, [0] * len(bounds))[k] = 1 if r["metric_output"] else 0
    for t in train:
        if lo <= t["step_counter"] <= hi and t["idx_in_split"] == 107:
            M.setdefault(("train", 107), [0] * len(bounds))[k] = 1 if t["metric_output"] else 0
for key in M:
    M[key] = M[key]
print("examples:", len(M))
scores = [100 * sum(v[k] for v in M.values()) / 300 for k in range(len(bounds))]
print("pass scores:", [round(s, 2) for s in scores])
gain = max(scores[1:]) - scores[0]
p = [sum(v) / len(v) for v in M.values()]
rng = random.Random(777)
B = 10000
gains = []
for _ in range(B):
    sim = []
    for k in range(26):
        sim.append(100 * sum(1 for pi in p if rng.random() < pi) / 300)
    gains.append(max(sim[1:]) - sim[0])
gains.sort()
print(f"T1 HoVer: obs gain {gain:.3f}; null mean {sum(gains)/B:.3f}; p95 {gains[int(0.95*B)]:.3f}; "
      f"share >= obs {sum(1 for x in gains if x >= gain - 1e-9)/B:.4f}")

# T3 McNemar
gt = {r["idx_in_split"]: r["metric_output"] for r in rows(os.path.join(g_dir, "metric_logs", "test.jsonl"))}
bt = {r["idx_in_split"]: r["metric_output"] for r in rows(os.path.join(b_dir, "metric_logs", "test.jsonl"))}
print("test idx: grpo", len(gt), "base", len(bt), "common", len(set(gt) & set(bt)))
b = sum(1 for i in gt if gt[i] and not bt[i])
c = sum(1 for i in gt if not gt[i] and bt[i])
n = b + c
p_exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
print(f"T3 HoVer: grpo {100*sum(gt.values())/300:.2f} base {100*sum(bt.values())/300:.2f} "
      f"b={b} c={c} exact McNemar p={p_exact:.4f}")
