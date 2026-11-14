# crosscheck_ifbench_pupa.py - independent plain re-implementation of IFBench T1/T2a and
# PUPA T3 (bootstrap), different code and seeds than grpo_tests.py.
# usage: python -I crosscheck_ifbench_pupa.py <IFBench GRPO folder> <PUPA GRPO folder> <PUPA Baseline folder>
import sys
import os
import json
import random

if_dir, pg_dir, pb_dir = sys.argv[1], sys.argv[2], sys.argv[3]


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---- IFBench: passes by train_counter jumps, first line per idx, missing = 0
val = rows(os.path.join(if_dir, "metric_logs", "val.jsonl"))
starts = [0] + [i for i in range(1, len(val)) if val[i]["train_counter"] - val[i - 1]["train_counter"] > 100]
bounds = list(zip(starts, starts[1:] + [len(val)]))
X = [[0.0] * len(bounds) for _ in range(300)]
for k, (a, b) in enumerate(bounds):
    seen = set()
    for r in val[a:b]:
        i = r["idx_in_split"]
        if i not in seen:
            seen.add(i)
            X[i][k] = float(r["metric_output"])
sc = [100 * sum(X[i][k] for i in range(300)) / 300 for k in range(len(bounds))]
print("IFBench passes", len(bounds), [round(s, 2) for s in sc])
gain = max(sc[1:]) - sc[0]
rng = random.Random(4242)
B = 10000
cnt = 0
tot = 0.0
for _ in range(B):
    sim = [100 * sum(rng.choice(X[i]) for i in range(300)) / 300 for _ in range(26)]
    g = max(sim[1:]) - sim[0]
    tot += g
    cnt += g >= gain - 1e-9
print(f"IFBench T1: obs gain {gain:.3f}; null mean {tot/B:.3f}; share >= obs {cnt/B:.4f}")
d = [sum(X[i][21:26]) / 5 - sum(X[i][0:5]) / 5 for i in range(300)]
obs = abs(sum(d))
cnt = sum(1 for _ in range(B) if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-9)
print(f"IFBench T2a: effect {100*sum(d)/300:.3f} points; sign-flip p ~ {(cnt+1)/(B+1):.5f}")

# ---- PUPA T3: all test-phase lines (test/val/train files) with step >= first test step
def test_phase(run):
    t = rows(os.path.join(run, "metric_logs", "test.jsonl"))
    s0 = min(r["step_counter"] for r in t)
    out = {}
    for split in ("test", "val", "train"):
        for r in rows(os.path.join(run, "metric_logs", split + ".jsonl")):
            if split == "test" or r["step_counter"] >= s0:
                out.setdefault((split, r["idx_in_split"]), []).append((r["step_counter"], float(r["metric_output"])))
    return {k: [v for _, v in sorted(vs)] for k, vs in out.items()}


G, Bs = test_phase(pg_dir), test_phase(pb_dir)
print("PUPA keys grpo", len(G), "base", len(Bs), "same keys:", set(G) == set(Bs),
      "lines grpo", sum(len(v) for v in G.values()), "base", sum(len(v) for v in Bs.values()))
diffs = []
for k in G:
    for a, b in zip(G[k], Bs[k]):
        diffs.append(a - b)
print(f"PUPA T3: n {len(diffs)} grpo {100*sum(sum(v) for v in G.values())/221:.2f} "
      f"base {100*sum(sum(v) for v in Bs.values())/221:.2f} diff {100*sum(diffs)/len(diffs):.3f}")
rng = random.Random(99)
bs = sorted(100 * sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(B))
print(f"PUPA T3 bootstrap 95% CI: [{bs[int(0.025*B)]:.2f}, {bs[int(0.975*B)]:.2f}]")
