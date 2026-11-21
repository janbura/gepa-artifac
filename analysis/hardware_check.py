# does the GPU type change the answers? compare the H100 rerun of IFBench seed 2 with the three A100 runs
# (Baseline seeds only change the train/val split, the test set and the program are the same, so the A100 runs are reruns)
#   python analysis/hardware_check.py results/b_baselines
import json, sys, itertools, os

base = sys.argv[1]  # results/b_baselines
N = 294
runs = {
    "A100 s0": base + "/seed_0/IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b",
    "A100 s1": base + "/seed_1/IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b",
    "A100 s2": base + "/seed_2/IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b",
    "H100 s2": base + "/h100_test/IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b",
}


def load(folder):
    scores = {}
    dup = 0
    for line in open(folder + "/metric_logs/test.jsonl", encoding="utf-8"):
        d = json.loads(line)
        i = d["idx_in_split"]
        if i in scores:
            # a straggler copy that DSPy sent again; the score uses the copy that finished first,
            # which is the first line (checked: this reproduces evaluation_result.txt for every run)
            dup += 1
            continue
        scores[i] = float(d["metric_output"]) if d["metric_output"] is not None else 0.0
    # examples that are not in the log failed (format error) and count as 0
    full = [scores.get(i, 0.0) for i in range(N)]
    failed = [i for i in range(N) if i not in scores]
    reported = open(folder + "/evaluation_results/evaluation_result.txt").read().splitlines()[1].split(",")[0]
    return full, failed, dup, reported


data = {}
for name, folder in runs.items():
    full, failed, dup, reported = load(folder)
    data[name] = (full, set(failed))
    print(f"{name}: score {100 * sum(full) / N:.2f} (reported {reported}), failed {len(failed)}, duplicate log lines {dup}")

print()
print("pair | same score on | mean abs diff | both failed | score diff")
rows = []
for a, b in itertools.combinations(runs, 2):
    xa, fa = data[a]
    xb, fb = data[b]
    same = sum(1 for i in range(N) if xa[i] == xb[i]) / N
    mad = sum(abs(xa[i] - xb[i]) for i in range(N)) / N
    kind = "cross" if ("H100" in a) != ("H100" in b) else "A100-A100"
    rows.append((kind, same, mad))
    print(f"{a} vs {b} ({kind}) | {100 * same:.1f} % | {mad:.3f} | {len(fa & fb)} | {100 * (sum(xa) - sum(xb)) / N:+.2f}")

for kind in ("A100-A100", "cross"):
    r = [x for x in rows if x[0] == kind]
    print(f"{kind}: mean agreement {100 * sum(x[1] for x in r) / len(r):.1f} %, mean abs diff {sum(x[2] for x in r) / len(r):.3f} ({len(r)} pairs)")

# per-example: is the H100 answer further from the A100 mean than an A100 run is from the mean of the other two?
h = data["H100 s2"][0]
a = [data[k][0] for k in ("A100 s0", "A100 s1", "A100 s2")]
mean_a = [sum(x[i] for x in a) / 3 for i in range(N)]
d_h = sum(abs(h[i] - mean_a[i]) for i in range(N)) / N
d_loo = []
for k in range(3):
    others = [a[j] for j in range(3) if j != k]
    m = [(others[0][i] + others[1][i]) / 2 for i in range(N)]
    d_loo.append(sum(abs(a[k][i] - m[i]) for i in range(N)) / N)
print(f"distance to the A100 mean: H100 {d_h:.3f} (against 3 runs); each A100 run to the mean of the other two: {', '.join(f'{x:.3f}' for x in d_loo)}")

# per example: mean distance H100 vs each A100 run, and mean distance between the A100 runs.
# if the GPU does not matter, both have the same expected value; bootstrap over examples for an interval
import random
pairs_in = [(0, 1), (0, 2), (1, 2)]
d_cross = [sum(abs(h[i] - a[k][i]) for k in range(3)) / 3 for i in range(N)]
d_within = [sum(abs(a[j][i] - a[k][i]) for j, k in pairs_in) / 3 for i in range(N)]
diff = [d_cross[i] - d_within[i] for i in range(N)]
obs = sum(diff) / N
rng = random.Random(12345)
boot = []
for _ in range(20000):
    smp = [diff[rng.randrange(N)] for _ in range(N)]
    boot.append(sum(smp) / N)
boot.sort()
lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
p_two = 2 * min(sum(1 for b in boot if b - obs <= -obs) / len(boot), sum(1 for b in boot if b - obs >= -obs) / len(boot))
print(f"mean distance cross {sum(d_cross) / N:.3f}, within A100 {sum(d_within) / N:.3f}, difference {obs:+.4f}, 95 % bootstrap interval [{lo:+.4f}, {hi:+.4f}], two-sided p about {min(1.0, p_two):.2f}")
# the same in score points: how far can a GPU effect on the mean score be? the H100 score vs the A100 mean
ma = sum(sum(x) for x in a) / (3 * N) * 100
print(f"H100 score {100 * sum(h) / N:.2f} vs A100 mean {ma:.2f}: {100 * sum(h) / N - ma:+.2f} points")
