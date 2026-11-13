# Sampling SD of one IFBench Baseline run from our reruns (scores are 0, 0.5 or 1 per example, so the 0/1
# disagreement formula of baseline_rerun_sd.py does not apply). Per example, the variance across runs estimates
# Var(x_i); the variance of one run's score is sum_i Var(x_i) / n^2. Descriptive check of the frozen 1.63 (plan 2.4);
# the rule stays frozen. Written before any IFBench optimizer result existed.
#   python analysis/ifbench_rerun_sd.py results/b_baselines
import sys, os, json, math
root = sys.argv[1]
N = 294
runs = {"A100 s0": "seed_0", "A100 s1": "seed_1", "A100 s2": "seed_2", "H100 s2": "h100_test"}


def load(folder):
    d = os.path.join(root, folder, "IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b")
    seen = {}
    for line in open(os.path.join(d, "metric_logs", "test.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["idx_in_split"] not in seen:  # first copy counts (F90)
                v = r["metric_output"]
                seen[r["idx_in_split"]] = 0.0 if v is None else float(v)
    x = [seen.get(i, 0.0) for i in range(N)]
    rep = float(open(os.path.join(d, "evaluation_results", "evaluation_result.txt")).read().splitlines()[1].split(",")[0])
    assert abs(100 * sum(x) / N - rep) < 0.006, (folder, 100 * sum(x) / N, rep)
    return x


data = {k: load(v) for k, v in runs.items()}
for label, keys in [("3 A100 runs", ["A100 s0", "A100 s1", "A100 s2"]), ("3 A100 + 1 H100 run", list(runs))]:
    xs = [data[k] for k in keys]
    k = len(xs)
    var_sum = 0.0
    for i in range(N):
        vals = [x[i] for x in xs]
        m = sum(vals) / k
        var_sum += sum((v - m) ** 2 for v in vals) / (k - 1)
    sd_ex = 100 * math.sqrt(var_sum) / N
    sc = [100 * sum(x) / N for x in xs]
    ms = sum(sc) / k
    sd_tot = math.sqrt(sum((s - ms) ** 2 for s in sc) / (k - 1))
    print(f"{label}: scores {', '.join(f'{s:.2f}' for s in sc)}; per-example sampling SD {sd_ex:.2f} (frozen 1.63); SD of the totals {sd_tot:.2f}")
