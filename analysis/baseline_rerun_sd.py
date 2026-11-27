# How noisy is one Baseline run? Compare three numbers for B's HotpotQA and HoVer Baselines (seeds 0-2):
#  1. the SD frozen in the match rule (plan 2.4, from F65's agreement rates),
#  2. the sampling SD estimated from per-example disagreements between our runs (0/1 metric:
#     P(two runs disagree on example i) = 2 p_i (1 - p_i), so the variance of one run's score is
#     sum_i p_i (1 - p_i) / n^2 = mean disagreements / 2 / n^2),
#  3. the SD of the three run scores.
# If 3 is much larger than 2, the runs move together more than independent examples would (run-level noise).
# This is a descriptive check; the rule stays frozen.
#   python analysis/baseline_rerun_sd.py <dir with seed_0, seed_1, seed_2>
import sys, os, json, math, itertools

root = sys.argv[1]
FROZEN = {"HotpotQABench_HotpotMultiHop": 1.47, "hoverBench_HoverMultiHop": 1.63}
N = 300


def per_example(d):
    scores = {}
    for line in open(os.path.join(d, "metric_logs", "test.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["idx_in_split"] in scores:
            continue  # re-sent straggler: the reported score uses the first copy (F90)
        v = r["metric_output"]
        scores[r["idx_in_split"]] = 0.0 if v is None else float(v)
    rep = float(open(os.path.join(d, "evaluation_results", "evaluation_result.txt")).read().splitlines()[1].split(",")[0])
    x = [scores.get(i, 0.0) for i in range(N)]
    assert abs(100 * sum(x) / N - rep) < 0.006, (d, 100 * sum(x) / N, rep)
    return x, N - len(scores)


for bench, frozen in FROZEN.items():
    runs = []
    for s in (0, 1, 2):
        d = os.path.join(root, f"seed_{s}", f"{bench}_Baseline_qwen3-8b")
        if os.path.exists(os.path.join(d, "evaluation_results", "evaluation_result.txt")):
            runs.append((s,) + per_example(d))
    if len(runs) < 2:
        print(bench, "fewer than 2 runs")
        continue
    vals = sorted(set(v for _, x, _ in runs for v in x))
    sc = [100 * sum(x) / N for _, x, _ in runs]
    m = sum(sc) / len(sc)
    sd_obs = math.sqrt(sum((v - m) ** 2 for v in sc) / (len(sc) - 1))
    dis = [sum(1 for i in range(N) if a[1][i] != b[1][i]) for a, b in itertools.combinations(runs, 2)]
    sd_ex = 100 * math.sqrt(sum(dis) / len(dis) / 2) / N
    print(f"{bench}: runs {', '.join(f'seed {s} {100 * sum(x) / N:.2f} (failed {f})' for s, x, f in runs)}; values {vals}")
    print(f"  disagreements per pair {dis} of {N}; sampling SD from them {sd_ex:.2f}; frozen {frozen:.2f}; SD of the {len(sc)} scores {sd_obs:.2f}")
    # chance of an SD at least this large if the frozen value were the truth: (k-1) s^2 / sigma^2 ~ chi2(k-1)
    k = len(sc)
    q = (k - 1) * sd_obs ** 2 / frozen ** 2
    if k == 3:
        p = math.exp(-q / 2)  # chi2 with 2 df
        print(f"  P(SD of 3 runs >= {sd_obs:.2f} | true SD {frozen:.2f}) = {p:.3f}; same with the per-example SD {sd_ex:.2f}: {math.exp(-(k - 1) * sd_obs ** 2 / sd_ex ** 2 / 2):.3f}")
