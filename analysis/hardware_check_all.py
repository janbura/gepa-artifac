# hardware check over the five benchmarks (plan section 2.4, addendum of 2026-11-09 12:24)
# score level: d_hw = mean of the three H100 Baselines - mean of the three A100 Baselines,
#   z = d_hw / (SD_samp * sqrt(2/3)), two-sided p, Holm over the five benchmarks at alpha = 0.05
# per example (not AIME, its index is per problem): A100-H100 pairs against A100-A100 pairs,
#   bootstrap over examples as in hardware_check.py (F90)
#   python analysis/hardware_check_all.py results
import gzip, json, math, os, random, sys

res = sys.argv[1]
A100 = res + "/b_baselines"
H100 = res + "/table1_h100"
SD_SAMP = {"HotpotQA": 1.47, "HoVer": 1.63, "IFBench": 1.63, "LiveBench-Math": 2.71, "AIME": 4.07}  # frozen in 2.4
BENCH = {
    "HotpotQA": ("HotpotQABench_HotpotMultiHop_Baseline_qwen3-8b", 300),
    "HoVer": ("hoverBench_HoverMultiHop_Baseline_qwen3-8b", 300),
    "IFBench": ("IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b", 294),
    "LiveBench-Math": ("LiveBenchMathBench_CoT_Baseline_qwen3-8b", 126),
    "AIME": ("AIMEBench_CoT_Baseline_qwen3-8b", 150),
}


def reported(folder):
    return float(open(folder + "/evaluation_results/evaluation_result.txt").read().splitlines()[1].split(",")[0])


def per_example(folder, n):
    p = folder + "/metric_logs/test.jsonl"
    f = open(p, encoding="utf-8") if os.path.exists(p) else gzip.open(p + ".gz", "rt", encoding="utf-8")
    scores = {}
    for line in f:
        d = json.loads(line)
        i = d["idx_in_split"]
        if i in scores:  # straggler copy sent again by DSPy, the first line is the one that counted
            continue
        scores[i] = float(d["metric_output"]) if d["metric_output"] is not None else 0.0
    return [scores.get(i, 0.0) for i in range(n)]  # missing examples failed (format) and count 0


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


print("score level (reported scores)")
print("benchmark | A100 s0 s1 s2 | H100 s0 s1 s2 | d_hw | SD | z | p")
pvals = {}
for b, (name, n) in BENCH.items():
    a = [reported(f"{A100}/seed_{s}/{name}") for s in range(3)]
    h = [reported(f"{H100}/seed_{s}/{name}") for s in range(3)]
    d = sum(h) / 3 - sum(a) / 3
    sd = SD_SAMP[b] * math.sqrt(2 / 3)
    z = d / sd
    p = 2 * (1 - phi(abs(z)))
    pvals[b] = p
    print(f"{b} | {' '.join(f'{x:.2f}' for x in a)} | {' '.join(f'{x:.2f}' for x in h)} | {d:+.2f} | {sd:.2f} | {z:+.2f} | {p:.3f}")

# Holm step-down over the five benchmarks
order = sorted(pvals, key=pvals.get)
stop = False
for k, b in enumerate(order):
    thr = 0.05 / (len(order) - k)
    rej = (not stop) and pvals[b] <= thr
    if not rej:
        stop = True
    print(f"Holm: {b} p {pvals[b]:.3f} vs {thr:.3f}: {'hardware effect' if rej else 'no evidence of a hardware effect'}")

print()
print("per example (A100-H100 pairs against A100-A100 pairs; H100-H100 shown for information)")
for b, (name, n) in BENCH.items():
    if b == "AIME":
        continue
    a = [per_example(f"{A100}/seed_{s}/{name}", n) for s in range(3)]
    h = [per_example(f"{H100}/seed_{s}/{name}", n) for s in range(3)]
    for k, runs in (("A100", a), ("H100", h)):
        for s in range(3):
            calc = 100 * sum(runs[s]) / n
            rep = reported(f"{A100 if k == 'A100' else H100}/seed_{s}/{name}")
            assert abs(calc - rep) < 0.01, (b, k, s, calc, rep)  # the log reproduces the reported score
    same = lambda x, y: sum(1 for i in range(n) if x[i] == y[i]) / n
    cross = [same(x, y) for x in h for y in a]
    within = [same(a[0], a[1]), same(a[0], a[2]), same(a[1], a[2])]
    within_h = [same(h[0], h[1]), same(h[0], h[2]), same(h[1], h[2])]
    d_cross = [sum(abs(x[i] - y[i]) for x in h for y in a) / 9 for i in range(n)]
    d_within = [sum(abs(a[j][i] - a[k][i]) for j, k in ((0, 1), (0, 2), (1, 2))) / 3 for i in range(n)]
    diff = [d_cross[i] - d_within[i] for i in range(n)]
    obs = sum(diff) / n
    rng = random.Random(12345)
    boot = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000))
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    p_two = min(1.0, 2 * min(sum(1 for x in boot if x <= 0), sum(1 for x in boot if x >= 0)) / len(boot))
    print(f"{b}: same score on {100 * sum(cross) / 9:.1f} % (A100-H100, 9 pairs) vs {100 * sum(within) / 3:.1f} % (A100-A100, 3 pairs), "
          f"H100-H100 {100 * sum(within_h) / 3:.1f} %; distance difference {obs:+.4f}, 95 % interval [{lo:+.4f}, {hi:+.4f}], p about {p_two:.2f}")
