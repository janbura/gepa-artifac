# final Table 1 verdicts with the match rule of REPRODUCTION_PLAN.md 2.4 (pre-registered on 2026-11-07).
# written on 2026-11-08, before any MIPROv2 or GEPA result existed, so the analysis is fixed before the data.
#   python analysis/table1_verdict.py <dir with seed_0, seed_1, seed_2> [more dirs] [--out file.md]
# every score is read from evaluation_results/evaluation_result.txt (never summed from test.jsonl)
import sys, os, math, json
from statistics import NormalDist

BENCHES = ["HotpotQA", "HoVer", "IFBench", "AIME", "LiveBench"]
OPTS = ["Baseline", "MIPROv2-Heavy", "GEPA", "GEPA-MERGE"]
FOLDER = {"HotpotQA": "HotpotQABench_HotpotMultiHop", "HoVer": "hoverBench_HoverMultiHop",
          "IFBench": "IFBench_IFBenchCoT2StageProgram", "AIME": "AIMEBench_CoT", "LiveBench": "LiveBenchMathBench_CoT"}
TEST_SIZE = {"HotpotQA": 300, "HoVer": 300, "IFBench": 294, "AIME": 150, "LiveBench": 126}

# paper Table 1, Qwen3-8B, in the order of OPTS
PAPER = {"HotpotQA": [42.33, 55.33, 62.33, 64.33], "HoVer": [35.33, 47.33, 52.33, 51.67],
         "IFBench": [36.90, 36.22, 38.61, 28.23], "AIME": [27.33, 20.00, 32.00, 32.00],
         "LiveBench": [48.70, 46.60, 51.95, 51.95]}
# sampling SD per cell at the paper's score level, frozen in the rule (match_criterion/noise_window.md 4.1)
SD_SAMP = {"HotpotQA": [1.48, 1.48, 1.45, 1.43], "HoVer": [1.61, 1.69, 1.69, 1.69],
           "IFBench": [1.63, 1.63, 1.64, 1.51], "AIME": [3.93, 3.53, 4.12, 4.12],
           "LiveBench": [2.71, 2.70, 2.71, 2.71]}
SD_SEARCH = 3.22  # pooled over the artifact's 7 GEPA vs GEPA+Merge pairs, 7 df
DF = 7
ALPHA = 0.05
MATH = ("AIME", "LiveBench")  # Merge cannot act on 1-predictor programs: GEPA + GEPA+Merge = one target


# Student's t from the regularized incomplete beta function, so no scipy is needed
def betacf(a, b, x):
    # continued fraction of the incomplete beta function (Numerical Recipes, section 6.4)
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * betacf(a, b, x) / a
    return 1 - math.exp(lbt) * betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betai(df / 2, 0.5, df / (df + t * t))


def t_quantile(q, df):
    # bisection on the cdf, for q > 0.5
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - t_two_sided_p(mid, df) / 2 < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def count_failed(d, b):
    # failed examples never reach the metric, so they are missing from test.jsonl; re-sent stragglers add lines
    f = os.path.join(d, "metric_logs", "test.jsonl")
    if not os.path.exists(f):
        return None
    idx = [json.loads(l)["idx_in_split"] for l in open(f, encoding="utf-8") if l.strip()]
    if b == "AIME":
        # 30 problems x 5 copies share one index: count at most 5 copies per problem
        return TEST_SIZE[b] - sum(min(idx.count(i), 5) for i in set(idx))
    return TEST_SIZE[b] - len(set(idx))


def read_runs(roots):
    runs = {}  # (bench, opt, seed) -> (score, failed, folder)
    for root in roots:
        for s in (0, 1, 2):
            for b in BENCHES:
                for o in OPTS:
                    d = os.path.join(root, f"seed_{s}", f"{FOLDER[b]}_{o}_qwen3-8b")
                    f = os.path.join(d, "evaluation_results", "evaluation_result.txt")
                    if not os.path.exists(f):
                        continue
                    score = float(open(f).read().splitlines()[1].split(",")[0])
                    key = (b, o, s)
                    if key in runs and abs(runs[key][0] - score) > 1e-9:
                        sys.exit(f"two different runs for {key}: {runs[key][2]} and {d}")
                    if key not in runs:
                        runs[key] = (score, count_failed(d, b), d)
    return runs


def mean(xs):
    return sum(xs) / len(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def holm(pvals):
    # indices rejected by Holm's step-down procedure (stops at the first p that is not rejected)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    rejected = set()
    for k, i in enumerate(order):
        if pvals[i] <= ALPHA / (m - k):
            rejected.add(i)
        else:
            break
    return rejected


def targets():
    out = []
    for b in BENCHES:
        out.append((b, "Baseline", ["Baseline"]))
        out.append((b, "MIPROv2", ["MIPROv2-Heavy"]))
        if b in MATH:
            out.append((b, "GEPA + GEPA+Merge (one target)", ["GEPA", "GEPA-MERGE"]))
        else:
            out.append((b, "GEPA", ["GEPA"]))
            out.append((b, "GEPA+Merge", ["GEPA-MERGE"]))
    return out


def verdicts(runs):
    # returns (rows, complete): one row per target with d, interval half-width, statistic, p and the SD used
    z975 = NormalDist().inv_cdf(0.975)
    t975 = t_quantile(0.975, DF)
    rows, complete = [], True
    for b, name, opts in targets():
        j = OPTS.index(opts[0])
        paper = PAPER[b][j]
        xs = [runs[(b, o, s)][0] for o in opts for s in (0, 1, 2) if (b, o, s) in runs]
        if len(xs) < 3 * len(opts):
            complete = False
            rows.append(dict(b=b, name=name, paper=paper, xs=xs, res=None))
            continue
        n = len(xs)
        d = mean(xs) - paper
        if opts == ["Baseline"]:
            sigma = SD_SAMP[b][0]
            sd_used = f"SD_samp {sigma:.2f}"
            stat = d / (sigma * math.sqrt(1 + 1 / n))
            pv = 2 * (1 - NormalDist().cdf(abs(stat)))
            half = z975 * sigma * math.sqrt(1 + 1 / n)
        else:
            sd_run = math.hypot(SD_SAMP[b][j], SD_SEARCH)
            # pooled SD of the benchmark's three optimizer cells (3 runs each, 6 df)
            cell_vars = []
            for o in ("MIPROv2-Heavy", "GEPA", "GEPA-MERGE"):
                ys = [runs[(b, o, s)][0] for s in (0, 1, 2) if (b, o, s) in runs]
                if len(ys) == 3:
                    cell_vars.append(var(ys))
            if len(cell_vars) == 3:
                s_pooled = math.sqrt(mean(cell_vars))
                sigma = max(sd_run, s_pooled)
                sd_used = f"max(SD_run {sd_run:.2f}, s_pooled {s_pooled:.2f})"
            else:
                complete = False
                sigma = sd_run
                sd_used = f"SD_run {sd_run:.2f} (s_pooled missing)"
            stat = d / (sigma * math.sqrt(1 + 1 / n))
            pv = t_two_sided_p(stat, DF)
            half = t975 * sigma * math.sqrt(1 + 1 / n)
        rows.append(dict(b=b, name=name, paper=paper, xs=xs, res=(d, half, stat, pv, sd_used)))
    return rows, complete


def report(runs):
    lines = []
    p = lines.append
    p("## Our runs per cell: mean ± SD over seeds (n) [paper]\n")
    p("| Benchmark | " + " | ".join(OPTS) + " |")
    p("|---|" + "---|" * len(OPTS))
    for b in BENCHES:
        cells = []
        for j, o in enumerate(OPTS):
            xs = [runs[(b, o, s)][0] for s in (0, 1, 2) if (b, o, s) in runs]
            if not xs:
                cells.append(f"– [{PAPER[b][j]:.2f}]")
            elif len(xs) == 1:
                cells.append(f"{xs[0]:.2f} (1) [{PAPER[b][j]:.2f}]")
            else:
                cells.append(f"{mean(xs):.2f} ± {math.sqrt(var(xs)):.2f} ({len(xs)}) [{PAPER[b][j]:.2f}]")
        p(f"| {b} | " + " | ".join(cells) + " |")

    p("\n## Runs: score (failed examples)\n")
    for b in BENCHES:
        for o in OPTS:
            xs = [(s, runs[(b, o, s)]) for s in (0, 1, 2) if (b, o, s) in runs]
            if xs:
                p(f"- {b} {o}: " + "; ".join(f"seed {s} = {r[0]:.2f} ({r[1] if r[1] is not None else '?'})" for s, r in xs))

    rows, complete = verdicts(runs)
    pvals = [r["res"][3] if r["res"] else 1.0 for r in rows]
    rejected = holm(pvals) if complete else set()
    p("\n## Verdict per target: d = our mean − paper, 95 % interval of d, Holm over the 18 targets at 0.05\n")
    if not complete:
        p("**Not all 18 targets have their runs yet: the numbers below are provisional and no verdict is given.**\n")
    p("| Target | Paper | Ours (n) | d | 95 % interval | SD used | z or t | p (unadjusted) | Verdict |")
    p("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        if r["res"] is None:
            p(f"| {r['b']} {r['name']} | {r['paper']:.2f} | {len(r['xs'])} runs | | | | | | missing runs |")
            continue
        d, half, stat, pv, sd_used = r["res"]
        if not complete:
            verdict = "provisional"
        elif i in rejected:
            verdict = "inconsistent with the paper"
        else:
            verdict = "consistent (no evidence of a gap)"
        p(f"| {r['b']} {r['name']} | {r['paper']:.2f} | {mean(r['xs']):.2f} ({len(r['xs'])}) | {d:+.2f} | [{d - half:+.2f}, {d + half:+.2f}] | {sd_used} | {stat:+.2f} | {pv:.3f} | {verdict} |")
    p(f"\nBaselines: z = d / (SD_samp · √(1 + 1/n)), normal. Optimizer targets: t = d / (σ · √(1 + 1/n)) on {DF} df, "
      f"σ = max(SD_run, s_pooled), SD_run = √(SD_samp² + {SD_SEARCH}²); t(0.975, {DF}) = {t_quantile(0.975, DF):.3f}. "
      "n = 3, or 6 for a math GEPA pair.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # the report has symbols the Windows console code page lacks
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    text = report(read_runs(args))
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)
