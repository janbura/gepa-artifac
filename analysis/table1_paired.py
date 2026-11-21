# per-example paired tests of Table 1, method of REPRODUCTION_PLAN.md 2.13 (written on 2026-11-16, before this
# script was run on the real runs)
#   python analysis/table1_paired.py <dir with seed_0, seed_1, seed_2> [--out numbers.json]
# One unit = one test example; AIME: one problem, the mean of its 5 tries. In each run the first logged line of an
# example (in step order) counts, because re-sent requests add later lines, and an example without a line scores 0.
# For each method: the mean over its seeds. Standard library only.
import sys, os, json, gzip, random, statistics

BENCHES = ["HotpotQA", "HoVer", "IFBench", "AIME", "LiveBench"]
OPTS = ["Baseline", "MIPROv2-Heavy", "GEPA", "GEPA-MERGE"]
NAME = {"Baseline": "Baseline", "MIPROv2-Heavy": "MIPROv2", "GEPA": "GEPA", "GEPA-MERGE": "GEPA+Merge"}
FOLDER = {"HotpotQA": "HotpotQABench_HotpotMultiHop", "HoVer": "hoverBench_HoverMultiHop",
          "IFBench": "IFBench_IFBenchCoT2StageProgram", "AIME": "AIMEBench_CoT", "LiveBench": "LiveBenchMathBench_CoT"}
UNITS = {"HotpotQA": 300, "HoVer": 300, "IFBench": 294, "AIME": 30, "LiveBench": 126}
TRIES = 5  # AIME asks every problem 5 times and the 5 copies share one index

# paper Table 1, Qwen3-8B, in the order of OPTS (same numbers as table1_verdict.py)
PAPER = {"HotpotQA": [42.33, 55.33, 62.33, 64.33], "HoVer": [35.33, 47.33, 52.33, 51.67],
         "IFBench": [36.90, 36.22, 38.61, 28.23], "AIME": [27.33, 20.00, 32.00, 32.00],
         "LiveBench": [48.70, 46.60, 51.95, 51.95]}
PAIRS = [("GEPA", "Baseline"), ("GEPA", "MIPROv2-Heavy"), ("GEPA-MERGE", "GEPA"),
         ("GEPA-MERGE", "Baseline"), ("GEPA-MERGE", "MIPROv2-Heavy"), ("MIPROv2-Heavy", "Baseline")]
MATH = ("AIME", "LiveBench")  # one predictor, so Merge cannot act: GEPA+Merge - GEPA is the same search twice
# runs with lost requests (Table 1 dagger marks, rule of 2026-11-11)
DAGGER = {("HotpotQA", "GEPA-MERGE", 2), ("HoVer", "GEPA", 0), ("HoVer", "GEPA", 1), ("HoVer", "GEPA-MERGE", 0),
          ("HoVer", "GEPA-MERGE", 1), ("HoVer", "GEPA-MERGE", 2), ("IFBench", "GEPA", 2), ("IFBench", "GEPA-MERGE", 2)}
B = 10000
ALPHA = 0.05


def read_lines(run):
    f = os.path.join(run, "metric_logs", "test.jsonl")
    if os.path.exists(f):
        fh = open(f, encoding="utf-8")
    elif os.path.exists(f + ".gz"):  # the copies in results/ are gzipped
        fh = gzip.open(f + ".gz", "rt", encoding="utf-8")
    else:
        sys.exit("no test log in " + run)
    rows = []
    for line in fh:
        if line.strip():
            r = json.loads(line)
            v = r["metric_output"]
            rows.append((r["step_counter"], r["idx_in_split"], 0.0 if v is None else float(v)))
    fh.close()
    rows.sort()  # step order
    return rows


def unit_scores(run, b):
    per = {}
    for step, idx, v in read_lines(run):
        if idx < 0 or idx >= UNITS[b]:
            sys.exit(f"index {idx} out of range in {run}")
        per.setdefault(idx, []).append(v)
    if b == "AIME":
        return [sum(per.get(u, [])[:TRIES]) / TRIES for u in range(UNITS[b])]
    return [per[u][0] if u in per else 0.0 for u in range(UNITS[b])]


def reported(run):
    f = os.path.join(run, "evaluation_results", "evaluation_result.txt")
    return float(open(f).read().splitlines()[1].split(",")[0])


def load(root):
    S, REP = {}, {}
    worst = 0.0
    for b in BENCHES:
        for o in OPTS:
            for s in (0, 1, 2):
                run = os.path.join(root, f"seed_{s}", f"{FOLDER[b]}_{o}_qwen3-8b")
                if not os.path.isdir(run):
                    sys.exit("missing run " + run)
                S[(b, o, s)] = unit_scores(run, b)
                REP[(b, o, s)] = reported(run)
                gap = abs(100 * statistics.mean(S[(b, o, s)]) - REP[(b, o, s)])
                worst = max(worst, gap)
                if gap > 0.005 + 1e-9:
                    sys.exit(f"stop (plan 2.13): {run} rebuilds to {100 * statistics.mean(S[(b, o, s)]):.4f}, "
                             f"reported {REP[(b, o, s)]:.2f}")
    return S, REP, worst


def method_means(S, b, o, seeds):
    return [statistics.mean(S[(b, o, s)][u] for s in seeds) for u in range(UNITS[b])]


def boot(diffs):
    # percentile interval: the 250th and the 9,750th of 10,000 sorted resample means
    rng = random.Random(0)
    n = len(diffs)
    means = []
    for _ in range(B):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * B) - 1], means[int(0.975 * B) - 1]


def signflip(diffs):
    # two-sided paired sign-flip test; a unit with difference 0 does not change the sum, so it is skipped
    rng = random.Random(1)
    n = len(diffs)
    nz = [x for x in diffs if x != 0]
    obs = abs(sum(diffs) / n)
    hits = 0
    for _ in range(B):
        t = sum(x if rng.random() < 0.5 else -x for x in nz) / n
        if abs(t) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1)


def holm_adjust(p):
    # Holm step-down adjusted p-values
    order = sorted(range(len(p)), key=lambda i: p[i])
    m = len(p)
    adj = [0.0] * m
    top = 0.0
    for k, i in enumerate(order):
        top = max(top, min(1.0, (m - k) * p[i]))
        adj[i] = top
    return adj


def pair_row(S, REP, b, a, c, seeds_a, seeds_c, test):
    ma = method_means(S, b, a, seeds_a)
    mc = method_means(S, b, c, seeds_c)
    diffs = [100 * (x - y) for x, y in zip(ma, mc)]
    d = statistics.mean(diffs)
    cell = statistics.mean(REP[(b, a, s)] for s in seeds_a) - statistics.mean(REP[(b, c, s)] for s in seeds_c)
    if abs(d - cell) > 0.01:
        sys.exit(f"stop (plan 2.13): {b} {a} - {c}: d {d:.4f} but the cell means differ by {cell:.4f}")
    lo, hi = boot(diffs)
    row = {"bench": b, "pair": f"{NAME[a]} - {NAME[c]}", "seeds_a": list(seeds_a), "seeds_b": list(seeds_c),
           "units": len(diffs), "d": d, "ci95": [lo, hi],
           "a_better": sum(1 for x in diffs if x > 0), "b_better": sum(1 for x in diffs if x < 0)}
    if test:
        row["p"] = signflip(diffs)
    return row, diffs


def aggregate(S, a, c):
    # mean of the five benchmark differences; the bootstrap resamples the units within each benchmark
    per_bench = []
    for b in BENCHES:
        ma = method_means(S, b, a, (0, 1, 2))
        mc = method_means(S, b, c, (0, 1, 2))
        per_bench.append([100 * (x - y) for x, y in zip(ma, mc)])
    d = statistics.mean(statistics.mean(x) for x in per_bench)
    rng = random.Random(0)
    vals = []
    for _ in range(B):
        tot = 0.0
        for dl in per_bench:
            n = len(dl)
            tot += sum(dl[rng.randrange(n)] for _ in range(n)) / n
        vals.append(tot / len(per_bench))
    vals.sort()
    return d, vals[int(0.025 * B) - 1], vals[int(0.975 * B) - 1]


def fmt_p(x):
    return f"{x:.4f}"


def report(S, REP, worst):
    numbers = {"rebuild_largest_gap": worst, "rows": [], "aggregate": [], "without_dagger": []}
    lines = []
    p = lines.append
    rows = []
    for b in BENCHES:
        for a, c in PAIRS:
            test = not (b in MATH and (a, c) == ("GEPA-MERGE", "GEPA"))
            row, _ = pair_row(S, REP, b, a, c, (0, 1, 2), (0, 1, 2), test)
            row["per_seed"] = [REP[(b, a, s)] - REP[(b, c, s)] for s in (0, 1, 2)]
            row["paper"] = PAPER[b][OPTS.index(a)] - PAPER[b][OPTS.index(c)]
            rows.append(row)
    tested = [r for r in rows if "p" in r]
    for r, x in zip(tested, holm_adjust([r["p"] for r in tested])):
        r["p_holm"] = x
        r["holm_rejects"] = x <= ALPHA
    numbers["rows"] = rows

    p(f"## Per-example paired tests of Table 1 (plan 2.13)\n")
    p(f"Rebuild: all 60 runs give their reported score (largest gap {worst:.4f} points). Units: test examples "
      f"(AIME: problems, mean of 5 tries); each method = mean over its 3 seeds. d = mean of A − B in points, 95 % "
      f"percentile bootstrap over units ({B:,} resamples), p = paired sign-flip test ({B:,} sign vectors), Holm over "
      f"the {len(tested)} tests. Ours per seed = mean ± SD of the three per-seed differences of the reported scores.\n")
    p("| Benchmark | A − B | Paper | Ours per seed | d | 95 % interval | units A better / B better | p | p (Holm) | Holm at 0.05 |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ps = r["per_seed"]
        seed_txt = f"{statistics.mean(ps):+.2f} ± {statistics.stdev(ps):.2f}"
        if "p" in r:
            tail = f"{fmt_p(r['p'])} | {fmt_p(r['p_holm'])} | {'rejects' if r['holm_rejects'] else 'no'}"
        else:
            tail = "same search twice | not tested | –"
        p(f"| {r['bench']} | {r['pair']} | {r['paper']:+.2f} | {seed_txt} | {r['d']:+.2f} | "
          f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['a_better']} / {r['b_better']} | {tail} |")

    p("\n## Aggregate over the five benchmarks (mean of the benchmark differences, no test)\n")
    p("| A − B | Paper (5 benchmarks, no PUPA) | d | 95 % interval |")
    p("|---|---|---|---|")
    for a, c in PAIRS:
        d, lo, hi = aggregate(S, a, c)
        paper = statistics.mean(PAPER[b][OPTS.index(a)] - PAPER[b][OPTS.index(c)] for b in BENCHES)
        numbers["aggregate"].append({"pair": f"{NAME[a]} - {NAME[c]}", "paper": paper, "d": d, "ci95": [lo, hi]})
        p(f"| {NAME[a]} − {NAME[c]} | {paper:+.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

    p("\n## Without the † runs (descriptive, no test; only the rows that contain a † run)\n")
    p("| Benchmark | A − B | seeds of A / of B | d | 95 % interval | d with all runs |")
    p("|---|---|---|---|---|---|")
    for r_all, (b, (a, c)) in zip(rows, [(b, pc) for b in BENCHES for pc in PAIRS]):
        sa = [s for s in (0, 1, 2) if (b, a, s) not in DAGGER]
        sc = [s for s in (0, 1, 2) if (b, c, s) not in DAGGER]
        if len(sa) == 3 and len(sc) == 3:
            continue
        if not sa or not sc:
            numbers["without_dagger"].append({"bench": b, "pair": f"{NAME[a]} - {NAME[c]}", "no_run_left": True})
            p(f"| {b} | {NAME[a]} − {NAME[c]} | {sa} / {sc} | no run left | | {r_all['d']:+.2f} |")
            continue
        row, _ = pair_row(S, REP, b, a, c, sa, sc, False)
        numbers["without_dagger"].append(row)
        p(f"| {b} | {NAME[a]} − {NAME[c]} | {sa} / {sc} | {row['d']:+.2f} | "
          f"[{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}] | {r_all['d']:+.2f} |")
    return "\n".join(lines) + "\n", numbers


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    S, REP, worst = load(args[0])
    text, numbers = report(S, REP, worst)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(numbers, f, indent=1)
