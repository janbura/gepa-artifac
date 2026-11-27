# Table 3 of the paper (candidate selection) with three seeds, method of REPRODUCTION_PLAN.md 2.14 (written
# 2026-11-20 16:46 before any Table 3 run; this script was written before any Table 3 result existed)
#   python analysis/table3_paired.py <results dir> [--out numbers.json]
# <results dir> holds table1_h100/ (Baseline and GEPA of HotpotQA, IFBench, HoVer), pupa_gptoss/ (Baseline and
# GEPA of PUPA, and split_overlap.json) and table3/ (SelectBestCandidate and BeamSearch of all four tasks).
# Per-example scores: as in 2.13 (first logged line of each test example, missing = 0); PUPA as in 2.10 (the
# metric logger files test copies of train/val rows under train/val, F108; changed check of 2026-11-15 22:08).
# Standard library only; the bootstrap, sign-flip test and Holm come from table1_paired.py.
import sys, os, json, gzip, re, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table1_paired import read_lines, boot, signflip, holm_adjust, B, ALPHA

TASKS = ["HotpotQA", "IFBench", "HoVer", "PUPA"]  # the paper's column order
FOLDER = {"HotpotQA": "HotpotQABench_HotpotMultiHop", "IFBench": "IFBench_IFBenchCoT2StageProgram",
          "HoVer": "hoverBench_HoverMultiHop", "PUPA": "Papillon_PAPILLON"}
UNITS = {"HotpotQA": 300, "IFBench": 294, "HoVer": 300, "PUPA": 221}
ROWS = ["Baseline", "Abl-SelectBestCandidate", "Abl-BeamSearch", "GEPA"]
NAME = {"Baseline": "Baseline", "Abl-SelectBestCandidate": "SelectBestCandidate", "Abl-BeamSearch": "BeamSearch",
        "GEPA": "GEPA"}
# paper Table 3, Qwen3-8B, in the order of ROWS (one run per cell)
PAPER = {"HotpotQA": [42.33, 58.33, 57.33, 62.33], "IFBench": [36.90, 30.44, 36.39, 38.61],
         "HoVer": [35.33, 45.33, 41.00, 52.33], "PUPA": [80.82, 85.45, 81.08, 91.85]}
COMPARABLE = ["HotpotQA", "IFBench", "HoVer"]  # our PUPA is judged by gpt-oss-120b (plan 2.10)
TESTS = [("GEPA", "Abl-SelectBestCandidate"), ("GEPA", "Abl-BeamSearch")]
SEEDS = (0, 1, 2)
# runs with lost requests (dagger, rule of 2026-11-11); a Table 3 run is added here when its check reports losses
DAGGER = {("HoVer", "GEPA", 0), ("HoVer", "GEPA", 1), ("IFBench", "GEPA", 2),
          ("IFBench", "Abl-SelectBestCandidate", 0),  # one test example lost to an NLTK error (plan F113)
          ("HotpotQA", "Abl-SelectBestCandidate", 0),  # one search request lost to a client connection error (as F104)
          ("HoVer", "Abl-SelectBestCandidate", 0)}  # one test request lost to a client connection error (as F104)


def run_dir(root, task, opt, s):
    if opt in ("Abl-SelectBestCandidate", "Abl-BeamSearch"):
        sub = "table3"
    elif task == "PUPA":
        sub = "pupa_gptoss"
    else:
        sub = "table1_h100"
    return os.path.join(root, sub, f"seed_{s}", f"{FOLDER[task]}_{opt}_qwen3-8b")


def reported(run):
    f = os.path.join(run, "evaluation_results", "evaluation_result.txt")
    return float(open(f).read().splitlines()[1].split(",")[0])


def open_text(path):
    if os.path.exists(path):
        return open(path, encoding="utf-8", errors="replace")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8", errors="replace")
    return None


def table1_scores(run, task):
    per = {}
    for step, idx, v in read_lines(run):
        if idx < 0 or idx >= UNITS[task]:
            sys.exit(f"index {idx} out of range in {run}")
        per.setdefault(idx, []).append(v)
    return [per[u][0] if u in per else 0.0 for u in range(UNITS[task])]


def pupa_scores(run, s, mapping):
    # same rebuild as pupa_analysis.py: the last block of metric calls holds the 221 test positions
    lines = []
    for split in ("train", "val", "test"):
        fh = open_text(os.path.join(run, "metric_logs", split + ".jsonl"))
        if fh is None:
            continue
        for line in fh:
            if line.strip():
                r = json.loads(line)
                lines.append((r["step_counter"], split, r["idx_in_split"], r["metric_output"]))
        fh.close()
    lines.sort(key=lambda x: x[0])
    n = UNITS["PUPA"]
    want = {}
    for pos, (split, idx) in enumerate(mapping[str(s)]):
        want.setdefault((split, idx), []).append(pos)
    e = 0
    while True:
        block = lines[-(n + e):]
        got = {}
        for _, split, idx, val in block:
            got.setdefault((split, idx), []).append(val)
        if all(len(got.get(k, [])) >= len(v) for k, v in want.items()):
            break
        e += 1
        if n + e > len(lines):
            sys.exit(f"stop (plan 2.14): not every PUPA test position found in {run}")
    before = set((sp, ix) for _, sp, ix, _ in lines[:-(n + e)])
    foreign = sum(len(v) for k, v in got.items() if k not in want)
    foreign_late = sum(len(v) for k, v in got.items() if k not in want and k[0] != "test" and k in before)
    extra_copies = [k for k, v in want.items() if k[0] != "test" and len(got[k]) > len(v)]
    steps = [x[0] for x in lines]
    if steps != list(range(1, len(steps) + 1)) or foreign != foreign_late or extra_copies:
        sys.exit(f"stop (plan 2.14, PUPA check of 2.10): {run}: step gaps or lines that are not test copies")
    scores = [0.0] * n
    for k, poss in want.items():
        for pos, val in zip(poss, got[k]):
            scores[pos] = 0.0 if val is None else float(val)
    return scores


def load(root):
    mapping = json.load(open(os.path.join(root, "pupa_gptoss", "split_overlap.json")))["logged_per_seed"]
    S, REP, missing, worst = {}, {}, [], 0.0
    for t in TASKS:
        for o in ROWS:
            for s in SEEDS:
                run = run_dir(root, t, o, s)
                if not os.path.isdir(run):
                    missing.append((t, o, s))
                    continue
                S[(t, o, s)] = pupa_scores(run, s, mapping) if t == "PUPA" else table1_scores(run, t)
                REP[(t, o, s)] = reported(run)
                gap = abs(100 * statistics.mean(S[(t, o, s)]) - REP[(t, o, s)])
                worst = max(worst, gap)
                if gap > 0.005 + 1e-9:
                    sys.exit(f"stop (plan 2.14): {run} rebuilds to {100 * statistics.mean(S[(t, o, s)]):.4f}, "
                             f"reported {REP[(t, o, s)]:.2f}")
    return S, REP, missing, worst


def seeds_of(S, t, o, drop_dagger=False):
    return [s for s in SEEDS if (t, o, s) in S and not (drop_dagger and (t, o, s) in DAGGER)]


def method_means(S, t, o, seeds):
    return [statistics.mean(S[(t, o, s)][u] for s in seeds) for u in range(UNITS[t])]


def diffs_of(S, t, a, c, sa, sc):
    return [100 * (x - y) for x, y in zip(method_means(S, t, a, sa), method_means(S, t, c, sc))]


def pair_row(S, REP, t, a, c, sa, sc, test):
    diffs = diffs_of(S, t, a, c, sa, sc)
    d = statistics.mean(diffs)
    cell = statistics.mean(REP[(t, a, s)] for s in sa) - statistics.mean(REP[(t, c, s)] for s in sc)
    if abs(d - cell) > 0.01:
        sys.exit(f"stop (plan 2.14): {t} {a} - {c}: d {d:.4f} but the cell means differ by {cell:.4f}")
    lo, hi = boot(diffs)
    row = {"task": t, "pair": f"{NAME[a]} - {NAME[c]}", "seeds_a": sa, "seeds_b": sc, "d": d, "ci95": [lo, hi],
           "a_better": sum(1 for x in diffs if x > 0), "b_better": sum(1 for x in diffs if x < 0),
           "per_seed": [REP[(t, a, s)] - REP[(t, c, s)] for s in SEEDS if s in sa and s in sc],
           "paper": PAPER[t][ROWS.index(a)] - PAPER[t][ROWS.index(c)]}
    if test:
        row["p"] = signflip(diffs)
    return row


def aggregate(S, a, c, tasks):
    # mean of the task differences; the bootstrap resamples the test examples within each task (as in 2.13)
    import random
    per = [diffs_of(S, t, a, c, seeds_of(S, t, a), seeds_of(S, t, c)) for t in tasks]
    d = statistics.mean(statistics.mean(x) for x in per)
    rng = random.Random(0)
    vals = []
    for _ in range(B):
        tot = 0.0
        for dl in per:
            n = len(dl)
            tot += sum(dl[rng.randrange(n)] for _ in range(n)) / n
        vals.append(tot / len(per))
    vals.sort()
    return d, vals[int(0.025 * B) - 1], vals[int(0.975 * B) - 1]


def search_stats(run):
    # from run_log.txt only (no pickle): iterations, proposals, accepted candidates, and the parent picks whose
    # tracked val score is below the best val score found so far
    fh = open_text(os.path.join(run, "run_log.txt"))
    if fh is None:
        return None
    best, last, prop, acc, picks, nonbest = None, 0, 0, 0, 0, 0
    for line in fh:
        m = re.match(r"Iteration (\d+): (.*)", line)
        if not m:
            continue
        last = max(last, int(m.group(1)))
        rest = m.group(2)
        if rest.startswith("Base program full valset score:"):
            best = float(rest.rsplit(":", 1)[1])
        elif rest.startswith("Full valset score for new program:"):
            v = float(rest.rsplit(":", 1)[1])
            best = v if best is None else max(best, v)
        elif rest.startswith("New program candidate index:"):
            acc += 1
        elif rest.startswith("Proposed new instruction"):
            prop += 1
        elif rest.startswith("Selected program candidate"):
            x = float(re.search(r"with base score: ([-0-9.eE+]+)", rest).group(1))
            picks += 1
            if best is not None and x < best - 1e-9:
                nonbest += 1
    fh.close()
    return {"iterations": last, "proposals": prop, "accepted": acc, "picks": picks, "picks_not_best": nonbest}


def msd_text(v):
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.2f} (n=1)"
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}" + ("" if len(v) == 3 else f" (n={len(v)})")


def report(root, S, REP, missing, worst):
    num = {"rebuild_largest_gap": worst, "missing_runs": missing, "cells": {}, "aggregate": {}, "tests": [],
           "descriptive": [], "without_dagger": [], "seed0_selectbest": {}, "search": []}
    L = []
    p = L.append
    p("## Table 3 with three seeds (plan 2.14)\n")
    p(f"Rebuild: every run gives its reported score (largest gap {worst:.4f} points). Missing runs: "
      f"{', '.join(f'{t} {NAME[o]} s{s}' for t, o, s in missing) or 'none'}. Test score in %, mean ± SD over the "
      f"seeds [paper, one run]. Our PUPA uses gpt-oss-120b as external model and judge (plan 2.10), so its column "
      f"and the four-task aggregate are not comparable with the paper.\n")
    p("| Method | " + " | ".join(TASKS) + " | Aggregate (4) | Improvement (4) | Aggregate (3, no PUPA) | Improvement (3) |")
    p("|---|" + "---|" * (len(TASKS) + 4))
    agg = {}
    for o in ROWS:
        cells = []
        for t in TASKS:
            v = [REP[(t, o, s)] for s in seeds_of(S, t, o)]
            num["cells"][f"{t} {NAME[o]}"] = {"per_seed": {s: REP[(t, o, s)] for s in seeds_of(S, t, o)},
                                               "paper": PAPER[t][ROWS.index(o)]}
            mark = "†" if any((t, o, s) in DAGGER for s in seeds_of(S, t, o)) else ""
            cells.append(f"{msd_text(v)}{mark} [{PAPER[t][ROWS.index(o)]:.2f}]")
        means4 = [statistics.mean(REP[(t, o, s)] for s in seeds_of(S, t, o)) for t in TASKS if seeds_of(S, t, o)]
        means3 = [statistics.mean(REP[(t, o, s)] for s in seeds_of(S, t, o)) for t in COMPARABLE if seeds_of(S, t, o)]
        agg[o] = (statistics.mean(means4) if len(means4) == 4 else None,
                  statistics.mean(means3) if len(means3) == 3 else None)
        cells_txt = " | ".join(cells)
        pa4 = statistics.mean(PAPER[t][ROWS.index(o)] for t in TASKS)
        pa3 = statistics.mean(PAPER[t][ROWS.index(o)] for t in COMPARABLE)
        base4 = statistics.mean(PAPER[t][0] for t in TASKS)
        base3 = statistics.mean(PAPER[t][0] for t in COMPARABLE)
        a4, a3 = agg[o]
        b4, b3 = agg["Baseline"]
        imp4 = "–" if a4 is None or b4 is None or o == "Baseline" else f"{a4 - b4:+.2f}"
        imp3 = "–" if a3 is None or b3 is None or o == "Baseline" else f"{a3 - b3:+.2f}"
        pimp4 = "" if o == "Baseline" else f" [{pa4 - base4:+.2f}]"
        pimp3 = "" if o == "Baseline" else f" [{pa3 - base3:+.2f}]"
        p(f"| {NAME[o]} | {cells_txt} | {'–' if a4 is None else f'{a4:.2f}'} [{pa4:.2f}] | {imp4}{pimp4} | "
          f"{'–' if a3 is None else f'{a3:.2f}'} [{pa3:.2f}] | {imp3}{pimp3} |")
        num["aggregate"][NAME[o]] = {"agg4": a4, "agg3": a3, "paper_agg4": pa4, "paper_agg3": pa3}
    p("\n† = contains a run with lost requests or a lost test score (rule of 2026-11-11; F113).\n")

    p("### Per seed (seed 0 / 1 / 2)\n")
    p("| Method | " + " | ".join(TASKS) + " |")
    p("|---|" + "---|" * len(TASKS))
    for o in ROWS:
        cells = []
        for t in TASKS:
            cells.append(" / ".join(f"{REP[(t, o, s)]:.2f}{'†' if (t, o, s) in DAGGER else ''}" if (t, o, s) in REP
                                    else "–" for s in SEEDS))
        p(f"| {NAME[o]} | " + " | ".join(cells) + " |")

    p("\n### Observation 3: paired per-example tests (Holm over the 8 tests)\n")
    p(f"d = mean over test examples of A − B in points (each method: mean over its seeds); 95 % percentile bootstrap "
      f"({B:,} resamples); p = paired sign-flip test ({B:,} sign vectors). The interval and p cover the test "
      f"examples, not the run-to-run variation; the per-seed differences show that.\n")
    p("| Task | A − B | Paper | Per seed | d | 95 % interval | A better / B better | p | p (Holm) | Holm at 0.05 |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for t in TASKS:
        for a, c in TESTS:
            sa, sc = seeds_of(S, t, a), seeds_of(S, t, c)
            if not sa or not sc:
                continue
            rows.append(pair_row(S, REP, t, a, c, sa, sc, True))
    for r, x in zip(rows, holm_adjust([r["p"] for r in rows])):
        r["p_holm"] = x
        r["holm_rejects"] = x <= ALPHA
    for r in rows:
        ps = " / ".join(f"{v:+.2f}" for v in r["per_seed"])
        p(f"| {r['task']} | {r['pair']} | {r['paper']:+.2f} | {ps} | {r['d']:+.2f} | [{r['ci95'][0]:+.2f}, "
          f"{r['ci95'][1]:+.2f}] | {r['a_better']} / {r['b_better']} | {r['p']:.4f} | {r['p_holm']:.4f} | "
          f"{'rejects' if r['holm_rejects'] else 'no'} |")
    num["tests"] = rows

    p("\n### Mean over tasks (no test)\n")
    p("| A − B | Tasks | Paper | d | 95 % interval |")
    p("|---|---|---|---|---|")
    for a, c in TESTS + [("Abl-SelectBestCandidate", "Abl-BeamSearch")]:
        for label, tasks in (("HotpotQA, IFBench, HoVer", COMPARABLE), ("all four (PUPA not comparable)", TASKS)):
            if not all(seeds_of(S, t, a) and seeds_of(S, t, c) for t in tasks):
                continue
            d, lo, hi = aggregate(S, a, c, tasks)
            paper = statistics.mean(PAPER[t][ROWS.index(a)] - PAPER[t][ROWS.index(c)] for t in tasks)
            num["aggregate"][f"{NAME[a]} - {NAME[c]} ({len(tasks)})"] = {"d": d, "ci95": [lo, hi], "paper": paper}
            p(f"| {NAME[a]} − {NAME[c]} | {label} | {paper:+.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

    p("\n### Descriptive: SelectBestCandidate − BeamSearch per task\n")
    p("| Task | Paper | Per seed | d | 95 % interval |")
    p("|---|---|---|---|---|")
    for t in TASKS:
        a, c = "Abl-SelectBestCandidate", "Abl-BeamSearch"
        sa, sc = seeds_of(S, t, a), seeds_of(S, t, c)
        if not sa or not sc:
            continue
        r = pair_row(S, REP, t, a, c, sa, sc, False)
        num["descriptive"].append(r)
        ps = " / ".join(f"{v:+.2f}" for v in r["per_seed"])
        p(f"| {t} | {r['paper']:+.2f} | {ps} | {r['d']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] |")

    p("\n### Descriptive: the tested rows without the † runs\n")
    p("| Task | A − B | seeds of A / of B | d | 95 % interval | d with all runs |")
    p("|---|---|---|---|---|---|")
    for r_all in rows:
        t = r_all["task"]
        a, c = [k for k in NAME if NAME[k] == r_all["pair"].split(" - ")[0]][0], \
               [k for k in NAME if NAME[k] == r_all["pair"].split(" - ")[1]][0]
        sa, sc = seeds_of(S, t, a, True), seeds_of(S, t, c, True)
        if sa == r_all["seeds_a"] and sc == r_all["seeds_b"]:
            continue
        if not sa or not sc:
            p(f"| {t} | {r_all['pair']} | {sa} / {sc} | no run left | | {r_all['d']:+.2f} |")
            continue
        r = pair_row(S, REP, t, a, c, sa, sc, False)
        num["without_dagger"].append(r)
        p(f"| {t} | {r['pair']} | {sa} / {sc} | {r['d']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
          f"{r_all['d']:+.2f} |")

    p("\n### Our seed-0 SelectBestCandidate and the authors' released seed-0 run\n")
    p("| Task | ours, seed 0 | authors (= paper) |")
    p("|---|---|---|")
    for t in TASKS:
        k = (t, "Abl-SelectBestCandidate", 0)
        ours = f"{REP[k]:.2f}" if k in REP else "–"
        num["seed0_selectbest"][t] = {"ours": REP.get(k), "authors": PAPER[t][1]}
        p(f"| {t} | {ours} | {PAPER[t][1]:.2f}{' (GPT-4.1-mini judge)' if t == 'PUPA' else ''} |")

    p("\n### Search statistics from the run logs (descriptive; sum over the seeds)\n")
    p("| Task | Method | runs | iterations | proposals | accepted candidates | parent picks | picks below the best so far |")
    p("|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        for o in ROWS[1:]:
            st = [search_stats(run_dir(root, t, o, s)) for s in seeds_of(S, t, o)]
            st = [x for x in st if x]
            if not st:
                continue
            tot = {k: sum(x[k] for x in st) for k in st[0]}
            num["search"].append({"task": t, "method": NAME[o], "runs": len(st), **tot})
            share = f"{tot['picks_not_best']} ({100 * tot['picks_not_best'] / tot['picks']:.0f} %)" if tot["picks"] else "–"
            p(f"| {t} | {NAME[o]} | {len(st)} | {tot['iterations']} | {tot['proposals']} | {tot['accepted']} | "
              f"{tot['picks']} | {share} |")
    return "\n".join(L) + "\n", num


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    S, REP, missing, worst = load(args[0])
    text, numbers = report(args[0], S, REP, missing, worst)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(numbers, f, indent=1)
