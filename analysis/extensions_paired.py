# Extensions E1 (GEPA-SubVal100) and E2 (GEPA-SubVal100-Gate) against the Table 1 GEPA runs, method of
# REPRODUCTION_PLAN.md 2.15 (written 2026-11-22 and 2026-11-23, before any full E1 or E2 run).
#   python analysis/extensions_paired.py <results dir> [--out numbers.json]
# <results dir> holds table1_h100/ (GEPA of HotpotQA and HoVer), extensions_e1_e2/ (the 12 extension runs) and
# checks/ (the job reports).
# Primary part (test scores, paired tests): from the logs only, with the bootstrap, sign-flip test and Holm of
# table1_paired.py. Descriptive parts (search, validation, race): from run_log.txt, final_race.json and
# gepa_state.bin. The state is read with an unpickler that only builds plain containers and random.Random.
import sys, os, json, gzip, re, pickle, random, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table1_paired import read_lines, boot, signflip, holm_adjust, B, ALPHA

TASKS = ["HotpotQA", "HoVer"]
FOLDER = {"HotpotQA": "HotpotQABench_HotpotMultiHop", "HoVer": "hoverBench_HoverMultiHop"}
N_TEST, N_VAL, N_SEARCH, RACE_CALLS = 300, 300, 100, 1000
BUDGET = {"HotpotQA": 6871, "HoVer": 7051}
ARMS = ["GEPA", "GEPA-SubVal100", "GEPA-SubVal100-Gate"]
NAME = {"GEPA": "GEPA", "GEPA-SubVal100": "E1", "GEPA-SubVal100-Gate": "E2"}
TESTS = [("GEPA-SubVal100", "GEPA"), ("GEPA-SubVal100-Gate", "GEPA"), ("GEPA-SubVal100-Gate", "GEPA-SubVal100")]
SEEDS = (0, 1, 2)
# runs with lost requests (rule of 2026-11-11): all of them lost single requests in the search, none in the test
DAGGER = {("HoVer", "GEPA", 0): 2, ("HoVer", "GEPA", 1): 3, ("HoVer", "GEPA-SubVal100", 0): 3,
          ("HoVer", "GEPA-SubVal100-Gate", 0): 1, ("HotpotQA", "GEPA-SubVal100", 1): 1}
JOBS = {"GEPA": {("HotpotQA", s): f"hg-hotpot-s{s}" for s in SEEDS} | {("HoVer", s): f"hg-hover-s{s}" for s in SEEDS},
        "EXT": {("HotpotQA", s): f"e12-hotpot-s{s}" for s in SEEDS} | {("HoVer", s): f"e12-hover-s{s}" for s in SEEDS}}
CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 5871]


def run_dir(root, task, arm, s):
    sub = "table1_h100" if arm == "GEPA" else "extensions_e1_e2"
    return os.path.join(root, sub, f"seed_{s}", f"{FOLDER[task]}_{arm}_qwen3-8b")


def reported(run):
    return float(open(os.path.join(run, "evaluation_results", "evaluation_result.txt")).read().splitlines()[1].split(",")[0])


def open_text(path):
    if os.path.exists(path):
        return open(path, encoding="utf-8", errors="replace")
    return gzip.open(path + ".gz", "rt", encoding="utf-8", errors="replace")


def test_scores(run):
    per = {}
    for step, idx, v in read_lines(run):
        if idx < 0 or idx >= N_TEST:
            sys.exit(f"index {idx} out of range in {run}")
        per.setdefault(idx, []).append(v)
    return [per[u][0] if u in per else 0.0 for u in range(N_TEST)]


class PlainUnpickler(pickle.Unpickler):
    ALLOWED = {("builtins", "set"), ("builtins", "frozenset"), ("builtins", "list"), ("builtins", "dict"),
               ("builtins", "tuple"), ("random", "Random"), ("builtins", "object"), ("copyreg", "_reconstructor")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"gepa_state.bin wants {module}.{name}, which this script does not load")


def load_state(run):
    with open(os.path.join(run, "gepa_state.bin"), "rb") as f:
        return PlainUnpickler(f).load()


def f0(x):
    return 0.0 if x is None else float(x)


def search_ids(seed):
    ids = list(range(N_VAL))
    random.Random(f"search_val_subset:{seed}").shuffle(ids)
    return sorted(ids[:N_SEARCH]), sorted(ids[N_SEARCH:])


def log_stats(run):
    it, prop, acc, gate, cut, kept = 0, 0, 0, 0, 0, 0
    fh = open_text(os.path.join(run, "run_log.txt"))
    for line in fh:
        m = re.match(r"Iteration (\d+): (.*)", line)
        if not m:
            continue
        it = max(it, int(m.group(1)))
        rest = m.group(2)
        if rest.startswith("Proposed new instruction"):
            prop += 1
        elif rest.startswith("New program candidate index:"):
            acc += 1
        elif rest.startswith("Val gate:"):
            gate += 1
            if "not discarded" in rest:
                kept += 1
            elif rest.rstrip().endswith("discarded"):
                cut += 1
    fh.close()
    return {"iterations": it, "proposals": prop, "accepted": acc, "gate": gate, "cut": cut, "guard_kept": kept}


def job_report(root, name):
    d = os.path.join(root, "checks")
    for f in sorted(os.listdir(d)):
        if not re.fullmatch(r"\d+\.md", f):
            continue
        text = open(os.path.join(d, f), encoding="utf-8").read()
        m = re.search(r"- job: " + re.escape(name) + r", state (\w+).*?elapsed (\d+):(\d+):(\d+), SBU (\d+)", text)
        if m:
            tok = re.search(r"generated ([0-9.]+) M tokens", text)
            req = re.search(r"- requests (\d+), responses (\d+)", text)
            return {"job": f[:-3], "state": m.group(1), "hours": int(m.group(2)) + int(m.group(3)) / 60 + int(m.group(4)) / 3600,
                    "sbu": int(m.group(5)), "mtok": float(tok.group(1)), "requests": int(req.group(1))}
    sys.exit("no check report for job " + name)


def msd(v):
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}"


def main(root, out):
    L = []
    p = L.append
    num = {"cells": {}, "tests": [], "aggregate": [], "without_dagger": [], "search": [], "validation": [], "curves": [],
           "race": [], "cost": []}
    # ---------- primary: test scores
    S, REP, worst = {}, {}, 0.0
    for t in TASKS:
        for a in ARMS:
            for s in SEEDS:
                run = run_dir(root, t, a, s)
                S[(t, a, s)] = test_scores(run)
                REP[(t, a, s)] = reported(run)
                gap = abs(100 * statistics.mean(S[(t, a, s)]) - REP[(t, a, s)])
                worst = max(worst, gap)
                if gap > 0.005 + 1e-9:
                    sys.exit(f"stop (plan 2.15): {run} rebuilds to {100 * statistics.mean(S[(t, a, s)]):.4f}, reported {REP[(t, a, s)]:.2f}")
    num["rebuild_largest_gap"] = worst

    def means(t, a, seeds):
        return [statistics.mean(S[(t, a, s)][u] for s in seeds) for u in range(N_TEST)]

    def diffs(t, a, c, sa=SEEDS, sc=SEEDS):
        return [100 * (x - y) for x, y in zip(means(t, a, sa), means(t, c, sc))]

    p("## Extensions E1 and E2 against GEPA (plan 2.15)\n")
    p(f"E1 = GEPA-SubVal100 (search on 100 of the 300 val examples, final race of the top 5 on the other 200). "
      f"E2 = GEPA-SubVal100-Gate (E1 plus the 50-example gate). GEPA = the Table 1 runs. Same rollout budget in "
      f"all arms. Rebuild: every run gives its reported score (largest gap {worst:.4f} points).\n")
    p("### Test score (%, 300 test examples per run)\n")
    p("| Task | Method | seed 0 | seed 1 | seed 2 | mean ± SD |")
    p("|---|---|---|---|---|---|")
    for t in TASKS:
        for a in ARMS:
            v = [REP[(t, a, s)] for s in SEEDS]
            cells = " | ".join(f"{REP[(t, a, s)]:.2f}{'†' if (t, a, s) in DAGGER else ''}" for s in SEEDS)
            p(f"| {t} | {NAME[a]} | {cells} | {msd(v)} |")
            num["cells"][f"{t} {NAME[a]}"] = {"per_seed": v, "mean": statistics.mean(v), "sd": statistics.stdev(v)}
    p("\n† = run with single requests lost in the search (rule of 2026-11-11): "
      + ", ".join(f"{t} {NAME[a]} s{s} {n}" for (t, a, s), n in sorted(DAGGER.items())) + ". No test request was lost.\n")

    p("### Paired per-example tests (Holm over the 6 tests)\n")
    p(f"d = mean over the 300 test examples of A − B in points (each method: mean over its three seeds); 95 % "
      f"percentile bootstrap ({B:,} resamples); p = paired sign-flip test ({B:,} sign vectors). The interval and p "
      f"cover the test examples, not the run-to-run variation; the per-seed differences show that.\n")
    p("| Task | A − B | Per seed | d | 95 % interval | A better / B better | p | p (Holm) | Holm at 0.05 |")
    p("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for t in TASKS:
        for a, c in TESTS:
            dl = diffs(t, a, c)
            d = statistics.mean(dl)
            cell = statistics.mean(REP[(t, a, s)] for s in SEEDS) - statistics.mean(REP[(t, c, s)] for s in SEEDS)
            if abs(d - cell) > 0.01:
                sys.exit(f"stop (plan 2.15): {t} {a} - {c}: d {d:.4f} but the cell means differ by {cell:.4f}")
            lo, hi = boot(dl)
            rows.append({"task": t, "pair": f"{NAME[a]} - {NAME[c]}", "d": d, "ci95": [lo, hi],
                         "a_better": sum(1 for x in dl if x > 0), "b_better": sum(1 for x in dl if x < 0),
                         "per_seed": [REP[(t, a, s)] - REP[(t, c, s)] for s in SEEDS], "p": signflip(dl)})
    for r, x in zip(rows, holm_adjust([r["p"] for r in rows])):
        r["p_holm"], r["holm_rejects"] = x, x <= ALPHA
    for r in rows:
        ps = " / ".join(f"{v:+.2f}" for v in r["per_seed"])
        p(f"| {r['task']} | {r['pair'].replace(' - ', ' − ')} | {ps} | {r['d']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
          f"{r['a_better']} / {r['b_better']} | {r['p']:.4f} | {r['p_holm']:.4f} | {'rejects' if r['holm_rejects'] else 'no'} |")
    num["tests"] = rows

    p("\n### Mean over the two tasks (no test)\n")
    p("| A − B | d | 95 % interval |")
    p("|---|---|---|")
    for a, c in TESTS:
        per = [diffs(t, a, c) for t in TASKS]
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
        lo, hi = vals[int(0.025 * B) - 1], vals[int(0.975 * B) - 1]
        num["aggregate"].append({"pair": f"{NAME[a]} - {NAME[c]}", "d": d, "ci95": [lo, hi]})
        p(f"| {NAME[a]} − {NAME[c]} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

    p("\n### Descriptive: the rows with a † run, without the † runs\n")
    p("| Task | A − B | seeds of A / of B | d | 95 % interval | d with all runs |")
    p("|---|---|---|---|---|---|")
    for r in rows:
        t = r["task"]
        a, c = [next(k for k in NAME if NAME[k] == x) for x in r["pair"].split(" - ")]
        sa = [s for s in SEEDS if (t, a, s) not in DAGGER]
        sc = [s for s in SEEDS if (t, c, s) not in DAGGER]
        if len(sa) == 3 and len(sc) == 3:
            continue
        dl = diffs(t, a, c, sa, sc)
        lo, hi = boot(dl)
        num["without_dagger"].append({"task": t, "pair": r["pair"], "seeds_a": sa, "seeds_b": sc, "d": statistics.mean(dl), "ci95": [lo, hi]})
        p(f"| {t} | {r['pair'].replace(' - ', ' − ')} | {sa} / {sc} | {statistics.mean(dl):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['d']:+.2f} |")

    # ---------- descriptive: the search
    ST, RACE = {}, {}
    for t in TASKS:
        for a in ARMS:
            for s in SEEDS:
                run = run_dir(root, t, a, s)
                ST[(t, a, s)] = load_state(run)
                if a != "GEPA":
                    RACE[(t, a, s)] = json.load(open(os.path.join(run, "final_race.json")))
    # checks of the gate rule, the race and the budget, recomputed from the saved state
    n_gate, over = 0, []
    for (t, a, s), fr in RACE.items():
        st = ST[(t, a, s)]
        sub = [100 * sum(f0(x) for x in v) / len(v) for v in st["prog_candidate_val_subscores"]]
        if any(len(v) != N_SEARCH for v in st["prog_candidate_val_subscores"]):
            sys.exit(f"stop: {t} {a} s{s}: a candidate has no score on all {N_SEARCH} search examples")
        fin = fr["finalists"]
        if [r["search_score"] for r in fin] != sorted(sub, reverse=True)[:len(fin)] or len(fin) != 5:
            sys.exit(f"stop: {t} {a} s{s}: the finalists are not the top 5 on the search examples")
        for r in fin:
            full = (N_SEARCH * r["search_score"] + (N_VAL - N_SEARCH) * r["held_out_score"]) / N_VAL
            if abs(full - r["full_val_score"]) > 1e-6 or abs(sub[r["program_idx"]] - r["search_score"]) > 1e-6:
                sys.exit(f"stop: {t} {a} s{s}: finalist {r['program_idx']} has inconsistent scores")
        if fr["returned_program_idx"] != max(fin, key=lambda r: r["full_val_score"])["program_idx"]:
            sys.exit(f"stop: {t} {a} s{s}: the returned program is not the best finalist on all 300")
        over.append(st["total_num_evals"] - BUDGET[t])
        if over[-1] < 0 or fr["total_num_evals"] != st["total_num_evals"]:
            sys.exit(f"stop: {t} {a} s{s}: metric calls {st['total_num_evals']} against a budget of {BUDGET[t]}")
        for x in st["full_program_trace"]:
            if "val_gate" not in x:
                continue
            g = x["val_gate"]
            n_gate += 1
            trails = g["child_gate_score"] < g["parent_gate_score"] - 1
            if (abs(sum(f0(v) for v in g["child_gate_subscores"]) - g["child_gate_score"]) > 1e-6 or len(g["child_gate_subscores"]) != 50
                    or g["trails"] != trails or g["cut"] != (trails and g["unanswered"] == 0) or (a == "GEPA-SubVal100")):
                sys.exit(f"stop: {t} {a} s{s}: gate decision of iteration {x['i']} does not follow the rule")
    num["checks"] = {"gate_decisions": n_gate, "overshoot_min": min(over), "overshoot_max": max(over)}
    p(f"\nChecks from the saved state, all passed: in the {len(RACE)} extension runs the finalists are the top 5 on the 100 "
      f"search examples, the returned program is the best finalist on all 300, and the metric calls are {min(over)} to "
      f"{max(over)} above the budget; the {n_gate} gate decisions follow the rule (discard if more than 1 example behind "
      f"the parent and no example unanswered).")
    p("\n### The search (descriptive; mean over the three seeds, range in brackets)\n")
    p("Metric calls: all calls counted against the budget. Val calls: full val passes of candidates (300 examples in "
      "GEPA, 100 in E1 and E2), gate passes of discarded children (50) and the final race (1,000). The rest are the "
      "three-example tests of parent and child.\n")
    p("| Task | Method | iterations | proposals | candidates (with the seed program) | gate decisions | discarded | metric calls | of which val | share on val |")
    p("|---|---|---|---|---|---|---|---|---|---|")

    def rng_text(v, fmt="{:.0f}"):
        return f"{fmt.format(statistics.mean(v))} [{fmt.format(min(v))}, {fmt.format(max(v))}]"

    for t in TASKS:
        for a in ARMS:
            ls = [log_stats(run_dir(root, t, a, s)) for s in SEEDS]
            cand = [len(ST[(t, a, s)]["prog_candidate_val_subscores"]) for s in SEEDS]
            calls = [ST[(t, a, s)]["total_num_evals"] for s in SEEDS]
            if a == "GEPA":
                val = [N_VAL * c for c in cand]
            else:
                val = [N_SEARCH * c + 50 * l["cut"] + RACE_CALLS for c, l in zip(cand, ls)]
            for s, l, c in zip(SEEDS, ls, cand):
                if l["accepted"] != c - 1:
                    sys.exit(f"stop: {t} {a} s{s}: run log has {l['accepted']} accepted candidates, the state {c - 1}")
                st_gate = [x["val_gate"] for x in ST[(t, a, s)]["full_program_trace"] if "val_gate" in x]
                if len(st_gate) != l["gate"] or sum(g["cut"] for g in st_gate) != l["cut"]:
                    sys.exit(f"stop: {t} {a} s{s}: gate decisions in the run log and in the state differ")
            share = [100 * v / c for v, c in zip(val, calls)]
            num["search"].append({"task": t, "method": NAME[a], "per_seed": ls, "candidates": cand, "metric_calls": calls, "val_calls": val})
            gd = "–" if a != "GEPA-SubVal100-Gate" else rng_text([l["gate"] for l in ls])
            gc = "–" if a != "GEPA-SubVal100-Gate" else f"{rng_text([l['cut'] for l in ls])} ({100 * sum(l['cut'] for l in ls) / sum(l['gate'] for l in ls):.0f} %)"
            p(f"| {t} | {NAME[a]} | {rng_text([l['iterations'] for l in ls])} | {rng_text([l['proposals'] for l in ls])} | {rng_text(cand)} | "
              f"{gd} | {gc} | {rng_text(calls)} | {rng_text(val)} | {statistics.mean(share):.0f} % |")
    kept = sum(l["guard_kept"] for x in num["search"] for l in x["per_seed"])
    unans = sum(g["unanswered"] for k, st in ST.items() for x in st["full_program_trace"] if "val_gate" in x for g in [x["val_gate"]])
    num["guard_kept"], num["gate_unanswered"] = kept, unans
    p(f"\nGate passes in which the server left an example unanswered: {unans}. Children the guard kept: {kept}.\n")

    # ---------- descriptive: validation scores of the returned program
    p("### The returned program on the val set (descriptive)\n")
    p("Seen = the seed's 100 search examples; other = the other 200 val examples. GEPA searched and selected on all "
      "300, so both columns are selection-biased for GEPA. E1 and E2 searched on the 100 and picked the best of 5 "
      "finalists on all 300, so the 200 are only lightly selected on. The test score is the unbiased comparison.\n")
    p("| Task | Method | seen 100 (per seed) | mean | other 200 (per seed) | mean | test mean |")
    p("|---|---|---|---|---|---|---|")
    RET = {}
    for t in TASKS:
        for a in ARMS:
            seen, other = [], []
            for s in SEEDS:
                st = ST[(t, a, s)]
                ids, rest = search_ids(s)
                if a == "GEPA":
                    sc = st["per_program_tracked_scores"]
                    best = max(range(len(sc)), key=lambda i: (sc[i], -i))
                    v = [f0(x) for x in st["prog_candidate_val_subscores"][best]]
                    seen.append(100 * sum(v[i] for i in ids) / len(ids))
                    other.append(100 * sum(v[i] for i in rest) / len(rest))
                else:
                    fr = RACE[(t, a, s)]
                    if fr["search_val_ids"] != ids:
                        sys.exit(f"stop: {t} {a} s{s}: the search ids are not the seeded draw")
                    w = next(r for r in fr["finalists"] if r["program_idx"] == fr["returned_program_idx"])
                    seen.append(w["search_score"])
                    other.append(w["held_out_score"])
                RET[(t, a, s)] = (seen[-1], other[-1])
            num["validation"].append({"task": t, "method": NAME[a], "seen100": seen, "other200": other})
            p(f"| {t} | {NAME[a]} | {' / '.join(f'{x:.1f}' for x in seen)} | {statistics.mean(seen):.2f} | "
              f"{' / '.join(f'{x:.1f}' for x in other)} | {statistics.mean(other):.2f} | {statistics.mean(REP[(t, a, s)] for s in SEEDS):.2f} |")

    # ---------- descriptive: best score on the search examples against metric calls
    p("\n### Best score on the seed's 100 search examples against metric calls (descriptive; mean over the seeds)\n")
    p("For every arm: the best score on these 100 examples among the candidates found so far. GEPA scored these "
      "examples as part of its 300, but did not select on them alone, and E1 and E2 did, so this comparison favours "
      "E1 and E2. The maximum over more candidates of a 100-example score is also higher by chance alone.\n")
    p("| Task | Method | " + " | ".join(f"{c:,} calls" for c in CHECKPOINTS) + " | end of search | calls to reach GEPA's final score (per seed) |")
    p("|---|---|" + "---|" * (len(CHECKPOINTS) + 2))
    for t in TASKS:
        target = {}
        for a in ARMS:
            curve_rows, ends, reach = [], [], []
            for s in SEEDS:
                st = ST[(t, a, s)]
                ids, _ = search_ids(s)
                V = st["prog_candidate_val_subscores"]
                if a == "GEPA":
                    sub = [100 * sum(f0(v[i]) for i in ids) / len(ids) for v in V]
                else:
                    sub = [100 * sum(f0(x) for x in v) / len(v) for v in V]
                calls = st["num_metric_calls_by_discovery"]
                if len(calls) != len(V):
                    sys.exit(f"stop: {t} {a} s{s}: discovery calls and candidates differ in number")
                pts = sorted(zip(calls, sub))
                row = []
                for c in CHECKPOINTS:
                    got = [x for k, x in pts if k <= c]
                    row.append(max(got) if got else None)
                curve_rows.append(row)
                ends.append(max(sub))
                if a == "GEPA":
                    target[s] = RET[(t, a, s)][0]
                reach.append(next((k for k, x in pts if x >= target[s] - 1e-9), None))
            mean_row = [statistics.mean(r[j] for r in curve_rows) for j in range(len(CHECKPOINTS))]
            num["curves"].append({"task": t, "method": NAME[a], "checkpoints": CHECKPOINTS, "per_seed": curve_rows, "end": ends, "calls_to_reach_gepa_final": reach})
            reach_txt = " / ".join("never" if k is None else f"{k:,}" for k in reach)
            p(f"| {t} | {NAME[a]} | " + " | ".join(f"{x:.1f}" for x in mean_row) + f" | {statistics.mean(ends):.1f} | {reach_txt} |")
    p("\nGEPA's final score = the score of the program GEPA returned on these 100 examples "
      "(not the best 100-example score among its candidates).\n")

    # ---------- descriptive: the race
    p("### The final race (descriptive)\n")
    p("| Task | Method | seed | finalists: seen 100 | finalists: other 200 | returned | best on the 100 | their other-200 scores (returned / best on the 100) |")
    p("|---|---|---|---|---|---|---|---|")
    changed, gaps, opt = 0, [], []
    for t in TASKS:
        for a in ARMS[1:]:
            for s in SEEDS:
                fr = RACE[(t, a, s)]
                fin = fr["finalists"]
                w = next(r for r in fin if r["program_idx"] == fr["returned_program_idx"])
                b = next(r for r in fin if r["program_idx"] == fr["search_subset_best_program_idx"])
                if w is not b:
                    changed += 1
                    gaps.append(w["held_out_score"] - b["held_out_score"])
                opt += [r["search_score"] - r["held_out_score"] for r in fin]
                num["race"].append({"task": t, "method": NAME[a], "seed": s, "returned": w["program_idx"], "subset_best": b["program_idx"],
                                    "finalists_seen": [r["search_score"] for r in fin], "finalists_other": [r["held_out_score"] for r in fin]})
                p(f"| {t} | {NAME[a]} | {s} | {' / '.join(f'{r['search_score']:.0f}' for r in fin)} | {' / '.join(f'{r['held_out_score']:.1f}' for r in fin)} | "
                  f"{w['program_idx']} | {b['program_idx']} | {w['held_out_score']:.1f} / {b['held_out_score']:.1f} |")
    num["race_summary"] = {"changed": changed, "runs": len(RACE), "mean_gain_other200_when_changed": statistics.mean(gaps) if gaps else None,
                           "mean_seen_minus_other_finalists": statistics.mean(opt), "finalists": len(opt)}
    p(f"\nThe race returned another program than the best on the 100 in {changed} of {len(RACE)} runs; there the returned "
      f"program is {statistics.mean(gaps):+.2f} points better on the other 200 on average (it was chosen with these 200, "
      f"so part of this is selection). Over all {len(opt)} finalists the score on the 100 search examples is "
      f"{statistics.mean(opt):+.2f} points above the score on the other 200.\n")

    # ---------- cost
    p("### Cost per job (two runs sharing one H100; descriptive)\n")
    p("GEPA jobs: GEPA and GEPA+Merge of one task and seed. Extension jobs: E1 and E2 of one task and seed.\n")
    p("| Task | Jobs | hours (per seed) | mean | SBU, mean | generated tokens, mean (M) | requests, mean |")
    p("|---|---|---|---|---|---|---|")
    for t in TASKS:
        base = None
        for kind, label in (("GEPA", "GEPA + GEPA+Merge"), ("EXT", "E1 + E2")):
            js = [job_report(root, JOBS[kind][(t, s)]) for s in SEEDS]
            h = statistics.mean(j["hours"] for j in js)
            num["cost"].append({"task": t, "jobs": label, "per_seed": js})
            ratio = "" if base is None else f" ({h / base:.2f}× GEPA)"
            base = base or h
            p(f"| {t} | {label} | {' / '.join(f'{j['hours']:.1f}' for j in js)} | {h:.1f}{ratio} | {statistics.mean(j['sbu'] for j in js):.0f} | "
              f"{statistics.mean(j['mtok'] for j in js):.1f} | {statistics.mean(j['requests'] for j in js):,.0f} |")
    text = "\n".join(L) + "\n"
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(num, f, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    main(args[0], out)
