# pupa_analysis.py - PUPA with gpt-oss-120b (plan 2.10 and its addenda of 2026-11-12 19:24,
# 2026-11-15 12:58 and 2026-11-15 20:45).
#
# 1. rebuilds the 221 per-example test scores of each run (the logger files test copies of
#    train/val rows under train/val, F108) and checks them against the reported score
# 2. mean +- SD per cell, the paper's three PUPA claims, with and without the run marked ‡
# 3. paired per-example differences, bootstrap over the test positions
# 4. exploratory: the same without the test copies of train/val rows / the query overlaps
#
# usage: python -I pupa_analysis.py <results/pupa_gptoss> <pupa_split_overlap.json> [out.json]
# Only the standard library is used.

import gzip
import json
import os
import random
import statistics
import sys
from collections import defaultdict

root, overlap_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else None

METHODS = ["Baseline", "MIPROv2-Heavy", "GEPA", "GEPA-MERGE"]
NAMES = {"Baseline": "Baseline", "MIPROv2-Heavy": "MIPROv2", "GEPA": "GEPA", "GEPA-MERGE": "GEPA+Merge"}
SEEDS = [0, 1, 2]
DAGGER = {("MIPROv2-Heavy", 0)}  # kept, empty gpt-oss answers above 1 % (addendum 2026-11-15 12:58)
N = 221
# the paper's PUPA claims (authors' seed 0: Baseline 80.82, MIPROv2 81.55, GEPA 91.85, GEPA+Merge 86.26)
CLAIMS = [("GEPA", "Baseline", 11.03), ("GEPA", "MIPROv2-Heavy", 10.30), ("GEPA-MERGE", "GEPA", -5.59)]
PAIRS = [("GEPA", "Baseline"), ("GEPA", "MIPROv2-Heavy"), ("GEPA-MERGE", "GEPA"),
         ("GEPA-MERGE", "Baseline"), ("GEPA-MERGE", "MIPROv2-Heavy"), ("MIPROv2-Heavy", "Baseline")]
B = 10000

overlap = json.load(open(overlap_path))
mapping = overlap["logged_per_seed"]  # seed -> 221 x [split, idx]


def run_dir(m, s):
    return os.path.join(root, "seed_%d" % s, "Papillon_PAPILLON_%s_qwen3-8b" % m)


def reported(m, s):
    with open(os.path.join(run_dir(m, s), "evaluation_results", "evaluation_result.txt")) as f:
        return float(f.read().splitlines()[1].split(",")[0])


def load_lines(d):
    lines = []
    for split in ("train", "val", "test"):
        p = os.path.join(d, "metric_logs", split + ".jsonl.gz")
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    lines.append((r["step_counter"], split, r["idx_in_split"], r["metric_output"]))
    lines.sort(key=lambda x: x[0])
    return lines


def rebuild(m, s):
    lines = load_lines(run_dir(m, s))
    steps = [x[0] for x in lines]
    gaps = steps != list(range(1, len(steps) + 1))
    want = defaultdict(list)  # (split, idx) -> test positions, in position order
    for pos, (split, idx) in enumerate(mapping[str(s)]):
        want[(split, idx)].append(pos)
    E = 0
    while True:
        block = lines[-(N + E):]
        got = defaultdict(list)
        for _, split, idx, val in block:
            got[(split, idx)].append(val)
        if all(len(got[k]) >= len(v) for k, v in want.items()):
            break
        E += 1
        if N + E > len(lines):
            return None, {"error": "not every test position found"}
    foreign = sum(len(v) for k, v in got.items() if k not in want)
    extra = sum(len(got[k]) - len(v) for k, v in want.items())
    # where do the lines that are not test copies come from? a late answer of the optimizer's
    # last evaluation (re-sent requests, F58/F71) is a train/val example already logged before the block
    before = set((sp, ix) for _, sp, ix, _ in lines[:-(N + E)])
    foreign_keys = [k for k, v in got.items() if k not in want for _ in v]
    foreign_late = sum(1 for k in foreign_keys if k[0] != "test" and k in before)
    # extra lines of mapped train/val copies could also be such late answers
    extra_copy_keys = [k for k, v in want.items() if k[0] != "test" and len(got[k]) > len(v)]
    scores = [None] * N
    nones = 0
    for k, poss in want.items():
        for pos, val in zip(poss, got[k]):  # first lines in step order count (F90)
            if val is None:
                nones += 1
                val = 0.0
            scores[pos] = float(val)
    differing = sum(1 for k, poss in want.items() if len(set(got[k][:len(poss)])) > 1)
    mean = round(100 * sum(scores) / N, 2)
    info = {"lines": len(lines), "step_gaps": gaps, "E": E, "foreign_lines": foreign, "extra_lines": extra,
            "foreign_lines_late_optimizer_answers": foreign_late,
            "extra_lines_of_train_val_copies": [[k[0], k[1], got[k]] for k in extra_copy_keys],
            "none_scores": nones, "copies_with_different_scores": differing,
            "rebuilt": mean, "reported": reported(m, s)}
    info["ok"] = (not gaps) and foreign == 0 and abs(mean - info["reported"]) < 1e-9
    # changed check (plan 2.10, deviation of 2026-11-15 22:08): lines that are not test copies are
    # allowed if they are late answers of the optimizer, and none of them may be counted for a test example
    info["ok_changed"] = ((not gaps) and abs(mean - info["reported"]) < 1e-9
                          and foreign == foreign_late and not extra_copy_keys)
    return scores, info


def msd(v):
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else float("nan"))


def seeds_for(*ms, dagger=True):
    return [s for s in SEEDS if dagger or not any((m, s) in DAGGER for m in ms)]


def boot(dpos):
    # percentile interval: the 250th and the 9,750th of 10,000 sorted resample means
    rng = random.Random(0)
    n = len(dpos)
    means = []
    for _ in range(B):
        means.append(sum(dpos[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[249], means[9749]


def paired(a, b, positions, dagger):
    ss = seeds_for(a, b, dagger=dagger)
    dpos = [100 * statistics.mean(S[(a, s)][p] - S[(b, s)][p] for s in ss) for p in positions]
    lo, hi = boot(dpos)
    return {"seeds": ss, "positions": len(positions), "mean_d": statistics.mean(dpos), "ci95": [lo, hi],
            "share_A_better": sum(1 for x in dpos if x > 0) / len(dpos),
            "share_B_better": sum(1 for x in dpos if x < 0) / len(dpos)}


# 1. rebuild and check
S, INFO = {}, {}
all_ok = True
all_ok_changed = True
print("1. per-example rebuild (must reproduce the reported score)")
for s in SEEDS:
    for m in METHODS:
        sc, info = rebuild(m, s)
        INFO["%s seed %d" % (NAMES[m], s)] = info
        if sc is None:
            all_ok = False
            print("  %-10s seed %d: FAILED, %s" % (NAMES[m], s, info["error"]))
            continue
        S[(m, s)] = sc
        all_ok = all_ok and info["ok"]
        all_ok_changed = all_ok_changed and info["ok_changed"]
        print("  %-10s seed %d: lines %d, block 221+%d, foreign %d (late optimizer answers %d), extra %d"
              " (of train/val copies: %s), None %d, copies with different scores %d,"
              " rebuilt %.2f, reported %.2f -> %s" % (NAMES[m], s, info["lines"], info["E"], info["foreign_lines"],
                                                      info["foreign_lines_late_optimizer_answers"],
                                                      info["extra_lines"], info["extra_lines_of_train_val_copies"],
                                                      info["none_scores"],
                                                      info["copies_with_different_scores"], info["rebuilt"],
                                                      info["reported"], "OK" if info["ok"] else "MISMATCH"))
n_strict = sum(1 for v in INFO.values() if not v.get("ok"))
n_changed = sum(1 for v in INFO.values() if not v.get("ok_changed"))
print("  check fixed at 20:45: failed in %d of 12 runs; changed check (deviation of 22:08): failed in %d of 12 runs"
      % (n_strict, n_changed))
res = {"rebuild": INFO, "rebuild_ok_fixed_rule": all_ok, "rebuild_ok_changed_rule": all_ok_changed}

# 2. cells and claims (reported scores, the table values)
print("\n2. mean +- SD over seeds (reported scores); without ‡ = MIPROv2 seed 0 left out")
cells = {}
for m in METHODS:
    for dag in (True, False):
        v = [reported(m, s) for s in seeds_for(m, dagger=dag)]
        mu, sd = msd(v)
        cells["%s%s" % (NAMES[m], "" if dag else " without ‡")] = {"values": v, "mean": mu, "sd": sd}
    print("  %-10s %s  mean %.2f +- %.2f" % (NAMES[m], " / ".join("%.2f" % reported(m, s) for s in SEEDS),
                                            cells[NAMES[m]]["mean"], cells[NAMES[m]]["sd"])
          + ("   without ‡: %.2f +- %.2f" % (cells[NAMES[m] + " without ‡"]["mean"], cells[NAMES[m] + " without ‡"]["sd"])
             if m == "MIPROv2-Heavy" else ""))
res["cells"] = cells

print("\n   the paper's claims: mean of the per-seed differences +- their SD; holds in direction if the sign matches")
claims = []
for a, b, paper in CLAIMS:
    for dag in (True, False):
        ss = seeds_for(a, b, dagger=dag)
        d = [reported(a, s) - reported(b, s) for s in ss]
        mu, sd = msd(d)
        holds = (mu > 0) == (paper > 0)
        claims.append({"claim": "%s - %s" % (NAMES[a], NAMES[b]), "with_dagger": dag, "seeds": ss, "per_seed": d,
                       "mean": mu, "sd": sd, "paper": paper, "holds_in_direction": holds})
        if dag or a == "MIPROv2-Heavy" or b == "MIPROv2-Heavy":
            print("  %-22s%s per seed %s  mean %+.2f +- %.2f   paper %+.2f   %s" % (
                "%s - %s" % (NAMES[a], NAMES[b]), "" if dag else " (no ‡)",
                " / ".join("%+.2f" % x for x in d), mu, sd, paper, "holds" if holds else "does not hold"))
res["claims"] = claims

if not all_ok_changed:
    print("\nrebuild check failed, no per-example analysis (plan 2.10, deviation of 2026-11-15 22:08)")
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)
    sys.exit(0)
if not all_ok:
    print("\nthe check fixed at 20:45 failed; the paired analysis below runs under the changed check"
          " (plan 2.10, deviation of 2026-11-15 22:08, user's OK)")

# 3. paired per-example analysis (fixed) and 4. exploratory subsets
kinds = overlap["test"]
subsets = {
    "all 221 test examples": list(range(N)),
    "exploratory: without the 25 copies of train/val rows": [r["test_pos"] for r in kinds if r["logged_as"] == "test"],
    "exploratory: without the 43 query overlaps": [r["test_pos"] for r in kinds
                                                   if not (r["query_in_train"] or r["query_in_val"])],
}
res["paired"] = {}
res["subset_cells"] = {}
for name, pos in subsets.items():
    print("\n3/4. %s (%d positions)" % (name, len(pos)))
    if name != "all 221 test examples":
        sc = {}
        for m in METHODS:
            v = [100 * statistics.mean(S[(m, s)][p] for p in pos) for s in SEEDS]
            mu, sd = msd(v)
            sc[NAMES[m]] = {"per_seed": v, "mean": mu, "sd": sd}
            print("  cell %-10s %s  mean %.2f +- %.2f" % (NAMES[m], " / ".join("%.2f" % x for x in v), mu, sd))
        res["subset_cells"][name] = sc
    out = []
    for a, b in PAIRS:
        for dag in (True, False):
            if not dag and "MIPROv2-Heavy" not in (a, b):
                continue
            r = paired(a, b, pos, dag)
            r.update({"pair": "%s - %s" % (NAMES[a], NAMES[b]), "with_dagger": dag})
            out.append(r)
            print("  %-22s%s mean d %+.2f  95%% CI [%+.2f, %+.2f]  A better on %.0f %%, B better on %.0f %% of positions" % (
                r["pair"], "" if dag else " (no ‡)", r["mean_d"], r["ci95"][0], r["ci95"][1],
                100 * r["share_A_better"], 100 * r["share_B_better"]))
    res["paired"][name] = out

if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
