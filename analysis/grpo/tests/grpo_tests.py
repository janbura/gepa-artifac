"""
grpo_tests.py - statistical checks of the four GRPO (Qwen3-8B, DSPy + Arbor) runs in the
GEPA paper artifact (arXiv 2507.19457), seed_0.

usage:  python -I grpo_tests.py <seed_0 folder> <output folder> [B]

All data files are read as text / JSON lines only (nothing from the data folder is
executed, imported or unpickled; *.pkl / *.bin are never opened).

ANALYSIS PLAN (written 2026-11-14, before running T1-T4; only the log structure and the
official per-pass scores printed in run_log_stderr.txt had been looked at)
------------------------------------------------------------------------------------
Pass reconstruction
  R1  If test.jsonl exists, val/train lines with step_counter >= first test step belong to
      the final test evaluation and are removed from the validation data.
  R2  val.jsonl is split into passes at jumps of the global step_counter > 100 (between two
      passes there are ~900+ training metric calls; inside a pass jumps are 1-4).
      Exactly 26 passes are expected.
  R3  The logger files some val examples under "train" (they recur in every pass with the
      same train idx).  k = n_val - (number of val examples seen under "val") such train
      idx are taken: the k train idx that occur inside the most passes.  All other
      train lines inside a pass are late training calls and are ignored.
  R4  Example key = (split label, idx).  Multiplicity m(key) = most common number of lines
      of that key per pass (2 when two val examples share one idx).  In each pass the
      first m lines (file order) are the outcomes; extra lines are straggler duplicates
      and are ignored; missing lines are errors and score 0 (DSPy Evaluate does the same).
  R5  Check: 100 * pass sum / n_val must equal the official score in run_log_stderr.txt
      (2 decimals) for all 26 passes, and the examples must add up to n_val.
Tests (B simulations / permutations / bootstrap resamples each, fixed seeds)
  T1  Selection effect.  G = max(pass 2..26) - pass 1.  Null "no learning": every example
      keeps its own outcome distribution over the 26 passes (binary: Bernoulli(p_i) with
      p_i = mean over 26 passes; fractional: resample its own 26 observed scores).
      Simulate 26 passes B times; report E[G], 95th percentile, share G_sim >= G_obs.
      Extra: observed SD of the 26 pass scores vs the null SD.
  T2  Trend.  (a) per example d_i = mean(passes 22-26) - mean(passes 1-5); effect in points
      = 100*mean(d); two-sided paired sign-flip permutation test and exact sign test.
      (b) OLS slope of pass score on pass index; permutation of the pass order; one-sided
      (upward) and two-sided p; effect = slope * 25 (fitted change over training).
  T3  Test set, GRPO vs Baseline, only where both runs have per-example test logs.  Pairs
      are formed by key (split label, idx); test examples missing in a run are errors = 0
      (reproduces the official test scores, checked).  Binary: exact McNemar.
      Fractional: paired bootstrap 95% CI (percentile) + paired sign-flip p.
      Sensitivity for PUPA: only examples logged under "test".
  T4  Binomial SE sqrt(p(1-p)/n) of one validation pass (p = mean over the 26 passes) and
      E[max of 25 iid N(0,1)] (numerical integral + simulation).
p-values from simulation are (count+1)/(B+1).  Nothing is rerun or tuned after seeing results.
"""
import sys
import os
import re
import json
import math
import random
import statistics
from collections import Counter, defaultdict

GAP = 100
SEED = 20261114

RUNS = [
    # name, GRPO folder, Baseline folder, n_val, binary scores?
    ("HotpotQA", "HotpotQABench_HotpotMultiHop_GRPO_qwen3-8b",
     "HotpotQABench_HotpotMultiHop_Baseline_qwen3-8b", 300, True),
    ("HoVer", "hoverBench_HoverMultiHop_GRPO_qwen3-8b",
     "hoverBench_HoverMultiHop_Baseline_qwen3-8b", 300, True),
    ("IFBench", "IFBench_IFBenchCoT2StageProgram_GRPO_qwen3-8b",
     "IFBench_IFBenchCoT2StageProgram_Baseline_qwen3-8b", 300, False),
    ("PUPA", "Papillon_PAPILLON_GRPO_qwen3-8b",
     "Papillon_PAPILLON_Baseline_qwen3-8b", 111, False),
]


# ----------------------------------------------------------------- reading the logs
def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def num(x):
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return float(x)


def official_val_scores(run_dir):
    pat = re.compile(r"validation set score (before training loop|after training step (\d+)/500): ([0-9.]+)")
    out = []
    with open(os.path.join(run_dir, "run_log_stderr.txt"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out.append((0 if m.group(2) is None else int(m.group(2)), float(m.group(3))))
    return out


def selected_checkpoint(run_dir):
    pat = re.compile(r"Updated the model .*model_checkpoint_(\d+)")
    found = None
    with open(os.path.join(run_dir, "run_log_stderr.txt"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                found = int(m.group(1))
    return found


def reconstruct_passes(run_dir, n_val):
    """Return (examples, info). examples: list of rows, one row per val example, 26 outcomes each."""
    logs = os.path.join(run_dir, "metric_logs")
    val = load_jsonl(os.path.join(logs, "val.jsonl"))
    train = load_jsonl(os.path.join(logs, "train.jsonl"))
    test = load_jsonl(os.path.join(logs, "test.jsonl"))
    info = {"val_lines_total": len(val)}
    test_start = min((t["step_counter"] for t in test), default=None)
    if test_start is not None:                                      # R1
        info["val_lines_in_test_phase"] = sum(1 for r in val if r["step_counter"] >= test_start)
        val = [r for r in val if r["step_counter"] < test_start]
        train = [t for t in train if t["step_counter"] < test_start]
    segs = [[val[0]]]                                                # R2
    for prev, cur in zip(val, val[1:]):
        if cur["step_counter"] - prev["step_counter"] > GAP:
            segs.append([])
        segs[-1].append(cur)
    info["n_passes"] = len(segs)
    # train lines inside each pass
    inside = []
    for seg in segs:
        lo, hi = seg[0]["step_counter"], seg[-1]["step_counter"]
        inside.append([t for t in train if lo <= t["step_counter"] <= hi])
    # per pass: key -> outcomes in file order (merged by step_counter)
    per_pass_val = []
    for seg in segs:
        d = defaultdict(list)
        for r in seg:
            d[("val", r["idx_in_split"])].append((r["step_counter"], num(r["metric_output"])))
        per_pass_val.append(d)
    # multiplicity of val keys (R4)
    def mode_mult(per_pass, key):
        c = Counter(len(p.get(key, [])) for p in per_pass)
        # most common count among passes where the key is present; ties -> smaller
        present = [(cnt, -m) for m, cnt in c.items() if m > 0]
        return -max(present)[1] if present else 0
    val_keys = sorted({k for p in per_pass_val for k in p})
    mult = {k: mode_mult(per_pass_val, k) for k in val_keys}
    n_seen_val = sum(mult.values())
    k_train = n_val - n_seen_val                                     # R3
    rec = Counter()
    for ins in inside:
        for i in set(t["idx_in_split"] for t in ins):
            rec[i] += 1
    ranked = sorted(rec.items(), key=lambda x: (-x[1], x[0]))
    chosen = [i for i, _ in ranked[:max(k_train, 0)]]
    info["val_examples_seen_under_val"] = n_seen_val
    info["val_examples_filed_under_train"] = k_train
    info["train_idx_taken"] = [(i, rec[i]) for i in chosen]
    info["next_train_idx_not_taken"] = ranked[k_train:k_train + 3] if k_train >= 0 else ranked[:3]
    per_pass = []
    for d, ins in zip(per_pass_val, inside):
        dd = defaultdict(list, {k: list(v) for k, v in d.items()})
        for t in ins:
            if t["idx_in_split"] in chosen:
                dd[("train", t["idx_in_split"])].append((t["step_counter"], num(t["metric_output"])))
        per_pass.append(dd)
    for i in chosen:
        mult[("train", i)] = mode_mult(per_pass, ("train", i))
    # R3b (added 2026-11-14 after the first reconstruction check failed for PUPA in 12/26
    # passes): a val example filed under "train" can be the very first or very last call of
    # the pass, i.e. at step lo-1 or hi+1, just outside the window spanned by the val lines.
    # Only if such a key has fewer than m lines inside the window, the line at exactly
    # lo-1 / hi+1 with that idx is added.  (Lines further out are training rollouts.)
    by_step = {t["step_counter"]: t for t in train}
    filled = 0
    for seg, dd in zip(segs, per_pass):
        lo, hi = seg[0]["step_counter"], seg[-1]["step_counter"]
        for i in chosen:
            key = ("train", i)
            for s in (lo - 1, hi + 1):
                t = by_step.get(s)
                if t is not None and t["idx_in_split"] == i and len(dd.get(key, [])) < mult[key]:
                    dd[key].append((s, num(t["metric_output"])))
                    filled += 1
    info["edge_lines_added_R3b"] = filled
    # build example rows
    examples, keys = [], []
    extra_lines = 0
    missing = 0
    errmask = []
    for key in sorted(mult):
        m = mult[key]
        for j in range(m):
            row, erow = [], []
            for p in per_pass:
                v = sorted(p.get(key, []))             # file order == step order
                if j < len(v):
                    row.append(v[j][1])
                    erow.append(False)
                else:
                    row.append(0.0)
                    erow.append(True)
                    missing += 1
            examples.append(row)
            errmask.append(erow)
            keys.append((key[0], key[1], j))
    info["errors_per_pass"] = [sum(e[k] for e in errmask) for k in range(len(per_pass))]
    info["_errmask"] = errmask
    for p in per_pass:
        for key, v in p.items():
            extra_lines += max(0, len(v) - mult.get(key, 0))
    info["n_examples"] = len(examples)
    info["error_cells_scored_0"] = missing
    info["straggler_lines_ignored"] = extra_lines
    info["keys_with_multiplicity_2plus"] = [list(k) for k, m in mult.items() if m >= 2]
    return examples, keys, info


def pass_scores(examples, n_val):
    n_pass = len(examples[0])
    return [100.0 * sum(row[k] for row in examples) / n_val for k in range(n_pass)]


# ----------------------------------------------------------------- small stat helpers
def quantile(sorted_vals, q):
    # linear interpolation between order statistics (type 7)
    n = len(sorted_vals)
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def binom_two_sided_sign(k_small, n):
    # exact two-sided sign / McNemar p = min(1, 2 * P(X <= k_small)), X ~ Bin(n, 1/2)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k_small + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ols_slope(y):
    n = len(y)
    xm = (n - 1) / 2
    ym = sum(y) / n
    sxx = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (y[i] - ym) for i in range(n)) / sxx


# ----------------------------------------------------------------- T1
def t1_selection(examples, n_val, binary, B, seed):
    rng = random.Random(seed)
    obs = pass_scores(examples, n_val)
    g_obs = max(obs[1:]) - obs[0]
    n_pass = len(obs)
    if binary:
        # group examples by success count k (p = k / n_obs) -> binomial draws
        groups = Counter()
        for row in examples:
            groups[(sum(row), len(row))] += 1
        plan = [(cnt, s / L) for (s, L), cnt in groups.items() if 0 < s < L]
        const = sum(cnt for (s, L), cnt in groups.items() if s == L)  # always-correct examples
        def one_pass():
            tot = const
            for cnt, p in plan:
                tot += rng.binomialvariate(cnt, p)
            return 100.0 * tot / n_val
    else:
        # examples whose 26 observed scores are all equal are constant under the null
        const = sum(row[0] for row in examples if min(row) == max(row))
        rows = [row for row in examples if min(row) != max(row)]
        L = len(examples[0])
        def one_pass():
            rnd = rng.random
            return 100.0 * (const + sum(row[int(rnd() * L)] for row in rows)) / n_val
    gains, sds = [], []
    for _ in range(B):
        sim = [one_pass() for _ in range(n_pass)]
        gains.append(max(sim[1:]) - sim[0])
        sds.append(statistics.stdev(sim))
    gs = sorted(gains)
    hit = sum(1 for g in gains if g >= g_obs - 1e-9)
    sd_obs = statistics.stdev(obs)
    hit_sd = sum(1 for s in sds if s >= sd_obs - 1e-9)
    # analytic null SD of one pass: sqrt(sum var_i) / n
    var_sum = 0.0
    for row in examples:
        var_sum += statistics.pvariance(row)
    return {
        "observed_pass1": obs[0], "observed_best_2_26": max(obs[1:]),
        "best_pass_index_1based": 1 + obs.index(max(obs)),
        "observed_gain": g_obs,
        "null_mean_gain": statistics.fmean(gains),
        "null_median_gain": quantile(gs, 0.5),
        "null_p95_gain": quantile(gs, 0.95),
        "share_sim_gain_ge_obs": hit / B,
        "p_value_gain": (hit + 1) / (B + 1),
        "observed_sd_of_26_passes": sd_obs,
        "null_mean_sd_of_26_passes": statistics.fmean(sds),
        "p_value_sd_ge_obs": (hit_sd + 1) / (B + 1),
        "null_sd_one_pass_analytic": 100.0 * math.sqrt(var_sum) / n_val,
        "B": B, "seed": seed,
    }


# ----------------------------------------------------------------- T2
def t2_trend(examples, n_val, B, seed):
    rng = random.Random(seed)
    obs = pass_scores(examples, n_val)
    # (a) first 5 vs last 5, per example
    d = [statistics.fmean(row[-5:]) - statistics.fmean(row[:5]) for row in examples]
    # effect in points of the official score (errors = 0, divide by n_val)
    eff = 100.0 * sum(d) / n_val
    tot = abs(sum(d))
    nz = [x for x in d if x != 0.0]
    hit = 0
    for _ in range(B):
        s = 0.0
        for x in nz:
            s += x if rng.random() < 0.5 else -x
        if abs(s) >= tot - 1e-9:
            hit += 1
    npos = sum(1 for x in d if x > 0)
    nneg = sum(1 for x in d if x < 0)
    p_sign = binom_two_sided_sign(min(npos, nneg), npos + nneg)
    # (b) pass-order permutation of the OLS slope
    slope = ols_slope(obs)
    y = list(obs)
    up = two = 0
    for _ in range(B):
        rng.shuffle(y)
        s = ols_slope(y)
        if s >= slope - 1e-12:
            up += 1
        if abs(s) >= abs(slope) - 1e-12:
            two += 1
    return {
        "mean_first5": statistics.fmean(obs[:5]), "mean_last5": statistics.fmean(obs[-5:]),
        "effect_last5_minus_first5_points": eff,
        "p_signflip_two_sided": (hit + 1) / (B + 1),
        "n_examples_up": npos, "n_examples_down": nneg, "p_sign_test": p_sign,
        "slope_points_per_pass": slope, "fitted_change_25_intervals": 25 * slope,
        "p_perm_slope_one_sided_up": (up + 1) / (B + 1),
        "p_perm_slope_two_sided": (two + 1) / (B + 1),
        "B": B, "seed": seed,
    }


# ----------------------------------------------------------------- S1 (post-hoc)
def s1_errors_excluded(examples, errmask, B, seed):
    """POST-HOC sensitivity, added 2026-11-14 AFTER seeing the T1/T2 results: failed calls
    (no metric line) are left out instead of being scored 0.  Not part of the plan above."""
    rng = random.Random(seed)
    n_pass = len(examples[0])
    obs = []
    for k in range(n_pass):
        vals = [row[k] for row, e in zip(examples, errmask) if not e[k]]
        obs.append(100.0 * sum(vals) / len(vals))
    d = []
    for row, e in zip(examples, errmask):
        a = [row[k] for k in range(5) if not e[k]]
        b = [row[k] for k in range(n_pass - 5, n_pass) if not e[k]]
        if a and b:
            d.append(statistics.fmean(b) - statistics.fmean(a))
    eff = 100.0 * statistics.fmean(d)
    tot = abs(sum(d))
    nz = [x for x in d if x != 0.0]
    hit = 0
    for _ in range(B):
        s = 0.0
        for x in nz:
            s += x if rng.random() < 0.5 else -x
        if abs(s) >= tot - 1e-9:
            hit += 1
    slope = ols_slope(obs)
    y = list(obs)
    up = 0
    for _ in range(B):
        rng.shuffle(y)
        if ols_slope(y) >= slope - 1e-12:
            up += 1
    return {"pass_scores_errors_excluded": [round(x, 3) for x in obs],
            "gain_best_2_26_minus_pass1": max(obs[1:]) - obs[0],
            "effect_last5_minus_first5_points": eff, "n_examples_used": len(d),
            "p_signflip_two_sided": (hit + 1) / (B + 1),
            "fitted_change_25_intervals": 25 * slope,
            "p_perm_slope_one_sided_up": (up + 1) / (B + 1), "B": B, "seed": seed}


# ----------------------------------------------------------------- T3
def test_outcomes(run_dir, is_grpo):
    """key -> list of outcomes of the final test evaluation (file order)."""
    logs = os.path.join(run_dir, "metric_logs")
    test = load_jsonl(os.path.join(logs, "test.jsonl"))
    if not test:
        return None, None
    start = min(t["step_counter"] for t in test)
    out = defaultdict(list)
    for split in ("test", "val", "train"):
        for r in load_jsonl(os.path.join(logs, split + ".jsonl")):
            if split != "test" and r["step_counter"] < start:
                continue   # validation / training calls, not part of the test evaluation
            out[(split, r["idx_in_split"])].append((r["step_counter"], num(r["metric_output"])))
    return out, start


def official_test_score(run_dir):
    with open(os.path.join(run_dir, "evaluation_results", "evaluation_result.txt"), "r",
              encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]
    return float(lines[1].split(",")[0])


def t3_paired(grpo_dir, base_dir, n_test, binary, B, seed, only_test_split=False):
    g, _ = test_outcomes(grpo_dir, True)
    b, _ = test_outcomes(base_dir, False)
    if g is None or b is None:
        return None
    keys = sorted(set(g) | set(b))
    pairs = []
    info = {"keys_only_in_grpo": [], "keys_only_in_base": [], "multi_line_keys": []}
    for k in keys:
        gv = [v for _, v in sorted(g.get(k, []))]
        bv = [v for _, v in sorted(b.get(k, []))]
        m = max(len(gv), len(bv))
        if m > 1:
            info["multi_line_keys"].append([k[0], k[1], len(gv), len(bv)])
        if not gv:
            info["keys_only_in_base"].append(list(k))
        if not bv:
            info["keys_only_in_grpo"].append(list(k))
        for j in range(m):
            pairs.append((k, gv[j] if j < len(gv) else 0.0, bv[j] if j < len(bv) else 0.0))
    n_logged = len(pairs)
    # examples with no line in either run (errors in both): (0, 0) pairs
    n_missing_both = n_test - n_logged
    if only_test_split:
        pairs = [p for p in pairs if p[0][0] == "test"]
        n_missing_both = 0
    gs = [p[1] for p in pairs] + [0.0] * max(n_missing_both, 0)
    bs = [p[2] for p in pairs] + [0.0] * max(n_missing_both, 0)
    n = len(gs)
    d = [x - y for x, y in zip(gs, bs)]
    res = {"n": n, "n_logged_pairs": n_logged, "n_errors_both_scored_0": max(n_missing_both, 0),
           "grpo_mean": 100 * statistics.fmean(gs), "base_mean": 100 * statistics.fmean(bs),
           "diff_points": 100 * statistics.fmean(d), "info": info}
    rng = random.Random(seed)
    if binary:
        bb = sum(1 for x, y in zip(gs, bs) if x == 1.0 and y == 0.0)
        cc = sum(1 for x, y in zip(gs, bs) if x == 0.0 and y == 1.0)
        res["discordant_grpo_only"] = bb
        res["discordant_base_only"] = cc
        res["p_mcnemar_exact"] = binom_two_sided_sign(min(bb, cc), bb + cc)
    # paired bootstrap CI of the mean difference
    boots = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots.append(100 * s / n)
    boots.sort()
    res["boot_ci95_points"] = [quantile(boots, 0.025), quantile(boots, 0.975)]
    # paired sign-flip permutation p (two-sided)
    tot = abs(sum(d))
    nz = [x for x in d if x != 0.0]
    hit = 0
    for _ in range(B):
        s = 0.0
        for x in nz:
            s += x if rng.random() < 0.5 else -x
        if abs(s) >= tot - 1e-9:
            hit += 1
    res["p_signflip_two_sided"] = (hit + 1) / (B + 1)
    res["B"] = B
    res["seed"] = seed
    return res


# ----------------------------------------------------------------- T4
def expected_max_normal(k=25):
    phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    a, b, n = -12.0, 12.0, 24000          # Simpson's rule
    h = (b - a) / n
    f = lambda x: x * k * phi(x) * Phi(x) ** (k - 1)
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def expected_max_normal_sim(k=25, B=100000, seed=SEED + 4):
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(B):
        tot += max(rng.gauss(0.0, 1.0) for _ in range(k))
    return tot / B


# ----------------------------------------------------------------- main
def main():
    base = sys.argv[1]
    outdir = sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    os.makedirs(outdir, exist_ok=True)
    results = {"B": B, "seed": SEED, "runs": {}}
    emax = expected_max_normal(25)
    emax_sim = expected_max_normal_sim(25)
    results["T4_expected_max_25_normal"] = {"integral": emax, "simulation_100k": emax_sim}
    print(f"T4: E[max of 25 iid N(0,1)] = {emax:.4f} (integral), {emax_sim:.4f} (simulation, 100k)")
    for j, (name, grpo, basel, n_val, binary) in enumerate(RUNS):
        gdir = os.path.join(base, grpo)
        bdir = os.path.join(base, basel)
        examples, keys, info = reconstruct_passes(gdir, n_val)
        errmask = info.pop("_errmask")
        obs = pass_scores(examples, n_val)
        off = official_val_scores(gdir)
        diffs = [abs(round(o, 2) - s) for o, (_, s) in zip(obs, off)]
        info["official_scores"] = [s for _, s in off]
        info["reconstructed_scores"] = [round(o, 4) for o in obs]
        info["max_abs_diff_vs_official"] = max(diffs) if diffs else None
        info["all_26_match"] = (len(off) == 26 and len(obs) == 26 and max(diffs) < 0.006
                                and info["n_examples"] == n_val)
        info["checkpoint_selected_by_dspy"] = selected_checkpoint(gdir)
        print(f"\n=== {name}: passes {info['n_passes']}, examples {info['n_examples']}/{n_val}, "
              f"match official: {info['all_26_match']} (max diff {info['max_abs_diff_vs_official']:.4f})")
        print("    info:", {k: v for k, v in info.items() if k not in ("official_scores", "reconstructed_scores")})
        r = {"pass_info": info}
        if not info["all_26_match"]:
            print("    !! reconstruction does not reproduce the official scores - tests still run, flagged")
            for k, (o, (st, s)) in enumerate(zip(obs, off)):
                if abs(round(o, 2) - s) >= 0.006:
                    print(f"       pass {k+1} (step {st}): reconstructed {o:.3f} official {s}")
        if B == 0:          # reconstruction check only, no tests
            continue
        r["T1"] = t1_selection(examples, n_val, binary, B, SEED + 10 * j + 1)
        r["T2"] = t2_trend(examples, n_val, B, SEED + 10 * j + 2)
        if any(any(e) for e in errmask):
            r["S1_posthoc_errors_excluded"] = s1_errors_excluded(examples, errmask, B, SEED + 10 * j + 6)
            r["S1_posthoc_errors_excluded"]["errors_first5_last5"] = [
                sum(info["errors_per_pass"][:5]), sum(info["errors_per_pass"][-5:])]
        mean_p = statistics.fmean(obs) / 100.0
        r["T4"] = {"p_mean_over_passes": mean_p,
                   "binomial_se_points": 100 * math.sqrt(mean_p * (1 - mean_p) / n_val),
                   "null_sd_one_pass_points": r["T1"]["null_sd_one_pass_analytic"],
                   "observed_sd_26_passes": r["T1"]["observed_sd_of_26_passes"]}
        r["T4"]["expected_max25_gain_binomial_points"] = emax * r["T4"]["binomial_se_points"]
        r["T4"]["expected_max25_gain_nullsd_points"] = emax * r["T4"]["null_sd_one_pass_points"]
        n_test = {"HotpotQA": 300, "HoVer": 300, "IFBench": 294, "PUPA": 221}[name]
        t3 = t3_paired(gdir, bdir, n_test, binary, B, SEED + 10 * j + 3)
        if t3 is not None:
            t3["official_grpo"] = official_test_score(gdir)
            t3["official_base"] = official_test_score(bdir)
            t3["grpo_matches_official"] = abs(t3["grpo_mean"] - t3["official_grpo"]) < 0.006
            t3["base_matches_official"] = abs(t3["base_mean"] - t3["official_base"]) < 0.006
            if name == "PUPA":
                t3["sensitivity_test_split_only"] = t3_paired(gdir, bdir, n_test, binary, B,
                                                              SEED + 10 * j + 5, only_test_split=True)
        r["T3"] = t3 if t3 is not None else {
            "available": False,
            "grpo_test_jsonl": os.path.exists(os.path.join(gdir, "metric_logs", "test.jsonl")),
            "base_test_jsonl": os.path.exists(os.path.join(bdir, "metric_logs", "test.jsonl")),
            "official_grpo": official_test_score(gdir), "official_base": official_test_score(bdir)}
        results["runs"][name] = r
        for t in ("T1", "T2", "T4"):
            print(f"    {t}:", json.dumps(r[t], default=str))
        print("    T3:", json.dumps({k: v for k, v in r["T3"].items() if k != "info"}, default=str))
        if t3 is not None:
            print("    T3 info:", json.dumps(t3["info"], default=str)[:600])
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    print("\nwritten:", os.path.join(outdir, "results.json"))


if __name__ == "__main__":
    main()
