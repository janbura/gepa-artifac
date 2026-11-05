"""Are the differences between optimizers in the GEPA artifact bigger than test-set noise?

Every run (except two GRPO runs) logs the metric for each test example in
metric_logs/test.jsonl.  We pair two runs of the same benchmark and model on the
test examples they share and compute
  - the mean difference (in points, same scale as the paper's tables),
  - a 95% bootstrap confidence interval over test examples,
  - a two-sided paired sign-flip permutation p-value,
  - the minimum detectable difference (80% power, alpha 0.05) for this test set.

This only covers the noise from the finite test set.  Run-to-run noise of the
optimizer itself (different seeds) comes on top of it, so a difference that is
not significant here is certainly not significant once seeds are included.

Notes on the logs:
  - AIME runs every one of the 30 problems 5 times; the 5 tries are averaged
    per problem, so the unit of the test is a problem (n = 30).
  - Some logs have duplicate or missing examples (mostly the Qwen IFBench runs).
    Duplicates are averaged, and only examples present in both runs are used.
    The completeness of each log is written to test_log_completeness.csv.

Run from the repository root:
    python analysis/significance.py
By default this reads our own runs in experiment_runs_data/ (step 8.2).  For the
authors' runs, pass --runs-dir reference_runs_data/experiment_runs/seed_0.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")

BENCH_LABELS = {
    "HotpotQABench": "HotpotQA",
    "IFBench": "IFBench",
    "hoverBench": "HoVer",
    "Papillon": "PUPA",
    "AIMEBench": "AIME",
    "LiveBenchMathBench": "LiveBench-Math",
}
MODEL_LABELS = {"gpt-41-mini": "GPT-4.1 mini", "qwen3-8b": "Qwen3 8B"}
OPT_LABELS = {
    "Baseline": "Baseline",
    "MIPROv2-Heavy": "MIPROv2",
    "GEPA": "GEPA",
    "GEPA-MERGE": "GEPA+Merge",
    "Abl-SelectBestCandidate": "SelectBest",
    "GRPO": "GRPO",
}

# (A, B): we test A - B.  These are the comparisons the paper's claims rest on.
COMPARISONS = [
    ("GEPA", "Baseline"),
    ("GEPA", "MIPROv2-Heavy"),
    ("GEPA", "GRPO"),
    ("GEPA-MERGE", "GEPA"),
    ("GEPA", "Abl-SelectBestCandidate"),
    ("GEPA-MERGE", "MIPROv2-Heavy"),
]

N_RESAMPLES = 10000


def parse_run_name(name):
    bench, program, optimizer, model = name.split("_")
    return bench, program, optimizer, model


def read_test_log(run_dir):
    """Returns ({example index: mean metric}, number of log lines), or (None, 0) without a log."""
    path = os.path.join(run_dir, "metric_logs", "test.jsonl")
    if not os.path.exists(path):
        return None, 0
    scores = defaultdict(list)
    n_lines = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_lines += 1
            value = row["metric_output"]
            scores[row["idx_in_split"]].append(0.0 if value is None else float(value))
    return {k: float(np.mean(v)) for k, v in scores.items()}, n_lines


def read_reported_score(run_dir):
    path = os.path.join(run_dir, "evaluation_results", "evaluation_result.txt")
    with open(path) as f:
        lines = f.read().splitlines()
    return float(lines[1].split(",")[0])


def paired_test(a, b, rng):
    """a, b: arrays of per-example scores (0-1 scale) for the same examples."""
    d = (a - b) * 100
    n = len(d)
    mean = d.mean()
    # bootstrap over examples
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # sign-flip permutation test
    signs = rng.choice([-1.0, 1.0], size=(N_RESAMPLES, n))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(mean) - 1e-12) + 1) / (N_RESAMPLES + 1)
    # minimum detectable difference, normal approximation: (1.96 + 0.84) * sd / sqrt(n)
    sd = d.std(ddof=1) if n > 1 else 0.0
    mdd = 2.8 * sd / np.sqrt(n)
    n_a_better = int(np.sum(d > 0))
    n_b_better = int(np.sum(d < 0))
    return mean, lo, hi, p, mdd, n, n_a_better, n_b_better


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--fig-dir", default=FIG_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    runs = {}
    completeness = []
    for name in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, name)
        if not os.path.isdir(run_dir) or name.count("_") != 3:
            continue
        bench, _, optimizer, model = parse_run_name(name)
        scores, n_lines = read_test_log(run_dir)
        reported = read_reported_score(run_dir)
        row = {"benchmark": BENCH_LABELS.get(bench, bench), "model": MODEL_LABELS.get(model, model),
               "optimizer": OPT_LABELS.get(optimizer, optimizer), "reported": reported,
               "log_lines": n_lines, "unique_examples": 0, "log_mean": None, "log_minus_reported": None}
        if scores is not None:
            runs[(bench, model, optimizer)] = scores
            log_mean = 100 * np.mean(list(scores.values()))
            row.update(unique_examples=len(scores), log_mean=round(log_mean, 2),
                       log_minus_reported=round(log_mean - reported, 2))
        completeness.append(row)
    comp = pd.DataFrame(completeness)
    comp.to_csv(os.path.join(args.out_dir, "test_log_completeness.csv"), index=False)

    rows = []
    for bench in BENCH_LABELS:
        for model in MODEL_LABELS:
            for opt_a, opt_b in COMPARISONS:
                a = runs.get((bench, model, opt_a))
                b = runs.get((bench, model, opt_b))
                if a is None or b is None:
                    continue
                shared = sorted(set(a) & set(b))
                xa = np.array([a[k] for k in shared])
                xb = np.array([b[k] for k in shared])
                mean, lo, hi, p, mdd, n, n_a, n_b = paired_test(xa, xb, rng)
                rows.append({
                    "benchmark": BENCH_LABELS[bench], "model": MODEL_LABELS[model],
                    "comparison": f"{OPT_LABELS[opt_a]} - {OPT_LABELS[opt_b]}",
                    "n_examples": n, "diff": round(mean, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                    "p_value": p, "min_detectable_diff": round(mdd, 2),
                    "n_A_better": n_a, "n_B_better": n_b,
                })
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"].values)
    df["significant_holm"] = df["p_holm"] < 0.05
    df["p_value"] = df["p_value"].round(4)
    df["p_holm"] = df["p_holm"].round(4)
    df.to_csv(os.path.join(args.out_dir, "significance.csv"), index=False)

    with open(os.path.join(args.out_dir, "significance.md"), "w") as f:
        f.write("# Paired significance of optimizer differences on the test sets (artifact, seed 0)\n\n")
        f.write("diff = A - B in points; CI = 95% bootstrap over test examples; p = paired sign-flip test; "
                "p_holm = Holm correction over all rows; min_detectable_diff = smallest difference this test "
                "set can detect with 80% power. Only test-set noise is included, not seed-to-seed noise.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Completeness of the per-example test logs\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n")

    # forest plot: one panel per model
    n_rows = max(len(df[df["model"] == m]) for m in MODEL_LABELS.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.3 * n_rows + 1.8), sharex=True)
    for ax, model in zip(axes, MODEL_LABELS.values()):
        sub = df[df["model"] == model].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        colors = ["#1f77b4" if s else "#999999" for s in sub["significant_holm"]]
        for yi, (_, r), c in zip(y, sub.iterrows(), colors):
            ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=c, linewidth=1.6)
            ax.plot(r["diff"], yi, "o", color=c, markersize=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{b}: {c}" for b, c in zip(sub["benchmark"], sub["comparison"])], fontsize=8)
        ax.set_title(model)
        ax.set_xlabel("difference in test score (points), 95% CI")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Paired test-set differences (artifact, seed 0); blue = significant after Holm correction")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "significance_forest.png"), dpi=150)
    print(df.to_string(index=False))
    print()
    print(comp[comp["log_minus_reported"].abs() > 0.5].to_string(index=False))


if __name__ == "__main__":
    main()
