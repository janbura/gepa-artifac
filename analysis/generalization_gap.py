"""Validation-to-test generalization gap of the optimized programs in the GEPA artifact.

For every run that is not a baseline we read the best validation score reached by
the optimizer, the validation score of the program that was actually tested and
the final test score.  The seed program's validation score is compared with the
baseline test score in the same way.  For the GEPA family we additionally compare
the best candidate with the seed program instance by instance on the validation
set, to see whether the validation gain is spread over many instances or
concentrated on a few.

Run from the repository root:
    python analysis/generalization_gap.py
"""

import argparse
import csv
import json
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")

# display names; the dict order is also the order used in tables and figure
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
GEPA_FAMILY = ("GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate")


def parse_run_name(name):
    """Run folders are named <Benchmark>_<Program>_<Optimizer>_<Model>."""
    bench, program, optimizer, model = name.split("_")
    return bench, program, optimizer, model


def read_test_score(run_dir):
    path = os.path.join(run_dir, "evaluation_results", "evaluation_result.txt")
    with open(path, encoding="utf-8") as f:
        header, values = list(csv.reader(f))[:2]
    return float(values[header.index("score")])


def load_state(run_dir):
    with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
        return pickle.load(f)


def gepa_validation(state):
    """Full validation scores of all candidates; the optimizer returns the first one reaching the maximum."""
    scores = state["program_full_scores_val_set"]
    best = max(scores)
    best_idx = scores.index(best)
    return dict(seed_val=scores[0], best_val=best, best_id=best_idx, tested_val=best,
                n_val=len(state["pareto_front_valset"]), n_candidates=len(scores))


def mipro_validation(run_dir):
    """MIPROv2 logs the full validation score of every fully evaluated candidate (first entry: seed program)."""
    n_val, scores = None, None
    with open(os.path.join(run_dir, "run_log_stderr.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"valset size: (\d+)", line)
            if m:
                n_val = int(m.group(1))
            m = re.search(r"Full eval scores so far: \[([^\]]*)\]", line)
            if m:
                scores = [float(x) for x in m.group(1).split(",")]
    best = max(scores)
    return dict(seed_val=scores[0], best_val=best, best_id=scores.index(best), tested_val=best,
                n_val=n_val, n_candidates=len(scores))


def grpo_validation(run_dir):
    """GRPO scores the student on the validation set every 20 steps; the tested model is the last checkpoint."""
    seed, n_val, steps = None, None, []
    with open(os.path.join(run_dir, "run_log_stderr.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"validation set score before training loop: ([\d.]+)", line)
            if m:
                seed = float(m.group(1))
            m = re.search(r"validation set score after training step (\d+)/\d+: ([\d.]+)", line)
            if m:
                steps.append((int(m.group(1)), float(m.group(2))))
            m = re.search(r"Average Metric: [\d.]+ / (\d+)", line)
            if m and n_val is None:
                n_val = int(m.group(1))  # the first evaluation in the log is the validation set
    best_step, best = max(steps, key=lambda s: s[1])
    return dict(seed_val=seed, best_val=best, best_id=best_step, tested_val=steps[-1][1],
                n_val=n_val, n_candidates=len(steps))


def metric_score(output):
    """Logged metric outputs are numbers or bools, or a dict with a score field."""
    if isinstance(output, dict):
        return float(output["score"])
    return float(output)


def seed_scores_from_log(run_dir, n_val):
    """Per-instance validation scores of the seed program, read from metric_logs/val.jsonl.

    The full validation pass of the seed program is the first thing written to this
    log, so its rows come first.  Rows are attributed by idx_in_split.  An index that
    appears more than once among the first n_val rows cannot be attributed (two
    validation examples that compare equal share one index, and when a pass skips
    examples the next pass starts before row n_val), so only indices seen exactly
    once are returned.
    """
    counts, scores = {}, {}
    with open(os.path.join(run_dir, "metric_logs", "val.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_val:
                break
            row = json.loads(line)
            idx = row["idx_in_split"]
            counts[idx] = counts.get(idx, 0) + 1
            scores[idx] = metric_score(row["metric_output"])
    return {idx: s for idx, s in scores.items() if counts[idx] == 1}


def compare_best_with_seed(state, best_idx, seed_log):
    """Count validation instances on which the best candidate beats, ties or loses to the seed.

    prog_candidate_val_subscores[0] cannot be used for the seed: in the shipped
    states it is the same list object as pareto_front_valset, which was updated in
    place during the run, so it holds the best score of any candidate per instance.
    The membership of candidate 0 in program_at_pareto_front_valset is exact: the
    seed ties the per-instance best exactly when 0 is in that set.  This settles
    every instance for 0/1 metrics; for graded metrics the case where both programs
    are below the best is settled with the seed scores from the metric log.
    """
    fronts = state["program_at_pareto_front_valset"]
    front_scores = state["pareto_front_valset"]
    subscores = state["prog_candidate_val_subscores"]
    binary = {float(x) for scores in subscores[1:] for x in scores} <= {0.0, 1.0}
    counts = dict(better=0, equal=0, worse=0, undetermined=0)
    from_log, mismatches = 0, 0
    for i, front in enumerate(fronts):
        seed_on_front = 0 in front
        best_on_front = best_idx in front
        if i in seed_log:
            # the log must agree with the state about the seed tying the best score
            if seed_on_front != (abs(seed_log[i] - float(front_scores[i])) < 1e-9):
                mismatches += 1
        if best_on_front and seed_on_front:
            key = "equal"
        elif best_on_front:
            key = "better"
        elif seed_on_front:
            key = "worse"
        elif binary:
            key = "equal"  # both scored 0 on an instance that another candidate solved
        elif i in seed_log:
            diff = float(subscores[best_idx][i]) - seed_log[i]
            key = "better" if diff > 1e-9 else ("worse" if diff < -1e-9 else "equal")
            from_log += 1
        else:
            key = "undetermined"
        counts[key] += 1
    return counts, binary, from_log, mismatches


def markdown_table(df, digits=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.{digits}f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect(runs_dir):
    """One row per optimized run plus one row per GEPA-family run for the per-instance comparison."""
    baseline_test = {}
    optimized = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        bench, _, optimizer, model = parse_run_name(name)
        test = read_test_score(run_dir)
        if optimizer == "Baseline":
            baseline_test[(bench, model)] = test
        else:
            optimized.append((name, run_dir, bench, optimizer, model, test))

    rows, change_rows = [], []
    for name, run_dir, bench, optimizer, model, test in optimized:
        if optimizer in GEPA_FAMILY:
            state = load_state(run_dir)
            info = gepa_validation(state)
            seed_log = seed_scores_from_log(run_dir, info["n_val"])
            counts, binary, from_log, mismatches = compare_best_with_seed(state, info["best_id"], seed_log)
            n_val = info["n_val"]
            change_rows.append(dict(
                benchmark=BENCH_LABELS[bench], model=model, optimizer=OPT_LABELS[optimizer],
                n_val=n_val, best_id=info["best_id"], seed_val=info["seed_val"], best_val=info["best_val"],
                val_gain=info["best_val"] - info["seed_val"],
                n_better=counts["better"], n_equal=counts["equal"], n_worse=counts["worse"],
                n_undetermined=counts["undetermined"],
                frac_better=counts["better"] / n_val, frac_equal=counts["equal"] / n_val,
                frac_worse=counts["worse"] / n_val,
                frac_changed=(counts["better"] + counts["worse"]) / n_val,
                binary_metric=binary, n_seed_from_log=len(seed_log), n_settled_by_log=from_log,
                n_log_state_mismatch=mismatches,
            ))
        elif optimizer == "MIPROv2-Heavy":
            info = mipro_validation(run_dir)
        elif optimizer == "GRPO":
            info = grpo_validation(run_dir)
        else:
            raise ValueError("unknown optimizer in run " + name)
        base = baseline_test[(bench, model)]
        gap = info["best_val"] - test
        seed_gap = info["seed_val"] - base
        rows.append(dict(
            benchmark=BENCH_LABELS[bench], model=model, optimizer=OPT_LABELS[optimizer],
            n_val=info["n_val"], n_candidates=info["n_candidates"],
            seed_val=info["seed_val"], baseline_test=base, seed_gap=seed_gap,
            best_val=info["best_val"], best_id=info["best_id"], tested_val=info["tested_val"], test=test,
            gap=gap, gap_tested=info["tested_val"] - test, excess_gap=gap - seed_gap,
            val_gain=info["best_val"] - info["seed_val"], test_gain=test - base,
        ))
    return pd.DataFrame(rows), pd.DataFrame(change_rows)


def sort_rows(df):
    bench_order = {b: i for i, b in enumerate(BENCH_LABELS.values())}
    model_order = {m: i for i, m in enumerate(MODEL_LABELS)}
    opt_order = {o: i for i, o in enumerate(OPT_LABELS.values())}
    key = df.apply(lambda r: (model_order[r.model], bench_order[r.benchmark], opt_order[r.optimizer]), axis=1)
    return df.assign(_key=key).sort_values("_key").drop(columns="_key").reset_index(drop=True)


def summarize(df):
    """Mean gaps per optimizer, overall and per model, with a paired comparison against MIPROv2."""
    mipro = df[df.optimizer == "MIPROv2"].set_index(["benchmark", "model"])
    out = []
    for model in ["all"] + list(MODEL_LABELS):
        part = df if model == "all" else df[df.model == model]
        for opt in OPT_LABELS.values():
            sub = part[part.optimizer == opt]
            if sub.empty:
                continue
            paired = sub.set_index(["benchmark", "model"]).join(mipro[["gap", "excess_gap"]], rsuffix="_mipro")
            paired = paired.dropna(subset=["gap_mipro"])
            out.append(dict(
                model=model, optimizer=opt, n_runs=len(sub),
                mean_gap=sub.gap.mean(), median_gap=sub.gap.median(),
                mean_seed_gap=sub.seed_gap.mean(), mean_excess_gap=sub.excess_gap.mean(),
                mean_val_gain=sub.val_gain.mean(), mean_test_gain=sub.test_gain.mean(),
                gap_below_mipro="%d/%d" % ((paired.gap < paired.gap_mipro).sum(), len(paired)),
                excess_gap_below_mipro="%d/%d" % ((paired.excess_gap < paired.excess_gap_mipro).sum(), len(paired)),
            ))
    return pd.DataFrame(out)


def make_figure(df, path):
    """Grouped bars, one panel per benchmark and model: best validation score next to the test score."""
    colors = {"validation": "#2a78d6", "test": "#eb6834"}
    ink, ink2, hair = "#0b0b0b", "#52514e", "#d6d5d0"
    benches = list(BENCH_LABELS.values())
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(len(models), len(benches), figsize=(18, 7.4), sharey=True)
    width = 0.36
    for r, model in enumerate(models):
        for c, bench in enumerate(benches):
            ax = axes[r][c]
            sub = df[(df.model == model) & (df.benchmark == bench)]
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "no run in the artifact", ha="center", va="center",
                        fontsize=8, color=ink2, transform=ax.transAxes)
                continue
            # the seed program's validation score (from the GEPA run) next to the baseline test score
            seed_row = sub[sub.optimizer == "GEPA"].iloc[0] if (sub.optimizer == "GEPA").any() else sub.iloc[0]
            labels = ["Baseline"] + list(sub.optimizer)
            val = [seed_row.seed_val] + list(sub.best_val)
            tst = [seed_row.baseline_test] + list(sub.test)
            xs = range(len(labels))
            ax.bar([x - 0.2 for x in xs], val, width, color=colors["validation"], zorder=2)
            ax.bar([x + 0.2 for x in xs], tst, width, color=colors["test"], zorder=2)
            for x, v, t in zip(xs, val, tst):
                ax.text(x, max(v, t) + 1.5, "%+.1f" % (v - t), ha="center", va="bottom", fontsize=6.5, color=ink2)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5, color=ink2)
            ax.set_title("%s / %s" % (bench, MODEL_LABELS[model]), fontsize=9, color=ink)
            ax.set_ylim(0, 108)
            ax.grid(axis="y", color="#ebeae6", linewidth=0.6, zorder=0)
            ax.tick_params(axis="y", labelsize=7.5, colors=ink2, length=0)
            ax.tick_params(axis="x", length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(hair)
            if c == 0:
                ax.set_ylabel("score (%)", fontsize=8, color=ink2)
    handles = [Patch(color=colors["validation"], label="best validation score (seed program for Baseline)"),
               Patch(color=colors["test"], label="test score")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Validation vs test score per optimizer (seed 0); the label above each pair is validation minus test",
                 fontsize=10, color=ink, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


NOTES = """
Columns: `seed_val` is the full validation score of the seed program as logged by
that optimizer, `baseline_test` the test score of the Baseline run of the same
benchmark and model, `seed_gap = seed_val - baseline_test`.  `best_val` is the
best full validation score reached (GEPA family: max over candidates; MIPROv2: max
of the "Full eval scores so far" list; GRPO: max over the evaluations done every
20 steps), `best_id` identifies it (candidate index, position in the MIPROv2 list
with 0 = seed program, or training step), `tested_val` is the validation score of
the program that was evaluated on the test set (equal to `best_val` except for
GRPO, which keeps the last checkpoint), `gap = best_val - test`, `gap_tested =
tested_val - test`, `excess_gap = gap - seed_gap` (the part of the gap not already
present for the unoptimized seed program), `val_gain = best_val - seed_val`,
`test_gain = test - baseline_test`.  All scores are in percent.

Caveats: the AIME GEPA run used a 45-example validation set (benchmark
"AIMEBench2"), the AIME MIPROv2 run a 90-example one ("AIMEBench"); the GRPO
runs were tested with the final checkpoint, not the best one; only seed 0 exists,
so differences of one or two points are within evaluation noise (the validation
score of the same seed program varies by about one point between optimizer runs
of the same benchmark).
"""

CHANGE_NOTES = """
For every GEPA-family run the best validation candidate (`best_id`) is compared
with the seed program (candidate 0) on each validation instance.  `n_better`,
`n_equal`, `n_worse` count instances where the best candidate scores higher, the
same or lower than the seed; `frac_changed` is the share of instances whose score
changed in either direction.  The comparison uses the per-instance Pareto front
stored in gepa_state.bin (the seed row of `prog_candidate_val_subscores` is not
usable, see the script); for graded metrics, instances where both programs are
below the per-instance best are settled with the seed scores from
metric_logs/val.jsonl (`n_settled_by_log`), and `n_undetermined` instances could
not be settled either way.  `n_log_state_mismatch` counts instances where the log
and the state disagree (expected 0).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--fig-dir", default=FIG_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    gaps, changes = collect(args.runs_dir)
    gaps = sort_rows(gaps)
    changes = sort_rows(changes)
    summary = summarize(gaps)

    gaps.to_csv(os.path.join(args.out_dir, "generalization_gap.csv"), index=False, float_format="%.4f")
    summary.to_csv(os.path.join(args.out_dir, "generalization_gap_summary.csv"), index=False, float_format="%.4f")
    changes.to_csv(os.path.join(args.out_dir, "per_instance_changes.csv"), index=False, float_format="%.4f")

    with open(os.path.join(args.out_dir, "generalization_gap.md"), "w", encoding="utf-8") as f:
        f.write("# Validation-to-test generalization gap (seed 0)\n\n")
        f.write(markdown_table(gaps) + "\n\n")
        f.write("## Summary per optimizer\n\n")
        f.write("`gap_below_mipro` counts the benchmark/model pairs on which the gap of the optimizer is smaller "
                "than the gap of MIPROv2 on the same pair (same for `excess_gap_below_mipro`).\n\n")
        f.write(markdown_table(summary) + "\n")
        f.write(NOTES)
    with open(os.path.join(args.out_dir, "per_instance_changes.md"), "w", encoding="utf-8") as f:
        f.write("# Per-instance validation changes of the best candidate relative to the seed program\n\n")
        f.write(markdown_table(changes, digits=3) + "\n")
        f.write(CHANGE_NOTES)

    make_figure(gaps, os.path.join(args.fig_dir, "generalization_gap.png"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(gaps[["benchmark", "model", "optimizer", "n_val", "seed_val", "baseline_test", "seed_gap", "best_val",
                "tested_val", "test", "gap", "gap_tested", "excess_gap"]].round(2).to_string(index=False))
    print()
    print(summary.round(2).to_string(index=False))
    print()
    print(changes[["benchmark", "model", "optimizer", "n_val", "best_id", "seed_val", "best_val", "n_better",
                   "n_equal", "n_worse", "n_undetermined", "frac_better", "frac_worse", "n_settled_by_log",
                   "n_log_state_mismatch"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
