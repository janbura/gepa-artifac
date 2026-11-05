"""Best validation score vs number of rollouts for every optimizer run.

Curve sources:
  GEPA, GEPA-MERGE, Abl-SelectBestCandidate: gepa_state.bin
      (program_full_scores_val_set against num_metric_calls_by_discovery)
  MIPROv2: run_log_stderr.txt (full validation evaluations; rollouts counted
      the same way as the authors' notebook scripts/generate_figures.ipynb)
  GRPO: run_log_stderr.txt (validation score logged every 20 training steps,
      48 rollouts per step)

Outputs, relative to the repo root:
  analysis/figures/budget_<bench>_<model>.png   one figure per task and model
  analysis/outputs/budget_curves.csv            every curve point
  analysis/outputs/sample_efficiency.md         summary tables (also as csv)

Run from the repo root:  python analysis/budget_curves.py [--logx]
By default this reads our own runs in experiment_runs_data/ (step 8.2).  For the
authors' runs, pass --runs-dir reference_runs_data/experiment_runs/seed_0.
"""

import argparse
import json
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")

# Rollout budget of the prompt optimizers per task, copied from
# get_max_invocations in scripts/experiment_configs.py (GEPA gets the number
# of metric calls that MIPROv2-Heavy used).
BUDGETS = {
    "HotpotQABench": 6871,
    "Papillon": 2426,
    "hoverBench": 7051,
    "IFBench": 3593,
    "LiveBenchMathBench": 1839,
    "AIMEBench": 1839,
}
MIPRO_MINIBATCH = 35
GRPO_ROLLOUTS_PER_STEP = 48  # scores are logged every 20 steps = 960 rollouts
GRPO_BUDGET = 24000

# Numbers quoted in the paper for Qwen3 8B: rollouts GEPA needed to reach the
# best validation score of GRPO, and the largest resulting ratio.
PAPER_ROLLOUTS_TO_MATCH_GRPO = {
    "HotpotQABench": 243, "IFBench": 402, "hoverBench": 330, "Papillon": 1143}
PAPER_MAX_SPEEDUP = 78

GEPA_FAMILY = ("GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate")
PLOT_ORDER = ["GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate", "MIPROv2-Heavy", "GRPO"]
# fixed colour, marker and line style per optimizer
STYLES = {
    "GEPA": ("#2a78d6", "o", "-"),
    "GEPA-MERGE": ("#6250d6", "s", "--"),
    "Abl-SelectBestCandidate": ("#e87ba4", "^", "-."),
    "MIPROv2-Heavy": ("#1baf7a", "D", "-"),
    "GRPO": ("#eb6834", "v", "-"),
}
LABELS = {
    "GEPA": "GEPA",
    "GEPA-MERGE": "GEPA+Merge",
    "Abl-SelectBestCandidate": "GEPA-Abl-SBC",
    "MIPROv2-Heavy": "MIPROv2",
    "GRPO": "GRPO",
}
BENCH_NAMES = {
    "HotpotQABench": "HotpotQA", "IFBench": "IFBench", "hoverBench": "HoVer",
    "Papillon": "PUPA", "AIMEBench": "AIME 2025", "LiveBenchMathBench": "LiveBench Math"}
MODEL_NAMES = {"qwen3-8b": "Qwen3 8B", "gpt-41-mini": "GPT-4.1 mini"}


def parse_run_name(name):
    bench, program, optimizer, model = name.split("_")
    return bench, program, optimizer, model


def read_first_json_line(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def read_last_json_line(path, chunk_size=1 << 16):
    """Parse the last line of a jsonl file by reading it backwards from the end."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0:
            step = min(chunk_size, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            body = buf.rstrip(b"\r\n")
            if b"\n" in body:
                return json.loads(body[body.rfind(b"\n") + 1:].decode("utf-8"))
    return json.loads(buf.rstrip(b"\r\n").decode("utf-8"))


def read_test_score(run_dir):
    path = os.path.join(run_dir, "evaluation_results", "evaluation_result.txt")
    # the data row ends with a comma; index_col=False keeps the first column a column
    return float(pd.read_csv(path, index_col=False).iloc[0]["score"])


def metric_log_counters(run_dir):
    """Train, val and dev metric calls from the last line of train.jsonl and val.jsonl."""
    out = {"dev_calls": 0}
    for split in ("train", "val"):
        path = os.path.join(run_dir, "metric_logs", split + ".jsonl")
        if os.path.exists(path):
            last = read_last_json_line(path)
        else:
            # the AIME MIPROv2 run has no val.jsonl; the first test line carries
            # the same counters at the moment the test evaluation started
            last = read_first_json_line(os.path.join(run_dir, "metric_logs", "test.jsonl"))
        out[split + "_calls"] = last[split + "_counter"]
        out["dev_calls"] = max(out["dev_calls"], last.get("dev_counter", 0))
    return out


def gepa_curve(run_dir):
    with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
        state = pickle.load(f)
    info = {
        "valset_size": len(state["pareto_front_valset"]),
        "total_num_evals": state["total_num_evals"],
        "num_full_evals": state["num_full_ds_evals"],
        "num_iterations": state["i"] + 1,
    }
    rollouts = [int(x) for x in state["num_metric_calls_by_discovery"]]
    scores = [float(x) for x in state["program_full_scores_val_set"]]
    return rollouts, scores, info


def mipro_curve(run_dir):
    """Full validation evaluations of MIPROv2 against rollouts.

    The k-th full evaluation (k = 0 is the default program) is placed at
    bootstrapping train calls + 35 per minibatch trial so far + k * valset size,
    which is the accounting of the authors' notebook: like GEPA's discovery
    count it excludes the full evaluation of the candidate itself.
    """
    with open(os.path.join(run_dir, "run_log_stderr.txt"), "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    valset_size = None
    scores = []
    search_calls = []  # search calls spent before each full eval after the default program
    n_minibatch = 0
    for line in lines:
        line = line.strip()
        if valset_size is None and "valset size: " in line:
            valset_size = int(line.split("valset size: ")[1])
        elif "Default program score: " in line:
            scores.append(float(line.split("Default program score: ")[1]))
        elif "on minibatch of size " in line and "Score: " in line:
            assert "on minibatch of size %d " % MIPRO_MINIBATCH in line, line
            n_minibatch += 1
        elif "Full eval scores so far: " in line or "Scores so far: " in line:
            seen = json.loads(line.split("so far: ")[1])
            if len(seen) > len(scores):
                scores.append(float(seen[-1]))
                search_calls.append(n_minibatch * MIPRO_MINIBATCH + (len(scores) - 1) * valset_size)
    assert valset_size is not None and len(scores) == len(search_calls) + 1
    # counters at the start of the test evaluation = calls spent by the optimizer
    first_test = read_first_json_line(os.path.join(run_dir, "metric_logs", "test.jsonl"))
    search = n_minibatch * MIPRO_MINIBATCH + len(scores) * valset_size
    if first_test["val_counter"] > 0:
        # authors' accounting: train calls are bootstrapping, val calls are the
        # search; val calls beyond the nominal search count are added as well
        bootstrap = first_test["train_counter"] + max(0, first_test["val_counter"] - search)
    else:
        # AIME: the search calls were logged under train/dev, so bootstrapping
        # is whatever the logged total leaves after the nominal search calls
        logged = sum(first_test.get(k, 0) for k in ("train_counter", "val_counter", "dev_counter"))
        bootstrap = max(0, logged - search)
    rollouts = [0] + [bootstrap + c for c in search_calls]
    info = {
        "valset_size": valset_size,
        "num_full_evals": len(scores),
        "num_minibatches": n_minibatch,
        "bootstrap_calls": bootstrap,
        "authors_total": bootstrap + search,
    }
    return rollouts, scores, info


def grpo_curve(run_dir):
    """Validation score logged by GRPO before training and every 20 steps."""
    rollouts, scores = [], []
    step_re = re.compile(r"after training step (\d+)/\d+: ([0-9.]+)")
    with open(os.path.join(run_dir, "run_log_stderr.txt"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Student program validation set score before training loop: " in line:
                rollouts.append(0)
                scores.append(float(line.strip().rsplit(": ", 1)[1]))
            elif "Student program validation set score after training step" in line:
                m = step_re.search(line)
                rollouts.append(GRPO_ROLLOUTS_PER_STEP * int(m.group(1)))
                scores.append(float(m.group(2)))
    return rollouts, scores, {"num_full_evals": len(scores)}


def collect_runs():
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        bench, program, optimizer, model = parse_run_name(name)
        run = {"run": name, "bench": bench, "program": program, "optimizer": optimizer,
               "model": model, "test_score": read_test_score(run_dir)}
        if optimizer in GEPA_FAMILY:
            rollouts, scores, info = gepa_curve(run_dir)
        elif optimizer.startswith("MIPROv2"):
            rollouts, scores, info = mipro_curve(run_dir)
        elif optimizer == "GRPO":
            rollouts, scores, info = grpo_curve(run_dir)
        else:  # Baseline has no optimization curve
            rollouts, scores, info = [], [], {}
        run.update(info)
        run["rollouts"] = rollouts
        run["scores"] = scores
        run["best_so_far"] = [float(x) for x in np.maximum.accumulate(scores)] if scores else []
        if optimizer != "Baseline":
            run.update(metric_log_counters(run_dir))
        runs.append(run)
    return runs


def group_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["bench"], run["model"]), []).append(run)
    return groups


def plot_rank(run):
    return PLOT_ORDER.index(run["optimizer"]) if run["optimizer"] in PLOT_ORDER else len(PLOT_ORDER)


def curve_table(runs):
    rows = []
    for run in runs:
        for r, s, b in zip(run["rollouts"], run["scores"], run["best_so_far"]):
            rows.append({"run": run["run"], "optimizer": run["optimizer"], "bench": run["bench"],
                         "model": run["model"], "rollouts": r, "val_score": s, "best_so_far": b})
    return pd.DataFrame(rows)


def plot_group(bench, model, group, logx):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in sorted(group, key=plot_rank):
        if not run["scores"]:
            continue
        color, marker, ls = STYLES[run["optimizer"]]
        label = "%s (best val %.1f, test %.1f)" % (LABELS[run["optimizer"]], max(run["scores"]), run["test_score"])
        ax.plot(run["rollouts"], run["best_so_far"], drawstyle="steps-post", color=color, marker=marker,
                linestyle=ls, linewidth=2, markersize=5, label=label)
        # test score of the program that was finally selected, at the rollouts
        # where it was found (GRPO: the final checkpoint)
        if run["optimizer"] == "GRPO":
            x_best = run["rollouts"][-1]
        else:
            x_best = run["rollouts"][int(np.argmax(run["scores"]))]
        ax.plot([x_best], [run["test_score"]], marker="*", markersize=13, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5, linestyle="none")

    # the seed program is the baseline: its validation score is candidate 0 of GEPA
    seed = [r for r in sorted(group, key=plot_rank) if r["scores"]]
    baseline = [r for r in group if r["optimizer"] == "Baseline"]
    if seed:
        seed_val = seed[0]["scores"][0]
        label = "seed program (val %.1f" % seed_val
        label += ", test %.1f)" % baseline[0]["test_score"] if baseline else ")"
        ax.axhline(seed_val, color="#888888", linestyle=":", linewidth=1.5, label=label)
    ax.axvline(BUDGETS[bench], color="#aaaaaa", linestyle="--", linewidth=1,
               label="prompt optimizer budget (%d)" % BUDGETS[bench])

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", markersize=13, markerfacecolor="white",
                          markeredgecolor="#555555", linestyle="none"))
    labels.append("test score of selected program")
    ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.95)

    if logx:
        ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("rollouts (metric calls)")
    ax.set_ylabel("validation score (%), best so far")
    ax.set_title("%s, %s" % (BENCH_NAMES.get(bench, bench), MODEL_NAMES.get(model, model)))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    name = "budget_%s_%s%s.png" % (bench, model, "_logx" if logx else "")
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)


def first_reach(run, target):
    """Index and rollouts of the first curve point whose best-so-far reaches target."""
    for idx, (r, b) in enumerate(zip(run["rollouts"], run["best_so_far"])):
        if b >= target - 1e-9:
            return idx, r
    return None, None


def sample_efficiency_table(runs):
    rows = []
    for (bench, model), group in sorted(group_runs(runs).items()):
        grpo = [r for r in group if r["optimizer"] == "GRPO"]
        if not grpo:
            continue
        grpo = grpo[0]
        grpo_best = max(grpo["scores"])
        _, grpo_needed = first_reach(grpo, grpo_best)
        for run in sorted(group, key=plot_rank):
            if run["optimizer"] not in GEPA_FAMILY + ("MIPROv2-Heavy",):
                continue
            idx, needed = first_reach(run, grpo_best)
            incl = np.nan
            speedup = np.nan
            speedup_incl = np.nan
            if needed is not None:
                # the discovery count excludes the full validation evaluation of
                # the matching candidate itself, so also report the count with it
                incl = needed + (run["valset_size"] if idx > 0 else 0)
                speedup = grpo_needed / needed if needed > 0 else np.inf
                speedup_incl = grpo_needed / incl if incl > 0 else np.inf
            rows.append({
                "bench": bench, "model": model, "optimizer": LABELS[run["optimizer"]],
                "grpo_best_val": grpo_best, "grpo_rollouts_needed": grpo_needed,
                "opt_best_val": max(run["scores"]),
                "rollouts_to_match_grpo": needed,
                "val_score_at_match": run["best_so_far"][idx] if idx is not None else np.nan,
                "rollouts_incl_val_eval": incl,
                "speedup": speedup, "speedup_incl_val_eval": speedup_incl,
                "paper_rollouts": PAPER_ROLLOUTS_TO_MATCH_GRPO.get(bench) if run["optimizer"] == "GEPA" else np.nan,
            })
    return pd.DataFrame(rows)


def best_candidate_table(runs):
    rows = []
    for run in runs:
        if run["optimizer"] not in GEPA_FAMILY:
            continue
        scores, rollouts = run["scores"], run["rollouts"]
        best_idx = int(np.argmax(scores))
        budget = BUDGETS[run["bench"]]
        rows.append({
            "run": run["run"], "bench": run["bench"], "model": run["model"],
            "optimizer": LABELS[run["optimizer"]], "num_candidates": len(scores),
            "seed_val": scores[0], "best_val": scores[best_idx], "test_score": run["test_score"],
            "best_candidate_idx": best_idx, "rollouts_at_best": rollouts[best_idx],
            "budget": budget, "fraction_of_budget": rollouts[best_idx] / budget,
            "last_candidate_rollouts": rollouts[-1], "total_num_evals": run["total_num_evals"],
        })
    return pd.DataFrame(rows)


def budget_accounting_tables(runs):
    """Logged metric calls (train + val) against the budget, GEPA family and the rest."""
    gepa_rows, other_rows = [], []
    for run in runs:
        opt = run["optimizer"]
        if opt == "Baseline":
            continue
        budget = GRPO_BUDGET if opt == "GRPO" else BUDGETS[run["bench"]]
        logged = run["train_calls"] + run["val_calls"] + run["dev_calls"]
        row = {"run": run["run"], "bench": run["bench"], "model": run["model"], "optimizer": LABELS[opt],
               "budget": budget, "train_calls": run["train_calls"], "val_calls": run["val_calls"],
               "dev_calls": run["dev_calls"], "logged_calls": logged,
               "logged_minus_budget": logged - budget, "logged_pct_of_budget": 100.0 * logged / budget}
        if opt in GEPA_FAMILY:
            row.update({
                "state_total_num_evals": run["total_num_evals"],
                "state_minus_budget": run["total_num_evals"] - budget,
                "logged_minus_state": logged - run["total_num_evals"],
                "valset_size": run["valset_size"], "num_full_evals": run["num_full_evals"],
            })
            gepa_rows.append(row)
        else:
            if opt.startswith("MIPROv2"):
                row["authors_total"] = run["authors_total"]
                row["authors_minus_budget"] = run["authors_total"] - budget
            other_rows.append(row)
    return pd.DataFrame(gepa_rows), pd.DataFrame(other_rows)


def fmt_cell(value, as_int):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isinf(value):
            return "inf"
        return "%d" % round(value) if as_int else "%.2f" % value
    return str(value)


def to_markdown(df):
    """Small markdown table writer; integer-valued columns print without decimals."""
    as_int = {}
    for c in df.columns:
        vals = [v for v in df[c].dropna() if isinstance(v, (int, float, np.integer, np.floating))]
        as_int[c] = bool(vals) and all(np.isinf(v) or float(v).is_integer() for v in vals)
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt_cell(row[c], as_int[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def summary_text(eff, best, acc_gepa, acc_other):
    lines = ["# Rollout budget curves and sample efficiency", ""]
    lines.append("Rollouts are metric calls. GEPA family: `num_metric_calls_by_discovery` from "
                 "gepa_state.bin (calls spent before a candidate's full validation evaluation; the "
                 "seed program sits at 0). MIPROv2: bootstrapping train calls + 35 per minibatch trial "
                 "+ valset size per earlier full evaluation. GRPO: 48 rollouts per training step, "
                 "scores logged every 20 steps (960 rollouts), 24000 in total.")
    lines.append("")

    lines += ["## Sample efficiency against GRPO (Qwen3 8B)", ""]
    lines.append("`grpo_best_val` is the best of GRPO's logged validation scores and "
                 "`grpo_rollouts_needed` the rollouts at which it was first reached. "
                 "`rollouts_to_match_grpo` is the first curve point of each prompt optimizer whose "
                 "best-so-far validation score is at least GRPO's best; `rollouts_incl_val_eval` adds "
                 "the full validation evaluation of that candidate itself. `paper_rollouts` is the "
                 "number quoted in the paper for GEPA.")
    lines += ["", to_markdown(eff), ""]
    gepa = eff[eff["optimizer"] == "GEPA"]
    lines += ["Comparison with the paper:", ""]
    for _, row in gepa.iterrows():
        name = BENCH_NAMES.get(row["bench"], row["bench"])
        if pd.isna(row["rollouts_to_match_grpo"]):
            lines.append("- %s: GEPA never reaches GRPO's best validation score of %.2f (GEPA best %.2f)."
                         % (name, row["grpo_best_val"], row["opt_best_val"]))
            continue
        lines.append("- %s: GEPA reaches GRPO's best validation score (%.2f) after %d rollouts "
                     "(paper: %d); GRPO needed %d rollouts, so GEPA used %.1fx fewer "
                     "(%.1fx when the matching candidate's own validation evaluation is counted)."
                     % (name, row["grpo_best_val"], row["rollouts_to_match_grpo"], row["paper_rollouts"],
                        row["grpo_rollouts_needed"], row["speedup"], row["speedup_incl_val_eval"]))
    if len(gepa) and gepa["speedup"].notna().any():
        top = gepa.loc[gepa["speedup"].idxmax()]
        lines.append("- Largest ratio for GEPA: %.1fx on %s (paper: up to %dx)."
                     % (top["speedup"], BENCH_NAMES.get(top["bench"], top["bench"]), PAPER_MAX_SPEEDUP))
    lines.append("")

    lines += ["## Rollouts at which the best candidate was found (GEPA family)", ""]
    lines.append("`rollouts_at_best` is the discovery count of the candidate with the highest full "
                 "validation score, `budget` the rollout budget of the task.")
    lines += ["", to_markdown(best), ""]
    lines.append("- Best candidate found after %.0f%% of the budget on average (min %.0f%%, max %.0f%%) "
                 "over %d runs; in %d of them the best candidate is the last one found."
                 % (100 * best["fraction_of_budget"].mean(), 100 * best["fraction_of_budget"].min(),
                    100 * best["fraction_of_budget"].max(), len(best),
                    int((best["best_candidate_idx"] == best["num_candidates"] - 1).sum())))
    lines.append("")

    lines += ["## Budget accounting (GEPA family)", ""]
    lines.append("`train_calls` and `val_calls` are the counters on the last line of "
                 "metric_logs/train.jsonl and val.jsonl (`logged_calls` adds the dev counter, which is "
                 "0 for the GEPA family), `state_total_num_evals` is the counter GEPA itself keeps in "
                 "gepa_state.bin and checks against the budget before every iteration.")
    lines += ["", to_markdown(acc_gepa), ""]
    over = acc_gepa["logged_minus_budget"]
    lines.append("- Logged train+val calls minus budget: mean %+.0f (%+.1f%%), min %+d, max %+d; "
                 "%d of %d runs exceed the budget."
                 % (over.mean(), 100 * (acc_gepa["logged_calls"] / acc_gepa["budget"] - 1).mean(),
                    over.min(), over.max(), int((over > 0).sum()), len(over)))
    state_over = acc_gepa["state_minus_budget"]
    lines.append("- GEPA's own counter minus budget: mean %+.0f, min %+d, max %+d; the counter can "
                 "only overshoot by the last iteration (one subsample round plus one full validation "
                 "evaluation) because the loop stops once it reaches the budget."
                 % (state_over.mean(), state_over.min(), state_over.max()))
    diff = acc_gepa["logged_minus_state"]
    lines.append("- Logged calls minus GEPA's counter: mean %+.0f, min %+d, max %+d (metric calls "
                 "that GEPA did not count, e.g. retried or failed rollouts)."
                 % (diff.mean(), diff.min(), diff.max()))
    lines.append("")

    lines += ["## Budget accounting (MIPROv2 and GRPO, for reference)", ""]
    lines.append("`authors_total` is the MIPROv2 rollout count the authors' notebook computes "
                 "(bootstrapping + 35 per minibatch + valset size per full evaluation); the task "
                 "budgets were taken from the MIPROv2 runs. For AIME the MIPROv2 search calls were "
                 "logged under train/dev (val_calls is 0), so bootstrapping is taken as the logged "
                 "total minus the nominal search calls.")
    lines += ["", to_markdown(acc_other), ""]
    grpo = acc_other[acc_other["optimizer"] == "GRPO"]
    if len(grpo):
        lines.append("- GRPO: the 24000 budget counts training rollouts only (48 per step, 500 steps); "
                     "logged train calls are %d to %d, and the %d validation checkpoints (val_calls) "
                     "come on top of it."
                     % (grpo["train_calls"].min(), grpo["train_calls"].max(), 26))
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    global RUNS_DIR
    parser = argparse.ArgumentParser(description="Best validation score vs rollouts per optimizer run.")
    parser.add_argument("--logx", action="store_true",
                        help="use a symlog x axis; figures get a _logx suffix")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    args = parser.parse_args()
    RUNS_DIR = args.runs_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    runs = collect_runs()
    curves = curve_table(runs)
    curves.to_csv(os.path.join(OUT_DIR, "budget_curves.csv"), index=False)
    print("curve points:", len(curves), "runs with a curve:", curves["run"].nunique())

    for (bench, model), group in sorted(group_runs(runs).items()):
        plot_group(bench, model, group, args.logx)

    eff = sample_efficiency_table(runs)
    best = best_candidate_table(runs)
    acc_gepa, acc_other = budget_accounting_tables(runs)
    for name, df in [("sample_efficiency", eff), ("best_candidate_discovery", best),
                     ("budget_accounting_gepa", acc_gepa), ("budget_accounting_other", acc_other)]:
        df.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)
        with open(os.path.join(OUT_DIR, name + ".md"), "w", encoding="utf-8") as f:
            f.write(to_markdown(df) + "\n")

    text = summary_text(eff, best, acc_gepa, acc_other)
    with open(os.path.join(OUT_DIR, "sample_efficiency.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
