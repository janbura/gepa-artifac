"""Two small analyses of the shipped GEPA runs.

Part A (feedback texts): reads instruction_proposer_inpouts.jsonl of every
GEPA-family run, counts the examples shown to the reflection LM per call,
classifies the feedback of each example as success / failure with a few
keyword rules per task, and measures the size of the reflection prompt.

Part B (does Pareto selection matter?): replays full_program_trace from
gepa_state.bin and compares the parent GEPA actually picked with the parent a
greedy rule (highest full validation score among the candidates known at that
time) would have picked.  It also looks at the final candidate pool: how many
candidates are needed to reach the per-instance validation maxima and how far
that "oracle" is above the best single candidate.

Run from the repo root:
    python analysis/feedback_and_pareto.py
"""
import argparse
import json
import math
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken

RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")

GEPA_FAMILY = ("GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate")
BENCH_SHORT = {"AIMEBench": "AIME", "LiveBenchMathBench": "LiveBench",
               "HotpotQABench": "HotpotQA", "hoverBench": "HoVer",
               "IFBench": "IFBench", "Papillon": "PUPA"}
OPT_SHORT = {"GEPA": "GEPA", "GEPA-MERGE": "Merge", "Abl-SelectBestCandidate": "Abl"}
MODEL_SHORT = {"gpt-41-mini": "gpt", "qwen3-8b": "qwen"}


def parse_run_name(run):
    """Run folders are named <Bench>_<Program>_<Optimizer>_<Model>."""
    bench, program, optimizer, model = run.split("_")
    return bench, program, optimizer, model


def short_label(run):
    bench, _, optimizer, model = parse_run_name(run)
    return "%s/%s/%s" % (BENCH_SHORT[bench], OPT_SHORT[optimizer], MODEL_SHORT[model])


def gepa_family_runs(runs_dir):
    runs = []
    for run in sorted(os.listdir(runs_dir)):
        if parse_run_name(run)[2] in GEPA_FAMILY:
            runs.append(run)
    return runs


def write_markdown(df, path):
    """Minimal markdown table writer (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else "%.3f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_table(df, base_path):
    df.to_csv(base_path + ".csv", index=False, encoding="utf-8")
    write_markdown(df, base_path + ".md")


# Part A: feedback texts

def classify_feedback(bench, text):
    """Return 'success', 'failure' or 'unclear' for one example's feedback."""
    t = text.strip()
    if bench in ("AIMEBench", "LiveBenchMathBench"):
        m = re.match(r"Your answer is ([\d.]+)% correct", t)
        if m:  # partial credit (edit distance / positional matching)
            return "success" if float(m.group(1)) >= 100 else "failure"
        if t.startswith("Your answer is correct") or t.startswith("Correct answer detected"):
            return "success"
        if t.startswith("Your answer is incorrect") or t.startswith("Incorrect answer detected"):
            return "failure"
        if "must be a valid integer" in t:  # answer could not be parsed
            return "failure"
    elif bench == "HotpotQABench":
        if "is correct. Here's some additional context" in t:
            return "success"
        if "is incorrect. The correct answer is" in t:
            return "failure"
        # first-hop summariser: were all supporting documents already retrieved?
        m = re.search(r"still missing after first hop: \[(.*?)\]", t)
        if m:
            return "success" if m.group(1).strip() == "" else "failure"
        # second-hop query writer: "retrieved k out of m remaining", m is what is still missing
        m = re.search(r"successfully retrieved (\d+) out of (\d+) remaining", t)
        if m:
            return "success" if int(m.group(2)) == 0 else "failure"
        # the second summariser only receives the ideal summary, no verdict
    elif bench == "hoverBench":
        if "Missing evidence" in t:
            return "failure"
        if "correct and useful" in t:
            return "success"
    elif bench == "IFBench":
        if "did not follow the following instructions" in t:
            return "failure"
        if "correctly followed the following instructions" in t:
            return "success"
    elif bench == "Papillon":
        m = re.match(r"The overall score is (-?[\d.]+)", t)
        if m:  # mean of quality and (1 - leakage); only a perfect score counts as success
            return "success" if float(m.group(1)) >= 1.0 else "failure"
    return "unclear"


def split_examples(user_examples):
    """Yield the feedback text of every '# Example k' block."""
    for block in re.split(r"(?m)^# Example \d+\s*$", user_examples)[1:]:
        m = re.search(r"(?m)^## Feedback\s*$", block)
        yield block[m.end():] if m else ""


def feedback_stats_for_run(run, runs_dir, enc):
    bench = parse_run_name(run)[0]
    path = os.path.join(runs_dir, run, "instruction_proposer_inpouts.jsonl")
    calls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            inputs = rec["all_inputs"]
            labels = [classify_feedback(bench, fb) for fb in split_examples(inputs["user_examples_and_feedback"])]
            n_tok = {k: len(enc.encode(str(v), disallowed_special=())) for k, v in inputs.items()}
            calls.append({
                "n_examples": len(labels),
                "n_success": labels.count("success"),
                "n_failure": labels.count("failure"),
                "n_unclear": labels.count("unclear"),
                "instruction_tokens": n_tok["current_instruction_doc"],
                "examples_tokens": n_tok["user_examples_and_feedback"],
                "prompt_tokens": sum(n_tok.values()),
            })
    c = pd.DataFrame(calls)
    n_ex = c["n_examples"].sum()
    bench, _, optimizer, model = parse_run_name(run)
    return {
        "run": run, "benchmark": bench, "optimizer": optimizer, "model": model,
        "n_calls": len(c),
        "examples_per_call": c["n_examples"].mean(),
        "n_examples": int(n_ex),
        "frac_examples_success": c["n_success"].sum() / n_ex,
        "frac_examples_failure": c["n_failure"].sum() / n_ex,
        "frac_examples_unclear": c["n_unclear"].sum() / n_ex,
        "frac_calls_with_failure": (c["n_failure"] > 0).mean(),
        "frac_calls_all_success": (c["n_success"] == c["n_examples"]).mean(),
        "frac_calls_all_failure": (c["n_failure"] == c["n_examples"]).mean(),
        "frac_calls_with_unclear": (c["n_unclear"] > 0).mean(),
        "mean_prompt_tokens": c["prompt_tokens"].mean(),
        "median_prompt_tokens": c["prompt_tokens"].median(),
        "max_prompt_tokens": int(c["prompt_tokens"].max()),
        "mean_examples_tokens": c["examples_tokens"].mean(),
        "mean_instruction_tokens": c["instruction_tokens"].mean(),
    }


def pooled_feedback(df):
    """Pool the per-run numbers per benchmark (weighted by calls / examples)."""
    rows = []
    for bench, g in df.groupby("benchmark", sort=False):
        n_calls = g["n_calls"].sum()
        n_ex = g["n_examples"].sum()
        rows.append({
            "benchmark": bench, "n_runs": len(g), "n_calls": int(n_calls),
            "frac_examples_failure": (g["frac_examples_failure"] * g["n_examples"]).sum() / n_ex,
            "frac_examples_unclear": (g["frac_examples_unclear"] * g["n_examples"]).sum() / n_ex,
            "frac_calls_with_failure": (g["frac_calls_with_failure"] * g["n_calls"]).sum() / n_calls,
            "frac_calls_all_success": (g["frac_calls_all_success"] * g["n_calls"]).sum() / n_calls,
            "frac_calls_all_failure": (g["frac_calls_all_failure"] * g["n_calls"]).sum() / n_calls,
            "mean_prompt_tokens": (g["mean_prompt_tokens"] * g["n_calls"]).sum() / n_calls,
        })
    return pd.DataFrame(rows)


# Part B: Pareto selection vs greedy selection

def load_state(run, runs_dir):
    with open(os.path.join(runs_dir, run, "gepa_state.bin"), "rb") as f:
        return pickle.load(f)


def replay_selection(state):
    """Walk the trace and compare each parent GEPA picked with the greedy pick.

    The greedy pick is the candidate with the highest full validation score
    among the candidates that existed at that iteration (ties: lowest index,
    like the optimizer's own idxmax).  Merge iterations pick two parents in a
    different way and are kept apart.
    """
    scores = [float(s) for s in state["program_full_scores_val_set"]]
    rows = []
    n_known = 1  # candidate 0 is the seed program
    record = scores[0]
    for step in state["full_program_trace"]:
        new_idx = step.get("new_program_idx")
        known = scores[:n_known]
        best_score = max(known)
        sel = None if step.get("merged") else step["selected_program_candidate"]
        rows.append({
            "iteration": step["i"],
            "kind": "merge" if sel is None else "reflection",
            "selected": sel,
            "greedy_pick": known.index(best_score),
            "selected_score": np.nan if sel is None else scores[sel],
            "best_score_at_time": best_score,
            "is_best_pick": sel is not None and scores[sel] >= best_score - 1e-9,
            # no proposal is made when the minibatch is skipped (all perfect / no usable trace)
            "proposed": sel is not None and "new_subsample_scores" in step,
            "accepted": new_idx is not None,
            "new_program_idx": new_idx,
            "new_score": np.nan if new_idx is None else scores[new_idx],
            "new_record": new_idx is not None and scores[new_idx] > record + 1e-9,
        })
        if new_idx is not None:
            assert new_idx == n_known, "candidates should be numbered in discovery order"
            n_known += 1
            record = max(record, scores[new_idx])
    assert n_known == len(scores)
    return rows


def ancestry(parents, idx):
    """All ancestors of candidate idx (merges have two parents), idx included."""
    seen, todo = set(), [idx]
    while todo:
        k = todo.pop()
        if k in seen or k is None:
            continue
        seen.add(k)
        todo.extend(parents[k])
    return seen


def remove_dominated(fronts, scores):
    """Port of the optimizer's own routine: drop a candidate when every
    instance front it sits on also holds another remaining candidate
    (lowest-scoring candidates are dropped first)."""
    programs = sorted({p for front in fronts for p in front}, key=lambda p: scores[p])
    dominated = set()
    changed = True
    while changed:
        changed = False
        for y in programs:
            if y in dominated:
                continue
            others = set(programs) - {y} - dominated
            if all(any(o in f for o in others) for f in fronts if y in f):
                dominated.add(y)
                changed = True
                break
    return [p for p in programs if p not in dominated]


def greedy_cover(fronts, scores):
    """Greedy set cover: how many candidates are needed so that every
    validation instance has one of them on its front."""
    uncovered = set(range(len(fronts)))
    chosen = []
    while uncovered:
        candidates = {p for t in uncovered for p in fronts[t]}
        best = max(candidates, key=lambda p: (sum(p in fronts[t] for t in uncovered), scores[p], -p))
        chosen.append(best)
        uncovered = {t for t in uncovered if best not in fronts[t]}
    return chosen


def score_matrix(state):
    """Per-candidate, per-instance validation scores.

    In the shipped states the seed's row is the same object as the running
    per-instance maximum (pareto_front_valset), so the seed's own scores are
    lost.  Where candidate 0 is still on an instance's front the seed equals
    that maximum; for binary metrics it must otherwise be 0; anything else is
    left as NaN.
    """
    M = np.array([[float(x) for x in row] for row in state["prog_candidate_val_subscores"]])
    fronts = state["program_at_pareto_front_valset"]
    front_max = np.array([float(x) for x in state["pareto_front_valset"]])
    binary = set(np.unique(M[1:])) <= {0.0, 1.0}
    seed = np.full(M.shape[1], np.nan)
    for t in range(M.shape[1]):
        if 0 in fronts[t]:
            seed[t] = front_max[t]
        elif binary:
            seed[t] = 0.0
    M[0] = seed
    return M, binary


def pareto_stats_for_run(run, runs_dir, matrix_dir):
    state = load_state(run, runs_dir)
    scores = [float(s) for s in state["program_full_scores_val_set"]]
    assert scores == [float(s) for s in state["per_program_tracked_scores"]]
    rows = replay_selection(state)
    it = pd.DataFrame(rows)
    it.insert(0, "run", run)

    refl = it[it["kind"] == "reflection"]
    best_picks = refl[refl["is_best_pick"]]
    other_picks = refl[~refl["is_best_pick"]]

    # how was each candidate created, and what does the final best descend from?
    created_by = {}
    for r in rows:
        if r["accepted"]:
            created_by[r["new_program_idx"]] = r["kind"] if r["kind"] == "merge" else (
                "best_pick" if r["is_best_pick"] else "nonbest_pick")
    parents = state["parent_program_for_candidate"]
    final_best = int(np.argmax(scores))
    family = ancestry(parents, final_best) - {0}
    kinds = [created_by[k] for k in family]

    # final pool
    fronts = state["program_at_pareto_front_valset"]
    front_max = np.array([float(x) for x in state["pareto_front_valset"]])
    on_any_front = set().union(*fronts)
    unique_best = {next(iter(f)) for f in fronts if len(f) == 1}
    nondominated = remove_dominated(fronts, scores)
    cover = greedy_cover(fronts, scores)
    M, binary = score_matrix(state)
    assert np.allclose(np.nanmax(M, axis=0), front_max)
    if binary:
        assert abs(100 * M[0].mean() - scores[0]) < 0.05
    pd.DataFrame(M, index=pd.Index(range(len(scores)), name="candidate")).to_csv(
        os.path.join(matrix_dir, run + ".csv"), float_format="%.4f", encoding="utf-8")

    bench, _, optimizer, model = parse_run_name(run)
    stats = {
        "run": run, "benchmark": bench, "optimizer": optimizer, "model": model,
        "n_candidates": len(scores), "n_val": len(fronts),
        "n_iterations": len(it), "n_reflection_iters": len(refl), "n_merge_iters": int((it["kind"] == "merge").sum()),
        "n_accepted_reflection": int(refl["accepted"].sum()),
        "n_accepted_merge": int(it.loc[it["kind"] == "merge", "accepted"].sum()),
        "frac_nonbest_picks": len(other_picks) / len(refl),
        "n_best_picks": len(best_picks), "n_nonbest_picks": len(other_picks),
        "acc_rate_best_picks": best_picks["accepted"].mean() if len(best_picks) else np.nan,
        "acc_rate_nonbest_picks": other_picks["accepted"].mean() if len(other_picks) else np.nan,
        "frac_skipped_best_picks": 1 - best_picks["proposed"].mean() if len(best_picks) else np.nan,
        "frac_skipped_nonbest_picks": 1 - other_picks["proposed"].mean() if len(other_picks) else np.nan,
        "acc_rate_best_picks_proposed": best_picks.loc[best_picks["proposed"], "accepted"].mean() if best_picks["proposed"].any() else np.nan,
        "acc_rate_nonbest_picks_proposed": other_picks.loc[other_picks["proposed"], "accepted"].mean() if other_picks["proposed"].any() else np.nan,
        "mean_gap_nonbest_picks": (other_picks["best_score_at_time"] - other_picks["selected_score"]).mean() if len(other_picks) else np.nan,
        "n_record_from_best_picks": int(best_picks["new_record"].sum()),
        "n_record_from_nonbest_picks": int(other_picks["new_record"].sum()),
        "n_record_from_merge": int(it.loc[it["kind"] == "merge", "new_record"].sum()),
        "final_best_idx": final_best,
        "final_best_created_by": created_by[final_best] if final_best else "seed",
        "final_best_n_ancestors": len(family),
        "final_best_ancestors_from_nonbest_picks": kinds.count("nonbest_pick"),
        "final_best_ancestors_from_merge": kinds.count("merge"),
        "seed_score": scores[0],
        "best_single_score": max(scores),
        "oracle_score": 100 * front_max.mean(),
        "oracle_gain": 100 * front_max.mean() - max(scores),
        "frac_instances_best_single_on_front": np.mean([final_best in f for f in fronts]),
        "n_on_any_front": len(on_any_front),
        "n_unique_best": len(unique_best),
        "n_nondominated": len(nondominated),
        "greedy_cover_size": len(cover),
        "metric_binary": binary,
        "n_seed_entries_unknown": int(np.isnan(M[0]).sum()),
    }
    return stats, it


def two_proportion_test(k1, n1, k2, n2):
    """Normal-approximation test that two rates are equal: (z, two-sided p)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))
    z = (k2 / n2 - k1 / n1) / se if se > 0 else 0.0
    return z, math.erfc(abs(z) / math.sqrt(2))


def pooled_pareto(df, iters):
    """Pool the selection comparison over runs, per optimizer."""
    rows = []
    for opt, g in df.groupby("optimizer", sort=False):
        sub = iters[(iters["run"].isin(g["run"])) & (iters["kind"] == "reflection")]
        best = sub[sub["is_best_pick"]]
        other = sub[~sub["is_best_pick"]]
        best_p = best[best["proposed"]]
        other_p = other[other["proposed"]]
        z, p = (np.nan, np.nan)
        if len(other_p):
            z, p = two_proportion_test(best_p["accepted"].sum(), len(best_p), other_p["accepted"].sum(), len(other_p))
        rows.append({
            "optimizer": opt, "n_runs": len(g),
            "n_reflection_iters": len(sub),
            "n_best_picks": len(best), "n_nonbest_picks": len(other),
            "frac_nonbest_picks": len(other) / len(sub),
            "acc_rate_best_picks": best["accepted"].mean(),
            "acc_rate_nonbest_picks": other["accepted"].mean() if len(other) else np.nan,
            "frac_skipped_best_picks": 1 - best["proposed"].mean(),
            "frac_skipped_nonbest_picks": 1 - other["proposed"].mean() if len(other) else np.nan,
            "n_best_picks_proposed": len(best_p), "n_nonbest_picks_proposed": len(other_p),
            "acc_rate_best_picks_proposed": best_p["accepted"].mean(),
            "acc_rate_nonbest_picks_proposed": other_p["accepted"].mean() if len(other_p) else np.nan,
            "acc_rate_proposed_diff_z": z, "acc_rate_proposed_diff_p": p,
            "n_record_from_best_picks": int(best["new_record"].sum()),
            "n_record_from_nonbest_picks": int(other["new_record"].sum()),
            "record_rate_best_picks": best["new_record"].mean(),
            "record_rate_nonbest_picks": other["new_record"].mean() if len(other) else np.nan,
            "final_best_n_ancestors": int(g["final_best_n_ancestors"].sum()),
            "final_best_ancestors_from_nonbest_picks": int(g["final_best_ancestors_from_nonbest_picks"].sum()),
            "n_runs_final_best_from_nonbest_pick": int((g["final_best_created_by"] == "nonbest_pick").sum()),
            "mean_oracle_gain": g["oracle_gain"].mean(),
            "mean_greedy_cover_size": g["greedy_cover_size"].mean(),
            "mean_n_nondominated": g["n_nondominated"].mean(),
            "mean_n_candidates": g["n_candidates"].mean(),
        })
    return pd.DataFrame(rows)


# figures

BLUE, ORANGE = "#2a78d6", "#eb6834"


def style_axis(ax):
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)


def paired_barh(ax, labels, a, b, name_a, name_b, xlabel):
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y - h / 2 - 0.02, a, height=h, color=BLUE, label=name_a)
    ax.barh(y + h / 2 + 0.02, b, height=h, color=ORANGE, label=name_b)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style_axis(ax)


def make_figures(feedback, pareto, fig_dir):
    labels = [short_label(r) for r in feedback["run"]]
    fig, ax = plt.subplots(figsize=(7, 8))
    paired_barh(ax, labels, feedback["frac_calls_with_failure"], feedback["frac_calls_all_failure"],
                "at least one failed example", "all examples failed", "fraction of reflection calls")
    ax.set_title("What the reflection LM saw per call", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feedback_failure_fraction.png"), dpi=150)
    plt.close(fig)

    labels = [short_label(r) for r in pareto["run"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    paired_barh(axes[0], labels, pareto["acc_rate_best_picks_proposed"], pareto["acc_rate_nonbest_picks_proposed"],
                "parent was the best candidate", "parent was not the best", "acceptance rate among evaluated proposals")
    axes[0].set_title("Acceptance rate by type of parent pick", fontsize=10)
    paired_barh(axes[1], labels, pareto["best_single_score"], pareto["oracle_score"],
                "best single candidate", "oracle: best candidate per instance", "validation score (%)")
    axes[1].set_title("Headroom in the final candidate pool", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pareto_selection.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Feedback texts and Pareto selection in the GEPA runs.")
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument("--only", default=None, help="only runs whose folder name contains this text")
    args = ap.parse_args()

    matrix_dir = os.path.join(args.out_dir, "val_score_matrices")
    for d in (args.out_dir, args.fig_dir, matrix_dir):
        os.makedirs(d, exist_ok=True)

    runs = gepa_family_runs(args.runs_dir)
    if args.only:
        runs = [r for r in runs if args.only in r]
    print("%d GEPA-family runs" % len(runs))

    # Part A
    enc = tiktoken.get_encoding("cl100k_base")
    feedback = pd.DataFrame([feedback_stats_for_run(r, args.runs_dir, enc) for r in runs])
    save_table(feedback, os.path.join(args.out_dir, "feedback_stats"))
    by_bench = pooled_feedback(feedback)
    save_table(by_bench, os.path.join(args.out_dir, "feedback_stats_by_benchmark"))

    # Part B
    stats, iters = [], []
    for run in runs:
        s, it = pareto_stats_for_run(run, args.runs_dir, matrix_dir)
        stats.append(s)
        iters.append(it)
    pareto = pd.DataFrame(stats)
    iters = pd.concat(iters, ignore_index=True)
    save_table(pareto, os.path.join(args.out_dir, "pareto_stats"))
    iters.to_csv(os.path.join(args.out_dir, "pareto_selection_iterations.csv"), index=False, encoding="utf-8")
    pooled = pooled_pareto(pareto, iters)
    save_table(pooled, os.path.join(args.out_dir, "pareto_stats_pooled"))

    make_figures(feedback, pareto, args.fig_dir)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["run", "n_calls", "examples_per_call", "frac_examples_failure", "frac_examples_unclear",
            "frac_calls_with_failure", "frac_calls_all_success", "frac_calls_all_failure", "mean_prompt_tokens", "max_prompt_tokens"]
    print("\nPart A: feedback texts per run")
    print(feedback[cols].round(3).to_string(index=False))
    print("\nPart A: pooled per benchmark")
    print(by_bench.round(3).to_string(index=False))
    cols = ["run", "n_candidates", "n_reflection_iters", "frac_nonbest_picks", "acc_rate_best_picks", "acc_rate_nonbest_picks",
            "frac_skipped_best_picks", "frac_skipped_nonbest_picks", "acc_rate_best_picks_proposed", "acc_rate_nonbest_picks_proposed",
            "mean_gap_nonbest_picks", "n_record_from_best_picks", "n_record_from_nonbest_picks", "final_best_created_by",
            "final_best_n_ancestors", "final_best_ancestors_from_nonbest_picks", "best_single_score", "oracle_score",
            "n_on_any_front", "n_nondominated", "n_unique_best", "greedy_cover_size"]
    print("\nPart B: Pareto vs greedy selection per run")
    print(pareto[cols].round(3).to_string(index=False))
    print("\nPart B: pooled per optimizer")
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
