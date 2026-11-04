"""Search-tree statistics for the GEPA, GEPA-MERGE and Abl-SelectBestCandidate runs.

For every run the script loads gepa_state.bin (a plain pickle of the optimizer
state), cross-checks a few counts against run_log.txt, writes one row per run to
analysis/outputs/search_tree_stats.csv (all columns) and .md (main columns),
writes a per optimizer/model summary, and draws the candidate tree of every run
to analysis/figures/tree_<run>.png.

Run from the repo root:
    python analysis/search_tree_stats.py
"""

import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")
OPTIMIZERS = ("GEPA", "GEPA-MERGE", "Abl-SelectBestCandidate")

# run_log.txt lines (all start with "Iteration N:") that are counted to cross-check the trace
LOG_PATTERNS = {
    "log_selected": "Selected program candidate",
    "log_better": "New subsample score is better",
    "log_merged": "Merged programs",
    "log_skip_merge": "skipping merge",
    "log_no_merge_cand": "No merge candidates found",
}

# columns shown in the markdown table (the csv has all of them)
MD_COLUMNS = [
    "run", "n_candidates", "n_iterations", "n_accepted", "acceptance_rate",
    "accept_rate_tested", "frac_sel_seed", "frac_sel_current_best",
    "pareto_front_size", "pareto_front_size_excl_ties", "pareto_front_nondominated",
    "n_best_off_front", "frac_best_off_front",
    "best_idx", "best_val_score", "best_depth", "max_depth", "best_is_last",
    "n_merge_invoked", "n_merge_attempts", "n_merges_kept", "log_consistent",
]

SUMMARY_COLUMNS = [
    "n_candidates", "n_iterations", "acceptance_rate", "accept_rate_tested",
    "frac_sel_seed", "frac_sel_current_best", "pareto_front_size",
    "pareto_front_size_excl_ties", "pareto_front_nondominated", "n_best_off_front",
    "frac_best_off_front", "best_depth", "max_depth", "best_is_last",
    "n_merge_invoked", "n_merge_attempts", "n_merges_kept",
]

MD_NOTES = """Main columns; the csv next to this file has all of them.

- acceptance_rate: iterations that added a candidate (reflective proposal or kept merge) / iterations
- accept_rate_tested: accepted reflective proposals / proposals that were scored on a minibatch
- frac_sel_seed, frac_sel_current_best: share of parent selections that picked the seed program,
  or the candidate with the best full validation score known at that time
- pareto_front_size: distinct candidates on any per-instance validation front (ties included);
  excl_ties drops instances where every candidate ties; nondominated applies the optimizer's
  dominated-program removal
- n_best_off_front: validation instances whose front does not contain the final best candidate
- log_consistent: the trace counts agree with the run_log.txt lines

"""


def parse_run_name(run):
    # e.g. hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b
    bench, program, optimizer, model = run.split("_")
    return bench, program, optimizer, model


def load_state(run_dir):
    with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
        return pickle.load(f)


def count_log_lines(run_dir):
    counts = {key: 0 for key in LOG_PATTERNS}
    path = os.path.join(run_dir, "run_log.txt")
    if not os.path.exists(path):
        return counts
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("Iteration"):
                continue
            for key, pattern in LOG_PATTERNS.items():
                if pattern in line:
                    counts[key] += 1
    return counts


def node_depths(parents):
    # generation of each candidate: seed is 0, a child sits one below its deepest parent
    depth = []
    for ps in parents:
        ps = [p for p in ps if p is not None]
        depth.append(0 if not ps else 1 + max(depth[p] for p in ps))
    return depth


def remove_dominated(fronts, scores):
    """Same greedy rule as the optimizer's remove_dominated_programs: drop a
    program (lowest aggregate score first) when every instance front it sits on
    also holds another surviving program. Returns the surviving candidate ids."""
    progs = sorted({p for front in fronts for p in front}, key=lambda p: scores[p])
    dominated = set()
    changed = True
    while changed:
        changed = False
        for y in progs:
            if y in dominated:
                continue
            others = set(progs) - dominated - {y}
            if all(front & others for front in fronts if y in front):
                dominated.add(y)
                changed = True
                break
    return [p for p in progs if p not in dominated]


def analyze_run(run, state, log_counts):
    bench, program, optimizer, model = parse_run_name(run)
    scores = state["program_full_scores_val_set"]
    parents = state["parent_program_for_candidate"]
    trace = state["full_program_trace"]
    fronts = state["program_at_pareto_front_valset"]
    n_cand = len(scores)
    n_iter = state["i"] + 1
    assert n_iter == len(trace), run

    best = int(np.argmax(scores))  # first index among ties, like the optimizer's idxmax
    depth = node_depths(parents)

    # walk the trace in order so that "current best" only looks at candidates found so far
    n_known = 1
    n_sel = n_sel_seed = n_sel_best = 0
    selected = []
    for t in trace:
        if "selected_program_candidate" in t:
            sel = t["selected_program_candidate"]
            n_sel += 1
            n_sel_seed += int(sel == 0)
            n_sel_best += int(sel == int(np.argmax(scores[:n_known])))
            selected.append(sel)
        if "new_program_idx" in t:
            assert t["new_program_idx"] == n_known, run
            n_known += 1
    assert n_known == n_cand, run

    n_accepted = sum(1 for t in trace if "new_program_idx" in t)
    merge_invoked = sum(1 for t in trace if t.get("invoked_merge"))
    merge_attempts = sum(1 for t in trace if t.get("merged"))
    merges_kept = sum(1 for t in trace if t.get("merged") and "new_program_idx" in t)
    n_accept_reflect = n_accepted - merges_kept
    # proposals that were actually scored on a minibatch (skipped when all scores were perfect)
    n_tested = sum(1 for t in trace if "new_subsample_scores" in t)
    n_no_feedback = sum(1 for t in trace
                        if "selected_program_candidate" in t and t.get("subsample_scores") is None)

    front_members = set().union(*fronts)
    # instances where every candidate ties (e.g. all score 0) put all candidates on the front
    n_tied = sum(1 for front in fronts if len(front) == n_cand)
    strict_members = set().union(*[front for front in fronts if len(front) < n_cand])
    nondominated = remove_dominated(fronts, scores)
    best_off_front = sum(1 for front in fronts if best not in front)

    best_iter = -1  # the seed program has no discovery iteration
    for t in trace:
        if t.get("new_program_idx") == best:
            best_iter = t["i"]

    log_ok = (log_counts["log_selected"] == n_sel
              and log_counts["log_better"] == n_accept_reflect
              and log_counts["log_merged"] == merge_attempts
              and log_counts["log_skip_merge"] == merge_attempts - merges_kept
              and log_counts["log_no_merge_cand"] == merge_invoked - merge_attempts)

    row = {
        "run": run,
        "benchmark": bench,
        "optimizer": optimizer,
        "model": model,
        "n_val_instances": len(fronts),
        "n_candidates": n_cand,
        "n_iterations": n_iter,
        "n_accepted": n_accepted,
        "n_accepted_reflect": n_accept_reflect,
        "n_merges_kept": merges_kept,
        "acceptance_rate": n_accepted / n_iter,
        "n_proposals_tested": n_tested,
        "accept_rate_tested": n_accept_reflect / n_tested if n_tested else np.nan,
        "n_no_feedback_iters": n_no_feedback,
        "n_selected": n_sel,
        "n_sel_seed": n_sel_seed,
        "frac_sel_seed": n_sel_seed / n_sel if n_sel else np.nan,
        "n_distinct_selected": len(set(selected)),
        "frac_sel_current_best": n_sel_best / n_sel if n_sel else np.nan,
        "always_selects_current_best": n_sel_best == n_sel,
        "n_children_of_seed": sum(1 for ps in parents if ps == [0]),
        "pareto_front_size": len(front_members),
        "n_tied_instances": n_tied,
        "pareto_front_size_excl_ties": len(strict_members),
        "pareto_front_nondominated": len(nondominated),
        "n_best_off_front": best_off_front,
        "frac_best_off_front": best_off_front / len(fronts),
        "seed_val_score": scores[0],
        "best_idx": best,
        "best_val_score": scores[best],
        "best_depth": depth[best],
        "max_depth": max(depth),
        "best_is_last": best == n_cand - 1,
        "best_is_merge": len(parents[best]) == 2,
        "best_found_iter": best_iter,
        "best_found_budget_frac": state["num_metric_calls_by_discovery"][best] / state["total_num_evals"],
        "n_merge_invoked": merge_invoked,
        "n_merge_attempts": merge_attempts,
        "n_merge_rejected": merge_attempts - merges_kept,
        "n_merge_no_candidates": merge_invoked - merge_attempts,
        "total_num_evals": state["total_num_evals"],
        "log_consistent": log_ok,
    }
    row.update(log_counts)
    return row


def draw_tree(run, state, best, depth, fig_path):
    scores = state["program_full_scores_val_set"]
    parents = state["parent_program_for_candidate"]
    n = len(scores)

    # x = generation, y = slot inside the generation (centred), in index order
    generations = {}
    for k in range(n):
        generations.setdefault(depth[k], []).append(k)
    pos = {}
    for d, members in generations.items():
        for j, k in enumerate(members):
            pos[k] = (d, j - (len(members) - 1) / 2.0)
    width = max(len(m) for m in generations.values())
    n_gen = max(depth) + 1

    fig, ax = plt.subplots(figsize=(max(5.0, 1.0 * n_gen + 2.0), max(3.0, 0.65 * width + 1.5)))
    # edges first so the nodes sit on top; a merge child has two parents and dashed edges
    for k in range(n):
        ps = [p for p in parents[k] if p is not None]
        style = "--" if len(ps) == 2 else "-"
        for p in ps:
            ax.plot([pos[p][0], pos[k][0]], [pos[p][1], pos[k][1]], style, color="0.6", lw=1, zorder=1)

    # single-hue ramp, cut below 25% so the lightest node is still visible on white
    cmap = LinearSegmentedColormap.from_list("blues", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    norm = Normalize(vmin=min(scores), vmax=max(scores))
    xs = [pos[k][0] for k in range(n)]
    ys = [pos[k][1] for k in range(n)]
    ax.scatter(xs, ys, c=scores, cmap=cmap, norm=norm, s=320, edgecolors="0.3", linewidths=0.8, zorder=2)
    ax.scatter([pos[best][0]], [pos[best][1]], s=620, facecolors="none", edgecolors="#d9541e",
               linewidths=2.5, zorder=3)
    for k in range(n):
        color = "white" if norm(scores[k]) > 0.55 else "black"
        ax.text(pos[k][0], pos[k][1], str(k), ha="center", va="center", fontsize=7, color=color, zorder=4)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                 label="full validation score (%)")
    ax.set_xticks(range(n_gen))
    ax.set_yticks([])
    ax.set_xlim(-0.6, n_gen - 0.4)
    ax.set_ylim(-width / 2.0 - 0.7, width / 2.0 + 0.7)
    ax.set_xlabel("generation (depth in parent tree)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("%s\nbest candidate %d (%.2f) outlined in orange; dashed edges = merge"
                 % (run, best, scores[best]), fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else ("%.3f" % v).rstrip("0").rstrip("."))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Search-tree statistics of the GEPA family runs.")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--fig-dir", default=FIG_DIR)
    parser.add_argument("--no-figures", action="store_true", help="skip the tree figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        run = os.path.basename(run_dir)
        if parse_run_name(run)[2] not in OPTIMIZERS:
            continue
        if not os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
            print("no gepa_state.bin in", run)
            continue
        state = load_state(run_dir)
        row = analyze_run(run, state, count_log_lines(run_dir))
        rows.append(row)
        if not args.no_figures:
            draw_tree(run, state, row["best_idx"], node_depths(state["parent_program_for_candidate"]),
                      os.path.join(args.fig_dir, "tree_%s.png" % run))
        print("%-72s cands=%3d iters=%4d accepted=%3d front=%3d merges=%d/%d log_ok=%s"
              % (run, row["n_candidates"], row["n_iterations"], row["n_accepted"],
                 row["pareto_front_size"], row["n_merges_kept"], row["n_merge_attempts"],
                 row["log_consistent"]))

    df = pd.DataFrame(rows).sort_values(["optimizer", "model", "benchmark"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.out_dir, "search_tree_stats.csv"), index=False)
    with open(os.path.join(args.out_dir, "search_tree_stats.md"), "w", encoding="utf-8") as f:
        f.write("# Search-tree statistics per run (seed 0)\n\n")
        f.write(MD_NOTES)
        f.write(to_markdown(df[MD_COLUMNS]))

    # means per optimizer and model (booleans become fractions)
    summary = df.groupby(["optimizer", "model"])[SUMMARY_COLUMNS].mean().reset_index()
    summary.insert(2, "n_runs", df.groupby(["optimizer", "model"]).size().values)
    summary.to_csv(os.path.join(args.out_dir, "search_tree_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "search_tree_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Search-tree statistics, mean per optimizer and model\n\n")
        f.write(to_markdown(summary))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print()
    print(summary.round(3).to_string(index=False))
    print()
    print("runs whose log counts disagree with the trace:",
          list(df.loc[~df["log_consistent"], "run"]) or "none")
    abl = df[df["optimizer"] == "Abl-SelectBestCandidate"]
    print("Abl runs always selecting the current best candidate: %d of %d"
          % (abl["always_selects_current_best"].sum(), len(abl)))


if __name__ == "__main__":
    main()
