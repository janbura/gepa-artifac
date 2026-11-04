"""Prompt evolution in the GEPA artifact runs (seed 0).

Three parts:

1. Reflection calls. Every GEPA-family run (GEPA, GEPA-MERGE, Abl-SelectBestCandidate)
   appends one line per call of the instruction proposer to
   instruction_proposer_inpouts.jsonl. We measure the length of the current
   instruction, the proposed instruction and the feedback text given to the
   reflection LM (characters and cl100k tokens), plot the instruction length per
   call for every run, and compare the number of calls with the paper's appendix.
2. Final prompt sizes. For every non-baseline run the optimized program is loaded
   and the tokens of each predictor's instruction and few-shot demos are counted,
   so MIPROv2 can be compared with GEPA and GEPA-MERGE (the paper says GEPA
   prompts are up to 9.2x shorter).
3. Prompt examples. Seed and final instructions of two GEPA runs, verbatim.

Run from the repository root:  python analysis/prompt_evolution.py
"""
import argparse
import csv
import json
import os
import pickle
import re
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tiktoken

# all paths are relative to the repository root
RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
OUT_DIR = os.path.join("analysis", "outputs")
FIG_DIR = os.path.join("analysis", "figures")

TASK_NAMES = {"AIMEBench": "AIME", "LiveBenchMathBench": "LiveBench", "HotpotQABench": "HotpotQA",
              "IFBench": "IFBench", "hoverBench": "HoVer", "Papillon": "PUPA"}
TASK_ORDER = ["HotpotQA", "IFBench", "HoVer", "PUPA", "AIME", "LiveBench"]
MODELS = ["gpt-41-mini", "qwen3-8b"]
COMPARED_OPTIMIZERS = ["MIPROv2-Heavy", "GEPA", "GEPA-MERGE"]
EXAMPLE_RUNS = ["AIMEBench_CoT_GEPA_gpt-41-mini", "HotpotQABench_HotpotMultiHop_GEPA_gpt-41-mini"]

# number of reflection LM calls in the paper's appendix table
PAPER_REFLECTION_CALLS = {
    "gpt-41-mini": {"AIME": 24, "LiveBench": 34, "HotpotQA": 69, "IFBench": 21, "HoVer": 92, "PUPA": 46},
    "qwen3-8b": {"AIME": 90, "LiveBench": 38, "HotpotQA": 64, "IFBench": 17, "HoVer": 50, "PUPA": 38},
}

# categorical colours in a fixed order, plus chart chrome
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text):
    return len(ENC.encode(text, disallowed_special=()))


def parse_run_name(run):
    bench, program, optimizer, model = run.split("_")
    return {"run": run, "task": TASK_NAMES[bench], "program": program,
            "optimizer": optimizer, "model": model}


def list_runs():
    return sorted(os.listdir(RUNS_DIR))


def load_gepa_state(run):
    with open(os.path.join(RUNS_DIR, run, "gepa_state.bin"), "rb") as f:
        return pickle.load(f)


def read_test_score(run):
    with open(os.path.join(RUNS_DIR, run, "evaluation_results", "evaluation_result.txt")) as f:
        return float(next(csv.DictReader(f))["score"])


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return ("%.2f" % value).rstrip("0").rstrip(".")
    return str(value)


def save_table(df, name):
    """Write a dataframe as csv and as a markdown table with the same stem."""
    df.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    with open(os.path.join(OUT_DIR, name + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- part 1

def read_reflection_calls(run):
    """One row per reflection call, with the lengths of the texts involved."""
    rows = []
    path = os.path.join(RUNS_DIR, run, "instruction_proposer_inpouts.jsonl")
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            rec = json.loads(line)
            cur = rec["all_inputs"]["current_instruction_doc"]
            new = rec["new_instruction"]
            feedback = rec["all_inputs"]["user_examples_and_feedback"]
            rows.append({"call_index": k,
                         "len_current": len(cur), "len_new": len(new), "len_feedback_input": len(feedback),
                         "len_current_tokens": n_tokens(cur), "len_new_tokens": n_tokens(new),
                         "len_feedback_input_tokens": n_tokens(feedback)})
    return rows


def reflection_call_iterations(run, n_calls, state):
    """Iteration number and updated predictor of each reflection call.

    The run log prints one 'Proposed new instruction' line per successful proposer
    call, in the same order as the jsonl lines, and the GEPA state remembers which
    predictor every iteration worked on."""
    with open(os.path.join(RUNS_DIR, run, "run_log.txt"), encoding="utf-8", errors="replace") as f:
        iters = [int(m) for m in re.findall(r"^Iteration (\d+): Proposed new instruction:", f.read(), re.M)]
    if len(iters) != n_calls:
        print("  warning: %d logged proposals but %d jsonl lines in %s" % (len(iters), n_calls, run))
        return [None] * n_calls, [None] * n_calls
    names = state["list_of_named_predictors"]
    trace = state["full_program_trace"]
    preds = [names[trace[i - 1]["predictor_to_update_id"]] for i in iters]
    return iters, preds


def plot_instruction_lengths(run, df, predictor_names):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(df["call_index"], df["len_current_tokens"], color=MUTED, lw=1.2,
            label="current instruction (proposer input)")
    if df["predictor"].notna().all():
        for j, pred in enumerate(predictor_names):
            part = df[df["predictor"] == pred]
            if len(part):
                ax.plot(part["call_index"], part["len_new_tokens"], color=COLORS[j % len(COLORS)],
                        lw=1.5, marker="o", ms=3.5, label="proposed for " + pred)
    else:
        ax.plot(df["call_index"], df["len_new_tokens"], color=COLORS[0], lw=1.5, marker="o", ms=3.5,
                label="proposed instruction")
    style_axes(ax)
    ax.set_xlabel("reflection call index", color=INK, fontsize=9)
    ax.set_ylabel("instruction length (cl100k tokens)", color=INK, fontsize=9)
    ax.set_title(run, color=INK, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "instruction_length_%s.png" % run), dpi=150)
    plt.close(fig)


def analyze_reflection_calls(skip_figures):
    all_rows, count_rows = [], []
    for run in list_runs():
        if not os.path.isfile(os.path.join(RUNS_DIR, run, "instruction_proposer_inpouts.jsonl")):
            continue
        info = parse_run_name(run)
        state = load_gepa_state(run)
        rows = read_reflection_calls(run)
        iters, preds = reflection_call_iterations(run, len(rows), state)
        for row, it, pred in zip(rows, iters, preds):
            row.update(info)
            row["iteration"] = it
            row["predictor"] = pred
        all_rows.extend(rows)

        paper = PAPER_REFLECTION_CALLS[info["model"]].get(info["task"])
        compared = info["optimizer"] in ("GEPA", "GEPA-MERGE") and paper is not None
        count_rows.append({**{k: info[k] for k in ("run", "task", "optimizer", "model")},
                           "n_reflection_calls": len(rows),
                           "n_iterations": len(state["full_program_trace"]),
                           "n_candidates": len(state["program_full_scores_val_set"]),
                           "n_merges": sum(1 for t in state["full_program_trace"] if t.get("invoked_merge")),
                           "paper_reflection_calls": paper if compared else None,
                           "matches_paper": (len(rows) == paper) if compared else None})
        if not skip_figures:
            plot_instruction_lengths(run, pd.DataFrame(rows), state["list_of_named_predictors"])

    cols = ["run", "task", "optimizer", "model", "call_index", "iteration", "predictor",
            "len_current", "len_new", "len_feedback_input",
            "len_current_tokens", "len_new_tokens", "len_feedback_input_tokens"]
    calls = pd.DataFrame(all_rows)[cols]
    save_table(calls, "reflection_calls")

    summary = calls.groupby(["run", "task", "optimizer", "model"], as_index=False).agg(
        n_calls=("call_index", "size"),
        mean_current_tokens=("len_current_tokens", "mean"),
        mean_proposed_tokens=("len_new_tokens", "mean"),
        max_proposed_tokens=("len_new_tokens", "max"),
        mean_feedback_tokens=("len_feedback_input_tokens", "mean"),
        max_feedback_tokens=("len_feedback_input_tokens", "max"))
    save_table(summary, "reflection_calls_summary")

    counts = pd.DataFrame(count_rows)
    save_table(counts, "reflection_call_counts")

    print("Reflection calls per run (paper's appendix value in brackets):")
    for row in counts.itertuples():
        ref = ""
        if not pd.isna(row.paper_reflection_calls):
            ref = " [paper %d, %s]" % (row.paper_reflection_calls, "match" if row.matches_paper else "differs")
        print("  %-62s %4d calls, %4d iterations, %3d candidates%s" % (
            row.run, row.n_reflection_calls, row.n_iterations, row.n_candidates, ref))
    for opt in ("GEPA", "GEPA-MERGE"):
        sel = counts[(counts.optimizer == opt) & counts.paper_reflection_calls.notna()]
        print("  %s runs matching the appendix table: %d of %d" % (opt, int(sel.matches_paper.sum()), len(sel)))
    print("  proposed instruction tokens: mean %.0f, max %d; feedback input tokens: mean %.0f, max %d" % (
        calls.len_new_tokens.mean(), calls.len_new_tokens.max(),
        calls.len_feedback_input_tokens.mean(), calls.len_feedback_input_tokens.max()))
    return calls, counts


# ---------------------------------------------------------------- part 2

def _not_available(*args, **kwargs):
    raise RuntimeError("not available in this analysis script")


def install_import_stubs():
    """Unpickling a program imports the benchmark package, which at import time pulls
    in HF datasets (needs pyarrow parquet) and spacy (needs srsly's msgpack extension),
    and even tries to download a spacy model and nltk data. None of that is needed to
    read a saved program, and those compiled extensions are blocked on this machine,
    so light stand-ins are registered before anything imports them."""
    for name in ("datasets", "spacy", "spacy.cli"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["datasets"].load_dataset = _not_available
    sys.modules["spacy"].load = _not_available
    sys.modules["spacy"].cli = sys.modules["spacy.cli"]
    sys.modules["spacy.cli"].download = lambda *args, **kwargs: None
    try:
        import nltk
        nltk.download = lambda *args, **kwargs: True
    except ImportError:
        pass


def setup_dspy():
    """Import dspy and the benchmark package under the names used when pickling."""
    sys.path.insert(0, os.getcwd())  # the gepa_artifact package lives in the repo root
    install_import_stubs()
    import dspy
    from gepa_artifact import benchmarks
    for alias in ("langProBe", "benchmarks", "benchmarks.langProBe.langProBe.langProBe"):
        sys.modules[alias] = benchmarks
    return dspy


def demo_text(pred):
    """Few-shot demos as the LM sees them: the signature's fields only, one
    'name: value' line each. Bookkeeping keys stored on the examples (ids, labels,
    the 'augmented' flag) never reach the prompt."""
    fields = list(pred.signature.input_fields) + list(pred.signature.output_fields)
    parts = []
    for demo in pred.demos or []:
        store = dict(demo.items())
        for name in fields:
            if name in store:
                value = store[name]
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                parts.append("%s: %s" % (name, value))
    return "\n".join(parts)


def measure_program(prog, info):
    rows = []
    for name, pred in prog.named_predictors():
        instruction = pred.signature.instructions
        demos = demo_text(pred)
        rows.append({**info, "predictor": name,
                     "instruction_chars": len(instruction), "instruction_tokens": n_tokens(instruction),
                     "n_demos": len(pred.demos or []), "demos_chars": len(demos), "demos_tokens": n_tokens(demos),
                     "total_tokens": n_tokens(instruction) + n_tokens(demos)})
    return rows


def plot_prompt_sizes(by_run, ratios):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    width = 0.26
    offset = 0.02 * by_run.total_tokens.max()
    for ax, model in zip(axes, MODELS):
        tasks = [t for t in TASK_ORDER if ((by_run.model == model) & (by_run.task == t)).any()]
        for j, opt in enumerate(COMPARED_OPTIMIZERS):
            xs, ys = [], []
            for i, task in enumerate(tasks):
                sel = by_run[(by_run.model == model) & (by_run.task == task) & (by_run.optimizer == opt)]
                if len(sel):
                    xs.append(i + (j - 1) * (width + 0.02))
                    ys.append(sel.total_tokens.iloc[0])
            ax.bar(xs, ys, width=width, color=COLORS[j], label=opt)
        # direct label: how much larger the MIPROv2 prompt is than the GEPA prompt
        for i, task in enumerate(tasks):
            sel = ratios[(ratios.model == model) & (ratios.task == task)]
            if len(sel) and not pd.isna(sel.ratio_mipro_over_gepa.iloc[0]):
                top = max(sel.mipro_tokens.iloc[0], sel.gepa_tokens.iloc[0], sel.gepa_merge_tokens.fillna(0).iloc[0])
                ax.text(i, top + offset, "%.1fx" % sel.ratio_mipro_over_gepa.iloc[0], ha="center",
                        va="bottom", fontsize=8, color=INK)
        style_axes(ax)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks)
        ax.set_title(model, color=INK, fontsize=10)
    axes[0].set_ylabel("final prompt size, cl100k tokens\n(instructions + demos, all predictors)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, title="label: MIPROv2 / GEPA", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prompt_sizes.png"), dpi=150)
    plt.close(fig)


def analyze_final_prompts(dspy):
    rows = []
    for run in list_runs():
        folder = os.path.join(RUNS_DIR, run, "evaluation_results", "optimized_program")
        if not os.path.isdir(folder):
            continue  # baseline runs ship no optimized program
        rows.extend(measure_program(dspy.load(folder), parse_run_name(run)))
    sizes = pd.DataFrame(rows)
    save_table(sizes, "final_prompt_sizes")

    by_run = sizes.groupby(["run", "task", "optimizer", "model"], as_index=False).agg(
        n_predictors=("predictor", "size"), n_demos=("n_demos", "sum"),
        instruction_tokens=("instruction_tokens", "sum"), demos_tokens=("demos_tokens", "sum"),
        total_tokens=("total_tokens", "sum"))
    save_table(by_run, "final_prompt_sizes_by_run")

    ratio_rows = []
    for model in MODELS:
        for task in TASK_ORDER:
            tokens = {}
            for opt in COMPARED_OPTIMIZERS:
                sel = by_run[(by_run.model == model) & (by_run.task == task) & (by_run.optimizer == opt)]
                tokens[opt] = float(sel.total_tokens.iloc[0]) if len(sel) else float("nan")
            if all(pd.isna(v) for v in tokens.values()):
                continue
            ratio_rows.append({"model": model, "task": task,
                               "mipro_tokens": tokens["MIPROv2-Heavy"], "gepa_tokens": tokens["GEPA"],
                               "gepa_merge_tokens": tokens["GEPA-MERGE"],
                               "ratio_mipro_over_gepa": tokens["MIPROv2-Heavy"] / tokens["GEPA"],
                               "ratio_mipro_over_gepa_merge": tokens["MIPROv2-Heavy"] / tokens["GEPA-MERGE"]})
    ratios = pd.DataFrame(ratio_rows)
    save_table(ratios, "prompt_size_ratio")
    plot_prompt_sizes(by_run, ratios)

    print("Final prompt size (tokens, instructions + demos) and MIPROv2 / GEPA ratio:")
    for row in ratios.itertuples():
        print("  %-12s %-9s MIPROv2 %6s  GEPA %6s  GEPA-MERGE %6s  ratio %s (vs merge %s)" % (
            row.model, row.task, fmt(row.mipro_tokens), fmt(row.gepa_tokens), fmt(row.gepa_merge_tokens),
            fmt(row.ratio_mipro_over_gepa), fmt(row.ratio_mipro_over_gepa_merge)))
    print("  ratio MIPROv2 / GEPA: min %.2f, max %.2f (paper: up to 9.2x shorter)" % (
        ratios.ratio_mipro_over_gepa.min(), ratios.ratio_mipro_over_gepa.max()))
    return sizes, ratios


# ---------------------------------------------------------------- part 3

def fence(text):
    """Fenced code block whose fence is longer than any backtick run in the text."""
    longest = max([len(m) for m in re.findall(r"`+", text)] + [2])
    marks = "`" * (longest + 1)
    return marks + "\n" + text + "\n" + marks


def write_prompt_examples(dspy):
    lines = ["# Seed and final instructions of two GEPA runs", "",
             "Instructions are copied verbatim from the pickled programs: the seed program is",
             "`prog_candidates/0` and the final program is `evaluation_results/optimized_program`",
             "of each run. Token counts use the cl100k tokenizer.", ""]
    for run in EXAMPLE_RUNS:
        seed = dspy.load(os.path.join(RUNS_DIR, run, "prog_candidates", "0"))
        final = dspy.load(os.path.join(RUNS_DIR, run, "evaluation_results", "optimized_program"))
        scores = load_gepa_state(run)["program_full_scores_val_set"]
        lines += ["## " + run, "",
                  "Validation score: seed %.2f, best of the %d candidates %.2f. Test score of the final program: %.2f." % (
                      scores[0], len(scores), max(scores), read_test_score(run)), ""]
        for (name, seed_pred), (_, final_pred) in zip(seed.named_predictors(), final.named_predictors()):
            seed_ins = seed_pred.signature.instructions
            final_ins = final_pred.signature.instructions
            lines += ["### Predictor `%s`" % name, "",
                      "Seed instruction (%d tokens):" % n_tokens(seed_ins), "", fence(seed_ins), ""]
            if final_ins == seed_ins:
                lines += ["Final instruction: unchanged from the seed.", ""]
            else:
                lines += ["Final instruction (%d tokens):" % n_tokens(final_ins), "", fence(final_ins), ""]
    with open(os.path.join(OUT_DIR, "prompt_examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Prompt evolution analysis of the GEPA artifact runs.")
    parser.add_argument("--skip-run-figures", action="store_true",
                        help="do not draw the per-run instruction length figures")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    analyze_reflection_calls(args.skip_run_figures)
    dspy = setup_dspy()
    analyze_final_prompts(dspy)
    write_prompt_examples(dspy)
    print("done; tables in %s, figures in %s" % (OUT_DIR, FIG_DIR))


if __name__ == "__main__":
    main()
