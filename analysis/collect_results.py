"""Collect the test scores of all shipped runs and compare them with the paper.

Reads every run folder under experiment_runs_data/experiment_runs/seed_0,
builds one score table per model (rows = optimizer, columns = benchmark),
compares every cell with Tables 1-3 of the GEPA paper, and writes a cost
table for the GPT-4.1 mini runs.

Run from the repo root:
    python analysis/collect_results.py
"""

import argparse
import json
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_DIR = os.path.join("experiment_runs_data", "experiment_runs", "seed_0")
DEFAULT_OUT_DIR = os.path.join("analysis", "outputs")

# folder-name token -> name used in the paper
BENCH_NAMES = {
    "HotpotQABench": "HotpotQA",
    "IFBench": "IFBench",
    "hoverBench": "HoVer",
    "Papillon": "PUPA",
    "AIMEBench": "AIME-2025",
    "LiveBenchMathBench": "LiveBench-Math",
}
OPT_NAMES = {
    "Baseline": "Baseline",
    "GRPO": "GRPO",
    "MIPROv2-Heavy": "MIPROv2",
    "GEPA": "GEPA",
    "GEPA-MERGE": "GEPA+Merge",
    "Abl-SelectBestCandidate": "SelectBestCandidate",
}
MODEL_NAMES = {"qwen3-8b": "Qwen3 8B", "gpt-41-mini": "GPT-4.1 mini"}

# older names that still appear inside config.json
CONFIG_ALIASES = {"GEPA8-Heavy": "GEPA", "AIMEBench2": "AIMEBench"}

BENCH_ORDER = list(BENCH_NAMES.values())
OPT_ORDER = list(OPT_NAMES.values())
MODEL_ORDER = ["Qwen3 8B", "GPT-4.1 mini"]

# Test scores printed in the paper, in BENCH_ORDER. Tables 1 and 2 give the
# first five/four optimizers; the SelectBestCandidate row is Table 3 (four tasks).
PAPER = {
    "Qwen3 8B": {
        "Baseline": [42.33, 36.90, 35.33, 80.82, 27.33, 48.70],
        "GRPO": [43.33, 35.88, 38.67, 86.66, 38.00, 51.26],
        "MIPROv2": [55.33, 36.22, 47.33, 81.55, 20.00, 46.60],
        "GEPA": [62.33, 38.61, 52.33, 91.85, 32.00, 51.95],
        "GEPA+Merge": [64.33, 28.23, 51.67, 86.26, 32.00, 51.95],
        "SelectBestCandidate": [58.33, 30.44, 45.33, 85.45, None, None],
    },
    "GPT-4.1 mini": {
        "Baseline": [38.00, 47.79, 46.33, 78.57, 49.33, 58.20],
        "MIPROv2": [58.00, 49.15, 48.33, 83.37, 51.33, 61.84],
        "GEPA": [69.00, 52.72, 51.67, 94.47, 59.33, 64.13],
        "GEPA+Merge": [65.67, 55.95, 56.67, 96.46, 59.33, 64.13],
    },
}
TOL = 0.005  # scores are printed with two decimals


def parse_number(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def read_eval_result(path):
    """Two CSV lines (header, values); both end with a trailing comma."""
    with open(path) as f:
        header, values = [line.strip().rstrip(",").split(",") for line in f if line.strip()]
    return {k: parse_number(v) for k, v in zip(header, values)}


def read_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return {
        "config_benchmark": CONFIG_ALIASES.get(cfg["benchmark_name"], cfg["benchmark_name"]),
        "config_optimizer": CONFIG_ALIASES.get(cfg["optimizer_name"], cfg["optimizer_name"]),
        "config_model": cfg["lm_name"],
    }


def collect_runs(runs_dir):
    """One row per run folder, named <Bench>_<Program>_<Optimizer>_<Model>."""
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        folder = os.path.join(runs_dir, name)
        if not os.path.isdir(folder):
            continue
        bench, program, opt, model = name.split("_")
        row = {
            "run": name,
            "model": MODEL_NAMES[model],
            "benchmark": BENCH_NAMES[bench],
            "program": program,
            "optimizer": OPT_NAMES[opt],
        }
        row.update(read_config(os.path.join(folder, "config.json")))
        result = read_eval_result(os.path.join(folder, "evaluation_results", "evaluation_result.txt"))
        # the result file repeats the optimizer under its old name; keep it apart from our label
        row["eval_optimizer_name"] = result.pop("optimizer", None)
        row.update(result)
        # the folder name is the reference; flag configs that disagree with it
        if (row["config_benchmark"], row["config_optimizer"], row["config_model"]) != (bench, opt, model):
            print("config.json names differ from folder name:", name)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Baseline runs have no optimizer columns: nothing was spent on optimization
    df["optimizer_cost"] = df["optimizer_cost"].fillna(0.0)
    for col in ["optimizer_input_tokens", "optimizer_output_tokens"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


def score_tables(df):
    """model -> DataFrame of test scores (rows = optimizer, columns = benchmark)."""
    tables = {}
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        table = sub.pivot(index="optimizer", columns="benchmark", values="score")
        opts = [o for o in OPT_ORDER if o in table.index]
        tables[model] = table.reindex(index=opts, columns=BENCH_ORDER)
    return tables


def fmt_cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def markdown_table(df, index_name=""):
    """Small Markdown renderer so the script does not need tabulate."""
    header = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmt_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_table(tables, out_dir):
    frames = []
    md = ["# Test scores in the artifact (seed 0)", ""]
    for model, table in tables.items():
        frames.append(table.reset_index().assign(model=model))
        md += [f"## {model}", "", markdown_table(table, "optimizer"), ""]
    wide = pd.concat(frames)[["model", "optimizer"] + BENCH_ORDER]
    wide.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    with open(os.path.join(out_dir, "results_table.md"), "w") as f:
        f.write("\n".join(md))


def inv(mapping, value):
    return [k for k, v in mapping.items() if v == value][0]


def compare_with_paper(df, tables):
    """Sort every paper cell into match / mismatch / no run; also list extra runs."""
    matches, mismatches, no_run = [], [], []
    program_of = df.drop_duplicates("benchmark").set_index("benchmark")["program"].to_dict()
    paper_cells = set()
    for model, rows in PAPER.items():
        for opt, values in rows.items():
            for bench, paper_val in zip(BENCH_ORDER, values):
                if paper_val is None:
                    continue
                paper_cells.add((model, opt, bench))
                table = tables.get(model)
                art_val = float(table.loc[opt, bench]) if table is not None and opt in table.index else float("nan")
                rec = {"model": model, "optimizer": opt, "benchmark": bench, "paper": paper_val, "artifact": art_val}
                if pd.isna(art_val):
                    rec["expected_run_folder"] = "{}_{}_{}_{}".format(
                        inv(BENCH_NAMES, bench), program_of.get(bench, "?"), inv(OPT_NAMES, opt), inv(MODEL_NAMES, model))
                    no_run.append(rec)
                elif abs(art_val - paper_val) < TOL:
                    matches.append(rec)
                else:
                    rec["diff"] = round(art_val - paper_val, 2)
                    mismatches.append(rec)
    extra = [
        {"model": r.model, "optimizer": r.optimizer, "benchmark": r.benchmark, "artifact": r.score, "run": r.run}
        for r in df.itertuples()
        if (r.model, r.optimizer, r.benchmark) not in paper_cells
    ]
    return matches, mismatches, no_run, extra


def write_comparison(matches, mismatches, no_run, extra, out_dir):
    def section(title, records, note):
        if not records:
            return [f"## {title}", "", "None.", ""]
        table = pd.DataFrame(records).set_index("model")
        return [f"## {title}", "", note, "", markdown_table(table, "model"), ""]

    n_cells = len(matches) + len(mismatches) + len(no_run)
    md = [
        "# Paper tables vs. artifact runs",
        "",
        f"Paper cells checked: {n_cells} (Tables 1, 2 and the SelectBestCandidate row of Table 3). "
        f"Exact matches: {len(matches)}. Mismatches: {len(mismatches)}. "
        f"Cells without a run in the artifact: {len(no_run)}. "
        f"Artifact runs without a paper cell: {len(extra)}.",
        "",
    ]
    md += section("Mismatches", mismatches, "Artifact score differs from the paper by more than 0.005.")
    md += section("Paper cells with no run in the artifact", no_run, "Only seed 0 is shipped; these runs are absent.")
    md += section("Artifact runs with no cell in the paper tables", extra, "Scores not printed in Tables 1-3.")
    md += section("Exact matches", matches, "Artifact score equals the paper to two decimals.")
    md += [
        "## Notes",
        "",
        "- LiveBench-Math / GEPA / GPT-4.1 mini: the paper prints 64.13, the artifact run scores 59.43. "
        "The artifact's GEPA+Merge run on LiveBench-Math scores 64.13, which is the number the paper prints "
        "for both GEPA and GEPA+Merge on that task.",
        "- The paper reports identical GEPA and GEPA+Merge scores for the two single-module math tasks "
        "(merging cannot change a one-predictor program). The artifact has no AIME-2025 GEPA+Merge run "
        "for GPT-4.1 mini, so that paper cell (59.33) is the GEPA run's score.",
        "- Qwen3 8B has no AIME-2025 or LiveBench-Math runs for any optimizer, so 10 cells of Table 1 cannot be checked.",
        "- GRPO was only run with Qwen3 8B on the four non-math tasks.",
        "- config.json keeps older names for the GEPA runs (optimizer_name GEPA8-Heavy, and benchmark_name AIMEBench2 "
        "for the AIME GEPA run); the folder names are used here.",
        "",
    ]
    with open(os.path.join(out_dir, "paper_vs_artifact.md"), "w") as f:
        f.write("\n".join(md))


def cost_table(df, out_dir, model="GPT-4.1 mini"):
    """Optimizer and final-evaluation cost of every run of one model (qwen runs cost 0)."""
    cols = ["optimizer_cost", "optimizer_input_tokens", "optimizer_output_tokens",
            "eval_cost", "input_tokens", "output_tokens"]
    sub = df[df["model"] == model].rename(columns={"cost": "eval_cost"}).copy()
    sub["total_cost"] = sub["optimizer_cost"] + sub["eval_cost"]
    sub["bench_rank"] = sub["benchmark"].map(BENCH_ORDER.index)
    sub["opt_rank"] = sub["optimizer"].map(OPT_ORDER.index)
    sub = sub.sort_values(["bench_rank", "opt_rank"])
    table = sub[["benchmark", "optimizer"] + cols + ["total_cost"]]
    table.to_csv(os.path.join(out_dir, "cost_table.csv"), index=False)

    per_opt = table.groupby("optimizer")[["optimizer_cost", "eval_cost", "total_cost"]].sum()
    per_opt = per_opt.reindex([o for o in OPT_ORDER if o in per_opt.index])
    per_opt["runs"] = table.groupby("optimizer").size().astype(str)  # keep as text so it is not shown as 6.00
    gepa = table[table["optimizer"] == "GEPA"].set_index("benchmark")[["optimizer_cost", "eval_cost", "total_cost"]]
    gepa = gepa.sort_values("total_cost")

    shown = table.set_index("benchmark").copy()
    for c in ["optimizer_input_tokens", "optimizer_output_tokens", "input_tokens", "output_tokens"]:
        shown[c] = shown[c].map(lambda v: f"{int(v):,}")
    total = table[cols + ["total_cost"]].sum()
    md = [
        f"# Cost of the {model} runs (USD, from evaluation_result.txt)",
        "",
        "Optimizer columns: spent during optimization (0 for Baseline). "
        "Eval columns: the final test-set evaluation of the returned program. "
        "Costs are the artifact's own accounting and are not exactly tokens times list price.",
        "",
        markdown_table(shown, "benchmark"),
        "",
        f"Total for all {len(table)} runs: optimizer {total['optimizer_cost']:.2f} USD + evaluation "
        f"{total['eval_cost']:.2f} USD = {total['total_cost']:.2f} USD "
        f"({int(total['optimizer_input_tokens'] + total['input_tokens']):,} input and "
        f"{int(total['optimizer_output_tokens'] + total['output_tokens']):,} output tokens).",
        "",
        "## Per optimizer (summed over benchmarks)",
        "",
        markdown_table(per_opt, "optimizer"),
        "",
        "## GEPA run per benchmark",
        "",
        markdown_table(gepa, "benchmark"),
        "",
        f"Cheapest full GEPA run: {gepa.index[0]} at {gepa['total_cost'].iloc[0]:.2f} USD; "
        f"most expensive: {gepa.index[-1]} at {gepa['total_cost'].iloc[-1]:.2f} USD; "
        f"all six GEPA runs: {gepa['total_cost'].sum():.2f} USD.",
        "",
    ]
    with open(os.path.join(out_dir, "cost_table.md"), "w") as f:
        f.write("\n".join(md))
    return total["total_cost"], gepa


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="folder with the run directories")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where tables are written")
    args = parser.parse_args()
    runs_dir = os.path.join(REPO_ROOT, args.runs_dir)
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = collect_runs(runs_dir)
    df.to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)
    print(f"{len(df)} runs:", df.groupby("model").size().to_dict())

    tables = score_tables(df)
    write_results_table(tables, out_dir)
    for model, table in tables.items():
        print(f"\n{model}\n{table.to_string(float_format=lambda v: f'{v:.2f}')}")

    matches, mismatches, no_run, extra = compare_with_paper(df, tables)
    write_comparison(matches, mismatches, no_run, extra, out_dir)
    print(f"\npaper cells: {len(matches)} match, {len(mismatches)} mismatch, {len(no_run)} without a run; "
          f"{len(extra)} artifact runs not in the paper tables")
    for rec in mismatches:
        print("  mismatch:", rec)

    total, gepa = cost_table(df, out_dir)
    print(f"\nGPT-4.1 mini: {total:.2f} USD to rerun everything; GEPA runs (USD):")
    print(gepa.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
