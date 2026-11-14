# Claims 4 and 7 from arbor_logs.txt (streamed as text; nothing is executed or literal-eval'ed).
# Usage: python -I arbor_analysis.py <run_dir> [...]
import os, re, sys, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NUM = r"(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
KEYS = ["loss", "kl", "reward", "reward_std", "completions/mean_length", "epoch", "grad_norm",
        "completions/clipped_ratio", "num_tokens"]


def grab(line, key):
    m = re.search(r"'" + re.escape(key) + r"': " + NUM, line)
    return float(m.group(1)) if m else None


for run in sys.argv[1:]:
    name = os.path.basename(run.rstrip("/\\"))
    print("=" * 90)
    print(name)
    dicts = []
    kl_lines_any = 0
    tables = 0
    rows_in_tables = 0
    think_open = 0
    think_empty = 0
    think_nonempty = 0
    second_think = 0
    example = None
    pending = None          # set after a "<think>" completion cell
    last_nonblank = None
    path = os.path.join(run, "arbor_logs.txt")
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            if "'kl': " in line:
                kl_lines_any += 1
            if "{'loss'" in line or ("'loss': " in line and "'kl': " in line):
                d = {k: grab(line, k) for k in KEYS}
                d["_line"] = ln
                dicts.append(d)
                continue
            if "[GRPO LOG]" not in line:
                continue
            if "╭" in line and " Step " in line:
                tables += 1
                continue
            if "│" not in line:
                continue
            parts = line.rstrip("\r\n").split("│")
            if len(parts) < 5:
                continue
            prompt_cell, comp = parts[2].strip(), parts[3].strip()
            if prompt_cell.startswith("<|im_start|>system"):
                rows_in_tables += 1
            if pending is not None:
                if comp == "</think>":
                    think_empty += 1
                    if example is None:
                        example = (pending, ln)
                    pending = None
                elif comp != "":
                    think_nonempty += 1
                    pending = None
                if comp:
                    last_nonblank = comp
                continue
            if comp == "<think>":
                think_open += 1
                if last_nonblank == "</think>":
                    second_think += 1
                pending = ln
            if comp:
                last_nonblank = comp
    kls = [d["kl"] for d in dicts if d["kl"] is not None]
    lens = [d["completions/mean_length"] for d in dicts if d["completions/mean_length"] is not None]
    print(f"[claim4] metric dict lines: {len(dicts)}; lines containing 'kl': {kl_lines_any}; KL values: {len(kls)}")
    print(f"[claim4] KL > 10: {sum(1 for k in kls if k > 10)} of {len(kls)}; max {max(kls):.4g}; values: {[float(f'{k:.3g}') for k in kls]}")
    print(f"[claim4] epochs logged: {[d['epoch'] for d in dicts]}")
    print(f"[claim4] loss values: {[d['loss'] for d in dicts]}")
    print(f"[claim7] completions/mean_length: n={len(lens)}; mean {statistics.mean(lens):.1f}; min {min(lens):.1f}; max {max(lens):.1f}")
    rw = [d["reward"] for d in dicts if d["reward"] is not None]
    print(f"[info] TRL reward per log: {[round(x, 3) for x in rw]}")
    print(f"[claim7] completion tables: {tables}; table rows (prompt starts): {rows_in_tables}; '<think>' completion cells: {think_open}; "
          f"empty think blocks (<think> then blank then </think>): {think_empty}; non-empty: {think_nonempty}; '<think>' right after an empty block: {second_think}")
    if example:
        print(f"[claim7] first empty think block at arbor_logs.txt lines {example[0]}-{example[1]}")
