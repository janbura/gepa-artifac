# Claims 1, 5, 8 from run_log_stderr.txt (+ evaluation_result.txt, lm_config.json), read as text only.
# Usage: python -I stderr_analysis.py <run_dir> [...]
import os, re, sys, statistics, datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TS_START = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) ")
TS_ANY = re.compile(r"(\d{4}[/-]\d\d[/-]\d\d[ T]\d\d:\d\d:\d\d)")


def parse_ts(s):
    s = s.replace("-", "/").replace("T", " ")
    return datetime.datetime.strptime(s, "%Y/%m/%d %H:%M:%S")


for run in sys.argv[1:]:
    name = os.path.basename(run.rstrip("/\\"))
    print("=" * 90)
    print(name)
    lines = open(os.path.join(run, "run_log_stderr.txt"), encoding="utf-8", errors="replace").read().splitlines()
    # ---- claim 8: wall clock ----
    ts = [parse_ts(m.group(1)) for l in lines for m in [TS_START.match(l)] if m]
    ts_any = [parse_ts(m.group(1)) for l in lines for m in TS_ANY.finditer(l)]
    print(f"[claim8] line-start timestamps: {len(ts)}; first {ts[0]} last {ts[-1]}; span {(ts[-1]-ts[0]).total_seconds()/3600:.2f} h; "
          f"monotonic: {all(a <= b for a, b in zip(ts, ts[1:]))}")
    print(f"[claim8] any timestamp anywhere in a line: min {min(ts_any)} max {max(ts_any)}; span {(max(ts_any)-min(ts_any)).total_seconds()/3600:.2f} h")
    # ---- claim 1: validation ----
    val = {}
    for l in lines:
        m = re.search(r"validation set score before training loop: ([0-9.]+)", l)
        if m:
            val[0] = float(m.group(1))
        m = re.search(r"validation set score after training step (\d+)/(\d+): ([0-9.]+)", l)
        if m:
            val[int(m.group(1))] = float(m.group(3))
    later = [val[k] for k in sorted(val) if k > 0]
    best = max(val.values())
    best_steps = [k for k in sorted(val) if val[k] == best]
    print(f"[claim1] before training: {val[0]}; n later: {len(later)}; mean {statistics.mean(later):.3f}; "
          f"SD(n-1) {statistics.stdev(later):.3f}; SD(n) {statistics.pstdev(later):.3f}; min {min(later)} max {max(later)}")
    print(f"[claim1] best overall {best} at step(s) {best_steps}; step 500: {val.get(500)}")
    nb = [l[20:].strip() for l in lines if "New best score" in l]
    print(f"[claim1] 'New best' lines: {len(nb)}; last: {nb[-1] if nb else None}")
    upd = [l for l in lines if "Updated the model for" in l]
    for l in upd:
        m = re.search(r"model_checkpoint_(\d+)", l)
        print(f"[claim1] final model set to checkpoint step {m.group(1) if m else '?'} -> equals best-validation step: {m and int(m.group(1)) in best_steps}")
    ck = [int(m.group(1)) for l in lines for m in [re.search(r"Checkpoint the model at step (\d+)/", l)] if m]
    print(f"[claim1] checkpoints saved at steps: {ck}")
    lmc = open(os.path.join(run, "lm_config.json"), encoding="utf-8").read()
    m = re.search(r'"new_model_name": "([^"]+)"', lmc)
    print(f"[claim1] lm_config.json new_model_name: {m.group(1) if m else None}")
    # ---- claim 5: final test evaluation ----
    k0 = next(i for i, l in enumerate(lines) if "GRPO compiler has finished compiling" in l)
    tail = lines[k0:]
    errs = [l for l in tail if re.match(r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d ERROR dspy\.utils\.parallelizer: Error for Example", l)]
    exc = {}
    for l in tail:
        m = re.match(r"^(openai\.\w+|litellm\.exceptions\.\w+|httpx\.\w+): (.*)$", l)
        if m:
            key = m.group(1) + ": " + m.group(2)[:90]
            exc[key] = exc.get(key, 0) + 1
    avg = [l for l in tail if "Average Metric" in l]
    print(f"[claim5] after 'finished compiling' (line {k0+1}/{len(lines)}): parallelizer ERROR lines {len(errs)}; Evaluate lines: {avg}")
    for k, v in sorted(exc.items(), key=lambda x: -x[1])[:6]:
        print(f"         {v:4d} x {k}")
    print(f"[claim5] evaluation_result.txt: {open(os.path.join(run, 'evaluation_results', 'evaluation_result.txt'), encoding='utf-8').read().strip().splitlines()[-1]}")
    # errors during training (before the final evaluation)
    errs_train = [l for l in lines[:k0] if " ERROR dspy.utils.parallelizer" in l]
    print(f"[info] parallelizer ERROR lines before the final evaluation: {len(errs_train)}")
