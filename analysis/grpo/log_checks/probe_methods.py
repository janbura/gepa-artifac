# Probe a few simpler (cruder) grouping methods to see how much the zero-advantage share depends on the method
# for the train-reward and per-example validation numbers. Uses the helper functions from metric_analysis.py.
import os, sys, collections, importlib.util
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ma", os.path.join(here, "metric_analysis.py"))
ma = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)
mean = ma.mean

for run in sys.argv[1:]:
    print("=" * 80); print(os.path.basename(run.rstrip("/\\")))
    train = ma.load(os.path.join(run, "metric_logs", "train.jsonl"))
    val = ma.load(os.path.join(run, "metric_logs", "val.jsonl"))
    test = ma.load(os.path.join(run, "metric_logs", "test.jsonl"))
    step_eval, val_eval, test_eval, vs = ma.parse_stderr(run)
    nval = val_eval[0][1]
    # (a) val passes as fixed chunks of nval rows by val_counter
    ch = collections.defaultdict(dict)
    for r in val:
        ch[(r["val_counter"] - 1) // nval].setdefault(r["idx_in_split"], float(r["metric_output"]))
    ks = sorted(ch)
    F, L = ks[:5], ks[-5:]
    common = set.intersection(*[set(ch[k]) for k in F + L])
    print(f"(a) val chunks of {nval} rows: n chunks {len(ks)}; common n={len(common)}: first5 {100*mean(mean(ch[k][i] for k in F) for i in common):.2f} "
          f"last5 {100*mean(mean(ch[k][i] for k in L) for i in common):.2f}")
    L2 = ks[21:26]
    common2 = set.intersection(*[set(ch[k]) for k in F + L2])
    print(f"    chunks 0-4 vs 21-25: n={len(common2)} {100*mean(mean(ch[k][i] for k in F) for i in common2):.2f} vs {100*mean(mean(ch[k][i] for k in L2) for i in common2):.2f}")
    # (b) train rows: steps = fixed chunks of 48 rows by train_counter
    c48 = collections.defaultdict(list)
    for r in train:
        c48[(r["train_counter"] - 1) // 48].append(float(r["metric_output"]))
    kk = sorted(c48)
    print(f"(b) 48-row chunks: n={len(kk)}; first50 rows-mean {100*mean(v for k in kk[:50] for v in c48[k]):.2f}; last50 {100*mean(v for k in kk[-50:] for v in c48[k]):.2f}; "
          f"chunk-means first50 {100*mean(mean(c48[k]) for k in kk[:50]):.2f} last50 {100*mean(mean(c48[k]) for k in kk[-50:]):.2f}")
    # (c) using the windows between validation passes, each split into 20 equal row ranges
    passes = ma.val_passes(val, test[0]["step_counter"] if test else None)
    pb = [p[0]["train_counter"] for p in passes]
    ev, ov = ma.classify_train_rows(train, val, test, passes)
    rows = [(r["train_counter"], float(r["metric_output"])) for i, r in enumerate(train)]
    steps = []
    for w in range(25):
        wr = [v for tc, v in rows if pb[w] < tc <= pb[w + 1]]
        n = len(wr)
        for s in range(20):
            steps.append(wr[s * n // 20:(s + 1) * n // 20])
    print(f"(c) windows split in 20 equal parts (all train rows): first50 {100*mean(v for s in steps[:50] for v in s):.2f} last50 {100*mean(v for s in steps[-50:] for v in s):.2f}; "
          f"step-means {100*mean(mean(s) for s in steps[:50]):.2f} / {100*mean(mean(s) for s in steps[-50:]):.2f}")
