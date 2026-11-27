# Extra variants for claims 3, 6, 9 (uses my own functions from metric_analysis.py in this folder).
# Usage: python -I extras.py <run_dir> [...]
import os, sys, math, collections, importlib.util

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ma", os.path.join(here, "metric_analysis.py"))
ma = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)
mean = ma.mean


def ols(xs, ys):
    n = len(xs); mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx; a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / (n - 2)
    se = math.sqrt(s2 / sxx)
    return b, se


for run in sys.argv[1:]:
    print("=" * 90)
    print(os.path.basename(run.rstrip("/\\")))
    train = ma.load(os.path.join(run, "metric_logs", "train.jsonl"))
    val = ma.load(os.path.join(run, "metric_logs", "val.jsonl"))
    test = ma.load(os.path.join(run, "metric_logs", "test.jsonl"))
    step_eval, val_eval, test_eval, vs = ma.parse_stderr(run)
    passes = ma.val_passes(val, test[0]["step_counter"] if test else None)
    ev, ov = ma.classify_train_rows(train, val, test, passes)
    keep = [not e for e in ev]
    steps, dropped, wcost = ma.segment_steps_dp(train, keep, [p[0]["train_counter"] for p in passes])
    assert len(steps) == 500
    vals = [[v for g in s["groups"].values() for v in g] for s in steps]
    rowmean = lambda a, b: 100 * mean(v for x in vals[a:b] for v in x)
    stepmean = lambda a, b: 100 * mean(mean(x) for x in vals[a:b] if x)
    per48 = lambda a, b: 100 * mean(sum(x) / 48 for x in vals[a:b])
    se = [100 * step_eval[k][0] / step_eval[k][1] for k in range(1, 501)]
    print(f"[claim3] first50 / last50: rows {rowmean(0,50):.2f}/{rowmean(450,500):.2f}; mean of step means {stepmean(0,50):.2f}/{stepmean(450,500):.2f}; "
          f"logged sum/48 {per48(0,50):.2f}/{per48(450,500):.2f}; DSPy per-step Evaluate {mean(se[:50]):.2f}/{mean(se[450:]):.2f}")
    print(f"[claim9] 100-step blocks: rows {[round(rowmean(k*100,(k+1)*100),2) for k in range(5)]}; step means {[round(stepmean(k*100,(k+1)*100),2) for k in range(5)]}; "
          f"DSPy Evaluate {[round(mean(se[k*100:(k+1)*100]),2) for k in range(5)]}")
    # trends: OLS slope per 100 steps
    xs = list(range(1, 501))
    b1, s1 = ols(xs, [100 * mean(x) for x in vals])
    b2, s2 = ols(xs, se)
    vx = list(range(20, 501, 20)); vy = [vs[s] for s in vx]
    b3, s3 = ols(vx, vy)
    print(f"[claim9] OLS slope per 100 steps: train (logged step means) {100*b1:+.2f} (SE {100*s1:.2f}); train (DSPy Evaluate) {100*b2:+.2f} (SE {100*s2:.2f}); "
          f"validation steps 20-500 {100*b3:+.2f} (SE {100*s3:.2f})")
    # failure counts: rollouts missing from the metric log per block (48 - kept rows)
    print(f"[info] rollouts without a metric row per 100-step block: {[sum(48 - min(48, s['kept']) for s in steps[k*100:(k+1)*100]) for k in range(5)]}")

    # claim 6 variants
    def per_pass(p, how):
        d = collections.defaultdict(list)
        for r in p:
            d[r["idx_in_split"]].append(float(r["metric_output"]))
        if how == "first":
            return {k: v[0] for k, v in d.items()}
        if how == "last":
            return {k: v[-1] for k, v in d.items()}
        return {k: mean(v) for k, v in d.items()}
    for how in ("first", "last", "mean"):
        P = [per_pass(p, how) for p in passes]
        F, L = list(range(0, 5)), list(range(21, 26))
        c10 = set.intersection(*[set(P[k]) for k in F + L])
        c26 = set.intersection(*[set(x) for x in P])
        r10 = (100 * mean(mean(P[k][i] for k in F) for i in c10), 100 * mean(mean(P[k][i] for k in L) for i in c10))
        r26 = (100 * mean(mean(P[k][i] for k in F) for i in c26), 100 * mean(mean(P[k][i] for k in L) for i in c26))
        print(f"[claim6] dup rule '{how}': common to 10 passes n={len(c10)} {r10[0]:.2f} vs {r10[1]:.2f}; common to all 26 n={len(c26)} {r26[0]:.2f} vs {r26[1]:.2f}")
    # Evaluate-based pass means (failures = 0, denominator = full val set)
    ve = [100 * s / n for s, n in val_eval]
    print(f"[claim6] DSPy Evaluate (failures=0): passes 0-4 mean {mean(ve[:5]):.2f}, passes 21-25 mean {mean(ve[21:]):.2f}; passes 1-5 mean {mean(ve[1:6]):.2f}")
