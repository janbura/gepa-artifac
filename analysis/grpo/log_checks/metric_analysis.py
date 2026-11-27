# Independent recomputation of metric-log based numbers for the four GRPO runs (v2).
# Reads metric_logs/{train,val,test}.jsonl as JSON text and run_log_stderr.txt as text.
# Nothing from the data folders is executed or imported.
# Usage: python -I metric_analysis.py <run_dir> [<run_dir> ...]
import json, os, sys, re, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PASS_JUMP = 100      # train_counter jump (in train rows) that separates two validation passes
NEIGH = 10           # neighbours on each side for the eval-phase test of a train row
NEIGH_FRAC = 0.75    # train row is "eval phase" if >= this share of its neighbours are val/test rows
LOOKBACK = 2         # a row whose idx belongs to one of the previous LOOKBACK steps is a late row


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- stderr parsing
TS = r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d "
def parse_stderr(run):
    """Return dict with per-step Evaluate sums, val Evaluate sums, final test Evaluate, val scores."""
    step_eval = {}      # step -> (sum, n)
    val_eval = []       # list of (sum, n) in order
    test_eval = None
    val_scores = {}
    state = None        # 'step', 'val', 'test'
    cur_step = None
    p = os.path.join(run, "run_log_stderr.txt")
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not re.match(TS, line):
                continue
            m = re.search(r"GRPO training step (\d+)/\d+\.\.\.$", line.rstrip())
            if m:
                state, cur_step = "step", int(m.group(1)); continue
            if "Evaluating the student program on the validation set" in line:
                state = "val"; continue
            if "GRPO compiler has finished compiling" in line:
                state = "test"; continue
            m = re.search(r"Average Metric: (-?[0-9.]+) / (\d+) ", line)
            if m:
                s, n = float(m.group(1)), int(m.group(2))
                if state == "step":
                    step_eval[cur_step] = (s, n); state = None
                elif state == "val":
                    val_eval.append((s, n)); state = None
                elif state == "test":
                    test_eval = (s, n); state = None
                continue
            m = re.search(r"validation set score before training loop: ([0-9.]+)", line)
            if m:
                val_scores[0] = float(m.group(1))
            m = re.search(r"validation set score after training step (\d+)/\d+: ([0-9.]+)", line)
            if m:
                val_scores[int(m.group(1))] = float(m.group(2))
    return step_eval, val_eval, test_eval, val_scores


# ---------------------------------------------------------------- metric logs
def val_passes(val, test_start_sc):
    passes = [[]]
    prev = None
    for r in val:
        if test_start_sc is not None and r["step_counter"] >= test_start_sc:
            continue    # val-split rows logged during the final test evaluation (content overlap)
        if prev is not None and r["train_counter"] - prev["train_counter"] > PASS_JUMP:
            passes.append([])
        passes[-1].append(r)
        prev = r
    return passes


def first_per_idx(rows):
    d = {}
    for r in rows:
        d.setdefault(r["idx_in_split"], float(r["metric_output"]))
    return d


def classify_train_rows(train, val, test, passes):
    """Eval-phase train rows = (a) rows mostly surrounded by val/test rows (neighbour rule), or
    (b) rows of an 'overlap' idx (a val example whose content equals a train example, so the logger
    files it under train; recognised because it shows up inside >= 3 validation passes) logged within
    DELTA global rows of a validation pass or of the test evaluation."""
    DELTA = 3
    seq = [("t", r["step_counter"], i) for i, r in enumerate(train)]
    seq += [("v", r["step_counter"], i) for i, r in enumerate(val)]
    seq += [("x", r["step_counter"], i) for i, r in enumerate(test)]
    seq.sort(key=lambda x: x[1])
    pos = {i: p for p, (k, sc, i) in enumerate(seq) if k == "t"}
    out = [False] * len(train)
    for i in range(len(train)):
        p = pos[i]
        neigh = [seq[q][0] for q in range(max(0, p - NEIGH), min(len(seq), p + NEIGH + 1)) if q != p]
        out[i] = sum(1 for k in neigh if k != "t") / len(neigh) >= NEIGH_FRAC
    spans = [(p[0]["step_counter"], p[-1]["step_counter"]) for p in passes]
    if test:
        spans.append((test[0]["step_counter"], test[-1]["step_counter"]))
    inside = collections.defaultdict(set)
    for r in train:
        for k, (a, b) in enumerate(spans[:len(passes)]):
            if a < r["step_counter"] < b:
                inside[r["idx_in_split"]].add(k)
    overlap = {idx for idx, ks in inside.items() if len(ks) >= len(passes) // 2}
    for i, r in enumerate(train):
        if not out[i] and r["idx_in_split"] in overlap:
            if any(a - DELTA <= r["step_counter"] <= b + DELTA for a, b in spans):
                out[i] = True
    return out, overlap


def segment_steps(train, keep, pass_bounds):
    """Steps = blocks of 4 distinct examples inside each window between two validation passes.
    Rows whose idx belongs to one of the previous LOOKBACK steps of the same window (or the last
    steps of the previous window) are 'late' rows (redundant copies of straggler jobs that finished
    after their step ended); they are kept aside, not added to any group."""
    rows = sorted((r["train_counter"], r["idx_in_split"], float(r["metric_output"])) for i, r in enumerate(train) if keep[i])
    windows = collections.defaultdict(list)
    for tc, idx, v in rows:
        w = 0
        while w + 1 < len(pass_bounds) and tc > pass_bounds[w + 1]:
            w += 1
        windows[w].append((tc, idx, v))
    steps, wcounts, late = [], {}, []
    for w in sorted(windows):
        wsteps = []
        cur = None
        for tc, idx, v in windows[w]:
            if cur is not None and idx in cur["groups"]:
                cur["groups"][idx].append(v); cur["rows"] += 1; continue
            prev_steps = (wsteps[-LOOKBACK:] if wsteps else steps[-1:])
            if any(idx in s["groups"] for s in prev_steps):
                late.append((w, tc, idx, v)); continue
            if cur is not None and len(cur["groups"]) < 4:
                cur["groups"][idx] = [v]; cur["rows"] += 1; continue
            if cur is not None:
                wsteps.append(cur)
            cur = {"window": w, "groups": collections.OrderedDict([(idx, [v])]), "rows": 1}
        if cur is not None:
            wsteps.append(cur)
        wcounts[w] = len(wsteps)
        steps.extend(wsteps)
    return steps, wcounts, late


def _block_cost(cnt, L):
    # main idx = idx with >= 3 rows in the block; an idx with > 16 rows counts as 2 example slots
    # (the same example drawn twice in one step at an epoch boundary).
    slots, nonmain = 0, 0
    for c in cnt.values():
        if c >= 3:
            slots += 2 if c > 16 else 1
        else:
            nonmain += c
    return 10.0 * abs(slots - 4) + 0.02 * abs(L - 48) + 0.5 * nonmain


def segment_steps_dp(train, keep, pass_bounds, k_per_window=20, lmin=8, lmax=72):
    """Optimal split of each window (rows between two validation passes) into exactly 20 contiguous
    blocks (= GRPO steps), minimising the summed _block_cost. Groups = rows of each main idx inside
    a block; non-main rows (1-2 stray rows of an idx: late redundant copies) are left out."""
    rows = sorted((r["train_counter"], r["idx_in_split"], float(r["metric_output"])) for i, r in enumerate(train) if keep[i])
    windows = collections.defaultdict(list)
    for tc, idx, v in rows:
        w = 0
        while w + 1 < len(pass_bounds) and tc > pass_bounds[w + 1]:
            w += 1
        windows[w].append((tc, idx, v))
    steps, dropped, wcost = [], 0, {}
    INF = float("inf")
    for w in sorted(windows):
        R = windows[w]
        N = len(R)
        K = k_per_window
        cost = [None] * N
        for i in range(N):
            cnt = collections.Counter()
            row = [INF] * (lmax + 1)
            for j in range(i, min(N, i + lmax)):
                cnt[R[j][1]] += 1
                L = j - i + 1
                if L >= lmin:
                    row[L] = _block_cost(cnt, L)
            cost[i] = row
        prev = [INF] * (N + 1); prev[0] = 0.0
        backs = []
        for k in range(1, K + 1):
            cur = [INF] * (N + 1); back = [-1] * (N + 1)
            for j in range(lmin, N + 1):
                best, arg = INF, -1
                for L in range(lmin, min(lmax, j) + 1):
                    i = j - L
                    pi = prev[i]
                    if pi == INF:
                        continue
                    c = pi + cost[i][L]
                    if c < best:
                        best, arg = c, i
                cur[j], back[j] = best, arg
            backs.append(back); prev = cur
        if prev[N] == INF:
            raise RuntimeError(f"window {w}: no segmentation into {K} blocks")
        wcost[w] = round(prev[N], 2)
        cuts, j = [N], N
        for k in range(K, 0, -1):
            j = backs[k - 1][j]; cuts.append(j)
        cuts.reverse()
        for a, b in zip(cuts, cuts[1:]):
            cnt = collections.Counter(R[q][1] for q in range(a, b))
            groups = collections.OrderedDict()
            for q in range(a, b):
                tc, idx, v = R[q]
                if cnt[idx] >= 3:
                    groups.setdefault(idx, []).append(v)
                else:
                    dropped += 1
            steps.append({"window": w, "groups": groups, "rows": b - a, "kept": sum(len(g) for g in groups.values()),
                          "slots": sum(2 if len(g) > 16 else 1 for g in groups.values())})
    return steps, dropped, wcost


def share_identical(groups, min_size=2):
    gs = [g for g in groups if len(g) >= min_size]
    if not gs:
        return (0, float("nan"))
    return (len(gs), round(100.0 * sum(1 for g in gs if max(g) == min(g)) / len(gs), 2))


def per_idx_clusters(train, keep, gap):
    by_idx = collections.defaultdict(list)
    for i, r in enumerate(train):
        if keep[i]:
            by_idx[r["idx_in_split"]].append((r["train_counter"], float(r["metric_output"])))
    groups = []
    for lst in by_idx.values():
        lst.sort()
        cur = [lst[0][1]]
        for (a, _), (b, v) in zip(lst, lst[1:]):
            if b - a > gap:
                groups.append(cur); cur = [v]
            else:
                cur.append(v)
        groups.append(cur)
    return groups


def chunk48_groups(train):
    g = collections.defaultdict(list)
    for r in train:
        g[((r["train_counter"] - 1) // 48, r["idx_in_split"])].append(float(r["metric_output"]))
    return list(g.values())


def main():
    for run in sys.argv[1:]:
        name = os.path.basename(run.rstrip("/\\"))
        print("=" * 100)
        print(name)
        train = load(os.path.join(run, "metric_logs", "train.jsonl"))
        val = load(os.path.join(run, "metric_logs", "val.jsonl"))
        test = load(os.path.join(run, "metric_logs", "test.jsonl"))
        step_eval, val_eval, test_eval, vs = parse_stderr(run)
        print(f"rows: train {len(train)} val {len(val)} test {len(test)} | stderr: per-step Evaluate {len(step_eval)}, val Evaluate {len(val_eval)}, test Evaluate {test_eval}, val scores {len(vs)}")
        test_start = test[0]["step_counter"] if test else None

        # ---------- validation passes, cross-checked against DSPy Evaluate sums ----------
        passes = val_passes(val, test_start)
        print(f"validation passes: {len(passes)}; rows per pass: {[len(p) for p in passes]}")
        diffs = []
        for k, p in enumerate(passes):
            s_first = sum(first_per_idx(p).values())
            diffs.append(round(val_eval[k][0] - s_first, 4) if k < len(val_eval) else None)
        print(f"Evaluate sum minus logged sum (first row per idx), per pass: {diffs}")
        # scores recomputed from Evaluate sums
        print("val score check (Evaluate sum / N *100 vs logged 'score' line):",
              all(abs(round(100 * val_eval[k][0] / val_eval[k][1], 2) - vs[0 if k == 0 else 20 * k]) < 0.006 for k in range(len(val_eval))))

        # ---------- train rows ----------
        eval_phase, overlap = classify_train_rows(train, val, test, passes)
        keep = [not e for e in eval_phase]
        ev_idx = collections.Counter(train[i]["idx_in_split"] for i in range(len(train)) if eval_phase[i])
        print(f"overlap idx (val examples filed under train): {sorted(overlap)}")
        print(f"train rows logged during evaluations (excluded): {sum(eval_phase)}; by idx: {sorted(ev_idx.items(), key=lambda x: -x[1])[:12]}")
        pb = [p[0]["train_counter"] for p in passes]
        gsteps, wcounts, late = segment_steps(train, keep, pb)
        bad = {w: c for w, c in wcounts.items() if c != 20}
        ggroups = [g for s in gsteps for g in s["groups"].values()]
        print(f"[greedy segmentation] steps {len(gsteps)}; windows with != 20 steps: {bad}; late rows {len(late)}; "
              f"share size>=2 {share_identical(ggroups)}; size==12 {share_identical([g for g in ggroups if len(g)==12])}")
        steps, dropped, wcost = segment_steps_dp(train, keep, pb)
        print(f"[DP segmentation] steps {len(steps)}; stray rows dropped {dropped}; windows with cost > 30: {{w: c for w, c in wcost.items() if c > 30}} -> "
              f"{ {w: c for w, c in wcost.items() if c > 30} }")
        print(f"   slots per step: {dict(collections.Counter(s['slots'] for s in steps))}; rows/step top: {collections.Counter(s['rows'] for s in steps).most_common(6)}")
        # compare per-step logged sum with stderr Evaluate per-step sum (DSPy's own per-step total)
        if len(steps) == 500:
            def ssum(s):
                return sum(v for g in s["groups"].values() for v in g)
            eq = sum(1 for k, s in enumerate(steps) if abs(ssum(s) - step_eval[k + 1][0]) < 1e-6)
            full = [k for k, s in enumerate(steps) if s["kept"] == 48 and s["slots"] == 4]
            eq_full = sum(1 for k in full if abs(ssum(steps[k]) - step_eval[k + 1][0]) < 1e-6)
            ge = sum(1 for k, s in enumerate(steps) if ssum(s) >= step_eval[k + 1][0] - 1e-6)
            print(f"   per-step logged sum == stderr Evaluate sum: {eq}/500; among steps with 48 kept rows & 4 slots: {eq_full}/{len(full)}; "
                  f"logged sum >= Evaluate sum: {ge}/500")

        # ---------- claim 2 ----------
        groups = [g for s in steps for g in s["groups"].values()]
        hist = dict(sorted(collections.Counter(len(g) for g in groups).items()))
        print(f"[claim2] M1 (DP step segmentation): groups {len(groups)}, size hist {hist}")
        print(f"[claim2] M1 share all-identical: size>=2 {share_identical(groups)}; size==12 {share_identical([g for g in groups if len(g)==12])}; "
              f"first 12 of size>=12 {share_identical([g[:12] for g in groups if len(g)>=12])}; size>=6 {share_identical(groups, 6)}; size 10-14 {share_identical([g for g in groups if 10 <= len(g) <= 14])}")
        if len(steps) == 500:
            clean = [g for s in steps if s["kept"] == 48 and s["slots"] == 4 and all(len(x) == 12 for x in s["groups"].values()) for g in s["groups"].values()]
            clean_ev = [g for k, s in enumerate(steps) if s["kept"] == 48 and s["slots"] == 4 and all(len(x) == 12 for x in s["groups"].values())
                        and abs(sum(v for x in s["groups"].values() for v in x) - step_eval[k + 1][0]) < 1e-6 for g in s["groups"].values()]
            print(f"[claim2] M1 clean steps only (4 groups x 12 rows): {share_identical(clean)}; clean and sum == stderr Evaluate: {share_identical(clean_ev)}")
        for gap in (60, 200, 400):
            gs = per_idx_clusters(train, keep, gap)
            print(f"[claim2] M2 per-idx clusters gap>{gap}: groups {len(gs)}; size>=2 {share_identical(gs)}; size==12 {share_identical([g for g in gs if len(g)==12])}")
        gs = chunk48_groups(train)
        print(f"[claim2] M5 (train_counter//48, idx) on all train rows: groups {len(gs)}; size>=1 {share_identical(gs,1)}; size>=2 {share_identical(gs)}; size==12 {share_identical([g for g in gs if len(g)==12])}")

        # ---------- claims 3 / 9 ----------
        if len(step_eval) == 500:
            se = [100 * step_eval[k][0] / step_eval[k][1] for k in range(1, 501)]
            print(f"[claim3] stderr per-step Evaluate (X/48, failures counted): first50 {mean(se[:50]):.2f} last50 {mean(se[-50:]):.2f}")
            print(f"[claim9] stderr per-step Evaluate, 100-step blocks: {[round(mean(se[k*100:(k+1)*100]), 2) for k in range(5)]}")
        if len(steps) == 500:
            def blk(a, b):
                return round(100 * mean(v for s in steps[a:b] for g in s["groups"].values() for v in g), 2)
            print(f"[claim3] metric-log rows by segmented step: first50 {blk(0,50)} last50 {blk(450,500)}")
            print(f"[claim9] metric-log rows by segmented step, 100-step blocks: {[blk(k*100,(k+1)*100) for k in range(5)]}")
        tp = [float(r["metric_output"]) for i, r in enumerate(train) if keep[i]]
        print(f"[claim3] position-based (first/last 2400 training-phase rows): {100*mean(tp[:2400]):.2f} / {100*mean(tp[-2400:]):.2f}")
        vb = [round(mean(vs[s] for s in range(k * 100 + 20, (k + 1) * 100 + 1, 20)), 2) for k in range(5)]
        print(f"[claim9] validation (logged scores) per 100-step block: {vb}; before training {vs[0]}")

        # ---------- claim 6 ----------
        if len(passes) == 26:
            P = [first_per_idx(p) for p in passes]
            for label, first, last in (("passes 0-4 vs 21-25", list(range(0, 5)), list(range(21, 26))),
                                       ("passes 1-5 vs 21-25", list(range(1, 6)), list(range(21, 26)))):
                common = set.intersection(*[set(P[k]) for k in first + last])
                fm = mean(mean(P[k][i] for k in first) for i in common)
                lm = mean(mean(P[k][i] for k in last) for i in common)
                print(f"[claim6] {label}: n={len(common)} first5 {100*fm:.2f} last5 {100*lm:.2f}")
            # variant: examples present in each of the first 5 and each of the last 5 (same as above), and
            # variant without the 'all 10' restriction (pass means of available examples)
            fm = mean(mean(P[k].values()) for k in range(0, 5)); lm = mean(mean(P[k].values()) for k in range(21, 26))
            print(f"[claim6-alt] no common-set restriction, mean of pass means: first5 {100*fm:.2f} last5 {100*lm:.2f}")
        if test:
            print(f"test.jsonl: rows {len(test)}, distinct idx {len(set(r['idx_in_split'] for r in test))}, sum {sum(float(r['metric_output']) for r in test):.4f}")


if __name__ == "__main__":
    main()
