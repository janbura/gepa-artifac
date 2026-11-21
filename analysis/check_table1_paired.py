# checks of table1_paired.py on made-up runs whose results can be worked out by hand
#   python analysis/check_table1_paired.py
import os, json, statistics, tempfile, importlib.util

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tp", os.path.join(here, "table1_paired.py"))
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

# 1. Holm by hand: sorted 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005, 3 x 0.01, 2 x 0.03, then max(1 x 0.04, 0.06)
adj = tp.holm_adjust([0.01, 0.04, 0.03, 0.005])
assert all(abs(x - y) < 1e-12 for x, y in zip(adj, [0.03, 0.06, 0.06, 0.02])), adj
print("Holm ok")

# 2. sign-flip test and bootstrap on cases with a known answer
assert tp.signflip([1.0] * 40) == 1 / (tp.B + 1)  # only the all-plus or all-minus vector reaches |mean| = 1
assert tp.signflip([1.0, -1.0] * 10) == 1.0  # observed mean 0, so every flipped mean reaches it
assert tp.boot([2.0] * 50) == (2.0, 2.0)
lo, hi = tp.boot([0.0] * 50 + [100.0] * 50)
assert 35 < lo < 50 < hi < 65, (lo, hi)  # mean 50, SE 5: about 40 to 60
print("sign-flip and bootstrap ok")


# 3. made-up runs for all 60 cells, with a missing example, a re-sent line and an AIME problem with 6 lines
def value(b, j, s, u, t):
    if b == "AIME":
        return 1.0 if (u + t + j + s) % 4 == 0 else 0.0
    if b in ("IFBench", "LiveBench"):
        return ((u + j + s) % 3) / 2  # partial credit like the real metrics
    return 1.0 if (u * (j + 2) + s) % 5 < 2 else 0.0


root = tempfile.mkdtemp()
truth = {}
for b in tp.BENCHES:
    for j, o in enumerate(tp.OPTS):
        for s in (0, 1, 2):
            lines, step, units = [], 0, []
            for u in range(tp.UNITS[b]):
                tries = range(tp.TRIES) if b == "AIME" else [0]
                got = []
                for t in tries:
                    if s == 0 and u == 0 and t == max(tries):
                        continue  # failed example (AIME: one failed try): no line, scores 0
                    step += 1
                    lines.append((step, u, value(b, j, s, u, t)))
                    got.append(value(b, j, s, u, t))
                units.append(sum(got) / tp.TRIES if b == "AIME" else (got[0] if got else 0.0))
            if b != "AIME":
                lines.append((step + 1, 1, 1.0 - value(b, j, s, 1, 0)))  # re-sent request: a later line, must not count
            elif s == 1:
                lines.append((step + 1, 2, 1.0))  # a 6th line for problem 2, must not count
            truth[(b, o, s)] = units
            d = os.path.join(root, f"seed_{s}", f"{tp.FOLDER[b]}_{o}_qwen3-8b")
            os.makedirs(os.path.join(d, "metric_logs"))
            os.makedirs(os.path.join(d, "evaluation_results"))
            with open(os.path.join(d, "metric_logs", "test.jsonl"), "w", encoding="utf-8") as f:
                for st, u, v in reversed(lines):  # file order reversed: the script must sort by step
                    f.write(json.dumps({"metric_output": v, "step_counter": st, "idx_in_split": u}) + "\n")
            with open(os.path.join(d, "evaluation_results", "evaluation_result.txt"), "w") as f:
                f.write(f"score,cost,input_tokens,output_tokens,\n{100 * statistics.mean(units):.2f},0,0,0,\n")

S, REP, worst = tp.load(root)
assert all(abs(S[k][u] - truth[k][u]) < 1e-12 for k in truth for u in range(len(truth[k]))), "rebuild differs"
assert worst <= 0.005
print("rebuild ok (first line in step order, missing = 0, AIME at most 5 tries per problem)")

# d by hand for HotpotQA GEPA - Baseline and AIME MIPROv2 - Baseline
for b, a, c in (("HotpotQA", "GEPA", "Baseline"), ("AIME", "MIPROv2-Heavy", "Baseline")):
    want = statistics.mean(100 * (statistics.mean(truth[(b, a, s)][u] for s in (0, 1, 2))
                                  - statistics.mean(truth[(b, c, s)][u] for s in (0, 1, 2)))
                           for u in range(tp.UNITS[b]))
    row, _ = tp.pair_row(S, REP, b, a, c, (0, 1, 2), (0, 1, 2), False)
    assert abs(row["d"] - want) < 1e-9, (b, row["d"], want)
print("d ok")

# 4. the whole report with fewer resamples (speed only)
tp.B = 400
text, numbers = tp.report(S, REP, worst)
rows = numbers["rows"]
assert len(rows) == 30 and sum(1 for r in rows if "p" in r) == 28 and text.count("same search twice") == 2
for agg in numbers["aggregate"]:
    want = statistics.mean(r["d"] for r in rows if r["pair"] == agg["pair"])
    assert abs(agg["d"] - want) < 1e-9, agg
sens = numbers["without_dagger"]
assert len(sens) == 13 and sum(1 for r in sens if r.get("no_run_left")) == 3, len(sens)
print("report ok (30 rows, 28 tests, aggregate = mean of the benchmark rows, 13 rows without the dagger runs)")

# 5. a run whose rebuild misses its reported score stops the script
f = os.path.join(root, "seed_1", f"{tp.FOLDER['HoVer']}_GEPA_qwen3-8b", "evaluation_results", "evaluation_result.txt")
with open(f, "w") as fh:
    fh.write(f"score,cost,input_tokens,output_tokens,\n{100 * statistics.mean(truth[('HoVer', 'GEPA', 1)]) + 0.34:.2f},0,0,0,\n")
try:
    tp.load(root)
    raise AssertionError("no stop")
except SystemExit as e:
    assert "stop (plan 2.13)" in str(e), e
print("stop rule ok")
print("all checks passed")
