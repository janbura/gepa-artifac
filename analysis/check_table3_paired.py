# Checks of table3_paired.py before any Table 3 result exists (plan 2.14).
#   python analysis/check_table3_paired.py <results dir with table1_h100/ and pupa_gptoss/>
# 1. search statistics on a made-up run log with known counts
# 2. the existing Baseline and GEPA runs rebuild to their reported scores (incl. PUPA's rebuild of plan 2.10)
# 3. fake Table 3 runs: SelectBestCandidate = a copy of the GEPA run, BeamSearch = a copy of the Baseline run.
#    Then GEPA - SelectBest must be exactly 0 and GEPA - BeamSearch must equal the published GEPA - Baseline
#    numbers (table1_h100/paired_output.md and the PUPA analysis).
import os, sys, shutil, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table3_paired as t3

root = sys.argv[1]
tmp = tempfile.mkdtemp()

# 1. made-up log: base 40, two accepted programs (45, 50), five picks of which two are below the best so far
log = """Iteration 0: Base program full valset score: 40.0
Iteration 1: Selected program candidate 0 with base score: 40.0
Iteration 1: Proposed new instruction for predictor x
Iteration 1: Full valset score for new program: 45.0
Iteration 1: New program candidate index: 1
Iteration 2: Selected program candidate 0 with base score: 40.0
Iteration 2: Proposed new instruction for predictor x
Iteration 3: Selected program candidate 1 with base score: 45.0
Iteration 3: Full valset score for new program: 50.0
Iteration 3: New program candidate index: 2
Iteration 4: Selected program candidate 1 with base score: 45.0
Iteration 5: Selected program candidate 2 with base score: 50.0
"""
fake = os.path.join(tmp, "fake_run")
os.makedirs(fake)
open(os.path.join(fake, "run_log.txt"), "w").write(log)
st = t3.search_stats(fake)
assert st == {"iterations": 5, "proposals": 2, "accepted": 2, "picks": 5, "picks_not_best": 2}, st

# 2. and 3. fake Table 3 runs from the real Baseline and GEPA runs
work = os.path.join(tmp, "results")
for sub in ("table1_h100", "pupa_gptoss"):
    shutil.copytree(os.path.join(root, sub), os.path.join(work, sub))
for t in t3.TASKS:
    for s in t3.SEEDS:
        for src, dst in (("GEPA", "Abl-SelectBestCandidate"), ("Baseline", "Abl-BeamSearch")):
            shutil.copytree(t3.run_dir(work, t, src, s), t3.run_dir(work, t, dst, s))

S, REP, missing, worst = t3.load(work)
assert not missing and len(S) == 48, (missing, len(S))
assert worst <= 0.005
text, num = t3.report(work, S, REP, missing, worst)
tests = {(r["task"], r["pair"]): r for r in num["tests"]}
assert len(tests) == 8
for t in t3.TASKS:
    r = tests[(t, "GEPA - SelectBestCandidate")]
    assert r["d"] == 0 and r["p"] == 1.0 and r["per_seed"] == [0, 0, 0], r
# GEPA - Baseline as published (Table 1 paired tests of 2026-11-16, PUPA analysis of 2026-11-15)
published = {"HotpotQA": (21.67, 17.33, 26.11), "IFBench": (-1.64, None, None), "HoVer": (13.89, None, None),
             "PUPA": (6.16, None, None)}
for t, (d, lo, hi) in published.items():
    r = tests[(t, "GEPA - BeamSearch")]
    assert round(r["d"], 2) == d, (t, r["d"])
    if lo is not None:
        assert [round(x, 2) for x in r["ci95"]] == [lo, hi], (t, r["ci95"])
# the table: our Baseline and GEPA means as in the report (HotpotQA 38.78 / 60.44, IFBench 34.63 / 32.99, HoVer
# 37.11 / 51.00, PUPA 86.22 / 92.37)
for t, (b, g) in {"HotpotQA": (38.78, 60.44), "IFBench": (34.63, 32.99), "HoVer": (37.11, 51.00),
                  "PUPA": (86.22, 92.37)}.items():
    vb = [REP[(t, "Baseline", s)] for s in t3.SEEDS]
    vg = [REP[(t, "GEPA", s)] for s in t3.SEEDS]
    assert round(sum(vb) / 3, 2) == b and round(sum(vg) / 3, 2) == g, (t, vb, vg)
assert "### Observation 3" in text and "### Search statistics" in text
# search statistics exist for the GEPA rows (run logs are in the packed runs)
assert all(x["runs"] == 3 for x in num["search"] if x["method"] == "GEPA"), num["search"]
shutil.rmtree(tmp)
print("all table3_paired checks passed")
