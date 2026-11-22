# Checks the smaller search val set and the final race of GEPA-SubVal100 on made-up data (no model, no GPU).
#   python -m scripts.check_search_val_subset      (from the repo root, in the run environment)
# The subset must be fixed per seed, the race must be paid from the budget and return the best candidate on
# the whole val set, and every other optimizer must be unchanged.
import json
import os
import tempfile
from types import SimpleNamespace

from gepa_artifact.gepa import gepa as gepa_module
from gepa_artifact.gepa.gepa import GEPA
from scripts.experiment_configs import get_optimizers


class Log:
    def __init__(self):
        self.lines = []

    def log(self, text):
        self.lines.append(text)


def fake_gepa(seed=0, search_val_size=100, top_k=5, run_dir=None):
    return SimpleNamespace(seed=seed, search_val_size=search_val_size, final_race_top_k=top_k, logger=Log(), run_dir=run_dir)


class FakeState:
    # candidate i is the list of its scores on the whole val set; the search has seen only search_ids
    def __init__(self, candidates, search_ids, total_num_evals=0):
        self.program_candidates = candidates
        self.prog_candidate_val_subscores = [[c[i] for i in search_ids] for c in candidates]
        self.per_program_tracked_scores = [100 * sum(s) / len(s) for s in self.prog_candidate_val_subscores]
        self.total_num_evals = total_num_evals
        self.saved = 0

    def save(self, run_dir):
        self.saved += 1


# 1. the subset: fixed for a seed, right sizes, no overlap, sorted, different for another seed
for n, k in ((300, 100), (111, 100), (300, 50)):
    a = GEPA.split_valset_for_search(fake_gepa(0, k), n)
    assert a == GEPA.split_valset_for_search(fake_gepa(0, k), n)
    search, held_out = a
    assert len(search) == k and len(held_out) == n - k
    assert sorted(search + held_out) == list(range(n)) and search == sorted(search) and held_out == sorted(held_out)
s0, s1, s2 = (GEPA.split_valset_for_search(fake_gepa(s), 300)[0] for s in (0, 1, 2))
assert s0 != s1 and s1 != s2 and s0 != s2
# a third of the examples, so about a third of each subset is shared with another seed's; not the same draw shifted
assert 15 < len(set(s0) & set(s1)) < 55

# 2. the race on 6 candidates over 30 val examples (search subset: 10): the top 5 on the subset are raced,
#    the winner is the best on all 30, and the race is counted
n, k = 30, 10
search_ids, held_out_ids = GEPA.split_valset_for_search(fake_gepa(0, k), n)


def cand(search_correct, held_out_correct):
    c = [0.0] * n
    for i in search_ids[:search_correct]:
        c[i] = 1.0
    for i in held_out_ids[:held_out_correct]:
        c[i] = 1.0
    return c


#             subset  held-out  whole val set
cands = [cand(3, 2),   # 0: 30   10   16.7   not a finalist (sixth on the subset)
         cand(8, 4),   # 1: 80   20   40.0   best on the subset, weak on the rest
         cand(7, 16),  # 2: 70   80   76.7   the winner
         cand(7, 10),  # 3: 70   50   56.7
         cand(6, 16),  # 4: 60   80   73.3
         cand(5, 20)]  # 5: 50  100   83.3 on the whole set, but fifth on the subset: still a finalist
calls = []


def evaluator(program):
    calls.append(program)
    sub = [program[i] for i in held_out_ids]
    return 100 * sum(sub) / len(sub), [None] * len(sub), sub


with tempfile.TemporaryDirectory() as d:
    g, st = fake_gepa(0, k, 5, d), FakeState(cands, search_ids, total_num_evals=1000)
    winner = GEPA.run_final_race(g, st, evaluator, search_ids, held_out_ids)
    assert calls == [cands[1], cands[2], cands[3], cands[4], cands[5]], "finalists in subset order, ties: the older one"
    assert winner == 5, winner
    assert st.total_num_evals == 1000 + 5 * 20 and st.saved == 1
    out = json.load(open(os.path.join(d, "final_race.json")))
    assert out["returned_program_idx"] == 5 and out["search_subset_best_program_idx"] == 1
    assert out["search_val_ids"] == search_ids and out["held_out_val_ids"] == held_out_ids and out["total_num_evals"] == 1100
    assert [r["program_idx"] for r in out["finalists"]] == [1, 2, 3, 4, 5]
    assert [round(r["full_val_score"], 1) for r in out["finalists"]] == [40.0, 76.7, 56.7, 73.3, 83.3]

    # 3. top 4 only: candidate 5 is not raced, so candidate 2 wins
    calls.clear()
    g, st = fake_gepa(0, k, 4, d), FakeState(cands, search_ids)
    assert GEPA.run_final_race(g, st, evaluator, search_ids, held_out_ids) == 2
    assert len(calls) == 4 and st.total_num_evals == 4 * 20

    # 4. a tie on the whole val set goes to the candidate that was better on the search subset
    tie = [cand(6, 10), cand(8, 8)]
    g, st = fake_gepa(0, k, 5, d), FakeState(tie, search_ids)
    assert GEPA.run_final_race(g, st, evaluator, search_ids, held_out_ids) == 1

    # 5. fewer candidates than top_k: all of them are raced; a single candidate is returned
    calls.clear()
    g, st = fake_gepa(0, k, 5, d), FakeState([cand(3, 2)], search_ids)
    assert GEPA.run_final_race(g, st, evaluator, search_ids, held_out_ids) == 0 and len(calls) == 1

# 6. compile(): the search gets the subset and the budget minus the race; the full budget is restored
seen = {}
valset = [f"val{i}" for i in range(300)]


class FakeEvaluate:
    def __init__(self, devset, **kwargs):
        self.devset = devset

    def __call__(self, program):
        seen.setdefault("race_devsets", []).append(list(self.devset))
        return 0.0, [None] * len(self.devset), [0.0] * len(self.devset)


def fake_search(base_dspy_program, trainset, valset):
    seen["search_valset"], seen["search_budget"] = list(valset), g.max_metric_calls
    st = FakeState([[0.0] * 300], GEPA.split_valset_for_search(g, 300)[0], total_num_evals=g.max_metric_calls + 40)
    return st


real_evaluate = gepa_module.dspy.Evaluate
gepa_module.dspy.Evaluate = FakeEvaluate
try:
    with tempfile.TemporaryDirectory() as d:
        for budget in (6871, 7051):
            g = fake_gepa(1, 100, 5, d)
            g.max_metric_calls, g.metric_fn, g.num_threads, g.failure_score, g.valset_provided = budget, None, 1, 0, None
            g.gepa = fake_search
            g.split_valset_for_search = lambda size: GEPA.split_valset_for_search(g, size)
            g.run_final_race = lambda *a: GEPA.run_final_race(g, *a)
            g.compile_with_search_val_subset = lambda *a: GEPA.compile_with_search_val_subset(g, *a)
            seen.clear()
            best = GEPA.compile(g, "student", ["train0"], valset)
            ids, rest = GEPA.split_valset_for_search(g, 300)
            assert seen["search_valset"] == [valset[i] for i in ids] and len(ids) == 100
            assert seen["search_budget"] == budget - 1000, seen["search_budget"]
            assert seen["race_devsets"] == [[valset[i] for i in rest]] and len(rest) == 200  # one candidate
            assert g.max_metric_calls == budget and best == [0.0] * 300
        # a budget that cannot pay the race is refused before anything runs
        g.max_metric_calls = 1050
        try:
            GEPA.compile(g, "student", ["train0"], valset)
            raise SystemExit("a budget below the race cost was accepted")
        except AssertionError:
            pass
finally:
    gepa_module.dspy.Evaluate = real_evaluate

# 7. optimizer list: every earlier entry keeps its index, GEPA-SubVal100 follows them and differs from GEPA only
#    by the two new arguments; GEPA itself has neither
names = [name for name, _ in get_optimizers()]
assert names[:8] == ["Baseline", "MIPROv2-Heavy", "GEPA-MERGE", "GEPA", "Abl-SelectBestCandidate", "GRPO", "Abl-BeamSearch", "GEPA-SubVal100"], names
cfg = dict(get_optimizers())
base, sub = cfg["GEPA"], cfg["GEPA-SubVal100"]
new_args = {"search_val_size": 100, "final_race_top_k": 5}
assert {k: v for k, v in sub.init_args.items() if k not in new_args} == base.init_args
assert {k: sub.init_args[k] for k in new_args} == new_args and not set(new_args) & set(base.init_args)
assert sub.langProBe_configs == base.langProBe_configs and sub.compile_args == base.compile_args
for name, opt in get_optimizers():
    if opt is not None and opt.optimizer is GEPA and not name.startswith("GEPA-SubVal100"):
        assert "search_val_size" not in opt.init_args, name

# 8. the new arguments are off by default, and need a metric-call budget and val-only tracking
import inspect
sig = inspect.signature(GEPA.__init__).parameters
assert sig["search_val_size"].default is None and sig["final_race_top_k"].default == 5
from scripts.run_experiments import uses_gepa_setup
assert uses_gepa_setup("GEPA-SubVal100")

print("all search val subset checks passed")
