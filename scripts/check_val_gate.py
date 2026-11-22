# Checks the staged acceptance gate of GEPA-SubVal100-Gate on made-up data (no model, no GPU).
#   python -m scripts.check_val_gate      (from the repo root, in the run environment)
# The gate examples must be fixed per seed, a child must be discarded exactly when it trails its parent on them
# by more than the tolerance, a discarded child must cost only the gate examples, a child that passes must get
# the same val evaluation as one full pass, and every other optimizer must be unchanged.
import inspect
from types import SimpleNamespace

from gepa_artifact.gepa import gepa as gepa_module
from gepa_artifact.gepa.gepa import GEPA
from scripts.experiment_configs import get_optimizers


class Log:
    def __init__(self):
        self.lines = []

    def log(self, text):
        self.lines.append(text)


def fake_gepa(seed=0, size=50, tol=1):
    return SimpleNamespace(seed=seed, val_gate_size=size, val_gate_tolerance=tol, logger=Log(), train_val_size=450)


# 1. gate positions: fixed for a seed, right sizes, no overlap, sorted, different for another seed
gate, rest = GEPA.split_valset_for_gate(fake_gepa(0), 100)
assert (gate, rest) == GEPA.split_valset_for_gate(fake_gepa(0), 100)
assert len(gate) == 50 and len(rest) == 50 and sorted(gate + rest) == list(range(100))
assert gate == sorted(gate) and rest == sorted(rest)
assert gate != GEPA.split_valset_for_gate(fake_gepa(1), 100)[0]
assert gate != list(range(50)), "the gate is a random half, not the first half"


# a program is the list of its scores on the 10 val examples of the search; an evaluator scores it on a devset
# of positions and counts how often it ran
class Evaluator:
    def __init__(self, positions):
        self.positions, self.calls = positions, 0

    def __call__(self, program):
        self.calls += 1
        sub = [program[p] for p in self.positions]
        return round(100 * sum(sub) / len(sub), 2), [("example%d" % p, "prediction%d" % p, program[p]) for p in self.positions], sub


def setup(parent, tol, size=4):
    g = fake_gepa(0, size, tol)
    gate_positions, rest_positions = GEPA.split_valset_for_gate(g, len(parent))
    g.val_gate = {"gate_positions": gate_positions, "rest_positions": rest_positions,
                  "gate_evaluator": Evaluator(gate_positions), "rest_evaluator": Evaluator(rest_positions)}
    st = SimpleNamespace(prog_candidate_val_subscores=[parent], full_program_trace=[{"i": 0}], i=0,
                         total_num_evals=1000, total_num_evals_per_trainval_instance=0.0)
    return g, st


def child_with(parent, g, gate_correct, rest_correct):
    c = [0.0] * len(parent)
    for p in g.val_gate["gate_positions"][:gate_correct]:
        c[p] = 1.0
    for p in g.val_gate["rest_positions"][:rest_correct]:
        c[p] = 1.0
    return c


n = 10
parent = [0.0] * n
g, _ = setup(parent, 1)
for p in g.val_gate["gate_positions"][:3]:
    parent[p] = 1.0   # the parent has 3 of the 4 gate examples right

# 2. tolerance 1: a child with 1 of 4 trails by 2 and is discarded; it costs the 4 gate examples only
g, st = setup(parent, 1)
out = GEPA.evaluate_val_with_gate(g, child_with(parent, g, 1, 6), st, 0)
assert out is None
assert st.total_num_evals == 1004 and g.val_gate["gate_evaluator"].calls == 1 and g.val_gate["rest_evaluator"].calls == 0
t = st.full_program_trace[-1]["val_gate"]
assert t["cut"] is True and t["parent_gate_score"] == 3 and t["child_gate_score"] == 1 and t["parent_program_idx"] == 0
assert "discarded" in g.logger.lines[-1]

# 3. tolerance 1: a child with 2 of 4 trails by exactly 1 and passes; nothing is counted here (the full
#    evaluation is counted by the caller, as before), and the result equals one full pass
g, st = setup(parent, 1)
child = child_with(parent, g, 2, 5)
out = GEPA.evaluate_val_with_gate(g, child, st, 0)
assert out is not None and st.total_num_evals == 1000
score, outputs, subscores = out
full = Evaluator(list(range(n)))(child)
assert (score, outputs, subscores) == full, "two stages must give the val evaluation of one full pass"
assert st.full_program_trace[-1]["val_gate"]["cut"] is False
assert g.val_gate["gate_evaluator"].calls == 1 and g.val_gate["rest_evaluator"].calls == 1

# 4. a child that is better on the gate passes; tolerance 0 discards a child that trails by 1; a larger
#    tolerance lets it pass
g, st = setup(parent, 1)
assert GEPA.evaluate_val_with_gate(g, child_with(parent, g, 4, 0), st, 0) is not None
g, st = setup(parent, 0)
assert GEPA.evaluate_val_with_gate(g, child_with(parent, g, 2, 6), st, 0) is None
g, st = setup(parent, 0)
assert GEPA.evaluate_val_with_gate(g, child_with(parent, g, 3, 0), st, 0) is not None   # equal on the gate
g, st = setup(parent, 2)
assert GEPA.evaluate_val_with_gate(g, child_with(parent, g, 1, 0), st, 0) is not None

# 5. the full-evaluation step uses the two-stage result instead of a second val pass, and counts the whole
#    val set once, as for a child without the gate
recorded = {}


class State(SimpleNamespace):
    def update_state_with_new_program(self, **kw):
        recorded.update(kw)
        return 1, 1


real_log = gepa_module.log_detailed_metrics_after_discovering_new_program
gepa_module.log_detailed_metrics_after_discovering_new_program = lambda **kw: None
try:
    g, st0 = setup(parent, 1)
    child = child_with(parent, g, 2, 5)
    out = GEPA.evaluate_val_with_gate(g, child, st0, 0)
    st = State(total_num_evals=1000, num_full_ds_evals=1, total_num_evals_per_trainval_instance=0.0,
               full_program_trace=[{"i": 0}], i=0)
    g.track_scores_on, g.run_dir, g.use_wandb = "val", None, False
    never = Evaluator(list(range(n)))
    GEPA.run_full_eval_add_new_program_to_gepa_tree(g, new_program=child, gepa_state=st, trainset_evaluator=None,
                                                    valset_evaluator=never, parent_program_idx=[0], valset_eval_output=out)
    assert never.calls == 0, "the val set was scored a second time"
    assert st.total_num_evals == 1000 + n and st.num_full_ds_evals == 2
    assert recorded["valset_subscores"] == child and recorded["valset_score"] == out[0]
    # without the gate: one normal val pass, same counting
    st = State(total_num_evals=1000, num_full_ds_evals=1, total_num_evals_per_trainval_instance=0.0,
               full_program_trace=[{"i": 0}], i=0)
    GEPA.run_full_eval_add_new_program_to_gepa_tree(g, new_program=child, gepa_state=st, trainset_evaluator=None,
                                                    valset_evaluator=never, parent_program_idx=[0])
    assert never.calls == 1 and st.total_num_evals == 1000 + n and recorded["valset_subscores"] == child
finally:
    gepa_module.log_detailed_metrics_after_discovering_new_program = real_log

# 6. optimizer list: every earlier entry keeps its index; GEPA-SubVal100-Gate is last and differs from
#    GEPA-SubVal100 only by the two gate arguments; no other optimizer has them
names = [name for name, _ in get_optimizers()]
assert names == ["Baseline", "MIPROv2-Heavy", "GEPA-MERGE", "GEPA", "Abl-SelectBestCandidate", "GRPO", "Abl-BeamSearch",
                 "GEPA-SubVal100", "GEPA-SubVal100-Gate"], names
cfg = dict(get_optimizers())
e1, e2 = cfg["GEPA-SubVal100"], cfg["GEPA-SubVal100-Gate"]
gate_args = {"val_gate_size": 50, "val_gate_tolerance": 1}
assert {k: v for k, v in e2.init_args.items() if k not in gate_args} == e1.init_args
assert {k: e2.init_args[k] for k in gate_args} == gate_args
assert e2.langProBe_configs == e1.langProBe_configs and e2.compile_args == e1.compile_args
for name, opt in get_optimizers():
    if opt is not None and opt.optimizer is GEPA and name != "GEPA-SubVal100-Gate":
        assert not set(gate_args) & set(opt.init_args), name

# 7. the gate is off by default
sig = inspect.signature(GEPA.__init__).parameters
assert sig["val_gate_size"].default is None and sig["val_gate_tolerance"].default == 1
from scripts.run_experiments import uses_gepa_setup
assert uses_gepa_setup("GEPA-SubVal100-Gate")

print("all val gate checks passed")
