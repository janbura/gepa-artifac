# Checks the parent selection of the Table 3 ablations on made-up candidate lists (no model, no GPU).
#   python -m scripts.check_beam_selection      (from the repo root, in the run environment)
# BeamSearch (beam_size=4) must pick the members of the top-4 in turn, newest/least-picked first;
# SelectBestCandidate and the optimizer list must be unchanged.
import random
from collections import Counter
from types import SimpleNamespace

from gepa_artifact.gepa.gepa import GEPA
from scripts.experiment_configs import get_optimizers


class Log:
    def __init__(self):
        self.lines = []

    def log(self, text):
        self.lines.append(text)


def fake_gepa(beam_size):
    return SimpleNamespace(beam_size=beam_size, beam_expansions=Counter(), logger=Log())


def fake_state(scores, linearized=True):
    return SimpleNamespace(per_program_tracked_scores=list(scores), program_candidates=list(range(len(scores))),
                           i=0, running_linearized_gepa=linearized, rng1=random.Random(0))


def pick(g, st):
    st.i += 1
    return GEPA.select_next_candidate_to_update(g, st)


# 1. one candidate: always the seed program
g, st = fake_gepa(4), fake_state([0.5])
assert [pick(g, st) for _ in range(3)] == [0, 0, 0]

# 2. six candidates: beam = 5 (0.9), 1 (0.7), 3 (0.7, tie -> older first), 2 (0.6); picked in turn
g, st = fake_gepa(4), fake_state([0.5, 0.7, 0.6, 0.7, 0.4, 0.9])
assert [pick(g, st) for _ in range(8)] == [5, 1, 3, 2, 5, 1, 3, 2]
assert g.logger.lines[0].startswith("Iteration 2: Beam [5, 1, 3, 2]")

# 3. a new best candidate enters the beam: picked next (never picked before), the worst member drops out
g, st = fake_gepa(4), fake_state([0.5, 0.7, 0.6, 0.7, 0.4, 0.9])
assert [pick(g, st) for _ in range(2)] == [5, 1]
st.per_program_tracked_scores.append(0.95)
st.program_candidates.append(6)
assert pick(g, st) == 6
# beam is now 6, 5, 1, 3: 3 was never picked, then a new round starts from the best
assert [pick(g, st) for _ in range(3)] == [3, 6, 5]

# 4. fewer candidates than the beam size: the ones there are, in turn
g, st = fake_gepa(4), fake_state([0.2, 0.3])
assert [pick(g, st) for _ in range(4)] == [1, 0, 1, 0]

# 5. beam_size None: SelectBestCandidate (linearized) still takes the best, first index on ties
g, st = fake_gepa(None), fake_state([0.5, 0.7, 0.7, 0.1])
assert [pick(g, st) for _ in range(3)] == [1, 1, 1]
assert g.logger.lines == []

# 6. optimizer list: the old entries keep their place, Abl-BeamSearch follows them (later entries come after
#    it) and differs from Abl-SelectBestCandidate only by beam_size
names = [name for name, _ in get_optimizers()]
assert names[6] == "Abl-BeamSearch", names
assert names[:6] == ["Baseline", "MIPROv2-Heavy", "GEPA-MERGE", "GEPA", "Abl-SelectBestCandidate", "GRPO"], names
cfg = dict(get_optimizers())
sb, bs = cfg["Abl-SelectBestCandidate"], cfg["Abl-BeamSearch"]
assert {k: v for k, v in bs.init_args.items() if k != "beam_size"} == sb.init_args
assert bs.init_args["beam_size"] == 4 and "beam_size" not in sb.init_args
assert bs.langProBe_configs == sb.langProBe_configs and bs.compile_args == sb.compile_args

# 7. run_experiments gives every GEPA-class optimizer the GEPA arguments (the released check missed the ablations)
from scripts.run_experiments import uses_gepa_setup
for name, opt in get_optimizers():
    is_gepa = opt is not None and opt.optimizer is GEPA
    assert uses_gepa_setup(name) == is_gepa, (name, is_gepa)

print("all beam selection checks passed")
