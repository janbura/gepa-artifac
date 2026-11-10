#!/usr/bin/env bash
# Prints the run_experiments command for a qwen3-8b GRPO run; grpo_job.sh fills in {portnum}.
#   usage: make_grpo_command.sh <benchmark> <program> [extra run_experiments args]
#   e.g.   make_grpo_command.sh hoverBench HoverMultiHop >> cmds.txt
# Unlike make_command.sh, --lm_config is the full qwen3-8b entry of LM_CONFIGS, including the
# train_kwargs that GRPO hands to Arbor's trainer, as in the authors' launch_commands and config.json.
# Without them Arbor trains with its own defaults. Shortened runs: put GEPA_EXPERIMENT_DIR=<dir>
# GEPA_GRPO_TRAIN_STEPS=<n> in front of the printed line in the commands file.
[ $# -ge 2 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
B=$1; P=$2; shift 2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv/bin/python
cd "$REPO" && "$PY" - "$PY" "$B" "$P" "$*" <<'EOF'
import json, sys
from scripts.experiment_configs import get_benchmarks, get_optimizers, LM_CONFIGS
py, b, p, extra = sys.argv[1:5]
metas, opts = get_benchmarks(), get_optimizers()
bnames = [m.name or m.benchmark.__name__ for m in metas]
onames = [name for name, _ in opts]
if b not in bnames:
    sys.exit(f"make_grpo_command.sh: unknown benchmark {b!r}; known: {', '.join(bnames)}")
bi, oi = bnames.index(b), onames.index("GRPO")
pnames = [getattr(x, "_name", x.__class__.__name__) for x in metas[bi].program]
if p not in pnames:
    sys.exit(f"make_grpo_command.sh: program {p!r} is not in {b}; known: {', '.join(pnames)}")
lm = next(c for c in LM_CONFIGS if c["name"] == "qwen3-8b")
lm_json = json.dumps(lm)
assert "'" not in lm_json and "train_kwargs" in lm and "{portnum}" in lm["api_base"]
print(f"{py} -m scripts.run_experiments --bm_idx {bi} --benchmark_name {b} --num_threads 32 "
      f"--program_idx {pnames.index(p)} --prog_name {p} --opt_idx {oi} --optim_name GRPO "
      f"--lm_config '{lm_json}' --seed 0" + (f" {extra}" if extra else ""))
EOF
