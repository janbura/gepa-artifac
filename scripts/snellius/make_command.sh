#!/usr/bin/env bash
# Prints one run_experiments command for qwen3-8b; qwen_job.sh fills in {portnum}.
#   usage: make_command.sh <benchmark> <program> <optimizer> [extra run_experiments args]
#   e.g.   make_command.sh HotpotQABench HotpotMultiHop Baseline >> cmds.txt
# The names are checked the way run_experiments asserts them, so a typo fails here instead of on
# the GPU after the server warm-up. Takes 30-60 s, because it builds every benchmark.
# Real chain runs need --use_cache_from_opt (MIPROv2-Heavy <- Baseline, GEPA family <- MIPROv2-Heavy),
# as in launch_commands; pilots leave it out (F44). A reminder goes to stderr.
[ $# -ge 3 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
B=$1; P=$2; O=$3; shift 3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv/bin/python
ERRS=$(mktemp)
OUT=$(cd "$REPO" && "$PY" - "$B" "$P" "$O" 2>"$ERRS" <<'EOF' | tail -1
import sys
from scripts.experiment_configs import get_benchmarks, get_optimizers
b, p, o = sys.argv[1:4]
metas, opts = get_benchmarks(), get_optimizers()
bnames = [m.name or m.benchmark.__name__ for m in metas]
onames = [name for name, _ in opts]
if b not in bnames:
    sys.exit(print(f"ERR unknown benchmark {b!r}; known: {', '.join(bnames)}"))
if o not in onames:
    sys.exit(print(f"ERR unknown optimizer {o!r}; known: {', '.join(onames)}"))
bi, oi = bnames.index(b), onames.index(o)
pnames = [getattr(x, "_name", x.__class__.__name__) for x in metas[bi].program]
if p not in pnames:
    sys.exit(print(f"ERR program {p!r} is not in {b}; known: {', '.join(pnames)}"))
cfg = opts[oi][1]
src = cfg.langProBe_configs.get("use_cache_from_opt") if cfg is not None else None
print("OK", bi, pnames.index(p), oi, src or "-")
EOF
)
case "$OUT" in
  "OK "*) read -r _ BI PI OI SRC <<< "$OUT" ;;
  "ERR "*) echo "make_command.sh: ${OUT#ERR }" >&2; rm -f "$ERRS"; exit 1 ;;
  *) echo "make_command.sh: could not read scripts/experiment_configs.py:" >&2; tail -5 "$ERRS" >&2; rm -f "$ERRS"; exit 1 ;;
esac
rm -f "$ERRS"
if [ "$SRC" != "-" ] && [[ " $* " != *" --use_cache_from_opt "* ]]; then
  echo "make_command.sh: note: a real $O run needs '--use_cache_from_opt $SRC' (as in launch_commands); pilots leave it out" >&2
fi
LM='{"name": "qwen3-8b", "model": "openai/arbor:qwen/qwen3-8b", "api_key": "API_KEY", "api_base": "http://localhost:{portnum}/v1/", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "launch_kwargs": {"max_context_length": 8192}}'
echo "$PY -m scripts.run_experiments --bm_idx $BI --benchmark_name $B --num_threads 32 --program_idx $PI --prog_name $P --opt_idx $OI --optim_name $O --lm_config '$LM' --seed 0${*:+ $*}"
