#!/usr/bin/env bash
# fails when a run has no result; also counts error lines, parse failures and logged test examples
# (failed examples are scored 0 and not written to test.jsonl, F29; an example can be logged twice,
# so unique examples are counted)
# usage: check_run.sh <run name>    e.g. check_run.sh hoverBench_HoverMultiHop_Baseline_qwen3-8b
#   SEED=<n> for another seed (default 0), GEPA_EXPERIMENT_DIR=<dir> for another experiment directory
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${SEED:-0}
R=${GEPA_EXPERIMENT_DIR:-$REPO/experiment_runs_data}/experiment_runs/seed_$SEED/$1
[ -f "$R/evaluation_results/evaluation_result.txt" ] || { echo "no result for $1 (seed $SEED)"; exit 1; }
E=$(cat "$R"/run_log*.txt 2>/dev/null | grep -ciE "error|exception")
P=$(cat "$R"/run_log*.txt 2>/dev/null | grep -c "Failed to parse response")
T=$(grep -o '"idx_in_split": [0-9]*' "$R/metric_logs/test.jsonl" 2>/dev/null | sort -u | wc -l)
echo "$1 (seed $SEED): score $(sed -n 2p "$R/evaluation_results/evaluation_result.txt" | cut -d, -f1), error lines $E, parse failures $P, test examples logged $T"
