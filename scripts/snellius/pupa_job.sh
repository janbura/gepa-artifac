#!/usr/bin/env bash
#SBATCH --job-name=gepa-pupa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --output=logs/%x-%j.out
# PUPA without an OpenAI key (plan section 2.5): serves the external model (gpt-oss-120b) with vLLM on
# the second GPU of the job, then runs qwen_job.sh unchanged (Arbor + Qwen3-8B on the first GPU) with
# the same arguments. The runs reach the external model through PUPA_EXTERNAL_* (papillon_utils.py).
# Submit from the repo root (logs/ must exist there):
#   sbatch -p gpu_h100 -t 04:00:00 -J p-pilot scripts/snellius/pupa_job.sh cmds_p_pilot.txt /home/scur0934/arbor091
# PAPILLON turns a failed call into an empty answer without any error (papillon_program.py), so when the
# external server exits or fails its health check 3 times in a row, this script stops the runs (job fails).
set -u
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
mkdir -p logs
JOB=${SLURM_JOB_ID:-local$$}
START=$(date +%s)
EXT_VENV=${EXT_VENV:-$HOME/vllm0110}
EXT_MODEL=${EXT_MODEL:-openai/gpt-oss-120b}
EXT_REVISION=${EXT_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}
EXT_NAME=gpt-oss-120b
EXT_PORT=$(( 40000 + ${SLURM_JOB_ID:-$$} % 20000 ))
ELOG=logs/external-$JOB.log
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# first GPU of the job for Qwen (Arbor), second one for the external model
IFS=, read -r G0 G1 _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
[ -n "${G1:-}" ] || { echo "this job needs 2 GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})"; exit 1; }
nvidia-smi
echo "external model: $EXT_MODEL revision $EXT_REVISION served as $EXT_NAME on port $EXT_PORT, GPU $G1, from $EXT_VENV (vllm $("$EXT_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo '?'))"
EWORK=/scratch-shared/$USER/external_$JOB
mkdir -p "$EWORK"
(export CUDA_VISIBLE_DEVICES=$G1 VLLM_CACHE_ROOT=$EWORK/vllm-cache && cd "$EWORK" && exec "$EXT_VENV/bin/vllm" serve "$EXT_MODEL" --revision "$EXT_REVISION" --served-model-name "$EXT_NAME" --port "$EXT_PORT") > "$ELOG" 2>&1 &
EXT_PID=$!
trap 'kill $EXT_PID 2>/dev/null' EXIT

# loading 65 GB of weights takes a while; wait up to 30 min for a real answer
READY=""
for i in $(seq 1 90); do
  kill -0 $EXT_PID 2>/dev/null || break
  if curl -s -m 300 "http://localhost:$EXT_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$EXT_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say ok.\"}], \"max_tokens\": 256}" | grep -q '"choices"'; then
    READY=1; break
  fi
  sleep 20
done
echo "external server ready: ${READY:-no} after $(( $(date +%s) - START )) s"
[ -n "$READY" ] || { echo "external server did not start"; tail -50 "$ELOG"; exit 1; }

export PUPA_EXTERNAL_MODEL=openai/$EXT_NAME PUPA_EXTERNAL_API_BASE=http://localhost:$EXT_PORT/v1 PUPA_EXTERNAL_API_KEY=local
# gpt-oss counts its hidden reasoning in max_tokens; 4000 instead of dspy's 1000 (plan 2.5)
export PUPA_EXTERNAL_MAX_TOKENS=4000
echo "runs use: PUPA_EXTERNAL_MODEL=$PUPA_EXTERNAL_MODEL PUPA_EXTERNAL_API_BASE=$PUPA_EXTERNAL_API_BASE PUPA_EXTERNAL_MAX_TOKENS=$PUPA_EXTERNAL_MAX_TOKENS"

CUDA_VISIBLE_DEVICES=$G0 bash scripts/snellius/qwen_job.sh "$@" &
QJ=$!

descendants() {
  local p
  for p in $(pgrep -P "$1"); do echo "$p"; descendants "$p"; done
}
stop_runs() {
  # the same stop file as qwen_job.sh, so it skips its remaining commands and exits 1
  echo "=== external watchdog $(date): $1; stopping the runs of this job" | tee -a "/scratch-shared/$USER/arbor_$JOB/stopped_by_watchdog"
  local p sig
  for sig in TERM KILL; do
    for p in $(descendants $QJ); do
      grep -qa -E "scripts.run_experiments|scripts.pupa_smoke" /proc/$p/cmdline 2>/dev/null && kill -$sig $p 2>/dev/null
    done
    [ $sig = TERM ] && sleep 30
  done
}
ext_watch() {
  local fails=0
  while sleep 60; do
    kill -0 $QJ 2>/dev/null || return
    kill -0 $EXT_PID 2>/dev/null || { stop_runs "the external server exited"; return; }
    if curl -sf -m 30 "http://localhost:$EXT_PORT/health" > /dev/null; then fails=0; else fails=$(( fails + 1 )); fi
    [ $fails -ge 3 ] && { stop_runs "the external server failed its health check 3 times in a row"; return; }
  done
}
ext_watch &
EW=$!

wait $QJ
STATUS=$?
kill $EW 2>/dev/null
echo "external server: $(grep -c 'POST /v1/chat/completions' "$ELOG") chat requests, $(grep 'POST /v1/chat/completions' "$ELOG" | grep -vc '" 200') not answered with HTTP 200"
kill $EXT_PID 2>/dev/null
sleep 15
pkill -P $EXT_PID 2>/dev/null
echo "pupa job done $(date), exit $STATUS"
exit $STATUS
