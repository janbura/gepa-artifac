#!/usr/bin/env bash
#SBATCH --job-name=gepa-qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
# Starts Arbor (vLLM) for Qwen3-8B on one GPU, waits until it answers, then runs the
# run_experiments commands listed in the file given as first argument ({portnum} is filled in).
# Submit from the repo root (logs/ must exist there for the --output file):
#   sbatch -p gpu_a100 -t 05:00:00 scripts/snellius/qwen_job.sh my_commands.txt
# Build the command lines with scripts/snellius/make_command.sh.
#
# Commands file: one shell command per line, run in order; empty lines and lines starting with #
# are skipped. A line ending in '&' runs in the background, so several runs share the server
# (pilot 4.4). A line 'wait' waits for the background runs started so far, and the job waits for
# all of them before it stops Arbor. A VAR=value prefix sets the environment of one line, e.g.
#   GEPA_EXPERIMENT_DIR=/scratch-shared/$USER/p44/n3 GEPA_MAX_METRIC_CALLS=1500 timeout 30m <command> &
#
# Watchdog: if Arbor exits, vLLM stops answering its health check, or Arbor writes no response for
# STALL_MINUTES (default 20) while runs are active, the job stops its runs and exits 1. Without it,
# a GEPA run whose LM calls fail loops forever without using up its budget (F54).
# Optional: ARBOR_VENV=<venv> serves from another Arbor environment, e.g. one with the authors'
# vLLM 0.9.1 (envs/arbor-vllm091-requirements.txt); default gepa_artifact/utils/arbor/.venv.
# Optional: PROGRESS_DIRS="<dir> ..." logs once a minute, for every run under those experiment
# dirs, its metric calls (lines in metric_logs/{train,val}.jsonl) and reflections (lines in
# instruction_proposer_inpouts.jsonl) to logs/progress-<job>.csv (pilot 4.4, F44).
set -u
REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
mkdir -p logs
export PATH=$HOME/.local/bin:$PATH
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} WANDB_API_KEY=${WANDB_API_KEY:-dummy}
export WANDB_MODE=${WANDB_MODE:-disabled} GEPA_USE_WANDB=${GEPA_USE_WANDB:-0}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
CMDS=$1
JOB=${SLURM_JOB_ID:-local$$}
STALL_MINUTES=${STALL_MINUTES:-20}
nvidia-smi -L
START=$(date +%s)
DIRTY=$(git diff --quiet HEAD 2>/dev/null || echo " + uncommitted changes")
echo "start $(date) on $(hostname), commit $(git rev-parse --short HEAD)$DIRTY, commands: $CMDS"
ARBOR_VENV=${ARBOR_VENV:-$REPO/gepa_artifact/utils/arbor/.venv}
echo "arbor venv: $ARBOR_VENV (vllm $("$ARBOR_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo '?'))"

# arbor writes request/response logs to ./storage, keep them out of the repo
WORK=/scratch-shared/$USER/arbor_$JOB
mkdir -p "$WORK"
STOPFILE=$WORK/stopped_by_watchdog
PORT=$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))
ALOG=logs/arbor-$JOB.log
(source "$ARBOR_VENV/bin/activate" && export PYTHONPATH="$REPO/gepa_artifact/utils/arbor" && cd "$WORK" && exec python -m arbor.cli serve --arbor-config "$REPO/scripts/snellius/arbor_inference_1gpu.yaml" --port $PORT > "$REPO/$ALOG" 2>&1) &
ARBOR_PID=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > "logs/gpu-$JOB.csv" &
SMI_PID=$!
WATCH_PID=
PROG_PID=

# Arbor's request log goes into logs/ unless it is large (long GEPA runs); it stays on scratch anyway
copy_arbor_logs() {
  local mb
  mb=$(du -sm "$WORK/storage/logs" 2>/dev/null | cut -f1)
  if [ -n "$mb" ] && [ "$mb" -le "${COPY_ARBOR_LOGS_MB:-1000}" ]; then
    cp -r "$WORK/storage/logs" "logs/arbor_requests_$JOB" && echo "arbor request logs: logs/arbor_requests_$JOB ($mb MB)"
  else
    echo "arbor request logs: $WORK/storage/logs (${mb:-?} MB, left on scratch)"
  fi
}
cleanup() {
  trap - EXIT
  kill ${SMI_PID:-} ${WATCH_PID:-} ${PROG_PID:-} 2>/dev/null
  copy_arbor_logs
  pkill -P $ARBOR_PID 2>/dev/null; kill $ARBOR_PID 2>/dev/null
}
trap cleanup EXIT

# first request makes arbor start vllm; wait until it returns an answer
READY=""
for i in $(seq 1 90); do
  kill -0 $ARBOR_PID 2>/dev/null || break
  grep -qiE "out of memory|larger than the maximum number of tokens|no available memory for the cache|larger than the available KV cache memory" "$ALOG" 2>/dev/null && { echo "vllm did not fit on this gpu"; break; }
  if curl -s -m 600 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model": "arbor:qwen/qwen3-8b", "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 64}' | grep -q '"choices"'; then
    READY=1; break
  fi
  sleep 20
done
UP=$(( $(date +%s) - START ))
echo "server ready: ${READY:-no} after $UP s"
[ -n "$READY" ] || { echo "server did not start"; tail -50 "$ALOG"; exit 1; }
# Arbor prints the vLLM command line, which has vLLM's own port (its /health/ route is the check below)
VLLM_PORT=$(grep -oE 'vllm_serve --model [^ ]+ --port [0-9]+' "$ALOG" | head -1 | grep -oE '[0-9]+$')
echo "vllm port: ${VLLM_PORT:-not found, health check off}"

# run_experiments processes started by this job (any depth below this script)
run_pids() {
  local p
  for p in $(pgrep -P "${1:-$$}"); do
    grep -qa scripts.run_experiments "/proc/$p/cmdline" 2>/dev/null && echo "$p"
    run_pids "$p"
  done
}
stop_runs() {
  echo "=== watchdog $(date): $1; stopping the runs of this job" | tee -a "$STOPFILE"
  kill -TERM $(run_pids) 2>/dev/null
  sleep 30
  kill -KILL $(run_pids) 2>/dev/null
}
watchdog() {
  local fails=0 size last_size=-1 last_change now
  last_change=$(date +%s)
  while sleep 60; do
    kill -0 $ARBOR_PID 2>/dev/null || { stop_runs "Arbor exited"; return; }
    if [ -n "$VLLM_PORT" ]; then
      if curl -sf -m 30 "http://localhost:$VLLM_PORT/health/" > /dev/null; then fails=0; else fails=$(( fails + 1 )); fi
      [ $fails -ge 3 ] && { stop_runs "vLLM failed its health check 3 times in a row"; return; }
    fi
    now=$(date +%s)
    # no runs active (between commands): idle time doesn't count as a stall
    [ -n "$(run_pids)" ] || { last_change=$now; continue; }
    size=$(du -sb "$WORK/storage/logs" 2>/dev/null | cut -f1)
    if [ "${size:-0}" != "$last_size" ]; then
      last_size=${size:-0}; last_change=$now
    elif [ $(( now - last_change )) -ge $(( STALL_MINUTES * 60 )) ]; then
      stop_runs "no new Arbor request or response for $STALL_MINUTES min"; return
    fi
  done
}
watchdog &
WATCH_PID=$!

progress() {
  local d r
  echo "time,run_dir,metric_calls,reflections" > "logs/progress-$JOB.csv"
  while sleep 60; do
    for d in $(find $PROGRESS_DIRS -maxdepth 6 -type d -name metric_logs -path '*/experiment_runs/seed_*' 2>/dev/null); do
      r=${d%/metric_logs}
      echo "$(date +%s),$r,$(cat "$d/train.jsonl" "$d/val.jsonl" 2>/dev/null | wc -l),$(cat "$r/instruction_proposer_inpouts.jsonl" 2>/dev/null | wc -l)"
    done >> "logs/progress-$JOB.csv"
  done
}
if [ -n "${PROGRESS_DIRS:-}" ]; then progress & PROG_PID=$!; fi

STATUS=0
BG_PIDS=()
run_cmd() {  # $1 = command, $2 = fg or bg
  local t0 rc
  t0=$(date +%s)
  eval "$1" < /dev/null
  rc=$?
  echo "=== [$2] took $(( $(date +%s) - t0 )) s, exit $rc: ${1:0:120}"
  return $rc
}
wait_bg() {
  local p
  for p in "${BG_PIDS[@]}"; do wait "$p" || STATUS=1; done
  BG_PIDS=()
}
BG_RE='^(.*[^[:space:]&])[[:space:]]*&[[:space:]]*$'
while IFS= read -r cmd || [ -n "$cmd" ]; do
  [[ $cmd =~ ^[[:space:]]*(#|$) ]] && continue
  [ -e "$STOPFILE" ] && { echo "=== stopped by the watchdog, skipping the remaining commands"; STATUS=1; break; }
  if [[ $cmd =~ ^[[:space:]]*wait[[:space:]]*$ ]]; then
    echo "=== $(date) waiting for ${#BG_PIDS[@]} background run(s)"
    wait_bg
    continue
  fi
  cmd=${cmd//\{portnum\}/$PORT}
  if [[ $cmd =~ $BG_RE ]]; then
    cmd=${BASH_REMATCH[1]}
    echo "=== $(date) starting in the background: ${cmd:0:160}"
    run_cmd "$cmd" bg &
    BG_PIDS+=($!)
  else
    echo "=== $(date) running: ${cmd:0:160}"
    run_cmd "$cmd" fg || STATUS=1
  fi
done < "$CMDS"
wait_bg
[ -e "$STOPFILE" ] && STATUS=1

echo "end $(date), total $(( $(date +%s) - START )) s, server start-up $UP s, exit $STATUS"
exit $STATUS
