#!/usr/bin/env bash
#SBATCH --job-name=gepa-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --output=logs/%x-%j.out
# GRPO version of qwen_job.sh: starts Arbor with the authors' GRPO layout and runs the run_experiments
# commands in the file given as first argument ({portnum} is filled in). Submit from the repo root:
#   sbatch -p gpu_h100 -t 01:30:00 scripts/snellius/grpo_job.sh my_commands.txt $HOME/arbor-envs/vllm091
# Build the command line with scripts/snellius/make_grpo_command.sh (make_command.sh leaves out the
# train_kwargs that GRPO needs).
#
# Differences from qwen_job.sh, all needed for GRPO:
# - 3 GPUs and gepa_artifact/utils/arbor/arbor_train.yaml (vLLM on GPUs 1 and 2 with tensor parallel 2,
#   LoRA training on GPU 0), the layout in the authors' GRPO logs. ARBOR_CONFIG=<yaml> overrides it.
# - No warm-up request. The first chat request would make Arbor start vLLM with its default 32,768-token
#   context, and GRPO's initialize then keeps that server (inference_manager.py:53-55) instead of
#   starting its own with max_context_length 8,192, as in the authors' runs. The job only waits for
#   Arbor's HTTP server; vLLM starts when the GRPO run initializes.
# - The watchdog reads vLLM's port from the Arbor log every minute: vLLM starts only at GRPO's
#   initialize, and at the end Arbor stops it and starts a new server for the test evaluation on a new
#   port. HEALTH_FAILS (default 10) consecutive failed checks stop the runs, so a restart fits.
# - A training error or an out-of-memory in the Arbor log is reported in the job log (not stopped:
#   the authors' HoVer run logs one at its very end, after the final checkpoint).
#
# As in qwen_job.sh: Snellius submits with --export=NONE, so put per-run variables (GEPA_EXPERIMENT_DIR,
# GEPA_GRPO_TRAIN_STEPS) at the start of the command line, and job-wide ones in sbatch --export=...
# The second argument is the Arbor environment (default gepa_artifact/utils/arbor/.venv).
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
HEALTH_FAILS=${HEALTH_FAILS:-10}
ARBOR_CONFIG=${ARBOR_CONFIG:-$REPO/gepa_artifact/utils/arbor/arbor_train.yaml}
nvidia-smi -L
START=$(date +%s)
DIRTY=$(git diff --quiet HEAD 2>/dev/null || echo " + uncommitted changes")
echo "start $(date) on $(hostname), commit $(git rev-parse --short HEAD)$DIRTY, commands: $CMDS"
ARBOR_VENV=${2:-${ARBOR_VENV:-$REPO/gepa_artifact/utils/arbor/.venv}}
[ -x "$ARBOR_VENV/bin/python" ] || { echo "no Arbor environment at $ARBOR_VENV"; exit 1; }
echo "arbor venv: $ARBOR_VENV (vllm $("$ARBOR_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo '?'), trl $("$ARBOR_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("trl"))' 2>/dev/null || echo '?'))"
[ -f "$ARBOR_CONFIG" ] || { echo "no Arbor config at $ARBOR_CONFIG"; exit 1; }
echo "arbor config: $ARBOR_CONFIG ($(tr '\n' ' ' < "$ARBOR_CONFIG"))"
# Same timeout edits as qwen_job.sh requires (F56, F67, F70, F72, F85).
V=$REPO/gepa_artifact/utils/arbor/arbor/server/services/inference
if ! grep -q "timeout=21600)" "$V/vllm_client.py" || ! grep -q "default=864000," "$V/vllm_serve.py"    || grep -q "timeout=60.0)" "$V/vllm_serve.py" || ! grep -q '"timeout": 21600' scripts/run_experiments.py; then
  echo "timeout edits missing (F56, F67, F70, F72): run scripts/snellius/apply_arbor_timeouts.sh and check scripts/run_experiments.py"; exit 1
fi
echo "job settings: STALL_MINUTES=$STALL_MINUTES HEALTH_FAILS=$HEALTH_FAILS"

# arbor writes request/response logs, checkpoints and models to ./storage, keep them out of the repo
WORK=/scratch-shared/$USER/arbor_$JOB
mkdir -p "$WORK"
STOPFILE=$WORK/stopped_by_watchdog
export VLLM_CACHE_ROOT=$WORK/vllm-cache
export GEPA_RETRIEVER_CACHE=${GEPA_RETRIEVER_CACHE:-$WORK/retriever_cache}
echo "retriever cache: $GEPA_RETRIEVER_CACHE"
PORT=$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))
ALOG=logs/arbor-$JOB.log
(source "$ARBOR_VENV/bin/activate" && export PYTHONPATH="$REPO/gepa_artifact/utils/arbor" && cd "$WORK" && exec python -m arbor.cli serve --arbor-config "$ARBOR_CONFIG" --port $PORT > "$REPO/$ALOG" 2>&1) &
ARBOR_PID=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 > "logs/gpu-$JOB.csv" &
SMI_PID=$!
WATCH_PID=

copy_arbor_logs() {
  local mb
  mb=$(du -sm "$WORK/storage/logs" 2>/dev/null | cut -f1)
  if [ -n "$mb" ] && [ "$mb" -le "${COPY_ARBOR_LOGS_MB:-1000}" ]; then
    cp -r "$WORK/storage/logs" "logs/arbor_requests_$JOB" && echo "arbor request logs: logs/arbor_requests_$JOB ($mb MB)"
  else
    echo "arbor request logs: $WORK/storage/logs (${mb:-?} MB, left on scratch)"
  fi
  echo "arbor storage (checkpoints, models): $WORK/storage ($(du -sh "$WORK/storage" 2>/dev/null | cut -f1))"
}
cleanup() {
  trap - EXIT
  kill ${SMI_PID:-} ${WATCH_PID:-} 2>/dev/null
  copy_arbor_logs
  pkill -P $ARBOR_PID 2>/dev/null; kill $ARBOR_PID 2>/dev/null
}
trap cleanup EXIT

# wait for Arbor's HTTP server only (no chat request, see the header)
READY=""
for i in $(seq 1 60); do
  kill -0 $ARBOR_PID 2>/dev/null || break
  if curl -sf -m 10 -o /dev/null "http://localhost:$PORT/openapi.json"; then READY=1; break; fi
  sleep 5
done
UP=$(( $(date +%s) - START ))
echo "arbor ready: ${READY:-no} after $UP s (vLLM starts when the GRPO run initializes)"
[ -n "$READY" ] || { echo "arbor did not start"; tail -50 "$ALOG"; exit 1; }

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
  local fails=0 size last_size=-1 last_change now vport reported=""
  last_change=$(date +%s)
  while sleep 60; do
    kill -0 $ARBOR_PID 2>/dev/null || { stop_runs "Arbor exited"; return; }
    # the vLLM server started last (GRPO's own, later the test-evaluation server)
    vport=$(grep -oE 'vllm_serve --model [^ ]+ --port [0-9]+' "$ALOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    if [ -n "$vport" ]; then
      if curl -sf -m 30 "http://localhost:$vport/health/" > /dev/null; then fails=0; else fails=$(( fails + 1 )); fi
      [ $fails -ge $HEALTH_FAILS ] && { stop_runs "vLLM (port $vport) failed its health check $HEALTH_FAILS times in a row"; return; }
    fi
    if [ -z "$reported" ] && grep -qE "Training error|CUDA out of memory|torch.OutOfMemoryError" "$ALOG" 2>/dev/null; then
      echo "=== watchdog $(date): the Arbor log reports a training error or an out-of-memory (not stopped, check $ALOG):"
      grep -m3 -E "Training error|CUDA out of memory|torch.OutOfMemoryError" "$ALOG" | cut -c1-300
      reported=1
    fi
    now=$(date +%s)
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

STATUS=0
while IFS= read -r cmd || [ -n "$cmd" ]; do
  [[ $cmd =~ ^[[:space:]]*(#|$) ]] && continue
  [ -e "$STOPFILE" ] && { echo "=== stopped by the watchdog, skipping the remaining commands"; STATUS=1; break; }
  cmd=${cmd//\{portnum\}/$PORT}
  echo "=== $(date) running: ${cmd:0:200}"
  t0=$(date +%s)
  eval "$cmd" < /dev/null || STATUS=1
  echo "=== took $(( $(date +%s) - t0 )) s, status $STATUS"
done < "$CMDS"
[ -e "$STOPFILE" ] && STATUS=1
# trainer throughput (F101): one 'Time since last step' line per optimizer step
NSTEPS=$(grep -c "Time since last step" "$ALOG" 2>/dev/null)
MEAN=$(grep -oE "Time since last step: [0-9.]+" "$ALOG" 2>/dev/null | awk '{s+=$5} END {if (NR) printf "%.1f", s/NR}')
echo "trainer: ${NSTEPS:-0} optimizer steps, mean ${MEAN:-?} s per step (20 sequences each)"
echo "end $(date), total $(( $(date +%s) - START )) s, exit $STATUS"
exit $STATUS
