#!/bin/bash
# 30-minute status of the Table 1 runs on Snellius: jobs, progress, GEPA monitors, server health and client
# errors of the runs that are still going, and at the END the new check lines (so a cut output keeps them). Read-only.
#   bash ~/repro_checks/status.sh ["2026-11-09 12:00"]   (check lines newer than this; default: last 2 hours)
SINCE=${1:-$(date -d '-2 hours' '+%F %T')}
date '+%F %T'
echo "--- run jobs (checks and monitors left out)"
squeue -u $USER -o "%.10i %.14j %.9P %.9T %.10M %.11l %.18R" | grep -vE " check-| mon-"
for X in ~/repro_runs ~/repro_h100; do
  [ -d $X ] || continue
  echo "--- progress of unfinished runs in $X"
  bash ~/repro_checks/progress.sh $X | grep -v '^|---' | tail -n +2 | cut -c1-230
done
echo "--- GEPA monitors (newest first)"
ls -t ~/repro_checks/mon-*.log 2>/dev/null | head -12 | while read f; do echo "$(basename $f): $(tail -1 $f | cut -c1-140)"; done
echo "--- server health of running jobs (requests - responses = requests still in flight)"
for J in $(squeue -h -u $USER -t R -o "%i %j" | grep -vE " check-| mon-" | awk '{print $1}'); do
  A=$(ls ~/gepa-repro/logs/arbor-$J.log 2>/dev/null); [ -n "$A" ] || continue
  L=$(ls -d /scratch-shared/$USER/arbor_$J/storage/logs/* 2>/dev/null | head -1)
  echo "job $J $(squeue -h -j $J -o %j): requests $(cat $L/inference_requests.jsonl 2>/dev/null | wc -l), responses $(cat $L/inference_responses.jsonl 2>/dev/null | wc -l), worker timeouts $(grep -c 'Timeout for chat request' $A), forwarder timeouts $(grep -c '^httpx.ReadTimeout' $A), HTTP 500 $(grep -c '500 Internal Server Error' $A), over-long $(grep -c 'Error in chat: The decoder prompt' $A), given up $(grep -c 'Request failed after' $A), watchdog $(grep -c '=== watchdog' $(ls ~/gepa-repro/logs/*-$J.out | grep -v arbor | head -1))"
done
echo "--- client errors in unfinished runs (format failures are normal; anything else needs a look)"
for X in ~/repro_runs ~/repro_h100; do for d in $X/experiment_runs/seed_*/*; do
  [ -f $d/config.json ] || continue; [ -f $d/evaluation_results/evaluation_result.txt ] && continue
  e=$(tr '\r' '\n' < $d/run_log_stderr.txt 2>/dev/null | grep -c 'ERROR dspy.utils.parallelizer')
  ef=$(tr '\r' '\n' < $d/run_log_stderr.txt 2>/dev/null | grep 'ERROR dspy.utils.parallelizer' | grep -c 'Failed to parse\|Expected dict_keys')
  echo "$(basename $X)/$(basename $(dirname $d))/$(basename $d): errors $e, not format $((e - ef))"
done; done
echo "--- check jobs still waiting or running: $(squeue -h -u $USER -o '%j' | grep -c '^check-') (a run that finished shows up in no list until its check has written its line)"
echo "--- check lines since $SINCE (OK / OK-FLAG / CHECK)"
awk -F'|' -v t=" $SINCE" 'NR > 2 && $3 > t' ~/repro_checks/SUMMARY.md | cut -c1-330
