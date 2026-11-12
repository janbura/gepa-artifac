#!/usr/bin/env bash
# Post-run check of one pupa_job.sh job (plan section 2.5), made from check_job.sh (Table 1): writes
# ~/pupa_checks/<job>.md and one line in ~/pupa_checks/SUMMARY.md with OK or CHECK, and exits 1 on CHECK,
# so a job submitted with --dependency=afterok:<check job> only starts after a clean run.
# Changes to check_job.sh: the PUPA checkout (~/gepa-pupa), test size 221, no seed-0 gate (there is no
# comparable paper value, plan 2.5), the smoke test line, and the external model server: every chat
# request must be answered with HTTP 200 and the external watchdog must not have stopped the runs.
#   usage: check_pupa.sh <job id> [<job id> ...]
set -u
BAD=0
OUT=${CHECK_OUT:-$HOME/pupa_checks}; mkdir -p "$OUT"
for J in "$@"; do
  F=$OUT/$J.md
  JL=$(ls $HOME/gepa-pupa/logs/*-$J.out 2>/dev/null | grep -v -E "/arbor-|/external-" | head -1)
  if [ -z "$JL" ]; then  # cancelled before it started, e.g. because a dependency failed
    echo "| $J | $(date '+%F %T') | CHECK | never ran: $(sacct -j $J -X -n -o JobName,State | tr -s ' ') |  |" >> $OUT/SUMMARY.md
    BAD=1; continue
  fi
  REPO=$(dirname "$(dirname "$JL")")
  AL=$REPO/logs/arbor-$J.log
  EL=$REPO/logs/external-$J.log
  {
  echo "# Check of PUPA job $J ($(date '+%F %T'))"
  echo
  sacct -j $J -X -n -P -o JobName,State,Partition,NodeList,Start,End,Elapsed,ElapsedRaw,AllocTRES | awk -F'|' '{b=0; n=split($9,t,","); for(i=1;i<=n;i++) if (t[i] ~ /^billing=/) {sub("billing=","",t[i]); b=t[i]}; printf "- job: %s, state %s, partition %s, node %s, %s to %s, elapsed %s, SBU %.0f\n", $1,$2,$3,$4,$5,$6,$7, b*$8/3600}'
  echo "- checkout: $REPO, $(grep -m1 '^start' "$JL" | sed 's/^start //')"
  echo "- $(grep -m1 '^arbor venv:' "$JL")"
  echo "- $(grep -m1 '^external model:' "$JL")"
  echo "- $(grep -m1 '^external server ready' "$JL"); $(grep -m1 'server ready: ' "$JL" | grep -v external)"
  echo "- $(grep -m1 '^runs use:' "$JL")"
  echo "- $(grep -m1 '^timeouts:' "$JL")"
  echo "- watchdog lines: $(grep -c '=== watchdog' "$JL"), external watchdog lines: $(grep -c '=== external watchdog' "$JL"), skipped runs: $(grep -c 'already exists. Skipping' "$JL")"
  echo "- smoke test: $(grep -m1 -o 'SMOKE TEST [A-Z]*' "$JL" || echo 'not in this job')"
  grep -E '^(external calls|judge calls|program errors|examples [0-9])' "$JL" | sed 's/^/  - /'
  echo
  echo "## Qwen server (Arbor) log"
  RQ=$(ls -d $REPO/logs/arbor_requests_$J/* /scratch-shared/$USER/arbor_$J/storage/logs/* 2>/dev/null | head -1)
  NQ=$(cat "$RQ"/inference_requests.jsonl 2>/dev/null | wc -l); NR=$(cat "$RQ"/inference_responses.jsonl 2>/dev/null | wc -l)
  CTXL=$(grep -c 'Error in chat: The decoder prompt' "$AL"); F3=$(grep -c 'Request failed after' "$AL")
  CTX=$((CTXL / 3)); [ "$F3" -lt "$CTX" ] && CTX=$F3
  UNEX=$((NQ-NR-CTX)); [ $UNEX -lt 0 ] && UNEX=0
  echo "- requests $NQ, responses $NR, unanswered $((NQ-NR)), of which rejected as longer than max_model_len $CTX (worker errors $CTXL, requests given up $F3), unexplained $UNEX (must be 0 unless the job hit its time limit)"
  echo "- worker timeouts (void the run, F72): $(grep -c 'Timeout for chat request' "$AL")"
  echo "- forwarder timeouts: $(grep -c '^httpx.ReadTimeout' "$AL"), HTTP 500: $(grep -c '500 Internal Server Error' "$AL"), HTTP 504: $(grep -c '504 Gateway' "$AL")"
  echo "- chunk-call timeouts (F70): $(grep -cE 'chunk chat after|Timeout waiting for worker' "$AL"), result mismatches (F68/F72): $(grep -c 'LLM result' "$AL"), connection errors: $(grep -cE 'ConnectError|Connection refused' "$AL")"
  echo
  echo "## External model server (gpt-oss-120b) log"
  [ -f "$EL" ] || echo "- NO-EXTERNAL-LOG: $EL is missing (the job ended before the external server started)"
  NE=$(grep -c 'POST /v1/chat/completions' "$EL" 2>/dev/null); NE=${NE:-0}
  NB=$(grep 'POST /v1/chat/completions' "$EL" 2>/dev/null | grep -vc '" 200')
  echo "- chat requests $NE, not answered with HTTP 200: $NB"
  [ "$NE" -gt 0 ] || echo "- NO-EXTERNAL-REQUESTS: the runs never reached the external model"
  echo "- error lines (reported only, the shutdown at the end can write some): $(grep -cE ' ERROR |Traceback' "$EL" 2>/dev/null)"
  python3 - "$EL" <<'PY'
import os, re, sys
# vLLM logs its throughput about every 10 s while it works
gen = []
if os.path.exists(sys.argv[1]):
    gen = [float(m.group(1)) for l in open(sys.argv[1], errors="replace") for m in [re.search(r"Avg generation throughput: ([0-9.]+) tokens/s", l)] if m]
if gen:
    print(f"- about {sum(gen) * 10 / 1e6:.2f} M tokens generated; busy log lines {sum(g > 0 for g in gen)}, mean {sum(gen) / len(gen):.0f} tok/s over all lines")
PY
  echo
  echo "## Runs"
  echo "| run | seed | score | scored / test size | failed | errors in stderr (of which format) | result file |"
  echo "|---|---|---|---|---|---|---|"
  CMDS=$(grep -m1 '^start' "$JL" | sed 's/.*commands: //')
  [ -f "$CMDS" ] || CMDS=$REPO/$CMDS
  grep 'run_experiments' "$CMDS" 2>/dev/null | while IFS= read -r line; do
    D=$(echo "$line" | grep -o 'GEPA_EXPERIMENT_DIR=[^ ]*' | cut -d= -f2 | sed "s#\$HOME#$HOME#")
    [ -n "$D" ] || D=$REPO/experiment_runs_data
    B=$(echo "$line" | grep -o -- '--benchmark_name [^ ]*' | cut -d' ' -f2)
    P=$(echo "$line" | grep -o -- '--prog_name [^ ]*' | cut -d' ' -f2)
    O=$(echo "$line" | grep -o -- '--optim_name [^ ]*' | cut -d' ' -f2)
    S=$(echo "$line" | grep -o -- '--seed [0-9]*' | tail -1 | cut -d' ' -f2)
    R=$D/experiment_runs/seed_$S/${B}_${P}_${O}_qwen3-8b
    N=221
    if [ -f $R/evaluation_results/evaluation_result.txt ]; then
      SC=$(sed -n 2p $R/evaluation_results/evaluation_result.txt | cut -d, -f1)
      DEN=$(tr '\r' '\n' < $R/run_log.txt | grep -o 'Average Metric: [0-9.]* / [0-9]*' | tail -1 | awk '{print $NF}')
      E=$(tr '\r' '\n' < $R/run_log_stderr.txt | grep -c 'ERROR dspy.utils.parallelizer')
      EF=$(tr '\r' '\n' < $R/run_log_stderr.txt | grep 'ERROR dspy.utils.parallelizer' | grep -c 'Failed to parse\|Expected dict_keys')
      EI=$(tr '\r' '\n' < $R/run_log_stderr.txt | grep 'ERROR dspy.utils.parallelizer' | grep -v 'Failed to parse\|Expected dict_keys' | grep -c 'InternalServerError')
      FLAG=""; [ "$E" -gt $((EF + EI)) ] && FLAG=" OTHER-ERRORS"
      echo "| ${B}_${O} | $S | $SC | ${DEN:-?} / $N | $(( N - ${DEN:-0} )) | $E ($EF)$FLAG | $(stat -c %y $R/evaluation_results/evaluation_result.txt | cut -c1-16) |"
      [ -n "$FLAG" ] && echo "  - first error that is not a format failure: $(tr '\r' '\n' < $R/run_log_stderr.txt | grep 'ERROR dspy.utils.parallelizer' | grep -v 'Failed to parse\|Expected dict_keys' | head -1 | cut -c1-300)"
      [ "$EI" -gt 0 ] && echo "  - client InternalServerErrors: $EI (allowed only up to the requests rejected for length)"
      echo "  - paper (authors' seed 0, GPT-4.1-mini as external model and judge, not comparable, plan 2.5): $(case $O in Baseline) echo 80.82;; MIPROv2-Heavy) echo 81.55;; GEPA) echo 91.85;; GEPA-MERGE) echo 86.26;; esac)"
      case "$O" in
        MIPROv2-Heavy) echo "  - MIPROv2: $(grep -h 'Default program score' $R/run_log_stderr.txt | tail -1 | sed 's/.*mipro_optimizer_v2: //') / $(grep -h 'Returning best identified program' $R/run_log_stderr.txt | tail -1 | sed 's/.*mipro_optimizer_v2: //') / saved programs $(ls $R/optimizer_logs/evaluated_programs 2>/dev/null | wc -l)" ;;
        GEPA|GEPA-MERGE) [ "$(cat $R/instruction_proposer_inpouts.jsonl 2>/dev/null | wc -l)" -gt 0 ] || echo "  - NO-REFLECTIONS: GEPA made no reflection, so it optimized nothing"
                         EP=$(grep -c 'Exception during instruction proposal' $R/run_log.txt); NP=$(grep -cE '^Iteration [0-9]+: Proposed new instruction' $R/run_log.txt)
                         echo "  - GEPA proposals: $NP, failed proposals: $EP"
                         [ "$EP" -ge 10 ] && [ "$EP" -gt "$NP" ] && echo "  - REFLECTION-FAILS: most proposals failed"
                         echo "  - GEPA: iterations $(grep -o 'Iteration [0-9]*:' $R/run_log.txt | tail -1 | grep -o '[0-9]*'), reflections $(cat $R/instruction_proposer_inpouts.jsonl 2>/dev/null | wc -l), metric log lines train $(cat $R/metric_logs/train.jsonl 2>/dev/null | wc -l) / val $(cat $R/metric_logs/val.jsonl 2>/dev/null | wc -l), merges $(grep -c 'Merged programs' $R/run_log.txt), candidates $(ls $R/prog_candidates 2>/dev/null | wc -l)" ;;
      esac
    else
      echo "| ${B}_${O} | $S | NO RESULT | – | – | – | – |"
    fi
  done
  } > "$F" 2>&1
  V=OK
  grep -qE "worker timeouts \(void the run, F72\): [1-9]|forwarder timeouts: [1-9]|chunk-call timeouts \(F70\): [1-9]|connection errors: [1-9]|NO RESULT|watchdog lines: [1-9]|external watchdog lines: [1-9]" "$F" && V=CHECK
  grep -qE "unexplained [1-9]|not answered with HTTP 200: [1-9]|NO-EXTERNAL-REQUESTS|NO-EXTERNAL-LOG|SMOKE TEST FAILED" "$F" && V=CHECK
  grep -qE "skipped runs: [1-9]" "$F" && V=CHECK
  grep -q "OTHER-ERRORS" "$F" && V=CHECK
  grep -q "^- job: .*, state COMPLETED," "$F" || V=CHECK
  ISE=$(grep -o 'client InternalServerErrors: [0-9]*' "$F" | awk '{s += $3} END {print s + 0}')
  RJ=$(grep -o 'rejected as longer than max_model_len [0-9]*' "$F" | awk '{print $NF + 0}')
  [ "$ISE" -gt "${RJ:-0}" ] && V=CHECK
  grep -q "REFLECTION-FAILS" "$F" && V=CHECK
  grep -q "NO-REFLECTIONS" "$F" && V=CHECK
  [ "$V" = CHECK ] && BAD=1
  echo "| $J | $(date '+%F %T') | $V | $(grep -m1 '^- job:' "$F" | cut -c8-) | $(grep '^| .*_' "$F" | awk -F'|' '{printf "%s s%s=%s; ", $2, $3, $4}') |" >> $OUT/SUMMARY.md
done
exit $BAD
