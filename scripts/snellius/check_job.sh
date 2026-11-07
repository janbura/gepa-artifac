#!/usr/bin/env bash
# Post-run check of one qwen_job.sh job (section 2.4 of the plan), written to ~/repro_checks/<job>.md
# and one line in ~/repro_checks/SUMMARY.md. Runs on its own after the job (sbatch --dependency=afterany),
# so the result is there even if nobody is logged in. Exits 1 if any checked job gets the verdict CHECK,
# so a job submitted with --dependency=afterok:<check job> only starts after a clean run (the gate).
#   usage: check_job.sh <job id> [<job id> ...]
set -u
BAD=0
OUT=$HOME/repro_checks; mkdir -p "$OUT"
size_of() { case "$1" in HotpotQABench|hoverBench) echo 300;; IFBench) echo 294;; AIMEBench) echo 150;; LiveBenchMathBench) echo 126;; *) echo 0;; esac; }

for J in "$@"; do
  F=$OUT/$J.md
  # which checkout ran it: the job log is in <repo>/logs/<name>-<id>.out
  JL=$(ls $HOME/gepa-repro/logs/*-$J.out $HOME/gepa-artifac/logs/*-$J.out 2>/dev/null | grep -v "/arbor-" | head -1)
  [ -n "$JL" ] || { echo "| $J | no job log found |" >> $OUT/SUMMARY.md; continue; }
  REPO=$(dirname "$(dirname "$JL")")
  AL=$REPO/logs/arbor-$J.log
  {
  echo "# Check of job $J ($(date '+%F %T'))"
  echo
  sacct -j $J -X -n -P -o JobName,State,Partition,NodeList,Start,End,Elapsed,ElapsedRaw,AllocTRES | awk -F'|' '{b=0; n=split($9,t,","); for(i=1;i<=n;i++) if (t[i] ~ /^billing=/) {sub("billing=","",t[i]); b=t[i]}; printf "- job: %s, state %s, partition %s, node %s, %s to %s, elapsed %s, SBU %.0f\n", $1,$2,$3,$4,$5,$6,$7, b*$8/3600}'
  echo "- checkout: $REPO, $(grep -m1 '^start' "$JL" | sed 's/^start //')"
  echo "- $(grep -m1 '^arbor venv:' "$JL" || grep -om1 'LLM engine (v[0-9.]*' "$AL")"
  echo "- $(grep -m1 '^timeouts:' "$JL" || echo 'timeouts: (old checkout: client 7200, forwarder 7200, worker 21600, chunk 600)')"
  echo "- $(grep -m1 'server ready' "$JL")"
  echo "- watchdog lines: $(grep -c '=== watchdog' "$JL"), skipped runs: $(grep -c 'already exists. Skipping' "$JL"), command exits: $(grep -o 'exit so far [0-9]*' "$JL" | tr '\n' ' ')"
  echo
  echo "## Server (Arbor) log"
  RQ=$(ls -d $REPO/logs/arbor_requests_$J/* /scratch-shared/$USER/arbor_$J/storage/logs/* 2>/dev/null | head -1)
  NQ=$(cat "$RQ"/inference_requests.jsonl 2>/dev/null | wc -l); NR=$(cat "$RQ"/inference_responses.jsonl 2>/dev/null | wc -l)
  echo "- requests $NQ, responses $NR, unanswered $((NQ-NR)) (must be 0 unless the job hit its time limit)"
  echo "- worker timeouts (void the run, F72): $(grep -c 'Timeout for chat request' "$AL")"
  echo "- forwarder timeouts: $(grep -c '^httpx.ReadTimeout' "$AL"), HTTP 500: $(grep -c '500 Internal Server Error' "$AL"), HTTP 504: $(grep -c '504 Gateway' "$AL")"
  echo "- chunk-call timeouts (F70): $(grep -cE 'chunk chat after|Timeout waiting for worker' "$AL"), result mismatches (F68/F72): $(grep -c 'LLM result' "$AL"), connection errors: $(grep -cE 'ConnectError|Connection refused' "$AL")"
  python3 - "$AL" <<'PY'
import re, sys, datetime, collections
T = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),")
ts = lambda l: datetime.datetime.strptime(T.search(l).group(1), "%Y-%m-%d %H:%M:%S").timestamp()
created, done, mx, kv = {}, {}, 0, ""
for l in open(sys.argv[1], errors="replace"):
    if "GPU KV cache size" in l and not kv:
        m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", l); kv = m.group(1) if m else ""
    for m in re.finditer(r"Processed prompts: 100%\|[^|]*\| (\d+)/(\d+) \[(\d+):(\d+)<", l):
        mx = max(mx, int(m.group(3)) * 60 + int(m.group(4)))
    if "LIFECYCLE" not in l: continue
    m = re.search(r"Created chat request (\S+?): max_tokens=(\d+)", l)
    if m and int(m.group(2)) > 64: created[m.group(1)] = ts(l); continue
    m = re.search(r"Request (\S+) is complete and will be finalized\. Generated (\d+) tokens, finish_reason=(\w+)", l)
    if m and m.group(1) in created: done[m.group(1)] = (ts(l), int(m.group(2)), m.group(3))
if done:
    t0 = min(created.values()); t1 = max(v[0] for v in done.values()); tok = sum(v[1] for v in done.values())
    lat = sorted(v[0] - created[k] for k, v in done.items())
    print(f"- generated {tok/1e6:.2f} M tokens in {(t1-t0)/3600:.2f} h = {tok/max(1,t1-t0):.0f} tok/s; KV cache {kv} tokens; longest chunk call {mx} s")
    print(f"- request latency median {lat[len(lat)//2]:.0f} s, p90 {lat[int(.9*len(lat))]:.0f} s, max {lat[-1]:.0f} s; finish reasons {dict(collections.Counter(v[2] for v in done.values()))}")
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
    N=$(size_of $B)
    if [ -f $R/evaluation_results/evaluation_result.txt ]; then
      SC=$(sed -n 2p $R/evaluation_results/evaluation_result.txt | cut -d, -f1)
      DEN=$(tr '\r' '\n' < $R/run_log.txt | grep -o 'Average Metric: [0-9.]* / [0-9]*' | tail -1 | awk '{print $NF}')
      E=$(tr '\r' '\n' < $R/run_log_stderr.txt | grep -c 'ERROR dspy.utils.parallelizer')
      EF=$(tr '\r' '\n' < $R/run_log_stderr.txt | grep 'ERROR dspy.utils.parallelizer' | grep -c 'Failed to parse\|Expected dict_keys')
      FLAG=""; [ "$E" -gt "$EF" ] && FLAG=" OTHER-ERRORS"
      echo "| ${B}_${O} | $S | $SC | ${DEN:-?} / $N | $(( N - ${DEN:-0} )) | $E ($EF)$FLAG | $(stat -c %y $R/evaluation_results/evaluation_result.txt | cut -c1-16) |"
      [ -n "$FLAG" ] && echo "  - first error that is not a format failure: $(tr '\r' '\n' < $R/run_log_stderr.txt | grep 'ERROR dspy.utils.parallelizer' | grep -v 'Failed to parse\|Expected dict_keys' | head -1 | cut -c1-300)"
      case "$O" in
        MIPROv2-Heavy) echo "  - MIPROv2: $(grep -h 'Default program score' $R/run_log_stderr.txt | tail -1 | sed 's/.*mipro_optimizer_v2: //') / $(grep -h 'Returning best identified program' $R/run_log_stderr.txt | tail -1 | sed 's/.*mipro_optimizer_v2: //') / saved programs $(ls $R/optimizer_logs/evaluated_programs 2>/dev/null | wc -l)" ;;
        GEPA|GEPA-MERGE) echo "  - GEPA: iterations $(grep -o 'Iteration [0-9]*:' $R/run_log.txt | tail -1 | grep -o '[0-9]*'), reflections $(cat $R/instruction_proposer_inpouts.jsonl 2>/dev/null | wc -l), metric log lines train $(cat $R/metric_logs/train.jsonl 2>/dev/null | wc -l) / val $(cat $R/metric_logs/val.jsonl 2>/dev/null | wc -l), merges $(grep -c 'Merged programs' $R/run_log.txt), candidates $(ls $R/prog_candidates 2>/dev/null | wc -l)" ;;
      esac
    else
      echo "| ${B}_${O} | $S | NO RESULT | – | – | – | – |"
    fi
  done
  } > "$F" 2>&1
  V=OK
  # void (CHECK): a lost or possibly mixed answer, or a crash. Internal HTTP 500/504 and F68 batch failures that
  # the forwarder retried successfully (every request answered, no client error) are only reported (F68 policy).
  grep -qE "worker timeouts \(void the run, F72\): [1-9]|forwarder timeouts: [1-9]|chunk-call timeouts \(F70\): [1-9]|connection errors: [1-9]|NO RESULT|watchdog lines: [1-9]" "$F" && V=CHECK
  grep -qE "unanswered [1-9]" "$F" && V=CHECK
  grep -q "OTHER-ERRORS" "$F" && V=CHECK
  [ "$V" = OK ] || BAD=1
  echo "| $J | $(date '+%F %T') | $V | $(grep -m1 '^- job:' "$F" | cut -c8-) | $(grep '^| .*_' "$F" | awk -F'|' '{printf "%s s%s=%s; ", $2, $3, $4}') |" >> $OUT/SUMMARY.md
done
exit $BAD
