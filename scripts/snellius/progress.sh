#!/bin/bash
# progress of the runs that have not finished yet, with a rough end-time guess, so the limits and the
# SBU budget of the next jobs can be checked while the runs are still going.
#   bash progress.sh [experiment dir, default ~/repro_runs]
# MIPROv2: last "Trial x / y" in run_log_stderr.txt, minutes per trial over the last 10 trials.
# GEPA: metric calls so far = lines in metric_logs/train.jsonl + val.jsonl (a few extra lines come from
# re-sent stragglers), against the fixed budget of the benchmark; the rate is the average since the start.
# Both guesses leave out the final test evaluation (minutes on an H100, up to ~30 min on an A100).
R=${1:-$HOME/repro_runs}/experiment_runs
budget_of() { case "$1" in HotpotQABench) echo 6871;; hoverBench) echo 7051;; IFBench) echo 3593;; AIMEBench|LiveBenchMathBench) echo 1839;; *) echo 0;; esac; }
now=$(date +%s)
echo "| run | seed | started | last log write | progress | guess for the end |"
echo "|---|---|---|---|---|---|"
for d in $R/seed_*/*; do
  [ -f "$d/config.json" ] || continue
  [ -f "$d/evaluation_results/evaluation_result.txt" ] && continue
  n=$(basename $d); s=$(basename $(dirname $d)); b=${n%%_*}
  t0=$(stat -c %Y "$d/config.json")
  last=$(stat -c %Y "$d/run_log_stderr.txt" 2>/dev/null || echo $t0)
  [ -f "$d/run_log.txt" ] && [ $(stat -c %Y "$d/run_log.txt") -gt $last ] && last=$(stat -c %Y "$d/run_log.txt")
  for f in $d/metric_logs/*.jsonl; do [ -f "$f" ] && [ $(stat -c %Y "$f") -gt $last ] && last=$(stat -c %Y "$f"); done
  el=$(( now - t0 ))
  case "$n" in
    *_MIPROv2-Heavy_*)
      tr '\r' '\n' < $d/run_log_stderr.txt | grep -E "Trial [0-9]+ / [0-9]+" > /tmp/trials_$$.txt
      if [ -s /tmp/trials_$$.txt ]; then
        cur=$(tail -1 /tmp/trials_$$.txt | grep -oE "Trial [0-9]+ / [0-9]+" | grep -oE "[0-9]+" | head -1)
        tot=$(tail -1 /tmp/trials_$$.txt | grep -oE "Trial [0-9]+ / [0-9]+" | grep -oE "[0-9]+" | tail -1)
        # minutes per trial over the last 10 trials, from the log timestamps
        per=$(tail -11 /tmp/trials_$$.txt | awk '{ split($1, d, "/"); split($2, t, ":"); s = mktime(d[1] " " d[2] " " d[3] " " t[1] " " t[2] " " t[3]); if (NR == 1) a = s; b = s; n = NR } END { if (n > 1) printf "%.1f", (b - a) / 60 / (n - 1); else print "?" }')
        if [ "$per" != "?" ]; then
          left=$(awk -v p=$per -v c=$cur -v t=$tot 'BEGIN { printf "%.0f", p * (t - c + 1) * 60 }')
          prog="trial $cur / $tot, $per min per trial"; guess="$(date -d @$(( now + left )) '+%m-%d %H:%M') (+$(( left / 3600 )) h $(( left % 3600 / 60 )) min)"
        else prog="trial $cur / $tot"; guess="?"; fi
      else
        prog="$(tr '\r' '\n' < $d/run_log_stderr.txt | grep -oE 'STEP [0-9]: [A-Z ]+' | tail -1 | sed 's/ *$//')"; guess="? (before the trials)"
      fi
      rm -f /tmp/trials_$$.txt ;;
    *_GEPA_*|*_GEPA-MERGE_*)
      B=$(budget_of $b)
      c=$(( $(cat $d/metric_logs/train.jsonl 2>/dev/null | wc -l) + $(cat $d/metric_logs/val.jsonl 2>/dev/null | wc -l) ))
      it=$(grep -oE "^Iteration [0-9]+:" $d/run_log.txt 2>/dev/null | tail -1 | grep -oE "[0-9]+")
      prog="$c / $B metric calls ($(( 100 * c / B )) %), iteration ${it:-0}, reflections $(cat $d/instruction_proposer_inpouts.jsonl 2>/dev/null | wc -l)"
      if [ $c -gt 0 ] && [ $el -gt 600 ]; then
        left=$(awk -v e=$el -v c=$c -v B=$B 'BEGIN { r = (B - c) * e / c; if (r < 0) r = 0; printf "%.0f", r }')
        guess="$(date -d @$(( now + left )) '+%m-%d %H:%M') (+$(( left / 3600 )) h $(( left % 3600 / 60 )) min)"
      else guess="?"; fi ;;
    *) prog="(Baseline)"; guess="-" ;;
  esac
  echo "| $n | ${s#seed_} | $(date -d @$t0 '+%m-%d %H:%M') ($(( el / 3600 )) h $(( el % 3600 / 60 )) min ago) | $(( (now - last) / 60 )) min ago | $prog | $guess |"
done
