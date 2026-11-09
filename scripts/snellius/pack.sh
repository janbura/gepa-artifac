#!/bin/bash
# Pack finished run folders for the repo's results/ folder and keep their server logs in the home archive.
#   bash ~/repro_checks/pack.sh <archive.tgz> <dest> <expdir>:<seed>:<run folder>:<job> [more entries]
#   dest = table1_h100 (runs that count in Table 1, all H100) or a100_hardware_check
#   example: bash ~/repro_checks/pack.sh pack_live.tgz table1_h100 $HOME/repro_runs:0:LiveBenchMathBench_CoT_GEPA_qwen3-8b:29003749
# Writes ~/<archive.tgz> with <dest>/seed_<s>/<run>/ and checks/<job>.md; download it and unpack into a fresh folder.
set -e
OUT=$1; DEST=$2; shift 2
P=$HOME/pack_tmp_$$; rm -rf $P; mkdir -p $P/$DEST $P/checks
for e in "$@"; do
  IFS=: read -r X S N J <<< "$e"
  src=$X/experiment_runs/seed_$S/$N; dst=$P/$DEST/seed_$S/$N
  [ -f $src/evaluation_results/evaluation_result.txt ] || { echo "no result in $src, stopping"; rm -rf $P; exit 1; }
  mkdir -p $dst/metric_logs
  cp $src/config.json $dst/; cp -r $src/evaluation_results $dst/
  for f in run_log.txt run_log_stderr.txt instruction_proposer_inpouts.jsonl; do [ -f $src/$f ] && gzip -c $src/$f > $dst/$f.gz; done
  for m in $src/metric_logs/*.jsonl; do gzip -c $m > $dst/metric_logs/$(basename $m).gz; done
  for t in optimizer_logs prog_candidates generated_best_outputs_valset; do [ -d $src/$t ] && tar czf $dst/$t.tgz -C $src $t; done
  [ -f $src/gepa_state.bin ] && cp $src/gepa_state.bin $dst/  # GEPA's binary state, kept as a file (never unpickle it)
  [ -f $HOME/repro_checks/$J.md ] && cp $HOME/repro_checks/$J.md $P/checks/
  A=$HOME/repro_archive/arbor_logs/$J; mkdir -p $A
  L=$(ls -d /scratch-shared/$USER/arbor_$J/storage/logs/* 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    [ -f $A/inference_requests.jsonl.gz ] || gzip -c $L/inference_requests.jsonl > $A/inference_requests.jsonl.gz
    [ -f $A/inference_responses.jsonl.gz ] || gzip -c $L/inference_responses.jsonl > $A/inference_responses.jsonl.gz
  fi
  AL=$(ls $HOME/gepa-repro/logs/arbor-$J.log $HOME/gepa-artifac/logs/arbor-$J.log 2>/dev/null | head -1)
  [ -n "$AL" ] && [ ! -f $A/arbor-$J.log.gz ] && gzip -c $AL > $A/arbor-$J.log.gz
  echo "packed $N seed $S: score $(sed -n 2p $src/evaluation_results/evaluation_result.txt | cut -d, -f1), job $J"
done
cp $HOME/repro_checks/SUMMARY.md $P/checks/SUMMARY_$(date +%F).md
tar czf $HOME/$OUT -C $P . && rm -rf $P && ls -la $HOME/$OUT
