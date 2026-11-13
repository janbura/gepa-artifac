#!/bin/bash
# Watches the GEPA runs of one job and cancels the job when GEPA has stopped proposing instructions.
# GEPA does not count an iteration toward its budget when the reflection call fails or when every
# minibatch answer is already perfect (gepa.py, the "continue" paths around lines 532-558), so a broken
# reflection keeps evaluating minibatches until the time limit, and the job watchdog does not notice
# because requests keep coming. In the authors' 19 GEPA runs the longest stretch without a proposed
# instruction was 14 iterations and no run had more than one failed proposal, so the limits below only
# stop runs that are clearly broken.
#   monitor_gepa.sh <job id> <run folder> [run folder ...]
# Every call does one pass and, while the job is still queued or running, submits the next pass 30 minutes
# later. Shared cpu jobs are charged for at least 16 cores, so one long-running monitor would cost about
# 16 SBU per hour; one pass takes seconds. Start the chain together with the GEPA job:
#   sbatch -p rome -n 1 -c 1 -t 00:05:00 -J mon-<job> -o $HOME/repro_checks/mon-<job>.log --open-mode=append \
#     --dependency=after:<job> --wrap "bash $HOME/repro_checks/monitor_gepa.sh <job> <run folders>"
# TEST=1 does one pass over the folders, prints what it would do and touches nothing.
# The settings below can be changed by environment only for the pass they are given to: Snellius does not
# pass the environment on to jobs (F63), so the later passes always use the defaults.
J=$1; shift
OUT=${OUT:-$HOME/repro_checks}  # where the report and the SUMMARY.md line go
MAX_GAP=${MAX_GAP:-40}   # iterations in a row without a proposed instruction
MAX_ERR=${MAX_ERR:-10}   # failed proposals (or other exceptions), and more of them than successful proposals
EVERY=${EVERY:-30}       # minutes between passes

check_run() {  # sets WHY (empty while the run looks fine) and the counters
  WHY=""; last=""; lastp=""; np=0; ep=0; eo=0
  L=$1/run_log.txt
  [ -f "$L" ] || return 0
  last=$(grep -oE "^Iteration [0-9]+:" $L | tail -1 | grep -oE "[0-9]+")
  lastp=$(grep -oE "^Iteration [0-9]+: Proposed new instruction" $L | tail -1 | grep -oE "^Iteration [0-9]+" | grep -oE "[0-9]+")
  np=$(grep -cE "^Iteration [0-9]+: Proposed new instruction" $L)
  ep=$(grep -c "Exception during instruction proposal" $L)
  eo=$(grep -c "Exception during optimization" $L)
  gap=$(( ${last:-0} - ${lastp:-0} ))
  if [ $gap -ge $MAX_GAP ]; then WHY="$gap iterations in a row without a proposed instruction (last iteration $last, proposals $np)"
  elif [ $ep -ge $MAX_ERR ] && [ $ep -gt $np ]; then WHY="$ep failed instruction proposals against $np successful ones"
  elif [ $eo -ge $MAX_ERR ] && [ $eo -gt $np ]; then WHY="$eo exceptions during optimization against $np proposals"
  fi
}

if [ "$TEST" = 1 ]; then
  for d in "$@"; do check_run $d; echo "$d: ${WHY:-fine} [last iteration ${last:-none}, last proposal ${lastp:-none}, proposals $np, failed proposals $ep, other exceptions $eo]"; done
  exit 0
fi

st=$(squeue -h -j $J -o %T 2>/dev/null)
if [ -z "$st" ]; then echo "$(date '+%F %T') job $J has ended, monitor stops"; exit 0; fi
if [ "$st" = RUNNING ]; then
  for d in "$@"; do
    check_run $d
    if [ -n "$WHY" ]; then
      scancel $J
      { echo "# GEPA monitor stopped job $J at $(date '+%F %T')"; echo
        echo "- run: $d"; echo "- reason: $WHY"; echo
        echo '```'; grep -E "^Iteration" $d/run_log.txt | tail -25 | cut -c1-300
        grep -A4 "Exception during" $d/run_log.txt | tail -25 | cut -c1-300; echo '```'; } > $OUT/monitor-$J.md
      echo "| $J | $(date '+%F %T') | CHECK | stopped by the GEPA monitor: $WHY, $(basename $d); see monitor-$J.md |  |" >> $OUT/SUMMARY.md
      echo "$(date '+%F %T') stopped job $J: $WHY ($d)"
      exit 1
    fi
    [ -f $d/run_log.txt ] && echo "$(date '+%F %T') $(basename $d): fine, last iteration ${last:-0}, proposals $np, failed proposals $ep"
  done
fi
next=$(sbatch --parsable -p rome -n 1 -c 1 -t 00:05:00 -J mon-$J -o $OUT/mon-$J.log --open-mode=append \
  --begin=now+${EVERY}minutes --wrap "bash $HOME/repro_checks/monitor_gepa.sh $J $*" 2>&1 | tail -1)
echo "$(date '+%F %T') job $J is $st, next pass: $next"
