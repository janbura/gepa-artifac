# Estimated SBU cost of one faithful GRPO run (500 steps, 24,000 rollouts) on Snellius.
#
# Inputs:
#   - wall-clock of the authors' four GRPO runs: the [claim8] lines of stderr_analysis_out.txt
#     (first to last timestamp of run_log_stderr.txt: training, 26 validations and the test step)
#   - 3 GPUs per run: vLLM rollouts with tensor parallel 2 on GPUs 1-2, training on GPU 0
#     (arbor_train.yaml, plan F28 and section 2.7)
#   - Snellius rates: 192 SBU per H100-hour, 128 SBU per A100-hour (plan F5)
#
# The logs do not say which GPU the authors used (the paper says H100/A100), so both are printed.
# The estimate assumes our GPUs run as fast as theirs. The test step after training takes 5-7 minutes
# in all four logs (failed or not), so the spans are almost all training and validation. Model loading
# and server start-up before the first log line are not counted.
#
# usage: python -I cost_estimate.py stderr_analysis_out.txt

import re
import sys

GPUS = 3
RATES = {"H100": 192, "A100": 128}
SEEDS = 3


def read_spans(path):
    spans = {}
    run = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("_GRPO_qwen3-8b"):
                run = line.strip()
            m = re.match(r"\[claim8\] line-start timestamps: .*; span ([0-9.]+) h", line)
            if m and run is not None:
                spans[run] = float(m.group(1))
    return spans


def main():
    spans = read_spans(sys.argv[1])
    if len(spans) != 4:
        sys.exit("expected 4 GRPO runs, found %d" % len(spans))

    print("GPUs per run: %d; rates: %s SBU per GPU-hour" % (GPUS, RATES))
    print()
    print("%-48s %8s %10s %10s" % ("run", "hours", "SBU H100", "SBU A100"))
    total_h = 0.0
    for run in sorted(spans):
        h = spans[run]
        total_h += h
        print("%-48s %8.2f %10.0f %10.0f" % (run, h, h * GPUS * RATES["H100"], h * GPUS * RATES["A100"]))
    print("%-48s %8.2f %10.0f %10.0f" % ("all four, one seed", total_h,
                                          total_h * GPUS * RATES["H100"], total_h * GPUS * RATES["A100"]))
    print("%-48s %8.2f %10.0f %10.0f" % ("all four, %d seeds" % SEEDS, SEEDS * total_h,
                                          SEEDS * total_h * GPUS * RATES["H100"],
                                          SEEDS * total_h * GPUS * RATES["A100"]))
    lo = min(spans.values())
    hi = max(spans.values())
    print()
    print("one run: %.0f-%.0f SBU on H100, %.0f-%.0f SBU on A100" % (
        lo * GPUS * RATES["H100"], hi * GPUS * RATES["H100"],
        lo * GPUS * RATES["A100"], hi * GPUS * RATES["A100"]))


if __name__ == "__main__":
    main()
