"""
Replay one of the shipped optimization runs (for example GEPA on AIME) using
only the cached LLM responses, write the result into a separate folder and
compare it with the original run.

The experiment code reads its base folder from the environment variable
GEPA_EXPERIMENT_DIR, so we build a small replay folder that only contains the
cache of the run we want to repeat and start scripts.run_experiments on it.
With a dummy API key every request that is not in the cache fails, which makes
cache misses visible instead of silently spending money.

Example:
    python -m scripts.replay_optimization --run AIMEBench_CoT_GEPA_gpt-41-mini --replay_dir ~/gepa_replay
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time


def read_test_score(run_dir):
    with open(os.path.join(run_dir, "evaluation_results", "evaluation_result.txt")) as f:
        f.readline()
        return float(f.readline().split(",")[0])


def load_state(run_dir):
    # gepa_state.bin holds everything except the candidate programs themselves
    with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="e.g. AIMEBench_CoT_GEPA_gpt-41-mini")
    p.add_argument("--replay_dir", required=True, help="folder for the replayed run (created if missing)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--dry_run", action="store_true", help="2 examples per split, just to see that it runs")
    p.add_argument("--out", default=None, help="json summary file")
    args = p.parse_args()

    from scripts.experiment_configs import BASE_EXPERIMENT_DIR, LM_CONFIGS, get_benchmarks, get_optimizers

    bench_name, prog_name, opt_name, lm_name = args.run.split("_")
    seed_dir = f"seed_{args.seed}"
    replay_dir = os.path.abspath(os.path.expanduser(args.replay_dir))

    src_cache = os.path.join(BASE_EXPERIMENT_DIR, "experiment_cache_dirs", seed_dir, args.run)
    dst_cache = os.path.join(replay_dir, "experiment_cache_dirs", seed_dir, args.run)
    if not os.path.isdir(dst_cache):
        print(f"copying cache {src_cache} -> {dst_cache}")
        shutil.copytree(src_cache, dst_cache)

    bench_names = [m.name or m.benchmark.__name__ for m in get_benchmarks()]
    opt_names = [name for name, _ in get_optimizers()]
    bm_idx = bench_names.index(bench_name)
    opt_idx = opt_names.index(opt_name)
    lm_config = [c for c in LM_CONFIGS if c["name"] == lm_name][0]

    cmd = [
        sys.executable, "-m", "scripts.run_experiments",
        "--bm_idx", str(bm_idx), "--benchmark_name", bench_name,
        "--num_threads", str(args.num_threads),
        "--program_idx", "0", "--prog_name", prog_name,
        "--opt_idx", str(opt_idx), "--optim_name", opt_name,
        "--lm_config", json.dumps(lm_config),
        "--seed", str(args.seed),
    ]
    if args.dry_run:
        cmd.append("--dry_run")

    env = dict(os.environ)
    env["GEPA_EXPERIMENT_DIR"] = replay_dir
    env["GEPA_USE_WANDB"] = "0"
    env.setdefault("OPENAI_API_KEY", "not-a-real-key")

    print(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    elapsed = time.time() - t0
    print(f"run finished in {elapsed:.0f}s")

    orig = os.path.join(BASE_EXPERIMENT_DIR, "experiment_runs", seed_dir, args.run)
    new = os.path.join(replay_dir, "experiment_runs", seed_dir, args.run)
    summary = {
        "run": args.run,
        "seconds": round(elapsed),
        "test_score_original": read_test_score(orig),
        "test_score_replay": read_test_score(new),
    }
    if "GEPA" in opt_name or opt_name == "Abl-SelectBestCandidate":
        s0, s1 = load_state(orig), load_state(new)
        summary["val_scores_original"] = s0["program_full_scores_val_set"]
        summary["val_scores_replay"] = s1["program_full_scores_val_set"]
        summary["calls_at_discovery_original"] = s0["num_metric_calls_by_discovery"]
        summary["calls_at_discovery_replay"] = s1["num_metric_calls_by_discovery"]
        summary["same_trajectory"] = (
            s0["program_full_scores_val_set"] == s1["program_full_scores_val_set"]
            and s0["num_metric_calls_by_discovery"] == s1["num_metric_calls_by_discovery"]
        )
    summary["same_test_score"] = abs(summary["test_score_original"] - summary["test_score_replay"]) < 1e-6
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
