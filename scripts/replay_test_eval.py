"""
Re-run the final test-set evaluation of one of the shipped experiment runs,
using only the cached LLM responses that come with the data artifact.

If every request is served from the cache, the score should be identical to
the one stored in evaluation_results/evaluation_result.txt and the token
counters stay at zero (cache hits report no usage), so we never need an API
key or a GPU for this check.

Example:
    python -m scripts.replay_test_eval --run AIMEBench_CoT_GEPA_gpt-41-mini
"""

import argparse
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="run name, e.g. AIMEBench_CoT_GEPA_gpt-41-mini")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="only use the first N test examples (quick check)")
    p.add_argument("--out", default=None,
                   help="where to write a small json summary")
    p.add_argument("--replay_dir", default=None,
                   help="work folder; the run's cache is copied here so the shipped data stays untouched "
                        "(default: <repo>/replay_runs)")
    return p.parse_args()


def main():
    args = parse_args()

    from scripts.experiment_configs import BASE_EXPERIMENT_DIR, LM_CONFIGS

    seed_dir = f"seed_{args.seed}"
    run_dir = os.path.join(BASE_EXPERIMENT_DIR, "experiment_runs", seed_dir, args.run)
    src_cache = os.path.join(BASE_EXPERIMENT_DIR, "experiment_cache_dirs", seed_dir, args.run)
    if not os.path.isdir(run_dir):
        sys.exit(f"run dir not found: {run_dir}")
    if not os.path.isdir(src_cache):
        sys.exit(f"cache dir not found: {src_cache}")

    # work on a copy: the sqlite files get opened for writing by the cache library
    replay_dir = args.replay_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "replay_runs")
    cache_copy = os.path.join(replay_dir, "experiment_cache_dirs", seed_dir, args.run)
    if not os.path.isdir(cache_copy):
        import shutil
        print(f"copying cache to {cache_copy}")
        shutil.copytree(src_cache, cache_copy)
    cache_dir = os.path.join(cache_copy, ".dspy_cache")

    # dspy picks up the cache directory from these variables when it is imported,
    # so they have to be set before the import below
    for var in ["DSPY_CACHEDIR", "DSP_CACHEDIR", "DSPY_NOTEBOOK_CACHEDIR", "DSP_NOTEBOOK_CACHEDIR"]:
        os.environ[var] = cache_dir
    # a dummy key is enough as long as everything comes from the cache
    os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")

    import dspy
    from gepa_artifact import benchmarks as bench_pkg
    # the saved programs were pickled when the package had a different name
    sys.modules["langProBe"] = bench_pkg
    sys.modules["benchmarks"] = bench_pkg
    sys.modules["benchmarks.langProBe.langProBe.langProBe"] = bench_pkg

    from scripts.experiment_configs import get_benchmarks
    from scripts.run_experiments import create_lm, calculate_stats

    bench_name, prog_name, opt_name, lm_name = args.run.split("_")

    meta = None
    for m in get_benchmarks():
        if (m.name or m.benchmark.__name__) == bench_name:
            meta = m
    if meta is None:
        sys.exit(f"benchmark {bench_name} not found in experiment_configs")

    lm_config = None
    for c in LM_CONFIGS:
        if c["name"] == lm_name:
            lm_config = dict(c)
    if lm_config is None:
        sys.exit(f"lm config {lm_name} not found")
    if lm_config["api_key"].startswith("env:"):
        lm_config["api_key"] = os.environ[lm_config["api_key"].split(":")[1]]
    if "api_base" in lm_config:
        # the port only mattered for the local server, the cache key ignores it
        lm_config["api_base"] = lm_config["api_base"].format(portnum=0)

    benchmark = meta.benchmark()
    test_set = benchmark.test_set
    if args.limit:
        test_set = test_set[:args.limit]
    print(f"{bench_name}: {len(test_set)} test examples")

    if opt_name == "Baseline":
        program = meta.program[0]
    else:
        program = dspy.load(os.path.join(run_dir, "evaluation_results", "optimized_program"))
    for name, pred in program.named_predictors():
        print(f"predictor {name}: {len(pred.signature.instructions)} chars of instruction, {len(pred.demos)} demos")

    lm = create_lm(lm_config)
    dspy.configure(lm=lm, adapter=dspy.settings.adapter)

    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=meta.metric,
        num_threads=args.num_threads,
        display_progress=True,
        max_errors=len(test_set) * 10,
        provide_traceback=False,
        return_all_scores=True,
    )
    t0 = time.time()
    score, per_example = evaluate(program)
    elapsed = time.time() - t0
    cost, in_tok, out_tok = calculate_stats(lm)

    # per-example scores of the original run, from the metric log of the test split
    stored_per_example = {}
    with open(os.path.join(run_dir, "metric_logs", "test.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("idx_in_split") is not None:
                stored_per_example[d["idx_in_split"]] = d["metric_output"]
    same = sum(1 for i, s in enumerate(per_example) if i in stored_per_example and float(stored_per_example[i]) == float(s))
    compared = sum(1 for i in range(len(per_example)) if i in stored_per_example)

    # the stored result from the original run
    with open(os.path.join(run_dir, "evaluation_results", "evaluation_result.txt")) as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    stored = dict(zip(header, values))
    stored_score = float(stored["score"])

    summary = {
        "run": args.run,
        "n_test": len(test_set),
        "replayed_score": float(score),
        "stored_score": stored_score,
        "match": (args.limit is None) and abs(float(score) - stored_score) < 1e-6,
        "examples_with_same_score_as_original": f"{same}/{compared}",
        "llm_calls_answered": len(lm.history),
        "tokens_not_from_cache": in_tok + out_tok,
        "cost": cost,
        "seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
