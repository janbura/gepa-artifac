"""Re-score the test examples that a run lost to server failures (plan, section 2.12).

The final program is rebuilt from text, never from a pickle: the lineage and merges from
run_log.txt, each proposed instruction from instruction_proposer_inpouts.jsonl, the seed
instructions from the program class. The run's own DSPy cache is read from a copy.

--offline only checks the rebuild, with no model server: every answered test example has to
come from the cache with its logged score, the format failures have to fail again, and only the
lost examples may need the server. Without --offline the whole test set is evaluated against a
live server (port), so the lost examples get new answers, and the re-scored result is written.

    cd ~/gepa-repro && .venv/bin/python /path/to/rescore_lost_test.py --run_dir RUN --cache_src CACHE \
        --work WORK --lm_config '{...}' --offline --out check.json
"""
import argparse
import gzip
import importlib.util
import itertools
import json
import logging
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            if p.endswith(".gz"):
                with gzip.open(p, "rt", encoding="utf-8", errors="replace", newline="") as f:
                    return f.read()
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                return f.read()
    raise FileNotFoundError(path)


def load_tool():
    # the log parser of the lost-request check (analysis/lost_request_impact.py)
    for cand in (os.path.join(HERE, "lost_request_impact.py"), os.path.join(HERE, "..", "analysis", "lost_request_impact.py")):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("lri", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    sys.exit("lost_request_impact.py not found")


def rebuild(run_dir, lri, seed_instr):
    """Instructions of the final program per predictor; a list of candidates when a merge broke a
    tie at random (then the offline check decides)."""
    msgs = lri.stdout_messages(read(os.path.join(run_dir, "run_log.txt")))
    expected, progs, sels, merges, other, timeline = lri.parse_search(msgs, 300)
    props = [json.loads(l) for l in read(os.path.join(run_dir, "instruction_proposer_inpouts.jsonl")).splitlines() if l.strip()]
    logged = [m for it, m in msgs if m.startswith("Proposed new instruction: ")]
    if len(props) != len(logged):
        sys.exit("%d proposals in the jsonl, %d in the log" % (len(props), len(logged)))
    same_text = sum(1 for p, l in zip(props, logged) if p["new_instruction"].strip() == l[len("Proposed new instruction: "):].strip())
    # which proposal made which child program
    child_prop, cur, k = {}, None, -1
    for it, m in msgs:
        first = m.split("\n", 1)[0]
        if first.startswith("Selected program candidate"):
            cur = None
        elif first.startswith("Proposed new instruction: "):
            k += 1
            cur = k
        else:
            mm = re.match(r"New program candidate index: (\d+)", first)
            if mm and progs[int(mm.group(1))]["kind"] == "child":
                child_prop[int(mm.group(1))] = cur
    preds = sorted(seed_instr)
    maps = {0: {p: ("text", seed_instr[p]) for p in preds}}
    for i in sorted(progs):
        if i == 0:
            continue
        pr = progs[i]
        if pr["kind"] == "child":
            m = dict(maps[pr["parents"][0]])
            m[pr["pred"]] = ("text", props[child_prop[i]]["new_instruction"])
        else:
            m = lri.merged_map(maps, progs, pr["parents"][0], pr["parents"][1], pr["anc"], preds)
        maps[i] = m
    scores = [progs[i]["score"] for i in sorted(progs)]
    final = scores.index(max(scores))
    # expand ties ('either') into candidate programs
    options = []
    for p in preds:
        v = maps[final][p]
        options.append([x[1] for x in v[1:]] if v[0] == "either" else [v[1]])
    candidates = [dict(zip(preds, combo)) for combo in itertools.product(*options)]
    info = dict(programs=len(progs), proposals=len(props), proposals_same_text_as_log=same_text,
                merges=sum(1 for p in progs.values() if p["kind"] == "merge"), final_program=final,
                final_val_score=progs[final]["score"], logged_best_after_last_program=progs[max(progs)].get("best"),
                final_lineage=lineage(progs, final),
                changed_predictors=[p for p in preds if maps[final][p] != maps[0][p]],
                candidates=len(candidates))
    return candidates, info


def lineage(progs, i):
    out = []
    while i is not None:
        pr = progs[i]
        out.append("%d (%s)" % (i, pr["kind"] if pr["kind"] != "child" else "child of %d, %s" % (pr["parents"][0], pr["pred"])))
        i = pr["parents"][0]
    return out


def logged_test(run_dir, test_keys):
    """Per test example: the logged scores (several when DSPy answered two copies), and which
    examples failed by format and which were lost (error lines of the test evaluation)."""
    scored = {}
    for line in read(os.path.join(run_dir, "metric_logs", "test.jsonl")).splitlines():
        if line.strip():
            d = json.loads(line)
            scored.setdefault(d["idx_in_split"], []).append(float(d["metric_output"]))
    text = read(os.path.join(run_dir, "run_log_stderr.txt"))
    stamp = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) (\w+) ([\w.]+): (.*)$")
    evs = []
    for line in text.split("\n"):
        m = stamp.match(line)
        if m:
            evs.append([m.group(1), m.group(4), []])
        elif evs:
            evs[-1][2].append(line)
    avg = [k for k, e in enumerate(evs) if "Average Metric" in e[1]]
    lost, fmt = set(), set()
    for e in evs[avg[-2] + 1:avg[-1]]:
        if not e[1].startswith("Error for "):
            continue
        m = re.search(r"'key': '?(\d+)'?", e[1])
        if not m or int(m.group(1)) not in test_keys:
            continue
        i = test_keys[int(m.group(1))]
        (lost if "InternalServerError" in "\n".join([e[1]] + e[2]) else fmt).add(i)
    # examples with a format-failure line that were still scored: DSPy ran two copies of them and
    # one failed to parse, so the cache may hold either copy's answer
    return scored, lost - set(scored), fmt - set(scored) - lost, (fmt & set(scored)) - lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--cache_src", required=True, help="the run's cache folder (with .dspy_cache)")
    ap.add_argument("--work", required=True, help="work folder; the cache is copied there once")
    ap.add_argument("--lm_config", required=True)
    ap.add_argument("--code_dir", default=os.getcwd(), help="checkout of the runs (gepa_artifact, scripts)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--candidate", type=int, default=None, help="live run: the candidate that passed the offline check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.offline == (args.port is not None):
        sys.exit("give either --offline or --port")

    cache_copy = os.path.join(args.work, "cache_copy")
    if not os.path.isdir(cache_copy):
        shutil.copytree(args.cache_src, cache_copy)
    # dspy reads the cache folder when it is imported
    for var in ["DSPY_CACHEDIR", "DSP_CACHEDIR"]:
        os.environ[var] = os.path.join(cache_copy, ".dspy_cache")
    sys.path.insert(0, args.code_dir)
    import dspy
    from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas
    from scripts.run_experiments import create_lm

    lri = load_tool()
    meta = ifbench_metas[0]
    base = meta.program[0]
    seed_instr = {name: pred.signature.instructions for name, pred in base.named_predictors()}
    candidates, info = rebuild(args.run_dir, lri, seed_instr)

    test_set = meta.benchmark().test_set
    test_keys = {int(ex.key): i for i, ex in enumerate(test_set)}
    scored, lost, fmt, fmt_scored = logged_test(args.run_dir, test_keys)
    official = float(read(os.path.join(args.run_dir, "evaluation_results", "evaluation_result.txt")).splitlines()[1].split(",")[0])
    first_sum = sum(v[0] for v in scored.values())
    last_sum = sum(v[-1] for v in scored.values())
    info.update(test_examples=len(test_set), scored=len(scored), lost=len(lost), format_failures=len(fmt),
                scored_after_a_failed_copy=sorted(fmt_scored),
                scored_twice=sum(1 for v in scored.values() if len(v) > 1),
                official_score=official, sum_first_logged=first_sum, sum_last_logged=last_sum)

    lm_config = json.loads(args.lm_config)
    lm_config["api_base"] = lm_config["api_base"].format(portnum=args.port if args.port else 9)
    lm = create_lm(lm_config)
    dspy.configure(lm=lm)
    logging.getLogger("dspy").setLevel(logging.ERROR)

    def run_candidate(instr):
        prog = base.deepcopy()
        for name, pred in prog.named_predictors():
            pred.signature = pred.signature.with_instructions(instr[name])

        def one(i):
            try:
                return i, "ok", prog(**test_set[i].inputs()), ""
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:300])
                server = any(t in msg for t in ("InternalServerError", "APIConnectionError", "Connection", "ServiceUnavailable", "Timeout"))
                return i, "server" if server else "failed", None, msg

        # DSPy's own executor (thread-local settings as in Evaluate), without second copies of
        # slow examples (timeout 0), so each example is answered once
        from dspy.utils.parallelizer import ParallelExecutor
        ex = ParallelExecutor(num_threads=args.threads, max_errors=10 ** 6, disable_progress_bar=True,
                              provide_traceback=False, timeout=0)
        res = ex.execute(one, list(range(len(test_set))))
        # the metric afterwards, one example at a time: NLTK's lazy corpus loading is not thread-safe
        # (in threads it can fail with "'WordListCorpusReader' object has no attribute ...")
        out = {}
        for i, st, pred, msg in res:
            if st == "ok":
                try:
                    out[i] = ("ok", float(meta.metric(test_set[i], pred)), "")
                except Exception as e:
                    out[i] = ("failed", 0.0, "metric: %s: %s" % (type(e).__name__, str(e)[:300]))
            else:
                out[i] = (st, 0.0, msg)
        return out

    report = dict(info=info, mode="offline" if args.offline else "live", started=time.strftime("%F %T"))
    passing = []
    todo = list(enumerate(candidates))
    if not args.offline:
        # the live run uses only the program that passed the offline check
        if args.candidate is None and len(candidates) != 1:
            sys.exit("%d candidate programs: give --candidate from the offline check" % len(candidates))
        todo = [todo[args.candidate or 0]]
    for n, instr in todo:
        res = run_candidate(instr)
        bad_scored = [i for i in scored if res[i][0] != "ok" or not any(abs(res[i][1] - v) < 1e-9 for v in scored[i])]
        bad_format = [i for i in fmt if res[i][0] == "ok"]
        needs_server = sorted(i for i in res if res[i][0] == "server")
        check = dict(candidate=n, scored_reproduced=len(scored) - len(bad_scored), scored_total=len(scored),
                     scored_mismatch=[(i, res[i][0], res[i][1], scored[i], res[i][2][:120]) for i in bad_scored[:10]],
                     format_failing_again=len(fmt) - len(bad_format), format_total=len(fmt),
                     needing_server=len(needs_server), needing_server_not_lost=[i for i in needs_server if i not in lost],
                     lost_needing_server=sum(1 for i in lost if res[i][0] == "server"))
        second_copy = [i for i in bad_scored if i in fmt_scored and res[i][0] == "failed"]
        check["scored_mismatch_explained_by_a_failed_copy"] = second_copy
        if args.offline:
            as_written = (not bad_scored and not bad_format and set(needs_server) == lost)
            # addendum of plan 2.12: an example scored from one copy while another copy failed to
            # parse (both in the log) also passes when the replay fails to parse it
            with_addendum = (set(bad_scored) <= set(second_copy) and not bad_format and set(needs_server) == lost)
            check["passes_as_written"] = as_written
            check["passes_with_addendum"] = with_addendum
            if with_addendum:
                passing.append(n)
        else:
            new = {i: res[i] for i in sorted(lost)}
            check["lost_examples"] = {i: dict(status=s, score=sc, error=msg[:200]) for i, (s, sc, msg) in new.items()}
            new_sum = sum(sc for s, sc, msg in new.values())
            check["lost_answered"] = sum(1 for s, sc, msg in new.values() if s == "ok")
            check["lost_again"] = sum(1 for s, sc, msg in new.values() if s == "server")
            check["lost_format_failed"] = sum(1 for s, sc, msg in new.values() if s == "failed")
            check["new_points_of_the_lost"] = new_sum
            base_sum = first_sum if abs(100 * first_sum / len(test_set) - official) < 0.006 else last_sum
            check["official_points"] = base_sum
            check["rescored_score"] = round(100 * (base_sum + new_sum) / len(test_set), 2)
            check["replay_score_all_examples"] = round(100 * sum(sc for s, sc, msg in res.values()) / len(test_set), 2)
        report.setdefault("candidates", []).append(check)
    if args.offline:
        report["passing_candidates"] = passing
        report["verdict"] = "REBUILD EXACT" if len(passing) == 1 else "REBUILD NOT EXACT (%d candidates pass)" % len(passing)
    report["ended"] = time.strftime("%F %T")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
