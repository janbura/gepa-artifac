"""Could the requests lost in a GEPA run have changed its search? (plan, section 2.11)

For every request lost to a client InternalServerError in a GEPA or GEPA+Merge run, this
script finds the step of the search the request belonged to and decides whether it could
have changed anything the search did, given every answer the run got:
CC = could change, NC = no change. The rules are the ones of section 2.11 of the plan
(written before the script was run on a run with losses).

Only the text logs of a run folder are read (run_log.txt and run_log_stderr.txt, plain or
.gz). No pickle is opened.

    python -I analysis/lost_request_impact.py --md report.md --json report.json RUN_DIR ...
"""
import argparse
import ast
import gzip
import json
import os
import re
import sys
from collections import Counter

# order in which the modules are called for one example (forward() of the two programs)
MODULE_ORDER = {
    "HotpotQABench": ["summarize1", "create_query_hop2", "summarize2", "final_answer"],
    "hoverBench": ["summarize1", "create_query_hop2", "summarize2", "create_query_hop3"],
}

ITER_RE = re.compile(r"^Iteration (\d+): (.*)$")
STAMP_RE = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) (\w+) ([\w.]+): (.*)$")
AVG_RE = re.compile(r"Average Metric: ([0-9.]+) / (\d+) \(")
NUM = r"([0-9]+(?:\.[0-9]+)?)"


def read_text(run_dir, name):
    for fn in (name, name + ".gz"):
        path = os.path.join(run_dir, fn)
        if os.path.exists(path):
            if fn.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
                    return f.read()
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                return f.read()
    raise FileNotFoundError(os.path.join(run_dir, name))


def num(s):
    x = float(s)
    return int(x) if x == int(x) else x


def to01(v):
    # logged scores are True / False, and 0 (DSPy's failure score) for examples that failed
    return 1 if v else 0


def pct(a, n):
    # the score DSPy's Evaluate returns
    return round(100 * a / n, 2)


# ---------------------------------------------------------------- copies of the GEPA code
# (gepa_artifact/gepa/entropy_utils.py and gepa_utils.py at 6ea0b87, unchanged logic, so
# that the rebuilt Pareto fronts give the same lists as in the run)

def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False
    return True


def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset:
        for p in front:
            freq[p] = freq.get(p, 0) + 1
    dominated = set()
    programs = list(freq.keys())
    if scores is None:
        scores = {p: 1 for p in programs}
    programs = sorted(programs, key=lambda x: scores[x], reverse=False)
    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break
    dominators = [p for p in programs if p not in dominated]
    return [{prog_idx for prog_idx in front if prog_idx in dominators} for front in program_at_pareto_front_valset]


def sampling_list(fronts, scores):
    # select_program_candidate_from_pareto_front without the random draw
    new_fronts = remove_dominated_programs(fronts, scores=scores)
    freq = {}
    for front in new_fronts:
        for prog_idx in front:
            if prog_idx not in freq:
                freq[prog_idx] = 0
            freq[prog_idx] += 1
    return [prog_idx for prog_idx, f in freq.items() for _ in range(f)]


def merge_candidates(fronts, scores):
    # find_dominator_programs
    new_fronts = remove_dominated_programs(fronts, scores=scores)
    uniq = []
    for front in new_fronts:
        uniq.extend(front)
    return list(set(uniq))


def idxmax(lst):
    return lst.index(max(lst))


# ---------------------------------------------------------------- reading the logs

def stdout_messages(text):
    """GEPA's messages in order as [iteration, text]. A message ends at the next message or
    progress bar, so a proposed instruction over several lines stays whole."""
    msgs = []
    cur = None
    for line in text.split("\n"):
        m = ITER_RE.match(line)
        if m:
            cur = [int(m.group(1)), m.group(2)]
            msgs.append(cur)
        elif "\r" in line or re.match(r"^\s*\d+%\|", line) or line.startswith("Updating shuffled trainset"):
            cur = None
        elif cur is not None:
            cur[1] += "\n" + line
    return msgs


def parse_search(msgs, val_n):
    """The evaluations GEPA made, in order, and the programs, selections and merge calls."""
    evals, progs, sels, merges, other, timeline = [], {}, [], [], [], []
    cur = {}
    for it, msg in msgs:
        first = msg.split("\n", 1)[0]
        if first.startswith("Base program full valset score: "):
            score = float(first.split(": ", 1)[1])
            evals.append(dict(kind="base val", n=val_n, iter=it, score=score, prog=0))
            progs[0] = dict(idx=0, iter=0, parents=[None], kind="base", score=score)
            continue
        m = re.match(r"Merged programs (\d+) and (\d+) via ancestor (\d+)", first)
        if m:
            a, b, c = map(int, m.groups())
            merges.append(dict(iter=it, found=True, a=a, b=b, anc=c))
            timeline.append(("merge", len(merges) - 1))
            evals.append(dict(kind="merge subsample", n=5, iter=it, a=a, b=b, anc=c))
            continue
        if first.startswith("No merge candidates found"):
            merges.append(dict(iter=it, found=False))
            timeline.append(("merge", len(merges) - 1))
            continue
        m = re.match(r"New program subsample score %s for merged program is better than min of both parents %s and %s" % (NUM, NUM, NUM), first)
        if m:
            e = evals[-1]
            assert e["kind"] == "merge subsample" and e["iter"] == it, (it, first)
            e.update(score=num(m.group(1)), p1=num(m.group(2)), p2=num(m.group(3)), kept=True)
            evals.append(dict(kind="val", n=val_n, iter=it, source="merge", parents=[e["a"], e["b"]], anc=e["anc"]))
            continue
        m = re.match(r"New program subsample score %s is worse than both parent programs %s and %s" % (NUM, NUM, NUM), first)
        if m:
            e = evals[-1]
            assert e["kind"] == "merge subsample" and e["iter"] == it, (it, first)
            e.update(score=num(m.group(1)), p1=num(m.group(2)), p2=num(m.group(3)), kept=False)
            continue
        m = re.match(r"Selected program candidate (\d+) with base score: (\S+)", first)
        if m:
            cur = dict(iter=it, parent=int(m.group(1)))
            sels.append(dict(iter=it, prog=int(m.group(1)), base_score=float(m.group(2))))
            timeline.append(("select", len(sels) - 1))
            continue
        if first.startswith("Updating predictor "):
            assert cur.get("iter") == it, (it, first)
            cur["pred"] = first[len("Updating predictor "):].strip()
            evals.append(dict(kind="parent minibatch", n=3, iter=it, parent=cur["parent"], pred=cur["pred"]))
            continue
        if first.startswith("Proposed new instruction: "):
            cur["instr"] = msg[len("Proposed new instruction: "):]
            continue
        if first.startswith("Exception during instruction proposal"):
            other.append(dict(iter=it, kind="proposal exception", text=msg))
            continue
        if first.startswith("Exception during optimization"):
            other.append(dict(iter=it, kind="iteration exception", text=msg))
            continue
        if first.startswith("Loading gepa state"):
            other.append(dict(iter=it, kind="state loaded from disk", text=msg))
            continue
        m = re.match(r"New subsample score: %s$" % NUM, first)
        if m:
            assert cur.get("iter") == it, (it, first)
            evals.append(dict(kind="child minibatch", n=3, iter=it, score=num(m.group(1)), parent=cur["parent"],
                              pred=cur["pred"], instr=cur.get("instr")))
            continue
        if first.startswith("New subsample score is not better"):
            assert evals[-1]["kind"] == "child minibatch" and evals[-1]["iter"] == it, (it, first)
            evals[-1]["kept"] = False
            continue
        m = re.match(r"New subsample score is better, going from %s to %s" % (NUM, NUM), first)
        if m:
            child, parent_eval = evals[-1], evals[-2]
            assert child["kind"] == "child minibatch" and parent_eval["kind"] == "parent minibatch", (it, first)
            child["kept"] = True
            parent_eval["score"] = num(m.group(1))
            assert num(m.group(2)) == child["score"], (it, first)
            evals.append(dict(kind="val", n=val_n, iter=it, source="child", parents=[cur["parent"]],
                              pred=cur["pred"], instr=cur["instr"]))
            continue
        if first.startswith("Full valset score for new program: "):
            evals[-1]["score"] = float(first.split(": ", 1)[1])
            continue
        if first.startswith("Individual valset scores for new program: "):
            evals[-1]["subscores"] = ast.literal_eval(first.split(": ", 1)[1])
            continue
        if first.startswith("Updated valset pareto front programs: "):
            evals[-1]["fronts"] = ast.literal_eval(first.split(": ", 1)[1])
            continue
        if first.startswith("Best program as per aggregate score on valset: "):
            evals[-1]["best"] = int(first.split(": ", 1)[1])
            continue
        m = re.match(r"New program candidate index: (\d+)", first)
        if m:
            e = evals[-1]
            assert e["kind"] == "val" and e["iter"] == it, (it, first)
            idx = int(m.group(1))
            e["prog"] = idx
            progs[idx] = dict(idx=idx, iter=it, parents=e["parents"], kind=e["source"], score=e["score"],
                              subscores=e["subscores"], fronts=e["fronts"], best=e.get("best"),
                              pred=e.get("pred"), instr=e.get("instr"), anc=e.get("anc"))
            timeline.append(("add", idx))
            continue
    # after the search the run evaluates the final program on the test set
    evals.append(dict(kind="test", n=None, iter=None))
    return evals, progs, sels, merges, other, timeline


def stderr_lines(text):
    """Timestamped lines of the DSPy loggers, each with the untimed lines (tracebacks) after it."""
    evs = []
    for line in text.split("\n"):
        m = STAMP_RE.match(line)
        if m:
            evs.append(dict(time=m.group(1), level=m.group(2), logger=m.group(3), text=m.group(4), extra=[]))
        elif evs:
            evs[-1]["extra"].append(line)
    return evs


def module_from_traceback(lines):
    # innermost frame of the program's forward(): the line after it calls self.<module>(...)
    mod = None
    for k, line in enumerate(lines):
        if re.search(r'_program\.py", line \d+, in forward', line) and k + 1 < len(lines):
            m = re.search(r"self\.(\w+)\(", lines[k + 1])
            if m:
                mod = m.group(1)
    return mod


def example_label(text):
    for key in ("id", "claim", "question"):
        m = re.search(r"'%s': (['\"])(.*?)\1" % key, text)
        if m:
            return "%s=%s" % (key, m.group(2)[:70])
    return text[:70]


def observed_evals(evs):
    """Evaluations as logged by DSPy (Average Metric a / n) with the errors logged during each."""
    obs, pending = [], []
    for e in evs:
        if e["logger"] == "dspy.evaluate.evaluate":
            m = AVG_RE.search(e["text"])
            if m:
                obs.append(dict(a=num(m.group(1)), n=int(m.group(2)), time=e["time"], errors=pending))
                pending = []
        elif e["logger"] == "dspy.utils.parallelizer" and e["text"].startswith("Error for "):
            first = e["text"]
            body = "\n".join([first] + e["extra"])
            lost = "InternalServerError" in body
            pending.append(dict(time=e["time"], lost=lost, example=example_label(first), first=first,
                                module=module_from_traceback(e["extra"]) if lost else None,
                                message=(re.findall(r"litellm\.InternalServerError[^\n]*", body) or [""])[0][:160]))
    return obs, pending


def align(expected, obs):
    problems = []
    if len(expected) != len(obs):
        problems.append("expected %d evaluations, the log has %d" % (len(expected), len(obs)))
    for k, (x, o) in enumerate(zip(expected, obs)):
        x["obs"] = o
        if x["n"] is not None and x["n"] != o["n"]:
            problems.append("evaluation %d (%s, iteration %s): size %s, logged %s" % (k, x["kind"], x["iter"], x["n"], o["n"]))
            continue
        if "score" not in x:
            continue
        if x["kind"] in ("base val", "val"):
            ok = pct(o["a"], o["n"]) == x["score"]
        else:
            ok = o["a"] == x["score"]
        if not ok:
            problems.append("evaluation %d (%s, iteration %s): score %s, DSPy logged %s / %s" % (k, x["kind"], x["iter"], x["score"], o["a"], o["n"]))
    return problems


# ---------------------------------------------------------------- rebuilding the search

def base_subscores(progs, val_n):
    """Per-example val scores of the base program: not logged as such, but fixed by the
    first new program's scores and the front logged after it (scores are 0 or 1)."""
    if 1 not in progs:
        return None
    s1 = [to01(v) for v in progs[1]["subscores"]]
    s0 = []
    for j, front in enumerate(progs[1]["fronts"]):
        if front == {0, 1}:
            s0.append(s1[j])
        elif front == {1} and s1[j] == 1:
            s0.append(0)
        elif front == {0} and s1[j] == 0:
            s0.append(1)
        else:
            raise ValueError("front %s with score %s at example %d" % (front, s1[j], j))
    return s0


def replay(s0, progs, timeline, sels, merges, val_n, flip=None, check=False):
    """Plays the logged additions again and records, at every selection and merge call, the
    list GEPA samples from and the merge candidates. flip = (k, j, v): program k scores v on
    val example j. With check=True the rebuilt state is compared with the log."""
    subs = {0: list(s0)}
    for i in progs:
        if i > 0:
            subs[i] = [to01(v) for v in progs[i]["subscores"]]
    if flip is not None:
        k, j, v = flip
        subs[k] = list(subs[k])
        subs[k][j] = v
    scores = [pct(sum(subs[0]), val_n)]
    fscore = list(subs[0])
    fronts = [{0} for _ in range(val_n)]
    rec = dict(select={}, merge={}, final=None, problems=[])
    if check and scores[0] != progs[0]["score"]:
        rec["problems"].append("base program: rebuilt score %s, logged %s" % (scores[0], progs[0]["score"]))
    for what, x in timeline:
        if what == "add":
            s = subs[x]
            for j in range(val_n):
                if s[j] > fscore[j]:
                    fscore[j] = s[j]
                    fronts[j] = {x}
                elif s[j] == fscore[j]:
                    fronts[j].add(x)
            scores.append(pct(sum(s), val_n))
            assert len(scores) == x + 1, "programs not added in order"
            if check:
                if scores[x] != progs[x]["score"]:
                    rec["problems"].append("program %d: rebuilt score %s, logged %s" % (x, scores[x], progs[x]["score"]))
                if [set(f) for f in fronts] != [set(f) for f in progs[x]["fronts"]]:
                    bad = sum(1 for a, b in zip(fronts, progs[x]["fronts"]) if set(a) != set(b))
                    rec["problems"].append("after program %d: rebuilt front differs at %d examples" % (x, bad))
                if progs[x].get("best") is not None and idxmax(scores) != progs[x]["best"]:
                    rec["problems"].append("after program %d: best %d, logged %d" % (x, idxmax(scores), progs[x]["best"]))
        elif what == "select":
            lst = sampling_list(fronts, scores)
            rec["select"][x] = lst
            if check:
                sel = sels[x]
                if sel["prog"] not in lst:
                    rec["problems"].append("iteration %d: selected program %d not in the rebuilt list" % (sel["iter"], sel["prog"]))
                if scores[sel["prog"]] != sel["base_score"]:
                    rec["problems"].append("iteration %d: score of program %d %s, logged %s" % (sel["iter"], sel["prog"], scores[sel["prog"]], sel["base_score"]))
        elif what == "merge":
            cands = merge_candidates(fronts, scores)
            rec["merge"][x] = (cands, len(scores))
            mg = merges[x]
            if check and mg["found"] and not (mg["a"] in cands and mg["b"] in cands):
                rec["problems"].append("iteration %d: merged %d and %d, not both in the rebuilt candidates" % (mg["iter"], mg["a"], mg["b"]))
    rec["final"] = idxmax(scores)
    rec["scores"] = scores
    return rec


def ancestors(progs, i):
    out, todo = set(), [i]
    while todo:
        for p in progs[todo.pop()]["parents"]:
            if p is not None and p not in out:
                out.add(p)
                todo.append(p)
    return out


def instruction_maps(progs, preds):
    """Instructions of every program per predictor: ('seed', p), ('text', ...) or, when a
    merge broke a tie at random, ('either', a, b)."""
    maps = {0: {p: ("seed", p) for p in preds}}
    for i in sorted(progs):
        if i == 0:
            continue
        pr = progs[i]
        if pr["kind"] == "child":
            m = dict(maps[pr["parents"][0]])
            m[pr["pred"]] = ("text", pr["instr"])
        else:
            m = merged_map(maps, progs, pr["parents"][0], pr["parents"][1], pr["anc"], preds)
        maps[i] = m
    return maps


def merged_map(maps, progs, a, b, c, preds):
    # sample_and_attempt_merge_programs_by_common_predictors (merge_programs.py)
    m = {}
    for p in preds:
        ia, ib, ic = maps[a][p], maps[b][p], maps[c][p]
        if (ic == ia or ic == ib) and ia != ib:
            m[p] = ib if ic == ia else ia
        elif ic != ia and ic != ib:
            sa, sb = progs[a]["score"], progs[b]["score"]
            m[p] = ia if sa > sb else (ib if sb > sa else ("either", ia, ib))
        else:
            m[p] = ia
    return m


def same_up_to(m1, m2, preds):
    for p in preds:
        a, b = m1[p], m2[p]
        aa = a[1:] if a[0] == "either" else (a,)
        bb = b[1:] if b[0] == "either" else (b,)
        if not any(u == v for u in aa for v in bb):
            return False
    return True


# ---------------------------------------------------------------- one run

# split of the examples an evaluation uses; CERTAIN = evaluations that use every example of it
SPLIT = {"base val": "val", "val": "val", "merge subsample": "val",
         "parent minibatch": "train", "child minibatch": "train", "test": "test"}
CERTAIN = {"base val", "val", "test"}


def load_splits(path, bench, seed):
    data = json.load(open(path, encoding="utf-8"))[bench][seed]
    where = {}
    for split in ("train", "val", "test"):
        for i, v in enumerate(data[split]):
            where.setdefault(v[:60], []).append((split, i))
    return where


def example_key(first, bench):
    key = "id" if bench == "HotpotQABench" else "claim"
    m = re.search(r"'%s': (['\"])(.*?)(?<!\\)\1" % key, first)
    if not m:
        return None
    return m.group(2).replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def eval_program(e, maps, progs, preds):
    """Instructions of the program an evaluation ran."""
    if e["kind"] == "base val":
        return maps[0]
    if e["kind"] == "val":
        return maps[e["prog"]] if "prog" in e else None
    if e["kind"] == "merge subsample":
        return merged_map(maps, progs, e["a"], e["b"], e["anc"], preds)
    if e["kind"] == "parent minibatch":
        return maps[e["parent"]]
    if e["kind"] == "child minibatch":
        m = dict(maps[e["parent"]])
        m[e["pred"]] = ("text", e["instr"])
        return m
    return None


def analyse_run(run_dir, splits_path):
    name = os.path.basename(os.path.normpath(run_dir))
    seed_dir = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    bench = name.split("_")[0]
    out = dict(run=seed_dir + "/" + name, problems=[], losses=[], checks={})
    where = load_splits(splits_path, bench, seed_dir.split("_")[1])
    stdout = read_text(run_dir, "run_log.txt")
    evs = stderr_lines(read_text(run_dir, "run_log_stderr.txt"))
    obs, trailing = observed_evals(evs)
    val_n = obs[0]["n"]
    msgs = stdout_messages(stdout)
    expected, progs, sels, merges, other, timeline = parse_search(msgs, val_n)
    out["problems"] += align(expected, obs)
    out["checks"].update(evaluations=len(obs), programs=len(progs), selections=len(sels),
                         merge_calls=len(merges), merges=sum(1 for mg in merges if mg["found"]),
                         other=[dict(iter=o["iter"], kind=o["kind"], text=o["text"][:200]) for o in other],
                         error_lines=sum(len(o["errors"]) for o in obs) + len(trailing),
                         error_lines_after_last_evaluation=len(trailing))
    preds = sorted({e["pred"] for e in expected if e.get("pred")})
    order = MODULE_ORDER[bench]
    ctx = dict(expected=expected, progs=progs, sels=sels, merges=merges, timeline=timeline, val_n=val_n,
               preds=preds, order=order, maps=instruction_maps(progs, preds))
    ctx["s0"] = base_subscores(progs, val_n)
    ctx["rec_a"] = replay(ctx["s0"], progs, timeline, sels, merges, val_n, check=True) if ctx["s0"] is not None else None
    if ctx["rec_a"] is not None:
        out["problems"] += ctx["rec_a"]["problems"]
        out["checks"]["final_program"] = ctx["rec_a"]["final"]

    # a lost reflection call is logged by GEPA itself, not by DSPy's evaluation
    for o in other:
        if "InternalServerError" in o["text"]:
            out["losses"].append(dict(time=None, iter=o["iter"], step=o["kind"], verdict="CC",
                                      readings=[dict(reading="GEPA message", verdict="CC",
                                                     why="no candidate in this iteration" if o["kind"] == "proposal exception"
                                                     else "the iteration stopped part-way")]))
    for k, x in enumerate(expected):
        o = x.get("obs")
        if o is None:
            continue
        for err in o["errors"]:
            if err["lost"]:
                out["losses"].append(judge_loss(err, k, x, bench, where, ctx))
    for err in trailing:
        if err["lost"]:
            out["problems"].append("a lost request after the last evaluation: %s" % err["time"])
    return out


def judge_loss(err, k, x, bench, where, ctx):
    key = example_key(err["first"], bench)
    places = where.get(key[:60], []) if key else []
    loss = dict(time=err["time"], iter=x["iter"], in_progress=x["kind"], example=err["example"],
                places=places, module=err["module"], message=err["message"],
                started=ctx["expected"][k - 1]["obs"]["time"] if k > 0 else None, ended=x["obs"]["time"])
    readings = []
    if len(places) != 1:
        readings.append(dict(reading="-", verdict="CC", why="example found %d times in the splits" % len(places)))
    else:
        split, idx = places[0]
        loss["split"], loss["idx"] = split, idx
        mod = err["module"] or ctx["order"][0]
        loss["module_used"] = mod + ("" if err["module"] else " (not in the log; first module assumed)")
        if SPLIT[x["kind"]] == split:
            readings.append(judge_in_e(k, x, idx, mod, ctx))
        if SPLIT[x["kind"]] != split or x["kind"] not in CERTAIN:
            readings.append(judge_left_behind(k, split, mod, ctx))
        elif x["kind"] != "test":
            # not part of the rule (E contains the example): reported only
            extra = judge_left_behind(k, split, mod, ctx)
            extra["reading"] += " (extra, not in the rule)"
            loss["extra"] = extra
    loss["readings"] = readings
    verdicts = [r["verdict"] for r in readings]
    loss["verdict"] = "CC" if "CC" in verdicts else ("test" if "test" in verdicts else "NC")
    return loss


def judge_in_e(k, x, idx, mod, ctx):
    """The lost copy belonged to E: the example's score in E could have been the other value."""
    expected, progs = ctx["expected"], ctx["progs"]
    kind = x["kind"]
    r = dict(reading="in %s of iteration %s" % (kind, x["iter"]))
    if kind == "test":
        r.update(verdict="test", why="no search effect; the example counts as wrong if it has no other answer")
        return r
    if kind == "parent minibatch":
        r.update(verdict="CC", why="the parent's minibatch gives the reflection input and the score the child must beat")
        return r
    if kind == "child minibatch":
        parent_eval = expected[k - 1]
        assert parent_eval["kind"] == "parent minibatch" and parent_eval["iter"] == x["iter"]
        p, c = parent_eval["obs"]["a"], x["score"]
        flips = (c - 1 <= p) if x["kept"] else (c + 1 > p)
        r["sums"] = "child %s, parent %s, child %s" % (c, p, "kept" if x["kept"] else "rejected")
        if flips:
            r.update(verdict="CC", why="one point %s for the child flips the decision" % ("less" if x["kept"] else "more"))
            return r
        hits = cache_hits(eval_program(x, ctx["maps"], progs, ctx["preds"]), mod, "train", k + 1, ctx)
    elif kind == "merge subsample":
        m, top = x["score"], max(x["p1"], x["p2"])
        flips = (m - 1 < top) if x["kept"] else (m + 1 >= top)
        r["sums"] = "merged %s, parents %s and %s, merge %s" % (m, x["p1"], x["p2"], "kept" if x["kept"] else "rejected")
        if flips:
            r.update(verdict="CC", why="one point %s for the merged program flips the decision" % ("less" if x["kept"] else "more"))
            return r
        hits = cache_hits(eval_program(x, ctx["maps"], progs, ctx["preds"]), mod, "val", k + 1, ctx)
    else:
        kprog = x["prog"]
        if ctx["rec_a"] is None or ctx["rec_a"]["problems"]:
            r.update(verdict="CC", why="the rebuild of the search does not match the log")
            return r
        logged = progs[kprog]["subscores"][idx] if kprog > 0 else ctx["s0"][idx]
        other_value = 0 if logged is True or (kprog == 0 and logged == 1) else 1
        r["sums"] = "program %d, val score %s; example %d logged %r, other value %d" % (kprog, x["score"], idx, logged, other_value)
        effect = flip_effect(kprog, idx, other_value, ctx)
        r["context"] = flip_context(kprog, idx, other_value, ctx)
        if effect:
            r.update(verdict="CC", why=effect)
            return r
        hits = cache_hits(ctx["maps"][kprog], mod, "val", k + 1, ctx)
    if hits:
        r.update(verdict="CC", why="decision unchanged, but a later identical request: " + summary(hits))
    else:
        r.update(verdict="NC", why="decision unchanged and no later identical request")
    return r


def judge_left_behind(k, split, mod, ctx):
    """The lost copy was left behind by an earlier evaluation of the same split: no result
    changed, only a later identical request could differ."""
    expected = ctx["expected"]
    r = dict(reading="left behind by an earlier %s evaluation" % split)
    found = []
    for e in expected[:k]:
        if SPLIT.get(e["kind"]) != split:
            continue
        prog = eval_program(e, ctx["maps"], ctx["progs"], ctx["preds"])
        if prog is None:
            continue
        hits = cache_hits(prog, mod, split, k, ctx)
        if hits:
            found.append("from %s of iteration %s: %s" % (e["kind"], e["iter"], summary(hits, 2)))
    if found:
        r.update(verdict="CC", why="%d earlier evaluations with a later identical request, e.g. %s" % (len(found), found[-1]))
    else:
        r.update(verdict="NC", why="no earlier evaluation of this split has a later identical request")
    return r


def flip_effect(kprog, j, value, ctx):
    rec_a = ctx["rec_a"]
    rec_b = replay(ctx["s0"], ctx["progs"], ctx["timeline"], ctx["sels"], ctx["merges"], ctx["val_n"], flip=(kprog, j, value))
    added = kprog == 0
    for what, x in ctx["timeline"]:
        if what == "add" and x == kprog:
            added = True
            continue
        if not added:
            continue
        if what == "select" and rec_a["select"][x] != rec_b["select"][x]:
            fa, fb = Counter(rec_a["select"][x]), Counter(rec_b["select"][x])
            diff = ["program %d %d -> %d" % (p, fa[p], fb[p]) for p in sorted(set(fa) | set(fb)) if fa[p] != fb[p]]
            return "the list sampled at iteration %d changes (%s)" % (
                ctx["sels"][x]["iter"], "entries: " + ", ".join(diff[:4]) if diff else "order only")
        if what == "merge":
            ca, _ = rec_a["merge"][x]
            cb, _ = rec_b["merge"][x]
            if ca != cb:
                return "the merge candidates at iteration %d change" % ctx["merges"][x]["iter"]
            inv = (set(ca) | set().union(*[ancestors(ctx["progs"], c) for c in ca])) if ca else set()
            if kprog in inv:
                return "program %d takes part in the merge call at iteration %d" % (kprog, ctx["merges"][x]["iter"])
    if rec_a["final"] != rec_b["final"]:
        return "the final program changes from %d to %d" % (rec_a["final"], rec_b["final"])
    return None


def flip_context(kprog, j, value, ctx):
    """Size of the change, for the report only (does not enter the verdict): how many
    selections and merge calls came after program k, at the first changed selection the
    share of list positions that hold another program, and k's val score against the final
    program's."""
    rec_a = ctx["rec_a"]
    rec_b = replay(ctx["s0"], ctx["progs"], ctx["timeline"], ctx["sels"], ctx["merges"], ctx["val_n"], flip=(kprog, j, value))
    added = kprog == 0
    n_sel = n_merge = 0
    first = None
    in_merge = []
    for what, x in ctx["timeline"]:
        if what == "add" and x == kprog:
            added = True
            continue
        if not added:
            continue
        if what == "select":
            n_sel += 1
            la, lb = rec_a["select"][x], rec_b["select"][x]
            if first is None and la != lb:
                n = max(len(la), len(lb))
                mism = sum(1 for i in range(n) if i >= len(la) or i >= len(lb) or la[i] != lb[i])
                first = "first changed selection at iteration %d: %d of %d list positions hold another program" % (
                    ctx["sels"][x]["iter"], mism, n)
        elif what == "merge":
            n_merge += 1
            mg = ctx["merges"][x]
            if mg["found"] and kprog in (mg["a"], mg["b"], mg["anc"]):
                in_merge.append("iteration %d merged %d and %d via %d" % (mg["iter"], mg["a"], mg["b"], mg["anc"]))
    final = rec_a["final"]
    parts = ["%d selections and %d merge calls after program %d" % (n_sel, n_merge, kprog)]
    if first:
        parts.append(first)
    if in_merge:
        parts.append("program %d is in the merges of %s" % (kprog, "; ".join(in_merge)))
    parts.append("program %d val score %s, with the other value %s; final program %d with %s" % (
        kprog, rec_a["scores"][kprog], rec_b["scores"][kprog], final, rec_a["scores"][final]))
    return "; ".join(parts)


def cache_hits(program, mod, split, start, ctx):
    """Evaluations from index start on, on the same split, whose program has the same
    instructions as `program` for module `mod` and every module before it: their request
    for the same example would have been answered from the cache in a run without the loss
    (which examples a minibatch or merge subsample used is not read, so they all count)."""
    order = ctx["order"]
    upto_mods = order[:order.index(mod) + 1]
    upto = [p for p in ctx["preds"] if p.split(".")[0] in upto_mods]
    hits = []
    for e in ctx["expected"][start:]:
        if SPLIT.get(e["kind"]) != split:
            continue
        q = eval_program(e, ctx["maps"], ctx["progs"], ctx["preds"])
        if q is not None and same_up_to(program, q, upto):
            hits.append("%s of iteration %s" % (e["kind"], e["iter"]))
    return hits


def summary(hits, n=3):
    return "; ".join(hits[:n]) + (" (and %d more)" % (len(hits) - n) if len(hits) > n else "")


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args()
    results = [analyse_run(r, args.splits) for r in args.runs]
    lines = []
    for r in results:
        c = r["checks"]
        lines.append("### %s" % r["run"])
        lines.append("")
        lines.append("- evaluations %d, programs %d, selections %d, merge calls %d (%d merges), final program %s; "
                     "error lines %d (%d after the last evaluation)" % (
                         c["evaluations"], c["programs"], c["selections"], c["merge_calls"], c["merges"],
                         c.get("final_program"), c["error_lines"], c["error_lines_after_last_evaluation"]))
        lines.append("- check of the rebuild against the log: %s" % ("all match" if not r["problems"] else "; ".join(r["problems"])))
        if c["other"]:
            lines.append("- other GEPA messages: %s" % c["other"])
        lines.append("- lost requests: %d" % len(r["losses"]))
        lines.append("")
        for n, l in enumerate(r["losses"], 1):
            lines.append("%d. **%s**: error %s, iteration %s, evaluation in progress: %s (started %s, ended %s); example %s, %s; module %s" % (
                n, l["verdict"], l.get("time"), l.get("iter"), l.get("in_progress"), l.get("started"), l.get("ended"),
                ", ".join("%s %d" % p for p in l.get("places", [])) or "?", l.get("example", ""), l.get("module_used", l.get("module"))))
            for rd in l["readings"] + ([l["extra"]] if l.get("extra") else []):
                lines.append("   - %s: **%s**%s; %s" % (rd["reading"], rd["verdict"], (" (" + rd["sums"] + ")") if rd.get("sums") else "", rd["why"]))
                if rd.get("context"):
                    lines.append("     - size (not part of the verdict): %s" % rd["context"])
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=str)
    if any(r["problems"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
