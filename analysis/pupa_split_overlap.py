# pupa_split_overlap.py - which PUPA test examples also occur in train or val.
#
# The loader (gepa_artifact/benchmarks/papillon/papillon_data.py) takes the first 443 rows
# of Columbia-NLP/PUPA (pupa_new) in order: 111 train, 111 val, 221 test, no shuffle.
# The metric logger (gepa_artifact/utils/metric_logger.py, forward) looks an example up in
# train, then val, then test, and logs it under the first split that has an equal example,
# with the index of the first equal one. This script repeats that lookup with dspy.Example,
# so it shows which test examples end up in train.jsonl / val.jsonl and which share an index.
# For seeds 1 and 2 scripts/run_experiments.py (lines 204-211) shuffles train+val with
# random.Random(seed) before the run, so the lookup is done for each seed (test never changes).
#
# usage (on a machine with the HF cache of the runs, same python as the runs):
#   HF_DATASETS_OFFLINE=1 python -I pupa_split_overlap.py out.json

import json
import random
import sys
from collections import Counter

import dspy
from datasets import load_dataset

data = load_dataset("Columbia-NLP/PUPA", "pupa_new")["train"]
ex = [
    dspy.Example(
        {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
    ).with_inputs("user_query")
    for x in data
]
train, val, test = ex[:111], ex[111:222], ex[222:443]
assert len(train) == 111 and len(val) == 111 and len(test) == 221


def logged_as(e):
    # same order as MetricLogger.forward
    if e in train:
        return "train", train.index(e)
    if e in val:
        return "val", val.index(e)
    return "test", test.index(e)


q_train = set(e.user_query for e in train)
q_val = set(e.user_query for e in val)

rows = []
for j, e in enumerate(test):
    split, idx = logged_as(e)
    if split == "test":
        kind = "first copy in test" if idx == j else "repeat of an earlier test row"
    else:
        kind = "equal to a " + split + " row"
    rows.append({
        "test_pos": j,
        "logged_as": split,
        "idx": idx,
        "kind": kind,
        "query_in_train": e.user_query in q_train,
        "query_in_val": e.user_query in q_val,
    })

val_rows = []
for j, e in enumerate(val):
    in_train = e in train
    val_rows.append({"val_pos": j, "equal_to_train_row": in_train,
                     "first_copy_in_val": (not in_train) and val.index(e) == j,
                     "query_in_train": e.user_query in q_train})

kinds = Counter(r["kind"] for r in rows)
summary = {
    "dataset_rows": len(ex),
    "cache_files": [c["filename"] for c in data.cache_files],
    "fingerprint": data._fingerprint,
    "test_kinds": dict(kinds),
    "test_query_in_train_or_val": sum(1 for r in rows if r["query_in_train"] or r["query_in_val"]),
    "test_rows_logged_in_test_jsonl": sum(1 for r in rows if r["logged_as"] == "test"),
    "distinct_test_indices": len(set(r["idx"] for r in rows if r["logged_as"] == "test")),
    "val_equal_to_train_row": sum(1 for r in val_rows if r["equal_to_train_row"]),
    "val_repeats_inside_val": sum(1 for r in val_rows if not r["equal_to_train_row"] and not r["first_copy_in_val"]),
    "val_query_in_train": sum(1 for r in val_rows if r["query_in_train"]),
    "train_distinct_rows": len(set(json.dumps(e.toDict(), sort_keys=True) for e in train)),
}
for k, v in summary.items():
    if k != "cache_files":
        print(k, v)

# where each test example is logged in the runs of each seed
per_seed = {}
for seed in (0, 1, 2):
    tr, va = train, val
    if seed != 0:
        # same as scripts/run_experiments.py
        combined = train + val
        random.Random(seed).shuffle(combined)
        tr, va = combined[:111], combined[111:]
    out = []
    for j, e in enumerate(test):
        if e in tr:
            out.append(["train", tr.index(e)])
        elif e in va:
            out.append(["val", va.index(e)])
        else:
            out.append(["test", test.index(e)])
    per_seed[str(seed)] = out
    c = Counter(s for s, _ in out)
    print("seed", seed, "test examples logged as", dict(c))

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump({"summary": summary, "test": rows, "val": val_rows, "logged_per_seed": per_seed}, f, indent=1)
