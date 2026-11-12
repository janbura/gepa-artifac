# Collects the authors' PUPA val outputs with the score their judge (GPT-4.1-mini) gave, for the judge
# check of plan section 2.10 (addendum of 2026-11-12). The authors' GEPA-family runs keep, for every val
# example, the outputs of the programs they tried:
#   generated_best_outputs_valset/task_<i>/iter_<k>_prog_<j>.json = [example, prediction, score]
# The same output in several files is kept once. Val examples only, no test example.
#   python make_authors_val_triples.py <authors' experiment_runs/seed_0 folder> <out.jsonl>
import glob
import json
import os
import sys

runs_dir, out = sys.argv[1], sys.argv[2]
runs = [
    "Papillon_PAPILLON_GEPA_qwen3-8b",
    "Papillon_PAPILLON_GEPA-MERGE_qwen3-8b",
    "Papillon_PAPILLON_Abl-SelectBestCandidate_qwen3-8b",
]
rows = {}
files = 0
conflicts = 0
for run in runs:
    for f in sorted(glob.glob(os.path.join(runs_dir, run, "generated_best_outputs_valset", "task_*", "*.json"))):
        example, pred, score = json.load(open(f, encoding="utf-8"))
        files += 1
        key = (example["user_query"], pred["llm_request"], pred["llm_response"], pred["response"])
        src = f"{run}/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
        if key in rows:
            # the authors' judge scored the same output again: keep every score, it shows how much
            # GPT-4.1-mini disagrees with itself
            if abs(rows[key]["authors_score"] - float(score)) > 1e-9:
                conflicts += 1
            rows[key]["authors_scores"].append(float(score))
            rows[key]["sources"].append(src)
            continue
        rows[key] = {
            "user_query": example["user_query"],
            "target_response": example["target_response"],
            "pii_str": example["pii_str"],
            "llm_request": pred["llm_request"],
            "llm_response": pred["llm_response"],
            "response": pred["response"],
            "authors_score": float(score),
            "authors_scores": [float(score)],
            "sources": [src],
        }

with open(out, "w", encoding="utf-8", newline="\n") as fo:
    for r in rows.values():
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

scores = [r["authors_score"] for r in rows.values()]
repeated = [r for r in rows.values() if len(r["authors_scores"]) > 1]
differ = [r for r in repeated if max(r["authors_scores"]) - min(r["authors_scores"]) > 1e-9]
print(f"files {files}, unique outputs {len(rows)}, scored more than once {len(repeated)}, of these with different scores "
      f"{len(differ)} ({len(differ) / max(1, len(repeated)):.0%}); later scores that differ from the first {conflicts}")
print(f"authors' scores: mean {sum(scores) / len(scores):.3f}, share 1.0 {sum(s == 1.0 for s in scores) / len(scores):.2f}, "
      f"val examples covered {len({r['user_query'] for r in rows.values()})}")
