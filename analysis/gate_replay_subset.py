"""Gate replay inside the E1 search subset (zero compute), to fix the E2 tolerance before any run.

For every accepted reflective child of our 12 HotpotQA/HoVer GEPA and GEPA+Merge runs: compare child and parent on
the 50 gate examples, a fixed half of the 100-example search subset that E1 draws for that seed (same draw as
gepa.py: random.Random(f"search_val_subset:{seed}")). Cut the child if it trails by more than tol examples.
The searches behind these logs used all 300 val examples, so this shows what the gate would cut, not how an
E1 search would then go on.
Usage: python analysis/gate_replay_subset.py results/table1_h100 > analysis/outputs/gate_replay_subset.md
"""
import collections, glob, os, pickle, random, sys
import numpy as np

ROOT = sys.argv[1]
f = lambda x: float(x) if x is not None else 0.0
R = collections.defaultdict(lambda: collections.defaultdict(float))
for d in sorted(glob.glob(os.path.join(ROOT, "seed_*", "*_GEPA*_qwen3-8b"))):
    name = os.path.basename(d)
    b = {"HotpotQABench": "HotpotQA", "hoverBench": "HoVer"}.get(name.split("_")[0])
    if b is None: continue
    seed = int(d.split("seed_")[1][0])
    st = pickle.load(open(os.path.join(d, "gepa_state.bin"), "rb"))
    V = np.array([[f(s) for s in subs] for subs in st["prog_candidate_val_subscores"]])
    nv = V.shape[1]
    ids = list(range(nv)); random.Random(f"search_val_subset:{seed}").shuffle(ids)
    search, gate = sorted(ids[:100]), sorted(ids[:50])
    sub_scores, full = V[:, search].mean(1), V.mean(1)
    best_sub, best_full = int(np.argmax(sub_scores)), int(np.argmax(full))
    par = st["parent_program_for_candidate"]
    def ancestors(c):
        anc, stack = set(), [c]
        while stack:
            x = stack.pop()
            for p in par[x] or []:
                if p is not None and p not in anc: anc.add(p); stack.append(p)
        return anc
    keep_sub, keep_full = ancestors(best_sub) | {best_sub}, ancestors(best_full) | {best_full}
    top5 = set(np.argsort(-sub_scores, kind="stable")[:5].tolist())
    seen, acc = set(), []
    for x in st["full_program_trace"]:
        if id(x) in seen: continue
        seen.add(id(x))
        if "new_program_idx" in x and "selected_program_candidate" in x and "merged_entities" not in x:
            acc.append((x["new_program_idx"], x["selected_program_candidate"]))
    for tol in (0, 1, 2, 3, 4):
        r = R[(b, tol)]
        for c, p in acc:
            r["n"] += 1
            if V[c, gate].sum() - V[p, gate].sum() < -tol:
                r["cut"] += 1
                r["better_sub"] += sub_scores[c] > sub_scores[p]
                r["better_full"] += full[c] > full[p]
                r["lineage_sub"] += c in keep_sub
                r["lineage_full"] += c in keep_full
                r["top5"] += c in top5
            else:
                r["kept_worse_sub"] += sub_scores[c] < sub_scores[p]
        r["worse_sub"] += sum(sub_scores[c] < sub_scores[p] for c, p in acc)
print("| bench | tol | children | cut | search-val calls saved | cut but better on the 100 | cut but better on all 300 | cut in the lineage of the subset best | of the full-val best | cut that were top 5 on the subset | worse-on-100 children still passed |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (b, tol), r in sorted(R.items()):
    n, c = int(r["n"]), int(r["cut"])
    print(f"| {b} | {tol} | {n} | {c} ({c/n:.0%}) | {50*c/(100*n):.0%} | {int(r['better_sub'])} | {int(r['better_full'])} | {int(r['lineage_sub'])} | {int(r['lineage_full'])} | {int(r['top5'])} | {int(r['kept_worse_sub'])} of {int(r['worse_sub'])} |")
