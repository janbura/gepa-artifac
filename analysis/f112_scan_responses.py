# F112: find the Qwen answers that contain a long run of one character, in the Arbor response log of a job
# (inference_responses.jsonl.gz), with the time each answer was created (CET).
#   python analysis/f112_scan_responses.py <inference_responses.jsonl.gz> [shortest run to list, default 10000]
# Standard library only. The log is read as data; nothing in it is run.
import sys, gzip, json, datetime, itertools

CET = datetime.timezone(datetime.timedelta(hours=1))
path = sys.argv[1]
min_run = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

answers = 0
longest_overall = 0
hits = []
for line in gzip.open(path, "rt", encoding="utf-8", errors="replace"):
    if not line.strip():
        continue
    r = json.loads(line)
    resp = r["response"]
    choice = resp["choices"][0]
    text = choice["message"]["content"] or ""
    answers += 1
    run, char, start, pos = 0, "", 0, 0
    for c, group in itertools.groupby(text):
        k = sum(1 for _ in group)
        if k > run:
            run, char, start = k, c, pos
        pos += k
    longest_overall = max(longest_overall, run)
    if run >= min_run:
        t = datetime.datetime.fromtimestamp(resp.get("created", 0), CET).strftime("%Y-%m-%d %H:%M:%S")
        before = text[max(0, start - 60):start].replace("\n", "\\n")
        hits.append((t, r.get("id", "")[:8], char, run, len(text), choice.get("finish_reason"), before))

print(f"{path}: {answers} answers, longest run of one character {longest_overall:,}")
print(f"answers with a run of at least {min_run:,} copies of one character: {len(hits)}")
for t, rid, char, run, length, finish, before in sorted(hits):
    print(f"  {t} id {rid} {char!r} x {run:,} (answer {length:,} characters, finish {finish}); before: ...{before}")
