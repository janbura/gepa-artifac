# how long did the authors Qwen MIPROv2 and GEPA runs take? first and last timestamp in run_log_stderr.txt,
# plus for MIPROv2 the time per minibatch trial and per full evaluation
#   python analysis/run_durations.py <authors experiment_runs/seed_0>
import re, sys, os, glob, datetime
root = sys.argv[1]
ts = re.compile(r"(20\d\d/\d\d/\d\d \d\d:\d\d:\d\d)")
def t(s):
    return datetime.datetime.strptime(s, "%Y/%m/%d %H:%M:%S")
for d in sorted(glob.glob(os.path.join(root, "*_qwen3-8b"))):
    name = os.path.basename(d)
    if "MIPROv2" not in name and "GEPA" not in name:
        continue
    f = os.path.join(d, "run_log_stderr.txt")
    if not os.path.exists(f):
        continue
    text = open(f, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    stamps = [t(m) for m in ts.findall(text)]
    if not stamps:
        print(name, "no timestamps"); continue
    total = (max(stamps) - min(stamps)).total_seconds() / 3600
    # gaps longer than 2 h mean a restart or a pause, report them
    s2 = sorted(stamps)
    gaps = [(s2[i], (s2[i + 1] - s2[i]).total_seconds() / 3600) for i in range(len(s2) - 1) if (s2[i + 1] - s2[i]).total_seconds() > 7200]
    line = f"{name}: {min(stamps)} -> {max(stamps)} = {total:.1f} h"
    if gaps:
        line += f"; gaps > 2 h: {[(str(a), round(g, 1)) for a, g in gaps]}"
    if "MIPROv2" in name:
        trials = [(int(m.group(2)), t(m.group(1)), m.group(3)) for m in re.finditer(r"(20\d\d/\d\d/\d\d \d\d:\d\d:\d\d) INFO dspy.teleprompt.mipro_optimizer_v2: =+ Trial (\d+) / \d+ - ([A-Za-z ]+)", text)]
        mini, full = [], []
        for (n1, t1, k1), (n2, t2, k2) in zip(trials, trials[1:]):
            dt = (t2 - t1).total_seconds() / 60
            (full if "Full" in k1 else mini).append(dt)
        if mini:
            line += f"; minibatch trial {sum(mini)/len(mini):.1f} min (n={len(mini)}), full eval {sum(full)/max(1,len(full)):.1f} min (n={len(full)})"
        st3 = re.search(r"(20\d\d/\d\d/\d\d \d\d:\d\d:\d\d) INFO dspy.teleprompt.mipro_optimizer_v2: ==> STEP 3", text)
        if st3:
            line += f"; steps 1-2 took {(t(st3.group(1)) - min(stamps)).total_seconds() / 60:.0f} min"
    print(line)
