# checks of table1_verdict.py on made-up runs whose results can be worked out by hand
#   python analysis/check_table1_verdict.py
import os, sys, math, tempfile, importlib.util

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tv", os.path.join(here, "table1_verdict.py"))
tv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv)

# 1. t distribution against printed tables
assert abs(tv.t_quantile(0.975, 7) - 2.364624) < 1e-5
assert abs(tv.t_quantile(0.975, 6) - 2.446912) < 1e-5
assert abs(tv.t_quantile(0.975, 2) - 4.302653) < 1e-5
assert abs(tv.t_two_sided_p(2.0, 7) - 0.085619) < 1e-5  # also checked by integrating the density
assert abs(tv.t_two_sided_p(1.0, 2) - (1 - 1 / math.sqrt(3))) < 1e-9  # exact for 2 df
assert abs(tv.t_two_sided_p(0.0, 7) - 1.0) < 1e-12
print("t distribution ok")

# 2. Holm: 4 p-values, alpha 0.05 -> thresholds 0.0125, 0.0167, 0.025, 0.05
assert tv.holm([0.01, 0.04, 0.03, 0.005]) == {0, 3}  # 0.005 and 0.01 pass, 0.03 > 0.025 stops
assert tv.holm([0.2, 0.001]) == {1}
assert tv.holm([0.03, 0.04]) == set()  # 0.03 > 0.025
print("Holm ok")

# 3. made-up runs: every cell at paper -1, 0, +1 (SD 1), with three exceptions
real_baselines = {"IFBench": [34.69, 33.33, 36.22], "AIME": [30.00, 31.33, 30.67], "LiveBench": [50.20, 48.93, 49.37]}
special = {("HoVer", "GEPA"): [30.0, 31.0, 32.0],                 # far below the paper's 52.33
           ("IFBench", "MIPROv2-Heavy"): [26.22, 36.22, 46.22]}   # wide spread, so s_pooled > SD_run
root = tempfile.mkdtemp()
for b in tv.BENCHES:
    for j, o in enumerate(tv.OPTS):
        xs = special.get((b, o)) or (real_baselines[b] if o == "Baseline" and b in real_baselines else [tv.PAPER[b][j] + k for k in (-1, 0, 1)])
        for s, x in enumerate(xs):
            d = os.path.join(root, f"seed_{s}", f"{tv.FOLDER[b]}_{o}_qwen3-8b", "evaluation_results")
            os.makedirs(d)
            open(os.path.join(d, "evaluation_result.txt"), "w").write(f"score,cost,input_tokens,output_tokens,\n{x},0,0,0,\n")
runs = tv.read_runs([root])
rows, complete = tv.verdicts(runs)
assert complete and len(rows) == 18
by = {(r["b"], r["name"]): r["res"] for r in rows}

# HoVer GEPA by hand: d = 31 - 52.33; s_pooled = 1 < SD_run = sqrt(1.69^2 + 3.22^2); t = d / (SD_run * sqrt(4/3))
sd_run = math.sqrt(1.69 ** 2 + 3.22 ** 2)
d, half, t, p, used = by[("HoVer", "GEPA")]
assert abs(d - (31 - 52.33)) < 1e-9 and abs(t - d / (sd_run * math.sqrt(4 / 3))) < 1e-9, (d, t)
assert abs(half - 2.364624 * sd_run * math.sqrt(4 / 3)) < 1e-4
print(f"HoVer GEPA: d {d:+.2f}, t {t:+.3f}, p {p:.5f} ({used})")

# IFBench MIPROv2: s_pooled = sqrt((100 + 1 + 1) / 3) beats SD_run = sqrt(1.63^2 + 3.22^2)
d, half, t, p, used = by[("IFBench", "MIPROv2")]
sp = math.sqrt(102 / 3)
assert abs(d) < 1e-9 and abs(half - 2.364624 * sp * math.sqrt(4 / 3)) < 1e-4 and "s_pooled 5.83" in used, (half, used)
print(f"IFBench MIPROv2: interval ±{half:.2f} ({used})")

# Baselines from our real runs: z = d / (SD_samp * sqrt(4/3)); the noise note printed z +0.73 (AIME) and +0.26 (LiveBench)
for b, z_note in (("AIME", 0.73), ("LiveBench", 0.26)):
    d, half, z, p, used = by[(b, "Baseline")]
    assert abs(z - z_note) < 0.006, (b, z)
d, half, z, p, used = by[("IFBench", "Baseline")]
assert abs(d - (sum(real_baselines["IFBench"]) / 3 - 36.90)) < 1e-9 and abs(z - d / (1.63 * math.sqrt(4 / 3))) < 1e-9
print(f"IFBench Baseline: d {d:+.2f}, z {z:+.2f}, p {p:.3f}")

# math pair: 6 runs, mean = paper, n = 6 in the formula
d, half, t, p, used = by[("AIME", "GEPA + GEPA+Merge (one target)")]
assert abs(d) < 1e-9 and abs(half - 2.364624 * math.sqrt(4.12 ** 2 + 3.22 ** 2) * math.sqrt(1 + 1 / 6)) < 1e-4
print("math pair ok")

# Holm over the 18 targets: only HoVer GEPA (p about 0.0014 < 0.05 / 18) is rejected
pvals = [r["res"][3] for r in rows]
rej = tv.holm(pvals)
assert [(rows[i]["b"], rows[i]["name"]) for i in rej] == [("HoVer", "GEPA")], rej
text = tv.report(runs)
assert text.count("inconsistent with the paper") == 1 and "provisional" not in text
print("Holm over 18 targets ok")

# 4. a missing run makes the whole table provisional
os.remove(os.path.join(root, "seed_2", f"{tv.FOLDER['AIME']}_GEPA_qwen3-8b", "evaluation_results", "evaluation_result.txt"))
text = tv.report(tv.read_runs([root]))
assert "provisional" in text and "inconsistent with the paper" not in text
print("missing run -> provisional ok")
print("all checks passed")
