"""Print the paper's result tables as Markdown from the bootstrap files in results/.
    python analysis/make_tables.py > docs/results_tables.md
Gain = 1 - nMAE(GateXer) / nMAE(reference), as a percentage, on 3-seed ensembles; [lo, hi] = 95% paired-bootstrap interval."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
NAME = {"eia_ERCO": "ERCOT", "eia_CISO": "CAISO", "eia_SWPP": "SWPP", "eia_MISO": "MISO", "opsd_DE": "Germany", "opsd_ES": "Spain",
        "opsd_FR": "France", "opsd_DK": "Denmark", "opsd_PT": "Portugal", "opsd_AT": "Austria", "opsd_GR": "Greece", "grid_india": "India"}
ORDER = ["eia_ERCO", "eia_CISO", "eia_SWPP", "eia_MISO", "opsd_DE", "opsd_ES", "opsd_FR", "opsd_DK", "opsd_PT", "opsd_AT", "opsd_GR", "grid_india"]
PRETTY = {"timexer": "TimeXer", "itransformer": "iTransformer", "patchtst": "PatchTST"}
load = lambda f: {j["grid"]: j for j in map(json.loads, open(os.path.join(R, f)))}
pc = lambda v: f"{100 * v:.1f}"

def main_table(file, title, grids):
    b = load(file)
    print(f"### {title}\n")
    print("| Grid | GateXer nMAE | Strongest baseline | Its nMAE | Gain % | 95% CI |\n|---|---|---|---|---|---|")
    gains = []
    for g in grids:
        if g not in b: continue
        j = b[g]; base = j["base"]; gl = j[f"gain_vs_{base}"]
        print(f"| {NAME[g]} | {j['ens']['gx']:.4f} | {PRETTY.get(base, base)} | {j['ens'][base]:.4f} | {pc(gl[0])} | [{pc(gl[1])}, {pc(gl[2])}] |")
        if g != "grid_india": gains.append(gl[0])
    return b, gains

b0, gains = main_table("boot_snb0.jsonl", "Plain protocol (3-seed ensembles)", ORDER)
print(f"\nMean gain over the 11 main grids: {pc(sum(gains) / len(gains))}% (range {pc(min(gains))} to {pc(max(gains))}).\n")
tg = [[b0[g][f"pt_gain_vs_{b0[g]['base']}"][k] for g in ORDER if g in b0 and g != "grid_india"] for k in range(3)]
print("Mean per-target gain over the strongest baseline, 11 grids: " + ", ".join(f"{n} {pc(sum(v) / len(v))}%" for n, v in zip(("load", "solar", "wind"), tg)) + "\n")
b1, _ = main_table("boot_snb1.jsonl", "Corrected protocol (every model predicts a correction on yesterday; GateXer vs corrected TimeXer)", ORDER)
print("\n### Which kind of sharing helps? (gain over the no-sharing control, 3-seed ensembles)\n")
ab = load("boot_ablate4.jsonl")
print("| Grid | Control | Full GateXer | Sources only | Weather only |\n|---|---|---|---|---|")
for g in ["eia_ERCO", "eia_CISO", "eia_SWPP", "eia_MISO", "opsd_PT", "opsd_DE"]:
    if g in ab:
        j = ab[g]; c = j["ctrl"]
        print(f"| {NAME[g]} | {'GateXer without sharing' if c == 'nosh' else 'ci (no sharing, no FAMO, no trend split)'} | "
              f"{pc(j['gx_over_' + c][0])} | {pc(j['srconly_over_' + c][0])} | {pc(j['exonly_over_' + c][0])} |")
