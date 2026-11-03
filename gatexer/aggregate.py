"""Collect results/**.json (+ chronos / blend npz) into markdown + CSV tables.   python aggregate.py results
Metrics: nMAE (MAE / mean actual) per target, net-load MAE, ramp MAE; mean +- std over seeds."""
import glob, json, os, sys, collections, numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = collections.defaultdict(list)
for f in glob.glob(f"{out}/main/*/*.json") + glob.glob(f"{out}/ablate/*/*.json"):
    parts = os.path.normpath(f).split(os.sep); ds, name, stage = parts[-2], parts[-1][:-5], parts[-3]   # normpath: Windows mixes / and \\
    preset = name.rsplit("_s", 1)[0]; m = json.load(open(f))["metrics"]
    rows[(stage, ds, preset)].append(dict(load=m["nmae"][0], solar=m["nmae"][1], wind=m["nmae"][2], avg=float(np.mean(m["nmae"])), net=m.get("net_mae", np.nan),
                                         ramp=float(np.mean(m["ramp_mae"])), h1=float(np.mean(m["nmae_by_h"]["1"])) if "nmae_by_h" in m else np.nan,
                                         h24=float(np.mean(m["nmae_by_h"]["24"])) if "nmae_by_h" in m else np.nan))
def ms(v): v = np.array(v, dtype=float); return f"{np.nanmean(v):.4f}" + (f"+-{np.nanstd(v):.4f}" if len(v) > 1 else "")
lines = []; csv = ["stage,dataset,preset,n_seeds,load,solar,wind,avg_nmae,net_mae,ramp_mae,h1,h24"]
for stage in ("main", "ablate"):
    ds_all = sorted({k[1] for k in rows if k[0] == stage})
    for ds in ds_all:
        lines += [f"\n### {stage}: {ds}", "| model | seeds | load | solar | wind | avg nMAE | net-load MAE | ramp MAE | 1h | 24h |", "|---|---|---|---|---|---|---|---|---|---|"]
        items = sorted([(k[2], v) for k, v in rows.items() if k[0] == stage and k[1] == ds], key=lambda x: np.mean([r["avg"] for r in x[1]]))
        for p, v in items:
            g = lambda key: [r[key] for r in v]
            lines.append(f"| {p} | {len(v)} | {ms(g('load'))} | {ms(g('solar'))} | {ms(g('wind'))} | {ms(g('avg'))} | {ms(g('net'))} | {ms(g('ramp'))} | {ms(g('h1'))} | {ms(g('h24'))} |")
            csv.append(",".join(map(str, [stage, ds, p, len(v), *[np.nanmean(g(k)) for k in ("load", "solar", "wind", "avg", "net", "ramp", "h1", "h24")]])))
open(f"{out}/summary.md", "w").write("\n".join(lines)); open(f"{out}/summary.csv", "w").write("\n".join(csv))
print("\n".join(lines) if lines else "no results yet")
