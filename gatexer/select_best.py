"""Pick, per preset, the (seq, d) with the lowest VALIDATION score summed over the tuning datasets -> results/tune/best.json"""
import glob, json, os, re, sys, collections
out = sys.argv[1] if len(sys.argv) > 1 else "runs"
score = collections.defaultdict(lambda: collections.defaultdict(float)); cnt = collections.defaultdict(lambda: collections.defaultdict(int))
for f in glob.glob(f"{out}/tune/*/*.json"):
    if f.endswith("best.json"): continue
    m = re.match(r"(.+)_seq(\d+)_d(\d+)\.json", os.path.basename(f))
    if not m: continue
    j = json.load(open(f)); v = j["extra"].get("val_best")
    if v is None: continue
    key = (int(m.group(2)), int(m.group(3))); score[m.group(1)][key] += v; cnt[m.group(1)][key] += 1
best = {}
for p, d in score.items():
    full = {k: s / cnt[p][k] for k, s in d.items() if cnt[p][k] == max(cnt[p].values())}
    k = min(full, key=full.get); best[p] = {"seq": k[0], "d": k[1], "val": round(full[k], 5)}
json.dump(best, open(f"{out}/tune/best.json", "w"), indent=1); print(json.dumps(best, indent=1))
