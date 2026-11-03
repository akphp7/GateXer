"""Resumable experiment runner. Every run writes results/<stage>/<dataset>/<preset>_s<seed>.json; finished runs are skipped,
so it is safe to stop and restart (e.g. Kaggle session limits).
  python run_experiments.py --stage tune --data_dir data/final --out runs
  python run_experiments.py --stage main --data_dir data/final --out runs --seeds 0 1 2
  python run_experiments.py --stage ablate --data_dir data/final --out runs --seeds 0 1 2
  python run_experiments.py --stage chronos --data_dir data/final --out runs      (needs chronos-forecasting + HF access)
Add --dry to print the commands only. Add --epochs 3 --max_train_windows 2000 for a quick smoke run."""
import argparse, json, os, subprocess, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
PY = '"%s"' % sys.executable            # quote every path: Windows user names / folders may contain spaces
q = lambda p: '"%s"' % p
EXOG = {"opsd": "temp,rad_dir,rad_dif"}          # OPSD has its own weather; EIA / Grid-India get weather only with --weather_dir
EXOG_W = "temp,ws100,ghi,cloud"                  # columns written by add_weather.py
START = {"eia": "2021-07-01"}                    # EIA: crop to 2021-07+ (solar/wind capacity grew 10-100x since 2018)
CORE = ["opsd_DE", "opsd_ES", "eia_ERCO", "eia_CISO", "grid_india"]
MORE = ["opsd_FR", "opsd_DK", "opsd_PT", "opsd_AT", "opsd_GR", "eia_SWPP", "eia_MISO"]
TUNE_DATA = ["opsd_DE", "eia_ERCO"]
BASE = ["snaive", "dlinear", "patchtst", "itransformer", "timexer", "ci", "naive"]
GX = {  # name -> train.py args
    "gx": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0",          # final model of the ablation check
    "gx_net": "--preset full --decomp 25 --ple 0",                        # with net-load + ramp-weighted loss
    "gx_fm": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --fm 1",  # + calendar marks of the forecast hours
    "ci_dec": "--preset ci --decomp 25",
    "ci_dec_fm": "--preset ci --decomp 25 --fm 1",
    "gx_gateonly": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --weighting equal",
    "gx_ple": "--preset full --decomp 25 --ple 1",
    "gx_nogate": "--preset full --decomp 25 --ple 0 --cross always",
    "gx_nocross": "--preset full --decomp 25 --ple 0 --cross off",
    "gx_equalw": "--preset full --decomp 25 --ple 0 --weighting equal",
    "gx_nonet": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0",
    "gx_srconly": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --cross src",   # clean ablation: gated mixing between sources only (no weather)
    "gx_exonly": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --cross ex",     # clean ablation: gated weather mixing only (no source sharing)
    "gx_nosh": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --cross off",      # clean ablation: FAMO, no sharing
    "gx_always": "--preset full --decomp 25 --ple 0 --net 0 --ramp 0 --cross always",  # clean ablation: always-on sharing
    "gx_nodec": "--preset full --decomp 0 --ple 0 --net 0 --ramp 0",                    # clean ablation: no trend/remainder split
    "gx_nodecomp": "--preset full --decomp 0 --ple 0",
}
ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["tune", "main", "ablate", "chronos"])
ap.add_argument("--data_dir", default="data/final"); ap.add_argument("--out", default="runs")
ap.add_argument("--datasets", default=""); ap.add_argument("--presets", default=""); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--max_train_windows", type=int, default=0); ap.add_argument("--dry", action="store_true")
ap.add_argument("--cstride", type=int, default=1, help="chronos stage: use every k-th window")
ap.add_argument("--snb", type=int, default=0, help="1 = all trainable models predict a correction on top of yesterday (seasonal-naive base)")
ap.add_argument("--pc", type=int, default=0, help="1 = add wind power-curve feature for ALL models (datasets with ws100)")
ap.add_argument("--weather_dir", default="", help="folder written by add_weather.py: EIA and Grid-India are then read from here WITH weather columns")
a = ap.parse_args()
def kind(d): return d.split("_")[0]
def has_w(ds): return bool(a.weather_dir) and kind(ds) in ("eia", "grid")
def csv_path(ds): return f"{a.weather_dir}/{ds}.csv" if has_w(ds) else f"{a.data_dir}/{ds}.csv"
def exog_of(ds): return EXOG_W if has_w(ds) else EXOG.get(kind(ds), "")
def preset_args(p): return GX[p] if p in GX else f"--preset {p}"
def run(cmd, outjson):
    if os.path.exists(outjson): return
    os.makedirs(os.path.dirname(outjson), exist_ok=True)
    print(">>", cmd, flush=True)
    if not a.dry: subprocess.run(cmd, shell=True, check=False)
def cmd_train(ds, preset, seed, out, seq=168, d=64, extra=""):
    k = kind(ds); ex = exog_of(ds)
    c = f"{PY} {q(HERE + '/train.py')} --csv {q(csv_path(ds))} --targets load,solar,wind --seq {seq} --pred 24 --patch 24 --d {d} " \
        f"--epochs {a.epochs} --bs {a.bs} --seed {seed} {preset_args(preset)} {extra} --out {q(out)}"
    if ex: c += f" --exog {ex}"
    if k in START: c += f" --start {START[k]}"
    if a.pc: c += " --pc 1"
    if a.snb: c += " --snb 1"
    if a.max_train_windows: c += f" --max_train_windows {a.max_train_windows}"
    return c
if a.stage == "tune":
    ds_list = a.datasets.split(",") if a.datasets else TUNE_DATA
    presets = a.presets.split(",") if a.presets else ["patchtst", "itransformer", "timexer", "ci", "gx"]
    for ds, p, (seq, d) in itertools.product(ds_list, presets, [(168, 64), (168, 128), (336, 64), (336, 128)]):
        o = f"{a.out}/tune/{ds}/{p}_seq{seq}_d{d}.json"; run(cmd_train(ds, p, 0, o, seq, d), o)
elif a.stage in ("main", "ablate"):
    best = json.load(open(f"{a.out}/tune/best.json")) if os.path.exists(f"{a.out}/tune/best.json") else {}
    if a.stage == "main":
        ds_list = a.datasets.split(",") if a.datasets else CORE + MORE
        presets = a.presets.split(",") if a.presets else BASE + ["gx"]
    else:
        ds_list = a.datasets.split(",") if a.datasets else ["opsd_DE", "eia_ERCO"]
        presets = a.presets.split(",") if a.presets else [p for p in GX if p != "gx"] + ["gx"]
    for ds, p, s in itertools.product(ds_list, presets, a.seeds):
        cfg = best.get(p if p in best else ("gx" if p in GX else p), {"seq": 168, "d": 64})
        o = f"{a.out}/{a.stage}/{ds}/{p}_s{s}.json"; run(cmd_train(ds, p, s, o, cfg["seq"], cfg["d"], f"--save_preds {q(o[:-5] + '.npz')}"), o)
elif a.stage == "chronos":
    ds_list = a.datasets.split(",") if a.datasets else CORE
    for ds in ds_list:
        k = kind(ds); ex = exog_of(ds)
        for tag, flags in (("chronos2_cov", ""), ("chronos2_uni", "--no_cov")):
            if a.presets and tag not in a.presets.split(","): continue
            o = f"{a.out}/main/{ds}/{tag}.npz"
            c = f"{PY} {q(HERE + '/chronos_zs.py')} --csv {q(csv_path(ds))} --date date --targets load,solar,wind --seq 168 --pred 24 --ctx 512 --stride {a.cstride} {flags} --out {q(o)}"
            if ex: c += f" --exog {ex}"
            if k in START: c += f" --start {START[k]}"
            if not os.path.exists(o): os.makedirs(os.path.dirname(o), exist_ok=True); print(">>", c, flush=True); (None if a.dry else subprocess.run(c, shell=True))
