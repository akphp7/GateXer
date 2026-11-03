"""Chronos-2 zero-shot baseline on the SAME windows as train.py (needs: pip install chronos-forecasting, GPU/HF access).
Exploratory baseline: it is NOT part of the comparison reported in the paper. Uses the documented Chronos2Pipeline.predict_df API.
All other columns (other sources + weather) are given as PAST covariates only, so the comparison with GateXer is fair."""
import argparse, time, numpy as np, pandas as pd, torch
T0 = time.time()
from chronos import Chronos2Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--date", default="date")
ap.add_argument("--targets", default="load,solar,wind"); ap.add_argument("--exog", default="")
ap.add_argument("--seq", type=int, default=168); ap.add_argument("--pred", type=int, default=24)
ap.add_argument("--ctx", type=int, default=512, help="Chronos context length in hours (use --seq for a like-for-like 168h run)")
ap.add_argument("--no_cov", action="store_true", help="univariate Chronos-2 (no covariates)")
ap.add_argument("--stride", type=int, default=1, help="use every k-th window (speed); window indices are saved so blends can align"); ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="amazon/chronos-2"); ap.add_argument("--start", default=""); ap.add_argument("--end", default="")
a = ap.parse_args()
df = pd.read_csv(a.csv); df[a.date] = pd.to_datetime(df[a.date]).dt.tz_localize(None)
if a.start: df = df[df[a.date] >= pd.Timestamp(a.start)]
if a.end: df = df[df[a.date] <= pd.Timestamp(a.end)]
df = df.reset_index(drop=True)
tg = a.targets.split(","); ex = [c for c in a.exog.split(",") if c]; S = len(tg); N = len(df); L, H = a.seq, a.pred
n_tr, n_va = int(N * 0.7), int(N * 0.1)
win = lambda s, e: s + np.arange(e - s - L - H + 1)          # identical to train.py
va, te = win(n_tr - L, n_tr + n_va), win(n_tr + n_va - L, N)
va, te = va[::a.stride], te[::a.stride]
pipe = Chronos2Pipeline.from_pretrained(a.model, device_map="cuda" if torch.cuda.is_available() else "cpu")

def forecast(idx):
    P = np.zeros((len(idx), H, S), dtype="float32")
    for si, tname in enumerate(tg):
        covs = [] if a.no_cov else [c for c in tg + ex if c != tname]
        for c0 in range(0, len(idx), a.chunk):
            rows = []
            for k, i in enumerate(idx[c0:c0 + a.chunk]):
                origin = i + L                                   # first forecast step
                lo = max(0, origin - a.ctx)
                d = df.iloc[lo:origin][[a.date, tname] + covs].copy()
                d.columns = ["timestamp", "target"] + covs
                d.insert(0, "id", f"{tname}_{c0 + k}")
                rows.append(d)
            ctx = pd.concat(rows, ignore_index=True)
            out = pipe.predict_df(ctx, prediction_length=H, quantile_levels=[0.5], id_column="id",
                                  timestamp_column="timestamp", target="target")
            for k in range(len(rows)):
                P[c0 + k, :, si] = out[out["id"] == f"{tname}_{c0 + k}"].sort_values("timestamp")["predictions"].values[:H]
        print("done", tname, "at", round(time.time() - T0), "s", flush=True)
    return P

Y = lambda idx: np.stack([df[tg].values[i + L: i + L + H] for i in idx]).astype("float32")
np.savez(a.out, val_P=forecast(va), val_Y=Y(va), test_P=forecast(te), test_Y=Y(te), te_idx=te, va_idx=va, seq=L)
print("saved", a.out)
