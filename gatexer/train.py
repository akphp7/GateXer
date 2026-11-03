import argparse, json, time, numpy as np, pandas as pd, torch
from model import GateXer, DLinear
from official import Official
from losses import FAMO, task_losses

PRESETS = {  # name: (cross, ple, weighting, net, ramp)
    "ci":    ("off",    False, "equal", False, 0.0),   # channel-independent backbone (PatchTST-like), no sharing
    "naive": ("always", False, "equal", False, 0.0),   # naive joint (MT-Transformer-style shared mixing)
    "full":  ("gated",  True,  "famo",  True,  1.0),   # GateXer: gate + PLE + FAMO + net-load + ramp loss
    "patchtst":    ("off", False, "equal", False, 0.0),   # official TSLib baselines (equal-weight loss, no extras)
    "itransformer": ("off", False, "equal", False, 0.0),
    "timexer":     ("off", False, "equal", False, 0.0),
}
OFFICIAL = ("patchtst", "itransformer", "timexer")

def marks_from(dates):
    d = pd.DatetimeIndex(dates)
    f = lambda v, p: np.stack([np.sin(2*np.pi*v/p), np.cos(2*np.pi*v/p)], -1)
    return np.concatenate([f(d.hour.values, 24), f(d.dayofweek.values, 7), f(d.dayofyear.values, 366)], -1).astype("float32")

def windows(X, M, Y_idx, start, end, L, H):
    n = end - start - L - H + 1
    return start + np.arange(n)

def batch(X, M, idx, L, H, S):
    a = idx[:, None] + np.arange(L)[None]
    b = idx[:, None] + L + np.arange(H)[None]
    return (torch.from_numpy(X[a]), torch.from_numpy(M[a]), torch.from_numpy(X[b][..., :S]), torch.from_numpy(M[b]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--date", default="date")
    ap.add_argument("--targets", default="load,solar,wind"); ap.add_argument("--exog", default="")
    ap.add_argument("--seq", type=int, default=168); ap.add_argument("--pred", type=int, default=24)
    ap.add_argument("--patch", type=int, default=24)
    ap.add_argument("--preset", default="full", choices=list(PRESETS) + ["dlinear", "snaive"])
    ap.add_argument("--cross"); ap.add_argument("--ple", type=int); ap.add_argument("--weighting")
    ap.add_argument("--net", type=int); ap.add_argument("--ramp", type=float)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--max_train_windows", type=int, default=0)
    ap.add_argument("--start", default=""); ap.add_argument("--end", default="")
    ap.add_argument("--snb", type=int, default=0, help="1 = every model predicts a correction on top of yesterday (seasonal-naive base)"); ap.add_argument("--fm", type=int, default=0, help="1 = give GateXer-family models the calendar marks of the FORECAST hours (known in advance)")
    ap.add_argument("--pc", type=int, default=0, help="1 = add wind power-curve feature min(ws100,12)^3 when ws100 exists (all models)")
    ap.add_argument("--decomp", type=int, default=0, help="moving-average kernel for trend/seasonal split (0=off, e.g. 25)")
    ap.add_argument("--save_preds", default="", help="npz with val/test predictions and targets in raw units")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.csv)
    if a.start: df = df[pd.to_datetime(df[a.date]) >= pd.Timestamp(a.start, tz=pd.to_datetime(df[a.date]).dt.tz)]
    if a.end: df = df[pd.to_datetime(df[a.date]) <= pd.Timestamp(a.end, tz=pd.to_datetime(df[a.date]).dt.tz)]
    df = df.reset_index(drop=True); tg = a.targets.split(","); ex = [c for c in a.exog.split(",") if c]
    if a.pc and "ws100" in ex:
        df["ws100_pc"] = np.minimum(df["ws100"].values, 12.0) ** 3; ex = ex + ["ws100_pc"]
    cols = tg + ex; S = len(tg)
    raw = df[cols].values.astype("float32"); M = marks_from(df[a.date]); N = len(df)
    n_tr, n_va = int(N*0.7), int(N*0.1)
    mu, sd = raw[:n_tr].mean(0), raw[:n_tr].std(0) + 1e-6
    X = ((raw - mu) / sd).astype("float32")
    L, H = a.seq, a.pred
    tr = windows(X, M, None, 0, n_tr, L, H); va = windows(X, M, None, n_tr - L, n_tr + n_va, L, H)
    te = windows(X, M, None, n_tr + n_va - L, N, L, H)
    if a.max_train_windows: tr = tr[::max(1, len(tr)//a.max_train_windows)]
    mu_t = torch.tensor(mu[:S], device=dev); sd_t = torch.tensor(sd[:S], device=dev)

    cross, ple, wt, net, ramp = PRESETS.get(a.preset, ("off", False, "equal", False, 0.0))
    cross = a.cross or cross; ple = bool(a.ple) if a.ple is not None else ple
    wt = a.weighting or wt; net = bool(a.net) if a.net is not None else net; ramp = a.ramp if a.ramp is not None else ramp

    if a.preset == "snaive":
        res = evaluate_fn(lambda x, m, fm: (x[:, -24:, :S].repeat(1, (H + 23)//24, 1)[:, :H], 0), X, M, te, L, H, S, mu_t, sd_t, dev, a)
        return finish(a, res, {}, 0)
    if a.preset == "dlinear":
        model = DLinear(S, L, H).to(dev)
    elif a.preset in OFFICIAL:
        model = Official(a.preset, S, S + len(ex), L, H, a.d, 4, a.layers, 2 * a.d, 0.1, a.patch).to(dev)
    else:
        model = GateXer(S, len(ex), M.shape[1], L, H, a.patch, a.d, 4, a.layers, 2*a.d, 0.1, cross, ple, 2, 1, a.decomp, (M.shape[1] if a.fm else 0)).to(dev)
    use_fm = bool(a.fm) and a.preset not in OFFICIAL and a.preset not in ("dlinear", "snaive")
    mf = lambda x, m, fm: (model(x, m, fm) if use_fm else model(x, m))
    if a.snb and a.preset not in ("snaive", "dlinear"):
        # residual-on-yesterday: every window-normalised model (ours and the official baselines) outputs ~window mean + deviation;
        # swap the window mean for yesterday's values, so an untrained model equals seasonal-naive and training learns the correction.
        mf0 = mf
        def mf(x, m, fm, _f=mf0):
            p, aux = _f(x, m, fm)
            base = x[:, -24:, :S].repeat(1, (H + 23) // 24, 1)[:, :H]
            return p + base - x[:, :, :S].mean(1, keepdim=True), aux
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    nt = S + (1 if net and S >= 3 else 0)
    famo = FAMO(nt, dev) if wt == "famo" else None
    best, bad, best_state, t0 = 1e9, 0, None, time.time()
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, m, y, fm = [t.to(dev) for t in batch(X, M, perm[i:i+a.bs], L, H, S)]
            pred, aux = mf(x, m, fm)
            Ls = task_losses(pred, y, mu_t, sd_t, ramp, net)
            if famo: loss, _ = famo.loss(Ls)
            else: loss = Ls.sum()
            loss = loss + 0.01 * aux
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if famo:
                with torch.no_grad(): famo.update(task_losses(mf(x, m, fm)[0], y, mu_t, sd_t, ramp, net))
        v = evaluate_fn(mf, X, M, va, L, H, S, mu_t, sd_t, dev, a, model)["val_score"]
        if v < best - 1e-5: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state)
    a._keep = bool(a.save_preds)
    res = evaluate_fn(mf, X, M, te, L, H, S, mu_t, sd_t, dev, a, model)
    if a.save_preds:
        rv = evaluate_fn(mf, X, M, va, L, H, S, mu_t, sd_t, dev, a, model)
        np.savez(a.save_preds, val_P=rv.pop('_arrays')[0], val_Y=evaluate_fn(mf, X, M, va, L, H, S, mu_t, sd_t, dev, a, model)['_arrays'][1],
                 test_P=res['_arrays'][0], test_Y=res['_arrays'][1], te_idx=te, va_idx=va, seq=L)
        res.pop('_arrays')
    gates = model.gate_stats() if hasattr(model, "gate_stats") else {}
    finish(a, res, gates, time.time() - t0, dict(cross=cross, ple=ple, weighting=wt, net=net, ramp=ramp, epochs_run=ep + 1, val_best=float(best), n_train=int(len(tr)),
           params=sum(p.numel() for p in model.parameters())))

def evaluate_fn(fn, X, M, idx, L, H, S, mu_t, sd_t, dev, a, model=None):
    if model is not None: model.eval()
    P, Y = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            x, m, y, fm = [t.to(dev) for t in batch(X, M, idx[i:i+256], L, H, S)]
            p = fn(x, m, fm)[0]; P.append(p * sd_t + mu_t); Y.append(y * sd_t + mu_t)
    P, Y = torch.cat(P), torch.cat(Y)
    mae = (P - Y).abs().mean((0, 1)); rmse = ((P - Y)**2).mean((0, 1)).sqrt(); nmae = mae / Y.abs().mean((0, 1))
    dy = (Y[:, 1:] - Y[:, :-1]).abs(); thr = torch.quantile(dy.flatten(0, 1), 0.9, dim=0)
    msk = (dy >= thr); ramp = (((P[:, 1:] - Y[:, 1:]).abs()) * msk).sum((0, 1)) / msk.sum((0, 1)).clamp(min=1)
    if getattr(a, '_keep', False): out_arrays = (P.cpu().numpy(), Y.cpu().numpy())
    hs = [h for h in (1, 6, 24) if h <= P.shape[1]]
    mae_h = {str(h): ((P[:, h-1] - Y[:, h-1]).abs().mean(0) / Y[:, h-1].abs().mean(0)).tolist() for h in hs}
    out = dict(nmae_by_h=mae_h, std_mae=(mae / sd_t).tolist(), std_mse=((rmse / sd_t) ** 2).tolist(), mae=mae.tolist(), rmse=rmse.tolist(), nmae=nmae.tolist(), ramp_mae=ramp.tolist(), val_score=float(nmae.mean()))
    if S >= 3:
        npred, ny = P[..., 0] - P[..., 1] - P[..., 2], Y[..., 0] - Y[..., 1] - Y[..., 2]
        out["net_mae"] = float((npred - ny).abs().mean()); out["net_rmse"] = float(((npred - ny)**2).mean().sqrt())
    if getattr(a, '_keep', False): out['_arrays'] = out_arrays
    return out

def finish(a, res, gates, secs, extra=None):
    r = dict(args=vars(a), metrics=res, gates=gates, seconds=round(secs, 1), extra=extra or {})
    print(json.dumps({k: r[k] for k in ("metrics", "extra", "seconds")}, indent=1))
    if a.out: json.dump(r, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
