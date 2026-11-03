"""After run_experiments (main + chronos): (1) turn chronos .npz into result .json, (2) build the hybrid = per-target blend of GateXer ('gx')
and Chronos-2 with weights fitted on VALIDATION only, per seed.   python postprocess.py results"""
import glob, json, os, sys, numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "runs"
def metrics(P, Y):
    mae = np.abs(P - Y).mean((0, 1)); nm = mae / np.abs(Y).mean((0, 1))
    dy = np.abs(Y[:, 1:] - Y[:, :-1]); thr = np.quantile(dy.reshape(-1, dy.shape[-1]), 0.9, axis=0); msk = dy >= thr
    ramp = (np.abs(P[:, 1:] - Y[:, 1:]) * msk).sum((0, 1)) / np.maximum(msk.sum((0, 1)), 1)
    r = dict(nmae=nm.tolist(), mae=mae.tolist(), rmse=np.sqrt(((P - Y) ** 2).mean((0, 1))).tolist(), ramp_mae=ramp.tolist(),
             nmae_by_h={str(h): (np.abs(P[:, h-1] - Y[:, h-1]).mean(0) / np.abs(Y[:, h-1]).mean(0)).tolist() for h in (1, 6, 24) if h <= P.shape[1]})
    if P.shape[-1] >= 3: r["net_mae"] = float(np.abs((P[..., 0] - P[..., 1] - P[..., 2]) - (Y[..., 0] - Y[..., 1] - Y[..., 2])).mean())
    return r
def save(path, m, extra=None): json.dump(dict(metrics=m, extra=extra or {}), open(path, "w"), indent=1)
for ds_dir in sorted(glob.glob(f"{out}/main/*")):
    ds = os.path.basename(ds_dir)
    chr_files = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(f"{ds_dir}/chronos2_*.npz")}
    for tag, z in chr_files.items(): save(f"{ds_dir}/{tag}_s0.json", metrics(z["test_P"], z["test_Y"]))
    if "chronos2_cov" not in chr_files: continue
    B = chr_files["chronos2_cov"]; grid = np.linspace(0, 1, 21)
    for f in sorted(glob.glob(f"{ds_dir}/gx_s*.npz")):
        A = np.load(f)
        if A["test_P"].shape != B["test_P"].shape or not np.allclose(A["test_Y"], B["test_Y"], atol=1e-2 * np.abs(B["test_Y"]).mean()):
            print("window mismatch, skip", f); continue
        S = A["val_Y"].shape[-1]
        w = [float(grid[np.argmin([np.abs(g * A["val_P"][..., s] + (1 - g) * B["val_P"][..., s] - A["val_Y"][..., s]).mean() for g in grid])]) for s in range(S)]
        P = np.stack([w[s] * A["test_P"][..., s] + (1 - w[s]) * B["test_P"][..., s] for s in range(S)], -1)
        seed = os.path.basename(f)[:-4].split("_s")[-1]
        save(f"{ds_dir}/gx+chronos2_s{seed}.json", metrics(P, A["test_Y"]), dict(weight_on_gx=w))
        print(ds, "blend seed", seed, "weights on GateXer:", w)
