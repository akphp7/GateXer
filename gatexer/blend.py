"""Fit a per-target convex blend between two prediction sets on VALIDATION only, report TEST metrics for
model A, model B and the blend. Inputs are the .npz files written by train.py --save_preds / chronos_zs.py."""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--names", default="gatexer,chronos2"); ap.add_argument("--out", default="")
args = ap.parse_args()
A, B = np.load(args.a), np.load(args.b)
assert np.allclose(A["test_Y"], B["test_Y"], atol=1e-3), "windows differ between the two files"
na, nb = args.names.split(",")
S = A["val_Y"].shape[-1]
grid = np.linspace(0, 1, 21)
w = [float(grid[np.argmin([np.abs(g * A["val_P"][..., s] + (1 - g) * B["val_P"][..., s] - A["val_Y"][..., s]).mean() for g in grid])]) for s in range(S)]
Pb = np.stack([w[s] * A["test_P"][..., s] + (1 - w[s]) * B["test_P"][..., s] for s in range(S)], -1)
Y = A["test_Y"]
def met(P):
    mae = np.abs(P - Y).mean((0, 1)); r = dict(mae=mae.tolist(), nmae=(mae / np.abs(Y).mean((0, 1))).tolist(),
        rmse=np.sqrt(((P - Y) ** 2).mean((0, 1))).tolist())
    if S >= 3: r["net_mae"] = float(np.abs((P[..., 0] - P[..., 1] - P[..., 2]) - (Y[..., 0] - Y[..., 1] - Y[..., 2])).mean())
    return r
res = {na: met(A["test_P"]), nb: met(B["test_P"]), "blend": met(Pb), "weight_on_" + na: w}
print(json.dumps(res, indent=1))
if args.out: json.dump(res, open(args.out, "w"), indent=1)
