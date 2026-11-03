"""Official Time-Series-Library baselines (MIT, vendored unchanged except optional-import guards) wrapped to the GateXer interface.
forward(x[B,L,S+E], marks[B,L,M]) -> (pred[B,H,S], 0)
  patchtst, itransformer : channel set = targets + weather, all channels predicted, first S used (standard TSLib 'M' usage)
  timexer                : one model per target, target = endogenous, every other channel = exogenous (official 'MS' usage)"""
import os, sys, types, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tslib_vendor"))
from models import PatchTST as _P, iTransformer as _I, TimeXer as _X


def _cfg(enc_in, seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, patch_len, features="M"):
    return types.SimpleNamespace(task_name="long_term_forecast", features=features, seq_len=seq_len, label_len=0, pred_len=pred_len,
                                 enc_in=enc_in, dec_in=enc_in, c_out=enc_in, d_model=d_model, n_heads=n_heads, e_layers=e_layers,
                                 d_ff=d_ff, dropout=dropout, activation="gelu", factor=1, embed="timeF", freq="h", use_norm=True,
                                 patch_len=patch_len, class_strategy="projection", num_class=0)


class Official(nn.Module):
    def __init__(self, kind, n_tgt, n_chan, seq_len, pred_len, d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1, patch_len=24):
        super().__init__()
        self.kind, self.S, self.C = kind, n_tgt, n_chan
        if kind == "patchtst":
            self.m = _P.Model(_cfg(n_chan, seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, 16), patch_len=16, stride=8)
        elif kind == "itransformer":
            self.m = _I.Model(_cfg(n_chan, seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, 16))
        elif kind == "timexer":
            self.ms = nn.ModuleList([_X.Model(_cfg(n_chan, seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, patch_len, "MS"))
                                     for _ in range(n_tgt)])
        else:
            raise ValueError(kind)

    def forward(self, x, marks):
        zero = x.new_zeros(())
        if self.kind == "patchtst":
            return self.m(x, None, None, None)[..., :self.S], zero
        if self.kind == "itransformer":
            return self.m(x, marks, None, None)[..., :self.S], zero
        outs = []
        for t, mt in enumerate(self.ms):
            idx = [c for c in range(self.C) if c != t] + [t]            # target last = endogenous
            outs.append(mt(x[..., idx], marks, None, None)[:, -mt.pred_len:, -1:])
        return torch.cat(outs, -1), zero
