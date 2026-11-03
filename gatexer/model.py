"""GateXer: TimeXer-style patch/global-token encoder + TimeFilter-style routed cross-source mixing
+ PLE/CGC-style shared/task experts, with a zero-initialised "do-no-harm" cross-source gate.

Inputs  x      : [B, L, Ct+E]  first Ct channels = targets (load, solar, wind, ...), rest = weather exog
        marks  : [B, L, M]     calendar features (known)
Outputs yhat   : [B, H, Ct]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEmb(nn.Module):
    def __init__(self, d, n=512):
        super().__init__()
        pe = torch.zeros(n, d)
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n):
        return self.pe[:n]


class Expert(nn.Module):
    def __init__(self, d, dff, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(drop), nn.Linear(dff, d))

    def forward(self, x):
        return self.net(x)


class PLEFFN(nn.Module):
    """Per-source gate over [shared experts + that source's own experts] (CGC layer of PLE)."""
    def __init__(self, d, dff, n_src, n_shared, n_task, drop):
        super().__init__()
        self.n_src = n_src
        self.shared = nn.ModuleList([Expert(d, dff, drop) for _ in range(n_shared)])
        self.task = nn.ModuleList([nn.ModuleList([Expert(d, dff, drop) for _ in range(n_task)]) for _ in range(n_src)])
        self.gates = nn.ModuleList([nn.Linear(d, n_shared + n_task) for _ in range(n_src)])

    def forward(self, x):  # x [B, S, T, D]
        sh = [e(x) for e in self.shared]  # computed once for all sources
        outs = []
        for s in range(self.n_src):
            xs = x[:, s]
            ex = [e[:, s] for e in sh] + [e(xs) for e in self.task[s]]
            ex = torch.stack(ex, -1)                        # [B,T,D,E]
            g = torch.softmax(self.gates[s](xs), -1)        # [B,T,E]
            outs.append((ex * g.unsqueeze(2)).sum(-1))
        return torch.stack(outs, 1)


class PlainFFN(nn.Module):
    def __init__(self, d, dff, drop):
        super().__init__()
        self.e = Expert(d, dff, drop)

    def forward(self, x):
        return self.e(x)


class CrossMix(nn.Module):
    """Routed cross-source mixing. Regions (TimeFilter-style): 0 = none, 1 = other sources at same patch,
    2 = exogenous (weather/calendar) tokens. mode: 'off' | 'always' | 'gated' | 'src' (gated, source mixing only) | 'ex' (gated, weather mixing only)."""
    def __init__(self, d, heads, n_src, drop, mode):
        super().__init__()
        self.mode, self.n_src = mode, n_src
        self.a_src = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.a_ex = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.router = nn.Linear(d, 3)
        nn.init.zeros_(self.router.weight)
        with torch.no_grad():
            self.router.bias.copy_(torch.tensor([0.0, 0.0, 0.0]))
        # per-source zero-init gate -> block starts as single-task ("do no harm")
        self.alpha_src = nn.Parameter(torch.zeros(n_src))
        self.alpha_ex = nn.Parameter(torch.zeros(n_src))
        self.norm = nn.LayerNorm(d)
        self.last_gate = None

    def forward(self, x, ex):  # x [B,S,T,D] ; ex [B,E,D] or None
        B, S, T, D = x.shape
        h = self.norm(x)
        out_src = torch.zeros_like(x)
        out_ex = torch.zeros_like(x)
        if self.mode in ("gated", "always", "src") and S > 1:
            q = h.permute(0, 2, 1, 3).reshape(B * T, S, D)
            mask = torch.eye(S, dtype=torch.bool, device=x.device)      # block self (own source)
            a, _ = self.a_src(q, q, q, attn_mask=mask, need_weights=False)
            out_src = a.reshape(B, T, S, D).permute(0, 2, 1, 3)
        if self.mode in ("gated", "always", "ex") and ex is not None:
            q = h.reshape(B * S, T, D)
            kv = ex.unsqueeze(1).expand(B, S, ex.shape[1], D).reshape(B * S, ex.shape[1], D)
            a, _ = self.a_ex(q, kv, kv, need_weights=False)
            out_ex = a.reshape(B, S, T, D)
        if self.mode == "always":
            return x + out_src + out_ex, x.new_zeros(())
        if self.mode == "off":
            return x, x.new_zeros(())
        g = torch.softmax(self.router(h), -1)                              # [B,S,T,3]
        self.last_gate = g.detach()
        a1 = torch.tanh(self.alpha_src).view(1, S, 1, 1)
        a2 = torch.tanh(self.alpha_ex).view(1, S, 1, 1)
        y = x + a1 * g[..., 1:2] * out_src + a2 * g[..., 2:3] * out_ex
        imp = g.mean((0, 2)).reshape(-1)                                   # importance balance (TimeFilter-like)
        aux = imp.var() / (imp.mean() ** 2 + 1e-8)
        return y, aux


class Block(nn.Module):
    def __init__(self, d, heads, dff, n_src, drop, cross, ple, n_shared, n_task):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = CrossMix(d, heads, n_src, drop, cross)
        self.ffn = PLEFFN(d, dff, n_src, n_shared, n_task, drop) if ple else PlainFFN(d, dff, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ex):  # x [B,S,T,D]
        B, S, T, D = x.shape
        h = x.reshape(B * S, T, D)
        a, _ = self.sa(h, h, h, need_weights=False)                          # temporal self-attn per source (TimeXer)
        x = self.n1(x + self.drop(a.reshape(B, S, T, D)))
        x, aux = self.cross(x, ex)                                           # routed cross-source mixing
        x = self.n2(x + self.drop(self.ffn(x)))
        return x, aux


class GateXer(nn.Module):
    def __init__(self, n_tgt, n_exog, n_marks, seq_len, pred_len, patch_len=24, d_model=64, heads=4,
                 layers=2, d_ff=128, drop=0.1, cross="gated", ple=True, n_shared=2, n_task=1, decomp=0, fm_dim=0):
        super().__init__()
        assert seq_len % patch_len == 0
        self.S, self.E, self.M = n_tgt, n_exog, n_marks
        self.L, self.H, self.pl = seq_len, pred_len, patch_len
        self.P = seq_len // patch_len
        self.val = nn.Linear(patch_len, d_model, bias=False)
        self.decomp = decomp                               # PowerMamba-style trend/seasonal split (kernel size)
        if decomp:
            self.val_s = nn.Linear(patch_len, d_model, bias=False)
        self.pos = PosEmb(d_model)
        self.src_emb = nn.Parameter(torch.randn(n_tgt, 1, d_model) * 0.02)
        self.glb = nn.Parameter(torch.randn(n_tgt, 1, d_model) * 0.02)   # TimeXer global token (per source)
        self.has_ex = (n_exog + n_marks) > 0
        if self.has_ex:
            self.ex_emb = nn.Linear(seq_len, d_model)                  # inverted (variate) embedding
        self.blocks = nn.ModuleList([Block(d_model, heads, d_ff, n_tgt, drop, cross, ple, n_shared, n_task)
                                     for _ in range(layers)])
        self.head_w = nn.Parameter(torch.randn(n_tgt, d_model * (self.P + 1), pred_len) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_tgt, pred_len))
        self.drop = nn.Dropout(drop)
        self.use_fm = fm_dim > 0                                   # calendar marks of the forecast hours (known in advance)
        if self.use_fm:
            F_in = d_model * (self.P + 1) + pred_len * fm_dim; hid = 256
            self.c1_w = nn.Parameter(torch.randn(n_tgt, F_in, hid) * (1.0 / math.sqrt(F_in))); self.c1_b = nn.Parameter(torch.zeros(n_tgt, hid))
            self.c2_w = nn.Parameter(torch.zeros(n_tgt, hid, pred_len)); self.c2_b = nn.Parameter(torch.zeros(n_tgt, pred_len))   # zero-init: starts identical to the model without it

    def forward(self, x, marks, fmarks=None):
        B = x.shape[0]
        mean = x.mean(1, keepdim=True).detach()                           # RevIN
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        xn = (x - mean) / std
        tgt = xn[..., :self.S].permute(0, 2, 1)                           # [B,S,L]
        if self.decomp:
            k = self.decomp
            tr = F.avg_pool1d(F.pad(tgt, (k // 2, k // 2), mode="replicate"), k, 1)
            pt, ps = tr.unfold(-1, self.pl, self.pl), (tgt - tr).unfold(-1, self.pl, self.pl)
            z = self.val(pt) + self.val_s(ps)
        else:
            z = self.val(tgt.unfold(-1, self.pl, self.pl))                # [B,S,P,D]
        z = z + self.pos(self.P).view(1, 1, self.P, -1) + self.src_emb.unsqueeze(0)
        z = torch.cat([z, self.glb.unsqueeze(0).expand(B, -1, -1, -1)], 2)  # [B,S,P+1,D]
        z = self.drop(z)
        ex = None
        if self.has_ex:
            e = torch.cat([xn[..., self.S:], marks], -1).permute(0, 2, 1)  # [B,E+M,L]
            ex = self.ex_emb(e)
        aux = z.new_zeros(())
        for b in self.blocks:
            z, a = b(z, ex)
            aux = aux + a
        flat = z.reshape(B, self.S, -1)                                    # [B,S,D*(P+1)]
        y = torch.einsum("bsf,sfh->bsh", flat, self.head_w) + self.head_b  # [B,S,H]
        if self.use_fm and fmarks is not None:
            fm = fmarks.reshape(B, 1, -1).expand(-1, self.S, -1)
            hcat = F.gelu(torch.einsum("bsf,sfh->bsh", torch.cat([flat, fm], -1), self.c1_w) + self.c1_b)
            y = y + torch.einsum("bsf,sfh->bsh", self.drop(hcat), self.c2_w) + self.c2_b      # normalised-space correction
        y = y.permute(0, 2, 1)                                             # [B,H,S]
        y = y * std[..., :self.S] + mean[..., :self.S]
        return y, aux

    def gate_stats(self):
        out = []
        for b in self.blocks:
            g = b.cross.last_gate
            if g is not None:
                out.append(dict(region_probs=g.mean((0, 2)).cpu().tolist(),
                                alpha_src=torch.tanh(b.cross.alpha_src).detach().cpu().tolist(),
                                alpha_ex=torch.tanh(b.cross.alpha_ex).detach().cpu().tolist()))
        return out


class DLinear(nn.Module):
    """Simple per-channel baseline (decomposition + linear), targets only."""
    def __init__(self, n_tgt, seq_len, pred_len, k=25):
        super().__init__()
        self.S, self.k = n_tgt, k
        self.trend = nn.Linear(seq_len, pred_len)
        self.seas = nn.Linear(seq_len, pred_len)

    def forward(self, x, marks):
        x = x[..., :self.S]
        pad = self.k // 2
        xp = F.pad(x.permute(0, 2, 1), (pad, pad), mode="replicate")
        tr = F.avg_pool1d(xp, self.k, 1)
        se = x.permute(0, 2, 1) - tr
        y = self.trend(tr) + self.seas(se)
        return y.permute(0, 2, 1), x.new_zeros(())
