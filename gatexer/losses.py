import torch
import torch.nn.functional as F


class FAMO:
    """Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023) - minimal port of the official
    Cranial-XIX/FAMO WeightMethod (same equations)."""
    def __init__(self, n_tasks, device, gamma=1e-5, w_lr=0.025):
        self.w = torch.zeros(n_tasks, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.prev = None

    def loss(self, losses):
        self.prev = losses.detach()
        z = F.softmax(self.w, -1)
        D = losses + 1e-8
        c = (z / D).sum().detach()
        return (D.log() * z / c).sum(), z.detach()

    def update(self, curr):
        delta = (self.prev + 1e-8).log() - (curr + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1), self.w, grad_outputs=delta.detach())[0]
        self.opt.zero_grad()
        self.w.grad = d
        self.opt.step()


def task_losses(pred, y, mu, sd, ramp_beta=0.0, net_w=True):
    """pred,y: [B,H,S] standardised. Column order: 0=load, 1=solar, 2=wind (net load = load-solar-wind).
    Returns per-task losses: [load, solar, wind, (net-load)]."""
    err = (pred - y).abs()
    if ramp_beta > 0:                                   # ramp-weighted MAE: emphasise large step changes
        dy = (y[:, 1:] - y[:, :-1]).abs()
        dy = torch.cat([dy[:, :1], dy], 1)
        w = 1.0 + ramp_beta * (dy / (dy.mean((0, 1), keepdim=True) + 1e-6)).clamp(max=5.0)
        err = err * w
    L = err.mean((0, 1))                                # [S]
    if net_w and pred.shape[-1] >= 3:
        rp, ry = pred * sd + mu, y * sd + mu            # back to raw units
        nl_p = rp[..., 0] - rp[..., 1] - rp[..., 2]
        nl_y = ry[..., 0] - ry[..., 1] - ry[..., 2]
        scale = nl_y.std() + 1e-6
        Ln = ((nl_p - nl_y).abs() / scale).mean()
        L = torch.cat([L, Ln.view(1)])
    return L
