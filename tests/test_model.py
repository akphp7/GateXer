"""Fast CPU tests of the properties that the paper relies on."""
import torch
from model import GateXer, CrossMix
from losses import FAMO

S, E, M, L, H = 3, 4, 6, 168, 24


def make(cross, **kw):
    torch.manual_seed(0)
    return GateXer(S, E, M, L, H, d_model=32, heads=4, d_ff=64, cross=cross, ple=False, decomp=25, **kw).eval()


def batch(B=5):
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, L, S + E, generator=g) * 3 + 10, torch.randn(B, L, M, generator=g)


def test_output_shape_and_finite():
    x, m = batch()
    y, aux = make("gated")(x, m)
    assert y.shape == (5, H, S)
    assert torch.isfinite(y).all() and torch.isfinite(aux)


def test_parameter_count_at_paper_width():
    net = GateXer(S, 4, 6, L, H, d_model=128, heads=4, d_ff=256, cross="gated", ple=False, decomp=25)
    n = sum(p.numel() for p in net.parameters())
    assert n > 0
    print("parameters at width 128 (3 sources, 4 weather channels, 6 calendar channels):", n)


def test_gates_start_closed_so_gated_equals_no_sharing():
    """Zero-initialised gates: at initialisation the gated model must output exactly what the
    no-sharing model outputs with the same weights ("starts as three independent forecasters")."""
    x, m = batch()
    gated, off = make("gated"), make("off")
    off.load_state_dict(gated.state_dict())
    with torch.no_grad():
        yg, _ = gated(x, m)
        yo, _ = off(x, m)
    assert torch.allclose(yg, yo, atol=1e-6)
    for blk in gated.blocks:
        assert torch.all(blk.cross.alpha_src == 0) and torch.all(blk.cross.alpha_ex == 0)


def test_always_on_differs_from_no_sharing():
    x, m = batch()
    always, off = make("always"), make("off")
    off.load_state_dict(always.state_dict())
    with torch.no_grad():
        ya, _ = always(x, m)
        yo, _ = off(x, m)
    assert not torch.allclose(ya, yo, atol=1e-4)


def test_gradients_open_the_gates():
    """The gates receive gradient from the very first step, so they can open if sharing helps."""
    x, m = batch()
    net = make("gated").train()
    y, aux = net(x, m)
    (y.abs().mean() + 0.01 * aux).backward()
    g = sum(blk.cross.alpha_src.grad.abs().sum() + blk.cross.alpha_ex.grad.abs().sum() for blk in net.blocks)
    assert g > 0


def test_single_source_model_runs():
    """With one source there is nothing to mix; source mixing is skipped and the model still runs."""
    net = GateXer(1, E, M, L, H, d_model=32, heads=4, d_ff=64, cross="gated", ple=False).eval()
    y, _ = net(torch.randn(2, L, 1 + E), torch.randn(2, L, M))
    assert y.shape == (2, H, 1)


def test_famo_weights_form_a_distribution():
    f = FAMO(3, "cpu")
    loss, z = f.loss(torch.tensor([0.3, 0.2, 0.5], requires_grad=True))
    assert abs(float(z.sum()) - 1.0) < 1e-6 and (z > 0).all()
