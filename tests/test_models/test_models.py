import io
import torch
import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from src.models import EMA, ScoreModel, EnergyBasedModel, FlowMatchingModel


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyNet(nn.Module):
    """A tiny network that is easy to reason about in tests.

    It concatenates time t (broadcast along feature dim) to x and applies a
    single linear layer, producing the same shape as x.
    """

    def __init__(self, dim: int):
        super().__init__()
        # x has shape (N, dim). We append one feature for t => dim+1
        self.lin = nn.Linear(dim + 1, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, *labels):
        # ensure t is broadcastable to x: shape (N, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt = torch.cat([x, t.expand_as(x[:, :1])], dim=1)
        return self.lin(xt)


def make_batch(n=4, d=3, device=None):
    return torch.randn(n, d, device=device)


def test_ema_update_and_restore():
    set_seed(123)
    net = TinyNet(dim=3)
    model = nn.Sequential(net)  # anything with named_parameters
    ema = EMA(model, decay=0.5)

    # initial shadow copied
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # change params to trigger update
    for p in model.parameters():
        with torch.no_grad():
            p.add_(1.0)

    ema.update()
    # shadow should have moved towards new params
    moved = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert name in ema.shadow
            # new shadow in between old and new
            assert torch.allclose(
                ema.shadow[name],
                (1 - ema.decay) * p.data + ema.decay * initial_shadow[name],
            )
            if not torch.allclose(ema.shadow[name], initial_shadow[name]):
                moved = True
    assert moved

    # apply shadow and restore bring back original params
    backup_before = {
        k: p.data.clone() for k, p in model.named_parameters() if p.requires_grad
    }
    ema.apply_shadow()
    # params now equal to shadow
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.data, ema.shadow[name])
    ema.restore()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.data, backup_before[name])


@pytest.mark.parametrize(
    "schedule", ["geometric", "linear"]
)  # exercise both schedules via ScoreModel
def test_scoremodel_forward_and_loss_and_trainstep(schedule):
    set_seed(0)
    d = 4
    net = TinyNet(dim=d)
    model = ScoreModel(network=net, schedule=schedule, device="cpu")

    # forward normalization: if network outputs zeros, output is zeros
    x = torch.zeros(2, d)
    t = torch.full((2,), 0.5)
    with torch.no_grad():
        # zero out weights/bias such that TinyNet returns zeros
        for p in net.parameters():
            p.zero_()
        out = model.forward(x, t)
        assert out.shape == x.shape
        assert torch.count_nonzero(out) == 0

    # loss is scalar and has grad
    batch = make_batch(n=8, d=d)
    loss = model.loss_fn(batch)
    assert loss.dim() == 0
    loss.backward()
    # optimizer step path without scheduler
    opt = SGD(model.parameters(), lr=1e-2)
    loss2 = model.train_step(batch, opt)
    assert isinstance(loss2.item(), float)

    # train_step with scheduler
    opt = SGD(model.parameters(), lr=1e-2)
    sch = StepLR(opt, step_size=1, gamma=0.9)
    _ = model.train_step(batch, opt, scheduler=sch)


def test_scoremodel_save_and_load(tmp_path):
    set_seed(42)
    d = 3
    net = TinyNet(dim=d)
    model = ScoreModel(network=net, device="cpu")

    # mutate some state and ema shadow
    model.history = ["epoch0", {"loss": 1.23}]
    for name, p in model.named_parameters():
        if p.requires_grad:
            model.ema.shadow[name] = p.data.clone() + 0.5  # nontrivial shadow

    # save
    f = tmp_path / "weights.pth"
    model._save_weights(str(f))
    assert f.exists()

    # create a fresh model and load
    new_model = ScoreModel(network=TinyNet(dim=d), device="cpu")
    new_model._load_weights(str(f))

    # state_dict equalities
    for k in model.state_dict().keys():
        assert torch.allclose(model.state_dict()[k], new_model.state_dict()[k])
    # history loaded
    assert new_model.history == model.history
    # ema shadow loaded
    assert set(new_model.ema.shadow.keys()) == set(model.ema.shadow.keys())
    for k in new_model.ema.shadow:
        assert torch.allclose(new_model.ema.shadow[k], model.ema.shadow[k])


def test_scoremodel_load_missing_optional_keys(tmp_path):
    # covers _load_weights fallbacks for missing HISTORY/EMA
    d = 3
    m1 = ScoreModel(network=TinyNet(dim=d), device="cpu")
    f = tmp_path / "minimal.pth"
    torch.save({"MODEL_STATE": m1.state_dict()}, str(f))

    m2 = ScoreModel(network=TinyNet(dim=d), device="cpu")
    m2._load_weights(str(f))

    for k in m1.state_dict().keys():
        assert torch.allclose(m1.state_dict()[k], m2.state_dict()[k])
    assert m2.history == []  # fallback exercised
    assert (
        isinstance(m2.ema.shadow, dict) and len(m2.ema.shadow) == 0
    )  # fallback exercised


def test_energy_based_model_forward_and_loss():
    set_seed(7)
    d = 5
    net = TinyNet(dim=d)
    ebm = EnergyBasedModel(network=net, device="cpu")

    x = make_batch(n=3, d=d)
    t = torch.rand(3)

    # direct forward (create_graph default False)
    g = ebm.forward(x.clone(), t)
    assert g.shape == x.shape

    # loss path (uses create_graph=True internally)
    batch = make_batch(n=6, d=d)
    loss = ebm.loss_fn(batch)
    assert loss.dim() == 0
    loss.backward()


def test_energy_based_model_forward_create_graph_flag():
    # explicitly exercise create_graph=True branch
    set_seed(11)
    d = 2
    ebm = EnergyBasedModel(network=TinyNet(dim=d), device="cpu")
    x = make_batch(n=2, d=d)
    t = torch.rand(2)
    g = ebm.forward(x, t, create_graph=True)
    assert g.shape == x.shape
    s = g.sum()
    # backprop through the created graph to ensure it's retained
    s.backward(create_graph=True)


def test_flow_matching_model_end_to_end(tmp_path):
    set_seed(9)
    d = 3
    net = TinyNet(dim=d)
    fmm = FlowMatchingModel(network=net, device="cpu")

    # forward returns same shape
    x = make_batch(n=2, d=d)
    t = torch.rand(2, 1)
    y = fmm.forward(x, t)
    assert y.shape == x.shape

    # loss is scalar and has gradient
    batch = make_batch(n=8, d=d)
    loss = fmm.loss_fn(batch)
    assert loss.dim() == 0
    loss.backward()

    # train_step without scheduler
    opt = SGD(fmm.parameters(), lr=1e-2)
    l = fmm.train_step(batch, opt)
    assert isinstance(l.item(), float)

    # train_step with scheduler path
    opt = SGD(fmm.parameters(), lr=1e-2)
    sch = StepLR(opt, step_size=1, gamma=0.9)
    _ = fmm.train_step(batch, opt, scheduler=sch)

    # save/load round trip
    f = tmp_path / "fmm_weights.pth"
    fmm.history = ["h1", 2]
    for name, p in fmm.named_parameters():
        if p.requires_grad:
            fmm.ema.shadow[name] = p.data.clone() + 0.1
    fmm._save_weights(str(f))

    fmm2 = FlowMatchingModel(network=TinyNet(dim=d), device="cpu")
    fmm2._load_weights(str(f))

    for k in fmm.state_dict().keys():
        assert torch.allclose(fmm.state_dict()[k], fmm2.state_dict()[k])
    assert fmm2.history == fmm.history
    for k in fmm.ema.shadow:
        assert torch.allclose(fmm.ema.shadow[k], fmm2.ema.shadow[k])
