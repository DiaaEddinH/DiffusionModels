import torch
import pytest

from torch import nn, Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from diffusion_models.noise.noise_scheduler import GeometricSchedule, LinearSchedule
from diffusion_models.models.models import (
    ExponentialMovingAverage,
    ScoreModel,
    EnergyBasedModel,
    FlowMatchingModel,
)

from tests.conftest import DummyNetwork
from torch.nn import Module, Parameter


@pytest.fixture
def score_model(dummy_network, dummy_schedule):
    return ScoreModel(network=dummy_network, schedule=dummy_schedule, decay_rate=0.9)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

class TestConstruction:
    def test_stores_network_and_schedule(self, score_model, dummy_network, dummy_schedule):
        assert score_model.network is dummy_network
        assert score_model.schedule is dummy_schedule

    def test_device_defaults_to_network_device(self, score_model):
        assert score_model.device == torch.device("cpu")

    def test_history_starts_empty(self, score_model):
        assert score_model.history == []

    def test_ema_shadow_seeded_from_initial_weights(self, score_model, dummy_network):
        assert torch.allclose(score_model.exponential_moving_average.shadow["network.scale"], dummy_network.scale.data)

    def test_ema_decay_is_configurable(self, dummy_network, dummy_schedule):
        model = ScoreModel(network=dummy_network, schedule=dummy_schedule, decay_rate=0.21)
        assert model.exponential_moving_average.decay_rate == pytest.approx(0.21)

    def test_unused_kwargs_warn(self, dummy_network, dummy_schedule):
        with pytest.warns(UserWarning, match="unused"):
            ScoreModel(network=dummy_network, schedule=dummy_schedule, not_a_real_arg=123)

    def test_explicit_device_moves_model(self, dummy_network, dummy_schedule):
        model = ScoreModel(network=dummy_network, schedule=dummy_schedule, device="cpu")
        assert model.device == torch.device("cpu")

# ----------------------------------------------------------------------
# device property/to()
# ----------------------------------------------------------------------

class TestDeviceSync:
    def test_device_property_reflects_to_call(self, score_model):
        score_model.to("cpu")
        assert score_model.device == torch.device("cpu")

    def test_to_moves_ema_shadow_dtype(self, score_model):
        score_model.to(dtype=torch.float64)
        assert score_model.exponential_moving_average.shadow["network.scale"].dtype == torch.float64

    def test_to_returns_self(self, score_model):
        assert score_model.to(dtype=torch.float32) is score_model


# ----------------------------------------------------------------------
# forward()
# ----------------------------------------------------------------------

class TestForward:
    def test_rescales_network_output_by_stddev(self, score_model, dummy_network, dummy_schedule):
        x = torch.randn(4, 3)
        t = torch.rand(4).clamp(min=0.1)  # avoid t=0 where stddev is 0
        out = score_model.forward(x, t)
        d = (x.dim() - 1) * (None,)
        expected = dummy_network(x, t) / dummy_schedule.stddev(t)[:, *d]
        assert torch.allclose(out, expected)
 
    def test_forwards_labels_to_network(self, dummy_schedule):
        seen = {}
 
        class LabelCapturingNetwork(Module):
            def __init__(self):
                super().__init__()
                self.scale = Parameter(torch.tensor(1.0))
 
            def forward(self, x, t, *labels):
                seen["labels"] = labels
                return self.scale * x
 
        model = ScoreModel(network=LabelCapturingNetwork(), schedule=dummy_schedule)
        t = torch.rand(2).clamp(min=0.1)
        model.forward(torch.randn(2, 2), t, "label_a")
        assert seen["labels"] == ("label_a",)

# ---------------------------------------------------------------------------
# loss_fn
# ---------------------------------------------------------------------------
 
class TestLossFn:
    def test_returns_scalar(self, score_model):
        loss = score_model.loss_fn(torch.randn(8, 3))
        assert loss.dim() == 0
 
    def test_is_finite_and_nonnegative(self, score_model):
        loss = score_model.loss_fn(torch.randn(8, 3))
        assert torch.isfinite(loss)
        assert loss.item() >= 0
 
    def test_gradients_flow_to_network_params(self, score_model, dummy_network):
        loss = score_model.loss_fn(torch.randn(8, 3))
        loss.backward()
        assert dummy_network.scale.grad is not None
 
    def test_eps_keeps_random_t_away_from_zero(self, score_model, monkeypatch):
        # Force torch.rand to return the minimum value (0.0) and confirm the
        # sampled time is still shifted up by eps, not exactly 0.
        monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.zeros(*a))
        captured = {}
        orig_mean_stddev = score_model.schedule.mean_stddev
 
        def spy(x, t):
            captured["t"] = t.clone()
            return orig_mean_stddev(x, t)
 
        score_model.schedule.mean_stddev = spy
        score_model.loss_fn(torch.randn(4, 3))
        assert (captured["t"] >= 1e-5 - 1e-12).all()


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------
 
class TestTrainStep:
    def test_updates_network_parameters(self, score_model, dummy_network):
        before = dummy_network.scale.item()
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)
        assert dummy_network.scale.item() != pytest.approx(before)
 
    def test_updates_ema_shadow(self, score_model):
        before = score_model.exponential_moving_average.shadow["network.scale"].clone()
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)
        assert not torch.allclose(before, score_model.exponential_moving_average.shadow["network.scale"])
 
    def test_appends_to_history(self, score_model):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        assert score_model.history == []
        score_model.train_step(torch.randn(8, 3), optimizer)
        assert len(score_model.history) == 1
        assert isinstance(score_model.history[0], float)
 
    def test_calls_scheduler_step_if_given(self, score_model):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_before = optimizer.param_groups[0]["lr"]
        score_model.train_step(torch.randn(8, 3), optimizer, lr_scheduler=lr_scheduler)
        assert optimizer.param_groups[0]["lr"] < lr_before
 
    def test_sets_train_mode(self, score_model):
        score_model.eval()
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)
        assert score_model.training is True
 
    def test_returns_loss_tensor(self, score_model):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        loss = score_model.train_step(torch.randn(8, 3), optimizer)
        assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# save / load weights
# ---------------------------------------------------------------------------
 
class TestCheckpointing:
    def test_roundtrip_preserves_weights(self, score_model, dummy_network, dummy_schedule, tmp_path):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)  # move weights off init
 
        ckpt_path = tmp_path / "weights.pt"
        score_model._save_weights(ckpt_path)
 
        fresh_network = DummyNetwork(scale=999.0)
        fresh_model = ScoreModel(network=fresh_network, schedule=dummy_schedule)
        fresh_model._load_weights(ckpt_path)
 
        assert torch.allclose(fresh_network.scale.data, dummy_network.scale.data)
 
    def test_roundtrip_preserves_history(self, score_model, dummy_schedule, tmp_path):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)
        score_model.train_step(torch.randn(8, 3), optimizer)
 
        ckpt_path = tmp_path / "weights.pt"
        score_model._save_weights(ckpt_path)
 
        fresh_model = ScoreModel(network=DummyNetwork(), schedule=dummy_schedule)
        fresh_model._load_weights(ckpt_path)
 
        assert fresh_model.history == score_model.history
 
    def test_roundtrip_preserves_ema_shadow(self, score_model, dummy_schedule, tmp_path):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        score_model.train_step(torch.randn(8, 3), optimizer)
 
        ckpt_path = tmp_path / "weights.pt"
        score_model._save_weights(ckpt_path)
 
        fresh_model = ScoreModel(network=DummyNetwork(), schedule=dummy_schedule)
        fresh_model._load_weights(ckpt_path)
 
        assert torch.allclose(
            fresh_model.exponential_moving_average.shadow["network.scale"],
            score_model.exponential_moving_average.shadow["network.scale"],
        )
 
    def test_save_creates_missing_parent_dirs(self, score_model, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "weights.pt"
        score_model._save_weights(nested_path)
        assert nested_path.exists()

# class TinyNet(nn.Module):
#     """A tiny network that is easy to reason about in tests.

#     It concatenates time t (broadcast along feature dim) to x and applies a
#     single linear layer, producing the same shape as x.
#     """

#     def __init__(self, dim: int):
#         super().__init__()
#         # x has shape (N, dim). We append one feature for t => dim+1
#         self.lin = nn.Linear(dim + 1, dim)

#     def forward(self, x: torch.Tensor, t: torch.Tensor, *labels):
#         # ensure t is broadcastable to x: shape (N, 1)
#         if t.dim() == 1:
#             t = t.unsqueeze(-1)
#         xt = torch.cat([x, t.expand_as(x[:, :1])], dim=1)
#         return self.lin(xt)


# def make_batch(n=4, d=3, device=None):
#     return torch.randn(n, d, device=device)


# def test_ema_update_and_restore():
#     model = TinyNet(dim=3)
#     ema = EMA(model, decay=0.5)

#     # initial shadow copied
#     initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

#     # change params to trigger update
#     for p in model.parameters():
#         with torch.no_grad():
#             p.add_(1.0)

#     ema.update()
#     # shadow should have moved towards new params
#     moved = False
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             assert name in ema.shadow
#             # new shadow in between old and new
#             assert torch.allclose(
#                 ema.shadow[name],
#                 (1 - ema.decay) * p.data + ema.decay * initial_shadow[name],
#             )
#             if not torch.allclose(ema.shadow[name], initial_shadow[name]):
#                 moved = True
#     assert moved

#     # apply shadow and restore bring back original params
#     backup_before = {
#         k: p.data.clone() for k, p in model.named_parameters() if p.requires_grad
#     }
#     ema.apply_shadow()
#     # params now equal to shadow
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             assert torch.allclose(p.data, ema.shadow[name])
#     ema.restore()
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             assert torch.allclose(p.data, backup_before[name])


# @pytest.mark.parametrize(
#     "schedule", [GeometricSchedule(), LinearSchedule()]
# )  # exercise both schedules via ScoreModel
# def test_scoremodel_forward_and_loss_and_trainstep(schedule):
#     d = 4
#     net = TinyNet(dim=d)
#     model = ScoreModel(network=net, schedule=schedule, device="cpu")

#     # forward normalization: if network outputs zeros, output is zeros
#     x = torch.zeros(2, d)
#     t = torch.full((2,), 0.5)
#     with torch.no_grad():
#         # zero out weights/bias such that TinyNet returns zeros
#         for p in net.parameters():
#             p.zero_()
#         out = model.forward(x, t)
#         assert out.shape == x.shape
#         assert torch.count_nonzero(out) == 0

#     # loss is scalar and has grad
#     batch = make_batch(n=8, d=d)
#     loss = model.loss_fn(batch)
#     assert loss.dim() == 0
#     loss.backward()
#     # optimizer step path without scheduler
#     opt = SGD(model.parameters(), lr=1e-2)
#     loss2 = model.train_step(batch, opt)
#     assert isinstance(loss2.item(), float)

#     # train_step with scheduler
#     opt = SGD(model.parameters(), lr=1e-2)
#     sch = StepLR(opt, step_size=1, gamma=0.9)
#     _ = model.train_step(batch, opt, scheduler=sch)


# def test_scoremodel_save_and_load(tmp_path):
#     d = 3
#     net = TinyNet(dim=d)
#     model = ScoreModel(network=net, device="cpu")

#     # mutate some state and ema shadow
#     model.history = ["epoch0", {"loss": 1.23}]
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             model.ema.shadow[name] = p.data.clone() + 0.5  # nontrivial shadow

#     # save
#     f = tmp_path / "weights.pth"
#     model._save_weights(str(f))
#     assert f.exists()

#     # create a fresh model and load
#     new_model = ScoreModel(network=TinyNet(dim=d), device="cpu")
#     new_model._load_weights(str(f))

#     # state_dict equalities
#     for k in model.state_dict().keys():
#         assert torch.allclose(model.state_dict()[k], new_model.state_dict()[k])
#     # history loaded
#     assert new_model.history == model.history
#     # ema shadow loaded
#     assert set(new_model.ema.shadow.keys()) == set(model.ema.shadow.keys())
#     for k in new_model.ema.shadow:
#         assert torch.allclose(new_model.ema.shadow[k], model.ema.shadow[k])


# def test_scoremodel_load_missing_optional_keys(tmp_path):
#     # covers _load_weights fallbacks for missing HISTORY/EMA
#     d = 3
#     m1 = ScoreModel(network=TinyNet(dim=d), device="cpu")
#     f = tmp_path / "minimal.pth"
#     torch.save({"MODEL_STATE": m1.state_dict()}, str(f))

#     m2 = ScoreModel(network=TinyNet(dim=d), device="cpu")
#     m2._load_weights(str(f))

#     for k in m1.state_dict().keys():
#         assert torch.allclose(m1.state_dict()[k], m2.state_dict()[k])
#     assert m2.history == []  # fallback exercised
#     assert (
#         isinstance(m2.ema.shadow, dict) and len(m2.ema.shadow) == 0
#     )  # fallback exercised


# def test_energy_based_model_forward_and_loss():
#     d = 5
#     net = TinyNet(dim=d)
#     ebm = EnergyBasedModel(network=net, device="cpu")

#     x = make_batch(n=3, d=d)
#     t = torch.rand(3)

#     # direct forward (create_graph default False)
#     g = ebm.forward(x.clone(), t)
#     assert g.shape == x.shape

#     # loss path (uses create_graph=True internally)
#     batch = make_batch(n=6, d=d)
#     loss = ebm.loss_fn(batch)
#     assert loss.dim() == 0
#     loss.backward()


# def test_energy_based_model_forward_create_graph_flag():
#     # explicitly exercise create_graph=True branch
#     d = 2
#     ebm = EnergyBasedModel(network=TinyNet(dim=d), device="cpu")
#     x = make_batch(n=2, d=d)
#     t = torch.rand(2)
#     g = ebm.forward(x, t, create_graph=True)
#     assert g.shape == x.shape
#     s = g.sum()
#     # backprop through the created graph to ensure it's retained
#     grad = torch.autograd.grad(outputs=s, inputs=x, create_graph=True)


# def test_flow_matching_model_end_to_end(tmp_path):
#     d = 3
#     net = TinyNet(dim=d)
#     fmm = FlowMatchingModel(network=net, device="cpu")

#     # forward returns same shape
#     x = make_batch(n=2, d=d)
#     t = torch.rand(2, 1)
#     y = fmm.forward(x, t)
#     assert y.shape == x.shape

#     # loss is scalar and has gradient
#     batch = make_batch(n=8, d=d)
#     loss = fmm.loss_fn(batch)
#     assert loss.dim() == 0
#     loss.backward()

#     # train_step without scheduler
#     opt = SGD(fmm.parameters(), lr=1e-2)
#     l = fmm.train_step(batch, opt)
#     assert isinstance(l.item(), float)

#     # train_step with scheduler path
#     opt = SGD(fmm.parameters(), lr=1e-2)
#     sch = StepLR(opt, step_size=1, gamma=0.9)
#     _ = fmm.train_step(batch, opt, scheduler=sch)

#     # save/load round trip
#     f = tmp_path / "fmm_weights.pth"
#     fmm.history = ["h1", 2]
#     for name, p in fmm.named_parameters():
#         if p.requires_grad:
#             fmm.ema.shadow[name] = p.data.clone() + 0.1
#     fmm._save_weights(str(f))

#     fmm2 = FlowMatchingModel(network=TinyNet(dim=d), device="cpu")
#     fmm2._load_weights(str(f))

#     for k in fmm.state_dict().keys():
#         assert torch.allclose(fmm.state_dict()[k], fmm2.state_dict()[k])
#     assert fmm2.history == fmm.history
#     for k in fmm.ema.shadow:
#         assert torch.allclose(fmm.ema.shadow[k], fmm2.ema.shadow[k])
