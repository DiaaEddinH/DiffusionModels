import torch

from diffusion_models.networks.networks import (
    GaussianFourierProjection,
    Embedding,
    LinearNet,
    EvenLinear,
    OddLinear,
    EvenResNet,
    ResBlock,
    ResNet,
    ConditionalLinearWrapper,
    ShiftWrapper,
    ws_conv,
    ws_convT,
    UNet,
    LinearAttention,
    AttentionBlock,
    UNetWAttention,
)


def test_gaussian_fourier_projection_shapes_and_values():
    embed_dim = 8
    proj = GaussianFourierProjection(embed_dim)

    # x as batch vector
    x = torch.tensor([0.0, 0.5, 1.0])
    out = proj(x)
    assert out.shape == (3, embed_dim)

    # At x=0, sin=0 and cos=1 for all components
    zero_out = proj(torch.tensor([0.0]))
    # first half sin part should be zeros
    assert torch.allclose(zero_out[0, : embed_dim // 2], torch.zeros(embed_dim // 2))
    # second half cos part should be ones
    assert torch.allclose(zero_out[0, embed_dim // 2 :], torch.ones(embed_dim // 2))

    # Buffer W exists and is not a parameter (i.e., not trainable)
    assert hasattr(proj, "W")
    names = [n for n, _ in proj.named_parameters()]
    assert "W" not in names


def test_gaussian_fourier_projection_W_is_buffer_not_trainable():
    proj = GaussianFourierProjection(embed_dim=8)
    # ensure W is a buffer and does not require grad
    assert isinstance(proj.W, torch.Tensor)
    assert proj.W.requires_grad is False


def test_gaussian_fourier_projection_odd_embed_dim_current_behavior():
    # Document current behavior: odd embed_dim yields output with size
    # 2 * floor(embed_dim/2), i.e., smaller than requested.
    embed_dim = 7
    proj = GaussianFourierProjection(embed_dim)
    x = torch.tensor([0.0, 1.0])
    out = proj(x)
    assert out.shape == (2, 2 * (embed_dim // 2))  # currently (2, 6)


def test_embedding_output_and_grad():
    time_channels = 16
    emb = Embedding(embed_dim=time_channels)
    x = torch.randn(4, requires_grad=True)
    y = emb(x)
    assert y.shape == (4, time_channels)
    y.sum().backward()
    # gradient must have flowed to input x
    assert x.grad is not None


def test_linear_net_time_conditioning_changes_output():
    net = LinearNet(in_channels=3, channels=[8, 5], time_channels=12)
    x = torch.randn(7, 3)
    t1 = torch.zeros(7)
    t2 = torch.ones(7)

    y1 = net(x, t1)
    y2 = net(x, t2)

    assert y1.shape == (7, 3)
    assert y2.shape == (7, 3)
    # Same x but different t should generally produce different outputs
    assert not torch.allclose(y1, y2)


def test_even_linear_is_even_function():
    net = EvenLinear(in_channels=4, channels=[6, 6], time_channels=10)
    x = torch.randn(5, 4)
    t = torch.randn(5)
    y = net(x, t)
    y_neg = net(-x, t)

    assert y.shape == (5, 4)
    assert torch.allclose(y, y_neg, atol=1e-5)


def test_odd_linear_is_odd_function():
    net = OddLinear(in_channels=2, channels=[5, 5], time_channels=8)
    x = torch.randn(6, 2)
    t = torch.randn(6)
    y = net(x, t)
    y_neg = net(-x, t)

    assert y.shape == (6, 2)
    assert torch.allclose(y, -y_neg, atol=1e-5)


def test_conditional_linear_wrapper_changes_with_y():
    base = LinearNet(in_channels=2, channels=[4, 4], time_channels=6)
    cond = ConditionalLinearWrapper(base, label_dim=12)

    x = torch.randn(3, 2)
    t = torch.zeros(3)
    y1 = torch.zeros(3)
    y2 = torch.ones(3)

    out1 = cond(x, t, y1)
    out2 = cond(x, t, y2)

    assert out1.shape == (3, 2)
    assert not torch.allclose(out1, out2)


def test_shift_wrapper_applies_expected_shift():
    # Dummy network that simply returns the (possibly shifted) x it receives
    class EchoNet(torch.nn.Module):
        def forward(self, x, t, y):
            return x

    wrapper = ShiftWrapper(EchoNet())

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2,2)
    t = torch.zeros(2)
    y = torch.tensor([0.5, -1.0])  # shape (2,)

    out = wrapper(x, t, y)
    shift_vec = torch.stack([y, -y], dim=-1)
    expected = x - shift_vec
    assert torch.allclose(out, expected)


def test_ws_conv_and_ws_convT_have_weight_norm():
    c = ws_conv(3, 5, kernel_size=3, padding=1)
    ct = ws_convT(4, 2, kernel_size=3)

    # weight_norm should register a parametrization on the weight
    assert hasattr(c, "parametrizations") and "weight" in c.parametrizations
    assert hasattr(ct, "parametrizations") and "weight" in ct.parametrizations


def test_unet_forward_shape():
    net = UNet(in_channels=2, channels=[8, 16], time_channels=12)
    x = torch.randn(2, 2, 8, 8)
    t = torch.randn(2)
    y = net(x, t)
    assert y.shape == (2, 2, 8, 8)


def test_linear_attention_forward_shape_and_finite():
    attn = LinearAttention(channels=12, heads=3, dim_head=4)
    x = torch.randn(2, 12, 5, 5)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attention_block_residual_and_grad():
    block = AttentionBlock(channels=8, num_heads=4, reduction=2)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_unet_with_attention_forward_shape():
    net = UNetWAttention(in_channels=2, channels=[8, 16], time_channels=10)
    x = torch.randn(1, 2, 8, 8)
    t = torch.randn(1)
    y = net(x, t)
    assert y.shape == (1, 2, 8, 8)


def test_residual_block_shape_and_finite():
    block = ResBlock(channels=2, time_channels=10)
    x = torch.randn(6, 2)
    t = torch.randn(10)
    y = block(x, t)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_resnet_forward_shape():
    net = ResNet(in_channels=2, channels=[8, 8], time_channels=16)
    x = torch.randn(8, 2)
    t = torch.randn(8)
    y = net(x, t)
    assert y.shape == (8, 2)


def test_even_resnet_is_even_function():
    net = EvenResNet(in_channels=4, channels=[6, 6], time_channels=10)
    x = torch.randn(5, 4)
    t = torch.randn(5)
    y = net(x, t)
    y_neg = net(-x, t)

    assert y.shape == (5, 4)
    assert torch.allclose(y, y_neg, atol=1e-5)


def test_no_bias_tanh_unet_is_odd_function():
    net = UNet(in_channels=4, channels=[6, 6], time_channels=10, activation=torch.nn.Tanh(), bias=False)
    x = torch.randn(5, 4, 8, 8)
    t = torch.randn(5)
    y = net(x, t)
    y_neg = -net(-x, t)

    assert y.shape == (5, 4, 8, 8)
    assert torch.allclose(y, y_neg, atol=1e-5)
