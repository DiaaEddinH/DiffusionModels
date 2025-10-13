from unittest import mock
import numpy as np
import pytest
import matplotlib.pyplot as plt

import torch

from diffusion_models.utils import (
    set_device,
    ddp_setup,
    destroy_ddp,
    count_trainable_parameters,
    grab,
    get_activation_func,
    set_default_plot_parameters,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
        # Make bias non-trainable to test filtering
        self.lin.bias.requires_grad = False


def test_set_device_cpu_default(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        dev = set_device("cpu")
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"


def test_set_device_gpu_path(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.backends.mps, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "set_device") as m_set_dev,
    ):
        dev = set_device("gpu")
        m_set_dev.assert_called_once_with(1)
        assert dev.type == "cuda"
        assert dev.index == 1


def test_set_device_mps_path(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        dev = set_device("gpu")
        # Even if the current platform doesn't support MPS, torch.device("mps") can be constructed
        assert isinstance(dev, torch.device)
        assert dev.type == "mps"


@pytest.mark.parametrize("inp", ["", "tpu", "blah"])
def test_set_device_invalid(inp):
    with pytest.raises(AssertionError):
        set_device(inp)


def test_ddp_setup_calls_init_with_gloo_when_single_cuda():
    with (
        mock.patch.object(torch.cuda, "device_count", return_value=1),
        mock.patch("torch.distributed.init_process_group") as m_init,
    ):
        ddp_setup(use_ddp=True)
        m_init.assert_called_once()
        # first positional arg is backend=... or keyword; check kwargs
        kwargs = m_init.call_args.kwargs
        if kwargs:
            assert kwargs.get("backend") == "gloo"
        else:
            # called with positional backend
            assert m_init.call_args.args[0] == "gloo"


def test_ddp_setup_uses_nccl_when_multi_cuda():
    with (
        mock.patch.object(torch.cuda, "device_count", return_value=2),
        mock.patch("torch.distributed.init_process_group") as m_init,
    ):
        ddp_setup(use_ddp=True)
        kwargs = m_init.call_args.kwargs
        if kwargs:
            assert kwargs.get("backend") == "nccl"
        else:
            assert m_init.call_args.args[0] == "nccl"


def test_ddp_setup_noop_when_disabled():
    with mock.patch("torch.distributed.init_process_group") as m_init:
        ddp_setup(use_ddp=False)
        m_init.assert_not_called()


def test_destroy_ddp_when_initialized():
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.barrier") as m_barrier,
        mock.patch("torch.distributed.destroy_process_group") as m_destroy,
    ):
        destroy_ddp(use_ddp=True)
        m_barrier.assert_called_once()
        m_destroy.assert_called_once()


def test_destroy_ddp_not_initialized():
    with (
        mock.patch("torch.distributed.is_initialized", return_value=False),
        mock.patch("torch.distributed.barrier") as m_barrier,
        mock.patch("torch.distributed.destroy_process_group") as m_destroy,
    ):
        destroy_ddp(use_ddp=True)
        m_barrier.assert_not_called()
        m_destroy.assert_not_called()


def test_destroy_ddp_disabled():
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.barrier") as m_barrier,
        mock.patch("torch.distributed.destroy_process_group") as m_destroy,
    ):
        destroy_ddp(use_ddp=False)
        m_barrier.assert_not_called()
        m_destroy.assert_not_called()


def test_count_trainable_parameters():
    model = DummyModel()
    expected = model.lin.weight.numel()  # only weight is trainable
    assert count_trainable_parameters(model) == expected


def test_grab_tensor_to_numpy():
    x = torch.randn(3, 4, requires_grad=True)
    arr = grab(x)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, x.detach().cpu().numpy())


def test_get_activation_func_valid():
    relu_cls = get_activation_func("relu")
    assert relu_cls is torch.nn.modules.activation.ReLU
    gelu_cls = get_activation_func("GELU")
    assert gelu_cls is torch.nn.modules.activation.GELU


def test_get_activation_func_invalid():
    with pytest.raises(ValueError):
        get_activation_func("not_an_activation")


def test_set_default_plot_parameters_updates_rcparams():
    # Ensure defaults can be set without rendering
    set_default_plot_parameters()
    assert plt.rcParams["text.usetex"] is True
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["font.serif"][0] == "Times New Roman"
    assert plt.rcParams["xtick.top"] is True
    assert plt.rcParams["ytick.right"] is True
    assert plt.rcParams["image.cmap"] == "viridis"
    # Color cycle first color
    first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    assert first_color == "#4477AA"


def test_set_device_gpu_requested_no_gpu_prints_and_uses_cpu(monkeypatch, capsys):
    # No CUDA, no MPS → should print the fallback message and return CPU device
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        dev = set_device("gpu")
        captured = capsys.readouterr()
        assert (
            "Supported GPUs are not available. Setting CPU as device." in captured.out
        )
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"
