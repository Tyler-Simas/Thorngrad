import numpy as np
import pytest

from thorngrad_pkg.tensor import Tensor
from thorngrad_pkg.nn import Module, Linear, ReLU, Sigmoid, Softmax, Sequential


# ---------- Linear ---------- #

def test_linear_forward_shape():
    layer = Linear(in_features=4, out_features=3)
    x = Tensor(np.random.randn(5, 4))  # batch of 5, 4 features each
    out = layer(x)
    assert out.data.shape == (5, 3)


def test_linear_parameters():
    layer = Linear(in_features=4, out_features=3)
    params = layer.parameters()
    assert len(params) == 2  # weight and bias
    assert params[0].data.shape == (4, 3)   # weight
    assert params[1].data.shape == (3,)     # bias


def test_linear_no_bias():
    layer = Linear(in_features=4, out_features=3, bias=False)
    params = layer.parameters()
    assert len(params) == 1  # only weight, no bias


def test_linear_backward_produces_gradients():
    layer = Linear(in_features=4, out_features=3)
    x = Tensor(np.random.randn(2, 4))
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert not np.allclose(layer.weight.grad, 0)
    assert not np.allclose(layer.bias.grad, 0)


# ---------- Activations ---------- #

def test_relu_module_matches_functional():
    from thorngrad_pkg import functional as F
    x_data = [-2.0, -1.0, 0.0, 1.0, 2.0]

    a = Tensor(x_data)
    out_module = ReLU()(a)

    b = Tensor(x_data)
    out_func = F.relu(b)

    np.testing.assert_allclose(out_module.data, out_func.data)


def test_sigmoid_module_output_range():
    x = Tensor(np.random.randn(10) * 5)  # wide range of values
    out = Sigmoid()(x)
    assert np.all(out.data > 0) and np.all(out.data < 1)


def test_softmax_module_sums_to_one():
    x = Tensor(np.random.randn(3, 5))
    out = Softmax(axis=-1)(x)
    sums = out.data.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones(3), rtol=1e-6)


# ---------- Sequential ---------- #

def test_sequential_forward_shape():
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 3),
    )
    x = Tensor(np.random.randn(5, 4))
    out = model(x)
    assert out.data.shape == (5, 3)


def test_sequential_parameters_collects_all_layers():
    model = Sequential(
        Linear(4, 8),
        ReLU(),          # contributes no parameters
        Linear(8, 3),
    )
    params = model.parameters()
    # Linear(4,8) -> weight+bias (2), Linear(8,3) -> weight+bias (2) = 4 total
    assert len(params) == 4


def test_sequential_backward_reaches_all_layers():
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 3),
    )
    x = Tensor(np.random.randn(2, 4))
    out = model(x)
    loss = out.sum()
    loss.backward()

    for p in model.parameters():
        assert not np.allclose(p.grad, 0)


def test_sequential_zero_grad_resets_all_gradients():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    x = Tensor(np.random.randn(2, 4))
    out = model(x)
    loss = out.sum()
    loss.backward()

    model.zero_grad()

    for p in model.parameters():
        np.testing.assert_allclose(p.grad, np.zeros_like(p.grad))