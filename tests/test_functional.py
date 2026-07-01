import numpy as np
import torch
import torch.nn.functional as F
import pytest

from thorngrad_pkg.tensor import Tensor
from thorngrad_pkg import functional as nf


# ---------- ReLU ---------- #

@pytest.mark.parametrize("a_vals, expected", [
    ([1.0, -2.0, 3.0], [1.0, 0.0, 3.0]),
    ([-5.0, -0.5, 0.0], [0.0, 0.0, 0.0]),
    ([[1.0, -1.0], [-2.0, 2.0]], [[1.0, 0.0], [0.0, 2.0]]),
])
def test_relu_forward(a_vals, expected):
    a = Tensor(a_vals)
    out = nf.relu(a)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


@pytest.mark.parametrize("a_vals", [
    ([1.0, -2.0, 3.0]),
    ([-5.0, -0.5, 0.0, 2.5]),
    ([[1.0, -1.0], [-2.0, 2.0]]),
])
def test_relu_backward_vs_torch(a_vals):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = nf.relu(a)
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    outt = F.relu(at)
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5)


# ---------- Sigmoid ---------- #

@pytest.mark.parametrize("a_vals", [
    ([0.0, 1.0, -1.0]),
    ([2.5, -3.0, 0.5]),
    ([[1.0, -1.0], [2.0, -2.0]]),
])
def test_sigmoid_forward(a_vals):
    a_np = np.array(a_vals)
    expected = 1 / (1 + np.exp(-a_np))

    a = Tensor(a_np.copy())
    out = nf.sigmoid(a)
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


@pytest.mark.parametrize("a_vals", [
    ([0.0, 1.0, -1.0]),
    ([2.5, -3.0, 0.5]),
    ([[1.0, -1.0], [2.0, -2.0]]),
])
def test_sigmoid_backward_vs_torch(a_vals):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = nf.sigmoid(a)
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    outt = torch.sigmoid(at)
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5)


# ---------- Softmax ---------- #

@pytest.mark.parametrize("a_vals, axis", [
    ([1.0, 2.0, 3.0], -1),
    ([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], -1),
    ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 0),
])
def test_softmax_forward(a_vals, axis):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = nf.softmax(a, axis=axis)

    # softmax should sum to 1 along the given axis
    sums = out.data.sum(axis=axis)
    np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-6)

    # and match torch's softmax directly
    at = torch.tensor(a_np)
    expected = F.softmax(at, dim=axis).numpy()
    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("a_vals, axis", [
    ([1.0, 2.0, 3.0], -1),
    ([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], -1),
    ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 0),
])
def test_softmax_backward_vs_torch(a_vals, axis):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = nf.softmax(a, axis=axis)
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    outt = F.softmax(at, dim=axis)
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5, atol=1e-8)