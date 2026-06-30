import numpy as np
import torch
import pytest

from thorngrad_pkg.tensor import Tensor


# --------- Numerical Gradient Check (Helper Function) ---------

def numerical_grad(f, x, h=1e-6):
    """
    Estimate df/dx_i for each element of x using central differences.
    f takes a plain np.ndarray and returns a scalar.
    """

    grad = np.zeros_like(x)
    iterations = np.nditer(x, flags = ['multi_index'])
    for _ in iterations:
        idx = iterations.multi_index
        orig = x[idx]

        x[idx] = orig + h
        f_plus = f(x)

        x[idx] = orig - h
        f_minus = f(x)

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad

# ------- Test: Basic Forward Correctness --------

@pytest.mark.parametrize("a_vals, b_vals, expected", [
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]),
    ([0.5, 1.5],      [2.25, 3.75],    [2.75, 5.25]),
    ([-1.2, 3.4],     [0.1, -0.4],     [-1.1, 3.0]),
    ([0.0, 0.0],      [0.0, 0.0],      [0.0, 0.0]),
])
def test_add_forward(a_vals, b_vals, expected):
    a, b = Tensor(a_vals), Tensor(b_vals)
    out = a + b
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

@pytest.mark.parametrize("a_vals, b_vals, expected", [
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 10.0, 18.0]),
    ([0.5, 1.5],      [2.0, 4.0],      [1.0, 6.0]),
    ([-1.2, 3.4],     [0.1, -0.4],     [-0.12, -1.36]),
    ([0.0, 5.0],      [10.0, 0.0],     [0.0, 0.0]),
])
def test_mul_forward(a_vals, b_vals, expected):
    a = Tensor(a_vals)
    b = Tensor(b_vals)
    out = a * b
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# TEST: subtraction
@pytest.mark.parametrize("a_vals, b_vals, expected", [
    ([5.0, 8.0, 10.0], [2.0, 3.0, 4.0], [3.0, 5.0, 6.0]),
    ([0.5, 1.5], [2.25, 3.75], [-1.75, -2.25]),
    ([-1.2, 3.4], [0.1, -0.4], [-1.3, 3.8]),
    ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
])
def test_sub_forward(a_vals, b_vals, expected):
    a = Tensor(a_vals)
    b = Tensor(b_vals)
    out = a - b
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# TEST: division
@pytest.mark.parametrize("a_vals, b_vals, expected", [
    ([4.0, 9.0, 16.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]),
    ([10.0, 5.0], [4.0, 2.0], [2.5, 2.5]),
    ([-6.0, 3.0], [2.0, -1.5], [-3.0, -2.0]),
    ([1.0, 1.0], [0.5, 4.0], [2.0, 0.25]),
])
def test_truediv_forward(a_vals, b_vals, expected):
    a = Tensor(a_vals)
    b = Tensor(b_vals)
    out = a / b
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# TEST: rsub
@pytest.mark.parametrize("const, b_vals, expected", [
    (10.0, [2.0, 3.0, 4.0], [8.0, 7.0, 6.0]),
    (5.0, [1.5, 2.5], [3.5, 2.5]),
    (0.0, [3.0, -2.0], [-3.0, 2.0]),
    (1.0, [[2.0, 4.0], [8.0, 0.5]], [[-1.0, -3.0], [-7.0, 0.5]]),
])
def test_rsub_forward(const, b_vals, expected):
    b = Tensor(b_vals)
    out = const - b   # exercises __rsub__
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# TEST: rtruediv
@pytest.mark.parametrize("const, b_vals, expected", [
    (10.0, [2.0, 4.0, 5.0], [5.0, 2.5, 2.0]),
    (6.0, [3.0, 2.0], [2.0, 3.0]),
    (1.0, [[2.0, 4.0], [8.0, 0.5]], [[0.5, 0.25], [0.125, 2.0]]),
])
def test_rtruediv_forward(const, b_vals, expected):
    b = Tensor(b_vals)
    out = const / b   # exercises __rtruediv__
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# TEST: Matrix Multiplication
@pytest.mark.parametrize("a_vals, b_vals, expected", [
    (
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[19.0, 22.0], [43.0, 50.0]],
    ),
    (
        [[1.0, 0.0], [0.0, 1.0]],   # identity matrix
        [[2.5, 3.5], [4.5, 5.5]],
        [[2.5, 3.5], [4.5, 5.5]],
    ),
    (
        [[2.0, 0.0, 1.0]],          # (1,3) @ (3,1) -> (1,1)
        [[1.0], [2.0], [3.0]],
        [[5.0]],
    ),
])
def test_matmul_forward(a_vals, b_vals, expected):
    a = Tensor(np.array(a_vals))
    b = Tensor(np.array(b_vals))
    out = a @ b
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)

# ----------------- Test: Gradients vs. PyTorch --------------

def _assert_grads_match(a, b, at, bt, rtol=1e-5): # Helper function
    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=rtol)
    np.testing.assert_allclose(b.grad, bt.grad.numpy(), rtol=rtol)


@pytest.mark.parametrize("a_vals, b_vals", [
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ([0.5, -1.5, 2.25], [3.0, 3.0, 3.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]),  # same shape, 2D
    ([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0]),                   # broadcast (2,2) + (2,)
])
def test_add_backward_vs_torch(a_vals, b_vals):
    a_np, b_np = np.array(a_vals), np.array(b_vals)

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a + b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at + bt
    outt.backward(torch.ones_like(outt))

    _assert_grads_match(a, b, at, bt)


@pytest.mark.parametrize("a_vals, b_vals", [
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ([0.5, -1.5, 2.25], [3.0, 3.0, 3.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]),
    ([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0]),  # broadcast
])
def test_mul_backward_vs_torch(a_vals, b_vals):
    a_np, b_np = np.array(a_vals), np.array(b_vals)

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a * b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at * bt
    outt.backward(torch.ones_like(outt))

    _assert_grads_match(a, b, at, bt)


@pytest.mark.parametrize("a_vals, b_vals", [
    ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
    ([[1.0, 0.0], [0.0, 1.0]], [[2.5, 3.5], [4.5, 5.5]]),
    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),  # (2,3) @ (3,2)
])
def test_matmul_backward_vs_torch(a_vals, b_vals):
    a_np, b_np = np.array(a_vals), np.array(b_vals)

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a @ b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at @ bt
    outt.backward(torch.ones_like(outt))

    _assert_grads_match(a, b, at, bt)


@pytest.mark.parametrize("a_vals, b_vals", [
    ([4.0, 9.0, 16.0], [2.0, 3.0, 4.0]),
    ([[10.0, 20.0], [30.0, 40.0]], [[2.0, 4.0], [5.0, 8.0]]),
    ([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0]),  # broadcast division
])
def test_truediv_backward_vs_torch(a_vals, b_vals):
    a_np, b_np = np.array(a_vals), np.array(b_vals)

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a / b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at / bt
    outt.backward(torch.ones_like(outt))

    _assert_grads_match(a, b, at, bt)


@pytest.mark.parametrize("a_vals, exponent", [
    ([2.0, 3.0, 4.0], 2),
    ([1.0, 2.0, 3.0], 3),
    ([[2.0, 4.0], [6.0, 8.0]], 2),
])
def test_pow_backward_vs_torch(a_vals, exponent):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = a ** exponent
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    outt = at ** exponent
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5)

@pytest.mark.parametrize("a_vals, b_vals", [
    ([5.0, 8.0, 10.0], [2.0, 3.0, 4.0]),
    ([0.5, -1.5, 2.25], [3.0, 3.0, 3.0]),
    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]),
    ([[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0]),  # broadcast
])
def test_sub_backward_vs_torch(a_vals, b_vals):
    a_np, b_np = np.array(a_vals), np.array(b_vals)

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a - b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at - bt
    outt.backward(torch.ones_like(outt))

    _assert_grads_match(a, b, at, bt)


@pytest.mark.parametrize("a_vals", [
    ([1.0, -2.0, 3.0]),
    ([[1.0, 2.0], [3.0, 4.0]]),
    ([0.0, 5.5, -7.25]),
])
def test_neg_backward_vs_torch(a_vals):
    a_np = np.array(a_vals)

    a = Tensor(a_np.copy())
    out = -a
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    outt = -at
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5)


@pytest.mark.parametrize("const, b_vals", [
    (10.0, [2.0, 4.0, 5.0]),
    (1.0, [[2.0, 4.0], [8.0, 0.5]]),
])
def test_rsub_backward_vs_torch(const, b_vals):
    b_np = np.array(b_vals)

    b = Tensor(b_np.copy())
    out = const - b          # exercises __rsub__
    out.backward()

    bt = torch.tensor(b_np, requires_grad=True)
    outt = const - bt
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(b.grad, bt.grad.numpy(), rtol=1e-5)


@pytest.mark.parametrize("const, b_vals", [
    (10.0, [2.0, 4.0, 5.0]),
    (1.0, [[2.0, 4.0], [8.0, 0.5]]),
])
def test_rtruediv_backward_vs_torch(const, b_vals):
    b_np = np.array(b_vals)

    b = Tensor(b_np.copy())
    out = const / b           # exercises __rtruediv__
    out.backward()

    bt = torch.tensor(b_np, requires_grad=True)
    outt = const / bt
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(b.grad, bt.grad.numpy(), rtol=1e-5)

# -------------- Test: Broadcasting -------------

def test_add_broadcast_backward():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2,2)
    b_np = np.array([10.0, 20.0])               # shape (2,) — broadcasts

    a, b = Tensor(a_np.copy()), Tensor(b_np.copy())
    out = a + b
    out.backward()

    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = at + bt
    outt.backward(torch.ones_like(outt))

    np.testing.assert_allclose(a.grad, at.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad, bt.grad.numpy(), rtol=1e-5)