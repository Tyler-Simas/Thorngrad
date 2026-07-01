import numpy as np
import pytest

from thorngrad_pkg.tensor import Tensor
from thorngrad_pkg.nn import Linear, ReLU, Sequential
from thorngrad_pkg.optim import SGD


def test_sgd_reduces_loss_on_toy_regression():
    np.random.seed(0)

    # Learn y = 2x + 1
    x_np = np.random.randn(20, 1)
    y_np = 2 * x_np + 1

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Linear(in_features=1, out_features=1)
    optimizer = SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x)
        diff = pred - y
        loss = (diff * diff).mean()   # MSE
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    # Loss should be lower at end than at start
    assert losses[-1] < losses[0] * 0.1


def test_sgd_trains_small_mlp_on_toy_classification():
    np.random.seed(0)

    # Separate two clusters of points
    x_np = np.vstack([
        np.random.randn(10, 2) + np.array([2, 2]),
        np.random.randn(10, 2) + np.array([-2, -2]),
    ])
    y_np = np.vstack([np.ones((10, 1)), np.zeros((10, 1))])

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
    )
    optimizer = SGD(model.parameters(), lr=0.05)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x)
        diff = pred - y
        loss = (diff * diff).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    assert losses[-1] < losses[0] * 0.2


def test_sgd_momentum_still_reduces_loss():
    np.random.seed(1)

    x_np = np.random.randn(20, 1)
    y_np = 3 * x_np - 2

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Linear(in_features=1, out_features=1)
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x)
        diff = pred - y
        loss = (diff * diff).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    assert losses[-1] < losses[0] * 0.1


def test_zero_grad_actually_clears_gradients_between_steps():
    x = Tensor(np.random.randn(5, 3))
    y = Tensor(np.random.randn(5, 1))

    model = Linear(3, 1)
    optimizer = SGD(model.parameters(), lr=0.01)

    pred = model(x)
    loss = ((pred - y) * (pred - y)).mean()
    loss.backward()

    # Gradients should be non-zero immediately after backward
    assert not np.allclose(model.weight.grad, 0)

    optimizer.zero_grad()

    # and zero again after zero_grad, before new backward call
    np.testing.assert_allclose(model.weight.grad, np.zeros_like(model.weight.grad))