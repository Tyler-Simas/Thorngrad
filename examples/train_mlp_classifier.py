"""
Train a small MLP with thorngrad to classify the two-moons dataset.
Demonstrates a nonlinear classification via ReLU, a NumPy-backed autograd engine,
and SGD optimization with momentum.
"""

import numpy as np
import matplotlib.pyplot as plt

from thorngrad_pkg.tensor import Tensor
from thorngrad_pkg.nn import Linear, ReLU, Sequential
from thorngrad_pkg.optim import SGD

def make_moons(n_samples = 200, noise = 0.1, seed = 0):
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 2

    theta1 = np.linspace(0, np.pi, n_per_class)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)

    theta2 = np.linspace(0, np.pi, n_per_class)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)

    X = np.vstack([x1, x2])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X.astype(np.float64), y.reshape(-1, 1).astype(np.float64)

def main():
    x_np, y_np = make_moons(n_samples = 200, noise = 0.15, seed = 0)
    x, y = Tensor(x_np), Tensor(y_np)

    model = Sequential(
        Linear(2, 16), ReLU(),
        Linear(16, 16), ReLU(),
        Linear(16, 1)
    )

    optimizer = SGD(model.parameters(), lr=0.1, momentum = 0.9)

    epochs = 500
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        diff = pred - y
        loss = (diff * diff).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
        if epoch % 50 == 0:
            print(f'epoch: {epoch} | loss: {loss.data.item():.4f}')
    
    print(f'Final Loss: {losses[-1]:.4f}')

     # ---- Plot loss curve ----
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(losses)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE loss")
    ax1.set_title("Training loss (thorngrad MLP on two-moons)")
    fig1.tight_layout()
    fig1.savefig("examples/loss_curve.png", dpi=150)

    # ---- Plot decision boundary ----
    x_min, x_max = x_np[:, 0].min() - 0.5, x_np[:, 0].max() + 0.5
    y_min, y_max = x_np[:, 1].min() - 0.5, x_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_pred = model(Tensor(grid)).data.reshape(xx.shape)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.contourf(xx, yy, grid_pred, levels=50, cmap="RdBu", alpha=0.6)
    ax2.scatter(x_np[:, 0], x_np[:, 1], c=y_np.ravel(), cmap="RdBu",
                edgecolors="k", linewidths=0.5)
    ax2.set_title("Decision boundary (thorngrad MLP on two-moons)")
    fig2.tight_layout()
    fig2.savefig("examples/decision_boundary.png", dpi=150)

    print("Saved examples/loss_curve.png and examples/decision_boundary.png")


if __name__ == "__main__":
    main()