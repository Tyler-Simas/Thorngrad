import numpy as np
from .tensor import Tensor
from .ops import _to_tensor

# ----- ReLU ---------
def relu(a):
    a = _to_tensor(a)
    out_tensor = Tensor(np.maximum(a.data, 0), _children = (a,), _op = 'relu')

    def _backward():
        if a.requires_grad:
            a.grad += (a.data > 0).astype(float) * out_tensor.grad
    out_tensor._backward = _backward
    return out_tensor

# ----- Sigmoid --------

def sigmoid(a):
    a = _to_tensor(a)
    out_data = 1 / (1 + np.exp(-a.data))
    out_tensor = Tensor(out_data, _children = (a,), _op = 'sigmoid')

    def _backward():
        if a.requires_grad:
            a.grad += (out_tensor.data * (1-out_tensor.data)) * out_tensor.grad
    out_tensor._backward = _backward
    return out_tensor

# ------- Softmax ---------

def softmax(a, axis=-1):
    # Following the softmax formula
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True) # shift to prevent overflow
    exp = np.exp(shifted)
    out_data = exp / np.sum(exp, axis=axis, keepdims=True)
    out_tensor = Tensor(out_data, _children=(a,), _op='softmax')

    def _backward():
        if a.requires_grad:
            g = out_tensor.grad
            s = out_tensor.data
            dot = np.sum(g*s, axis=axis, keepdims=True)
            a.grad += s * (g - dot)
    out_tensor._backward = _backward
    return out_tensor

