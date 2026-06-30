import numpy as np
from .tensor import Tensor

# -------- Broadcasting Helper ----------- #

def _reduce_grad_to_shape(grad, target_shape):
    pass

def _to_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad = False)

# ---------- Binary Ops -------------- #

def add(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data + b.data, _children=(a, b), _op='+')

    def _backward():
        if a.requires_grad:
            a.grad += _reduce_grad_to_shape(out.grad, a.data.shape)
