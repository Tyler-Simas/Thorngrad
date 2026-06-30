import numpy as np
from .tensor import Tensor

# -------- Broadcasting Helper ----------- #

def _reduce_grad_to_shape(grad, target_shape):
    """
    Sum a gradient array back down to target shape, undoing the 
    broadcasting that happened during the forward pass.
    """
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


def _to_tensor(x): # Turn object to tensor if it is not
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad = False)

# ---------- Binary Ops -------------- #

def add(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data + b.data, _children=(a, b), _op='+')

    def _backward():
        if a.requires_grad:
            a.grad += _reduce_grad_to_shape(out.grad, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_grad_to_shape(out.grad, b.data.shape)
    out._backward = _backward
    return out

def sub(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data - b.data, _children=(a, b), _op='-')

    def _backward():
        if a.requires_grad:
            a.grad += _reduce_grad_to_shape(out.grad, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_grad_to_shape(-out.grad, b.data.shape)
    out._backward = _backward
    return out

def mul(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data * b.data, _children=(a, b), _op='*')

    def _backward():
        if a.requires_grad:
            a.grad += _reduce_grad_to_shape(b.data * out.grad, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_grad_to_shape(a.data * out.grad, b.data.shape)
    out._backward = _backward
    return out

def truediv(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data / b.data, _children=(a, b), _op='/')

    def _backward():
        if a.requires_grad:
            a.grad += _reduce_grad_to_shape(out.grad / b.data, a.data.shape)
        if b.requires_grad:
            b.grad += _reduce_grad_to_shape(-a.data * out.grad / (b.data**2), b.data.shape)
    out._backward = _backward
    return out

def matmul(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out = Tensor(a.data @ b.data, _children=(a, b), _op='@')

    def _backward():
        if a.requires_grad:
            a.grad += out.grad @ b.data.T
        if b.requires_grad:
            b.grad += a.data.T @ out.grad
    out._backward = _backward
    return out


# ------ Unary Ops ------- #

def neg(a):
    a = _to_tensor(a)
    out = Tensor(-a.data, _children=(a,), _op='neg')

    def _backward():
        if a.requires_grad:
            a.grad += -out.grad
    out._backward = _backward
    return out

def pow(a, exponent):
    a = _to_tensor(a)
    out = Tensor(a.data ** exponent, _children=(a,), _op=f'**{exponent}')

    def _backward():
        if a.requires_grad:
            a.grad += (exponent * a.data ** (exponent - 1)) * out.grad
    out._backward = _backward
    return out