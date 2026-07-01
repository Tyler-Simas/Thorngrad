import numpy as np
from .tensor import Tensor
from . import functional as F

class Module:  
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(1 / in_features))
        self.bias = Tensor(np.zeros(out_features)) if bias else None
    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]

class ReLU(Module):
    def __call__(self, x):
        return F.relu(x)
    
class Sigmoid(Module):
    def __call__(self, x):
        return F.sigmoid(x)
    
class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        return F.softmax(x, axis = self.axis)
    

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers: # passes each layer through the next one
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params