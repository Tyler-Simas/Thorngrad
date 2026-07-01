import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64) # normalize arrays the same
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op # just for visualization
        self._backward = lambda: None
    
    @property
    def shape(self):
        return self.data.shape
    
    # ---------- Dunder Methods ------------ #

    def __add__(self, other):
        from . import ops
        return ops.add(self, other)
    
    def __sub__(self, other):
        from . import ops
        return ops.sub(self, other)
    
    def __mul__(self, other):
        from . import ops
        return ops.mul(self, other)
    
    def __truediv__(self, other):
        from . import ops
        return ops.truediv(self, other)
    
    def __neg__(self):
        from . import ops
        return ops.neg(self)
    
    def __pow__(self, other):
        from . import ops
        return ops.pow(self, other)
    
    def __matmul__(self, other):
        from . import ops
        return ops.matmul(self, other)
    
    # ---------- Right Side Dunder Methods ------------ #

    def __radd__(self, other):
        from . import ops
        return ops.add(self, other)
    
    def __rsub__(self, other):
        from . import ops
        return ops.sub(other, self)
    
    def __rmul__(self, other):
        from . import ops
        return ops.mul(self, other)
    
    def __rtruediv__(self, other):
        from . import ops
        return ops.truediv(other, self)
    
    # --------- Some Matrix Operations -----------
    def sum(self, axis=None, keepdims=False):
        from . import ops
        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from . import ops
        return ops.mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, new_shape):
        from . import ops
        return ops.reshape(self, new_shape)
    
    # -------- Functions ---------
    def relu(self):
      from . import functional as F
      return F.relu(self)

    def sigmoid(self):
      from . import functional as F
      return F.sigmoid(self)

    def softmax(self, axis=-1):
      from . import functional as F
      return F.softmax(self, axis=axis)

    # ------ Backward ------- #

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad = {self.grad})"
    