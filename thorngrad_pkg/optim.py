import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self): # PyTorch includes this redundancy, optimzer.zero_grad() is good to keep
        for p in self.params:
            p.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + p.grad # velocity 
            p.data -= self.lr * self.velocities[i]