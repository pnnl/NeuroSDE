import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Drift1D(nn.Module):
    """g_theta(x,u): [*,2] -> [*,1]"""
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        dims = [2] + [hidden]*layers + [1]
        self.net = MLP(dims)
    def forward(self, xu): return self.net(xu)

class DriftND(nn.Module):
    """g_theta(x,u): [*, nx+nu] -> [*, nx]"""
    def __init__(self, nx=2, nu=1, hidden=128, layers=3):
        super().__init__()
        dims = [nx+nu] + [hidden]*layers + [nx]
        self.net = MLP(dims)
    def forward(self, xu): return self.net(xu)
