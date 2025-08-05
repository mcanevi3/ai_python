import torch
import torch.nn as nn

class LyapunovNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.L_elements = nn.Parameter(torch.randn(n, n))
        # self.L_elements = nn.Parameter(torch.eye(n))

    def get_P(self):
        L = torch.tril(self.L_elements)
        P = L @ L.T
        return P+torch.eye(P.shape[0]) * 1e-2  # Add small identity matrix for numerical stability
    def get_V(self, x):
        P = self.get_P()
        V = (x @ P @ x.T)
        return V
    def get_Vdot(self, xdot, x):
        P = self.get_P()
        dotV = (xdot @ P @ x.T)+ (x @ P @ xdot.T)
        return dotV
    def forward(self, xdot,x):
        return self.get_Vdot(xdot,x)
