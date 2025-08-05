import torch
import torch.nn as nn

A=torch.tensor([[1., 2.],
                  [-3., -4.]])
B=torch.tensor([[0.],
                  [1.]])

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,1,bias=False),
        )
        
    def forward(self, x):
        return self.net(x)
    
    def print(self):
        for m in self.net:
            for name,param in m.named_parameters():
                print(f"{name}:\n{param.data}\n")
    def save(self,filename="controller.pt"):
        torch.save(self.net.state_dict(), filename)

    def load(self, filename="controller.pt"):
        self.net.load_state_dict(torch.load(filename))
        self.net.eval() 

def V(x,P):  # Lyapunov function: x^2
    # x: (N, 2), P: (2, 2)
    xP = x @ P
    res=(xP * x).sum(dim=1, keepdim=True)
    return res

def dVdt(x, u,P):
    # x: (N, 2), u: (N, 1)
    x_dot = x @ A.T + u @ B.T         # shape: (N, 2)
    xP = x @ P                       # shape: (N, 2)
    vdot = 2 * (xP * x_dot).sum(dim=1, keepdim=True)  # scalar per sample
    return vdot

