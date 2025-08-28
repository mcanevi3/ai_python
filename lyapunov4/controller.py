import torch
import torch.nn as nn


class LyapunovNet(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.L = nn.Parameter(torch.randn(n, n))

    def forward(self):
        L = torch.tril(self.L)
        P = L @ L.T
        return P #+torch.eye(P.shape[0]) * 1e-2  # Add small identity matrix for numerical stability
   
class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1,bias=False),
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

def V(P,x):
    return (x@P@x.T).diagonal()

def Vdot(P,x,xdot):
    return xdot@P@x.T+x@P@xdot.T

def sign_tanh(x,alpha=100):
    return torch.tanh(alpha*x)

def cost(u):
    return 0.5+0.5*sign_tanh(u)

