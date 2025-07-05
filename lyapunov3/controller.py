import torch
import torch.nn as nn

A = torch.tensor([[0., 1.],
                  [-2., -3.]])
B = torch.tensor([[0.],
                  [1.]])
Q = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 2,bias=False),
            nn.ReLU(),
            nn.Linear(2,1,bias=False)
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

def V(x):  # Lyapunov function: x^2
    # x: (N, 2), Q: (2, 2)
    xQ = x @ Q
    return (xQ * x).sum(dim=1, keepdim=True)

def dVdt(x, u):
    # x: (N, 2), u: (N, 1)
    x_dot = x @ A.T + u @ B.T         # shape: (N, 2)
    xQ = x @ Q                        # shape: (N, 2)
    vdot = 2 * (xQ * x_dot).sum(dim=1, keepdim=True)  # scalar per sample
    return vdot