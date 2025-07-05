import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2,bias=False),
            nn.ReLU(),
            nn.Linear(2, 1,bias=False)
        )
        self.net = nn.Sequential(
            nn.Linear(1, 1,bias=False)
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
    return x**2

def dVdt(x, u):
    return 2 * x * (2*x + u)  # derivative of V(x) = x^2
