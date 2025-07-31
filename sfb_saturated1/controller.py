import torch
import torch.nn as nn

As=torch.tensor([[1., 2.],
                  [-3., -4.]])
Bs=torch.tensor([[0.],
                  [1.]])
Cs=torch.tensor([[1.0, 0.0]])

A =  torch.cat([
        torch.cat([As, torch.zeros((2, 1))], dim=1),
        torch.cat([-Cs, torch.zeros((1, 1))], dim=1)
    ], dim=0)
B = torch.cat([Bs, torch.zeros((1, 1))], dim=0)
C = torch.cat([Cs, torch.zeros((1, 1))], dim=1)
Br=torch.tensor([[0.],
                 [0.],
                [1.]])
Q = torch.tensor([[0.001, 0.0, 0.0],
                  [0.0, 0.001, 0.0],
                  [0.0, 0.0, 0.001]])

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,1,bias=False),
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

def LQR_Controller():
    """
    Compute the LQR controller gain K.
    """
    import numpy as np
    import control        
    
    K, S, E = control.lqr(A, B, torch.tensor([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]), 10.0 * np.eye(1))  # LQR gain
    K=-K
    Acl=(A+B*K).numpy()
    eigs=np.linalg.eigvals(Acl)
    print("K =", K)  # [k1, k2]
    print("Eigs:", eigs)

    # Define full 3D gain manually (e.g., from extended system)
    Ktorch = torch.tensor(K, dtype=torch.float32)  # shape (1,3)
    
    return Ktorch