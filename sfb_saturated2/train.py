from matplotlib import pyplot as plt
import numpy as np
import torch

from controller import *

grid = torch.linspace(-1, 1, 11)
x1, x2= torch.meshgrid(grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)

controller = Controller()
optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3, weight_decay=1e-5)

nnP=LyapunovP()
optimizerP = torch.optim.Adam(nnP.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(1*10000+1):
    
    P = nnP()
    Vx = V(x_train, P)
    u = controller(x_train)  # your neural controller
    Vdot = dVdt(x_train, u, P)
    lyapunov_penalty = torch.relu(Vdot)

    loss = lyapunov_penalty.mean()
    loss.backward()
    optimizer.step()

    cond=(lyapunov_penalty>0).squeeze()
    count=cond.sum().item()
    if epoch % 500 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} Violations:{count}")
    if count==0:
        print(f"All violations cleared at epoch {epoch}")
        break
controller.print()
controller.save()

linear_layer = controller.net[0]  # nn.Linear(2, 1, bias=False)
weights = linear_layer.weight.data  # shape: (1, 2)
# Convert to numpy (optional)
K = weights.numpy().flatten()
Acl=(A+B*K).numpy()
eigs=np.linalg.eigvals(Acl)

print("K =", K)  # [k1, k2]
print("Eigs:", eigs)