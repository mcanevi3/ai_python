from matplotlib import pyplot as plt
import numpy as np
import torch
from controller import *

A=torch.tensor([[1., 2.],
                  [-3., -4.]])

grid = torch.linspace(-1, 1, 4)
x1, x2= torch.meshgrid(grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)
xdot_train = A @ x_train.T
xdot_train = xdot_train.T

lyap = LyapunovNet(2)
optimizer = torch.optim.Adam(lyap.parameters(), lr=1e-4) #, weight_decay=1e-5)

for epoch in range(10*10000+1):
    optimizer.zero_grad()
    # xdot_train.requires_grad = True
    # x_train.requires_grad = True
    Vx = lyap.get_V(x_train)
    Vdot = lyap.get_Vdot(xdot_train,x_train)
    Vdot2 = Vdot + 0.1*torch.eye(Vdot.shape[0], device=Vdot.device)

    lyapunov_penalty = torch.relu(Vdot2) 
    lyapunov_penalty = torch.nn.functional.softplus(Vdot2)
    lyapunov_penalty = torch.log(1 + torch.exp(Vdot2))

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

## Save the controller and print the results
optimizer.zero_grad()
Vx = lyap.get_V(x_train)
Vdot = lyap.get_Vdot(xdot_train,x_train)
lyapunov_penalty = torch.relu(Vdot)
cond=(lyapunov_penalty>0).squeeze()
count=cond.sum().item()
loss = lyapunov_penalty.mean()
print(f"Loss: {loss.item():.6f} Violations:{count}")

P=lyap.get_P().detach().numpy()
eigP= np.linalg.eigvals(P)
print(f"P:\n{P}")
print(f"Eigenvalues of P: {eigP}")
An=A.numpy()
print(f"Eigenvalues of A: {np.linalg.eigvals(An)}")
eigs= np.linalg.eigvals(An.transpose()*P+P*An)
print(f"Eigenvalues of A^T P + P A: {eigs}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Vx.detach().numpy().flatten(),'k')
plt.title("Lyapunov Function V(x)")
plt.subplot(1, 2, 2)
plt.plot(Vdot.detach().numpy().flatten(),'k')
plt.title("Time Derivative \\dot{V}(x)")
plt.show()
