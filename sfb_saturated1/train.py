from matplotlib import pyplot as plt
import numpy as np
import torch

from controller import *

alpha = 0.0

grid = torch.linspace(-1, 1, 10)
x1, x2, x3 = torch.meshgrid(grid, grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(), x3.flatten()], dim=1)


Klqr = LQR_Controller()
u = x_train @ Klqr.T
v_dot = dVdt(x_train, u)              # compute \dot{V}(x)
lyapunov_penalty_raw = v_dot ;#+ alpha * V(x_train)
lyapunov_penalty = torch.relu(lyapunov_penalty_raw)

plt.figure(figsize=(10, 6))
plt.plot(V(x_train).numpy().flatten(),'k', label='V(x)')
plt.plot(lyapunov_penalty_raw.numpy().flatten(),'b', label='Raw Lyapunov Penalty')
plt.plot(lyapunov_penalty.numpy().flatten(),'r', label='Clipped Lyapunov Penalty')
plt.show()

# model = Controller()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# model.print()
# # === 4. Training loop ===
# for epoch in range(1*100000+1):
#     u = model(x_train)                    # controller output
#     v_dot = dVdt(x_train, u)              # compute \dot{V}(x)
#     lyapunov_penalty = torch.relu(v_dot + alpha * V(x_train))

#     #count violations    
#     cond=(lyapunov_penalty>0).squeeze()
#     count=cond.sum().item() 
#     violation_prob=torch.sigmoid(lyapunov_penalty)
#     loss=violation_prob.mean()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} Violations:{count}")
#     if count==0:
#         print(f"All violations cleared at epoch {epoch}")
#         break
# model.print()
# model.save()

# linear_layer = model.net[0]  # nn.Linear(2, 1, bias=False)
# weights = linear_layer.weight.data  # shape: (1, 2)
# # Convert to numpy (optional)
# K = weights.numpy().flatten()
# Acl=(A+B*K).numpy()
# eigs=np.linalg.eigvals(Acl)

# print("K =", K)  # [k1, k2]
# print("Eigs:", eigs)