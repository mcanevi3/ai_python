import torch
import torch.nn as nn

from controller import *

model = Controller()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
alpha = 0.05

# Training data: sample x from range [-2, 2]
grid = torch.linspace(-1, 1, 100)
x1, x2 = torch.meshgrid(grid, grid, indexing='ij')
# Flatten and combine into (N, 2) tensor
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)

model.print()
# === 4. Training loop ===
for epoch in range(150+1):
    u = model(x_train)                    # controller output
    v_dot = dVdt(x_train, u)              # compute \dot{V}(x)
    lyapunov_penalty = torch.relu(v_dot + alpha * V(x_train))

    #count violations    
    cond=(lyapunov_penalty>0).squeeze()
    count=cond.sum().item() 
    violation_prob=torch.sigmoid(lyapunov_penalty)
    loss=violation_prob.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} Violations:{count}")

model.print()
model.save()

#linear_layer = model.net[0]  # nn.Linear(2, 1, bias=False)
#weights = linear_layer.weight.data  # shape: (1, 2)
# Convert to numpy (optional)
#K = weights.numpy().flatten()
#eigs=torch.linalg.eigvals(A+B*K)

#print("K =", K)  # [k1, k2]
#print("Eigs:", eigs)