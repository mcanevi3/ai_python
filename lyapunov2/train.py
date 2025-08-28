import torch
import torch.nn as nn

from controller import *

model = Controller()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
alpha = 0.1

# with torch.no_grad():
#    model.net[0].weight.fill_(10)

# Training data: sample x from range [-2, 2]
x_train = torch.linspace(-1,1, 1000).unsqueeze(1)

model.print()
# === 4. Training loop ===
for epoch in range(10000+1):
    u = model(x_train)                    # controller output
    v_dot = dVdt(x_train, u)              # compute \dot{V}(x)
    lyapunov_penalty = v_dot + alpha * V(x_train)

    violation_prob=cost(lyapunov_penalty)
    loss=violation_prob.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #count violations    
    cond=(lyapunov_penalty>0).squeeze()
    count=cond.sum().item() 
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} Violations:{count}")
    # if count==0:
        # break
model.print()
model.save()