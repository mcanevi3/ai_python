import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import control as ctrl

# === 1. Define the transfer function ===
num = [1, 2]
den = [1, 4, 5]
sys = ctrl.tf(num, den)

# === 2. Time vector and random input ===
t = np.linspace(0, 10, 500)
u = np.random.uniform(-1, 1, size=len(t))

# === 3. Simulate system output ===
t_out, y = ctrl.forced_response(sys, T=t, U=u)

# === 4. Prepare training data: (u[t], y[t-1]) → y[t]
X = np.stack([u[1:], y[:-1]], axis=1)
Y = y[1:]

# === 5. Convert to PyTorch ===
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 6. Neural network model
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 7. Train
for epoch in range(100):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}")

# === 8. Predict
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

# === 9. Plot results
plt.figure(figsize=(10, 4))
plt.plot(t_out[1:], y[1:], label="True Output")
plt.plot(t_out[1:], y_pred, '--', label="NN Prediction")
plt.xlabel("Time [s]")
plt.ylabel("Output y(t)")
plt.title("System Output vs NN Model")
plt.legend()
plt.grid(True)
plt.show()
