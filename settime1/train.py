import torch
import torch.nn as nn
import torch.optim as optim


class SettlingTimeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Load tensors
data = torch.load("settling_time_tensors.pt")
X_torch = data['X']  # shape (N, 2): [w, d]
y_torch = data['y']  # shape (N, 1): [ts]

model = SettlingTimeNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Training loop
epochs = 10000
for epoch in range(epochs):
    model.train()
    
    y_pred = model(X_torch)
    loss = criterion(y_pred, y_torch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    print(model(torch.tensor([1.0,0.0])))
