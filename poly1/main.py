import torch
import torch.nn as nn
import numpy as np

# Generate training data
def generate_data(n_samples=10000):
    a1 = np.random.uniform(-10, 10, n_samples)
    a0 = np.random.uniform(-10, 10, n_samples)

    coeffs = np.stack([a1, a0], axis=1)
    roots = []
    for a, b in zip(a1, a0):
        r = np.roots([1, a, b])
        roots.append([r[0].real, r[0].imag, r[1].real, r[1].imag])
    
    return torch.tensor(coeffs, dtype=torch.float32), torch.tensor(roots, dtype=torch.float32)

# Neural network definition
class RootFinderNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

# Training loop
def train_model():
    X, y = generate_data()
    model = RootFinderNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(5000):
        model.train()
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
    
    return model

# Train
model = train_model()

# test model s^2+a1s+a0
a1=4.0
a0=4.0
a1t = torch.tensor([a1])
a0t = torch.tensor([a0])
inp = torch.stack([a1t, a0t], dim=1)
out = model(inp)
print("Predicted roots:", out)
print("True roots:", np.roots([1, a1, a0]))