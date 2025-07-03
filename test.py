import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Training data (same as before)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = 2 * X + 1

# Define a simple neural network model: one layer, no activation
model = nn.Linear(in_features=1, out_features=1)
# Loss function: Mean Squared Error
criterion = nn.MSELoss()
# Optimizer: Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Training loop
for epoch in range(1000):
    # Forward pass
    Y_pred = model(X)
    # Compute loss
    loss = criterion(Y_pred, Y)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print progress
    if epoch % 100 == 0:
        [W, b] = model.parameters()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, W = {W.item():.4f}, b = {b.item():.4f}")

# Final model
W, b = model.parameters()
print("\nTrained model: y = {:.2f}x + {:.2f}".format(W.item(), b.item()))

# Plot results
plt.scatter(X.detach().numpy(), Y.detach().numpy(), label='True data')
plt.plot(X.detach().numpy(), Y_pred.detach().numpy(), color='red', label='Model prediction')
plt.legend()
plt.show()