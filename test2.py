import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
Y = 2*X + 1

# Define the model
model = nn.Sequential(
    nn.Linear(1, 4),  # input → hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # input → hidden layer
    nn.ReLU(),
    nn.Linear(4, 1)   # hidden → output layer
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Training loop
for epoch in range(1000):
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Print model structure again after training
print("\nFinal trained model:")
print(model)
for m in model:
    for name,param in m.named_parameters():
        print(f"{name}:\n{param.data}\n")

with torch.no_grad():
    xTest=torch.tensor([[6.0]])
    yTest=model(xTest)
    print("test:",xTest,yTest)
