import torch
import torch.nn as nn
import torch.optim as optim

# directly learning A,B from full state data

class LTIModel(nn.Module):
    def __init__(self, n_states=2, n_inputs=1):
        super().__init__()
        # Combine x and u into one vector: input dim = n_states + n_inputs
        self.linear = nn.Linear(n_states + n_inputs, n_states, bias=False) 

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)  # shape: (batch_size, n_states + n_inputs)
        return self.linear(xu)          # predicts next state: (batch_size, n_states)

# Example true system matrices
A_true = torch.tensor([[0.9, 0.3], [-0.2, 0.8]])
B_true = torch.tensor([[0.1], [0.05]])

def simulate_data(A, B, n_samples=1000,noise_std=0.01):
    x = torch.zeros((n_samples + 1, 2))
    u = torch.randn((n_samples, 1)) * 0.5  # random inputs

    for k in range(n_samples):
        process_noise = torch.randn(2) * noise_std
        x[k+1] = A @ x[k] + B @ u[k]+process_noise
    measurement_noise = torch.randn_like(x) * noise_std
    x_noisy = x + measurement_noise
    return x_noisy[:-1], u, x_noisy[1:]  # (x_k, u_k, x_{k+1})

# Generate data
x_k, u_k, x_kplus1 = simulate_data(A_true, B_true)
# Model
model = LTIModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    x_pred = model(x_k, u_k)
    loss = criterion(x_pred, x_kplus1)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Extract learned matrices
with torch.no_grad():
    learned_weights = model.linear.weight.data  # shape (n_states, n_states + n_inputs)
    A_learned = learned_weights[:, :2]
    B_learned = learned_weights[:, 2:]

print("Learned A:\n", A_learned)
print("Learned B:\n", B_learned)
