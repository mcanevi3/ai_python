import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# === 1. Define the neural controller: u(x) ===
class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.net(x)

# === 2. Lyapunov function and its derivative ===
def V(x):  # Lyapunov function: x^2
    return x**2

def dVdt(x, u):
    return 2 * x * (-x + u)  # derivative of V(x) = x^2

# === 3. Training setup ===
model = Controller()
optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
alpha = 0.25  # desired rate of decrease in Lyapunov

# Training data: sample x from range [-2, 2]
x_train = torch.linspace(-2, 2, 500).unsqueeze(1)

# === 4. Training loop ===
for epoch in range(50000+1):
    u = model(x_train)                    # controller output
    v_dot = dVdt(x_train, u)              # compute \dot{V}(x)
    # lyapunov_penalty = torch.relu(v_dot + alpha * V(x_train))  # should be ≤ 0
    # # loss = lyapunov_penalty.mean()       # minimize violation
    # loss = lyapunov_penalty.mean() + 10 * lyapunov_penalty.min()
    epsilon = 0.01
    lyapunov_penalty = v_dot + alpha * V(x_train) + epsilon

    soft_penalty = torch.nn.functional.softplus(lyapunov_penalty)
    loss = soft_penalty.mean()+ 50 * soft_penalty.max()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# === 5. Visualize results ===
x_test = torch.linspace(-2, 2, 200).unsqueeze(1)
u_test = model(x_test).detach()
v_dot_test = dVdt(x_test, u_test).detach()

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(x_test, u_test)
plt.title("Learned Control Law u(x)")
plt.grid(True)
plt.subplot(132)
plt.plot(x_test, v_dot_test, label=r'$\dot{V}(x)$')
plt.plot(x_test, -alpha * V(x_test).detach(), '--', label=r'$-\alpha V(x)$')
plt.title("Lyapunov Derivative vs Bound")
plt.legend()
plt.grid(True)
plt.tight_layout()
#########################################################
model.eval()
with torch.no_grad():
    u_vals = model(x_test).squeeze()
x_np = x_test.squeeze().numpy()
u_np = u_vals.numpy()
# Stability boundary with alpha=0.5
bound = (1 - alpha / 2) * x_np  # 0.75 * x
# Check violations (where u(x) > 0.75 * x)
violations = u_np > bound
print(f"Number of violations: {violations.sum()} out of {len(x_np)} points")

# Plot results
plt.subplot(133)
plt.plot(x_np, u_np, label='Learned control $u(x)$')
plt.plot(x_np, bound, 'r--', label=r'Stability bound $0.75 x$ (with $\alpha=0.5$)')
plt.fill_between(x_np, bound, u_np, where=violations, color='red', alpha=0.3, label='Violations')
plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Stability Condition Check')
plt.legend()
plt.grid(True)
plt.show()

print("\nFinal trained model:")
print(model.net)
for m in model.net:
    for name,param in m.named_parameters():
        print(f"{name}:\n{param.data}\n")
