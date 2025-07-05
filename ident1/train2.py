import torch
import torch.nn as nn
import torch.optim as optim
from LTINetwork import *
import matplotlib.pyplot as plt

# Example true system matrices
A_true = torch.tensor([[0.9, 0.3], [-0.2, 0.8]])
B_true = torch.tensor([[0.1], [0.05]])
C_true = torch.tensor([[1.0,0.0]])
def simulate_data(A, B, C, n_samples=1000,noise_std=0.01):
    x = torch.zeros((n_samples, 2))
    u = torch.randn((n_samples, 1)) * 0.5  # random inputs
    y = torch.zeros((n_samples,1))
    for k in range(1,n_samples):
        process_noise = torch.randn(2) * noise_std
        x[k] = A @ x[k-1] + B @ u[k-1]+process_noise
        y[k] = C @ x[k]        
    return u,x,y
# Generate data
u,x,y = simulate_data(A_true, B_true, C_true)

model=LTINetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for epoch in range(500+1):
    model.reset_state(x0=torch.tensor([0.0, 0.0]))
    loss_total = 0
    for k in range(len(u)):
        y_true_k = y[k]           # target x(k+1)
        y_pred_k = model(u[k])                # predicts x̂(k+1)
        loss_total+= loss_fn(y_pred_k, y_true_k)
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss_total.item():.6f}")

with torch.no_grad():
    print("Learned A:\n", model.A)
    print("Learned B:\n", model.B)
    print("Learned C:\n", model.C)

    uhat,xhat,yhat = simulate_data(model.A,model.B,model.C)


    # Plot
    plt.figure(figsize=(10, 6))
    #plt.subplot(3, 1, 1)
    #plt.plot(u, label='Input u')
    #plt.ylabel("u")
    #plt.legend()

    plt.subplot(2, 1, 1)
    plt.plot(x[:, 0],'k', label='x_1 measured')
    plt.plot(x[:, 0],'r', label='x_1 modeled')
#    plt.plot(x[:, 1], label='State x_2')
    plt.ylabel("States")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y,'k', label='Measured')
    plt.plot(yhat,'r', label='Modeled')
    plt.xlabel("Time step")
    plt.ylabel("Output")
    plt.legend()

    plt.tight_layout()
    plt.savefig("lti_simulation.png")