import numpy as np 
from controller import *

Fs=Controller()
Fs.load()
Fs.print()

samples=101
t=np.linspace(0,20,samples)
T=t[2]-t[1]

x0=np.array([1,-1])
x=np.zeros((samples,2))
x[0]=x0
u=np.ones((samples,1))
Ad = torch.eye(2) + T * A
Bd = T * B
print(Ad)
print(Bd)
for i in range(1, samples):
    xi = torch.tensor(x[i-1], dtype=torch.float32)        # shape: (2,)
    ui = Fs(xi.unsqueeze(0)).detach().numpy()[0, 0]       # scalar output

    u[i-1] = ui

    # Compute dx/dt = A x + B u
    dx = Ad @ xi + Bd.flatten() * ui                      # shape: (2,)
    x[i] = dx.numpy()

# Print final state
print("Initial state:", x[0])
print("Final state:", x[-1])
