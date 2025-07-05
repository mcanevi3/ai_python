import numpy as np 
from controller import *


Fs=Controller()
Fs.load()
Fs.print()

samples=101
t=np.linspace(0,10,samples)
T=t[2]-t[1]

x0=10
x=np.zeros((samples,1))
x[0]=x0
u=np.ones((samples,1))
for i in range(1, samples):
    # Convert x[i-1] to tensor, run through controller, then back to NumPy
    xi = torch.tensor([[x[i-1][0]]], dtype=torch.float32)  # shape (1,1)
    ui = Fs(xi).detach().numpy()[0][0]  # output is tensor -> numpy scalar
    u[i-1] = ui
    x[i] = (1 +T*(2)) * x[i-1] + T * u[i-1]
print(x[0])
print(x[-1])

