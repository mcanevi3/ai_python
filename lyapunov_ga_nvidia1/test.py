import torch
import numpy as np
import matplotlib.pyplot as plt

from controller import *
from plant import *
from device import *


Fs=Controller()
Fs.load_controller("controller.pth")

data=load_json("results.json")
pvec=np.array(data["pvec"])
fvec=np.array(data["fvec"])

P=make_symmetric_P(pvec)
print(f"Eigs of P:{eigs(P)}")

K=torch.tensor(fvec,dtype=torch.float32).view(1, 3)
Acl=A+B@K
print(f"Eigs of Acl:{eigs(Acl)}")

t= torch.arange(0, 20, 1e-3, dtype=torch.float32)
NSIM= t.shape[0]
x= torch.zeros((n, NSIM), dtype=torch.float32)
u=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
y=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
r=torch.sin(2*torch.pi*t).reshape(1, -1)
# x[:, 0] = torch.tensor([1., 1.,0.], dtype=torch.float32)
for i in range(1, NSIM):
    u[:, i-1] = Fs(x[:, i-1])
    x[:, i] = x[:, i-1] + (A @ x[:, i-1] + B @ u[:, i-1] + Br @ r[:,i-1]) * (t[i] - t[i-1])
    y[:, i] = C @ x[:, i]
plt.figure(num=1, figsize=(10, 5))
plt.plot(t.detach().numpy(), r.squeeze().detach().numpy(),'k', label='r')
plt.plot(t.detach().numpy(), y.squeeze().detach().numpy(),'b', label='y')
plt.plot(t.detach().numpy(), u.squeeze().detach().numpy(),'m', label='u')
plt.ylim(-1.1,1.1)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('System Response')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()