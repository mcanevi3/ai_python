import torch 
import matplotlib.pyplot as plt

from defs import *
from lyap import *
from util import *

delta=0.1
grid=torch.arange(-1, 1, 2*delta)+delta
x1, x2= torch.meshgrid(grid,grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)
pvec=torch.tensor([1.0,1.0,1.0],dtype=torch.float32)
P=make_symmetric_P(pvec)

x=x_train.clone()

plt.figure(figsize=(10, 10))
plt.grid()
ax=plt.gca()
plt.plot(0,0,'kx',linewidth=2)
plt.plot(x1, x2, 'x', color='red')


plt.plot(x[:, 0], x[:, 1], 'o', color='black', markersize=5)
step=grad_step(x,A,P)
x = x + 0.01 *step
plt.plot(x[:, 0], x[:, 1], 'o', color='red', markersize=5)

# for x in x_train:
#     circle = plt.Circle((x[0].item(), x[1].item()), radius=delta, color='blue', alpha=0.1)
#     ax.add_patch(circle)


plt.show()
