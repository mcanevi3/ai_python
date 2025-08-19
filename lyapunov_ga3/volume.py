import torch 
import matplotlib.pyplot as plt

from defs import *
from lyap import *
from util import *

delta=0.5
grid=torch.arange(-1, 1, 2*delta)+delta
x1, x2= torch.meshgrid(grid,grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)
pvec=torch.tensor([1.0,-1.0,1.0],dtype=torch.float32)
P=make_symmetric_P(pvec)

x=x_train.clone()

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
ax1=plt.gca()
plt.subplot(1, 3, 2)
ax2=plt.gca()
plt.subplot(1, 3, 3)
ax3=plt.gca()

ax1.grid()
ax2.grid()
ax3.grid()
ax1.plot(0,0,'kx',linewidth=2)

shade=0.0
ax1.plot(x[:, 0], x[:, 1], 'o', color=(shade,0,0), markersize=5)
for i in range(99):
    step=grad_step(x,A,P)
    x = x + 0.01 *step
    shade=shade+0.01
    ax1.plot(x[:, 0], x[:, 1], 'o', color=(shade,0,0), markersize=5)

    val=V(x,A,P)
    ax2.plot(range(len(val)), val,color=(shade,0,0))
    val=Vdot(x,A,P)
    ax3.plot(range(len(val)), val,color=(shade,0,0))
plt.show()
