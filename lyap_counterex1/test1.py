"""
Look for positive V(x) but also positive dot(V)(x) on a grid to understan V+alpha dot(V) beeeing zero despite instability
"""
import torch 

from util import *

A=torch.tensor([[1.0,2.0,3.0],[2.0,2.0,2.4],[-3.0,-4.0,7.0]])
n=A.shape[0]
nP=n*(n+1)//2

pvec=torch.tensor([164.3077, -153.4470,   -0.6598,  -16.6760,   -1.2490,   -1.1480])
pvec=torch.tensor([-206.4720,  166.1432,  -13.1968,   38.2916,  -15.1836,   37.3204])
pvec=torch.tensor([-206.5327,  132.3334,  -74.7703,   36.9112,  -39.7628,   21.9226])
P=make_symmetric_P(pvec,3)

grid = torch.linspace(-1, 1, 10, dtype=torch.float32)
x1, x2, x3= torch.meshgrid(grid, grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(),x3.flatten()], dim=1).T

# cost used before
V = (x_train.T @ P @ x_train).diagonal()
V_res=torch.relu(-V)
V_count = (V_res > 0).sum().item()

xdot_train = A @ x_train
Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
Vdot_res=torch.relu(Vdot+0.1*V)
Vdot_count = (Vdot_res > 0).sum().item()

print(f"Vdot_count:{Vdot_count}")

lmi1=A.T@P+P@A
eigVal,eigVec=torch.linalg.eig(lmi1)

print(eigVal)
print(eigVec)
xbad=eigVec[:,2].view(3,1).real
Vbad=xbad.T@P@xbad

xdotbad = A @ xbad
Vdotbad = xdotbad.T @ P @ xbad + xbad.T @ P @ xdotbad
print(f"xbad:{xbad}")
print(f"V(xbad):{Vbad}")
print(f"Vdot(xbad):{Vdotbad}")

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
    
plot3_points(ax,x_train, color='r', marker='*', markersize=2)
plot3_points(ax,xbad, color='b', marker='*', markersize=3)
plt.show()