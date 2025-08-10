"""
Write a gradient-ascend algorithm to maximize the delta V for a grid used for Lyapunov function
3D example
"""
from matplotlib import pyplot as plt
import torch 

from util import *

A=torch.tensor([[1.0,2.0,3.0],[2.0,2.0,2.4],[-3.0,-4.0,7.0]])

grid = torch.linspace(-2.0, 2.0, 5)
x1, x2, x3= torch.meshgrid(grid, grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(), x3.flatten()], dim=1).T

Plqr=get_lqr_P(A,torch.eye(3))
I=torch.eye(3,3)
L=torch.randn(3,3)

Q=L@L.T
Pr=get_lqr_P(A,-Q)
P=Pr

V=(x_train.T@P@x_train).diagonal()
xdot_train=A@x_train
gradxdot_train=A
Vdot=(xdot_train.T@P@x_train+x_train.T@P@xdot_train).diagonal()
Vdot_res=torch.relu(Vdot)
Vdot_count=Vdot_res.sum().item()

lmi1=A.T@P+P@A
eigVal,eigVec=torch.linalg.eig(lmi1)

eigsP,_=torch.linalg.eig(P)
print(f"Eigs of P:{eigsP.tolist()}")
print(f"Eigs of A^TP+PA:{eigVal.tolist()}")

costvec=torch.zeros((30+1,))
costvec[0]=Vdot_count
costindex=torch.zeros_like(costvec)
x_train2=x_train
for i in range(1,costvec.shape[0]):
    xdot_train2=A@x_train2
    gradVdot=2*(gradxdot_train.T@P@x_train2+P@xdot_train2)

    V2=(x_train2.T@P@x_train2).diagonal()
    Vdot2=(xdot_train2.T@P@x_train2+x_train2.T@P@xdot_train2).diagonal()
    Vdot2_res=torch.relu(Vdot2)
    Vdot2_sign=torch.sign(Vdot2_res)
    Vdot2_count=Vdot2_res.sum().item()

    # minimize values zero after ReLU, approach the boundary
    x_train2=x_train2-0.001*gradVdot*(1-Vdot2_sign)
    # maximize values zero after ReLU, approach the boundary
    x_train2=x_train2+0.001*gradVdot*(Vdot2_sign)
    x_train2=torch.clamp(x_train2,-1e2,1e2)

    costvec[i]=Vdot2_count
    costindex[i]=i

print(f"Cost:{costvec[-1].item()}")
plt.figure()
plt.plot(costindex,costvec,'b')
plt.show()

