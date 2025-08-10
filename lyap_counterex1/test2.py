"""
Write a gradient-ascend algorithm to maximize the delta V for a grid used for Lyapunov function
2D example
"""
from matplotlib import pyplot as plt
import torch 

A=torch.tensor([[1.0,2.0],[-3.0,-4.0]])

grid = torch.linspace(-10.0, -5, 3)
x1, x2= torch.meshgrid(grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1).T

P=torch.eye(2,2)
V=(x_train.T@P@x_train).diagonal()
xdot_train=A@x_train
gradxdot_train=A
Vdot=(xdot_train.T@P@x_train+x_train.T@P@xdot_train).diagonal()


lmi1=A.T@P+P@A
eigVal,eigVec=torch.linalg.eig(lmi1)
print(f"Eigs of A^TP+PA:{eigVal.tolist()}")

gradVdot=2*(gradxdot_train.T@P@x_train+P@xdot_train)

print("Vdot : [" + ", ".join(f"{v:10.3f}" for v in Vdot.tolist()) + "]")
plt.figure()
plt.plot(x_train[1,:],Vdot,'kx')

x_train2=x_train
for i in range(3):
    x_train2=x_train2+0.01*gradVdot*Vdot*+1

    xdot_train2=A@x_train2
    V2=(x_train2.T@P@x_train2).diagonal()
    Vdot2=(xdot_train2.T@P@x_train2+x_train2.T@P@xdot_train2).diagonal()
    print("Vdot2: [" + ", ".join(f"{v:10.3f}" for v in Vdot2.tolist()) + "]")
    plt.plot(x_train2[1,:],Vdot2,'o',color=(0.2+0.8*i/2,0,0))

plt.show()
