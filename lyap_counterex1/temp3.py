"""
Gradient Ascent for Lyapunov derivative
"""
import torch
import matplotlib.pyplot as plt

A=torch.tensor([[1.0,2.0],[-3.0,-4.0]])

def V(x,P):
    return x.T@P@x
def Vdot(x,P):
    xdot=A@x
    return xdot.T@P@x+x.T@P@xdot
def grad_Vdot(x,P):
    xdot=A@x
    grad_xdot=A
    return grad_xdot.T@P@x+P@xdot

P=torch.diag(torch.tensor([1.0,1.0]))

x0=torch.stack([
    torch.tensor([-1,-1]),
    torch.tensor([-1,1]),
    torch.tensor([1,-1]),
    torch.tensor([1,1]),
],dim=1)
N=x0.shape[0]
SAMPLES=x0.shape[1]
STEPS=40
xvec=torch.zeros((SAMPLES,N,STEPS))
vdotvec=torch.zeros((SAMPLES,STEPS))
for i in range(SAMPLES):
    xvec[i,:,0]=x0[:,i]
    vdotvec[i,0]=Vdot(xvec[i,:,0],P)

for k in range(SAMPLES):
    for i in range(STEPS):
        step=0.01*grad_Vdot(xvec[k,:,0],P)
        xvec[k,:,i]=xvec[k,:,i-1]+step
        vdotvec[k,i]=Vdot(xvec[k,:,i],P)

print(xvec)
print(vdotvec)

for k in range(SAMPLES): #SAMPLES
    plt.plot(xvec[k,0,0],vdotvec[k,0],'kx')
    plt.plot(xvec[k,0,:],vdotvec[k,:])
    plt.plot(xvec[k,0,-1],vdotvec[k,-1],'ko')

plt.grid()
plt.show()