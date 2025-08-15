"""
Gradient Ascent for Lyapunov derivative
"""
import torch
import matplotlib.pyplot as plt
import scipy

A=torch.tensor([[1.0,2.0],[-3.0,-4.0]])

def V(x,P):
    return x.T@P@x
def Vdot(x,P):
    x_col = x.unsqueeze(1) 
    xdot=A@x_col
    return (xdot.T@P@x_col+x_col.T@P@xdot).squeeze()
def grad_Vdot(x,P):
    x_col = x.unsqueeze(1) 
    xdot=A@x_col
    grad_xdot=A
    return (grad_xdot.T@P@x_col+P@xdot).squeeze()

P=torch.diag(torch.tensor([1.0,1.0]))
Q=torch.diag(torch.tensor([-1.0,1.0]))
P=torch.tensor(scipy.linalg.solve_continuous_lyapunov(A.numpy().T, Q.numpy()))

lmi1=A.T@P+P@A
pLambda,pV=torch.linalg.eig(P)
print(f"Eigs of P:{pLambda.tolist()}")
lmi1Lambda,lmi1V=torch.linalg.eig(lmi1)
print(f"Eigs of A^TP+PA:{lmi1Lambda.tolist()}")
x0=torch.stack([
    torch.tensor([-1,-1]),
    torch.tensor([-1,1]),
    torch.tensor([1,-1]),
    torch.tensor([1,1]),
],dim=1)
N=x0.shape[0]
SAMPLES=x0.shape[1]
STEPS=400
xvec=torch.zeros((SAMPLES,N,STEPS))
vdotvec=torch.zeros((SAMPLES,STEPS))
for i in range(SAMPLES):
    xvec[i,:,0]=x0[:,i]
    vdotvec[i,0]=Vdot(xvec[i,:,0],P)

for k in range(SAMPLES):
    for i in range(1,STEPS):
        step=0.01*grad_Vdot(xvec[k,:,i-1],P)
        xvec[k,:,i]=xvec[k,:,i-1]+step
        vdotvec[k,i]=Vdot(xvec[k,:,i],P)
        if k==0:
            cost=torch.relu(vdotvec[k,i]).sum().item()
            print(f"Cost[{i}]:{cost}")

for k in range(SAMPLES): #SAMPLES
    plt.plot(xvec[k,0,0],vdotvec[k,0],'kx')
    plt.plot(xvec[k,0,:],vdotvec[k,:])
    plt.plot(xvec[k,0,-1],vdotvec[k,-1],'ko')

    
plt.grid()
plt.show()