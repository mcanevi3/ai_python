"""
Gradient Ascent x^TPX
"""
import torch
import matplotlib.pyplot as plt

def f(x,P):
    return x.T@P@x
def grad_f(x,P):
    return 2*P@x

COLORS=([(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)
        ,(1.0,1.0,0.0),(1.0,0.0,1.0),(0.0,1.0,1.0)])

x0=torch.stack([
    torch.tensor([-1,-1]),
    torch.tensor([-1,0]),
    torch.tensor([-1,1]),
    torch.tensor([1,-1]),
    torch.tensor([1,0]),
    torch.tensor([1,1]),
],dim=1)
N=x0.shape[0]
SAMPLES=x0.shape[1]
STEPS=4
xvec=torch.zeros((SAMPLES,N,STEPS))
for i in range(SAMPLES):
    xvec[i,:,0]=x0[:,i]

P=torch.diag(torch.tensor([1.0,-1.0]))

for k in range(SAMPLES):
    for i in range(STEPS):
        step=0.1*grad_f(xvec[k,:,0],P)
        xvec[k,:,i]=xvec[k,:,i-1]+step

print(xvec)

for k in range(SAMPLES):
    plt.plot(xvec[k,0,0],xvec[k,1,0],'x',color=COLORS[k])
    plt.plot(xvec[k,0,:],xvec[k,1,:],color=COLORS[k])
    plt.plot(xvec[k,0,-1],xvec[k,1,-1],'o',color=COLORS[k])

plt.grid()
plt.show()

