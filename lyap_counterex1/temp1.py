"""
Gradient Ascent
"""
import torch
import matplotlib.pyplot as plt

def f(x):
    return x**2
def grad_f(x):
    return 2*x

COLORS=[(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)]
STEPS=30
x0=torch.tensor([-4,0.1,4])

N=len(x0)
xvec=torch.zeros((N,STEPS))
xvec[:,0]=x0
for i in range(1,STEPS):
    step=0.1*grad_f(xvec[:,i-1])
    xvec[:,i]=xvec[:,i-1]+step

tempX=torch.linspace(-10.0,10.0,100)
tempY=f(tempX)
plt.plot(tempX,tempY,'k')
for i in range(xvec.shape[0]):
    plt.plot(xvec[i,0],f(xvec[i,0]),'x',color=COLORS[i])
    plt.plot(xvec[i,:],f(xvec[i,:]),color=COLORS[i])
    plt.plot(xvec[i,-1],f(xvec[i,-1]),'o',color=COLORS[i])

plt.grid()
plt.show()

