import numpy as np
import matplotlib.pyplot as plt

def sign(x:np.ndarray):
    y=np.zeros_like(x)
    for i,val in enumerate(x):
        if val>0:
            y[i]=1
        elif val<0:
            y[i]=-1
    return y

def sign_tanh(x:np.ndarray,alpha=100):
    return np.tanh(alpha*x)

if __name__=="__main__":
    t=np.linspace(0,2,1000)
    uvec=np.sin(2*np.pi*1*t)
    uvec=t**2+t-2
    
    plt.plot(t,uvec,'k',label="sine")
    plt.plot(t,sign(uvec),'b',label="sign")
    plt.plot(t,0.5+0.5*sign_tanh(uvec),'r',label="0.5+0.5 tanh(100x)")

    plt.grid()
    plt.legend()
    plt.show()