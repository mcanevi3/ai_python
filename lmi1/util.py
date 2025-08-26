import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt 
import control

def print_pos_values(arr):
    mask = arr > 0
    indices = np.argwhere(mask)
    values = arr[mask]
    print("Indices:", indices.tolist())
    print("Values:", values)
def is_stable_eig(eig):
    return np.all(np.real(eig)<0)
def sign(x):
    return np.sign(x)
def relu(x):
    return np.maximum(0, x)
def sum(x):
    total = 0
    for item in x:
        total += item
    return total
def solve_lmi(x_train,xdot_train,alpha=0.3,eps=1e-5,verbose=False):
    P = cvx.Variable((2,2), symmetric=True)
    objective = cvx.Minimize(cvx.trace(P)*0)
    constraints=[
        P >> eps*np.eye(2),
    ]
    for i in range(x_train.shape[0]):
        x=x_train[i,:]
        dx=xdot_train[i,:]
        
        vx=x.T @ P @ x
        vxdot=dx.T @ P @ x + x.T @ P @ dx
        
        val=vxdot+alpha*vx
        constraints.append(val+eps<=0)

    prob = cvx.Problem(objective, constraints)
    result=prob.solve(solver=cvx.SCS,verbose=verbose)
    P=P.value
    return P
def test_lmi(x_train,xdot_train,P,alpha):
    constraints=np.array([])
    cost=0
    for i in range(x_train.shape[0]):
        x=x_train[i,:]
        dx=xdot_train[i,:]
        
        vx=x.T @ P @ x
        vxdot=dx.T @ P @ x + x.T @ P @ dx
        
        val=vxdot+alpha*vx
        constraints=np.append(constraints,val)
        cost=cost+relu(val)
    return cost, constraints

def plot_constraint(x):
    plt.figure(figsize=(15,10))
    plt.plot(x,'k',linewidth=2)
    plt.plot(sign(x),'r',linewidth=2)
    plt.grid()
    plt.xlabel("sample")
    plt.ylabel(r"$\dot{V}$")
    plt.ylim((-1.1,1.1))
    plt.title(r"Constraint $\dot{V}<0$")
    plt.show()

def plot_step_responses(A,B,x_train):
    np.random.seed(4)
    n=2
    K = np.random.rand(1, n)
    Ac=A+B@K
    eigVal,_=np.linalg.eig(Ac)
    print(f"eig(Ac):{eigVal}")

    P=np.array([[0.00019255,0.00040085],[0.00040085,0.00137185]])
    Gs=control.ss(Ac,B,np.eye(n),np.zeros((n,1)))
    for x0 in x_train:
        plt.subplot(2,1,1)
        x0=x0.reshape(2,1)
        t,x=control.initial_response(Gs,X0=x0)
        
        plt.plot(t,x[0,:],'r')
        plt.plot(t,x[1,:],'b')

        print(f"x0:{x0}")

        V=x0.T@P@x0
        print(f"V(x0):{V}")
        x1=x0
        x2=Ac@x1
        print(f"dotV(x1):{x2.T@P@x1+x1.T@P@x2}")

        x1=x0
        step=10
        vdotvec=np.zeros((step,))
        for i in range(step):
            x2=Ac@x1
            Vdot=x2.T@P@x1+x1.T@P@x2
            x1=x2
            vdotvec[i]=Vdot.item()

        plt.subplot(2,1,2)
        plt.plot(vdotvec)

        break
    plt.show()

def plot_lyap_with_lmi(A,B,x_train):
    np.random.seed(4)
    n=2
    K = np.random.rand(1, n)
    Ac=A+B@K
    P=np.array([[0.00019255,0.00040085],[0.00040085,0.00137185]])

    Q=Ac.T@P+P@Ac
    dotV_vec=np.zeros((x_train.shape[0],))
    
    for i,x0 in enumerate(x_train):
        x0=x0.reshape(2,1)
        V=x0.T@P@x0
        dotV_vec[i]=(x0.T@Q@x0).item()
    plt.plot(relu(dotV_vec))
    plt.show()

if __name__=="__main__":
    from defs import x_train,A,B
    plot_lyap_with_lmi(A,B,x_train)
