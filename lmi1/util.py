import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt 

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


if __name__=="__main__":
    test_eig1,_=np.linalg.eig(np.eye(2))
    print(f"eig:{test_eig1}")
    print(f"is stable? {is_stable_eig(test_eig1)}")