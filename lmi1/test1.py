import cvxpy as cvx
import numpy as np

A=np.array([[1.0,2.0],[-3.0,-4.0]])
B=np.array([[1.0],[2.0]])

grid=np.linspace(-1,1,6)
x1,x2=np.meshgrid(grid,grid)
x_train=np.vstack([x1.flatten(),x2.flatten()]).T

# K = np.array([[-32.62468001,-8.21394156]])
# K=np.array([[0.03934841, 0.71462091]])
K=np.array([[0.15300342, 0.00486872]])
# K = np.random.rand(1, 2)
u=x_train @ K.T
xdot_train=x_train @ A.T + u @ B.T

P = cvx.Variable((2,2), symmetric=True)
objective = cvx.Minimize(0)
constraints=[
    P >> 1e-5*np.eye(2),
]
for i in range(x_train.shape[0]):
    x=x_train[i,:]
    dx=xdot_train[i,:]
    
    vx=x.T @ P @ x
    vxdot=dx.T @ P @ x + x.T @ P @ dx
    
    # constraints.append(vx + 0.001*vxdot+1e-6 <= 0)
    constraints.append(vxdot+1e-6 <= 0)

prob = cvx.Problem(objective, constraints)
result = prob.solve(solver=cvx.SCS)
P=P.value

print(f"K:{K}")
Ac=A+B@K
eigVal,eigVec = np.linalg.eig(Ac)
print("Eigenvalues Ac:", eigVal)

print(f"P:{P}")
if P is not None:
    eigP,vecP = np.linalg.eig(P)
    print("Eigenvalues of P:", eigP)

    lmi1=Ac.T@P+P@Ac
    eigVal,eigVec = np.linalg.eig(lmi1)
    print("Eigenvalues LMI:", eigVal)
