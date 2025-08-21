import cvxpy as cvx
import numpy as np

A=np.array([[1.0,2.0],[-3.0,-4.0]])
B=np.array([[1.0],[2.0]])

P = cvx.Variable((2,2), symmetric=True)
S = cvx.Variable((1,2))
objective = cvx.Minimize(cvx.trace(P))
constraints = [
    P >> 1e-3*np.eye(2),
    A @ P + P @ A.T + B @ S + S.T @ B.T + 1e-1*np.eye(2) << 0
]
prob = cvx.Problem(objective, constraints)
result = prob.solve(solver=cvx.SCS)

P=P.value
S=S.value
K=S@np.linalg.inv(P)
print(f"K:{K}")

lmi1=A+B@K
eigVal,eigVec = np.linalg.eig(lmi1)
print("Closed-loop eigenvalues:", eigVal)

