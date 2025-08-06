"""
Genetic Algorithm for Lyapunov Function Training
Test known P, A^T P + P A
with using GA to find P
"""
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

A=torch.tensor([[1., 2.],
                  [-3., -4.]], dtype=torch.float32)
n= A.shape[0]
# UTIL FUNCTIONS
def get_lqr_P(A, Q):
    P = scipy.linalg.solve_continuous_lyapunov(A.numpy().T, -Q.numpy())
    return torch.tensor(P, dtype=torch.float32)
def eigs(P):
    return np.linalg.eigvals(P.numpy())
def make_symmetric_P(pvec):
    L = torch.zeros((n, n), dtype=torch.float32)
    idx = 0
    for i in range(n):
        for j in range(i+1):
            L[i, j] = pvec[idx]
            idx += 1
    P = L @ L.T
    return P

# GA FUNCTIONS
POP_SIZE = 200
N_GEN = 200
MUTATION_RATE = 0.2
MUTATION_SCALE = 0.1
# Data
grid = torch.linspace(-1, 1, 100, dtype=torch.float32)
x1, x2= torch.meshgrid(grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1).T
xdot_train = A @ x_train
def cost(pvec):
    P = make_symmetric_P(pvec)
    V = (x_train.T @ P @ x_train).diagonal()
    V_res=torch.relu(-V)
    V_count = (V_res > 0).sum().item()

    Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
    Vdot_res=torch.relu(Vdot)
    Vdot_count = (Vdot_res > 0).sum().item()

    return V_count + Vdot_count


# Algorithm
pop = np.random.uniform(low=-10, high=10, size=(POP_SIZE, 3))
best_idx= None
best_p = None
best_count = float('inf')
for gen in range(N_GEN):
    fitness = np.array([cost(p) for p in pop])
    best_idx = np.argmin(fitness)
    best_p = pop[best_idx]
    best_count = fitness[best_idx]
    print(f"Gen {gen:3d} | Best violations: {best_count}")

    if best_count == 0:
        break

    # Selection: Top 20% keep
    num_elite = POP_SIZE // 5
    elite = pop[np.argsort(fitness)[:num_elite]]

    # Crossover: fill the rest
    children = []
    while len(children) < POP_SIZE - num_elite:
        parents = elite[np.random.choice(num_elite, 2, replace=False)]
        alpha = np.random.rand()
        child = alpha * parents[0] + (1 - alpha) * parents[1]

        # Mutation
        if np.random.rand() < MUTATION_RATE:
            child += np.random.normal(0, MUTATION_SCALE, size=3)

        children.append(child)

    # New population
    pop = np.vstack([elite, children])

print("*******************************")
print(f"Best candidate: {best_p}")
print(f"Best count: {best_count}")
P = make_symmetric_P(best_p)
print(f"Best P:\n{P.numpy()}")
print(f"Eigenvalues of P: {eigs(P)}")
print(f"Eigenvalues of A^T P + P A: {eigs(A.T @ P + P @ A)}")
