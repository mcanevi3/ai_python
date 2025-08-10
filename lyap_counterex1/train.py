import torch

from util import *

A=torch.tensor([[1.0,2.0,3.0],[2.0,2.0,2.4],[-3.0,-4.0,7.0]])
n=A.shape[0]
nP=n*(n+1)//2

# GA FUNCTIONS
POP_SIZE = 1500
N_GEN = 40
MUTATION_RATE = 0.4
MUTATION_SCALE = 0.2

# Data
grid = torch.linspace(-1, 1, 10, dtype=torch.float32)
x1, x2, x3= torch.meshgrid(grid, grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(),x3.flatten()], dim=1).T
def cost(xvec):
    pvec= xvec[:nP]
    P = make_symmetric_P(pvec,3)
    V = (x_train.T @ P @ x_train).diagonal()
    V_res=torch.relu(-V)
    V_count = (V_res > 0).sum().item()

    xdot_train = A @ x_train 
    Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
    Vdot_res=torch.relu(Vdot+0.1*V)
    Vdot_count = (Vdot_res > 0).sum().item()

    return Vdot_count


pop = (torch.rand((POP_SIZE, nP)) * 500) - 250

best_idx = None
best_p = None
best_count = float("inf")

for gen in range(N_GEN):
    # Evaluate fitness
    fitness = torch.tensor([cost(p) for p in pop])  # cost must accept torch.Tensor

    # Best individual
    best_idx = torch.argmin(fitness)
    best_p = pop[best_idx]
    best_count = fitness[best_idx].item()

    print(f"Gen {gen:3d} | Best violations: {best_count}/{x_train.shape[1]}")

    if best_count == 0:
        break

    # Selection: Top 20%
    num_elite = POP_SIZE // 5
    elite_idx = torch.argsort(fitness)[:num_elite]
    elite = pop[elite_idx]

    # Crossover + mutation
    children = []
    while len(children) < POP_SIZE - num_elite:
        parent_idx = torch.randperm(num_elite)[:2]
        parents = elite[parent_idx]
        alpha = torch.rand(1)
        child = alpha * parents[0] + (1 - alpha) * parents[1]

        if torch.rand(1) < MUTATION_RATE:
            child += torch.randn_like(child) * MUTATION_SCALE

        children.append(child)

    children = torch.stack(children)

    # New population
    pop = torch.cat([elite, children], dim=0)

print("*******************************")
print(f"Best candidate: {best_p}")
print(f"Best count: {best_count}")
