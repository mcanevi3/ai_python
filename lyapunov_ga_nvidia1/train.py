import torch
import numpy as np
import matplotlib.pyplot as plt

from controller import *
from plant import *
from device import *

# GA FUNCTIONS
POP_SIZE = 1500
N_GEN = 40
MUTATION_RATE = 0.4
MUTATION_SCALE = 0.2


Fs=Controller()
fileName="controller.pth"

# Data
grid = torch.linspace(-1, 1, 14, dtype=torch.float32,device=device)
x1, x2,x3 = torch.meshgrid(grid, grid,grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(), x3.flatten()], dim=1).T

def cost(xvec):
    pvec= xvec[:nP]
    fvec= xvec[nP:]
    Fs.set_controller(fvec)
    P = make_symmetric_P(pvec)
    V = (x_train.T @ (P) @ x_train).diagonal()
    V_res=torch.relu(-V)
    V_count = (V_res > 0).sum().item()

    u = Fs(x_train.T).T
    xdot_train = A @ x_train #+ B @ u 
    Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
    Vdot_res=torch.relu(Vdot+0.1*V)
    Vdot_count = (Vdot_res > 0).sum().item()

    return Vdot_count


pop = (torch.rand((POP_SIZE, nP + Fs.controller_param_count()), device=device) * 20) - 10

best_idx = None
best_p = None
best_count = float("inf")

for gen in range(N_GEN):
    # Evaluate fitness
    fitness = torch.tensor([cost(p) for p in pop], device=device)  # cost must accept torch.Tensor

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
        parent_idx = torch.randperm(num_elite,device=device)[:2]
        parents = elite[parent_idx]
        alpha = torch.rand(1, device=device)
        child = alpha * parents[0] + (1 - alpha) * parents[1]

        if torch.rand(1, device=device) < MUTATION_RATE:
            child += torch.randn_like(child,device=device) * MUTATION_SCALE

        children.append(child)

    children = torch.stack(children)

    # New population
    pop = torch.cat([elite, children], dim=0)

print("*******************************")
print(f"Best candidate: {best_p}")
print(f"Best count: {best_count}")
pvec= best_p[:nP]
fvec= best_p[nP:]
Fs.set_controller(fvec)
P = make_symmetric_P(pvec)
print(f"Best P:\n{P.cpu().numpy()}")
print(f"Eigenvalues of P: {eigs(P.cpu())}")

K=Fs.Fs[0].weight.data
Acl=A+B @ K
temp=Acl.T @ P+P @ Acl
print(f"Eigenvalues of Acl.T @ P+P @ Acl: {eigs(temp.cpu())}")

Fs.print_controller()
Fs.save_controller(fileName)

data={
    "pvec":pvec.cpu().tolist(),
    "fvec":fvec.cpu().tolist(),
}
save_json(data,"results.json")

