import torch
import matplotlib.pyplot as plt

from util import *
from lyap import *
from defs import *

# GA FUNCTIONS
POP_SIZE = 15000
N_GEN = 100
MUTATION_RATE = 0.4
MUTATION_SCALE = 0.25

# Data
grid = torch.linspace(-1, 1, 5, dtype=torch.float32)
x1, x2, x3= torch.meshgrid(grid, grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten(),x3.flatten()], dim=1)
def ga_cost(xvec):
    pvec= xvec[:nP]
    P = make_symmetric_P(pvec,3)

    x=x_train
    cost=lyap_cost(x,A,P)
    for i in range(100):    
        if cost >0:
            break
        step = grad_step(x, A, P)
        x = x + 0.01 *step
        cost=lyap_cost(x,A,P)
    return cost

if __name__ == "__main__":
    pop = torch.randn((POP_SIZE, nP)) * 500-250  # Initial population with small random values
    # pop = (torch.rand((POP_SIZE, nP)) * 500) - 250

    best_idx = None
    best_p = None
    best_count = float("inf")

    for gen in range(N_GEN):
        # Evaluate fitness
        fitness = torch.tensor([ga_cost(p) for p in pop])  # cost must accept torch.Tensor

        # Best individual
        best_idx = torch.argmin(fitness)
        best_p = pop[best_idx]
        best_count = fitness[best_idx].item()

        print(f"Gen {gen:3d} | Best violations: {best_count}/{x_train.shape[0]}")

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

    data={"pvec":best_p.tolist()}
    save_json(data, "best_candidate.json")
