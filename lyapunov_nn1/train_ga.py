import torch
import numpy as np

def make_symmetric_P(p_vec):
    P = torch.tensor([[p_vec[0], p_vec[1]],
                      [p_vec[1], p_vec[2]]], dtype=torch.float32)
    return P

def count_violations(p_vec, x, xdot):
    P = make_symmetric_P(p_vec)
    V = (x @ P @ x.T).diagonal()
    countV= (V <= 0).sum().item()
    Vdot = (xdot @ P @ x.T).diagonal()+(x @ P @ xdot.T).diagonal()
    countVdot = (Vdot >= 0).sum().item()
    return countV+countVdot

POP_SIZE = 100
N_GEN = 200
MUTATION_RATE = 0.2
MUTATION_SCALE = 0.1


A=torch.tensor([[1., 2.],
                  [-3., -4.]], dtype=torch.float32)

grid = torch.linspace(-1, 1, 4, dtype=torch.float32)
x1, x2= torch.meshgrid(grid, grid, indexing='ij')
x_train = torch.stack([x1.flatten(), x2.flatten()], dim=1)
xdot_train = A @ x_train.T
xdot_train = xdot_train.T

# Initialize population: [p11, p12, p22]
pop = np.random.uniform(low=0.1, high=2.0, size=(POP_SIZE, 3))

for gen in range(N_GEN):
    # Evaluate fitness
    fitness = np.array([count_violations(p, x_train, xdot_train) for p in pop])
    
    best_idx = np.argmin(fitness)
    best_p = pop[best_idx]
    best_count = fitness[best_idx]

    print(f"Gen {gen:3d} | Best violations: {best_count}")

    if best_count == 0:
        print("Found perfect Lyapunov candidate.")
        P = make_symmetric_P(best_p)
        eigvals = np.linalg.eigvals(P.numpy())
        print(f"Best P:\n{P.numpy()}")
        print(f"Eigenvalues of P: {eigvals}")

        Vdot = (xdot_train @ P @ x_train.T).diagonal() + (x_train @ P @ xdot_train.T).diagonal()
        count = (Vdot > 0).sum().item()
        print(f"Vdot violations: {count}")

        temp=A.T @ P+P @ A
        print(f"Eig:\n{np.linalg.eigvals(temp.numpy())}")
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
