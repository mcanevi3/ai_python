"""
Genetic Algorithm for Lyapunov Function Training
Test known P, A^T P + P A using u=Kx and NN
with using GA to find P
"""
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

Ap=torch.tensor([[1., 2.],[-3., -4.]], dtype=torch.float32)
Bp=torch.tensor([[0.],[1.]], dtype=torch.float32)
Cp=torch.tensor([[1., 0.]], dtype=torch.float32)

A =  torch.cat([
        torch.cat([Ap, torch.zeros((Ap.shape[0], 1))], dim=1),
        torch.cat([-Cp, torch.zeros((1, 1))], dim=1)
    ], dim=0)
B = torch.cat([Bp, torch.zeros((1, 1))], dim=0)
C = torch.cat([Cp, torch.zeros((1, 1))], dim=1)
Br = torch.cat([torch.zeros((Ap.shape[0], 1)), torch.ones((1, 1))], dim=0)

n = 3
nP=n*(n+1)//2  # Number of unique elements in symmetric P
nF=n
# Controller
fileName = "best_controller.pth"
Fs=nn.Sequential(nn.Linear(nF,1,bias=False))

def set_controller(fvec):
    with torch.no_grad(): 
        Fs[0].weight.copy_(torch.tensor(fvec))  
def get_controller():
    return Fs[0].weight.data
def print_controller():
    print("Controller:", get_controller())
def save_controller(filename):
    torch.save(Fs.state_dict(), filename)
def load_controller(filename):
    Fs.load_state_dict(torch.load(filename))
    Fs.eval()

# simulate
def sim():
    NSIM= 100
    t= torch.linspace(0, 10, NSIM, dtype=torch.float32)
    x= torch.zeros((nF, NSIM), dtype=torch.float32)
    u=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
    y=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
    r=torch.ones((1, NSIM), dtype=torch.float32)  # Reference input
    # x[:, 0] = torch.tensor([1., 1.], dtype=torch.float32)
    for i in range(1, NSIM):
        u[:, i-1] = Fs(x[:, i-1])
        x[:, i] = x[:, i-1] + (A @ x[:, i-1] + B @ u[:, i-1] + Br @ r[:, i-1]) * (t[i] - t[i-1])
        y[:, i] = C @ x[:, i]
    plt.plot(t.detach().numpy(), r.squeeze().detach().numpy(),'k', label='r')
    plt.plot(t.detach().numpy(), y.squeeze().detach().numpy(),'b', label='y')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('System Response')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
# UTIL FUNCTIONS
def get_lqr_P(A, Q):
    P = scipy.linalg.solve_continuous_lyapunov(A.numpy().T, -Q.numpy())
    return torch.tensor(P, dtype=torch.float32)
def eigs(P):
    return np.linalg.eigvals(P.numpy())
def make_symmetric_P(pvec):
    L = torch.zeros((nF, nF), dtype=torch.float32)
    idx = 0
    for i in range(nF):
        for j in range(i+1):
            L[i, j] = pvec[idx]
            idx += 1
    P = L @ L.T
    return P

def train():
    # GA FUNCTIONS
    POP_SIZE = 30
    N_GEN = 100
    MUTATION_RATE = 0.2
    MUTATION_SCALE = 0.1
    # Data
    grid = torch.linspace(-1, 1, 20, dtype=torch.float32)
    x1, x2, x3= torch.meshgrid(grid, grid,grid, indexing='ij')
    x_train = torch.stack([x1.flatten(), x2.flatten(), x3.flatten()], dim=1).T
    def cost(xvec):
        pvec= xvec[:nP]
        fvec= xvec[nP:]
        set_controller(fvec)
        P = make_symmetric_P(pvec)
        V = (x_train.T @ P @ x_train).diagonal()
        V_res=torch.relu(-V)
        V_count = (V_res > 0).sum().item()

        u = Fs(x_train.T).T
        xdot_train = A @ x_train + B @ u

        Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
        Vdot_res=torch.relu(Vdot)
        Vdot_count = (Vdot_res > 0).sum().item()

        return V_count + Vdot_count


    # Algorithm
    pop = np.random.uniform(low=-2, high=2, size=(POP_SIZE, nP+nF))
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
                child += np.random.normal(0, MUTATION_SCALE, size=nP+nF)

            children.append(child)

        # New population
        pop = np.vstack([elite, children])

    print("*******************************")
    print(f"Best candidate: {best_p}")
    print(f"Best count: {best_count}")
    pvec= best_p[:6]
    fvec= best_p[6:]
    set_controller(fvec)
    P = make_symmetric_P(pvec)
    print(f"Best P:\n{P.numpy()}")
    print(f"Eigenvalues of P: {eigs(P)}")
    K=get_controller()
    Acl= A + B @ K
    print(f"Eigenvalues of A^T P + P A: {eigs((Acl).T @ P + P @ (Acl))}")
    print_controller()
    save_controller(fileName)

if exists(fileName):
    load_controller(fileName)
    print_controller()
else:
    train()
sim()
