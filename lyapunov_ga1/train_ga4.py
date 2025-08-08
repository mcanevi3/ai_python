"""
Genetic Algorithm for Lyapunov Function Training
Test known P, A^T P + P A using u=Kx and NN
with using GA to find P
P is generated directly from a vector instead of Cholesky decomposition
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

n = A.shape[0]  # Number of states
nP=n*(n+1)//2  # Number of unique elements in symmetric P

# Controller
fileName = "best_controller.pth"
Fs = nn.Sequential(
    nn.Linear(n, 3, bias=True),
    nn.Linear(3, 3, bias=True),
    nn.Linear(3, 1, bias=False)
)
def get_controller():
    with torch.no_grad():
        return torch.cat([p.view(-1) for p in Fs.parameters()])
def controller_param_count():
    total_params = sum(p.numel() for p in Fs.parameters())
    return total_params
# Set all parameters from a flat 1-D vector
def set_controller(fvec):
    with torch.no_grad():
        fvec_t = torch.as_tensor(fvec, dtype=torch.float32, device=next(Fs.parameters()).device)
        idx = 0
        for p in Fs.parameters():
            numel = p.numel()
            p.copy_(fvec_t[idx:idx + numel].view_as(p))
            idx += numel
        if idx != fvec_t.numel():
            raise ValueError("Size of fvec does not match total number of model parameters")
def print_controller():
    print("Controller structure and parameters:\n")
    for i, layer in enumerate(Fs):
        if isinstance(layer, nn.Linear):
            print(f"Layer {i} - Linear(in_features={layer.in_features}, out_features={layer.out_features}, bias={layer.bias is not None})")
            print("  Weights:")
            print(layer.weight.data)
            if layer.bias is not None:
                print("  Biases:")
                print(layer.bias.data)
        else:
            # Non-learnable layers (like ReLU)
            print(f"Layer {i} - {layer.__class__.__name__}")
    print("\nEnd of controller printout.\n")
def save_controller(filename):
    torch.save(Fs.state_dict(), filename)
def load_controller(filename):
    Fs.load_state_dict(torch.load(filename))
    Fs.eval()

# simulate
def sim():
    t= torch.arange(0, 20, 1e-3, dtype=torch.float32)
    NSIM= t.shape[0]
    x= torch.zeros((n, NSIM), dtype=torch.float32)
    u=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
    y=torch.zeros((1, NSIM), dtype=torch.float32)  # Reference input
    r=torch.sin(2*torch.pi*t).reshape(1, -1)
    # x[:, 0] = torch.tensor([1., 1.,0.], dtype=torch.float32)
    for i in range(1, NSIM):
        u[:, i-1] = Fs(x[:, i-1])
        x[:, i] = x[:, i-1] + (A @ x[:, i-1] + B @ u[:, i-1] + Br @ r[:,i-1]) * (t[i] - t[i-1])
        y[:, i] = C @ x[:, i]
    plt.figure(num=1, figsize=(10, 5))
    plt.plot(t.detach().numpy(), r.squeeze().detach().numpy(),'k', label='r')
    plt.plot(t.detach().numpy(), y.squeeze().detach().numpy(),'b', label='y')
    plt.plot(t.detach().numpy(), u.squeeze().detach().numpy(),'m', label='u')
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
    # P=torch.tensor([[pvec[0], pvec[1]],
    #                  [pvec[1], pvec[2]]], dtype=torch.float32)
    P=torch.tensor([[pvec[0], pvec[1], pvec[2]],
                     [pvec[1], pvec[3], pvec[4]],
                     [pvec[2], pvec[4], pvec[5]]], dtype=torch.float32)
    return P

def cartesian_data():
    grid = torch.linspace(-1, 1, 14, dtype=torch.float32)
    x1, x2,x3 = torch.meshgrid(grid, grid,grid, indexing='ij')
    x_train = torch.stack([x1.flatten(), x2.flatten(), x3.flatten()], dim=1).T
    return x_train

def train():
    # GA FUNCTIONS
    POP_SIZE = 1500
    N_GEN = 40
    MUTATION_RATE = 0.4
    MUTATION_SCALE = 0.2
    # Data
    x_train = cartesian_data()
    def cost(xvec):
        pvec= xvec[:nP]
        fvec= xvec[nP:]
        set_controller(fvec)
        P = make_symmetric_P(pvec)
        V = (x_train.T @ (P) @ x_train).diagonal()
        V_res=torch.relu(-V)
        V_count = (V_res > 0).sum().item()

        u = Fs(x_train.T).T
        xdot_train = A @ x_train + B @ u 
        Vdot = (xdot_train.T @ P @ x_train + x_train.T @ P @ xdot_train).diagonal()
        Vdot_res=torch.relu(Vdot+0.1*V)
        Vdot_count = (Vdot_res > 0).sum().item()

        return V_count + Vdot_count
    
    def plot_cost(xvec):
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

        plt.figure(num=2,figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(V.detach().numpy(), 'r.', label='V')
        plt.title(f'V: {V_count} violations')
        plt.xlabel('Sample')
        plt.ylabel('V')
        plt.grid()
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(Vdot.detach().numpy(), 'b.', label='Vdot')
        plt.title(f'Vdot: {Vdot_count} violations')
        plt.xlabel('Sample')
        plt.ylabel('Vdot')
        plt.grid()
        plt.legend()
        plt.tight_layout()

    
    # Algorithm
    pop = np.random.uniform(low=-10, high=10, size=(POP_SIZE, nP+controller_param_count()))
    best_idx= None
    best_p = None
    best_count = float('inf')
    for gen in range(N_GEN):
        fitness = np.array([cost(p) for p in pop])
        best_idx = np.argmin(fitness)
        best_p = pop[best_idx]
        best_count = fitness[best_idx]
        print(f"Gen {gen:3d} | Best violations: {best_count}/{x_train.shape[1]}")

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
                child += np.random.normal(0, MUTATION_SCALE, size=nP+controller_param_count())

            children.append(child)

        # New population
        pop = np.vstack([elite, children])

    print("*******************************")
    print(f"Best candidate: {best_p}")
    print(f"Best count: {best_count}")
    pvec= best_p[:nP]
    fvec= best_p[nP:]
    set_controller(fvec)
    P = make_symmetric_P(pvec)
    print(f"Best P:\n{P.numpy()}")
    print(f"Eigenvalues of P: {eigs(P)}")
    
    print_controller()
    save_controller(fileName)
    plot_cost(best_p)

if __name__=="__main__":
    train()
    # load_controller(fileName)
    # print_controller()
    sim()




