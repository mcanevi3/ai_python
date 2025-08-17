import torch

def V(x,A,P):
    return x@P@x.T
def Vdot(x,A,P):
    x_col = x
    xdot=A@x_col
    return (xdot.T@P@x_col+x_col.T@P@xdot).diagonal()
def grad_Vdot(x,A,P):
    xdot=A@x
    grad_xdot=A
    return (grad_xdot.T@P@x+P@xdot) 

def lyap_cost(x,A,P):
    x_train = x
    xdot_train=A@x_train
    Vdot=(xdot_train.T@P@x_train+x_train.T@P@xdot_train).diagonal()
    Vdot_res=torch.relu(Vdot)
    Vdot_count=Vdot_res.sum().item()

    return Vdot_count

if __name__ == "__main__":
    # Example usage
    A = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.4], [-3.0, -4.0, 7.0]])
    n = A.shape[0]
    nP = n * (n + 1) // 2

    grid = torch.linspace(-1, 1, 4, dtype=torch.float32)
    x1, x2, x3= torch.meshgrid(grid, grid, grid, indexing='ij')
    x_train = torch.stack([x1.flatten(), x2.flatten(),x3.flatten()], dim=1).T

    # import scipy
    # P = scipy.linalg.solve_continuous_lyapunov(A.numpy(), -torch.eye(3).numpy())
    # P = torch.tensor(P, dtype=torch.float32)
    P=torch.diag(torch.tensor([1.0,2.0,3.0], dtype=torch.float32))
    x=x_train
    for _ in range(10):
        cost=lyap_cost(x,A,P)
        if cost == 0:
            print("Lyapunov cost is zero, stopping.")
            break

        step=0.01*grad_Vdot(x,A,P)
        x=x+step
        print(f"Lyapunov cost: {cost}")

