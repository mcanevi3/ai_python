import torch

def V(x,A,P):
    return x@P@x.T
def Vdot(x,A,P):
    xdot=x@A.T
    return (xdot@P@x.T+x@P@xdot.T).diagonal()
def grad_Vdot(x,A,P):
    xdot=x@A.T
    grad_xdot=A
    res=grad_xdot@P@x.T+P@xdot.T
    return res.diagonal()

def lyap_cost(x,A,P):
    xdot_train=x@A.T
    Vdot=(xdot_train@P@x.T+x@P@xdot_train.T).diagonal()
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
            step=0.01*grad_Vdot(x,A,P)
            x=x+step
        else:
            break
        print(f"Lyapunov cost: {cost}")

