import torch
import json 
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_symmetric_P(pvec,n=2):
    L = torch.zeros((n,n), dtype=torch.float32)
    idx = 0
    for i in range(n):
        for j in range(i+1):
            L[i, j] = pvec[idx]
            idx += 1
    P = L @ L.T
    return P

def save_json(data:dict,file_name:str):
    json_str = json.dumps(data, indent=4)
    with open(file_name, "w") as f:
        f.write(json_str)   

def load_json(file_name:str):
    with open(file_name, "r") as f:
        data = json.load(f)    
        return data

def get_lqr_P(A, Q):
    P = scipy.linalg.solve_continuous_lyapunov(A.numpy().T, -Q.numpy())
    return torch.tensor(P, dtype=torch.float32)
def eigs(P):
    return torch.linalg.eigvals(P)

def plot3_points(ax,x_train, color='b', marker='o', markersize=3):
    """
    Plots 3D points from a torch tensor like MATLAB's plot3.
    
    Args:
        x_train: torch.Tensor of shape (3, N) — columns are points [x, y, z].
        color: color of points/lines.
        marker: marker style.
        markersize: marker size.
    """
    # Ensure CPU numpy
    points = x_train.cpu().numpy()
    
    for i in range(points.shape[1]):
        ax.plot([0, points[0, i]],
                [0, points[1, i]],
                [0, points[2, i]],
                color=color, marker=marker, markersize=markersize)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points')
    ax.grid(True)