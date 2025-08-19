import torch
import json 
import scipy

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
    return torch.linalg.eig(P)

def normalize(x):
    norm = torch.linalg.norm(x)
    if norm > 0:
        x_normed = x / norm
    else:
        x_normed = x
    return x_normed