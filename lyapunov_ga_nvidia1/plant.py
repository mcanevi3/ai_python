import scipy
import torch
import numpy as np
import json

from device import *

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
    return np.linalg.eigvals(P.numpy())
# def make_symmetric_P(pvec):
#     # P=torch.tensor([[pvec[0], pvec[1]],
#     #                  [pvec[1], pvec[2]]], dtype=torch.float32)
#     P=torch.tensor([[pvec[0], pvec[1], pvec[2]],
#                      [pvec[1], pvec[3], pvec[4]],
#                      [pvec[2], pvec[4], pvec[5]]], dtype=torch.float32)
#     return P
def make_symmetric_P(pvec):
    L = torch.zeros((n, n), dtype=torch.float32,device=device)
    idx = 0
    for i in range(n):
        for j in range(i+1):
            L[i, j] = pvec[idx]
            idx += 1
    P = L @ L.T
    return P

Ap=torch.tensor([[1., 2.],[-3., -4.]], dtype=torch.float32,device=device)
Bp=torch.tensor([[0.],[1.]], dtype=torch.float32,device=device)
Cp=torch.tensor([[1., 0.]], dtype=torch.float32,device=device)

A =  torch.cat([
        torch.cat([Ap, torch.zeros((Ap.shape[0], 1),device=device)], dim=1),
        torch.cat([-Cp, torch.zeros((1, 1),device=device)], dim=1)
    ], dim=0)
B = torch.cat([Bp, torch.zeros((1, 1),device=device)], dim=0)
C = torch.cat([Cp, torch.zeros((1, 1),device=device)], dim=1)
Br = torch.cat([torch.zeros((Ap.shape[0], 1),device=device), torch.ones((1, 1),device=device)], dim=0)

n = A.shape[0]  # Number of states
nP=n*(n+1)//2  # Number of unique elements in symmetric P

if __name__=="__main__":
    data = {
    "name": "sathiyajith",
    "rollno": 56,
    "cgpa": 8.6,
    "phone": "9976770500"
    }
    save_json(data,"test.json")
    data2=load_json("test.json")
    print(data2)