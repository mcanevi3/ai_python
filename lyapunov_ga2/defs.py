import torch

A=torch.tensor([[1.0,2.0,3.0],[2.0,2.0,2.4],[-3.0,-4.0,7.0]])
n=A.shape[0]
nP=n*(n+1)//2
