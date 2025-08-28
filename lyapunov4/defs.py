import torch

A=torch.tensor([[1.0,2.0],[-3.0,-4.0]])
B=torch.tensor([[1.0],[0.0]])

grid=torch.linspace(-1,1,10)
x1,x2=torch.meshgrid(grid,grid,indexing='ij')
x_train=torch.vstack([x1.flatten(),x2.flatten()]).T
