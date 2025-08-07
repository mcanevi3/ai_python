"""
Test polar coordinates
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# # 2x196
# theta = torch.linspace(-2*torch.pi, 2*torch.pi, 36, dtype=torch.float32)
# r=torch.linspace(0, 1, 10, dtype=torch.float32)
# theta, r = torch.meshgrid(theta, r, indexing='ij')
# theta = theta.flatten()
# r = r.flatten()
# x_train = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=0)  # shape (2, 100)

def sample_unit_ball(n_dim, n_points):
    x = torch.randn(n_points, n_dim)
    x = x / x.norm(dim=1, keepdim=True)  # on sphere
    r = torch.rand(n_points).pow(1.0 / n_dim).unsqueeze(1)  # radius scaled
    return (r * x).T
x_train=sample_unit_ball(2, 500)

x1=x_train[0, :].numpy()
x2=x_train[1, :].numpy()
plt.figure(figsize=(6, 6))
plt.plot(x1, x2, 'o')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Polar Coordinates')
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
