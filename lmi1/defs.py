import numpy as np

A=np.array([[1.0,2.0],[-3.0,-4.0]])
B=np.array([[1.0],[2.0]])

# grid=np.linspace(-1,1,100)
grid=np.linspace(-1,1,5)
x1,x2=np.meshgrid(grid,grid)
x_train=np.vstack([x1.flatten(),x2.flatten()]).T
