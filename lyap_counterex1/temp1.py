import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2-x**2
def grad_f(x):
    return -2*x

xi=1
print(f"i:0 xi:{xi}")
xvec=[xi]
fvec=[f(xi)]
plt.plot(xvec,fvec,'kx')
for i in range(30):
    xi=xi+0.1*grad_f(xi)

    xvec.append(xi)
    fvec.append(f(xi))

plt.plot(xvec,fvec)
plt.plot(xvec[-1],fvec[-1],'ko')
plt.grid()
plt.show()