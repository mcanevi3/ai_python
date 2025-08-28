import torch
import torch.nn as nn

from controller import *
from defs import *

controller = Controller()
optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
alpha = 0.001

controller.print()

lyapP=LyapunovNet()
lyapOptimizer = torch.optim.Adam(lyapP.parameters(), lr=0.01)
for epoch in range(10000+1):
    u = controller(x_train)        
    xdot_train=x_train@A.T+u@B.T

    P=lyapP()
    v_dot = Vdot(P,x_train,xdot_train) 
    lyapunov_penalty = v_dot + alpha * V(P,x_train)

    violation_prob=cost(lyapunov_penalty)
    loss=violation_prob.mean()
 
    optimizer.zero_grad()
    lyapOptimizer.zero_grad()
    loss.backward()

    optimizer.step()
    lyapOptimizer.step()

    #count violations    
    cond=(lyapunov_penalty>0).squeeze()
    count=cond.sum().item() 
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} Violations:{count}")

with torch.no_grad():
    P=lyapP()
    print(f"P:{P.detach().numpy()}")
    eigP,_=torch.linalg.eig(P)
    print(f"Eig(P):{eigP.detach().numpy()}")

    K=controller.net[0].weight[0].view(1,2)
    print(f"K:{K.detach().numpy()}")    
    Ac=A+B@K
    lmi1=Ac.T@P+P@Ac
    eigLmi,_=torch.linalg.eig(lmi1)
    print(f"Eig of LMI:{eigLmi.detach().numpy()}")
# controller.print()
controller.save()