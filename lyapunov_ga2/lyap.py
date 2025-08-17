import torch

def V(x,A,P):
    return x.T@P@x
def Vdot(x,A,P):
    x_col = x.unsqueeze(1) 
    xdot=A@x_col
    return (xdot.T@P@x_col+x_col.T@P@xdot).squeeze()
def grad_Vdot(x,A,P):
    x_col = x.unsqueeze(1) 
    xdot=A@x_col
    grad_xdot=A
    return (grad_xdot.T@P@x_col+P@xdot).squeeze()

def lyap_cost(x_train,A,P):
    xdot_train=A@x_train
    Vdot=(xdot_train.T@P@x_train+x_train.T@P@xdot_train).diagonal()
    Vdot_res=torch.relu(Vdot)
    Vdot_count=Vdot_res.sum().item()
    
    return Vdot_count
