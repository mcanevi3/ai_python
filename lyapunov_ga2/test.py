import torch 

from util import *
from lyap import *
from defs import *

data=load_json("best_candidate.json")

pvec=torch.tensor(data["pvec"])
P=make_symmetric_P(pvec,3)

eigsP=eigs(P)
print(f"P: {P}")
print(f"Eigenvalues of P: {eigsP}")
lmi1=A.T@P+P@A
eigsLMI1=eigs(lmi1)
print(f"LMI1: {lmi1}")
print(f"Eigenvalues of LMI1: {eigsLMI1}")
