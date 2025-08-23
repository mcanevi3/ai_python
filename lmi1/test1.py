from defs import *
from util import *

# K = np.array([[-32.62468001,-8.21394156]])
# K=np.array([[0.03934841, 0.71462091]])
np.random.seed(4)
K = np.random.rand(1, 2)
# K=np.array([[0.15300342, 0.00486872]])
print(f"K:{K}")
u=x_train @ K.T
xdot_train=x_train @ A.T + u @ B.T
alpha=0

Ac=A+B@K
eigVal,eigVec = np.linalg.eig(Ac)
print(f"eig(A+B@K):{eigVal}")

P=solve_lmi(x_train,xdot_train,alpha,verbose=False)
# print(f"P:{P}")
if P is not None:
    eigP,vecP = np.linalg.eig(P)
    print(f"eig(P):{eigP}")

    # lmi1=Ac.T@P+P@Ac+alpha*P
    # eigLmi,vecLmi=np.linalg.eig(lmi1)
    # print(f"eig(LMI):{eigLmi}")
    
    print(f"is stable? {is_stable_eig(eigVal)}")
    cost, constraints=test_lmi(x_train,xdot_train,P,alpha)
    print(f"cost:{cost}")

    print_pos_values(constraints)
    plot_constraint(constraints)
else:
    print(f"is stable? {is_stable_eig(eigVal)}")
    print("No solution!")