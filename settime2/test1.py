import numpy as np
import matplotlib.pyplot as plt 
import control 

def settime(w,d):
    return 3.91/w+1.92/(w+1.5*d)

def compute_gradients(w, d, k, ts_star):
    denom = (w + 1.5*d)**2
    ts_model = settime(w,d)
    e1 = ts_star - ts_model
    e2 = 2*w + d - 5
    e3 = w**2 + w*d - k
    dJ_dw = e1 * (3.91 / w**2 + 1.92 / denom) + 2*e2 + e3 * (2*w + d)
    dJ_dd = e1 * (2.88 / denom) + e2 + e3 * w
    dJ_dk = -e3
    return dJ_dw, dJ_dd, dJ_dk, e1, e2, e3

ts_star=15
alpha=1e-3

w, d = 6/ts_star,0
k = w*(w+d)

dJ_dw, dJ_dd, dJ_dk,e1,e2,e3 = compute_gradients(w, d, k, ts_star)
J = 0.5*e1**2 + 0.5*e2**2 + 0.5*e3**2
ts_now =settime(w,d)
print(f"J={J:.5f}, w={w:.3f}, d={d:.3f}, k={k:.3f}, ts={ts_now:.3f}")

for i in range(1000):
    dJ_dw, dJ_dd, dJ_dk,e1,e2,e3 = compute_gradients(w, d, k, ts_star)
    w -= alpha * dJ_dw
    d -= alpha * dJ_dd
    k -= alpha * dJ_dk
    w = max(w, 1e-1)
    d = max(d, 0.0)
    J = 0.5*e1**2 + 0.5*e2**2 + 0.5*e3**2
    ts_now =settime(w,d)
print(f"J={J:.5f}, w={w:.3f}, d={d:.3f}, k={k:.3f}, ts={ts_now:.3f}")

Gs=control.tf(1,[1,5,0])
Ts=control.feedback(k*Gs,1)
info=control.step_info(Ts)
#print(Ts)
print(f"ts:{info["SettlingTime"]} os:{info["Overshoot"]}")
print(f"1 {2*w+d} {w*(w+d)}")
print(control.poles(Ts))
 