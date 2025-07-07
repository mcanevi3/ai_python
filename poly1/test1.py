import numpy as np

def settling_time(w, lam):
    return (3.91 + 1.92 * lam) / w

def loss(w, lam, ts_star):
    return (settling_time(w, lam) - ts_star)**2

def gradients(w, lam, ts_star):
    e = settling_time(w, lam) - ts_star
    dL_dw = 2 * e * (-(3.91 + 1.92 * lam) / w**2)
    dL_dlam = 2 * e * (1.92 / w)
    return dL_dw, dL_dlam

# Initial values
ts_star = 200  # desired settling time
w, lam = 6/ts_star, 0
lr = 0.0001

for i in range(1000):
    dw, dlam = gradients(w, lam, ts_star)
    w -= lr * dw
    lam -= lr * dlam
    
    # Project back into feasible region
    w = max(w, 1e-3)
    lam = min(max(lam, 1e-6), 1.0)
    
    if i % 100 == 0:
        print(f"Iter {i}: w = {w:.4f}, λ = {lam:.4f}, ts = {settling_time(w, lam):.4f}, loss = {loss(w, lam, ts_star):.6f}")

print(ts_star)
print(settling_time(w,lam))
