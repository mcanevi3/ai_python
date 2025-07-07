import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control
import torch

def get_ts(w, d):
    try:
        Gs = control.tf(w*(w+d), [1, 2*w + d, w*(w+d)])
        info = control.step_info(Gs)
        return info['SettlingTime']
    except:
        return np.nan  # for stability or simulation errors

# Grids
wvec = np.logspace(-2, 2, 100)  # from 0.01 to 100
dvec = np.linspace(0, 1000, 100)

# Meshgrid
Wmat, Dmat = np.meshgrid(wvec, dvec)

# Flatten for iteration
Wflat = Wmat.flatten()
Dflat = Dmat.flatten()

# Compute t_s values
ts_list = []
for w, d in zip(Wflat, Dflat):
    ts = get_ts(w, d)
    ts_list.append(ts)

ts_array = np.array(ts_list)

# Filter out NaNs (e.g., unstable systems or simulation errors)
valid_mask = ~np.isnan(ts_array)
W_valid = Wflat[valid_mask]
D_valid = Dflat[valid_mask]
TS_valid = ts_array[valid_mask]

# Prepare input matrix: (N, 2)
X_np = np.vstack([W_valid, D_valid]).T  # shape: (N, 2)
y_np = TS_valid.reshape(-1, 1)          # shape: (N, 1)

# Convert to torch tensors
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# Optional: human-readable DataFrame
df = pd.DataFrame({
    'w': W_valid,
    'd': D_valid,
    'ts': TS_valid
})

print(df.head())
print(f"PyTorch input X shape: {X.shape}, y shape: {y.shape}")

df.to_csv("settling_time_data.csv", index=False)
torch.save({'X': X, 'y': y}, "settling_time_tensors.pt")
