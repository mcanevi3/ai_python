import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU via MPS")
else:
    device = torch.device("cpu")
    print("Falling back to CPU")

vec = torch.tensor([1.0, 2.0, 3.0], device=device)
print(vec)
