# import torch

# # Number of GPUs
# num_gpus = torch.cuda.device_count()
# print(f"Number of GPUs: {num_gpus}")

# # List all GPU devices
# for i in range(num_gpus):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#     print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
#     print(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using Apple Silicon GPU via MPS")
# else:
#     device = torch.device("cpu")
#     print("Falling back to CPU")

import torch
import time

def compare_cpu_mps(size=5000):
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    if not mps_available:
        print("MPS not available on this machine.")
        return

    # Create random matrices on CPU first
    a_cpu = torch.randn(size, size, dtype=torch.float32)
    b_cpu = torch.randn(size, size, dtype=torch.float32)

    # CPU timing
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    cpu_time = time.time() - start

    # Move to MPS
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    # Warm-up run to avoid startup overhead
    _ = torch.matmul(a_mps, b_mps)

    # MPS timing
    torch.mps.synchronize()  # Ensure previous ops are done
    start = time.time()
    c_mps = torch.matmul(a_mps, b_mps)
    torch.mps.synchronize()  # Wait for GPU
    mps_time = time.time() - start

    print(f"CPU time: {cpu_time:.4f} s")
    print(f"MPS time: {mps_time:.4f} s")
    print(f"Speedup: {cpu_time / mps_time:.2f}x faster on MPS")

# Run
compare_cpu_mps()
