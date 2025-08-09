import torch

# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using:{device}")

# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()
# end_event.record()
# torch.cuda.synchronize()

# print(f"Time: {start_event.elapsed_time(end_event):.3f} ms")
