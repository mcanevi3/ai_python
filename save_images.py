import os
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Directory to save images
save_dir = "saved_digits"
os.makedirs(save_dir, exist_ok=True)

# Load MNIST training data
transform = transforms.ToTensor()
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Save first 100 images
for i in range(100):
    img_tensor, label = mnist_dataset[i]
    filename = f"{save_dir}/{i:03d}_{label}_.png"
    save_image(img_tensor, filename)

print(f"Saved 100 images to '{save_dir}' folder.")
