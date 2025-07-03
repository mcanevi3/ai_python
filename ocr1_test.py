import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from SimpleNN import SimpleNN
from SimpleCNN import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("mnist_model_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# img = Image.open("saved_digits/001_0_.png").convert('L')  # Grayscale
img = Image.open("image4.png").convert('L')  # Grayscale
img = img.resize((28, 28))                  # Resize to MNIST format
# Optional: Invert colors if background is dark
# img = transforms.functional.invert(img)
img.save("transformed.png")

# Transform for the network: to tensor + normalize
transform = transforms.Compose([
    transforms.ToTensor(),                    # Shape: [1, 28, 28]
    transforms.Normalize((0.5,), (0.5,))      # Normalize to [-1, 1]
])
img_tensor = transform(img).unsqueeze(0)      # Shape: [1, 1, 28, 28]

# === 4. Predict ===
with torch.no_grad():
    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1).item()

#  === Plot the output logits as a bar chart ===
output_np = output.squeeze().numpy()  # shape: (10,)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.xticks(range(10))
plt.xlabel("Digit Class")
plt.ylabel("Logit Score")
plt.title("Model Output Scores")
plt.grid(True, axis='y')
probs = torch.softmax(output, dim=1).squeeze().numpy()
plt.bar(range(10), probs)

plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Digit: {prediction}")
plt.axis('off')
plt.show()