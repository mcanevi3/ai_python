import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from SimpleNN import SimpleNN
from SimpleCNN import SimpleCNN

# Device config (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1000,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1000,
                                          shuffle=False)

model = SimpleCNN().to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 4
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Save trained model
torch.save(model.state_dict(), "mnist_model_cnn.pth")
print(model.state_dict())
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# Visual test: predict a sample
sample_img, sample_label = test_dataset[0]
model.eval()
with torch.no_grad():
    output = model(sample_img.unsqueeze(0).to(device))
    prediction = output.argmax(dim=1).item()

plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f'Predicted: {prediction}, True: {sample_label}')
plt.axis('off')
plt.show()
