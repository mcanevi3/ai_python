import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 1, 28, 28] → [B, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → [B, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → [B, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                              # → [B, 32, 7, 7]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                # → [B, 32*7*7]
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                           # 10 classes
        )

    def forward(self, x):
        x = self.conv_net(x)
        return self.fc(x)
