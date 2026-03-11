# Week 18 Day 6
# Save and load CNN model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,8,3)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8,16,3)
        self.fc = nn.Linear(16*5*5,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = torch.flatten(x)

        x = self.fc(x)

        return x
    
model = CNN()

criterion = nn.CrossEntropyLoss()
optimization = optim.Adam(model.parameters(),lr=0.001)

#training
for epoch in range(5):
    for images, labels in train_loader:
        outcome = model(images)

        loss = criterion(outcome, labels)

        optimization.zero_grad()
        loss.backward()
        optimization.step()

        
        