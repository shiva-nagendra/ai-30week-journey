#week 18 day 3
# Train CNN on MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Transform images to tensors
transform_tensor = transforms.ToTensor()

#Load MNIST dataset
train_dataset = datasets.MNIST(
    root="./data",
    download=True,
    train=True,
    transform=transform_tensor
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

#CNN model
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

        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

model = CNN()

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#Trainig loop
for epoch in range(5):
    total_loss = 0

    for images,labels in train_loader:
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch{epoch+1}, loss:{total_loss:.4f}")


# git add .
# git commit -m "Week 18 Day 3: Trained CNN on MNIST handwritten digit dataset"
# git push



