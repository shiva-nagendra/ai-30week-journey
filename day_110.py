#week 18 day 3
# Train CNN on MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#transform images to tensors
transform = transforms.ToTensor()

#load MNIST dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

#CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,8,3)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8,16,3)

        self.fc = nn.Linear(16*5*5,10)

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))

        x = self.pool(torch.relu(self.conv2(x))) 

        x = torch.flatten(x,1)

        x = self.fc(x)

        return x
    
model = CNN()

#loss and optimizer
crieterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
for epoch in range(3):
    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = crieterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")
        
