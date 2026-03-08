#week 18 day 5
# Evaluate CNN on MNIST test set

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

# training dataset
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

# test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False
)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train model
for epoch in range(3):

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} finished")

