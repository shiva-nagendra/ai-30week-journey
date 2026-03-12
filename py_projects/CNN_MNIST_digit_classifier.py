#week 18 project
#CNN MNIST digit classifier with random predictions

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

#transform images to tensor
transform = transforms.ToTensor()

#load training data
train_datasets = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_datasets,
    batch_size=128,
    shuffle=True
)

#load test data
test_datasets = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_datasets,
    batch_size=128,
    shuffle=False
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

criterion = nn.CrossEntropyLoss()
optimization = optim.Adam(model.parameters(),lr=0.001)

#training

print("\nTraining model....\n")

for epoch in range(3):

    total_loss = 0

    for images, labels in train_loader:
        output = model(images)
        loss = criterion(output,labels)

        optimization.zero_grad()
        loss.backward()
        optimization.step()

        total_loss+=loss.item()

    print(f"\nEpoch {epoch+1}, Total loss: {total_loss:.3f}")


#Pick random test image
index = random.randint(0,9999)

image, label = test_datasets[index]

#add batch dimension
image = image.unsqueeze(0)

#prediction
model.eval()

with torch.no_grad():

    output = model(image)

    _, prediction = torch.max(output, 1)

print("\nPrediction:",prediction.item())
print("\nActual",label)


with torch.no_grad():

    for images, labels in test_loader:

        total = 0
        correct = 0

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

 