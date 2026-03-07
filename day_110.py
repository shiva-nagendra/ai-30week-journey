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
    batch_size=64,
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

        




