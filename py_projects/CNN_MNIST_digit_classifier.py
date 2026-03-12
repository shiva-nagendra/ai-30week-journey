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

