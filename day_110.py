#week 18 day 3
# Train CNN on MNIST

import torch
import torch.nn
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

    




