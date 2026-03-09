#week 18 day 5
# Evaluate CNN on MNIST test set

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader

transform = transforms.ToTensor

