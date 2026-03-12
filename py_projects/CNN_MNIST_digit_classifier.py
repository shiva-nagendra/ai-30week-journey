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
