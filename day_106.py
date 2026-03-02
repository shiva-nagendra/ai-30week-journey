#week 17 day 5
#From One Neuron → Real Neural Network

import torch
import torch.nn as nn
import torch.optim as optim

#Create fake data
x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = x**2

#Define model

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 10) #1 input 10 neurons
        self.output = nn.Linear(10, 1) #10 neurons 1 output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = self.relu(x)
        return x
    
model = SimpleNN()

