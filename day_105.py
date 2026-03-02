#week 17 day 4
#Training loop

import torch
import torch.nn as nn
import torch.optim as optim

#create fake data
x = torch.range(100, 1)
y = 2*x + 1

#Define model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1) #1 input 1 output

    def forward(self, x):
        return self.linear(x)
    
model = SimpleModel()

#loss function

