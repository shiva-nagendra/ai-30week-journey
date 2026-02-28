#week 17 day 3
#Neural networks 

import torch
import torch.nn as nn

#Define neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.linear = nn.Linear(2, 1) #2 inputs to 1 output

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
#Create model
model = SimpleNN()

#print model structure
print(model)

for name, param in model.named_parameters():
    print(name, param)