#week 17 day 4
#Training loop

import torch
import torch.nn as nn
import torch.optim as optim

#create fake data
x = torch.randn(100, 1)
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
criterion = nn.MSELoss()

#define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

#Training loop

for epoch in range(50):

    #forward pass(model prediction)
    pred = model(x)

    #Compute loss(measure error)
    loss = criterion(pred, y)

    #clear errors
    optimizer.zero_grad()

    #backward pass
    loss.backward()

    #update weights
    optimizer.step()

    if epoch % 10 == 0:
        print(f"\nEpoch:{epoch},Loss:{loss.item()}")


#print learned parameters

for name, params in model.named_parameters():
    print(name, params)



