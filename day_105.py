#week 17 day 4

import torch
import torch.nn as nn
import torch.optim as optim

# Fake dataset (y = 2x + 1)
x = torch.randn(100, 1)
y = 2 * x + 1


# Define simple linear model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
# Create model
model = SimpleModel()

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(50):

    # Forward pass
    predictions = model(x)
    loss = criterion(predictions, y)

     # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
