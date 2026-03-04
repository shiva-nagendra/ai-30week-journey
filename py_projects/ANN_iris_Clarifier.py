#week 17 project
#ANN iris clarifier

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load iris
iris = load_iris()
x = iris.data
y = iris.target

#split features
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

#feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#convert to tensors
x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)

y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

# Neural Network Architecture
class IrisNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 3)

    def forward(self, x):

        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x
    

model = IrisNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training Loop
for epoch in range(300):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
with torch.no_grad():

    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).sum().item() / len(y_test)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")