#week 17 day 6
#Training a Neural Network on a Real Dataset (Iris)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load data
iris = load_iris()
x = iris.data
y = iris.target

#split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

#scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#convert to tensor
x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

#Define model
class IrisNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4,16)
        self.relu = nn.ReLU
        self.output = nn.Linear(16,3)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
model = IrisNN()

#loss and optimizer
crieterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

#Training loop

for epoch in range(200):
    output = model(x_train)
    loss = crieterion(output, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"\nEpoch:{epoch}, Loss:{loss.item()}")

