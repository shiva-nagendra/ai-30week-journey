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