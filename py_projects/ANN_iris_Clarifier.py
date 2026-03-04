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

