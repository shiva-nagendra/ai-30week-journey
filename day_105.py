#week 17 day 4

import torch
import torch.nn as nn
import torch.optim as optim

# Fake dataset (y = 2x + 1)
x = torch.randn(100, 1)
y = 2 * x + 1