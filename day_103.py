#week 17, day 1 and day 2

import torch

x = torch.tensor([1.0,2.0,3.0])

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x +1

y.backward()

print("\nx gradient: ",x.grad)

#pytorch computed derivative automatically