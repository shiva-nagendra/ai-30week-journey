#week 13 day 4
#standardization vs normalization

import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

x = np.array([[10],[20],[30],[1000]])

std = StandardScaler()
norm = MinMaxScaler()

print("Standardized:")
print(np.round(std.fit_transform(x),2))

print("\nNormalized:")
print(np.round(norm.fit_transform(x),2))

