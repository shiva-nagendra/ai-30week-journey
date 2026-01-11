#scaling datasets

import numpy as np

data = np.array([1,2,3,4,5])

A = data * 2
print("raw data:", data)
print("scaled data:", A)

B = np.array([
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
])

v = np.array([10,20,30])
print("\ndata of B:\n", B)
print("\ndata B + v:\n",B+v)
print("\ndivide all values:\n",B/10)

c = B.dot(v)
print("\nB.dot(c) :\n",c)