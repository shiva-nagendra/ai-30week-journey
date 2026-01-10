#numpy practice
import numpy as np
A = [1,2,3,4]
B = np.array([1,2,3])

print("a = ",A)
print("b = ",B)

print("B + 10 = ", B + 10)
print("B * 2 = ", B * 2)

C = np.array([10,20,30])
print("B + C =", B + C)

print("dot product:", np.dot(B, C))
