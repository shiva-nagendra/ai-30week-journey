#2D matrices practice

import numpy as np

A = np.array([ #2x3 matrix
    [1,2,3],
    [4,5,6]
])

v = np.array([10,20,30])

print("mat A:",A)
print("vec v:",v)

res = A.dot(v)
print("A.v : \n", res)

B = np.array([ #3x2 matrix
    [2,3],
    [3,4],
    [4,5]
])

C = A.dot(B) #matix x matrix
print("B matrix: ", B)
print("A.B : \n", C)

print("\nShape of A:", A.shape)
print("\nShape of B:", B.shape)
print("\nShape of C:", C.shape)