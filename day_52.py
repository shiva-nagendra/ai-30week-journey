#Indexing, slicing & dataset prep

import numpy as np

data = np.array([
    [25, 50000, 78],
    [30, 60000, 85],
    [22, 45000, 72],
    [35, 70000, 90],
    [28, 52000, 80]
])

print("Full data:", data)

print("\nFirst person", data[0])#print first row

scores = data[:, 2] #print a column
print("\nscores:",scores)

print("\n first three rows:\n",data[:3])

high_scores = data[data[:, 2] > 80]
print("\nHigh scorers:\n", high_scores)

data[:, 1] = data[:, 1]*1.10
print("\ndata after salary raise:",data)

