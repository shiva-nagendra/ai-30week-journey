#Feature Engineering with NumPy

import numpy as np

dataset = np.array([
    #age. income. score
    [25,   50000,   78],
    [30,   60000,   85],
    [22,   45000,   72],
    [35,   70000,   90],
    [28,   52000,   80],
    [40,   80000,   95],
])

print("\nRaw data: \n", dataset)

#seperating features and target

x_raw = dataset[:, :2] #age, income
y = dataset[:, 2]#score

print("\n features X: \n", x_raw)
print("\nTarget Y: \n", y)

#normalize features

x_min = x_raw.min(axis=0)
x_max = x_raw.max(axis=0)

x_norm = round(((x_raw - x_min)/(x_max-x_min))*100,2)

print("\nNormalized X: \n", x_norm)

#create a new feature (income per age)

income_per_age = x_raw[:, 1]/x_raw[:, 0]

#Add it as a new column

X_new = np.column_stack((x_norm, income_per_age))

print("\nnew feature matrix: \n", X_new)



