#day 54 #project #student analysis #visualizing graphically

import numpy as np

# Dataset: [age, study_hours, score]
data = np.array([
    [18, 2, 65],
    [19, 4, 70],
    [18, 1, 50],
    [20, 6, 85],
    [21, 7, 90],
    [19, 3, 68],
    [22, 8, 92],
    [18, 2, 60]
])

print("Raw Data:\n", data)

#Remove students with score less than 60
filtered = data[data[:, 2]>= 60]
print("\nFilterd data: \n", filtered)

#seperate features and target
x_raw = data[:, :2]
y = data[:, 2]

x_raw = np.sort(x_raw)
y = np.sort(y)

print("\nFeatures: \n", x_raw)
print("\nTarget :\n",y)

#normalize features

x_min = x_raw.min(axis=0)
x_max = x_raw.max(axis=0)
X_norm = (x_raw - x_min)/(x_max-x_min)
X_norm = np.round(X_norm, 2)

print("\nNormalised features: \n", X_norm)

#feature engineering: study hours per age

study_per_age = x_raw[:, 1]/ x_raw[:, 0]
study_per_age = np.round(study_per_age, 2)
#final feature matrix

X_final = np.column_stack((X_norm, study_per_age))
print("\n Final matrix: \n", X_final)

#stats

print("\n sverage students marks: \n", np.mean(y))
print("\n standard deviation: \n", np.std(y))

import matplotlib.pyplot as plt

plt.figure()
plt.plot( y, x_raw, marker = "o", markersize = 9)
plt.grid()
plt.xlabel("age")
plt.ylabel("score")
plt.show()
