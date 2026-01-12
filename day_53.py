#mini project 
#Numpy data pipeline

import numpy as np

# Fake dataset: [age, income, score]
data = np.array([
    [25, 50000, 78],
    [30, 60000, 85],
    [22, 45000, 72],
    [35, 70000, 90],
    [28, 52000, 80],
    [40, 80000, 95],
    [19, 30000, 60]
])

print("\nOriginal dataset:\n", data)

#remove people with score less than 70
clean_data = data[data[:, 2] >= 70]
print("\nClean data: \n", clean_data)

#Normalize income(0-1 range)
income = clean_data[:, 0]
income_norm = (income - income.min())/(income.max()-income.min())
clean_data[:, 1] = income_norm*100 #scale to 0-100

print("\nAfter income normalization: \n", clean_data)

#compute stats

avg_score = np.mean(clean_data[:, 2])
std_score = np.std(clean_data[:, 2])

print("\nAverage score : \n", avg_score)
print("\nStandard deviation: \n", std_score)

