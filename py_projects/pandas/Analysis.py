#pandas project

import pandas as pd

employees = pd.read_csv("employees_prjct.csv")
performance = pd.read_csv("performance.csv")

#Load both CSVs
print("Employees:\n", employees)
print("\nPerformance:\n", performance)

#inspect missing values

missing_values = employees.isnull().sum()
missing_performance = performance.isnull().sum()
print("\nmissing employees values: \n",missing_values)
print("\nMissing performance values:\n",missing_performance)

#clean data

#1. fill age with median
#2. fill missing score with mean
employees["age"] = employees["age"].fillna(employees["age"].median())
performance["score"]= performance["score"].fillna(performance["score"].mean())

#merge tables on emp_id

merged_data = pd.merge(employees, performance, on="emp_id")
print("\nMerged data:\n", merged_data)

#Analyze

#1.avg. score per dept.
#2. avg. income per dept.
#3. top performers (score >= 85)

#1
average_score = merged_data.groupby("department")["score"].mean()
print("\nAverage score per dept :\n", average_score)

average_income = merged_data.groupby("department")["income"].mean()
print("\nAverage income per dept :\n", average_income)


top_performer = merged_data[merged_data["score"]>=85]
print("\nTop performers:\n", top_performer)


