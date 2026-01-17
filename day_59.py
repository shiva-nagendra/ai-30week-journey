#mini project on pandas

import pandas as pd

employees = pd.read_csv("employees2.csv")
scores = pd.read_csv("scores2.csv")

print("\nEmployees :\n", employees)
print("\nScores:\n",scores)

#merge data

merged_data = pd.merge(employees,scores,on="emp_id")
print("\nMerged data: \n", merged_data)

# check missing data
print("\nMissing data, \n", merged_data.isnull())


#fill missing score with avg
merged_data["score"] = merged_data["score"].fillna(merged_data["score"].mean())

print("\nfilled data: \n", merged_data)

#average score by dept.

print("\nAverage score per dept: \n")
print(merged_data.groupby("department")["score"].mean())

print("\nAverage income per dep: \n")
print(merged_data.groupby("department")["income"].mean())

high = merged_data[merged_data["score"]>= 85]
print("\nHigh score: \n", high)
