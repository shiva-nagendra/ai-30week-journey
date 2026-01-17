#merging on pandas

import pandas as pd

employees = pd.read_csv("employees.csv")
scores = pd.read_csv("scores.csv")

print("\nEmployees: \n", employees)
print("\nScores: \n", scores)

#merge function

merged = pd.merge(employees, scores, on="emp_id")
print("\nMerged data: \n", merged)

#average scores
avg_scores = merged.groupby("department")["score"].mean()
print("\n Average score : \n", avg_scores)