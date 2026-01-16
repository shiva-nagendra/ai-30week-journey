# department,age,income,score
# IT,25,50000,78
# IT,30,60000,85
# HR,22,45000,72
# HR,35,70000,90
# Sales,28,52000,80
# Sales,40,80000,95
# IT,27,55000,82
# HR,29,48000,75


import pandas as pd

df = pd.read_csv("week8_day3.csv")

print("Full Data:\n", df)

# 1. Sort by score (highest first)
sorted_df = df.sort_values(by="score", ascending=True)
print("\nSorted by score:\n", sorted_df)

# 2. Group by department
grouped = df.groupby("department")

# 3. Average score per department
print("\nAverage score per department:")
print(grouped["score"].mean())

# 4. Count people per department
print("\nCount per department:")
print(grouped["score"].count())

# 5. Max score per department
print("\nMax score per department:")
print(grouped["score"].max())