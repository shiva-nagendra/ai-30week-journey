


import pandas as pd

# Load CSV
df = pd.read_csv("week8_day1.csv")

print("Full DataFrame:")
print(df)

# Show first rows
print("\nHead:")
print(df.head())

# Show stats
print("\nDescribe:")
print(df.describe())

# Select a column
print("\nScores:")
print(df["score"])

# Filter rows
high = df[df["score"] > 80]
print("\nHigh scorers:")
print(high)




